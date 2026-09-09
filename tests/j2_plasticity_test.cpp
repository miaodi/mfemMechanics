#include "J2Plasticity.h"
#include "PostProc.h"
#include "SolidMechanicsIntegrator.h"
#include "Solvers.h"
#include "util.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-local-typedefs"
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>
#include <autodiff/forward/utils/gradient.hpp>
#pragma GCC diagnostic pop

#include <Eigen/Dense>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

TEST( ParaView2DVectorCoefficient, PadsPlanarGridFunctionWithZeroZComponent )
{
    mfem::Mesh mesh = mfem::Mesh::MakeCartesian2D( 1, 1, mfem::Element::QUADRILATERAL );
    mfem::H1_FECollection collection( 1, mesh.Dimension() );
    mfem::FiniteElementSpace space( &mesh, &collection, 2, mfem::Ordering::byVDIM );
    mfem::GridFunction displacement( &space );
    mfem::VectorFunctionCoefficient prescribedDisplacement( 2,
                                                            []( const mfem::Vector& position, mfem::Vector& value )
                                                            {
                                                                value( 0 ) = 2. * position( 0 ) + position( 1 );
                                                                value( 1 ) = -position( 0 ) + 3. * position( 1 );
                                                            } );
    displacement.ProjectCoefficient( prescribedDisplacement );

    plugin::ParaView2DVectorCoefficient outputDisplacement( displacement );
    mfem::IntegrationPoint integrationPoint;
    integrationPoint.Set2( .25, .75 );
    mfem::ElementTransformation& transformation = *mesh.GetElementTransformation( 0 );
    transformation.SetIntPoint( &integrationPoint );
    mfem::Vector value;
    outputDisplacement.Eval( value, transformation, integrationPoint );

    ASSERT_EQ( value.Size(), 3 );
    const mfem::real_t tolerance = 100. * std::numeric_limits<mfem::real_t>::epsilon();
    EXPECT_NEAR( value( 0 ), 1.25, tolerance );
    EXPECT_NEAR( value( 1 ), 2., tolerance );
    EXPECT_DOUBLE_EQ( value( 2 ), 0. );

    mfem::ParaViewDataCollection paraview( "paraview_vector_test", &mesh );
    paraview.RegisterVCoeffField( "displacement", &outputDisplacement );
    ASSERT_EQ( paraview.GetVCoeffFieldMap().count( "displacement" ), 1 );
    EXPECT_EQ( paraview.GetVCoeffFieldMap().at( "displacement" )->GetVDim(), 3 );
}

struct TransactionCheckingState
{
    int Evaluations{ 0 };
};

class TransactionCheckingHistory
{
public:
    const TransactionCheckingState& CommittedState() const noexcept
    {
        return mCommitted;
    }

    const TransactionCheckingState& TrialState() const noexcept
    {
        return mTrial;
    }

    void SetTrialState( const TransactionCheckingState& state )
    {
        MFEM_VERIFY( mActive, "Trial state requires an active material-point transaction." );
        mTrial = state;
    }

    void BeginStep() noexcept
    {
        MFEM_VERIFY( !mActive, "Material-point history was begun more than once." );
        mActive = true;
        mTrial = mCommitted;
        mBeginCalls++;
    }

    void CommitStep() noexcept
    {
        MFEM_VERIFY( mActive, "Material-point history commit requires BeginStep." );
        mCommitted = mTrial;
        mActive = false;
        mCommitCalls++;
    }

    void RollbackStep() noexcept
    {
        MFEM_VERIFY( mActive, "Material-point history rollback requires BeginStep." );
        mTrial = mCommitted;
        mActive = false;
        mRollbackCalls++;
    }

    int BeginCalls() const noexcept
    {
        return mBeginCalls;
    }

    int CommitCalls() const noexcept
    {
        return mCommitCalls;
    }

    int RollbackCalls() const noexcept
    {
        return mRollbackCalls;
    }

private:
    TransactionCheckingState mCommitted;
    TransactionCheckingState mTrial;
    int mBeginCalls{ 0 };
    int mCommitCalls{ 0 };
    int mRollbackCalls{ 0 };
    bool mActive{ false };
};

class TransactionCheckingMaterial
{
public:
    static constexpr plugin::SolidKinematics Kinematics = plugin::SolidKinematics::SmallStrain;

    struct Response : plugin::SolidMaterialResponse
    {
        TransactionCheckingState TrialState;
    };

    Response Evaluate( const plugin::SmallStrainMaterialPoint& materialPoint, const TransactionCheckingState& committedState ) const
    {
        Response response;
        response.Stress = materialPoint.MechanicalStrain;
        response.ConsistentTangent.diagonal() << 1., 1., 1., .5, .5, .5;
        response.TrialState = committedState;
        response.TrialState.Evaluations++;
        return response;
    }
};

namespace plugin
{
template <>
struct MaterialPointTraits<TransactionCheckingMaterial>
{
    using State = TransactionCheckingHistory;
};
} // namespace plugin

namespace
{
constexpr bool kSinglePrecision = std::is_same_v<mfem::real_t, float>;
constexpr mfem::real_t kTightTolerance = kSinglePrecision ? 2e-5f : 2e-12;
constexpr mfem::real_t kDerivativeTolerance = kSinglePrecision ? 2e-2f : 2e-6;

using J2PointStorage = plugin::SolidMechanicsPointStorage<J2PlasticityMaterial>;

class IdentitySmallStrainMaterial
{
public:
    static constexpr plugin::SolidKinematics Kinematics = plugin::SolidKinematics::SmallStrain;

    plugin::SolidMaterialResponse Evaluate( const plugin::SmallStrainMaterialPoint& materialPoint ) const
    {
        plugin::SolidMaterialResponse response;
        response.Stress = materialPoint.MechanicalStrain;
        response.ConsistentTangent.diagonal() << 1., 1., 1., .5, .5, .5;
        return response;
    }
};

class IdentityFiniteStrainMaterial
{
public:
    static constexpr plugin::SolidKinematics Kinematics = plugin::SolidKinematics::FiniteStrain;

    plugin::SolidMaterialResponse Evaluate( const plugin::FiniteStrainMaterialPoint& materialPoint ) const
    {
        plugin::SolidMaterialResponse response;
        response.Stress = .5 * ( materialPoint.DeformationGradient.transpose() * materialPoint.DeformationGradient -
                                 Eigen::Matrix3r::Identity() );
        response.ConsistentTangent.diagonal() << 1., 1., 1., .5, .5, .5;
        return response;
    }
};

using IdentityPointStorage = plugin::SolidMechanicsPointStorage<IdentitySmallStrainMaterial>;
using IdentityIntegrator = plugin::SolidMechanicsIntegrator<IdentitySmallStrainMaterial>;
static_assert( std::is_same_v<typename IdentityPointStorage::ElementStateType, plugin::NoIntegrationPointState>,
               "A stateless solid material should use geometry-only point storage." );
static_assert( !std::is_constructible_v<IdentityIntegrator, IdentitySmallStrainMaterial&&, IdentityPointStorage&>,
               "A solid mechanics integrator must not borrow a temporary material." );

plugin::J2PlasticityParameters Parameters()
{
    plugin::J2PlasticityParameters parameters;
    parameters.YoungsModulus = 200.;
    parameters.PoissonRatio = .25;
    parameters.Hardening.InitialYieldStress = 1.;
    parameters.Hardening.HardeningModulus = 10.;
    return parameters;
}

mfem::real_t EquivalentStress( const Eigen::Matrix3r& stress )
{
    const Eigen::Matrix3r deviatoricStress = stress - stress.trace() / 3. * Eigen::Matrix3r::Identity();
    return std::sqrt( 1.5 * deviatoricStress.squaredNorm() );
}

mfem::real_t YieldStress( const plugin::LinearIsotropicHardening& hardening, const mfem::real_t equivalentPlasticStrain )
{
    return hardening.InitialYieldStress + hardening.HardeningModulus * equivalentPlasticStrain;
}

Eigen::Vector6r StressVector( const Eigen::Vector6r& engineeringStrain,
                              const plugin::J2PlasticityState& committedState,
                              const plugin::J2PlasticityParameters& parameters )
{
    const auto response = plugin::EvaluateJ2Plasticity( util::InverseVoigt( engineeringStrain, true ), committedState, parameters );
    return util::Voigt<mfem::real_t, mfem::real_t>( response.Stress, false );
}

Eigen::Matrix6r FiniteDifferenceTangent( const Eigen::Vector6r& strain,
                                         const plugin::J2PlasticityState& committedState,
                                         const plugin::J2PlasticityParameters& parameters )
{
    const mfem::real_t step = kSinglePrecision ? 2e-4f : 1e-7;
    Eigen::Matrix6r tangent;
    for ( int column = 0; column < 6; column++ )
    {
        Eigen::Vector6r plus = strain;
        Eigen::Vector6r minus = strain;
        plus( column ) += step;
        minus( column ) -= step;
        tangent.col( column ) =
            ( StressVector( plus, committedState, parameters ) - StressVector( minus, committedState, parameters ) ) / ( 2. * step );
    }
    return tangent;
}

class FixedStepContext final : public plugin::NonlinearStepContext
{
public:
    FixedStepContext()
    {
        SetDelta( 1. );
    }

    bool Convergence() const override
    {
        return true;
    }
};

template <int Dimension>
mfem::Vector AffineDisplacement( const mfem::FiniteElement& element,
                                 mfem::ElementTransformation& transformation,
                                 const Eigen::Matrix<mfem::real_t, Dimension, Dimension>& gradient )
{
    MFEM_ASSERT( element.GetDim() == Dimension, "The affine displacement dimension must match the element." );
    const int dofs = element.GetDof();
    mfem::Vector displacement( Dimension * dofs );
    mfem::Vector position( Dimension );
    const auto& nodes = element.GetNodes();
    for ( int node = 0; node < dofs; node++ )
    {
        transformation.Transform( nodes.IntPoint( node ), position );
        const Eigen::Matrix<mfem::real_t, Dimension, 1> value =
            gradient * Eigen::Map<const Eigen::Matrix<mfem::real_t, Dimension, 1>>( position.GetData() );
        for ( int component = 0; component < Dimension; component++ )
        {
            displacement( node + component * dofs ) = value( component );
        }
    }
    return displacement;
}

template <typename Material, int Dimension>
void ExpectStatelessMaterialJacobianMatchesDirectionalDifference( mfem::Mesh& mesh,
                                                                  const Material& material,
                                                                  const Eigen::Matrix<mfem::real_t, Dimension, Dimension>& gradient )
{
    mfem::H1_FECollection collection( 1, mesh.Dimension() );
    mfem::FiniteElementSpace space( &mesh, &collection, mesh.Dimension(), mfem::Ordering::byVDIM );
    const auto* element = space.GetFE( 0 );
    auto* transformation = mesh.GetElementTransformation( 0 );
    ASSERT_NE( element, nullptr );
    ASSERT_NE( transformation, nullptr );

    plugin::SolidMechanicsPointStorage<Material> pointStorage( &mesh );
    plugin::SolidMechanicsIntegrator<Material> integrator( material, pointStorage );
    FixedStepContext context;
    integrator.SetStepContext( &context );

    const mfem::Vector displacement = AffineDisplacement( *element, *transformation, gradient );
    mfem::DenseMatrix tangent;
    integrator.AssembleElementGrad( *element, *transformation, displacement, tangent );

    mfem::Vector direction( displacement.Size() );
    for ( int i = 0; i < direction.Size(); i++ )
    {
        direction( i ) = static_cast<mfem::real_t>( ( i % 5 ) - 2 );
    }
    direction /= direction.Norml2();
    const mfem::real_t step = kSinglePrecision ? 2e-4f : 1e-7;
    mfem::Vector plus( displacement ), minus( displacement );
    plus.Add( step, direction );
    minus.Add( -step, direction );
    mfem::Vector plusResidual, minusResidual;
    integrator.AssembleElementVector( *element, *transformation, plus, plusResidual );
    integrator.AssembleElementVector( *element, *transformation, minus, minusResidual );
    plusResidual -= minusResidual;
    plusResidual /= 2. * step;

    mfem::Vector analytic( direction.Size() );
    tangent.Mult( direction, analytic );
    analytic -= plusResidual;
    EXPECT_LE( analytic.Norml2(), kDerivativeTolerance * ( 1. + plusResidual.Norml2() ) );
}

template <int Dimension>
void ExpectElementJacobianMatchesDirectionalDifference( mfem::Mesh& mesh,
                                                        const Eigen::Matrix<mfem::real_t, Dimension, Dimension>& gradient,
                                                        const mfem::real_t kinematicHardeningModulus = 0. )
{
    mfem::H1_FECollection collection( 1, mesh.Dimension() );
    mfem::FiniteElementSpace space( &mesh, &collection, mesh.Dimension(), mfem::Ordering::byVDIM );
    const auto* element = space.GetFE( 0 );
    auto* transformation = mesh.GetElementTransformation( 0 );
    ASSERT_NE( element, nullptr );
    ASSERT_NE( transformation, nullptr );

    mfem::ConstantCoefficient youngsModulus( 200. );
    mfem::ConstantCoefficient poissonRatio( .25 );
    mfem::ConstantCoefficient initialYieldStress( 1. );
    mfem::ConstantCoefficient hardeningModulus( 10. );
    mfem::ConstantCoefficient kinematicHardening( kinematicHardeningModulus );
    J2PlasticityMaterial material( youngsModulus, poissonRatio, initialYieldStress, hardeningModulus, kinematicHardening );
    J2PointStorage pointStorage( &mesh );
    plugin::SolidMechanicsIntegrator<J2PlasticityMaterial> integrator( material, pointStorage );
    FixedStepContext context;
    integrator.SetStepContext( &context );

    const mfem::Vector displacement = AffineDisplacement( *element, *transformation, gradient );
    mfem::DenseMatrix tangent;
    integrator.AssembleElementGrad( *element, *transformation, displacement, tangent );

    mfem::Vector direction( displacement.Size() );
    for ( int i = 0; i < direction.Size(); i++ )
    {
        direction( i ) = static_cast<mfem::real_t>( ( i % 5 ) - 2 );
    }
    direction /= direction.Norml2();
    const mfem::real_t step = kSinglePrecision ? 2e-4f : 1e-7;
    mfem::Vector plus( displacement ), minus( displacement );
    plus.Add( step, direction );
    minus.Add( -step, direction );
    mfem::Vector plusResidual, minusResidual;
    integrator.AssembleElementVector( *element, *transformation, plus, plusResidual );
    integrator.AssembleElementVector( *element, *transformation, minus, minusResidual );
    plusResidual -= minusResidual;
    plusResidual /= 2. * step;

    mfem::Vector analytic( direction.Size() );
    tangent.Mult( direction, analytic );
    analytic -= plusResidual;
    EXPECT_LE( analytic.Norml2(), kDerivativeTolerance * ( 1. + plusResidual.Norml2() ) );
}

class ConstantResidualOperator final : public mfem::Operator
{
public:
    ConstantResidualOperator() : mfem::Operator( 1 ), mGradient( 1 )
    {
        mGradient = 1.;
    }

    void Mult( const mfem::Vector&, mfem::Vector& residual ) const override
    {
        residual.SetSize( 1 );
        residual = 1.;
    }

    mfem::Operator& GetGradient( const mfem::Vector& ) const override
    {
        return mGradient;
    }

private:
    mutable mfem::DenseMatrix mGradient;
};

class IdentitySolver final : public mfem::Solver
{
public:
    IdentitySolver() : mfem::Solver( 1 )
    {
    }

    void SetOperator( const mfem::Operator& ) override
    {
    }

    void Mult( const mfem::Vector& rightHandSide, mfem::Vector& solution ) const override
    {
        solution = rightHandSide;
    }
};

class IdentityOperator final : public mfem::Operator
{
public:
    IdentityOperator() : mfem::Operator( 1 ), mGradient( 1 )
    {
        mGradient = 1.;
    }

    void Mult( const mfem::Vector& state, mfem::Vector& residual ) const override
    {
        residual = state;
    }

    mfem::Operator& GetGradient( const mfem::Vector& ) const override
    {
        return mGradient;
    }

private:
    mutable mfem::DenseMatrix mGradient;
};

class ZeroResidualOperator final : public mfem::Operator
{
public:
    ZeroResidualOperator() : mfem::Operator( 1 ), mGradient( 1 )
    {
        mGradient = 1.;
    }

    void Mult( const mfem::Vector&, mfem::Vector& residual ) const override
    {
        residual.SetSize( 1 );
        residual = 0.;
    }

    mfem::Operator& GetGradient( const mfem::Vector& ) const override
    {
        return mGradient;
    }

private:
    mutable mfem::DenseMatrix mGradient;
};

class SwitchableResidualOperator final : public mfem::Operator
{
public:
    SwitchableResidualOperator() : mfem::Operator( 1 ), mGradient( 1 )
    {
        mGradient = 1.;
    }

    void SetReject( const bool reject )
    {
        mReject = reject;
    }

    void Mult( const mfem::Vector&, mfem::Vector& residual ) const override
    {
        residual.SetSize( 1 );
        residual = mReject ? 1. : 0.;
    }

    mfem::Operator& GetGradient( const mfem::Vector& ) const override
    {
        return mGradient;
    }

private:
    bool mReject{ false };
    mutable mfem::DenseMatrix mGradient;
};

class PoisonableIntegrator final : public plugin::StepAwareNonlinearFormIntegrator
{
public:
    void SetResidual( const mfem::real_t residual )
    {
        mResidual = residual;
    }

    void AssembleElementVector( const mfem::FiniteElement& element, mfem::ElementTransformation&, const mfem::Vector&, mfem::Vector& residual ) override
    {
        residual.SetSize( element.GetDof() );
        residual = mResidual;
    }

    void AssembleElementGrad( const mfem::FiniteElement& element, mfem::ElementTransformation&, const mfem::Vector&, mfem::DenseMatrix& gradient ) override
    {
        gradient.SetSize( element.GetDof() );
        gradient = 0.;
        for ( int i = 0; i < element.GetDof(); i++ )
        {
            gradient( i, i ) = 1.;
        }
    }

    void BeginStep() noexcept override
    {
        StepAwareNonlinearFormIntegrator::BeginStep();
        mDepth++;
    }

    void CommitStep() noexcept override
    {
        MFEM_VERIFY( mDepth > 0, "Test commit requires a matching begin." );
        if ( --mDepth == 0 )
        {
            mCommits++;
            mRejected = false;
        }
        StepAwareNonlinearFormIntegrator::CommitStep();
    }

    void RollbackStep() noexcept override
    {
        MFEM_VERIFY( mDepth > 0, "Test rollback requires a matching begin." );
        mRejected = true;
        if ( --mDepth == 0 )
        {
            mRollbacks++;
            mRejected = false;
        }
        StepAwareNonlinearFormIntegrator::RollbackStep();
    }

    bool CanCommitStep() const noexcept override
    {
        return !mRejected;
    }

    int Commits() const noexcept
    {
        return mCommits;
    }

    int Rollbacks() const noexcept
    {
        return mRollbacks;
    }

private:
    int mDepth{ 0 };
    int mCommits{ 0 };
    int mRollbacks{ 0 };
    bool mRejected{ false };
    mfem::real_t mResidual{ 0. };
};

class CompositeTestOperator final : public mfem::Operator, public plugin::CompositeNonlinearOperator
{
public:
    explicit CompositeTestOperator( mfem::Operator& child )
        : mfem::Operator( child.Height(), child.Width() ), mChild( child )
    {
    }

    void Mult( const mfem::Vector& input, mfem::Vector& output ) const override
    {
        mChild.Mult( input, output );
    }

    mfem::Operator& GetGradient( const mfem::Vector& input ) const override
    {
        return mChild.GetGradient( input );
    }

    void GetChildOperators( std::vector<const mfem::Operator*>& children ) const override
    {
        children.push_back( &mChild );
        children.push_back( &mChild );
    }

private:
    mfem::Operator& mChild;
};
} // namespace

TEST( J2Plasticity, ZeroAndHydrostaticStrainRemainElastic )
{
    const auto parameters = Parameters();
    const plugin::J2PlasticityState state;

    const auto zero = plugin::EvaluateJ2Plasticity( Eigen::Matrix3r::Zero(), state, parameters );
    EXPECT_EQ( zero.Branch, plugin::J2PlasticityBranch::Elastic );
    EXPECT_LE( zero.Stress.norm(), kTightTolerance );
    EXPECT_EQ( zero.TrialState.EquivalentPlasticStrain, 0. );

    const Eigen::Matrix3r hydrostaticStrain = .01 * Eigen::Matrix3r::Identity();
    const auto hydrostatic = plugin::EvaluateJ2Plasticity( hydrostaticStrain, state, parameters );
    EXPECT_EQ( hydrostatic.Branch, plugin::J2PlasticityBranch::Elastic );
    EXPECT_LE( EquivalentStress( hydrostatic.Stress ), kTightTolerance );
    EXPECT_EQ( hydrostatic.TrialState.EquivalentPlasticStrain, 0. );
}

TEST( J2Plasticity, EngineeringShearUsesTheThreeDimensionalJ2Invariant )
{
    const auto parameters = Parameters();
    const mfem::real_t shearModulus = parameters.YoungsModulus / ( 2. * ( 1. + parameters.PoissonRatio ) );
    const mfem::real_t yieldShearStrain =
        parameters.Hardening.InitialYieldStress / ( std::sqrt( mfem::real_t{ 3. } ) * shearModulus );

    Eigen::Vector6r strain = Eigen::Vector6r::Zero();
    strain( 3 ) = .5 * yieldShearStrain;
    EXPECT_EQ( plugin::EvaluateJ2Plasticity( util::InverseVoigt( strain, true ), {}, parameters ).Branch,
               plugin::J2PlasticityBranch::Elastic );

    strain( 3 ) = 2. * yieldShearStrain;
    const auto response = plugin::EvaluateJ2Plasticity( util::InverseVoigt( strain, true ), {}, parameters );
    EXPECT_EQ( response.Branch, plugin::J2PlasticityBranch::Plastic );
    const mfem::real_t yieldStress = YieldStress( parameters.Hardening, response.TrialState.EquivalentPlasticStrain );
    EXPECT_NEAR( EquivalentStress( response.Stress ), yieldStress, kTightTolerance * ( 1. + yieldStress ) );
    EXPECT_GT( std::abs( response.TrialState.PlasticStrain( 0, 1 ) ), 0. );
}

TEST( J2Plasticity, PerfectPlasticAndHardeningReturnsSatisfyConsistency )
{
    Eigen::Matrix3r strain = Eigen::Matrix3r::Zero();
    strain( 0, 0 ) = .03;

    auto hardeningParameters = Parameters();
    const auto hardening = plugin::EvaluateJ2Plasticity( strain, {}, hardeningParameters );
    ASSERT_EQ( hardening.Branch, plugin::J2PlasticityBranch::Plastic );
    EXPECT_NEAR( EquivalentStress( hardening.Stress ),
                 YieldStress( hardeningParameters.Hardening, hardening.TrialState.EquivalentPlasticStrain ), kTightTolerance );

    auto perfectParameters = hardeningParameters;
    perfectParameters.Hardening.HardeningModulus = 0.;
    const auto perfect = plugin::EvaluateJ2Plasticity( strain, {}, perfectParameters );
    ASSERT_EQ( perfect.Branch, plugin::J2PlasticityBranch::Plastic );
    EXPECT_NEAR( EquivalentStress( perfect.Stress ), perfectParameters.Hardening.InitialYieldStress, kTightTolerance );
    EXPECT_LT( EquivalentStress( perfect.Stress ), EquivalentStress( hardening.Stress ) );
}

TEST( J2Plasticity, CombinedHardeningReturnSatisfiesShiftedYieldCondition )
{
    auto parameters = Parameters();
    parameters.KinematicHardening.Modulus = 15.;
    Eigen::Matrix3r strain = Eigen::Matrix3r::Zero();
    strain( 0, 0 ) = .03;

    const auto response = plugin::EvaluateJ2Plasticity( strain, {}, parameters );
    ASSERT_EQ( response.Branch, plugin::J2PlasticityBranch::Plastic );
    const mfem::real_t yieldStress = YieldStress( parameters.Hardening, response.TrialState.EquivalentPlasticStrain );
    EXPECT_NEAR( EquivalentStress( response.Stress - response.TrialState.BackStress ), yieldStress,
                 kTightTolerance * ( 1. + yieldStress ) );
    EXPECT_LE(
        ( response.TrialState.BackStress - ( 2. / 3. ) * parameters.KinematicHardening.Modulus * response.TrialState.PlasticStrain )
            .norm(),
        kTightTolerance * ( 1. + response.TrialState.BackStress.norm() ) );
    EXPECT_NEAR( response.TrialState.BackStress.trace(), 0., kTightTolerance );
    EXPECT_GT( response.TrialState.BackStress.norm(), 0. );
}

TEST( J2Plasticity, LinearKinematicHardeningDemonstratesBauschingerEffect )
{
    auto isotropicParameters = Parameters();
    auto kinematicParameters = Parameters();
    kinematicParameters.Hardening.HardeningModulus = 0.;
    kinematicParameters.KinematicHardening.Modulus = isotropicParameters.Hardening.HardeningModulus;

    Eigen::Vector6r loadingStrain = Eigen::Vector6r::Zero();
    loadingStrain( 3 ) = .04;
    const auto isotropicLoading = plugin::EvaluateJ2Plasticity( util::InverseVoigt( loadingStrain, true ), {}, isotropicParameters );
    const auto kinematicLoading = plugin::EvaluateJ2Plasticity( util::InverseVoigt( loadingStrain, true ), {}, kinematicParameters );
    ASSERT_EQ( isotropicLoading.Branch, plugin::J2PlasticityBranch::Plastic );
    ASSERT_EQ( kinematicLoading.Branch, plugin::J2PlasticityBranch::Plastic );
    EXPECT_LE( ( isotropicLoading.Stress - kinematicLoading.Stress ).norm(),
               kTightTolerance * ( 1. + isotropicLoading.Stress.norm() ) );

    Eigen::Vector6r reverseStrain = Eigen::Vector6r::Zero();
    reverseStrain( 3 ) = .024;
    const auto isotropicReverse = plugin::EvaluateJ2Plasticity( util::InverseVoigt( reverseStrain, true ),
                                                                isotropicLoading.TrialState, isotropicParameters );
    const auto kinematicReverse = plugin::EvaluateJ2Plasticity( util::InverseVoigt( reverseStrain, true ),
                                                                kinematicLoading.TrialState, kinematicParameters );
    EXPECT_EQ( isotropicReverse.Branch, plugin::J2PlasticityBranch::Elastic );
    EXPECT_EQ( kinematicReverse.Branch, plugin::J2PlasticityBranch::Plastic );
    EXPECT_GT( kinematicReverse.Stress( 0, 1 ), isotropicReverse.Stress( 0, 1 ) );
}

TEST( J2Plasticity, ProportionalLoadingIsIndependentOfIncrementSubdivision )
{
    const auto parameters = Parameters();
    Eigen::Matrix3r finalStrain = Eigen::Matrix3r::Zero();
    finalStrain( 0, 0 ) = .03;
    const auto singleStep = plugin::EvaluateJ2Plasticity( finalStrain, {}, parameters );
    ASSERT_EQ( singleStep.Branch, plugin::J2PlasticityBranch::Plastic );

    plugin::J2PlasticityState incrementedState;
    plugin::J2PlasticityResponse incrementedResponse;
    for ( int increment = 1; increment <= 3; increment++ )
    {
        const Eigen::Matrix3r strain = static_cast<mfem::real_t>( increment ) / 3. * finalStrain;
        incrementedResponse = plugin::EvaluateJ2Plasticity( strain, incrementedState, parameters );
        incrementedState = incrementedResponse.TrialState;
    }

    EXPECT_LE( ( incrementedResponse.Stress - singleStep.Stress ).norm(), kTightTolerance * ( 1. + singleStep.Stress.norm() ) );
    EXPECT_LE( ( incrementedState.PlasticStrain - singleStep.TrialState.PlasticStrain ).norm(),
               kTightTolerance * ( 1. + singleStep.TrialState.PlasticStrain.norm() ) );
    EXPECT_NEAR( incrementedState.EquivalentPlasticStrain, singleStep.TrialState.EquivalentPlasticStrain, kTightTolerance );
}

TEST( J2Plasticity, RejectsInvalidParametersAndHistory )
{
    GTEST_FLAG_SET( death_test_style, "threadsafe" );
    auto parameters = Parameters();
    parameters.PoissonRatio = .5;
    EXPECT_DEATH( (void)plugin::EvaluateJ2Plasticity( Eigen::Matrix3r::Zero(), {}, parameters ), "Poisson ratio" );

    parameters = Parameters();
    parameters.Hardening.HardeningModulus = -1.;
    EXPECT_DEATH( (void)plugin::EvaluateJ2Plasticity( Eigen::Matrix3r::Zero(), {}, parameters ), "hardening modulus" );

    parameters = Parameters();
    parameters.KinematicHardening.Modulus = -1.;
    EXPECT_DEATH( (void)plugin::EvaluateJ2Plasticity( Eigen::Matrix3r::Zero(), {}, parameters ),
                  "kinematic hardening modulus" );

    plugin::J2PlasticityState invalidState;
    invalidState.EquivalentPlasticStrain = -1.;
    EXPECT_DEATH( (void)plugin::EvaluateJ2Plasticity( Eigen::Matrix3r::Zero(), invalidState, Parameters() ),
                  "equivalent plastic strain" );

    Eigen::Matrix3r nonsymmetricStrain = Eigen::Matrix3r::Zero();
    nonsymmetricStrain( 0, 1 ) = 1e-3;
    EXPECT_DEATH( (void)plugin::EvaluateJ2Plasticity( nonsymmetricStrain, {}, Parameters() ), "must be symmetric" );

    invalidState = {};
    invalidState.PlasticStrain( 0, 0 ) = 1e-3;
    EXPECT_DEATH( (void)plugin::EvaluateJ2Plasticity( Eigen::Matrix3r::Zero(), invalidState, Parameters() ),
                  "must be deviatoric" );

    invalidState = {};
    invalidState.BackStress( 0, 1 ) = 1.;
    EXPECT_DEATH( (void)plugin::EvaluateJ2Plasticity( Eigen::Matrix3r::Zero(), invalidState, Parameters() ),
                  "backstress must be symmetric" );

    invalidState = {};
    invalidState.BackStress( 0, 0 ) = 1.;
    EXPECT_DEATH( (void)plugin::EvaluateJ2Plasticity( Eigen::Matrix3r::Zero(), invalidState, Parameters() ),
                  "backstress must be deviatoric" );

    parameters = Parameters();
    parameters.YoungsModulus = std::numeric_limits<mfem::real_t>::max();
    parameters.PoissonRatio = .49;
    EXPECT_DEATH( (void)plugin::EvaluateJ2Plasticity( Eigen::Matrix3r::Zero(), {}, parameters ),
                  "elastic moduli must be finite" );
}

TEST( J2Plasticity, ConsistentTangentMatchesCenteredDifferenceOnPlasticBranch )
{
    const auto parameters = Parameters();
    Eigen::Vector6r strain;
    strain << .025, -.004, .002, .011, -.006, .004;
    const auto response = plugin::EvaluateJ2Plasticity( util::InverseVoigt( strain, true ), {}, parameters );
    ASSERT_EQ( response.Branch, plugin::J2PlasticityBranch::Plastic );

    const Eigen::Matrix6r numerical = FiniteDifferenceTangent( strain, {}, parameters );
    EXPECT_LE( ( response.ConsistentTangent - numerical ).norm(), kDerivativeTolerance * ( 1. + numerical.norm() ) );
    EXPECT_LE( ( response.ConsistentTangent - response.ConsistentTangent.transpose() ).norm(), kTightTolerance );
}

TEST( J2Plasticity, ConsistentTangentMatchesExistingAutodiffOracle )
{
    const auto parameters = Parameters();
    autodiff::VectorXdual strain( 6 );
    strain << .025, -.004, .002, .011, -.006, .004;

    const auto stress = [parameters]( const autodiff::VectorXdual& engineeringStrain )
    {
        using Scalar = autodiff::dual;
        const Scalar shearModulus = parameters.YoungsModulus / ( 2. * ( 1. + parameters.PoissonRatio ) );
        const Scalar bulkModulus = parameters.YoungsModulus / ( 3. * ( 1. - 2. * parameters.PoissonRatio ) );
        const Eigen::Matrix<Scalar, 6, 1> fixedStrain = engineeringStrain;
        const Eigen::Matrix<Scalar, 3, 3> strainTensor = util::InverseVoigt( fixedStrain, true );
        const Eigen::Matrix<Scalar, 3, 3> identity = Eigen::Matrix<Scalar, 3, 3>::Identity();
        const Eigen::Matrix<Scalar, 3, 3> deviatoricStrain = strainTensor - strainTensor.trace() / 3. * identity;
        const Eigen::Matrix<Scalar, 3, 3> trialDeviatoricStress = 2. * shearModulus * deviatoricStrain;
        using std::sqrt;
        const Scalar equivalentStress = sqrt( 1.5 * trialDeviatoricStress.squaredNorm() );
        const Scalar plasticIncrement = ( equivalentStress - parameters.Hardening.InitialYieldStress ) /
                                        ( 3. * shearModulus + parameters.Hardening.HardeningModulus );
        const Scalar radialScale = 1. - 3. * shearModulus * plasticIncrement / equivalentStress;
        const Eigen::Matrix<Scalar, 3, 3> stressTensor =
            bulkModulus * strainTensor.trace() * identity + radialScale * trialDeviatoricStress;
        return util::Voigt<Scalar, Scalar>( stressTensor, false );
    };

    const Eigen::MatrixXd autodiffTangent = autodiff::jacobian( stress, autodiff::wrt( strain ), autodiff::at( strain ) );
    const Eigen::Vector6r realStrain = strain.cast<mfem::real_t>();
    const auto response = plugin::EvaluateJ2Plasticity( util::InverseVoigt( realStrain, true ), {}, parameters );
    EXPECT_LE( ( response.ConsistentTangent - autodiffTangent.cast<mfem::real_t>() ).norm(),
               kDerivativeTolerance * ( 1. + autodiffTangent.norm() ) );
}

TEST( J2Plasticity, PlaneStrainRetainsOutOfPlaneStressAndPlasticFlow )
{
    Eigen::Matrix3r planeStrain = Eigen::Matrix3r::Zero();
    planeStrain( 1, 1 ) = .03;
    const auto response = plugin::EvaluateJ2Plasticity( planeStrain, {}, Parameters() );
    ASSERT_EQ( response.Branch, plugin::J2PlasticityBranch::Plastic );
    EXPECT_GT( std::abs( response.Stress( 2, 2 ) ), 0. );
    EXPECT_GT( std::abs( response.TrialState.PlasticStrain( 2, 2 ) ), 0. );
    EXPECT_NEAR( response.TrialState.PlasticStrain.trace(), 0., kTightTolerance );
}

TEST( J2PlasticityHistory, TrialEvaluationIsDeterministicAndRollbackSafe )
{
    auto parameters = Parameters();
    parameters.KinematicHardening.Modulus = 15.;
    plugin::J2PlasticityHistory history;
    history.BeginStep();

    Eigen::Matrix3r largeStrain = Eigen::Matrix3r::Zero();
    largeStrain( 0, 0 ) = .04;
    history.SetTrialState( plugin::EvaluateJ2Plasticity( largeStrain, history.CommittedState(), parameters ).TrialState );
    ASSERT_GT( history.TrialState().EquivalentPlasticStrain, 0. );

    Eigen::Matrix3r smallStrain = Eigen::Matrix3r::Zero();
    smallStrain( 0, 0 ) = .001;
    history.SetTrialState( plugin::EvaluateJ2Plasticity( smallStrain, history.CommittedState(), parameters ).TrialState );
    EXPECT_EQ( history.TrialState().EquivalentPlasticStrain, 0. );
    history.CommitStep();
    EXPECT_EQ( history.CommittedState().EquivalentPlasticStrain, 0. );

    history.BeginStep();
    history.SetTrialState( plugin::EvaluateJ2Plasticity( largeStrain, history.CommittedState(), parameters ).TrialState );
    history.RollbackStep();
    EXPECT_EQ( history.TrialState().EquivalentPlasticStrain, history.CommittedState().EquivalentPlasticStrain );
    EXPECT_LE( ( history.TrialState().BackStress - history.CommittedState().BackStress ).norm(), kTightTolerance );
    EXPECT_LE( ( history.TrialState().PlasticStrain - history.CommittedState().PlasticStrain ).norm(), kTightTolerance );

    history.BeginStep();
    const auto firstPlasticResponse = plugin::EvaluateJ2Plasticity( largeStrain, history.CommittedState(), parameters );
    ASSERT_EQ( firstPlasticResponse.Branch, plugin::J2PlasticityBranch::Plastic );
    history.SetTrialState( firstPlasticResponse.TrialState );
    history.CommitStep();
    const mfem::real_t firstCommittedPlasticStrain = history.CommittedState().EquivalentPlasticStrain;
    const Eigen::Matrix3r firstCommittedBackStress = history.CommittedState().BackStress;
    ASSERT_GT( firstCommittedPlasticStrain, 0. );
    ASSERT_GT( firstCommittedBackStress.norm(), 0. );

    Eigen::Matrix3r unloadedStrain = largeStrain;
    unloadedStrain( 0, 0 ) -= .001;
    history.BeginStep();
    const auto unloading = plugin::EvaluateJ2Plasticity( unloadedStrain, history.CommittedState(), parameters );
    EXPECT_EQ( unloading.Branch, plugin::J2PlasticityBranch::Elastic );
    EXPECT_EQ( unloading.TrialState.EquivalentPlasticStrain, firstCommittedPlasticStrain );
    history.SetTrialState( unloading.TrialState );
    history.CommitStep();

    Eigen::Matrix3r reloadedStrain = largeStrain;
    reloadedStrain( 0, 0 ) += .01;
    history.BeginStep();
    const auto reloading = plugin::EvaluateJ2Plasticity( reloadedStrain, history.CommittedState(), parameters );
    ASSERT_EQ( reloading.Branch, plugin::J2PlasticityBranch::Plastic );
    EXPECT_GT( reloading.TrialState.EquivalentPlasticStrain, firstCommittedPlasticStrain );
    history.SetTrialState( reloading.TrialState );
    history.RollbackStep();
    EXPECT_EQ( history.CommittedState().EquivalentPlasticStrain, firstCommittedPlasticStrain );
    EXPECT_LE( ( history.TrialState().BackStress - firstCommittedBackStress ).norm(), kTightTolerance );
}

TEST( J2Plasticity, ConsistentTangentMatchesCenteredDifferenceAfterPlasticHistory )
{
    const auto parameters = Parameters();
    Eigen::Vector6r firstStrain = Eigen::Vector6r::Zero();
    firstStrain( 0 ) = .03;
    const auto firstResponse = plugin::EvaluateJ2Plasticity( util::InverseVoigt( firstStrain, true ), {}, parameters );
    ASSERT_EQ( firstResponse.Branch, plugin::J2PlasticityBranch::Plastic );

    Eigen::Vector6r secondStrain;
    secondStrain << .045, -.003, .001, .008, -.004, .006;
    const auto secondResponse =
        plugin::EvaluateJ2Plasticity( util::InverseVoigt( secondStrain, true ), firstResponse.TrialState, parameters );
    ASSERT_EQ( secondResponse.Branch, plugin::J2PlasticityBranch::Plastic );

    const Eigen::Matrix6r numerical = FiniteDifferenceTangent( secondStrain, firstResponse.TrialState, parameters );
    EXPECT_LE( ( secondResponse.ConsistentTangent - numerical ).norm(), kDerivativeTolerance * ( 1. + numerical.norm() ) );
}

TEST( J2Plasticity, KinematicTangentMatchesCenteredDifferenceAfterNonproportionalHistory )
{
    auto parameters = Parameters();
    parameters.KinematicHardening.Modulus = 15.;
    Eigen::Vector6r firstStrain = Eigen::Vector6r::Zero();
    firstStrain( 0 ) = .03;
    firstStrain( 3 ) = .01;
    const auto firstResponse = plugin::EvaluateJ2Plasticity( util::InverseVoigt( firstStrain, true ), {}, parameters );
    ASSERT_EQ( firstResponse.Branch, plugin::J2PlasticityBranch::Plastic );
    ASSERT_GT( firstResponse.TrialState.BackStress.norm(), 0. );

    Eigen::Vector6r secondStrain;
    secondStrain << .045, -.003, .001, .008, -.004, .006;
    const auto secondResponse =
        plugin::EvaluateJ2Plasticity( util::InverseVoigt( secondStrain, true ), firstResponse.TrialState, parameters );
    ASSERT_EQ( secondResponse.Branch, plugin::J2PlasticityBranch::Plastic );

    const Eigen::Matrix6r numerical = FiniteDifferenceTangent( secondStrain, firstResponse.TrialState, parameters );
    EXPECT_LE( ( secondResponse.ConsistentTangent - numerical ).norm(), kDerivativeTolerance * ( 1. + numerical.norm() ) );
    EXPECT_LE( ( secondResponse.ConsistentTangent - secondResponse.ConsistentTangent.transpose() ).norm(), kTightTolerance );
}

TEST( SolidMechanicsIntegrator, AcceptsStatelessSmallStrainMaterial )
{
    mfem::Mesh mesh = mfem::Mesh::MakeCartesian2D( 1, 1, mfem::Element::QUADRILATERAL, true, 1., 1. );
    Eigen::Matrix2r gradient;
    gradient << .018, .006, .002, -.003;
    ExpectStatelessMaterialJacobianMatchesDirectionalDifference( mesh, IdentitySmallStrainMaterial{}, gradient );
}

TEST( SolidMechanicsIntegrator, DispatchesFiniteStrainKinematicsFromMaterial )
{
    mfem::Mesh mesh2d = mfem::Mesh::MakeCartesian2D( 1, 1, mfem::Element::QUADRILATERAL, true, 1., 1. );
    Eigen::Matrix2r gradient2d;
    gradient2d << .12, .04, -.02, -.03;
    ExpectStatelessMaterialJacobianMatchesDirectionalDifference( mesh2d, IdentityFiniteStrainMaterial{}, gradient2d );

    mfem::Mesh mesh3d = mfem::Mesh::MakeCartesian3D( 1, 1, 1, mfem::Element::HEXAHEDRON, 1., 1., 1. );
    Eigen::Matrix3r gradient3d;
    gradient3d << .08, .04, -.03, -.02, -.05, .06, .03, -.01, .02;
    ExpectStatelessMaterialJacobianMatchesDirectionalDifference( mesh3d, IdentityFiniteStrainMaterial{}, gradient3d );
}

TEST( SolidMechanicsIntegrator, InitializesLazilyCreatedHistoryOncePerTransaction )
{
    mfem::Mesh mesh = mfem::Mesh::MakeCartesian2D( 1, 1, mfem::Element::QUADRILATERAL, true, 1., 1. );
    mfem::H1_FECollection collection( 1, mesh.Dimension() );
    mfem::FiniteElementSpace space( &mesh, &collection, mesh.Dimension(), mfem::Ordering::byVDIM );
    const auto* element = space.GetFE( 0 );
    auto* transformation = mesh.GetElementTransformation( 0 );
    ASSERT_NE( element, nullptr );
    ASSERT_NE( transformation, nullptr );

    TransactionCheckingMaterial material;
    plugin::SolidMechanicsPointStorage<TransactionCheckingMaterial> pointStorage( &mesh );
    plugin::SolidMechanicsIntegrator<TransactionCheckingMaterial> integrator( material, pointStorage );
    FixedStepContext context;
    integrator.SetStepContext( &context );
    mfem::Vector displacement( element->GetDof() * mesh.Dimension() );
    displacement = 0.;
    mfem::Vector residual;

    integrator.BeginStep();
    integrator.AssembleElementVector( *element, *transformation, displacement, residual );
    integrator.AssembleElementVector( *element, *transformation, displacement, residual );
    auto& history = pointStorage.GetElementPoint( 0 ).State.template Get<TransactionCheckingMaterial>();
    EXPECT_EQ( history.BeginCalls(), 1 );
    EXPECT_EQ( history.TrialState().Evaluations, 1 );
    integrator.CommitStep();
    EXPECT_EQ( history.CommitCalls(), 1 );
    EXPECT_EQ( history.CommittedState().Evaluations, 1 );

    integrator.BeginStep();
    integrator.BeginStep();
    integrator.AssembleElementVector( *element, *transformation, displacement, residual );
    EXPECT_EQ( history.BeginCalls(), 2 );
    integrator.RollbackStep();
    EXPECT_FALSE( integrator.CanCommitStep() );
    integrator.CommitStep();
    EXPECT_EQ( history.RollbackCalls(), 1 );
    EXPECT_EQ( history.TrialState().Evaluations, history.CommittedState().Evaluations );
}

TEST( SolidMechanicsIntegrator, J2TwoDimensionalElementJacobianMatchesResidualDirectionalDifference )
{
    mfem::Mesh mesh = mfem::Mesh::MakeCartesian2D( 1, 1, mfem::Element::QUADRILATERAL, true, 1., 1. );
    Eigen::Matrix2r gradient;
    gradient << .018, .006, .002, -.003;
    ExpectElementJacobianMatchesDirectionalDifference( mesh, gradient );
}

TEST( SolidMechanicsIntegrator, J2ThreeDimensionalElementJacobianMatchesResidualDirectionalDifference )
{
    mfem::Mesh mesh = mfem::Mesh::MakeCartesian3D( 1, 1, 1, mfem::Element::HEXAHEDRON, 1., 1., 1. );
    Eigen::Matrix3r gradient;
    gradient << .018, .006, -.004, .002, -.003, .005, .001, -.002, .004;
    ExpectElementJacobianMatchesDirectionalDifference( mesh, gradient );
}

TEST( SolidMechanicsIntegrator, J2KinematicElementJacobianMatchesResidualDirectionalDifference )
{
    mfem::Mesh mesh = mfem::Mesh::MakeCartesian2D( 1, 1, mfem::Element::QUADRILATERAL, true, 1., 1. );
    Eigen::Matrix2r gradient;
    gradient << .018, .006, .002, -.003;
    ExpectElementJacobianMatchesDirectionalDifference( mesh, gradient, 15. );
}

TEST( SolidMechanicsIntegrator, CommitsRollsBackAndProjectsJ2ElementHistory )
{
    mfem::Mesh mesh = mfem::Mesh::MakeCartesian2D( 1, 1, mfem::Element::QUADRILATERAL, true, 1., 1. );
    mfem::H1_FECollection displacementCollection( 1, mesh.Dimension() );
    mfem::FiniteElementSpace displacementSpace( &mesh, &displacementCollection, mesh.Dimension(), mfem::Ordering::byVDIM );
    const auto* element = displacementSpace.GetFE( 0 );
    auto* transformation = mesh.GetElementTransformation( 0 );
    ASSERT_NE( element, nullptr );
    ASSERT_NE( transformation, nullptr );

    mfem::ConstantCoefficient youngsModulus( 200. );
    mfem::ConstantCoefficient poissonRatio( .25 );
    mfem::ConstantCoefficient initialYieldStress( 1. );
    mfem::ConstantCoefficient hardeningModulus( 10. );
    J2PlasticityMaterial material( youngsModulus, poissonRatio, initialYieldStress, hardeningModulus );
    J2PointStorage pointStorage( &mesh );
    plugin::SolidMechanicsIntegrator<J2PlasticityMaterial> integrator( material, pointStorage );
    FixedStepContext context;
    integrator.SetStepContext( &context );

    Eigen::Matrix2r gradient = Eigen::Matrix2r::Zero();
    gradient( 0, 0 ) = .03;
    mfem::Vector residual;
    integrator.BeginStep();
    integrator.AssembleElementVector( *element, *transformation, AffineDisplacement( *element, *transformation, gradient ), residual );
    integrator.CommitStep();

    mfem::L2_FECollection plasticityCollection( 0, mesh.Dimension() );
    mfem::FiniteElementSpace plasticitySpace( &mesh, &plasticityCollection );
    mfem::GridFunction equivalentPlasticStrain( &plasticitySpace );
    plugin::ProjectCommittedEquivalentPlasticStrain( pointStorage, equivalentPlasticStrain );
    const mfem::real_t committedValue = equivalentPlasticStrain( 0 );
    ASSERT_GT( committedValue, 0. );

    gradient( 0, 0 ) = .05;
    integrator.BeginStep();
    integrator.AssembleElementVector( *element, *transformation, AffineDisplacement( *element, *transformation, gradient ), residual );
    integrator.RollbackStep();
    plugin::ProjectCommittedEquivalentPlasticStrain( pointStorage, equivalentPlasticStrain );
    EXPECT_EQ( equivalentPlasticStrain( 0 ), committedValue );

    GTEST_FLAG_SET( death_test_style, "threadsafe" );
    mfem::Mesh otherMesh = mfem::Mesh::MakeCartesian2D( 1, 1, mfem::Element::QUADRILATERAL );
    mfem::L2_FECollection otherCollection( 0, otherMesh.Dimension() );
    mfem::FiniteElementSpace otherSpace( &otherMesh, &otherCollection );
    mfem::GridFunction incompatibleField( &otherSpace );
    EXPECT_DEATH( plugin::ProjectCommittedEquivalentPlasticStrain( pointStorage, incompatibleField ),
                  "point-storage mesh" );
}

TEST( SolidMechanicsIntegrator, J2GlobalVectorOrderingsProduceTheSameResidual )
{
    mfem::Mesh mesh = mfem::Mesh::MakeCartesian2D( 1, 1, mfem::Element::QUADRILATERAL, true, 1., 1. );
    mfem::H1_FECollection collection( 1, mesh.Dimension() );
    mfem::FiniteElementSpace byNodesSpace( &mesh, &collection, mesh.Dimension(), mfem::Ordering::byNODES );
    mfem::FiniteElementSpace byVdimSpace( &mesh, &collection, mesh.Dimension(), mfem::Ordering::byVDIM );

    mfem::ConstantCoefficient youngsModulus( 200. );
    mfem::ConstantCoefficient poissonRatio( .25 );
    mfem::ConstantCoefficient initialYieldStress( 1. );
    mfem::ConstantCoefficient hardeningModulus( 10. );
    J2PlasticityMaterial material( youngsModulus, poissonRatio, initialYieldStress, hardeningModulus );
    J2PointStorage byNodesStorage( &mesh );
    J2PointStorage byVdimStorage( &mesh );
    mfem::NonlinearForm byNodesForm( &byNodesSpace );
    mfem::NonlinearForm byVdimForm( &byVdimSpace );
    auto* byNodesIntegrator = new plugin::SolidMechanicsIntegrator<J2PlasticityMaterial>( material, byNodesStorage );
    auto* byVdimIntegrator = new plugin::SolidMechanicsIntegrator<J2PlasticityMaterial>( material, byVdimStorage );
    byNodesForm.AddDomainIntegrator( byNodesIntegrator );
    byVdimForm.AddDomainIntegrator( byVdimIntegrator );
    FixedStepContext context;
    byNodesIntegrator->SetStepContext( &context );
    byVdimIntegrator->SetStepContext( &context );

    mfem::VectorFunctionCoefficient affineDisplacement( mesh.Dimension(),
                                                        []( const mfem::Vector& position, mfem::Vector& displacement )
                                                        {
                                                            displacement.SetSize( 2 );
                                                            displacement( 0 ) = .018 * position( 0 ) + .006 * position( 1 );
                                                            displacement( 1 ) = .002 * position( 0 ) - .003 * position( 1 );
                                                        } );
    mfem::GridFunction byNodesDisplacement( &byNodesSpace );
    mfem::GridFunction byVdimDisplacement( &byVdimSpace );
    byNodesDisplacement.ProjectCoefficient( affineDisplacement );
    byVdimDisplacement.ProjectCoefficient( affineDisplacement );
    mfem::Vector byNodesTrueDofs, byVdimTrueDofs;
    byNodesDisplacement.GetTrueDofs( byNodesTrueDofs );
    byVdimDisplacement.GetTrueDofs( byVdimTrueDofs );

    mfem::Vector byNodesResidual( byNodesForm.Height() ), byVdimResidual( byVdimForm.Height() );
    byNodesForm.Mult( byNodesTrueDofs, byNodesResidual );
    byVdimForm.Mult( byVdimTrueDofs, byVdimResidual );
    ASSERT_EQ( byNodesSpace.GetNDofs(), byVdimSpace.GetNDofs() );
    for ( int component = 0; component < mesh.Dimension(); component++ )
    {
        for ( int dof = 0; dof < byNodesSpace.GetNDofs(); dof++ )
        {
            const int byNodesIndex =
                mfem::Ordering::Map<mfem::Ordering::byNODES>( byNodesSpace.GetNDofs(), mesh.Dimension(), dof, component );
            const int byVdimIndex =
                mfem::Ordering::Map<mfem::Ordering::byVDIM>( byVdimSpace.GetNDofs(), mesh.Dimension(), dof, component );
            EXPECT_NEAR( byNodesResidual( byNodesIndex ), byVdimResidual( byVdimIndex ), kTightTolerance );
        }
    }
}

TEST( NonlinearStepLifecycle, RejectedIntegratorRollsBackWholeTransaction )
{
    mfem::Mesh mesh = mfem::Mesh::MakeCartesian2D( 1, 1, mfem::Element::QUADRILATERAL );
    mfem::H1_FECollection collection( 1, mesh.Dimension() );
    mfem::FiniteElementSpace space( &mesh, &collection );
    mfem::NonlinearForm form( &space );
    auto* rejected = new PoisonableIntegrator;
    auto* companion = new PoisonableIntegrator;
    form.AddDomainIntegrator( rejected );
    form.AddDomainIntegrator( companion );
    FixedStepContext context;

    context.BeginStep( &form );
    rejected->BeginStep();
    rejected->RollbackStep();
    EXPECT_FALSE( context.CommitStep( &form ) );
    EXPECT_EQ( rejected->Commits(), 0 );
    EXPECT_EQ( companion->Commits(), 0 );
    EXPECT_EQ( rejected->Rollbacks(), 1 );
    EXPECT_EQ( companion->Rollbacks(), 1 );
}

TEST( NonlinearStepLifecycle, CompositeOperatorForwardsLifecycleToNestedForm )
{
    mfem::Mesh mesh = mfem::Mesh::MakeCartesian2D( 1, 1, mfem::Element::QUADRILATERAL );
    mfem::H1_FECollection collection( 1, mesh.Dimension() );
    mfem::FiniteElementSpace space( &mesh, &collection );
    mfem::NonlinearForm form( &space );
    auto* integrator = new PoisonableIntegrator;
    form.AddDomainIntegrator( integrator );
    CompositeTestOperator composite( form );
    FixedStepContext context;

    context.BeginStep( &composite );
    EXPECT_TRUE( context.CommitStep( &composite ) );
    EXPECT_EQ( integrator->Commits(), 1 );
    EXPECT_EQ( integrator->Rollbacks(), 0 );

    context.BeginStep( &composite );
    context.RollbackStep( &composite );
    EXPECT_EQ( integrator->Commits(), 1 );
    EXPECT_EQ( integrator->Rollbacks(), 1 );
}

TEST( NewtonLineSearch, FailedSolveRestoresInputSolution )
{
    ConstantResidualOperator nonlinearOperator;
    IdentitySolver linearSolver;
    plugin::NewtonLineSearch solver;
    solver.SetOperator( nonlinearOperator );
    solver.SetSolver( linearSolver );
    solver.iterative_mode = true;
    solver.SetMaxIter( 1 );

    mfem::Vector solution( 1 );
    solution = 3.5;
    mfem::Vector rightHandSide;
    solver.Mult( rightHandSide, solution );

    EXPECT_FALSE( solver.GetConverged() );
    EXPECT_EQ( solution( 0 ), 3.5 );
}

TEST( NewtonLineSearch, AcceptsExactFullStepForLinearProblem )
{
    IdentityOperator nonlinearOperator;
    IdentitySolver linearSolver;
    plugin::NewtonLineSearch solver;
    solver.SetOperator( nonlinearOperator );
    solver.SetSolver( linearSolver );
    solver.iterative_mode = true;
    solver.SetLineSearch( true );
    solver.SetRelTol( 1e-12 );
    solver.SetAbsTol( 1e-14 );
    solver.SetMaxIter( 2 );

    mfem::Vector solution( 1 );
    solution = 1.;
    mfem::Vector rightHandSide;
    solver.Mult( rightHandSide, solution );

    EXPECT_TRUE( solver.GetConverged() );
    EXPECT_EQ( solver.GetNumIterations(), 1 );
    EXPECT_NEAR( solution( 0 ), 0., kTightTolerance );
}

TEST( MultiNewtonAdaptive, AppliesTrialStateAtGlobalPseudoTime )
{
    ZeroResidualOperator nonlinearOperator;
    IdentitySolver linearSolver;
    plugin::MultiNewtonAdaptive<plugin::NewtonLineSearch> solver;
    solver.SetOperator( nonlinearOperator );
    solver.SetSolver( linearSolver );
    solver.iterative_mode = true;
    solver.SetMaxIter( 2 );
    solver.SetDelta( .4 );
    solver.SetMaxDelta( .4 );
    solver.SetMinDelta( 1e-12 );
    solver.SetPseudoTimeInterval( 1., 2. );

    std::vector<mfem::real_t> trialPseudoTimes;
    std::vector<mfem::real_t> acceptedPseudoTimes;
    solver.SetTrialStateFunc(
        [&]( const mfem::real_t pseudoTime, mfem::Vector& solution )
        {
            trialPseudoTimes.push_back( pseudoTime );
            solution( 0 ) = 10. * pseudoTime;
        } );
    solver.SetDataCollectionFunc( [&]( int, int, const mfem::real_t pseudoTime )
                                  { acceptedPseudoTimes.push_back( pseudoTime ); } );

    mfem::Vector solution( 1 );
    solution = 0.;
    mfem::Vector rightHandSide;
    solver.Mult( rightHandSide, solution );

    ASSERT_TRUE( solver.GetConverged() );
    ASSERT_EQ( trialPseudoTimes.size(), 3 );
    ASSERT_EQ( acceptedPseudoTimes.size(), 3 );
    EXPECT_NEAR( trialPseudoTimes[0], 1.4, kTightTolerance );
    EXPECT_NEAR( trialPseudoTimes[1], 1.8, kTightTolerance );
    EXPECT_NEAR( trialPseudoTimes[2], 2., kTightTolerance );
    EXPECT_EQ( trialPseudoTimes, acceptedPseudoTimes );
    EXPECT_NEAR( solver.GetCurrentPseudoTime(), 2., kTightTolerance );
    EXPECT_EQ( solver.GetPseudoTimeIncrement(), 0. );
    EXPECT_EQ( solver.StepNumber(), 3 );
    EXPECT_NEAR( solution( 0 ), 20., kTightTolerance );

    // The configured initial increment remains available for another interval.
    trialPseudoTimes.clear();
    acceptedPseudoTimes.clear();
    solver.SetPseudoTimeInterval( 2., 3. );
    solver.Mult( rightHandSide, solution );
    EXPECT_TRUE( solver.GetConverged() );
    EXPECT_NEAR( solver.GetCurrentPseudoTime(), 3., kTightTolerance );
    EXPECT_NEAR( solution( 0 ), 30., kTightTolerance );
}

TEST( MultiNewtonAdaptive, ReappliesTrialStateAfterPseudoTimeCutback )
{
    SwitchableResidualOperator nonlinearOperator;
    IdentitySolver linearSolver;
    plugin::MultiNewtonAdaptive<plugin::NewtonLineSearch> solver;
    solver.SetOperator( nonlinearOperator );
    solver.SetSolver( linearSolver );
    solver.iterative_mode = true;
    solver.SetMaxIter( 2 );
    solver.SetDelta( .5 );
    solver.SetMaxDelta( .5 );
    solver.SetMinDelta( 1e-12 );
    solver.SetPseudoTimeInterval( 0., 1. );

    bool rejectFirstAttempt = true;
    std::vector<mfem::real_t> trialPseudoTimes;
    std::vector<mfem::real_t> acceptedPseudoTimes;
    solver.SetTrialStateFunc(
        [&]( const mfem::real_t pseudoTime, mfem::Vector& solution )
        {
            trialPseudoTimes.push_back( pseudoTime );
            nonlinearOperator.SetReject( rejectFirstAttempt );
            rejectFirstAttempt = false;
            solution( 0 ) = pseudoTime;
        } );
    solver.SetDataCollectionFunc( [&]( int, int, const mfem::real_t pseudoTime )
                                  { acceptedPseudoTimes.push_back( pseudoTime ); } );

    mfem::Vector solution( 1 );
    solution = 0.;
    mfem::Vector rightHandSide;
    solver.Mult( rightHandSide, solution );

    ASSERT_TRUE( solver.GetConverged() );
    ASSERT_GE( trialPseudoTimes.size(), 3 );
    ASSERT_FALSE( acceptedPseudoTimes.empty() );
    EXPECT_NEAR( trialPseudoTimes[0], .5, kTightTolerance );
    EXPECT_NEAR( trialPseudoTimes[1], .25, kTightTolerance );
    EXPECT_NEAR( acceptedPseudoTimes[0], .25, kTightTolerance );
    EXPECT_NEAR( acceptedPseudoTimes.back(), 1., kTightTolerance );
    EXPECT_NEAR( solver.GetCurrentPseudoTime(), 1., kTightTolerance );
    EXPECT_NEAR( solution( 0 ), 1., kTightTolerance );
}

TEST( MultiNewtonAdaptive, ThrowingTrialStateRestoresAcceptedPseudoTimeAndSolution )
{
    ZeroResidualOperator nonlinearOperator;
    IdentitySolver linearSolver;
    plugin::MultiNewtonAdaptive<plugin::NewtonLineSearch> solver;
    solver.SetOperator( nonlinearOperator );
    solver.SetSolver( linearSolver );
    solver.iterative_mode = true;
    solver.SetDelta( .5 );
    solver.SetPseudoTimeInterval( 1., 2. );
    solver.SetTrialStateFunc(
        []( const mfem::real_t, mfem::Vector& solution )
        {
            solution( 0 ) = 99.;
            throw std::runtime_error( "trial-state failure" );
        } );

    mfem::Vector solution( 1 );
    solution = 3.;
    mfem::Vector rightHandSide;
    EXPECT_THROW( solver.Mult( rightHandSide, solution ), std::runtime_error );

    EXPECT_FALSE( solver.GetConverged() );
    EXPECT_EQ( solution( 0 ), 3. );
    EXPECT_EQ( solver.GetCurrentPseudoTime(), 1. );
    EXPECT_EQ( solver.GetPseudoTimeIncrement(), 0. );
    EXPECT_EQ( solver.StepNumber(), 0 );
}

TEST( MultiNewtonAdaptive, ThrowingAcceptedStepCallbackRetainsCommittedStepBookkeeping )
{
    ZeroResidualOperator nonlinearOperator;
    IdentitySolver linearSolver;
    plugin::MultiNewtonAdaptive<plugin::NewtonLineSearch> solver;
    solver.SetOperator( nonlinearOperator );
    solver.SetSolver( linearSolver );
    solver.iterative_mode = true;
    solver.SetDelta( .5 );
    solver.SetPseudoTimeInterval( 0., 1. );
    solver.SetTrialStateFunc( []( const mfem::real_t pseudoTime, mfem::Vector& solution ) { solution( 0 ) = pseudoTime; } );
    solver.SetDataCollectionFunc( []( int, int, const mfem::real_t )
                                  { throw std::runtime_error( "accepted-step output failure" ); } );

    mfem::Vector solution( 1 );
    solution = 0.;
    mfem::Vector rightHandSide;
    EXPECT_THROW( solver.Mult( rightHandSide, solution ), std::runtime_error );

    EXPECT_FALSE( solver.GetConverged() );
    EXPECT_EQ( solution( 0 ), .5 );
    EXPECT_EQ( solver.GetCurrentPseudoTime(), .5 );
    EXPECT_EQ( solver.GetPseudoTimeIncrement(), 0. );
    EXPECT_EQ( solver.StepNumber(), 1 );
}

TEST( MultiNewtonAdaptive, CutbackRollsBackAndAcceptedPseudoTimeCommitsIntegratorState )
{
    mfem::Mesh mesh = mfem::Mesh::MakeCartesian1D( 1 );
    mfem::H1_FECollection collection( 1, mesh.Dimension() );
    mfem::FiniteElementSpace space( &mesh, &collection );
    mfem::NonlinearForm nonlinearOperator( &space );
    auto* integrator = new PoisonableIntegrator;
    nonlinearOperator.AddDomainIntegrator( integrator );

    IdentitySolver linearSolver;
    plugin::MultiNewtonAdaptive<plugin::NewtonLineSearch> solver;
    solver.SetOperator( nonlinearOperator );
    solver.SetSolver( linearSolver );
    solver.iterative_mode = true;
    solver.SetMaxIter( 1 );
    solver.SetDelta( .5 );
    solver.SetMaxDelta( .5 );
    solver.SetMinDelta( 1e-12 );

    bool rejectFirstAttempt = true;
    solver.SetTrialStateFunc(
        [&]( const mfem::real_t, mfem::Vector& )
        {
            integrator->SetResidual( rejectFirstAttempt ? 1. : 0. );
            rejectFirstAttempt = false;
        } );

    mfem::Vector solution( space.GetTrueVSize() );
    solution = 0.;
    mfem::Vector rightHandSide;
    solver.Mult( rightHandSide, solution );

    EXPECT_TRUE( solver.GetConverged() );
    EXPECT_EQ( integrator->Rollbacks(), 1 );
    EXPECT_EQ( integrator->Commits(), solver.StepNumber() );
    EXPECT_NEAR( solver.GetCurrentPseudoTime(), 1., kTightTolerance );
}
