#include "PhaseField.h"
#include "PostProc.h"
#include "Solvers.h"

#include <Eigen/Dense>
#include <cmath>
#include <functional>
#include <gtest/gtest.h>
#include <type_traits>

namespace
{
using Split = PhaseFieldElasticMaterial::StrainEnergySplit;
constexpr bool kSingle = std::is_same_v<mfem::real_t, float>;
constexpr mfem::real_t kStressTol = kSingle ? 2e-4f : 2e-10;
constexpr mfem::real_t kTangentTol = kSingle ? 8e-2f : 3e-5;
constexpr mfem::real_t kDifferenceStep = kSingle ? 2e-3f : 1e-6;
constexpr Split kSplits[] = { Split::MieheSpectral, Split::AmorVolumetricDeviatoric, Split::Isotropic };

mfem::Array<mfem::FiniteElementSpace*> Spaces( mfem::FiniteElementSpace& u, mfem::FiniteElementSpace& phi )
{
    mfem::Array<mfem::FiniteElementSpace*> spaces( 2 );
    spaces[0] = &u;
    spaces[1] = &phi;
    return spaces;
}

struct MaterialPoint
{
    mfem::Mesh mesh{ mfem::Mesh::MakeCartesian3D( 1, 1, 1, mfem::Element::HEXAHEDRON, 1., 1., 1. ) };
    mfem::ConstantCoefficient youngs{ 10. }, poisson{ .25 };

    void Set( PhaseFieldElasticMaterial& material, const Eigen::Vector6r& strain, mfem::real_t phi )
    {
        auto& transformation = *mesh.GetElementTransformation( 0 );
        const auto& point = mfem::Geometries.GetCenter( mfem::Geometry::CUBE );
        transformation.SetIntPoint( &point );
        material.at( transformation, point );
        material.setMechanicalStrain( util::InverseVoigt( strain, true ) );
        material.setPhaseField( phi );
    }
};

class CountingCoefficient : public mfem::ConstantCoefficient
{
public:
    using mfem::ConstantCoefficient::ConstantCoefficient;
    int calls{ 0 };
    mfem::real_t Eval( mfem::ElementTransformation& transformation, const mfem::IntegrationPoint& point ) override
    {
        ++calls;
        return mfem::ConstantCoefficient::Eval( transformation, point );
    }
};

class FixedStepContext final : public plugin::NonlinearStepContext
{
public:
    bool Convergence() const override
    {
        return true;
    }
};

struct ElementProblem
{
    mfem::Mesh mesh{ mfem::Mesh::MakeCartesian2D( 1, 1, mfem::Element::QUADRILATERAL, true, 1., 1. ) };
    mfem::H1_FECollection collection{ 1, 2 };
    mfem::FiniteElementSpace uSpace{ &mesh, &collection, 2, mfem::Ordering::byVDIM };
    mfem::FiniteElementSpace phiSpace{ &mesh, &collection };
    CountingCoefficient youngs{ 10. }, poisson{ .25 };
    PhaseFieldElasticMaterial material;
    plugin::PhaseFieldPointStorage storage{ &mesh };
    FixedStepContext context;
    mfem::Array<mfem::FiniteElementSpace*> spaces{ Spaces( uSpace, phiSpace ) };
    mfem::BlockNonlinearForm form{ spaces };
    mfem::BlockVector state{ form.GetBlockTrueOffsets() };
    plugin::PhaseFieldIntegrator<plugin::PhaseFieldPointStorage>* integrator;

    explicit ElementProblem( Split split ) : material( youngs, poisson, split, { 2.5, .3, .01 } )
    {
        integrator = new plugin::PhaseFieldIntegrator<plugin::PhaseFieldPointStorage>( material, storage );
        form.AddDomainIntegrator( integrator ); // Form owns the integrator.
        integrator->SetStepContext( &context );
        mfem::GridFunction u( &uSpace ), phi( &phiSpace );
        mfem::VectorFunctionCoefficient displacement( 2,
                                                      []( const mfem::Vector& x, mfem::Vector& value )
                                                      {
                                                          value.SetSize( 2 );
                                                          value( 0 ) = .018 * x( 0 ) + .004 * x( 1 );
                                                          value( 1 ) = .004 * x( 0 ) - .007 * x( 1 );
                                                      } );
        mfem::FunctionCoefficient damage( []( const mfem::Vector& x ) { return .2 + .03 * x( 0 ) + .02 * x( 1 ); } );
        u.ProjectCoefficient( displacement );
        phi.ProjectCoefficient( damage );
        u.GetTrueDofs( state.GetBlock( 0 ) );
        phi.GetTrueDofs( state.GetBlock( 1 ) );
    }

    void CheckJacobian( bool activeHistory )
    {
        youngs.calls = poisson.calls = 0;
        auto& jacobian = form.GetGradient( state );
        const int points = mfem::IntRules.Get( mfem::Geometry::SQUARE, 3 ).GetNPoints();
        EXPECT_EQ( youngs.calls, points );
        EXPECT_EQ( poisson.calls, points );
        mfem::Vector residual( state.Size() );
        youngs.calls = poisson.calls = 0;
        form.Mult( state, residual );
        EXPECT_EQ( youngs.calls, points );
        EXPECT_EQ( poisson.calls, points );

        // Excite each input block separately and check both output blocks.
        for ( int input = 0; input < 2; ++input )
        {
            mfem::BlockVector direction( form.GetBlockTrueOffsets() );
            direction = 0.;
            auto& block = direction.GetBlock( input );
            for ( int i = 0; i < block.Size(); ++i )
            {
                block( i ) = ( i % 7 ) - 3.;
            }
            direction /= direction.Norml2();
            mfem::Vector plus( state ), minus( state ), rp( state.Size() ), rm( state.Size() );
            mfem::BlockVector analytical( form.GetBlockTrueOffsets() ), numerical( form.GetBlockTrueOffsets() );
            jacobian.Mult( direction, analytical );
            plus.Add( kDifferenceStep, direction );
            minus.Add( -kDifferenceStep, direction );
            form.Mult( plus, rp );
            form.Mult( minus, rm );
            subtract( rp, rm, numerical );
            numerical /= 2 * kDifferenceStep;
            if ( input == 0 && !activeHistory )
            {
                EXPECT_EQ( analytical.GetBlock( 1 ).Norml2(), 0. );
            }
            for ( int output = 0; output < 2; ++output )
            {
                const mfem::real_t scale = 1. + numerical.GetBlock( output ).Norml2();
                analytical.GetBlock( output ) -= numerical.GetBlock( output );
                EXPECT_LE( analytical.GetBlock( output ).Norml2(), kTangentTol * scale );
            }
        }
    }
};

// Small algebraic problems isolate solver convergence and transaction behavior
// from constitutive/assembly errors. All diagonal blocks are scalar.
class ScalarSolver final : public mfem::Solver
{
public:
    explicit ScalarSolver( mfem::real_t fraction = 1. ) : fraction( fraction )
    {
    }
    std::function<void()> beforeSolve;
    mutable int calls{ 0 };
    mutable mfem::real_t lastRhs{ 0. };
    void SetOperator( const mfem::Operator& op ) override
    {
        MFEM_VERIFY( op.Height() == 1 && op.Width() == 1, "Expected scalar test block." );
        mfem::Vector one( 1 ), value( 1 );
        one = 1.;
        op.Mult( one, value );
        diagonal = value( 0 );
    }
    void Mult( const mfem::Vector& rhs, mfem::Vector& x ) const override
    {
        if ( beforeSolve )
        {
            beforeSolve();
        }
        ++calls;
        lastRhs = rhs( 0 );
        x = rhs;
        x *= fraction / diagonal;
    }

private:
    mfem::real_t fraction, diagonal{ 1. };
};

class HistoryProbe final : public plugin::BlockStepAwareNonlinearFormIntegrator
{
public:
    plugin::PhaseFieldHistory history;
    int begins{ 0 }, commits{ 0 }, rollbacks{ 0 };
    void BeginStep() noexcept override
    {
        BlockStepAwareNonlinearFormIntegrator::BeginStep();
        ++begins;
        history.BeginStep();
    }
    void CommitStep() noexcept override
    {
        ++commits;
        history.CommitStep();
        BlockStepAwareNonlinearFormIntegrator::CommitStep();
    }
    void RollbackStep() noexcept override
    {
        ++rollbacks;
        history.RollbackStep();
        BlockStepAwareNonlinearFormIntegrator::RollbackStep();
    }
};

class TwoBlockForm final : public mfem::BlockNonlinearForm
{
public:
    HistoryProbe* probe;
    mutable mfem::real_t currentU{ 0. }, currentPhi{ 0. };
    TwoBlockForm( mfem::Array<mfem::FiniteElementSpace*>& spaces, mfem::real_t coupling = 0., bool nonlinear = false )
        : mfem::BlockNonlinearForm( spaces ), coupling( coupling ), nonlinear( nonlinear ), gradient( GetBlockTrueOffsets() )
    {
        probe = new HistoryProbe;
        AddDomainIntegrator( probe );
        uu = pp = 1.;
        up = nonlinear ? -1. : coupling;
        pu = nonlinear ? -.2 : coupling;
        gradient.SetBlock( 0, 0, &uu );
        gradient.SetBlock( 0, 1, &up );
        gradient.SetBlock( 1, 0, &pu );
        gradient.SetBlock( 1, 1, &pp );
    }
    void Mult( const mfem::Vector& x, mfem::Vector& r ) const override
    {
        r.SetSize( 2 );
        r( 0 ) = nonlinear ? x( 0 ) + std::pow( x( 0 ), 3 ) - 1. - x( 1 ) : x( 0 ) + coupling * x( 1 );
        r( 1 ) = x( 1 ) + ( nonlinear ? -.2 : coupling ) * x( 0 );
        probe->history.EvaluateTrial( x( 0 ) * x( 0 ) );
    }
    mfem::Operator& GetGradient( const mfem::Vector& x ) const override
    {
        currentU = x( 0 );
        currentPhi = x( 1 );
        uu = nonlinear ? 1. + 3. * x( 0 ) * x( 0 ) : 1.;
        return gradient;
    }

private:
    mfem::real_t coupling;
    bool nonlinear;
    mutable mfem::DenseMatrix uu{ 1 }, up{ 1 }, pu{ 1 }, pp{ 1 };
    mutable mfem::BlockOperator gradient;
};

class PhaseFieldSolverTest : public testing::Test
{
protected:
    mfem::Mesh mesh{ mfem::Mesh::MakeCartesian1D( 1 ) };
    mfem::L2_FECollection collection{ 0, 1 };
    mfem::FiniteElementSpace uSpace{ &mesh, &collection }, phiSpace{ &mesh, &collection };
    mfem::Array<mfem::FiniteElementSpace*> spaces{ Spaces( uSpace, phiSpace ) };
    mfem::Vector state, rhs;
    const mfem::real_t tolerance = kSingle ? 1e-4f : 1e-9;
    void SetUp() override
    {
        state.SetSize( 2 );
        rhs.SetSize( 2 );
        state = 0.;
        rhs = 0.;
    }
};

static_assert( !std::is_copy_constructible_v<PhaseFieldElasticMaterial> );
} // namespace

TEST( PhaseFieldMaterial, EngineeringShearAndCompressionLimits )
{
    MaterialPoint point;
    constexpr mfem::real_t gamma = .04, mu = 4., phi = .6, k = .03;
    const mfem::real_t g = ( 1. - k ) * ( 1. - phi ) * ( 1. - phi ) + k;
    for ( Split split : kSplits )
    {
        PhaseFieldElasticMaterial material( point.youngs, point.poisson, split, { 2700., .015e-3, k } );
        Eigen::Vector6r strain = Eigen::Vector6r::Zero();
        for ( int shear = 3; shear < 6; ++shear )
        {
            strain.setZero();
            strain( shear ) = gamma;
            point.Set( material, strain, 0. );
            EXPECT_NEAR( material.getPK2StressVector()( shear ), mu * gamma, kStressTol );
            material.updateRefModuli();
            EXPECT_NEAR( material.getRefModuli()( shear, shear ), mu, kStressTol );
            EXPECT_NEAR( material.getPsiPos(), mu * gamma * gamma / ( split == Split::MieheSpectral ? 4. : 2. ), kStressTol );
            point.Set( material, strain, phi );
            EXPECT_NEAR( material.getPK2StressVector()( shear ),
                         mu * gamma * ( split == Split::MieheSpectral ? ( 1. + g ) / 2. : g ), kStressTol );
        }
        strain.setZero();
        strain.head<3>().setConstant( -.01 );
        point.Set( material, strain, 0. );
        const Eigen::Vector6r intact = material.getPK2StressVector();
        point.Set( material, strain, phi );
        EXPECT_LE( ( material.getPK2StressVector() - ( split == Split::Isotropic ? g : 1. ) * intact ).norm(), kStressTol );
        if ( split != Split::Isotropic )
        {
            EXPECT_NEAR( material.getPsiPos(), 0., kStressTol );
        }
    }
}

TEST( PhaseFieldMaterial, ResponseSnapshotAndIntactZeroStrainTangent )
{
    MaterialPoint point;
    Eigen::Matrix6r elastic = Eigen::Matrix6r::Zero();
    elastic.topLeftCorner<3, 3>().setConstant( 4. );
    elastic.diagonal().head<3>().setConstant( 12. );
    elastic.diagonal().tail<3>().setConstant( 4. );
    Eigen::Vector6r strain;
    strain << .018, -.007, .003, .008, .002, -.004;
    for ( Split split : kSplits )
    {
        PhaseFieldElasticMaterial material( point.youngs, point.poisson, split );
        point.Set( material, Eigen::Vector6r::Zero(), 0. );
        const auto zero = material.EvaluateResponse( true );
        EXPECT_LE( zero.stress.norm(), kStressTol );
        EXPECT_NEAR( zero.positiveEnergy, 0., kStressTol );
        ASSERT_TRUE( zero.tangent );
        EXPECT_LE( ( *zero.tangent - elastic ).norm(), kStressTol * elastic.norm() );
        point.Set( material, strain, .3 );
        const auto response = material.EvaluateResponse( true );
        const auto stressOnly = material.EvaluateResponse( false );
        ASSERT_TRUE( response.tangent );
        EXPECT_FALSE( stressOnly.tangent );
        EXPECT_EQ( response.positiveEnergy, material.getPsiPos() );
        EXPECT_EQ( stressOnly.positiveEnergy, response.positiveEnergy );
        EXPECT_LE( ( response.stress - material.getPK2StressVector() ).norm(), kStressTol );
        EXPECT_LE( ( response.positiveStress - material.getPositiveStressVector() ).norm(), kStressTol );
        EXPECT_LE( ( response.phaseStressDerivative - material.getPhaseStressDerivative() ).norm(), kStressTol );
        EXPECT_LE( ( stressOnly.stress - response.stress ).norm(), kStressTol );
        EXPECT_LE( ( stressOnly.positiveStress - response.positiveStress ).norm(), kStressTol );
        EXPECT_LE( ( stressOnly.phaseStressDerivative - response.phaseStressDerivative ).norm(), kStressTol );
        material.updateRefModuli();
        EXPECT_LE( ( *response.tangent - material.getRefModuli() ).norm(), kStressTol );
        material.setPhaseField( .7 );
        EXPECT_GT( ( material.EvaluateResponse( false ).stress - response.stress ).norm(), kStressTol );
    }
}

TEST( PhaseFieldMaterial, TangentsMatchCenteredDifferencesIncludingRepeatedRoots )
{
    MaterialPoint point;
    Eigen::Vector6r direction;
    direction << -.2, .3, .1, .4, -.25, .15;
    direction.normalize();
    Eigen::Vector6r mixed;
    mixed << .018, -.007, .004, .006, -.003, .002;
    for ( Split split : kSplits )
    {
        PhaseFieldElasticMaterial material( point.youngs, point.poisson, split );
        for ( int mode = 0; mode < 3; ++mode )
        {
            Eigen::Vector6r strain = mixed;
            if ( mode > 0 )
            {
                strain.setZero();
                strain.head<3>().setConstant( mode == 1 ? -.02 : .02 );
            }
            point.Set( material, strain, .35 );
            const auto response = material.EvaluateResponse( true );
            ASSERT_TRUE( response.tangent->allFinite() );
            EXPECT_LE( ( *response.tangent - response.tangent->transpose() ).norm(),
                       kStressTol * ( 1. + response.tangent->norm() ) );
            point.Set( material, strain + kDifferenceStep * direction, .35 );
            const Eigen::Vector6r plus = material.getPK2StressVector();
            point.Set( material, strain - kDifferenceStep * direction, .35 );
            const Eigen::Vector6r minus = material.getPK2StressVector();
            const Eigen::Vector6r numerical = ( plus - minus ) / ( 2 * kDifferenceStep );
            EXPECT_LE( ( *response.tangent * direction - numerical ).norm(), kTangentTol * ( 1. + numerical.norm() ) );
        }
    }
}

TEST( PhaseFieldMaterial, AmorVolumetricSwitchAndFullyCrackedSpectralLimit )
{
    MaterialPoint point;
    PhaseFieldElasticMaterial amor( point.youngs, point.poisson, Split::AmorVolumetricDeviatoric, { 2700., .015e-3, .03 } );
    Eigen::Vector6r volumetric = Eigen::Vector6r::Zero();
    volumetric.head<3>().setOnes();
    const mfem::real_t g = .97 * .4 * .4 + .03;
    const mfem::real_t bulk = 10. / ( 3 * ( 1 - 2 * .25 ) );
    for ( mfem::real_t dilation : { -.02, 0., .02 } )
    {
        Eigen::Vector6r strain = dilation * volumetric;
        strain( 3 ) = .04;
        point.Set( amor, strain, .6 );
        const auto response = amor.EvaluateResponse( true );
        const mfem::real_t weight = dilation > 0. ? g : ( dilation < 0. ? 1. : ( 1. + g ) / 2 );
        const Eigen::Vector6r expected = 3 * bulk * weight * volumetric;
        EXPECT_LE( ( *response.tangent * volumetric - expected ).norm(), kStressTol * ( 1. + expected.norm() ) );
        point.Set( amor, strain + kDifferenceStep * volumetric, .6 );
        const Eigen::Vector6r plus = amor.getPK2StressVector();
        point.Set( amor, strain - kDifferenceStep * volumetric, .6 );
        EXPECT_LE( ( ( plus - amor.getPK2StressVector() ) / ( 2 * kDifferenceStep ) - expected ).norm(),
                   kTangentTol * ( 1. + expected.norm() ) );
    }
    PhaseFieldElasticMaterial spectral( point.youngs, point.poisson, Split::MieheSpectral, { 2700., .015e-3, 1e-20 } );
    point.Set( spectral, .02 * volumetric, 0. );
    const auto intact = spectral.EvaluateResponse( true );
    spectral.setPhaseField( 1. );
    const auto cracked = spectral.EvaluateResponse( true );
    EXPECT_LE( ( cracked.stress / spectral.getK() - intact.stress ).norm(), kStressTol * intact.stress.norm() );
    EXPECT_LE( ( *cracked.tangent / spectral.getK() - *intact.tangent ).norm(), kStressTol * intact.tangent->norm() );
}

TEST( PhaseFieldIntegrator, AllFourJacobianBlocksOnLoadingAndUnloading )
{
    for ( Split split : kSplits )
    {
        ElementProblem problem( split );
        problem.integrator->BeginStep();
        problem.CheckJacobian( true );
        mfem::Vector residual( problem.state.Size() );
        problem.form.Mult( problem.state, residual );
        problem.integrator->CommitStep();
        problem.integrator->BeginStep();
        problem.state.GetBlock( 0 ) *= .5;
        problem.CheckJacobian( false );
        problem.integrator->RollbackStep();
    }
}

TEST( PhaseFieldIntegrator, HomogeneousAT2BalanceUsesCorrectLengthConvention )
{
    ElementProblem problem( Split::MieheSpectral );
    // The prescribed affine u has eps_xx=.018, eps_yy=-.007, eps_xy=.004,
    // eps_zz=0. With lambda=mu=4, its only positive principal strain is:
    const mfem::real_t positivePrincipal = ( .011 + std::sqrt( .025 * .025 + 4 * .004 * .004 ) ) / 2;
    const mfem::real_t history = 2 * .011 * .011 + 4 * positivePrincipal * positivePrincipal;
    const mfem::real_t driving = 2 * ( 1 - problem.material.getK() ) * history;
    const mfem::real_t equilibriumPhase = driving / ( problem.material.getGc() / problem.material.getL0() + driving );
    mfem::BlockVector residual( problem.form.GetBlockTrueOffsets() );
    problem.integrator->BeginStep();
    problem.state.GetBlock( 1 ) = 0.;
    problem.form.Mult( problem.state, residual );
    EXPECT_NEAR( residual.GetBlock( 1 ).Sum(), -driving, kStressTol );
    problem.state.GetBlock( 1 ) = equilibriumPhase;
    problem.form.Mult( problem.state, residual );
    EXPECT_LE( residual.GetBlock( 1 ).Norml2(), kStressTol );
    problem.integrator->RollbackStep();
}

TEST_F( PhaseFieldSolverTest, InitialPhaseResidualAndNonzeroRhsUseConfiguredSolvers )
{
    TwoBlockForm form( spaces );
    ScalarSolver uSolver, phiSolver, sharedSolver;
    plugin::NewtonForPhaseField solver;
    solver.SetOperator( form );
    solver.SetBlockSolvers( uSolver, phiSolver );
    solver.SetRelTol( 0. );
    solver.SetAbsTol( tolerance );
    solver.SetMaxIter( 3 );
    solver.iterative_mode = true;
    rhs( 1 ) = 2.;
    solver.Mult( rhs, state );
    ASSERT_TRUE( solver.GetConverged() );
    EXPECT_EQ( state( 0 ), 0. );
    EXPECT_EQ( state( 1 ), 2. );
    EXPECT_EQ( uSolver.calls, 0 );
    EXPECT_EQ( phiSolver.calls, 1 );
    EXPECT_EQ( phiSolver.lastRhs, -2. );
    state = 0.;
    rhs( 0 ) = 1.;
    solver.SetSolver( sharedSolver );
    solver.Mult( rhs, state );
    EXPECT_TRUE( solver.GetConverged() );
    EXPECT_EQ( sharedSolver.calls, 2 );
    EXPECT_EQ( phiSolver.calls, 1 );
    EXPECT_EQ( state( 0 ), 1. );
    EXPECT_EQ( state( 1 ), 2. );
}

TEST_F( PhaseFieldSolverTest, BothCoupledResidualsMustPassAndRejectedStepsRollBack )
{
    constexpr mfem::real_t coupling = .9;
    TwoBlockForm form( spaces, coupling );
    ScalarSolver linear;
    plugin::NewtonForPhaseField solver;
    solver.SetOperator( form );
    solver.SetSolver( linear );
    solver.SetRelTol( 0. );
    solver.SetAbsTol( tolerance );
    solver.iterative_mode = true;
    for ( int driven = 0; driven < 2; ++driven )
    {
        state = 0.;
        rhs = 0.;
        rhs( driven ) = 1.;
        solver.SetMaxIter( 1 );
        solver.Mult( rhs, state );
        EXPECT_FALSE( solver.GetConverged() );
        EXPECT_EQ( state.Norml2(), 0. );
        solver.SetMaxIter( 150 );
        solver.Mult( rhs, state );
        ASSERT_TRUE( solver.GetConverged() );
        EXPECT_GT( solver.GetNumIterations(), 1 );
        mfem::Vector residual;
        form.Mult( state, residual );
        residual -= rhs;
        EXPECT_LE( std::abs( residual( 0 ) ), tolerance );
        EXPECT_LE( std::abs( residual( 1 ) ), tolerance );
        EXPECT_NEAR( solver.GetFinalNorm(), residual.Norml2(), kStressTol );
        EXPECT_NEAR( state( driven ), 1. / ( 1 - coupling * coupling ), 2 * tolerance / ( 1 - coupling ) );
    }
}

TEST_F( PhaseFieldSolverTest, BlockAbsoluteFloorsAndSharedFallback )
{
    TwoBlockForm form( spaces );
    ScalarSolver linear;
    plugin::NewtonForPhaseField solver;
    solver.SetOperator( form );
    solver.SetSolver( linear );
    solver.SetRelTol( 0. );
    solver.SetAbsTol( 0. );
    solver.SetMaxIter( 0 );
    solver.iterative_mode = true;
    rhs( 0 ) = 1e-3;
    rhs( 1 ) = 1e-7;
    solver.SetBlockAbsTol( 1e-2, 1e-6 );
    solver.Mult( rhs, state );
    EXPECT_TRUE( solver.GetConverged() );
    solver.SetBlockAbsTol( 1e-2, 1e-9 );
    solver.Mult( rhs, state );
    EXPECT_FALSE( solver.GetConverged() );
    solver.SetBlockAbsTol( 1e-4, 1e-6 );
    solver.Mult( rhs, state );
    EXPECT_FALSE( solver.GetConverged() );
    solver.ClearBlockAbsTol();
    solver.SetAbsTol( 1e-2 );
    solver.Mult( rhs, state );
    EXPECT_TRUE( solver.GetConverged() );
    solver.SetBlockAbsTol( 1e-5, 1e-6 );
    solver.SetMaxIter( 2 );
    solver.SetMechanicsNewton( 2, 0., 1. ); // Inner goal must be capped by the outer goal.
    solver.Mult( rhs, state );
    EXPECT_TRUE( solver.GetConverged() );
    EXPECT_EQ( solver.GetNumMechanicsIterations(), 1 );
    EXPECT_EQ( state( 0 ), rhs( 0 ) );
}

TEST_F( PhaseFieldSolverTest, InnerNewtonHoldsPhaseFixedAndCommitsOnlyAfterOuterConvergence )
{
    TwoBlockForm form( spaces, 0., true );
    ScalarSolver uSolver, phiSolver;
    plugin::NewtonForPhaseField solver;
    solver.SetOperator( form );
    solver.SetBlockSolvers( uSolver, phiSolver );
    solver.SetRelTol( 0. );
    solver.SetBlockAbsTol( tolerance, tolerance );
    solver.SetMechanicsNewton( 30, 0., tolerance / 10 );
    solver.SetMaxIter( 40 );
    solver.iterative_mode = true;
    int previousPhaseSolves = -1;
    mfem::real_t fixedPhase = 0.;
    uSolver.beforeSolve = [&]()
    {
        if ( previousPhaseSolves != phiSolver.calls )
        {
            previousPhaseSolves = phiSolver.calls;
            fixedPhase = form.currentPhi;
        }
        EXPECT_EQ( form.currentPhi, fixedPhase );
        EXPECT_EQ( form.probe->commits, 0 );
    };
    phiSolver.beforeSolve = [&]()
    {
        const auto u = form.currentU;
        EXPECT_LE( std::abs( u + u * u * u - 1. - form.currentPhi ), tolerance / 10 );
        EXPECT_EQ( form.probe->commits, 0 );
    };
    solver.Mult( rhs, state );
    ASSERT_TRUE( solver.GetConverged() );
    EXPECT_GT( solver.GetMaxMechanicsIterationsUsed(), 1 );
    EXPECT_GT( solver.GetNumIterations(), 1 );
    EXPECT_EQ( solver.GetNumMechanicsIterations(), uSolver.calls );
    EXPECT_EQ( form.probe->begins, 1 );
    EXPECT_EQ( form.probe->commits, 1 );
    EXPECT_EQ( form.probe->rollbacks, 0 );
    mfem::Vector residual;
    form.Mult( state, residual );
    EXPECT_LE( std::abs( residual( 0 ) ), tolerance );
    EXPECT_LE( std::abs( residual( 1 ) ), tolerance );
    EXPECT_EQ( form.probe->history.CommittedValue(), state( 0 ) * state( 0 ) );
}

TEST_F( PhaseFieldSolverTest, InnerFailureRestoresUnknownsAndHistoryBeforePhaseSolve )
{
    TwoBlockForm form( spaces, 0., true );
    ScalarSolver uSolver, phiSolver;
    plugin::NewtonForPhaseField solver;
    solver.SetOperator( form );
    solver.SetBlockSolvers( uSolver, phiSolver );
    solver.SetRelTol( 0. );
    solver.SetAbsTol( tolerance );
    solver.SetMechanicsNewton( 1, 0., tolerance );
    solver.SetMaxIter( 20 );
    solver.iterative_mode = true;
    solver.Mult( rhs, state );
    EXPECT_FALSE( solver.GetConverged() );
    EXPECT_EQ( state.Norml2(), 0. );
    EXPECT_EQ( uSolver.calls, 1 );
    EXPECT_EQ( phiSolver.calls, 0 );
    EXPECT_EQ( form.probe->begins, 1 );
    EXPECT_EQ( form.probe->commits, 0 );
    EXPECT_EQ( form.probe->rollbacks, 1 );
    EXPECT_EQ( form.probe->history.CommittedValue(), 0. );
    EXPECT_EQ( form.probe->history.TrialValue(), 0. );
}

TEST_F( PhaseFieldSolverTest, PhaseReferenceUsesTheCompletedMechanicalSubsolve )
{
    TwoBlockForm form( spaces, 0., true );
    ScalarSolver uSolver, phiSolver( .4 );
    plugin::NewtonForPhaseField solver;
    solver.SetOperator( form );
    solver.SetBlockSolvers( uSolver, phiSolver );
    solver.SetAbsTol( 0. );
    solver.SetRelTol( .5 );
    solver.SetMaxIter( 1 );
    solver.SetMechanicsNewton( 30, 0., kSingle ? 1e-5f : 1e-12 );
    solver.iterative_mode = true;
    solver.Mult( rhs, state );
    // u=1 after the first correction would give phase goal .1. The converged
    // u~.6823 gives goal ~.0682, which rejects the deliberately partial phi solve.
    EXPECT_FALSE( solver.GetConverged() );
    EXPECT_GT( solver.GetMaxMechanicsIterationsUsed(), 1 );
    EXPECT_EQ( phiSolver.calls, 1 );
    EXPECT_EQ( state.Norml2(), 0. );
}

TEST( StressCoefficient, UsesCurrentPhaseFieldValue )
{
    ElementProblem problem( Split::MieheSpectral );
    mfem::GridFunction displacement( &problem.uSpace ), phase( &problem.phiSpace );
    mfem::VectorFunctionCoefficient prescribed( 2,
                                                []( const mfem::Vector& x, mfem::Vector& u )
                                                {
                                                    u.SetSize( 2 );
                                                    u( 0 ) = .01 * x( 0 );
                                                    u( 1 ) = 0.;
                                                } );
    displacement.ProjectCoefficient( prescribed );
    plugin::StressCoefficient coefficient( 2, problem.material );
    coefficient.SetDisplacement( displacement );
    coefficient.SetPhaseField( phase );
    auto& transformation = *problem.mesh.GetElementTransformation( 0 );
    const auto& point = mfem::Geometries.GetCenter( mfem::Geometry::SQUARE );
    mfem::Vector intact, degraded;
    phase = 0.;
    coefficient.Eval( intact, transformation, point );
    phase = .75;
    coefficient.Eval( degraded, transformation, point );
    const auto k = problem.material.getK();
    const auto g = ( 1 - k ) * .25 * .25 + k;
    for ( int i = 0; i < 6; ++i )
    {
        EXPECT_NEAR( degraded( i ), g * intact( i ), kStressTol * ( 1. + std::abs( intact( i ) ) ) );
    }
}

TEST( StressCoefficient, VonMisesIncludesEveryShearComponent )
{
    MaterialPoint point;
    mfem::H1_FECollection collection( 1, 3 );
    mfem::FiniteElementSpace space( &point.mesh, &collection, 3, mfem::Ordering::byVDIM );
    mfem::GridFunction u( &space );
    mfem::VectorFunctionCoefficient prescribed( 3,
                                                []( const mfem::Vector& x, mfem::Vector& value )
                                                {
                                                    value.SetSize( 3 );
                                                    value( 0 ) = .02 * x( 1 );
                                                    value( 1 ) = .03 * x( 2 );
                                                    value( 2 ) = .04 * x( 0 );
                                                } );
    u.ProjectCoefficient( prescribed );
    IsotropicElasticMaterial material( point.youngs, point.poisson );
    plugin::StressCoefficient coefficient( 3, material );
    coefficient.SetDisplacement( u );
    mfem::Vector stress;
    coefficient.Eval( stress, *point.mesh.GetElementTransformation( 0 ), mfem::Geometries.GetCenter( mfem::Geometry::CUBE ) );
    const auto expected =
        std::sqrt( .5 * ( std::pow( stress( 0 ) - stress( 1 ), 2 ) + std::pow( stress( 1 ) - stress( 2 ), 2 ) +
                          std::pow( stress( 2 ) - stress( 0 ), 2 ) ) +
                   3 * ( std::pow( stress( 3 ), 2 ) + std::pow( stress( 4 ), 2 ) + std::pow( stress( 5 ), 2 ) ) );
    EXPECT_GT( std::abs( stress( 3 ) ) + std::abs( stress( 4 ) ), 0. );
    EXPECT_NEAR( stress( 6 ), expected, kStressTol * ( 1. + expected ) );
}
