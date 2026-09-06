#include "PhaseField.h"
#include "PostProc.h"
#include "Solvers.h"

#include <Eigen/Dense>
#include <cmath>
#include <gtest/gtest.h>
#include <type_traits>

namespace
{
constexpr bool kSinglePrecision = std::is_same_v<mfem::real_t, float>;
constexpr mfem::real_t kStressTolerance = kSinglePrecision ? 2e-4f : 2e-10;
constexpr mfem::real_t kTangentTolerance = kSinglePrecision ? 8e-2f : 3e-5;
constexpr mfem::real_t kFiniteDifferenceStep = kSinglePrecision ? 2e-3f : 1e-6;

class FixedStepContext final : public plugin::NonlinearStepContext
{
public:
    bool Convergence() const override
    {
        return true;
    }
};

class IdentitySolver final : public mfem::Solver
{
public:
    explicit IdentitySolver( mfem::real_t correctionScale = 1. ) : mCorrectionScale( correctionScale )
    {
    }

    void SetOperator( const mfem::Operator& op ) override
    {
        MFEM_VERIFY( op.Height() == op.Width(), "IdentitySolver requires a square operator." );
        height = op.Height();
        width = op.Width();
    }

    void Mult( const mfem::Vector& rightHandSide, mfem::Vector& solution ) const override
    {
        ++calls;
        lastRightHandSide = rightHandSide( 0 );
        solution = rightHandSide;
        solution *= mCorrectionScale;
    }

    mutable int calls{ 0 };
    mutable mfem::real_t lastRightHandSide{ 0. };

private:
    mfem::real_t mCorrectionScale;
};

class AffineTwoBlockForm final : public mfem::BlockNonlinearForm
{
public:
    AffineTwoBlockForm( mfem::Array<mfem::FiniteElementSpace*>& spaces, const mfem::Vector& target, mfem::real_t coupling = 0. )
        : mfem::BlockNonlinearForm( spaces ),
          mTarget( target ),
          mFirstDiagonal( 1 ),
          mSecondDiagonal( 1 ),
          mCoupling( 1 ),
          mGradient( GetBlockTrueOffsets() )
    {
        MFEM_VERIFY( Height() == 2 && target.Size() == Height(), "The test form requires two scalar blocks." );
        mFirstDiagonal = 1.;
        mSecondDiagonal = 1.;
        mCoupling = coupling;
        mGradient.SetDiagonalBlock( 0, &mFirstDiagonal );
        mGradient.SetDiagonalBlock( 1, &mSecondDiagonal );
        mGradient.SetBlock( 0, 1, &mCoupling );
        mGradient.SetBlock( 1, 0, &mCoupling );
    }

    void Mult( const mfem::Vector& state, mfem::Vector& residual ) const override
    {
        residual.SetSize( state.Size() );
        mGradient.Mult( state, residual );
        residual -= mTarget;
    }

    mfem::Operator& GetGradient( const mfem::Vector& ) const override
    {
        return mGradient;
    }

private:
    mfem::Vector mTarget;
    mutable mfem::DenseMatrix mFirstDiagonal;
    mutable mfem::DenseMatrix mSecondDiagonal;
    mutable mfem::DenseMatrix mCoupling;
    mutable mfem::BlockOperator mGradient;
};

struct MaterialPoint
{
    mfem::Mesh mesh{ mfem::Mesh::MakeCartesian3D( 1, 1, 1, mfem::Element::HEXAHEDRON, 1., 1., 1. ) };
    mfem::ConstantCoefficient youngsModulus{ 10. };
    mfem::ConstantCoefficient poissonRatio{ .25 };
    mfem::ElementTransformation& transformation{ *mesh.GetElementTransformation( 0 ) };
    const mfem::IntegrationPoint& integrationPoint{ mfem::Geometries.GetCenter( mfem::Geometry::CUBE ) };
};

class CountingCoefficient : public mfem::ConstantCoefficient
{
public:
    using mfem::ConstantCoefficient::ConstantCoefficient;
    mfem::real_t Eval( mfem::ElementTransformation& transformation, const mfem::IntegrationPoint& point ) override
    {
        ++calls;
        return mfem::ConstantCoefficient::Eval( transformation, point );
    }
    int calls{ 0 };
};

void SetStrain( PhaseFieldElasticMaterial& material, MaterialPoint& point, const Eigen::Vector6r& engineeringStrain, const mfem::real_t phase )
{
    point.transformation.SetIntPoint( &point.integrationPoint );
    material.at( point.transformation, point.integrationPoint );
    material.setMechanicalStrain( util::InverseVoigt( engineeringStrain, true ) );
    material.setPhaseField( phase );
}

mfem::Vector AssembleElementResidual( plugin::PhaseFieldIntegrator<plugin::PhaseFieldPointStorage>& integrator,
                                      const mfem::Array<const mfem::FiniteElement*>& elements,
                                      mfem::ElementTransformation& transformation,
                                      const mfem::Vector& state,
                                      const int displacementSize )
{
    mfem::Vector displacement( const_cast<mfem::real_t*>( state.GetData() ), displacementSize );
    mfem::Vector phase( const_cast<mfem::real_t*>( state.GetData() ) + displacementSize, state.Size() - displacementSize );
    mfem::Array<const mfem::Vector*> elementState( 2 );
    elementState[0] = &displacement;
    elementState[1] = &phase;

    mfem::Vector displacementResidual;
    mfem::Vector phaseResidual;
    mfem::Array<mfem::Vector*> elementResidual( 2 );
    elementResidual[0] = &displacementResidual;
    elementResidual[1] = &phaseResidual;
    integrator.AssembleElementVector( elements, transformation, elementState, elementResidual );

    mfem::Vector residual( state.Size() );
    for ( int i = 0; i < displacementResidual.Size(); i++ )
    {
        residual( i ) = displacementResidual( i );
    }
    for ( int i = 0; i < phaseResidual.Size(); i++ )
    {
        residual( displacementSize + i ) = phaseResidual( i );
    }
    return residual;
}

mfem::DenseMatrix AssembleElementJacobian( plugin::PhaseFieldIntegrator<plugin::PhaseFieldPointStorage>& integrator,
                                           const mfem::Array<const mfem::FiniteElement*>& elements,
                                           mfem::ElementTransformation& transformation,
                                           mfem::Vector& state,
                                           const int displacementSize )
{
    mfem::Vector displacement( state.GetData(), displacementSize );
    mfem::Vector phase( state.GetData() + displacementSize, state.Size() - displacementSize );
    mfem::Array<const mfem::Vector*> elementState( 2 );
    elementState[0] = &displacement;
    elementState[1] = &phase;

    mfem::DenseMatrix blocks[2][2];
    mfem::Array2D<mfem::DenseMatrix*> elementJacobian( 2, 2 );
    for ( int row = 0; row < 2; row++ )
    {
        for ( int column = 0; column < 2; column++ )
        {
            elementJacobian( row, column ) = &blocks[row][column];
        }
    }
    integrator.AssembleElementGrad( elements, transformation, elementState, elementJacobian );

    mfem::DenseMatrix jacobian( state.Size() );
    jacobian = 0.;
    for ( int blockRow = 0; blockRow < 2; blockRow++ )
    {
        const int rowOffset = blockRow == 0 ? 0 : displacementSize;
        for ( int blockColumn = 0; blockColumn < 2; blockColumn++ )
        {
            const int columnOffset = blockColumn == 0 ? 0 : displacementSize;
            const auto& block = blocks[blockRow][blockColumn];
            for ( int column = 0; column < block.Width(); column++ )
            {
                for ( int row = 0; row < block.Height(); row++ )
                {
                    jacobian( rowOffset + row, columnOffset + column ) = block( row, column );
                }
            }
        }
    }
    return jacobian;
}

static_assert( !std::is_copy_constructible_v<PhaseFieldElasticMaterial> );
} // namespace

TEST( PhaseFieldMaterial, SpectralSplitPreservesEngineeringShearScaling )
{
    MaterialPoint point;
    PhaseFieldElasticMaterial material( point.youngsModulus, point.poissonRatio,
                                        PhaseFieldElasticMaterial::StrainEnergySplit::MieheSpectral );
    constexpr mfem::real_t engineeringShear = .04;
    const mfem::real_t shearModulus = point.youngsModulus.constant / ( 2. * ( 1. + point.poissonRatio.constant ) );
    Eigen::Vector6r strain = Eigen::Vector6r::Zero();
    strain( 3 ) = engineeringShear;
    SetStrain( material, point, strain, 0. );

    const Eigen::Vector6r stress = material.getPK2StressVector();
    material.updateRefModuli();

    EXPECT_NEAR( stress( 3 ), shearModulus * engineeringShear,
                 kStressTolerance * ( 1. + std::abs( shearModulus * engineeringShear ) ) );
    EXPECT_NEAR( material.getRefModuli()( 3, 3 ), shearModulus, kStressTolerance * ( 1. + std::abs( shearModulus ) ) );
    EXPECT_NEAR( material.getPsiPos(), shearModulus * engineeringShear * engineeringShear / 4.,
                 kStressTolerance * ( 1. + shearModulus * engineeringShear * engineeringShear ) );
}

TEST( PhaseFieldMaterial, BatchedResponseMatchesGettersAndOwnsOptionalTangent )
{
    MaterialPoint point;
    Eigen::Vector6r strain;
    strain << .018, -.007, .003, .008, .002, -.004;
    for ( const auto split : { PhaseFieldElasticMaterial::StrainEnergySplit::Isotropic,
                               PhaseFieldElasticMaterial::StrainEnergySplit::AmorVolumetricDeviatoric,
                               PhaseFieldElasticMaterial::StrainEnergySplit::MieheSpectral } )
    {
        PhaseFieldElasticMaterial material( point.youngsModulus, point.poissonRatio, split );
        SetStrain( material, point, strain, .3 );
        const auto response = material.EvaluateResponse( true );
        ASSERT_TRUE( response.tangent.has_value() );
        const auto stressOnly = material.EvaluateResponse( false );
        EXPECT_FALSE( stressOnly.tangent.has_value() );
        EXPECT_EQ( response.positiveEnergy, material.getPsiPos() );
        EXPECT_EQ( stressOnly.positiveEnergy, response.positiveEnergy );
        EXPECT_LE( ( response.stress - material.getPK2StressVector() ).norm(), kStressTolerance );
        EXPECT_LE( ( response.positiveStress - material.getPositiveStressVector() ).norm(), kStressTolerance );
        EXPECT_LE( ( response.phaseStressDerivative - material.getPhaseStressDerivative() ).norm(), kStressTolerance );
        EXPECT_LE( ( stressOnly.stress - response.stress ).norm(), kStressTolerance );
        EXPECT_LE( ( stressOnly.positiveStress - response.positiveStress ).norm(), kStressTolerance );
        EXPECT_LE( ( stressOnly.phaseStressDerivative - response.phaseStressDerivative ).norm(), kStressTolerance );
        material.updateRefModuli();
        EXPECT_LE( ( *response.tangent - material.getRefModuli() ).norm(), kStressTolerance );
        material.setPhaseField( .7 );
        EXPECT_GT( ( material.EvaluateResponse( false ).stress - response.stress ).norm(), kStressTolerance );
    }
}

TEST( PhaseFieldMaterial, NamedSplitsHaveTheirDocumentedCompressionBehavior )
{
    MaterialPoint point;
    PhaseFieldFractureParameters parameters;
    parameters.residualStiffness = .02;
    Eigen::Vector6r compression = Eigen::Vector6r::Zero();
    compression.head<3>().setConstant( -.01 );

    PhaseFieldElasticMaterial spectral( point.youngsModulus, point.poissonRatio,
                                        PhaseFieldElasticMaterial::StrainEnergySplit::MieheSpectral, parameters );
    SetStrain( spectral, point, compression, 0. );
    const Eigen::Vector6r intactSpectralStress = spectral.getPK2StressVector();
    SetStrain( spectral, point, compression, .8 );
    EXPECT_LE( spectral.getPsiPos(), kStressTolerance );
    EXPECT_LE( ( spectral.getPK2StressVector() - intactSpectralStress ).norm(), kStressTolerance );

    PhaseFieldElasticMaterial amor( point.youngsModulus, point.poissonRatio,
                                    PhaseFieldElasticMaterial::StrainEnergySplit::AmorVolumetricDeviatoric, parameters );
    SetStrain( amor, point, compression, 0. );
    const Eigen::Vector6r intactAmorStress = amor.getPK2StressVector();
    SetStrain( amor, point, compression, .8 );
    EXPECT_LE( amor.getPsiPos(), kStressTolerance );
    EXPECT_LE( ( amor.getPK2StressVector() - intactAmorStress ).norm(), kStressTolerance );

    PhaseFieldElasticMaterial isotropic( point.youngsModulus, point.poissonRatio,
                                         PhaseFieldElasticMaterial::StrainEnergySplit::Isotropic, parameters );
    SetStrain( isotropic, point, compression, 0. );
    const Eigen::Vector6r intactIsotropicStress = isotropic.getPK2StressVector();
    SetStrain( isotropic, point, compression, .8 );
    const mfem::real_t degradation = ( 1. - parameters.residualStiffness ) * .2 * .2 + parameters.residualStiffness;
    EXPECT_LE( ( isotropic.getPK2StressVector() - degradation * intactIsotropicStress ).norm(), kStressTolerance );
}

TEST( PhaseFieldMaterial, AmorSplitDegradesDeviatoricShear )
{
    MaterialPoint point;
    PhaseFieldFractureParameters parameters;
    parameters.residualStiffness = .03;
    PhaseFieldElasticMaterial material( point.youngsModulus, point.poissonRatio,
                                        PhaseFieldElasticMaterial::StrainEnergySplit::AmorVolumetricDeviatoric, parameters );
    constexpr mfem::real_t phase = .6;
    constexpr mfem::real_t engineeringShear = .04;
    const mfem::real_t shearModulus = point.youngsModulus.constant / ( 2. * ( 1. + point.poissonRatio.constant ) );
    const mfem::real_t degradation =
        ( 1. - parameters.residualStiffness ) * ( 1. - phase ) * ( 1. - phase ) + parameters.residualStiffness;
    Eigen::Vector6r strain = Eigen::Vector6r::Zero();
    strain( 3 ) = engineeringShear;
    SetStrain( material, point, strain, phase );

    EXPECT_NEAR( material.getPK2StressVector()( 3 ), degradation * shearModulus * engineeringShear,
                 kStressTolerance * ( 1. + shearModulus * engineeringShear ) );
    EXPECT_NEAR( material.getPsiPos(), shearModulus * engineeringShear * engineeringShear / 2., kStressTolerance );
    EXPECT_NEAR( material.getPositiveStressVector()( 3 ), shearModulus * engineeringShear, kStressTolerance );
    material.updateRefModuli();
    for ( int column = 3; column < 6; column++ )
    {
        const Eigen::Vector6r expected = degradation * shearModulus * Eigen::Vector6r::Unit( column );
        EXPECT_LE( ( material.getRefModuli().col( column ) - expected ).norm(), kStressTolerance * ( 1. + expected.norm() ) );
    }
}

TEST( PhaseFieldMaterial, AmorVolumetricTangentsUseComplementarySlopes )
{
    MaterialPoint point;
    PhaseFieldFractureParameters parameters;
    parameters.residualStiffness = .03;
    PhaseFieldElasticMaterial material( point.youngsModulus, point.poissonRatio,
                                        PhaseFieldElasticMaterial::StrainEnergySplit::AmorVolumetricDeviatoric, parameters );
    constexpr mfem::real_t phase = .6;
    const mfem::real_t degradation =
        ( 1. - parameters.residualStiffness ) * ( 1. - phase ) * ( 1. - phase ) + parameters.residualStiffness;
    const mfem::real_t bulkModulus = point.youngsModulus.constant / ( 3. * ( 1. - 2. * point.poissonRatio.constant ) );
    Eigen::Vector6r direction = Eigen::Vector6r::Zero();
    direction.head<3>().setOnes();
    for ( const mfem::real_t dilation : { -.02, 0., .02 } )
    {
        SCOPED_TRACE( dilation );
        Eigen::Vector6r strain = dilation * direction;
        strain( 3 ) = .04; // Pure shear at zero trace; it must retain bulk response.
        const mfem::real_t weight = dilation > 0. ? degradation : ( dilation < 0. ? 1. : ( 1. + degradation ) / 2. );
        const Eigen::Vector6r expected = 3. * bulkModulus * weight * direction;
        SetStrain( material, point, strain, phase );
        material.updateRefModuli();
        EXPECT_LE( ( material.getRefModuli() * direction - expected ).norm(), kStressTolerance * ( 1. + expected.norm() ) );
        SetStrain( material, point, strain + kFiniteDifferenceStep * direction, phase );
        const Eigen::Vector6r plusStress = material.getPK2StressVector();
        SetStrain( material, point, strain - kFiniteDifferenceStep * direction, phase );
        const Eigen::Vector6r minusStress = material.getPK2StressVector();
        const Eigen::Vector6r numerical = ( plusStress - minusStress ) / ( 2. * kFiniteDifferenceStep );
        EXPECT_LE( ( numerical - expected ).norm(), kTangentTolerance * ( 1. + expected.norm() ) );
    }
}

TEST( PhaseFieldMaterial, EverySplitRecoversIntactZeroStrainTangent )
{
    MaterialPoint point;
    // For E=10, nu=1/4: lambda=mu=4, normal diagonal=12, shear diagonal=4.
    Eigen::Matrix6r expected = Eigen::Matrix6r::Zero();
    expected.topLeftCorner<3, 3>().setConstant( 4. );
    expected.diagonal().head<3>().setConstant( 12. );
    expected.diagonal().tail<3>().setConstant( 4. );
    for ( const auto split : { PhaseFieldElasticMaterial::StrainEnergySplit::MieheSpectral,
                               PhaseFieldElasticMaterial::StrainEnergySplit::AmorVolumetricDeviatoric,
                               PhaseFieldElasticMaterial::StrainEnergySplit::Isotropic } )
    {
        SCOPED_TRACE( static_cast<int>( split ) );
        PhaseFieldElasticMaterial material( point.youngsModulus, point.poissonRatio, split );
        SetStrain( material, point, Eigen::Vector6r::Zero(), 0. );
        EXPECT_LE( material.getPK2StressVector().norm(), kStressTolerance );
        EXPECT_LE( material.getPositiveStressVector().norm(), kStressTolerance );
        EXPECT_NEAR( material.getPsiPos(), 0., kStressTolerance );
        material.updateRefModuli();
        EXPECT_LE( ( material.getRefModuli() - expected ).norm(), kStressTolerance * ( 1. + expected.norm() ) );
    }
}

TEST( PhaseFieldMaterial, TangentsMatchDirectionalFiniteDifferencesForEverySplit )
{
    MaterialPoint point;
    const Eigen::Vector6r strain = ( Eigen::Vector6r() << .018, -.007, .004, .006, -.003, .002 ).finished();
    Eigen::Vector6r direction = ( Eigen::Vector6r() << -.2, .3, .1, .4, -.25, .15 ).finished();
    direction.normalize();

    for ( const auto split : { PhaseFieldElasticMaterial::StrainEnergySplit::MieheSpectral,
                               PhaseFieldElasticMaterial::StrainEnergySplit::AmorVolumetricDeviatoric,
                               PhaseFieldElasticMaterial::StrainEnergySplit::Isotropic } )
    {
        PhaseFieldElasticMaterial material( point.youngsModulus, point.poissonRatio, split );
        SetStrain( material, point, strain, .35 );
        material.updateRefModuli();
        const Eigen::Vector6r analyticalDerivative = material.getRefModuli() * direction;

        SetStrain( material, point, strain + kFiniteDifferenceStep * direction, .35 );
        const Eigen::Vector6r plusStress = material.getPK2StressVector();
        SetStrain( material, point, strain - kFiniteDifferenceStep * direction, .35 );
        const Eigen::Vector6r minusStress = material.getPK2StressVector();
        const Eigen::Vector6r numericalDerivative = ( plusStress - minusStress ) / ( 2. * kFiniteDifferenceStep );

        EXPECT_LE( ( analyticalDerivative - numericalDerivative ).norm(), kTangentTolerance * ( 1. + numericalDerivative.norm() ) );
    }
}

TEST( PhaseFieldMaterial, RepeatedPrincipalStrainsHaveFiniteConsistentTangents )
{
    MaterialPoint point;
    PhaseFieldElasticMaterial material( point.youngsModulus, point.poissonRatio );
    Eigen::Vector6r direction;
    direction << .2, -.1, .3, .4, -.2, .1;
    for ( const mfem::real_t dilation : { -.02, .02 } )
    {
        Eigen::Vector6r strain = Eigen::Vector6r::Zero();
        strain.head<3>().setConstant( dilation );
        SetStrain( material, point, strain, .4 );
        material.updateRefModuli();
        const Eigen::Vector6r analytical = material.getRefModuli() * direction;
        ASSERT_TRUE( analytical.allFinite() );
        SetStrain( material, point, strain + kFiniteDifferenceStep * direction, .4 );
        const Eigen::Vector6r plus = material.getPK2StressVector();
        SetStrain( material, point, strain - kFiniteDifferenceStep * direction, .4 );
        const Eigen::Vector6r minus = material.getPK2StressVector();
        const Eigen::Vector6r numerical = ( plus - minus ) / ( 2. * kFiniteDifferenceStep );
        EXPECT_LE( ( analytical - numerical ).norm(), kTangentTolerance * ( 1. + numerical.norm() ) );
    }
    SetStrain( material, point, Eigen::Vector6r::Zero(), 0. );
    EXPECT_LE( material.getPK2StressVector().norm(), kStressTolerance );
    material.updateRefModuli();
    EXPECT_TRUE( material.getRefModuli().allFinite() );
}

TEST( PhaseFieldMaterial, FullyCrackedTensionRetainsSmallResidualStiffness )
{
    MaterialPoint point;
    PhaseFieldFractureParameters parameters;
    parameters.residualStiffness = 1e-20;
    PhaseFieldElasticMaterial material( point.youngsModulus, point.poissonRatio,
                                        PhaseFieldElasticMaterial::StrainEnergySplit::MieheSpectral, parameters );
    Eigen::Vector6r strain = Eigen::Vector6r::Zero();
    strain.head<3>().setConstant( .02 );
    SetStrain( material, point, strain, 0. );
    const Eigen::Vector6r intactStress = material.getPK2StressVector();
    material.updateRefModuli();
    const Eigen::Matrix6r intactTangent = material.getRefModuli();
    material.setPhaseField( 1. );
    const Eigen::Vector6r scaledStress = material.getPK2StressVector() / parameters.residualStiffness;
    material.updateRefModuli();
    const Eigen::Matrix6r scaledTangent = material.getRefModuli() / parameters.residualStiffness;
    EXPECT_LE( ( scaledStress - intactStress ).norm(), kStressTolerance * intactStress.norm() );
    EXPECT_LE( ( scaledTangent - intactTangent ).norm(), kStressTolerance * intactTangent.norm() );
}

TEST( PhaseFieldIntegrator, AllFourJacobianBlocksMatchDirectionalFiniteDifference )
{
    for ( const auto split : { PhaseFieldElasticMaterial::StrainEnergySplit::Isotropic,
                               PhaseFieldElasticMaterial::StrainEnergySplit::AmorVolumetricDeviatoric,
                               PhaseFieldElasticMaterial::StrainEnergySplit::MieheSpectral } )
    {
        mfem::Mesh mesh = mfem::Mesh::MakeCartesian2D( 1, 1, mfem::Element::QUADRILATERAL, true, 1., 1. );
        mfem::H1_FECollection collection( 1, mesh.Dimension() );
        mfem::FiniteElementSpace displacementSpace( &mesh, &collection, mesh.Dimension(), mfem::Ordering::byVDIM );
        mfem::FiniteElementSpace phaseSpace( &mesh, &collection );
        const auto* displacementElement = displacementSpace.GetFE( 0 );
        const auto* phaseElement = phaseSpace.GetFE( 0 );
        auto* transformation = mesh.GetElementTransformation( 0 );
        ASSERT_NE( displacementElement, nullptr );
        ASSERT_NE( phaseElement, nullptr );
        ASSERT_NE( transformation, nullptr );

        CountingCoefficient youngsModulus( 10. );
        CountingCoefficient poissonRatio( .25 );
        PhaseFieldFractureParameters parameters{ 2.5, .3, .01 };
        PhaseFieldElasticMaterial material( youngsModulus, poissonRatio, split, parameters );
        plugin::PhaseFieldPointStorage pointStorage( &mesh );
        plugin::PhaseFieldIntegrator<plugin::PhaseFieldPointStorage> integrator( material, pointStorage );
        FixedStepContext context;
        integrator.SetStepContext( &context );

        mfem::Array<const mfem::FiniteElement*> elements( 2 );
        elements[0] = displacementElement;
        elements[1] = phaseElement;
        const int displacementDofs = displacementElement->GetDof();
        const int displacementSize = displacementDofs * mesh.Dimension();
        mfem::Vector state( displacementSize + phaseElement->GetDof() );
        for ( int node = 0; node < displacementDofs; node++ )
        {
            mfem::Vector position;
            transformation->Transform( displacementElement->GetNodes().IntPoint( node ), position );
            state( node ) = .018 * position( 0 ) + .004 * position( 1 );
            state( displacementDofs + node ) = .004 * position( 0 ) - .007 * position( 1 );
            state( displacementSize + node ) = .2 + .03 * position( 0 ) + .02 * position( 1 );
        }

        integrator.BeginStep();
        mfem::DenseMatrix jacobian = AssembleElementJacobian( integrator, elements, *transformation, state, displacementSize );
        const int points = mfem::IntRules.Get( displacementElement->GetGeomType(), 3 ).GetNPoints();
        EXPECT_EQ( youngsModulus.calls, points );
        EXPECT_EQ( poissonRatio.calls, points );
        youngsModulus.calls = poissonRatio.calls = 0;
        AssembleElementResidual( integrator, elements, *transformation, state, displacementSize );
        EXPECT_EQ( youngsModulus.calls, points );
        EXPECT_EQ( poissonRatio.calls, points );
        Eigen::Map<const Eigen::MatrixXr> jacobianMap( jacobian.Data(), jacobian.Height(), jacobian.Width() );
        EXPECT_GT( jacobianMap.block( 0, displacementSize, displacementSize, phaseElement->GetDof() ).norm(), 0. );
        EXPECT_GT( jacobianMap.block( displacementSize, 0, phaseElement->GetDof(), displacementSize ).norm(), 0. );

        mfem::Vector direction( state.Size() );
        for ( int i = 0; i < direction.Size(); i++ )
        {
            direction( i ) = static_cast<mfem::real_t>( ( i % 7 ) - 3 );
        }
        direction /= direction.Norml2();
        mfem::Vector plus( state );
        mfem::Vector minus( state );
        plus.Add( kFiniteDifferenceStep, direction );
        minus.Add( -kFiniteDifferenceStep, direction );
        const mfem::Vector plusResidual = AssembleElementResidual( integrator, elements, *transformation, plus, displacementSize );
        const mfem::Vector minusResidual = AssembleElementResidual( integrator, elements, *transformation, minus, displacementSize );
        mfem::Vector numericalDerivative( plusResidual );
        numericalDerivative -= minusResidual;
        numericalDerivative /= 2. * kFiniteDifferenceStep;
        mfem::Vector analyticalDerivative( state.Size() );
        jacobian.Mult( direction, analyticalDerivative );
        analyticalDerivative -= numericalDerivative;

        EXPECT_LE( analyticalDerivative.Norml2(), kTangentTolerance * ( 1. + numericalDerivative.Norml2() ) );
        // Commit the loaded state, then unload well away from the history switch.
        AssembleElementResidual( integrator, elements, *transformation, state, displacementSize );
        integrator.CommitStep();
        integrator.BeginStep();
        for ( int i = 0; i < displacementSize; i++ )
        {
            state( i ) *= .5;
        }
        youngsModulus.calls = poissonRatio.calls = 0;
        jacobian = AssembleElementJacobian( integrator, elements, *transformation, state, displacementSize );
        EXPECT_EQ( youngsModulus.calls, points );
        EXPECT_EQ( poissonRatio.calls, points );
        youngsModulus.calls = poissonRatio.calls = 0;
        AssembleElementResidual( integrator, elements, *transformation, state, displacementSize );
        EXPECT_EQ( youngsModulus.calls, points );
        EXPECT_EQ( poissonRatio.calls, points );
        Eigen::Map<const Eigen::MatrixXr> unloadedJacobian( jacobian.Data(), jacobian.Height(), jacobian.Width() );
        EXPECT_EQ( unloadedJacobian.block( displacementSize, 0, phaseElement->GetDof(), displacementSize ).norm(), 0. );
        plus = state;
        minus = state;
        plus.Add( kFiniteDifferenceStep, direction );
        minus.Add( -kFiniteDifferenceStep, direction );
        numericalDerivative = AssembleElementResidual( integrator, elements, *transformation, plus, displacementSize );
        numericalDerivative -= AssembleElementResidual( integrator, elements, *transformation, minus, displacementSize );
        numericalDerivative /= 2. * kFiniteDifferenceStep;
        jacobian.Mult( direction, analyticalDerivative );
        analyticalDerivative -= numericalDerivative;
        EXPECT_LE( analyticalDerivative.Norml2(), kTangentTolerance * ( 1. + numericalDerivative.Norml2() ) );
        integrator.RollbackStep();
    }
}

TEST( NewtonForPhaseField, DoesNotIgnoreInitialPhaseResidual )
{
    mfem::Mesh mesh = mfem::Mesh::MakeCartesian1D( 1 );
    mfem::L2_FECollection collection( 0, mesh.Dimension() );
    mfem::FiniteElementSpace firstSpace( &mesh, &collection );
    mfem::FiniteElementSpace secondSpace( &mesh, &collection );
    mfem::Array<mfem::FiniteElementSpace*> spaces( 2 );
    spaces[0] = &firstSpace;
    spaces[1] = &secondSpace;
    mfem::Vector target( 2 );
    target( 0 ) = 0.;
    target( 1 ) = 2.;
    AffineTwoBlockForm form( spaces, target );
    IdentitySolver linearSolver;
    plugin::NewtonForPhaseField solver;
    solver.SetOperator( form );
    solver.SetSolver( linearSolver );
    solver.SetRelTol( 1e-12 );
    solver.SetAbsTol( 1e-12 );
    solver.SetMaxIter( 3 );
    solver.iterative_mode = true;
    mfem::Vector solution( form.Height() );
    solution = 0.;
    mfem::Vector zeroRightHandSide;

    solver.Mult( zeroRightHandSide, solution );

    EXPECT_TRUE( solver.GetConverged() );
    EXPECT_NEAR( solution( 0 ), 0., kStressTolerance );
    EXPECT_NEAR( solution( 1 ), 2., kStressTolerance );
    EXPECT_NEAR( solver.GetFinalNorm(), 0., kStressTolerance );
}

TEST( NewtonForPhaseField, AppliesRightHandSideToBothBlocksAtEveryEvaluation )
{
    mfem::Mesh mesh = mfem::Mesh::MakeCartesian1D( 1 );
    mfem::L2_FECollection collection( 0, mesh.Dimension() );
    mfem::FiniteElementSpace firstSpace( &mesh, &collection );
    mfem::FiniteElementSpace secondSpace( &mesh, &collection );
    mfem::Array<mfem::FiniteElementSpace*> spaces( 2 );
    spaces[0] = &firstSpace;
    spaces[1] = &secondSpace;
    mfem::Vector target( 2 );
    target = 0.;
    AffineTwoBlockForm form( spaces, target );
    IdentitySolver linearSolver;
    plugin::NewtonForPhaseField solver;
    solver.SetOperator( form );
    solver.SetSolver( linearSolver );
    solver.SetRelTol( 1e-12 );
    solver.SetAbsTol( 1e-12 );
    solver.SetMaxIter( 3 );
    solver.iterative_mode = true;
    mfem::Vector solution( form.Height() );
    solution = 0.;
    mfem::BlockVector rightHandSide( form.GetBlockTrueOffsets() );
    rightHandSide( 0 ) = 1.;
    rightHandSide( 1 ) = 2.;

    solver.Mult( rightHandSide, solution );

    EXPECT_TRUE( solver.GetConverged() );
    EXPECT_NEAR( solution( 0 ), 1., kStressTolerance );
    EXPECT_NEAR( solution( 1 ), 2., kStressTolerance );
    EXPECT_NEAR( solver.GetFinalNorm(), 0., kStressTolerance );
}

TEST( NewtonForPhaseField, RoutesEachBlockToItsConfiguredSolver )
{
    mfem::Mesh mesh = mfem::Mesh::MakeCartesian1D( 1 );
    mfem::L2_FECollection collection( 0, mesh.Dimension() );
    mfem::FiniteElementSpace firstSpace( &mesh, &collection );
    mfem::FiniteElementSpace secondSpace( &mesh, &collection );
    mfem::Array<mfem::FiniteElementSpace*> spaces( 2 );
    spaces[0] = &firstSpace;
    spaces[1] = &secondSpace;
    mfem::Vector target( 2 );
    target = 0.;
    AffineTwoBlockForm form( spaces, target );
    IdentitySolver displacementSolver;
    IdentitySolver phaseSolver;
    plugin::NewtonForPhaseField solver;
    solver.SetOperator( form );
    solver.SetBlockSolvers( displacementSolver, phaseSolver );
    solver.SetRelTol( 1e-12 );
    solver.SetAbsTol( 1e-12 );
    solver.SetMaxIter( 3 );
    solver.iterative_mode = true;
    mfem::Vector solution( form.Height() );
    solution = 0.;
    mfem::BlockVector rightHandSide( form.GetBlockTrueOffsets() );
    rightHandSide( 0 ) = 1.;
    rightHandSide( 1 ) = 2.;

    solver.Mult( rightHandSide, solution );

    EXPECT_TRUE( solver.GetConverged() );
    EXPECT_NEAR( solution( 0 ), 1., kStressTolerance );
    EXPECT_NEAR( solution( 1 ), 2., kStressTolerance );
    EXPECT_NEAR( solver.GetFinalNorm(), 0., kStressTolerance );
    EXPECT_EQ( displacementSolver.calls, 1 );
    EXPECT_EQ( phaseSolver.calls, 1 );
    EXPECT_NEAR( displacementSolver.lastRightHandSide, -1., kStressTolerance );
    EXPECT_NEAR( phaseSolver.lastRightHandSide, -2., kStressTolerance );

    // Calling SetSolver afterwards restores the shared-solver contract.
    IdentitySolver sharedSolver;
    solver.SetSolver( sharedSolver );
    solution = 0.;
    solver.Mult( rightHandSide, solution );
    EXPECT_TRUE( solver.GetConverged() );
    EXPECT_EQ( sharedSolver.calls, 2 );
    EXPECT_EQ( phaseSolver.calls, 1 );
}

TEST( NewtonForPhaseField, CoupledUpdatesReactivateInitiallyConvergedBlocks )
{
    mfem::Mesh mesh = mfem::Mesh::MakeCartesian1D( 1 );
    mfem::L2_FECollection collection( 0, mesh.Dimension() );
    mfem::FiniteElementSpace displacementSpace( &mesh, &collection );
    mfem::FiniteElementSpace phaseSpace( &mesh, &collection );
    mfem::Array<mfem::FiniteElementSpace*> spaces( 2 );
    spaces[0] = &displacementSpace;
    spaces[1] = &phaseSpace;
    constexpr mfem::real_t tolerance = kSinglePrecision ? 2e-5f : 1e-11;
    mfem::Vector zeroRightHandSide;

    for ( int drivenBlock = 0; drivenBlock < 2; ++drivenBlock )
    {
        SCOPED_TRACE( drivenBlock );
        mfem::Vector target( 2 );
        target = 0.;
        target( drivenBlock ) = 1.;
        // Ru = u + phi/2 - target_u, Rphi = phi + u/2 - target_phi.
        // At the zero initial state the other block is exactly converged.
        AffineTwoBlockForm form( spaces, target, .5 );
        IdentitySolver displacementSolver;
        IdentitySolver phaseSolver;
        plugin::NewtonForPhaseField solver;
        solver.SetOperator( form );
        solver.SetBlockSolvers( displacementSolver, phaseSolver );
        solver.SetRelTol( 0. );
        solver.SetAbsTol( tolerance );
        solver.iterative_mode = true;
        mfem::Vector solution( form.Height() );
        solution = 0.;

        // Neither loading direction can converge in one sweep: the latest
        // phase update disturbs displacement equilibrium. Reject and restore.
        solver.SetMaxIter( 1 );
        solver.Mult( zeroRightHandSide, solution );
        EXPECT_FALSE( solver.GetConverged() );
        EXPECT_GT( solver.GetFinalNorm(), tolerance );
        EXPECT_EQ( solution.Norml2(), 0. );

        displacementSolver.calls = 0;
        phaseSolver.calls = 0;
        solver.SetMaxIter( 40 );
        solver.Mult( zeroRightHandSide, solution );
        ASSERT_TRUE( solver.GetConverged() );
        EXPECT_GT( displacementSolver.calls, 0 );
        EXPECT_GT( phaseSolver.calls, 0 );
        EXPECT_GT( solver.GetNumIterations(), 1 );
        mfem::Vector residual;
        form.Mult( solution, residual );
        EXPECT_LE( std::abs( residual( 0 ) ), tolerance );
        EXPECT_LE( std::abs( residual( 1 ) ), tolerance );
        EXPECT_NEAR( solution( drivenBlock ), 4. / 3., 3. * tolerance );
        EXPECT_NEAR( solution( 1 - drivenBlock ), -2. / 3., 3. * tolerance );
    }
}

TEST( NewtonForPhaseField, ExactBlockSolvesDoNotGuaranteeCoupledConvergence )
{
    mfem::Mesh mesh = mfem::Mesh::MakeCartesian1D( 1 );
    mfem::L2_FECollection collection( 0, mesh.Dimension() );
    mfem::FiniteElementSpace displacementSpace( &mesh, &collection );
    mfem::FiniteElementSpace phaseSpace( &mesh, &collection );
    mfem::Array<mfem::FiniteElementSpace*> spaces( 2 );
    spaces[0] = &displacementSpace;
    spaces[1] = &phaseSpace;
    mfem::Vector target( 2 );
    target( 0 ) = 1.;
    target( 1 ) = 0.;
    constexpr mfem::real_t coupling = .999;
    constexpr int sweeps = 20;
    AffineTwoBlockForm form( spaces, target, coupling );
    IdentitySolver displacementSolver;
    IdentitySolver phaseSolver;
    plugin::NewtonForPhaseField solver;
    solver.SetOperator( form );
    solver.SetBlockSolvers( displacementSolver, phaseSolver );
    solver.SetRelTol( 0. );
    solver.SetAbsTol( kStressTolerance );
    solver.SetMaxIter( sweeps );
    solver.iterative_mode = true;
    mfem::Vector solution( form.Height() );
    solution = 0.;
    mfem::Vector zeroRightHandSide;

    solver.Mult( zeroRightHandSide, solution );

    // Original scalar model: Ru = u + a*phi - 1, Rphi = phi + a*u.
    // Both diagonal Jacobians are exactly 1 and the coupled matrix is SPD.
    // Each phase solve gives Rphi = 0, but after n sweeps Ru = -a^(2*n).
    // This near-stagnation is coupling error, not linear-solver or tangent error.
    EXPECT_FALSE( solver.GetConverged() );
    EXPECT_EQ( solver.GetNumIterations(), sweeps );
    EXPECT_EQ( displacementSolver.calls, sweeps );
    EXPECT_EQ( phaseSolver.calls, sweeps );
    EXPECT_NEAR( solver.GetFinalNorm(), std::pow( coupling, 2 * sweeps ), 10. * kStressTolerance );
    EXPECT_GT( solver.GetFinalNorm(), .95 );
    EXPECT_EQ( solution.Norml2(), 0. ); // The rejected trial is rolled back.
}

TEST( NewtonForPhaseField, LargerSweepBudgetResolvesStrongCouplingWithoutRelaxingTolerance )
{
    mfem::Mesh mesh = mfem::Mesh::MakeCartesian1D( 1 );
    mfem::L2_FECollection collection( 0, mesh.Dimension() );
    mfem::FiniteElementSpace displacementSpace( &mesh, &collection );
    mfem::FiniteElementSpace phaseSpace( &mesh, &collection );
    mfem::Array<mfem::FiniteElementSpace*> spaces( 2 );
    spaces[0] = &displacementSpace;
    spaces[1] = &phaseSpace;
    mfem::Vector target( 2 );
    target( 0 ) = 1.;
    target( 1 ) = 0.;
    constexpr mfem::real_t coupling = .99;
    constexpr mfem::real_t tolerance = kSinglePrecision ? 2e-4f : 1e-8;
    AffineTwoBlockForm form( spaces, target, coupling );
    IdentitySolver linearSolver;
    plugin::NewtonForPhaseField solver;
    solver.SetOperator( form );
    solver.SetSolver( linearSolver );
    solver.SetRelTol( 0. );
    solver.SetAbsTol( tolerance );
    solver.iterative_mode = true;
    mfem::Vector solution( form.Height() );
    solution = 0.;
    mfem::Vector zeroRightHandSide;

    solver.SetMaxIter( 12 );
    solver.Mult( zeroRightHandSide, solution );
    EXPECT_FALSE( solver.GetConverged() );
    EXPECT_EQ( solution.Norml2(), 0. );

    // Only the work budget changes. Check both actual residuals at the returned
    // state, not merely the solver flag or the last exact phase subsolve.
    solver.SetMaxIter( 1200 );
    solver.Mult( zeroRightHandSide, solution );
    ASSERT_TRUE( solver.GetConverged() );
    EXPECT_GT( solver.GetNumIterations(), 12 );
    mfem::Vector residual;
    form.Mult( solution, residual );
    EXPECT_LE( std::abs( residual( 0 ) ), tolerance );
    EXPECT_LE( std::abs( residual( 1 ) ), tolerance );
    EXPECT_NEAR( solver.GetFinalNorm(), residual.Norml2(), kStressTolerance );
    // The smallest eigenvalue is 1-a, so residual tolerances must be scaled
    // by the inverse eigenvalue when checking the solution itself.
    const mfem::real_t solutionTolerance = 2. * tolerance / ( 1. - coupling );
    EXPECT_NEAR( solution( 0 ), 1. / ( 1. - coupling * coupling ), solutionTolerance );
    EXPECT_NEAR( solution( 1 ), -coupling / ( 1. - coupling * coupling ), solutionTolerance );
}

TEST( NewtonForPhaseField, RelativeOnlyToleranceForInitiallyZeroBlocks )
{
    mfem::Mesh mesh = mfem::Mesh::MakeCartesian1D( 1 );
    mfem::L2_FECollection collection( 0, mesh.Dimension() );
    mfem::FiniteElementSpace displacementSpace( &mesh, &collection );
    mfem::FiniteElementSpace phaseSpace( &mesh, &collection );
    mfem::Array<mfem::FiniteElementSpace*> spaces( 2 );
    spaces[0] = &displacementSpace;
    spaces[1] = &phaseSpace;
    constexpr mfem::real_t relativeTolerance = mfem::real_t( 1 ) / 128;
    mfem::Vector zeroRightHandSide;

    for ( int drivenBlock = 0; drivenBlock < 2; ++drivenBlock )
    {
        SCOPED_TRACE( drivenBlock );
        mfem::Vector target( 2 );
        target = 0.;
        target( drivenBlock ) = 1.;
        AffineTwoBlockForm form( spaces, target, .5 );
        // Half corrections deliberately leave nonzero residuals in both blocks;
        // exact affine phase solves could mask a permanently zero phase goal.
        IdentitySolver linearSolver( .5 );
        plugin::NewtonForPhaseField solver;
        solver.SetOperator( form );
        solver.SetSolver( linearSolver );
        solver.SetRelTol( relativeTolerance );
        solver.SetAbsTol( 0. );
        solver.SetMaxIter( 40 );
        solver.iterative_mode = true;
        mfem::Vector solution( form.Height() );
        solution = 0.;

        solver.Mult( zeroRightHandSide, solution );

        ASSERT_TRUE( solver.GetConverged() );
        EXPECT_GT( solver.GetNumIterations(), 1 );
        mfem::Vector residual;
        form.Mult( solution, residual );
        // The driven block starts at norm 1; the other first activates at
        // norm .25 (coupling .5 times the first half correction .5).
        EXPECT_LE( std::abs( residual( drivenBlock ) ), relativeTolerance );
        EXPECT_LE( std::abs( residual( 1 - drivenBlock ) ), relativeTolerance * mfem::real_t( .25 ) );
        EXPECT_GT( std::abs( residual( 0 ) ), 0. );
        EXPECT_GT( std::abs( residual( 1 ) ), 0. );
        EXPECT_NEAR( solver.GetFinalNorm(), residual.Norml2(), kStressTolerance );
    }
}

TEST( StressCoefficient, UsesCurrentPhaseFieldValue )
{
    mfem::Mesh mesh = mfem::Mesh::MakeCartesian2D( 1, 1, mfem::Element::QUADRILATERAL, true, 1., 1. );
    mfem::H1_FECollection collection( 1, mesh.Dimension() );
    mfem::FiniteElementSpace displacementSpace( &mesh, &collection, mesh.Dimension(), mfem::Ordering::byVDIM );
    mfem::FiniteElementSpace phaseSpace( &mesh, &collection );
    mfem::GridFunction displacement( &displacementSpace );
    mfem::GridFunction phase( &phaseSpace );
    mfem::VectorFunctionCoefficient displacementCoefficient( mesh.Dimension(),
                                                             []( const mfem::Vector& position, mfem::Vector& value )
                                                             {
                                                                 value.SetSize( 2 );
                                                                 value( 0 ) = .01 * position( 0 );
                                                                 value( 1 ) = 0.;
                                                             } );
    displacement.ProjectCoefficient( displacementCoefficient );

    mfem::ConstantCoefficient youngsModulus( 10. );
    mfem::ConstantCoefficient poissonRatio( .25 );
    PhaseFieldFractureParameters parameters;
    parameters.residualStiffness = .02;
    PhaseFieldElasticMaterial material( youngsModulus, poissonRatio,
                                        PhaseFieldElasticMaterial::StrainEnergySplit::MieheSpectral, parameters );
    plugin::StressCoefficient stressCoefficient( mesh.Dimension(), material );
    stressCoefficient.SetDisplacement( displacement );
    stressCoefficient.SetPhaseField( phase );
    auto& transformation = *mesh.GetElementTransformation( 0 );
    const auto& integrationPoint = mfem::Geometries.GetCenter( mfem::Geometry::SQUARE );
    mfem::Vector intactStress;
    mfem::Vector degradedStress;

    phase = 0.;
    stressCoefficient.Eval( intactStress, transformation, integrationPoint );
    phase = .75;
    stressCoefficient.Eval( degradedStress, transformation, integrationPoint );

    const mfem::real_t degradation = ( 1. - parameters.residualStiffness ) * .25 * .25 + parameters.residualStiffness;
    for ( int component = 0; component < 6; component++ )
    {
        EXPECT_NEAR( degradedStress( component ), degradation * intactStress( component ),
                     kStressTolerance * ( 1. + std::abs( intactStress( component ) ) ) );
    }
}

TEST( StressCoefficient, VonMisesIncludesEveryShearComponent )
{
    mfem::Mesh mesh = mfem::Mesh::MakeCartesian3D( 1, 1, 1, mfem::Element::HEXAHEDRON, 1., 1., 1. );
    mfem::H1_FECollection collection( 1, mesh.Dimension() );
    mfem::FiniteElementSpace displacementSpace( &mesh, &collection, mesh.Dimension(), mfem::Ordering::byVDIM );
    mfem::GridFunction displacement( &displacementSpace );
    mfem::VectorFunctionCoefficient displacementCoefficient( mesh.Dimension(),
                                                             []( const mfem::Vector& position, mfem::Vector& value )
                                                             {
                                                                 value.SetSize( 3 );
                                                                 value( 0 ) = .02 * position( 1 );
                                                                 value( 1 ) = .03 * position( 2 );
                                                                 value( 2 ) = .04 * position( 0 );
                                                             } );
    displacement.ProjectCoefficient( displacementCoefficient );

    mfem::ConstantCoefficient youngsModulus( 10. );
    mfem::ConstantCoefficient poissonRatio( .25 );
    IsotropicElasticMaterial material( youngsModulus, poissonRatio );
    plugin::StressCoefficient stressCoefficient( mesh.Dimension(), material );
    stressCoefficient.SetDisplacement( displacement );
    auto& transformation = *mesh.GetElementTransformation( 0 );
    const auto& integrationPoint = mfem::Geometries.GetCenter( mfem::Geometry::CUBE );
    mfem::Vector stress;

    stressCoefficient.Eval( stress, transformation, integrationPoint );

    const mfem::real_t expected =
        std::sqrt( .5 * ( std::pow( stress( 0 ) - stress( 1 ), 2 ) + std::pow( stress( 1 ) - stress( 2 ), 2 ) +
                          std::pow( stress( 2 ) - stress( 0 ), 2 ) ) +
                   3. * ( std::pow( stress( 3 ), 2 ) + std::pow( stress( 4 ), 2 ) + std::pow( stress( 5 ), 2 ) ) );
    EXPECT_GT( std::abs( stress( 3 ) ) + std::abs( stress( 4 ) ), 0. );
    EXPECT_NEAR( stress( 6 ), expected, kStressTolerance * ( 1. + expected ) );
}
