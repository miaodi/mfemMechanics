#include "FEMPlugin.h"
#include "Material.h"
#include "NeoHookeanMaterial.h"
#include "Solvers.h"
#include "StressFreeDeformation.h"

#include <cmath>
#include <gtest/gtest.h>
#include <type_traits>

namespace
{
constexpr bool kSinglePrecision = std::is_same_v<mfem::real_t, float>;
constexpr mfem::real_t kResidualTolerance = kSinglePrecision ? 1e-3f : 1e-10;
constexpr mfem::real_t kMatrixTolerance = kSinglePrecision ? 1e-5f : 1e-12;
constexpr mfem::real_t kTangentTolerance = kSinglePrecision ? 5e-2f : 5e-5;

class FixedLoadState final : public plugin::IterAuxilliary
{
public:
    explicit FixedLoadState( const mfem::real_t loadFactor )
    {
        SetDelta( loadFactor );
    }

    bool Convergence() const override
    {
        return true;
    }
};

mfem::Vector UniformExpansion( const mfem::FiniteElement& element, mfem::ElementTransformation& transformation, const mfem::real_t expansion )
{
    const int dof = element.GetDof();
    const int dimension = element.GetDim();
    mfem::Vector displacement( dof * dimension );
    mfem::Vector position( dimension );

    const auto& nodes = element.GetNodes();
    for ( int node = 0; node < dof; node++ )
    {
        transformation.Transform( nodes.IntPoint( node ), position );
        for ( int component = 0; component < dimension; component++ )
        {
            displacement( node + component * dof ) = expansion * position( component );
        }
    }
    return displacement;
}

void ExpectMatricesNear( const mfem::DenseMatrix& actual, const mfem::DenseMatrix& expected )
{
    ASSERT_EQ( actual.Height(), expected.Height() );
    ASSERT_EQ( actual.Width(), expected.Width() );
    for ( int column = 0; column < actual.Width(); column++ )
    {
        for ( int row = 0; row < actual.Height(); row++ )
        {
            EXPECT_NEAR( actual( row, column ), expected( row, column ),
                         kMatrixTolerance * ( 1. + std::abs( expected( row, column ) ) ) );
        }
    }
}
} // namespace

TEST( ThermalStrain, IsComposedOutsideMaterialAndScalesWithLoadFactor )
{
    constexpr mfem::real_t thermalExpansion = 1e-4;
    constexpr mfem::real_t referenceTemperature = 20.;
    constexpr mfem::real_t targetTemperature = 120.;
    constexpr mfem::real_t targetStrain = thermalExpansion * ( targetTemperature - referenceTemperature );

    mfem::Mesh mesh = mfem::Mesh::MakeCartesian3D( 1, 1, 1, mfem::Element::HEXAHEDRON, 1., 1., 1. );
    mfem::H1_FECollection collection( 1, mesh.Dimension() );
    mfem::FiniteElementSpace space( &mesh, &collection );
    const auto* element = space.GetFE( 0 );
    auto* transformation = mesh.GetElementTransformation( 0 );
    ASSERT_NE( element, nullptr );
    ASSERT_NE( transformation, nullptr );

    mfem::ConstantCoefficient elasticModulus( 1000. );
    mfem::ConstantCoefficient poissonRatio( .25 );
    mfem::ConstantCoefficient expansionCoefficient( thermalExpansion );
    mfem::ConstantCoefficient referenceTemperatureCoefficient( referenceTemperature );
    mfem::ConstantCoefficient targetTemperatureCoefficient( targetTemperature );

    IsotropicElasticMaterial material( elasticModulus, poissonRatio );
    plugin::IsotropicThermalExpansion thermalExpansionModel( expansionCoefficient, targetTemperatureCoefficient,
                                                             referenceTemperatureCoefficient );
    plugin::IntegrationPointStorage pointStorage( &mesh );
    plugin::NonlinearElasticityIntegrator integrator( material, pointStorage );
    integrator.setNonlinear( false );
    integrator.AddStressFreeDeformation( thermalExpansionModel );

    FixedLoadState unitLoad( 1. );
    integrator.SetIterAux( &unitLoad );

    mfem::Vector constrainedDisplacement( element->GetDof() * element->GetDim() );
    constrainedDisplacement = 0.;
    mfem::Vector constrainedResidual;
    integrator.AssembleElementVector( *element, *transformation, constrainedDisplacement, constrainedResidual );
    ASSERT_GT( constrainedResidual.Norml2(), 1e-2 );

    const mfem::Vector freeExpansion = UniformExpansion( *element, *transformation, targetStrain );
    mfem::Vector freeResidual;
    integrator.AssembleElementVector( *element, *transformation, freeExpansion, freeResidual );
    EXPECT_LE( freeResidual.Norml2(), kResidualTolerance * ( 1. + constrainedResidual.Norml2() ) );

    FixedLoadState zeroLoad( 0. );
    integrator.SetIterAux( &zeroLoad );
    mfem::Vector zeroLoadResidual;
    integrator.AssembleElementVector( *element, *transformation, constrainedDisplacement, zeroLoadResidual );
    EXPECT_LE( zeroLoadResidual.Norml2(), kResidualTolerance );

    FixedLoadState halfLoad( .5 );
    integrator.SetIterAux( &halfLoad );
    const mfem::Vector halfExpansion = UniformExpansion( *element, *transformation, targetStrain * .5 );
    mfem::Vector halfLoadResidual;
    integrator.AssembleElementVector( *element, *transformation, halfExpansion, halfLoadResidual );
    EXPECT_LE( halfLoadResidual.Norml2(), kResidualTolerance * ( 1. + constrainedResidual.Norml2() ) );

    mfem::DenseMatrix thermalTangent;
    integrator.AssembleElementGrad( *element, *transformation, constrainedDisplacement, thermalTangent );

    IsotropicElasticMaterial referenceMaterial( elasticModulus, poissonRatio );
    plugin::IntegrationPointStorage referencePointStorage( &mesh );
    plugin::NonlinearElasticityIntegrator referenceIntegrator( referenceMaterial, referencePointStorage );
    referenceIntegrator.setNonlinear( false );
    referenceIntegrator.SetIterAux( &halfLoad );
    mfem::DenseMatrix referenceTangent;
    referenceIntegrator.AssembleElementGrad( *element, *transformation, constrainedDisplacement, referenceTangent );

    ExpectMatricesNear( thermalTangent, referenceTangent );
}

TEST( ThermalStrain, UsesMultiplicativeSplitAtLargeDeformation )
{
    constexpr mfem::real_t thermalExpansion = 1e-2;
    constexpr mfem::real_t referenceTemperature = 20.;
    constexpr mfem::real_t targetTemperature = 30.;
    constexpr mfem::real_t thermalLogStrain = thermalExpansion * ( targetTemperature - referenceTemperature );
    const mfem::real_t thermalStretch = std::exp( thermalLogStrain );

    mfem::Mesh mesh = mfem::Mesh::MakeCartesian3D( 1, 1, 1, mfem::Element::HEXAHEDRON, 1., 1., 1. );
    mfem::H1_FECollection collection( 1, mesh.Dimension() );
    mfem::FiniteElementSpace space( &mesh, &collection );
    const auto* element = space.GetFE( 0 );
    auto* transformation = mesh.GetElementTransformation( 0 );
    ASSERT_NE( element, nullptr );
    ASSERT_NE( transformation, nullptr );

    mfem::ConstantCoefficient shearModulus( 400. );
    mfem::ConstantCoefficient lameLambda( 600. );
    mfem::ConstantCoefficient expansionCoefficient( thermalExpansion );
    mfem::ConstantCoefficient referenceTemperatureCoefficient( referenceTemperature );
    mfem::ConstantCoefficient targetTemperatureCoefficient( targetTemperature );

    NeoHookeanMaterial material( shearModulus, lameLambda, NeoHookeanType::Ln );
    plugin::IsotropicThermalExpansion thermalExpansionModel( expansionCoefficient, targetTemperatureCoefficient,
                                                             referenceTemperatureCoefficient );
    plugin::IntegrationPointStorage pointStorage( &mesh );
    plugin::NonlinearElasticityIntegrator integrator( material, pointStorage );
    integrator.AddStressFreeDeformation( thermalExpansionModel );

    FixedLoadState unitLoad( 1. );
    integrator.SetIterAux( &unitLoad );

    mfem::Vector constrainedDisplacement( element->GetDof() * element->GetDim() );
    constrainedDisplacement = 0.;
    mfem::Vector constrainedResidual;
    integrator.AssembleElementVector( *element, *transformation, constrainedDisplacement, constrainedResidual );
    ASSERT_GT( constrainedResidual.Norml2(), 1e-2 );

    const mfem::Vector freeExpansion = UniformExpansion( *element, *transformation, thermalStretch - 1. );
    mfem::Vector freeResidual;
    integrator.AssembleElementVector( *element, *transformation, freeExpansion, freeResidual );
    EXPECT_LE( freeResidual.Norml2(), kResidualTolerance * ( 1. + constrainedResidual.Norml2() ) );

    mfem::DenseMatrix tangent;
    integrator.AssembleElementGrad( *element, *transformation, freeExpansion, tangent );

    mfem::Vector direction( freeExpansion.Size() );
    for ( int i = 0; i < direction.Size(); i++ )
    {
        direction( i ) = static_cast<mfem::real_t>( ( i % 7 ) - 3 );
    }
    direction /= direction.Norml2();

    const mfem::real_t step = kSinglePrecision ? 2e-3f : 1e-6;
    mfem::Vector plus( freeExpansion );
    mfem::Vector minus( freeExpansion );
    plus.Add( step, direction );
    minus.Add( -step, direction );

    mfem::Vector plusResidual;
    mfem::Vector minusResidual;
    integrator.AssembleElementVector( *element, *transformation, plus, plusResidual );
    integrator.AssembleElementVector( *element, *transformation, minus, minusResidual );

    mfem::Vector numericalDerivative( plusResidual );
    numericalDerivative -= minusResidual;
    numericalDerivative /= 2. * step;

    mfem::Vector analyticalDerivative( direction.Size() );
    tangent.Mult( direction, analyticalDerivative );
    analyticalDerivative -= numericalDerivative;
    EXPECT_LE( analyticalDerivative.Norml2(), kTangentTolerance * ( 1. + numericalDerivative.Norml2() ) );
}
