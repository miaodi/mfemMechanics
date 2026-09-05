#include "Contact.h"

#include <array>
#include <cmath>
#include <gtest/gtest.h>
#include <initializer_list>
#include <type_traits>

namespace
{
constexpr bool kSinglePrecision = std::is_same_v<mfem::real_t, float>;
constexpr mfem::real_t kGeometryTolerance = kSinglePrecision ? 2e-5f : 2e-13;
constexpr mfem::real_t kAssemblyTolerance = kSinglePrecision ? 2e-4f : 2e-12;
constexpr mfem::real_t kDerivativeTolerance = kSinglePrecision ? 8e-3f : 2e-6;
constexpr mfem::real_t kFiniteDifferenceStep = kSinglePrecision ? 1e-3f : 1e-7;

mfem::Vector MakeVector( const std::initializer_list<mfem::real_t> values )
{
    mfem::Vector vector( static_cast<int>( values.size() ) );
    int index = 0;
    for ( const mfem::real_t value : values )
    {
        vector( index++ ) = value;
    }
    return vector;
}

int FindBoundaryElement( const mfem::Mesh& mesh, const int attribute )
{
    for ( int boundaryElement = 0; boundaryElement < mesh.GetNBE(); boundaryElement++ )
    {
        if ( mesh.GetBdrAttribute( boundaryElement ) == attribute )
        {
            return boundaryElement;
        }
    }
    return -1;
}

void ExpectVectorNear( const mfem::Vector& actual, const mfem::Vector& expected, const mfem::real_t tolerance )
{
    ASSERT_EQ( actual.Size(), expected.Size() );
    for ( int index = 0; index < actual.Size(); index++ )
    {
        EXPECT_NEAR( actual( index ), expected( index ), tolerance * ( 1. + std::abs( expected( index ) ) ) );
    }
}

std::array<mfem::real_t, 2> AssembleContactResultant( const mfem::Ordering::Type ordering )
{
    constexpr mfem::real_t width = 1.7;
    constexpr mfem::real_t penetration = .08;
    constexpr mfem::real_t penalty = 23.;
    mfem::Mesh mesh = mfem::Mesh::MakeCartesian2D( 2, 1, mfem::Element::QUADRILATERAL, true, width, 1. );
    mfem::H1_FECollection collection( 1, mesh.Dimension() );
    mfem::FiniteElementSpace space( &mesh, &collection, mesh.Dimension(), ordering );

    mfem::Vector planePoint = MakeVector( { 0., 0. } );
    mfem::Vector planeNormal = MakeVector( { 0., 1. } );
    plugin::RigidPlaneObstacle obstacle( planePoint, planeNormal );
    mfem::Array<int> contactBoundary( mesh.bdr_attributes.Max() );
    contactBoundary = 0;
    contactBoundary[0] = 1;
    mfem::NonlinearForm contactForm( &space );
    contactForm.AddBdrFaceIntegrator( new plugin::FrictionlessPenaltyContactIntegrator( obstacle, penalty ), contactBoundary );

    mfem::Vector displacement( space.GetVSize() );
    displacement = 0.;
    for ( int degreeOfFreedom = 0; degreeOfFreedom < space.GetNDofs(); degreeOfFreedom++ )
    {
        displacement( space.DofToVDof( degreeOfFreedom, 1 ) ) = -penetration;
    }

    mfem::Vector residual( space.GetTrueVSize() );
    contactForm.Mult( displacement, residual );
    std::array<mfem::real_t, 2> resultant{ 0., 0. };
    for ( int component = 0; component < mesh.Dimension(); component++ )
    {
        for ( int degreeOfFreedom = 0; degreeOfFreedom < space.GetNDofs(); degreeOfFreedom++ )
        {
            resultant[component] += residual( space.DofToVDof( degreeOfFreedom, component ) );
        }
    }
    return resultant;
}
} // namespace

TEST( RigidObstacle, PlaneNormalIsNormalizedAndOrientedTowardPositiveGap )
{
    const mfem::Vector point = MakeVector( { 1., -2. } );
    const mfem::Vector normal = MakeVector( { 3., 4. } );
    const plugin::RigidPlaneObstacle obstacle( point, normal );
    const mfem::Vector position = MakeVector( { 4., 2. } );
    plugin::SignedDistanceEvaluation evaluation;

    obstacle.Evaluate( position, evaluation );

    EXPECT_NEAR( evaluation.Gap, 5., kGeometryTolerance );
    const mfem::Vector expectedGradient = MakeVector( { .6, .8 } );
    ExpectVectorNear( evaluation.Gradient, expectedGradient, kGeometryTolerance );
    EXPECT_LE( evaluation.Hessian.FNorm(), kGeometryTolerance );
}

TEST( RigidObstacle, SphereReturnsSignedDistanceGradientAndHessian )
{
    const mfem::Vector center = MakeVector( { 1., -1. } );
    const plugin::RigidSphereObstacle obstacle( center, 2. );
    const mfem::Vector position = MakeVector( { 4., 3. } );
    plugin::SignedDistanceEvaluation evaluation;

    obstacle.Evaluate( position, evaluation );

    EXPECT_NEAR( evaluation.Gap, 3., kGeometryTolerance );
    const mfem::Vector expectedGradient = MakeVector( { .6, .8 } );
    ExpectVectorNear( evaluation.Gradient, expectedGradient, kGeometryTolerance );
    EXPECT_NEAR( evaluation.Hessian( 0, 0 ), .128, kGeometryTolerance );
    EXPECT_NEAR( evaluation.Hessian( 0, 1 ), -.096, kGeometryTolerance );
    EXPECT_NEAR( evaluation.Hessian( 1, 0 ), -.096, kGeometryTolerance );
    EXPECT_NEAR( evaluation.Hessian( 1, 1 ), .072, kGeometryTolerance );

    mfem::DenseMatrix numericalHessian( obstacle.Dimension() );
    for ( int column = 0; column < obstacle.Dimension(); column++ )
    {
        mfem::Vector plus( position );
        mfem::Vector minus( position );
        plus( column ) += kFiniteDifferenceStep;
        minus( column ) -= kFiniteDifferenceStep;
        plugin::SignedDistanceEvaluation plusEvaluation;
        plugin::SignedDistanceEvaluation minusEvaluation;
        obstacle.Evaluate( plus, plusEvaluation );
        obstacle.Evaluate( minus, minusEvaluation );
        for ( int row = 0; row < obstacle.Dimension(); row++ )
        {
            numericalHessian( row, column ) =
                ( plusEvaluation.Gradient( row ) - minusEvaluation.Gradient( row ) ) / ( 2. * kFiniteDifferenceStep );
        }
    }
    mfem::DenseMatrix hessianError( evaluation.Hessian );
    hessianError -= numericalHessian;
    EXPECT_LE( hessianError.FNorm(), kDerivativeTolerance * ( 1. + numericalHessian.FNorm() ) );
}

TEST( RigidObstacle, RejectsInvalidGeometryAndPenaltyParameters )
{
    GTEST_FLAG_SET( death_test_style, "threadsafe" );
    const mfem::Vector point = MakeVector( { 0., 0. } );
    const mfem::Vector zeroNormal = MakeVector( { 0., 0. } );
    EXPECT_DEATH( (void)plugin::RigidPlaneObstacle( point, zeroNormal ), "normal" );
    EXPECT_DEATH( (void)plugin::RigidSphereObstacle( point, 0. ), "radius" );

    const plugin::RigidSphereObstacle sphere( point, 1. );
    plugin::SignedDistanceEvaluation evaluation;
    EXPECT_DEATH( sphere.Evaluate( point, evaluation ), "not differentiable" );

    const mfem::Vector planeNormal = MakeVector( { 0., 1. } );
    const plugin::RigidPlaneObstacle plane( point, planeNormal );
    EXPECT_DEATH( (void)plugin::FrictionlessPenaltyContactIntegrator( plane, 0. ), "penalty" );
}

TEST( PenaltyContact, PenetratedPlaneHasCorrectResidualSignScaleAndNormalTangent )
{
    constexpr mfem::real_t width = 2.3;
    constexpr mfem::real_t penetration = .12;
    constexpr mfem::real_t penalty = 17.;
    mfem::Mesh mesh = mfem::Mesh::MakeCartesian2D( 1, 1, mfem::Element::QUADRILATERAL, true, width, 1. );
    mfem::H1_FECollection collection( 1, mesh.Dimension() );
    mfem::FiniteElementSpace space( &mesh, &collection, mesh.Dimension(), mfem::Ordering::byVDIM );
    const int boundaryElement = FindBoundaryElement( mesh, 1 );
    ASSERT_GE( boundaryElement, 0 );
    auto* transformation = mesh.GetBdrFaceTransformations( boundaryElement );
    ASSERT_NE( transformation, nullptr );
    const auto* element = space.GetFE( transformation->Elem1No );
    ASSERT_NE( element, nullptr );

    const mfem::Vector planePoint = MakeVector( { 0., 0. } );
    const mfem::Vector planeNormal = MakeVector( { 0., 1. } );
    const plugin::RigidPlaneObstacle obstacle( planePoint, planeNormal );
    plugin::FrictionlessPenaltyContactIntegrator integrator( obstacle, penalty );
    const int degreeOfFreedomCount = element->GetDof();
    mfem::Vector displacement( degreeOfFreedomCount * mesh.Dimension() );
    displacement = 0.;
    for ( int degreeOfFreedom = 0; degreeOfFreedom < degreeOfFreedomCount; degreeOfFreedom++ )
    {
        displacement( degreeOfFreedom + degreeOfFreedomCount ) = -penetration;
    }

    mfem::Vector residual;
    mfem::DenseMatrix tangent;
    integrator.AssembleFaceVector( *element, *element, *transformation, displacement, residual );
    integrator.AssembleFaceGrad( *element, *element, *transformation, displacement, tangent );

    mfem::real_t horizontalResidual = 0.;
    mfem::real_t verticalResidual = 0.;
    for ( int degreeOfFreedom = 0; degreeOfFreedom < degreeOfFreedomCount; degreeOfFreedom++ )
    {
        horizontalResidual += residual( degreeOfFreedom );
        verticalResidual += residual( degreeOfFreedom + degreeOfFreedomCount );
    }
    EXPECT_NEAR( horizontalResidual, 0., kAssemblyTolerance );
    EXPECT_NEAR( verticalResidual, -penalty * penetration * width, kAssemblyTolerance * ( 1. + penalty * penetration * width ) );

    mfem::Vector uniformNormalDisplacement( displacement.Size() );
    uniformNormalDisplacement = 0.;
    for ( int degreeOfFreedom = 0; degreeOfFreedom < degreeOfFreedomCount; degreeOfFreedom++ )
    {
        uniformNormalDisplacement( degreeOfFreedom + degreeOfFreedomCount ) = 1.;
    }
    mfem::Vector tangentAction( displacement.Size() );
    tangent.Mult( uniformNormalDisplacement, tangentAction );
    mfem::real_t horizontalTangentResultant = 0.;
    mfem::real_t verticalTangentResultant = 0.;
    for ( int degreeOfFreedom = 0; degreeOfFreedom < degreeOfFreedomCount; degreeOfFreedom++ )
    {
        horizontalTangentResultant += tangentAction( degreeOfFreedom );
        verticalTangentResultant += tangentAction( degreeOfFreedom + degreeOfFreedomCount );
    }
    EXPECT_NEAR( horizontalTangentResultant, 0., kAssemblyTolerance );
    EXPECT_NEAR( verticalTangentResultant, penalty * width, kAssemblyTolerance * ( 1. + penalty * width ) );
}

TEST( PenaltyContact, SupportsThreeDimensionalPlaneContact )
{
    constexpr mfem::real_t width = 1.3;
    constexpr mfem::real_t depth = .7;
    constexpr mfem::real_t penetration = .09;
    constexpr mfem::real_t penalty = 19.;
    mfem::Mesh mesh = mfem::Mesh::MakeCartesian3D( 1, 1, 1, mfem::Element::HEXAHEDRON, width, depth, 1. );
    mfem::H1_FECollection collection( 1, mesh.Dimension() );
    mfem::FiniteElementSpace space( &mesh, &collection, mesh.Dimension(), mfem::Ordering::byVDIM );
    const int boundaryElement = FindBoundaryElement( mesh, 1 );
    ASSERT_GE( boundaryElement, 0 );
    auto* transformation = mesh.GetBdrFaceTransformations( boundaryElement );
    ASSERT_NE( transformation, nullptr );
    const auto* element = space.GetFE( transformation->Elem1No );
    ASSERT_NE( element, nullptr );

    const mfem::Vector planePoint = MakeVector( { 0., 0., 0. } );
    const mfem::Vector planeNormal = MakeVector( { 0., 0., 1. } );
    const plugin::RigidPlaneObstacle obstacle( planePoint, planeNormal );
    plugin::FrictionlessPenaltyContactIntegrator integrator( obstacle, penalty );
    const int degreeOfFreedomCount = element->GetDof();
    mfem::Vector displacement( degreeOfFreedomCount * mesh.Dimension() );
    displacement = 0.;
    for ( int degreeOfFreedom = 0; degreeOfFreedom < degreeOfFreedomCount; degreeOfFreedom++ )
    {
        displacement( degreeOfFreedom + 2 * degreeOfFreedomCount ) = -penetration;
    }

    mfem::Vector residual;
    integrator.AssembleFaceVector( *element, *element, *transformation, displacement, residual );
    std::array<mfem::real_t, 3> resultant{ 0., 0., 0. };
    for ( int component = 0; component < mesh.Dimension(); component++ )
    {
        for ( int degreeOfFreedom = 0; degreeOfFreedom < degreeOfFreedomCount; degreeOfFreedom++ )
        {
            resultant[component] += residual( degreeOfFreedom + component * degreeOfFreedomCount );
        }
    }

    EXPECT_NEAR( resultant[0], 0., kAssemblyTolerance );
    EXPECT_NEAR( resultant[1], 0., kAssemblyTolerance );
    EXPECT_NEAR( resultant[2], -penalty * penetration * width * depth,
                 kAssemblyTolerance * ( 1. + penalty * penetration * width * depth ) );
}

TEST( PenaltyContact, TouchingUsesActiveTangentWhileSeparationIsInactive )
{
    constexpr mfem::real_t penalty = 13.;
    mfem::Mesh mesh = mfem::Mesh::MakeCartesian2D( 1, 1, mfem::Element::QUADRILATERAL );
    mfem::H1_FECollection collection( 1, mesh.Dimension() );
    mfem::FiniteElementSpace space( &mesh, &collection, mesh.Dimension(), mfem::Ordering::byVDIM );
    const int boundaryElement = FindBoundaryElement( mesh, 1 );
    ASSERT_GE( boundaryElement, 0 );
    auto* transformation = mesh.GetBdrFaceTransformations( boundaryElement );
    ASSERT_NE( transformation, nullptr );
    const auto* element = space.GetFE( transformation->Elem1No );
    ASSERT_NE( element, nullptr );

    const mfem::Vector planePoint = MakeVector( { 0., 0. } );
    const mfem::Vector planeNormal = MakeVector( { 0., 1. } );
    const plugin::RigidPlaneObstacle obstacle( planePoint, planeNormal );
    plugin::FrictionlessPenaltyContactIntegrator integrator( obstacle, penalty );
    mfem::Vector displacement( element->GetDof() * mesh.Dimension() );
    displacement = 0.;

    mfem::Vector touchingResidual;
    mfem::DenseMatrix touchingTangent;
    integrator.AssembleFaceVector( *element, *element, *transformation, displacement, touchingResidual );
    integrator.AssembleFaceGrad( *element, *element, *transformation, displacement, touchingTangent );
    EXPECT_LE( touchingResidual.Norml2(), kAssemblyTolerance );
    EXPECT_GT( touchingTangent.FNorm(), penalty * .1 );

    for ( int degreeOfFreedom = 0; degreeOfFreedom < element->GetDof(); degreeOfFreedom++ )
    {
        displacement( degreeOfFreedom + element->GetDof() ) = .1;
    }
    mfem::Vector separatedResidual;
    mfem::DenseMatrix separatedTangent;
    integrator.AssembleFaceVector( *element, *element, *transformation, displacement, separatedResidual );
    integrator.AssembleFaceGrad( *element, *element, *transformation, displacement, separatedTangent );
    EXPECT_LE( separatedResidual.Norml2(), kAssemblyTolerance );
    EXPECT_LE( separatedTangent.FNorm(), kAssemblyTolerance );
}

TEST( PenaltyContact, CurvedObstacleTangentMatchesDirectionalDifference )
{
    mfem::Mesh mesh = mfem::Mesh::MakeCartesian2D( 1, 1, mfem::Element::QUADRILATERAL, true, 1., 1. );
    mfem::H1_FECollection collection( 2, mesh.Dimension() );
    mfem::FiniteElementSpace space( &mesh, &collection, mesh.Dimension(), mfem::Ordering::byVDIM );
    const int boundaryElement = FindBoundaryElement( mesh, 1 );
    ASSERT_GE( boundaryElement, 0 );
    auto* transformation = mesh.GetBdrFaceTransformations( boundaryElement );
    ASSERT_NE( transformation, nullptr );
    const auto* element = space.GetFE( transformation->Elem1No );
    ASSERT_NE( element, nullptr );

    const mfem::Vector center = MakeVector( { .5, -.4 } );
    const plugin::RigidSphereObstacle obstacle( center, 1.2 );
    plugin::FrictionlessPenaltyContactIntegrator integrator( obstacle, 31. );
    mfem::Vector displacement( element->GetDof() * mesh.Dimension() );
    mfem::Vector direction( displacement.Size() );
    for ( int index = 0; index < displacement.Size(); index++ )
    {
        displacement( index ) = .007 * static_cast<mfem::real_t>( ( index % 5 ) - 2 );
        direction( index ) = static_cast<mfem::real_t>( ( index * 7 ) % 11 - 5 );
    }
    direction /= direction.Norml2();

    mfem::DenseMatrix tangent;
    integrator.AssembleFaceGrad( *element, *element, *transformation, displacement, tangent );
    mfem::Vector analyticalDerivative( displacement.Size() );
    tangent.Mult( direction, analyticalDerivative );

    mfem::Vector plus( displacement );
    mfem::Vector minus( displacement );
    plus.Add( kFiniteDifferenceStep, direction );
    minus.Add( -kFiniteDifferenceStep, direction );
    mfem::Vector plusResidual;
    mfem::Vector minusResidual;
    integrator.AssembleFaceVector( *element, *element, *transformation, plus, plusResidual );
    integrator.AssembleFaceVector( *element, *element, *transformation, minus, minusResidual );
    mfem::Vector numericalDerivative( plusResidual );
    numericalDerivative -= minusResidual;
    numericalDerivative /= 2. * kFiniteDifferenceStep;

    analyticalDerivative -= numericalDerivative;
    EXPECT_LE( analyticalDerivative.Norml2(), kDerivativeTolerance * ( 1. + numericalDerivative.Norml2() ) );
}

TEST( PenaltyContact, ThreeDimensionalSphereTangentMatchesDirectionalDifference )
{
    mfem::Mesh mesh = mfem::Mesh::MakeCartesian3D( 1, 1, 1, mfem::Element::HEXAHEDRON, 1., .8, 1. );
    mfem::H1_FECollection collection( 1, mesh.Dimension() );
    mfem::FiniteElementSpace space( &mesh, &collection, mesh.Dimension(), mfem::Ordering::byVDIM );
    const int boundaryElement = FindBoundaryElement( mesh, 1 );
    ASSERT_GE( boundaryElement, 0 );
    auto* transformation = mesh.GetBdrFaceTransformations( boundaryElement );
    ASSERT_NE( transformation, nullptr );
    const auto* element = space.GetFE( transformation->Elem1No );
    ASSERT_NE( element, nullptr );

    const mfem::Vector center = MakeVector( { .5, .4, -.5 } );
    const plugin::RigidSphereObstacle obstacle( center, 1.5 );
    plugin::FrictionlessPenaltyContactIntegrator integrator( obstacle, 29. );
    mfem::Vector displacement( element->GetDof() * mesh.Dimension() );
    mfem::Vector direction( displacement.Size() );
    for ( int index = 0; index < displacement.Size(); index++ )
    {
        displacement( index ) = .004 * static_cast<mfem::real_t>( ( index % 7 ) - 3 );
        direction( index ) = static_cast<mfem::real_t>( ( index * 5 ) % 13 - 6 );
    }
    direction /= direction.Norml2();

    mfem::DenseMatrix tangent;
    integrator.AssembleFaceGrad( *element, *element, *transformation, displacement, tangent );
    mfem::Vector analyticalDerivative( displacement.Size() );
    tangent.Mult( direction, analyticalDerivative );

    mfem::Vector plus( displacement );
    mfem::Vector minus( displacement );
    plus.Add( kFiniteDifferenceStep, direction );
    minus.Add( -kFiniteDifferenceStep, direction );
    mfem::Vector plusResidual;
    mfem::Vector minusResidual;
    integrator.AssembleFaceVector( *element, *element, *transformation, plus, plusResidual );
    integrator.AssembleFaceVector( *element, *element, *transformation, minus, minusResidual );
    mfem::Vector numericalDerivative( plusResidual );
    numericalDerivative -= minusResidual;
    numericalDerivative /= 2. * kFiniteDifferenceStep;

    analyticalDerivative -= numericalDerivative;
    EXPECT_LE( analyticalDerivative.Norml2(), kDerivativeTolerance * ( 1. + numericalDerivative.Norml2() ) );
}

TEST( PenaltyContact, GlobalAssemblyIsIndependentOfVectorOrdering )
{
    constexpr mfem::real_t width = 1.7;
    constexpr mfem::real_t penetration = .08;
    constexpr mfem::real_t penalty = 23.;
    const auto byVdimResultant = AssembleContactResultant( mfem::Ordering::byVDIM );
    const auto byNodesResultant = AssembleContactResultant( mfem::Ordering::byNODES );

    EXPECT_NEAR( byVdimResultant[0], 0., kAssemblyTolerance );
    EXPECT_NEAR( byVdimResultant[1], -penalty * penetration * width, kAssemblyTolerance * ( 1. + penalty * penetration * width ) );
    EXPECT_NEAR( byNodesResultant[0], byVdimResultant[0], kAssemblyTolerance );
    EXPECT_NEAR( byNodesResultant[1], byVdimResultant[1], kAssemblyTolerance * ( 1. + std::abs( byVdimResultant[1] ) ) );
}
