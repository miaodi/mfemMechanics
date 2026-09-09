#include "../examples/contact/GapSampling.h"
#include "Contact.h"

#include <array>
#include <chrono>
#include <cmath>
#include <gtest/gtest.h>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

namespace
{
constexpr bool kSinglePrecision = std::is_same_v<mfem::real_t, float>;
constexpr mfem::real_t kGeometryTolerance = kSinglePrecision ? 2e-5f : 2e-13;
constexpr mfem::real_t kAssemblyTolerance = kSinglePrecision ? 2e-4f : 2e-12;
constexpr mfem::real_t kDerivativeTolerance = kSinglePrecision ? 8e-3f : 2e-6;
constexpr mfem::real_t kFiniteDifferenceStep = kSinglePrecision ? 1e-3f : 1e-7;

static_assert( !std::is_constructible_v<plugin::SemismoothRigidContactOperator,
                                        mfem::Operator&,
                                        mfem::IdentityOperator&&,
                                        mfem::FiniteElementSpace&,
                                        const plugin::RigidObstacle&,
                                        plugin::BoundaryMultiplierSpace&,
                                        const mfem::Array<int>&,
                                        mfem::real_t> );
static_assert( !std::is_constructible_v<plugin::SemismoothRigidContactOperator,
                                        mfem::Operator&,
                                        const mfem::IdentityOperator&&,
                                        mfem::FiniteElementSpace&,
                                        const plugin::RigidObstacle&,
                                        plugin::BoundaryMultiplierSpace&,
                                        const mfem::Array<int>&,
                                        mfem::real_t> );
static_assert( !std::is_constructible_v<plugin::SemismoothRigidContactOperator,
                                        mfem::Operator&,
                                        const mfem::Operator&,
                                        mfem::FiniteElementSpace&,
                                        plugin::RigidPlaneObstacle&&,
                                        plugin::BoundaryMultiplierSpace&,
                                        const mfem::Array<int>&,
                                        mfem::real_t> );
static_assert( !std::is_constructible_v<plugin::SemismoothRigidContactOperator,
                                        mfem::Operator&,
                                        const mfem::Operator&,
                                        mfem::FiniteElementSpace&,
                                        const plugin::RigidPlaneObstacle&&,
                                        plugin::BoundaryMultiplierSpace&,
                                        const mfem::Array<int>&,
                                        mfem::real_t> );

static_assert( std::is_constructible_v<plugin::SemismoothRigidContactOperator,
                                       mfem::Operator&,
                                       const mfem::Operator&,
                                       mfem::FiniteElementSpace&,
                                       const plugin::RigidObstacle&,
                                       plugin::BoundaryMultiplierSpace&,
                                       const mfem::Array<int>&,
                                       mfem::real_t> );
static_assert( !std::is_constructible_v<plugin::SemismoothRigidContactOperator,
                                        mfem::Operator&,
                                        const mfem::Operator&,
                                        mfem::FiniteElementSpace&,
                                        const plugin::RigidObstacle&,
                                        plugin::BoundaryMultiplierSpace&&,
                                        const mfem::Array<int>&,
                                        mfem::real_t> );
static_assert( !std::is_constructible_v<plugin::SemismoothRigidContactOperator,
                                        mfem::Operator&,
                                        const mfem::Operator&,
                                        mfem::FiniteElementSpace&,
                                        const plugin::RigidObstacle&,
                                        const plugin::BoundaryMultiplierSpace&&,
                                        const mfem::Array<int>&,
                                        mfem::real_t> );
static_assert( !std::is_copy_constructible_v<plugin::BoundaryMultiplierSpace> );
static_assert( !std::is_move_constructible_v<plugin::BoundaryMultiplierSpace> );
static_assert( !std::is_copy_assignable_v<plugin::BoundaryMultiplierSpace> );
static_assert( !std::is_move_assignable_v<plugin::BoundaryMultiplierSpace> );
static_assert( !std::is_constructible_v<plugin::BoundaryMultiplierSpace, mfem::Mesh&&, const mfem::Array<int>&> );
static_assert( !std::is_copy_constructible_v<plugin::SemismoothRigidContactOperator> );
static_assert( !std::is_move_constructible_v<plugin::SemismoothRigidContactOperator> );
static_assert( !std::is_copy_assignable_v<plugin::SemismoothRigidContactOperator> );
static_assert( !std::is_move_assignable_v<plugin::SemismoothRigidContactOperator> );

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
    contactForm.AddBoundaryIntegrator( new plugin::FrictionlessPenaltyContactIntegrator( obstacle, penalty ), contactBoundary );

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

class DiagonalNonlinearOperator final : public mfem::Operator
{
public:
    explicit DiagonalNonlinearOperator( const mfem::Vector& diagonal )
        : mfem::Operator( diagonal.Size() ), mDiagonal( diagonal ), mGradient( diagonal.Size() )
    {
        mGradient = 0.;
        for ( int i = 0; i < diagonal.Size(); i++ )
        {
            mGradient( i, i ) = diagonal( i );
        }
    }

    void Mult( const mfem::Vector& input, mfem::Vector& output ) const override
    {
        ASSERT_EQ( input.Size(), Width() );
        output.SetSize( Height() );
        for ( int i = 0; i < input.Size(); i++ )
        {
            output( i ) = mDiagonal( i ) * input( i );
        }
    }

    mfem::Operator& GetGradient( const mfem::Vector& ) const override
    {
        return mGradient;
    }

private:
    mfem::Vector mDiagonal;
    mutable mfem::DenseMatrix mGradient;
};

class LeadingIdentityOperator final : public mfem::Operator
{
public:
    LeadingIdentityOperator( const int selectedSize, const int inputSize ) : mfem::Operator( selectedSize, inputSize )
    {
        MFEM_VERIFY( selectedSize <= inputSize, "A leading-field extraction cannot exceed its input size." );
    }

    void Mult( const mfem::Vector& input, mfem::Vector& output ) const override
    {
        output.SetSize( Height() );
        for ( int i = 0; i < Height(); i++ )
        {
            output( i ) = input( i );
        }
    }

    void MultTranspose( const mfem::Vector& input, mfem::Vector& output ) const override
    {
        output.SetSize( Width() );
        output = 0.;
        for ( int i = 0; i < Height(); i++ )
        {
            output( i ) = input( i );
        }
    }
};

mfem::Array<int> ContactAttributes( const int attribute )
{
    mfem::Array<int> attributes( 1 );
    attributes[0] = attribute;
    return attributes;
}

void SetUniformDisplacement( const mfem::FiniteElementSpace& space, mfem::Vector& unknown, const int component, const mfem::real_t value )
{
    for ( int degreeOfFreedom = 0; degreeOfFreedom < space.GetNDofs(); degreeOfFreedom++ )
    {
        unknown( space.DofToVDof( degreeOfFreedom, component ) ) = value;
    }
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
    auto* transformation = mesh.GetBdrElementTransformation( boundaryElement );
    ASSERT_NE( transformation, nullptr );
    const auto* element = space.GetBE( boundaryElement );
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
    integrator.AssembleElementVector( *element, *transformation, displacement, residual );
    integrator.AssembleElementGrad( *element, *transformation, displacement, tangent );

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
    auto* transformation = mesh.GetBdrElementTransformation( boundaryElement );
    ASSERT_NE( transformation, nullptr );
    const auto* element = space.GetBE( boundaryElement );
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
    integrator.AssembleElementVector( *element, *transformation, displacement, residual );
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
    auto* transformation = mesh.GetBdrElementTransformation( boundaryElement );
    ASSERT_NE( transformation, nullptr );
    const auto* element = space.GetBE( boundaryElement );
    ASSERT_NE( element, nullptr );

    const mfem::Vector planePoint = MakeVector( { 0., 0. } );
    const mfem::Vector planeNormal = MakeVector( { 0., 1. } );
    const plugin::RigidPlaneObstacle obstacle( planePoint, planeNormal );
    plugin::FrictionlessPenaltyContactIntegrator integrator( obstacle, penalty );
    mfem::Vector displacement( element->GetDof() * mesh.Dimension() );
    displacement = 0.;

    mfem::Vector touchingResidual;
    mfem::DenseMatrix touchingTangent;
    integrator.AssembleElementVector( *element, *transformation, displacement, touchingResidual );
    integrator.AssembleElementGrad( *element, *transformation, displacement, touchingTangent );
    EXPECT_LE( touchingResidual.Norml2(), kAssemblyTolerance );
    EXPECT_GT( touchingTangent.FNorm(), penalty * .1 );

    for ( int degreeOfFreedom = 0; degreeOfFreedom < element->GetDof(); degreeOfFreedom++ )
    {
        displacement( degreeOfFreedom + element->GetDof() ) = .1;
    }
    mfem::Vector separatedResidual;
    mfem::DenseMatrix separatedTangent;
    integrator.AssembleElementVector( *element, *transformation, displacement, separatedResidual );
    integrator.AssembleElementGrad( *element, *transformation, displacement, separatedTangent );
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
    auto* transformation = mesh.GetBdrElementTransformation( boundaryElement );
    ASSERT_NE( transformation, nullptr );
    const auto* element = space.GetBE( boundaryElement );
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
    integrator.AssembleElementGrad( *element, *transformation, displacement, tangent );
    mfem::Vector analyticalDerivative( displacement.Size() );
    tangent.Mult( direction, analyticalDerivative );

    mfem::Vector plus( displacement );
    mfem::Vector minus( displacement );
    plus.Add( kFiniteDifferenceStep, direction );
    minus.Add( -kFiniteDifferenceStep, direction );
    mfem::Vector plusResidual;
    mfem::Vector minusResidual;
    integrator.AssembleElementVector( *element, *transformation, plus, plusResidual );
    integrator.AssembleElementVector( *element, *transformation, minus, minusResidual );
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
    auto* transformation = mesh.GetBdrElementTransformation( boundaryElement );
    ASSERT_NE( transformation, nullptr );
    const auto* element = space.GetBE( boundaryElement );
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
    integrator.AssembleElementGrad( *element, *transformation, displacement, tangent );
    mfem::Vector analyticalDerivative( displacement.Size() );
    tangent.Mult( direction, analyticalDerivative );

    mfem::Vector plus( displacement );
    mfem::Vector minus( displacement );
    plus.Add( kFiniteDifferenceStep, direction );
    minus.Add( -kFiniteDifferenceStep, direction );
    mfem::Vector plusResidual;
    mfem::Vector minusResidual;
    integrator.AssembleElementVector( *element, *transformation, plus, plusResidual );
    integrator.AssembleElementVector( *element, *transformation, minus, minusResidual );
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

TEST( SemismoothContact, RejectsInvalidComplianceAndJumpWeight )
{
    GTEST_FLAG_SET( death_test_style, "threadsafe" );
    mfem::Mesh mesh = mfem::Mesh::MakeCartesian2D( 1, 1, mfem::Element::QUADRILATERAL );
    mfem::H1_FECollection collection( 1, mesh.Dimension() );
    mfem::FiniteElementSpace space( &mesh, &collection, mesh.Dimension() );
    mfem::NonlinearForm primal( &space );
    mfem::IdentityOperator displacementExtraction( space.GetTrueVSize() );
    const plugin::RigidPlaneObstacle obstacle( MakeVector( { 0., 0. } ), MakeVector( { 0., 1. } ) );
    const mfem::Array<int> contactAttributes = ContactAttributes( 1 );
    plugin::BoundaryMultiplierSpace multipliers( mesh, contactAttributes );
    mfem::Array<int> essentialDofs;
    constexpr mfem::real_t infinity = std::numeric_limits<mfem::real_t>::infinity();
    constexpr mfem::real_t nan = std::numeric_limits<mfem::real_t>::quiet_NaN();
    for ( const mfem::real_t gamma : std::array<mfem::real_t, 4>{ 0., -1., infinity, nan } )
    {
        EXPECT_DEATH( (void)plugin::SemismoothRigidContactOperator( primal, displacementExtraction, space, obstacle,
                                                                    multipliers, essentialDofs, gamma ),
                      "gamma must be positive and finite" );
    }
    // Positive and finite is insufficient if 1/gamma overflows mfem::real_t.
    constexpr mfem::real_t tinyGamma = std::numeric_limits<mfem::real_t>::denorm_min();
    EXPECT_DEATH( (void)plugin::SemismoothRigidContactOperator( primal, displacementExtraction, space, obstacle,
                                                                multipliers, essentialDofs, tinyGamma ),
                  "gamma must have a finite reciprocal" );

    constexpr mfem::real_t gamma = 1. / 17.;
    for ( const mfem::real_t delta : std::array<mfem::real_t, 3>{ -1., infinity, nan } )
    {
        EXPECT_DEATH( (void)plugin::SemismoothRigidContactOperator( primal, displacementExtraction, space, obstacle,
                                                                    multipliers, essentialDofs, gamma, 1., 1., delta ),
                      "delta must be finite and nonnegative" );
    }
}

TEST( SemismoothContact, BoundaryP0HasOneMultiplierPerSelectedFace )
{
    mfem::Mesh mesh = mfem::Mesh::MakeCartesian2D( 3, 2, mfem::Element::QUADRILATERAL );
    mfem::H1_FECollection collection( 1, mesh.Dimension() );
    mfem::FiniteElementSpace space( &mesh, &collection, mesh.Dimension() );
    mfem::Vector diagonal( space.GetTrueVSize() );
    diagonal = 0.;
    DiagonalNonlinearOperator primal( diagonal );
    mfem::IdentityOperator displacementExtraction( space.GetTrueVSize() );
    const plugin::RigidPlaneObstacle obstacle( MakeVector( { 0., 0. } ), MakeVector( { 0., 1. } ) );
    const mfem::Array<int> contactAttributes = ContactAttributes( 1 );
    mfem::Array<int> essentialDofs;

    plugin::BoundaryMultiplierSpace multipliers( mesh, contactAttributes );
    plugin::SemismoothRigidContactOperator contact( primal, displacementExtraction, space, obstacle, multipliers,
                                                    essentialDofs, 1. / 17. );

    EXPECT_EQ( contact.GetContactSubMesh().GetNE(), 3 );
    EXPECT_EQ( contact.GetMultiplierSpace().GetTrueVSize(), 3 );
    EXPECT_EQ( contact.GetBlockOffsets()[1], space.GetTrueVSize() );
    EXPECT_EQ( contact.Height(), space.GetTrueVSize() + 3 );
    EXPECT_EQ( &contact.GetMultiplierSpace(), &multipliers.GetSpace() );
    EXPECT_EQ( &contact.GetContactSubMesh(), &multipliers.GetMesh() );
}

TEST( BoundaryMultiplierSpace, DefaultP0AndConfiguredP1Q1MatchParentBoundaryPoints )
{
    for ( const int dimension : { 2, 3 } )
        for ( const bool simplex : { false, true } )
        {
            SCOPED_TRACE( ::testing::Message() << dimension << " simplex=" << simplex );
            mfem::Mesh mesh =
                dimension == 2
                    ? mfem::Mesh::MakeCartesian2D( 2, 1, simplex ? mfem::Element::TRIANGLE : mfem::Element::QUADRILATERAL )
                    : mfem::Mesh::MakeCartesian3D( 2, 1, 1, simplex ? mfem::Element::TETRAHEDRON : mfem::Element::HEXAHEDRON );
            mfem::Array<int> attributes( mesh.bdr_attributes );
            plugin::BoundaryMultiplierSpace p0( mesh, attributes );
            auto collection = std::make_unique<mfem::L2_FECollection>( 1, dimension - 1 );
            const auto* ownedCollection = collection.get();
            plugin::BoundaryMultiplierSpace p1( mesh, attributes, std::move( collection ) );
            EXPECT_EQ( collection, nullptr );
            EXPECT_EQ( p1.GetSpace().FEColl(), ownedCollection );
            EXPECT_EQ( p1.GetSpace().GetMesh(), &p1.GetMesh() );
            EXPECT_EQ( p1.GetSpace().GetVDim(), 1 );
            EXPECT_EQ( p0.GetSpace().GetTrueVSize(), mesh.GetNBE() );
            const int dofsPerFace = dimension == 2 ? 2 : ( simplex ? 3 : 4 );
            EXPECT_EQ( p1.GetSpace().GetTrueVSize(), dofsPerFace * mesh.GetNBE() );
            mfem::FunctionCoefficient affine(
                [dimension]( const mfem::Vector& x )
                { return 2. + .7 * x( 0 ) - .4 * x( 1 ) + ( dimension == 3 ? .2 * x( 2 ) : 0. ); } );
            mfem::GridFunction lambda( &p1.GetSpace() );
            lambda.ProjectCoefficient( affine );
            const auto& view = p1;
            EXPECT_EQ( &view.GetMesh(), &p1.GetMesh() );
            EXPECT_EQ( &view.GetSpace(), &p1.GetSpace() );
            mfem::Array<int> seen( mesh.GetNBE() );
            seen = 0;
            for ( int e = 0; e < view.GetMesh().GetNE(); ++e )
            {
                const int be = view.GetParentBoundaryElement( e );
                ASSERT_GE( be, 0 );
                ASSERT_LT( be, mesh.GetNBE() );
                EXPECT_EQ( seen[be]++, 0 );
                mfem::Array<int> dofs, directDofs, constantDofs;
                view.GetElementDofs( e, dofs );
                view.GetSpace().GetElementDofs( e, directDofs );
                p0.GetElementDofs( e, constantDofs );
                ASSERT_EQ( dofs.Size(), dofsPerFace );
                ASSERT_EQ( directDofs.Size(), dofs.Size() );
                ASSERT_EQ( constantDofs.Size(), 1 );
                for ( int j = 0; j < dofs.Size(); ++j )
                {
                    EXPECT_EQ( dofs[j], directDofs[j] );
                }
                mfem::Vector localLambda;
                lambda.GetSubVector( dofs, localLambda );
                for ( const auto t : { mfem::real_t{ .13 }, mfem::real_t{ .61 } } )
                {
                    mfem::IntegrationPoint parentPoint;
                    parentPoint.Set2w( t, .21, 1. );
                    mfem::Vector shape, constantShape, position( dimension );
                    view.CalcShape( e, parentPoint, shape );
                    p0.CalcShape( e, parentPoint, constantShape );
                    ASSERT_EQ( shape.Size(), dofsPerFace );
                    ASSERT_EQ( constantShape.Size(), 1 );
                    EXPECT_NEAR( constantShape( 0 ), 1., kGeometryTolerance );
                    auto* transform = mesh.GetBdrElementTransformation( be );
                    transform->SetIntPoint( &parentPoint );
                    transform->Transform( parentPoint, position );
                    EXPECT_NEAR( shape * localLambda, affine.Eval( *transform, parentPoint ), kAssemblyTolerance );
                    // Reconstruct every physical coordinate from the submesh's FE nodes.
                    // This checks the whole basis permutation, not just partition of unity.
                    const auto& nodes = view.GetSpace().GetFE( e )->GetNodes();
                    mfem::Vector reconstructed( dimension ), node( dimension );
                    reconstructed = 0.;
                    auto* subTransform = p1.GetMesh().GetElementTransformation( e );
                    for ( int j = 0; j < dofsPerFace; ++j )
                    {
                        subTransform->Transform( nodes.IntPoint( j ), node );
                        reconstructed.Add( shape( j ), node );
                    }
                    ExpectVectorNear( reconstructed, position, kGeometryTolerance );
                }
            }
        }
}

TEST( BoundaryMultiplierSpace, RejectsUnsupportedCollectionsAndInvalidAttributes )
{
    GTEST_FLAG_SET( death_test_style, "threadsafe" );
    mfem::Mesh mesh = mfem::Mesh::MakeCartesian2D( 1, 1, mfem::Element::QUADRILATERAL );
    const auto attributes = ContactAttributes( 1 );
    EXPECT_DEATH( (void)plugin::BoundaryMultiplierSpace( mesh, attributes, std::make_unique<mfem::H1_FECollection>( 1, 1 ) ),
                  "L2|discontinuous" );
    EXPECT_DEATH( (void)plugin::BoundaryMultiplierSpace(
                      mesh, attributes,
                      std::make_unique<mfem::L2_FECollection>( 1, 1, mfem::BasisType::GaussLegendre, mfem::FiniteElement::INTEGRAL ) ),
                  "VALUE|value" );
    EXPECT_DEATH( (void)plugin::BoundaryMultiplierSpace( mesh, attributes, std::make_unique<mfem::L2_FECollection>( 1, 2 ) ),
                  "dimension|geometry|finite element|FiniteElement" );
    mfem::Array<int> empty, duplicate( 2 );
    duplicate = 1;
    EXPECT_DEATH( (void)plugin::BoundaryMultiplierSpace( mesh, empty ), "attribute|Attribute" );
    EXPECT_DEATH( (void)plugin::BoundaryMultiplierSpace( mesh, duplicate ), "attribute|Attribute" );
    EXPECT_DEATH( (void)plugin::BoundaryMultiplierSpace( mesh, ContactAttributes( 99 ) ), "attribute|Attribute" );
}

TEST( SemismoothContact, WrapsBlockNonlinearPrimalOperator )
{
    mfem::Mesh mesh = mfem::Mesh::MakeCartesian2D( 1, 1, mfem::Element::QUADRILATERAL );
    mfem::H1_FECollection displacementCollection( 1, mesh.Dimension() );
    mfem::H1_FECollection scalarCollection( 1, mesh.Dimension() );
    mfem::FiniteElementSpace displacementSpace( &mesh, &displacementCollection, mesh.Dimension(), mfem::Ordering::byVDIM );
    mfem::FiniteElementSpace scalarSpace( &mesh, &scalarCollection );
    mfem::Array<mfem::FiniteElementSpace*> primalSpaces( 2 );
    primalSpaces[0] = &displacementSpace;
    primalSpaces[1] = &scalarSpace;
    mfem::BlockNonlinearForm primal( primalSpaces );
    LeadingIdentityOperator displacementExtraction( displacementSpace.GetTrueVSize(), primal.Width() );
    const plugin::RigidPlaneObstacle obstacle( MakeVector( { 0., 0. } ), MakeVector( { 0., 1. } ) );
    const mfem::Array<int> contactAttributes = ContactAttributes( 1 );
    mfem::Array<int> essentialDofs;
    plugin::BoundaryMultiplierSpace multipliers( mesh, contactAttributes );
    plugin::SemismoothRigidContactOperator contact( primal, displacementExtraction, displacementSpace, obstacle,
                                                    multipliers, essentialDofs, 1. / 13. );

    mfem::Vector unknown( contact.Width() );
    unknown = 0.;
    SetUniformDisplacement( displacementSpace, unknown, 1, -.05 );
    unknown( contact.GetBlockOffsets()[1] ) = .2;
    mfem::Vector residual;
    contact.Mult( unknown, residual );
    mfem::Vector scalarResidual( residual.GetData() + displacementSpace.GetTrueVSize(), scalarSpace.GetTrueVSize() );
    EXPECT_LE( scalarResidual.Norml2(), kAssemblyTolerance );

    mfem::Vector scalarDirection( contact.Width() );
    scalarDirection = 0.;
    for ( int i = displacementSpace.GetTrueVSize(); i < primal.Width(); i++ )
    {
        scalarDirection( i ) = 1.;
    }
    mfem::Vector tangentAction( contact.Height() );
    contact.GetGradient( unknown ).Mult( scalarDirection, tangentAction );
    EXPECT_LE( tangentAction.Norml2(), kAssemblyTolerance );
}

TEST( SemismoothContact, ActivePlaneResidualHasExpectedPrimalAndMultiplierResultants )
{
    constexpr mfem::real_t width = 2.3;
    constexpr mfem::real_t penetration = .12;
    constexpr mfem::real_t multiplierValue = .7;
    constexpr mfem::real_t gamma = 1. / 17.;
    mfem::Mesh mesh = mfem::Mesh::MakeCartesian2D( 1, 1, mfem::Element::QUADRILATERAL, true, width, 1. );
    mfem::H1_FECollection collection( 1, mesh.Dimension() );
    mfem::FiniteElementSpace space( &mesh, &collection, mesh.Dimension(), mfem::Ordering::byVDIM );
    mfem::Vector diagonal( space.GetTrueVSize() );
    diagonal = 0.;
    DiagonalNonlinearOperator primal( diagonal );
    mfem::IdentityOperator displacementExtraction( space.GetTrueVSize() );
    const plugin::RigidPlaneObstacle obstacle( MakeVector( { 0., 0. } ), MakeVector( { 0., 1. } ) );
    const mfem::Array<int> contactAttributes = ContactAttributes( 1 );
    mfem::Array<int> essentialDofs;
    plugin::BoundaryMultiplierSpace multipliers( mesh, contactAttributes );
    plugin::SemismoothRigidContactOperator contact( primal, displacementExtraction, space, obstacle, multipliers,
                                                    essentialDofs, gamma );

    mfem::Vector unknown( contact.Width() );
    unknown = 0.;
    SetUniformDisplacement( space, unknown, 1, -penetration );
    unknown( contact.GetBlockOffsets()[1] ) = multiplierValue;
    mfem::Vector residual;
    contact.Mult( unknown, residual );

    mfem::real_t horizontalResultant = 0.;
    mfem::real_t verticalResultant = 0.;
    for ( int degreeOfFreedom = 0; degreeOfFreedom < space.GetNDofs(); degreeOfFreedom++ )
    {
        horizontalResultant += residual( space.DofToVDof( degreeOfFreedom, 0 ) );
        verticalResultant += residual( space.DofToVDof( degreeOfFreedom, 1 ) );
    }
    const mfem::real_t pressure = multiplierValue + penetration / gamma;
    EXPECT_NEAR( horizontalResultant, 0., kAssemblyTolerance );
    EXPECT_NEAR( verticalResultant, -width * pressure, kAssemblyTolerance * ( 1. + width * pressure ) );
    EXPECT_NEAR( residual( contact.GetBlockOffsets()[1] ), width * penetration, kAssemblyTolerance * ( 1. + width * penetration ) );

    const plugin::SemismoothContactDiagnostics diagnostics = contact.ComputeContactDiagnostics( unknown );
    EXPECT_NEAR( diagnostics.MinimumGap, -penetration, kAssemblyTolerance );
    EXPECT_NEAR( diagnostics.MaximumPressure, pressure, kAssemblyTolerance * ( 1. + pressure ) );
    EXPECT_NEAR( diagnostics.MaximumComplementarityResidual, penetration, kAssemblyTolerance );
    EXPECT_NEAR( diagnostics.Resultant( 1 ), width * pressure, kAssemblyTolerance * ( 1. + width * pressure ) );
    EXPECT_EQ( diagnostics.ActiveQuadraturePointCount, diagnostics.QuadraturePointCount );
}

TEST( SemismoothContact, ScaleSeparatedCompliancePreservesActiveGapResidualAndDerivative )
{
    const mfem::real_t gamma = std::ldexp( mfem::real_t{ 1. }, std::numeric_limits<mfem::real_t>::digits + 1 );
    for ( const auto ordering : { mfem::Ordering::byNODES, mfem::Ordering::byVDIM } )
    {
        mfem::Mesh mesh = mfem::Mesh::MakeCartesian2D( 1, 1, mfem::Element::QUADRILATERAL, true, 1., 1. );
        mfem::H1_FECollection collection( 1, mesh.Dimension() );
        mfem::FiniteElementSpace space( &mesh, &collection, mesh.Dimension(), ordering );
        mfem::Vector diagonal( space.GetTrueVSize() );
        diagonal = 0.;
        DiagonalNonlinearOperator primal( diagonal );
        mfem::IdentityOperator extraction( space.GetTrueVSize() );
        const plugin::RigidPlaneObstacle obstacle( MakeVector( { 0., 0. } ), MakeVector( { 0., 1. } ) );
        plugin::BoundaryMultiplierSpace multipliers( mesh, ContactAttributes( 1 ) );
        mfem::Array<int> essentialDofs;
        plugin::SemismoothRigidContactOperator contact( primal, extraction, space, obstacle, multipliers, essentialDofs, gamma );
        const int multiplierDof = contact.GetBlockOffsets()[1];
        ASSERT_EQ( contact.Width() - multiplierDof, 1 );
        mfem::Vector unknown( contact.Width() );
        unknown = 0.;
        SetUniformDisplacement( space, unknown, 1, -1. );
        unknown( multiplierDof ) = 1.;
        mfem::Vector residual;
        contact.Mult( unknown, residual );
        EXPECT_NEAR( residual( multiplierDof ), 1., kAssemblyTolerance );
        const auto diagnostics = contact.ComputeContactDiagnostics( unknown );
        EXPECT_NEAR( diagnostics.MinimumGap, -1., kGeometryTolerance );
        EXPECT_EQ( diagnostics.MaximumPressure, 1. ); // The gap increment is lost in p, but not in c.
        EXPECT_NEAR( diagnostics.MaximumComplementarityResidual, 1., kAssemblyTolerance );
        EXPECT_EQ( diagnostics.ActiveQuadraturePointCount, diagnostics.QuadraturePointCount );

        mfem::Vector direction( contact.Width() );
        direction = 0.;
        SetUniformDisplacement( space, direction, 1, 1. );
        mfem::Vector tangentAction( contact.Height() );
        contact.GetGradient( unknown ).Mult( direction, tangentAction );
        const mfem::real_t step = std::cbrt( std::numeric_limits<mfem::real_t>::epsilon() );
        mfem::Vector plus( unknown ), minus( unknown ), plusResidual, minusResidual;
        plus.Add( step, direction );
        minus.Add( -step, direction );
        contact.Mult( plus, plusResidual );
        contact.Mult( minus, minusResidual );
        const mfem::real_t derivative = ( plusResidual( multiplierDof ) - minusResidual( multiplierDof ) ) / ( 2. * step );
        EXPECT_NEAR( tangentAction( multiplierDof ), -1., kAssemblyTolerance );
        EXPECT_NEAR( derivative, tangentAction( multiplierDof ), 10. * step * step );
    }
}

TEST( SemismoothContact, InactivePlaneConstrainsMultiplierWithoutDisplacementForce )
{
    constexpr mfem::real_t width = 1.7;
    constexpr mfem::real_t separation = .1;
    constexpr mfem::real_t multiplierValue = .2;
    constexpr mfem::real_t gamma = 1. / 10.;
    mfem::Mesh mesh = mfem::Mesh::MakeCartesian2D( 1, 1, mfem::Element::QUADRILATERAL, true, width, 1. );
    mfem::H1_FECollection collection( 1, mesh.Dimension() );
    mfem::FiniteElementSpace space( &mesh, &collection, mesh.Dimension() );
    mfem::Vector diagonal( space.GetTrueVSize() );
    diagonal = 0.;
    DiagonalNonlinearOperator primal( diagonal );
    mfem::IdentityOperator displacementExtraction( space.GetTrueVSize() );
    const plugin::RigidPlaneObstacle obstacle( MakeVector( { 0., 0. } ), MakeVector( { 0., 1. } ) );
    const mfem::Array<int> contactAttributes = ContactAttributes( 1 );
    mfem::Array<int> essentialDofs;
    plugin::BoundaryMultiplierSpace multipliers( mesh, contactAttributes );
    plugin::SemismoothRigidContactOperator contact( primal, displacementExtraction, space, obstacle, multipliers,
                                                    essentialDofs, gamma );

    mfem::Vector unknown( contact.Width() );
    unknown = 0.;
    SetUniformDisplacement( space, unknown, 1, separation );
    unknown( contact.GetBlockOffsets()[1] ) = multiplierValue;
    mfem::Vector residual;
    contact.Mult( unknown, residual );

    mfem::Vector displacementResidual( residual.GetData(), space.GetTrueVSize() );
    EXPECT_LE( displacementResidual.Norml2(), kAssemblyTolerance );
    EXPECT_NEAR( residual( contact.GetBlockOffsets()[1] ), -width * gamma * multiplierValue,
                 kAssemblyTolerance * ( 1. + width * gamma * multiplierValue ) );

    mfem::Vector multiplierDirection( contact.Width() );
    multiplierDirection = 0.;
    multiplierDirection( contact.GetBlockOffsets()[1] ) = 1.;
    mfem::Vector tangentAction( contact.Height() );
    contact.GetGradient( unknown ).Mult( multiplierDirection, tangentAction );
    mfem::Vector displacementTangent( tangentAction.GetData(), space.GetTrueVSize() );
    EXPECT_LE( displacementTangent.Norml2(), kAssemblyTolerance );
    EXPECT_NEAR( tangentAction( contact.GetBlockOffsets()[1] ), -width * gamma, kAssemblyTolerance );

    const auto diagnostics = contact.ComputeContactDiagnostics( unknown );
    EXPECT_NEAR( diagnostics.MaximumComplementarityResidual, gamma * multiplierValue, kAssemblyTolerance );
}

TEST( SemismoothContact, IndependentGapSamplingUsesCurrentCurvedGeometry )
{
    for ( const auto ordering : { mfem::Ordering::byNODES, mfem::Ordering::byVDIM } )
    {
        mfem::Mesh mesh = mfem::Mesh::MakeCartesian2D( 2, 2, mfem::Element::QUADRILATERAL );
        mesh.SetCurvature( 2 );
        mfem::VectorFunctionCoefficient warp( 2,
                                              []( const mfem::Vector& x, mfem::Vector& y )
                                              {
                                                  y = x;
                                                  y( 1 ) += .12 * x( 0 ) * x( 0 );
                                              } );
        mesh.Transform( warp );
        mfem::H1_FECollection collection( 4, 2 );
        mfem::FiniteElementSpace space( &mesh, &collection, 2, ordering );
        mfem::GridFunction displacement( &space );
        mfem::VectorFunctionCoefficient field( 2,
                                               []( const mfem::Vector& x, mfem::Vector& u )
                                               {
                                                   u( 0 ) = .02 * x( 0 );
                                                   u( 1 ) = .03 * x( 0 ) * x( 0 ) - .15 * x( 0 );
                                               } );
        displacement.ProjectCoefficient( field );
        plugin::RigidPlaneObstacle obstacle( MakeVector( { 0., 1. } ), MakeVector( { 0., 1. } ) );
        // Current top gap = .15*x^2 - .15*x; its minimum is at x=.5,
        // a face endpoint deliberately absent from the assembly Gauss rule.
        EXPECT_NEAR( contact_example::SampleMinimumGap( mesh, displacement, obstacle, 3, 200 ), -.0375, kGeometryTolerance );
    }
}

TEST( SemismoothContact, CoarseQuadraticCircleNeedsResolvedQuadrature )
{
    for ( const auto ordering : { mfem::Ordering::byNODES, mfem::Ordering::byVDIM } )
    {
        mfem::Mesh mesh = mfem::Mesh::MakeCartesian2D( 2, 2, mfem::Element::QUADRILATERAL );
        mfem::H1_FECollection collection( 2, 2 );
        mfem::FiniteElementSpace space( &mesh, &collection, 2, ordering );
        mfem::NonlinearForm primal( &space );
        mfem::IdentityOperator extraction( space.GetTrueVSize() );
        plugin::RigidSphereObstacle obstacle( MakeVector( { .65, 1.26 } ), .25 );
        mfem::Array<int> attributes = ContactAttributes( 3 ), essential;
        constexpr mfem::real_t gamma = 1. / 20000.;
        plugin::BoundaryMultiplierSpace multipliers( mesh, attributes );
        plugin::SemismoothRigidContactOperator contact( primal, extraction, space, obstacle, multipliers, essential, gamma );
        mfem::BlockVector unknown( contact.GetBlockOffsets() );
        unknown = 0.;
        SetUniformDisplacement( space, unknown, 1, .02 );
        mfem::GridFunction displacement( &space );
        displacement.SetFromTrueDofs( unknown.GetBlock( 0 ) );
        const auto missed = contact.ComputeContactDiagnostics( unknown );
        EXPECT_EQ( missed.QuadraturePointCount, 6 );
        EXPECT_EQ( missed.ActiveQuadraturePointCount, 0 );
        EXPECT_GT( missed.MinimumGap, 0. );
        EXPECT_NEAR( contact_example::SampleMinimumGap( mesh, displacement, obstacle, 3, 200 ), -.01, kGeometryTolerance );

        // Independent continuous resultant for rigid translation: integrate
        // -(d/gamma)*(R/sqrt(s^2+d^2)-1) over s in [-a,a], a=sqrt(R^2-d^2).
        const mfem::real_t d = .24, radius = .25;
        const mfem::real_t a = std::sqrt( radius * radius - d * d );
        const mfem::real_t exactResultant = -2. * d / gamma * ( radius * std::asinh( a / d ) - a );
        for ( const int integrationOrder : { 127, 255 } )
        {
            contact.SetIntegrationRule( &mfem::IntRules.Get( mfem::Geometry::SEGMENT, integrationOrder ) );
            const auto diagnostics = contact.ComputeContactDiagnostics( unknown );
            EXPECT_GT( diagnostics.ActiveQuadraturePointCount, 0 );
            EXPECT_NEAR( diagnostics.Resultant( 1 ), exactResultant, .01 * std::abs( exactResultant ) );
            mfem::Vector direction( contact.Width() ), action( contact.Height() ), rp, rm;
            direction = 0.;
            SetUniformDisplacement( space, direction, 1, 1. );
            // A small step stays away from the sampled active-set transitions.
            const mfem::real_t step = kSinglePrecision ? 1e-5f : 1e-7;
            contact.GetGradient( unknown ).Mult( direction, action );
            mfem::Vector plus( unknown ), minus( unknown );
            plus.Add( step, direction );
            minus.Add( -step, direction );
            contact.Mult( plus, rp );
            contact.Mult( minus, rm );
            rp -= rm;
            rp /= 2. * step;
            action -= rp;
            EXPECT_LE( action.Norml2(), kDerivativeTolerance * ( 1. + rp.Norml2() ) );
        }
    }
}

namespace
{
void CheckHighOrderBoundaryTraceAgainstVolumeOracle( const int multiplierOrder )
{
    for ( const int dimension : { 2, 3 } )
        for ( const bool simplex : { false, true } )
            for ( const auto ordering : { mfem::Ordering::byNODES, mfem::Ordering::byVDIM } )
            {
                SCOPED_TRACE( ::testing::Message() << dimension << " simplex=" << simplex << " ordering=" << ordering );
                mfem::Mesh mesh =
                    dimension == 2
                        ? mfem::Mesh::MakeCartesian2D( 1, 1, simplex ? mfem::Element::TRIANGLE : mfem::Element::QUADRILATERAL )
                        : mfem::Mesh::MakeCartesian3D( 1, 1, 1, simplex ? mfem::Element::TETRAHEDRON : mfem::Element::HEXAHEDRON );
                mesh.SetCurvature( 2 );
                mfem::VectorFunctionCoefficient warp( dimension,
                                                      [dimension]( const mfem::Vector& x, mfem::Vector& y )
                                                      {
                                                          y = x;
                                                          y( dimension - 1 ) += .12 * x( 0 ) * x( 0 );
                                                      } );
                mesh.Transform( warp );
                mfem::H1_FECollection collection( 4, dimension );
                mfem::FiniteElementSpace space( &mesh, &collection, dimension, ordering );
                mfem::NonlinearForm primal( &space );
                mfem::IdentityOperator extraction( space.GetTrueVSize() );
                mfem::Vector center( dimension );
                center = -2.;
                plugin::RigidSphereObstacle obstacle( center, 4. );
                mfem::Array<int> attributes( mesh.bdr_attributes );
                mfem::Array<int> essential;
                constexpr mfem::real_t gamma = 1. / 17.;
                // Include a nonsymmetric rule: preserving just the order is insufficient
                // when the boundary element and mesh face have different orientations.
                mfem::IntegrationRule customRule( 2 );
                customRule.IntPoint( 0 ).Set2w( .13, .21, .37 );
                customRule.IntPoint( 1 ).Set2w( .61, .12, .63 );
                plugin::BoundaryMultiplierSpace multipliers(
                    mesh, attributes,
                    multiplierOrder == 0 ? nullptr : std::make_unique<mfem::L2_FECollection>( multiplierOrder, dimension - 1 ) );
                plugin::SemismoothRigidContactOperator contact( primal, extraction, space, obstacle, multipliers,
                                                                essential, gamma, 1., 1., 0. );
                mfem::Vector unknown( contact.Width() ), direction( contact.Width() );
                for ( int i = 0; i < unknown.Size(); ++i )
                {
                    unknown( i ) = .003 * std::sin( i + 1. );
                    direction( i ) = std::cos( i + .7 );
                }
                direction /= direction.Norml2();
                const int offset = space.GetTrueVSize();
                for ( int i = offset; i < unknown.Size(); ++i )
                {
                    unknown( i ) = ( i % 2 == 0 ) ? 100. : -100.;
                }
                if ( multiplierOrder > 0 )
                {
                    // Project physical nodal values, not a guessed coefficient ordering.
                    // This affine field is exactly representable even after the warp,
                    // which changes only the last physical coordinate.
                    mfem::FunctionCoefficient affine( [dimension]( const mfem::Vector& x )
                                                      { return .4 * x( 0 ) + ( dimension == 3 ? .7 * x( 1 ) : 0. ); } );
                    mfem::GridFunction lambda( &multipliers.GetSpace() );
                    lambda.ProjectCoefficient( affine );
                    for ( int e = 0; e < multipliers.GetMesh().GetNE(); ++e )
                    {
                        mfem::Array<int> dofs;
                        multipliers.GetSpace().GetElementDofs( e, dofs );
                        for ( const int dof : dofs )
                        {
                            unknown( offset + dof ) = lambda( dof ) + ( e % 2 == 0 ? 100. : -100. );
                        }
                    }
                }
                // Reuse the same operator through override and default restoration.
                for ( const bool custom : { false, true, false } )
                {
                    SCOPED_TRACE( ::testing::Message() << "custom=" << custom );
                    contact.SetIntegrationRule( custom ? &customRule : nullptr );
                    mfem::Vector expected( contact.Height() ), expectedAction( contact.Height() );
                    expected = 0.;
                    expectedAction = 0.;
                    int expectedPointCount = 0;
                    const auto& submesh = contact.GetContactSubMesh();
                    for ( int e = 0; e < submesh.GetNE(); ++e )
                    {
                        const int be = submesh.GetParentElementIDMap()[e];
                        auto* transform = mesh.GetBdrFaceTransformations( be );
                        const auto& fe = *space.GetFE( transform->Elem1No );
                        mfem::Array<int> vdofs, ldofs;
                        space.GetElementVDofs( transform->Elem1No, vdofs );
                        contact.GetMultiplierSpace().GetElementDofs( e, ldofs );
                        mfem::Vector lambdaBlock( unknown.GetData() + offset, unknown.Size() - offset );
                        mfem::Vector lambdaDirectionBlock( direction.GetData() + offset, direction.Size() - offset );
                        mfem::Vector localLambda, localLambdaDirection, multiplierShape( ldofs.Size() );
                        lambdaBlock.GetSubVector( ldofs, localLambda );
                        lambdaDirectionBlock.GetSubVector( ldofs, localLambdaDirection );
                        mfem::InverseElementTransformation inverse( multipliers.GetMesh().GetElementTransformation( e ) );
                        mfem::Vector localU, localDirection, shape( fe.GetDof() ), position( dimension );
                        unknown.GetSubVector( vdofs, localU );
                        direction.GetSubVector( vdofs, localDirection );
                        const int order = transform->Elem1->OrderW() + 2 * fe.GetOrder() +
                                          ( fe.Space() == mfem::FunctionSpace::Pk ? 1 : 0 );
                        const auto& rule = custom ? customRule : mfem::IntRules.Get( transform->GetGeometryType(), order );
                        expectedPointCount += rule.GetNPoints();
                        for ( int q = 0; q < rule.GetNPoints(); ++q )
                        {
                            const auto& ip = rule.IntPoint( q );
                            transform->SetAllIntPoints( &ip );
                            fe.CalcShape( transform->GetElement1IntPoint(), shape );
                            transform->Transform( ip, position );
                            if ( multiplierOrder == 0 )
                            {
                                multiplierShape = 1.;
                            }
                            else
                            {
                                // Independent physical inversion into the submesh: do not
                                // reuse BoundaryMultiplierSpace::CalcShape or its orientation map.
                                mfem::IntegrationPoint multiplierPoint;
                                ASSERT_EQ( inverse.Transform( position, multiplierPoint ), mfem::InverseElementTransformation::Inside );
                                multipliers.GetSpace().GetFE( e )->CalcShape( multiplierPoint, multiplierShape );
                            }
                            mfem::Vector dx( dimension );
                            dx = 0.;
                            for ( int c = 0; c < dimension; ++c )
                                for ( int j = 0; j < fe.GetDof(); ++j )
                                {
                                    position( c ) += shape( j ) * localU( c * fe.GetDof() + j );
                                    dx( c ) += shape( j ) * localDirection( c * fe.GetDof() + j );
                                }
                            plugin::SignedDistanceEvaluation evaluation( dimension );
                            obstacle.Evaluate( position, evaluation );
                            const auto lambda = multiplierShape * localLambda;
                            const auto dlambda = multiplierShape * localLambdaDirection;
                            const auto pressure = std::max( mfem::real_t{ 0. }, lambda - evaluation.Gap / gamma );
                            const bool active = lambda - evaluation.Gap / gamma >= 0.;
                            const auto dp = active ? dlambda - ( evaluation.Gradient * dx ) / gamma : 0.;
                            const auto weight = ip.weight * transform->Weight();
                            for ( int j = 0; j < ldofs.Size(); ++j )
                            {
                                const int encoded = ldofs[j];
                                const int dof = offset + ( encoded < 0 ? -1 - encoded : encoded );
                                const auto w = weight * multiplierShape( j ) * ( encoded < 0 ? -1. : 1. );
                                expected( dof ) += w * gamma * ( pressure - lambda );
                                expectedAction( dof ) += w * gamma * ( dp - dlambda );
                            }
                            mfem::Vector dn( dimension );
                            evaluation.Hessian.Mult( dx, dn );
                            for ( int c = 0; c < dimension; ++c )
                                for ( int j = 0; j < fe.GetDof(); ++j )
                                {
                                    const int encoded = vdofs[c * fe.GetDof() + j];
                                    const int dof = encoded < 0 ? -1 - encoded : encoded;
                                    const auto w = weight * shape( j ) * ( encoded < 0 ? -1. : 1. );
                                    expected( dof ) -= w * pressure * evaluation.Gradient( c );
                                    expectedAction( dof ) -= w * ( dp * evaluation.Gradient( c ) + pressure * dn( c ) );
                                }
                        }
                    }
                    mfem::Vector residual, action( contact.Height() );
                    contact.Mult( unknown, residual );
                    auto& jacobian = dynamic_cast<mfem::BlockOperator&>( contact.GetGradient( unknown ) );
                    jacobian.Mult( direction, action );
                    ExpectVectorNear( residual, expected, kAssemblyTolerance );
                    ExpectVectorNear( action, expectedAction, kAssemblyTolerance );
                    EXPECT_EQ( contact.ComputeContactDiagnostics( unknown ).QuadraturePointCount, expectedPointCount );
                    mfem::Vector plus( unknown ), minus( unknown ), rp, rm;
                    plus.Add( kFiniteDifferenceStep, direction );
                    minus.Add( -kFiniteDifferenceStep, direction );
                    contact.Mult( plus, rp );
                    contact.Mult( minus, rm );
                    rp -= rm;
                    rp /= 2. * kFiniteDifferenceStep;
                    ExpectVectorNear( action, rp, kDerivativeTolerance );

                    // The public Jacobian wraps its sparse blocks in product operators.
                    // Check their trace support through actions in both directions.
                    mfem::Array<int> traceMarker( offset );
                    traceMarker = 0;
                    for ( int e = 0; e < submesh.GetNE(); ++e )
                    {
                        mfem::Array<int> trace, ldofs;
                        space.GetBdrElementVDofs( submesh.GetParentElementIDMap()[e], trace );
                        for ( const int dof : trace )
                        {
                            traceMarker[dof < 0 ? -1 - dof : dof] = 1;
                        }
                        contact.GetMultiplierSpace().GetElementDofs( e, ldofs );
                        mfem::Vector unit( unknown.Size() - offset ), column( offset ), row( offset );
                        for ( const int lambdaDof : ldofs )
                        {
                            unit = 0.;
                            unit( lambdaDof ) = 1.;
                            jacobian.GetBlock( 0, 1 ).Mult( unit, column );
                            jacobian.GetBlock( 1, 0 ).MultTranspose( unit, row );
                            for ( int i = 0; i < offset; ++i )
                            {
                                if ( trace.Find( i ) < 0 )
                                {
                                    EXPECT_EQ( column( i ), 0. );
                                    EXPECT_EQ( row( i ), 0. );
                                }
                            }
                        }
                    }
                    mfem::Vector interiorDirection( unknown.Size() );
                    interiorDirection = 0.;
                    int interiorDofs = 0;
                    for ( int i = 0; i < offset; ++i )
                    {
                        if ( traceMarker[i] == 0 )
                        {
                            interiorDirection( i ) = direction( i );
                            ++interiorDofs;
                        }
                    }
                    ASSERT_GT( interiorDofs, 0 );
                    jacobian.Mult( interiorDirection, action );
                    EXPECT_EQ( action.Norml2(), 0. );
                    plus = unknown;
                    plus += interiorDirection;
                    contact.Mult( plus, rp );
                    ExpectVectorNear( rp, residual, kAssemblyTolerance );
                }
            }
}
} // namespace

TEST( SemismoothContact, HighOrderBoundaryTraceMatchesVolumeOracle )
{
    CheckHighOrderBoundaryTraceAgainstVolumeOracle( 0 );
}

TEST( SemismoothContact, BoundaryP1Q1HighOrderTraceMatchesVolumeOracle )
{
    CheckHighOrderBoundaryTraceAgainstVolumeOracle( 1 );
}

TEST( PenaltyContact, BoundaryAssemblyMatchesVolumeShapeOracle )
{
    for ( const int dimension : { 2, 3 } )
        for ( const bool simplex : { false, true } )
            for ( const auto ordering : { mfem::Ordering::byNODES, mfem::Ordering::byVDIM } )
            {
                SCOPED_TRACE( ::testing::Message() << dimension << " simplex=" << simplex << " ordering=" << ordering );
                mfem::Mesh mesh =
                    dimension == 2
                        ? mfem::Mesh::MakeCartesian2D( 1, 1, simplex ? mfem::Element::TRIANGLE : mfem::Element::QUADRILATERAL )
                        : mfem::Mesh::MakeCartesian3D( 1, 1, 1, simplex ? mfem::Element::TETRAHEDRON : mfem::Element::HEXAHEDRON );
                mesh.SetCurvature( 2 );
                mfem::VectorFunctionCoefficient warp( dimension,
                                                      [dimension]( const mfem::Vector& x, mfem::Vector& y )
                                                      {
                                                          y = x;
                                                          y( dimension - 1 ) += .12 * x( 0 ) * x( 0 );
                                                      } );
                mesh.Transform( warp );
                mfem::H1_FECollection collection( 4, dimension );
                mfem::FiniteElementSpace space( &mesh, &collection, dimension, ordering );
                mfem::Vector center( dimension );
                center = -2.;
                plugin::RigidSphereObstacle obstacle( center, 6. );
                mfem::IntegrationRule customRule( 2 );
                customRule.IntPoint( 0 ).Set2w( .13, .21, .37 );
                customRule.IntPoint( 1 ).Set2w( .61, .12, .63 );
                constexpr mfem::real_t penalty = 17.;
                mfem::NonlinearForm boundary( &space );
                auto* traceIntegrator = new plugin::FrictionlessPenaltyContactIntegrator( obstacle, penalty );
                boundary.AddBoundaryIntegrator( traceIntegrator );
                mfem::Vector u( space.GetTrueVSize() ), direction( u.Size() );
                for ( int i = 0; i < u.Size(); ++i )
                {
                    u( i ) = .003 * std::sin( i + 1. );
                    direction( i ) = std::cos( i + .7 );
                }
                direction /= direction.Norml2();
                mfem::Array<int> traceMarker( u.Size() );
                traceMarker = 0;
                for ( int e = 0; e < mesh.GetNBE(); ++e )
                {
                    mfem::Array<int> dofs;
                    space.GetBdrElementVDofs( e, dofs );
                    for ( const int dof : dofs )
                    {
                        traceMarker[dof < 0 ? -1 - dof : dof] = 1;
                    }
                }
                for ( const bool custom : { false, true } )
                {
                    traceIntegrator->SetIntRule( custom ? &customRule : nullptr );
                    mfem::Vector residual( u.Size() ), reference( u.Size() ), action( u.Size() ), expected( u.Size() );
                    boundary.Mult( u, residual );
                    auto& matrix = dynamic_cast<mfem::SparseMatrix&>( boundary.GetGradient( u ) );
                    matrix.Mult( direction, action );
                    // Inspect stored CSR entries, not just their numerical values.
                    int interior = 0;
                    for ( int row = 0; row < u.Size(); ++row )
                    {
                        if ( !traceMarker[row] )
                        {
                            ++interior;
                            EXPECT_EQ( matrix.GetI()[row], matrix.GetI()[row + 1] );
                        }
                        for ( int k = matrix.GetI()[row]; k < matrix.GetI()[row + 1]; ++k )
                        {
                            EXPECT_TRUE( traceMarker[matrix.GetJ()[k]] );
                        }
                    }
                    EXPECT_GT( interior, 0 );
                    mfem::Vector plus( u ), minus( u ), rp( u.Size() ), rm( u.Size() );
                    plus.Add( kFiniteDifferenceStep, direction );
                    minus.Add( -kFiniteDifferenceStep, direction );
                    boundary.Mult( plus, rp );
                    boundary.Mult( minus, rm );
                    rp -= rm;
                    rp /= 2. * kFiniteDifferenceStep;
                    ExpectVectorNear( action, rp, kDerivativeTolerance );
                    const auto energyDerivative =
                        ( boundary.GetEnergy( plus ) - boundary.GetEnergy( minus ) ) / ( 2. * kFiniteDifferenceStep );
                    EXPECT_NEAR( energyDerivative, residual * direction,
                                 kDerivativeTolerance * ( 1. + std::abs( residual * direction ) ) );

                    // Independent volume-shape oracle at the original mesh-face points:
                    // no boundary shape evaluation or production assembly helpers.
                    mfem::real_t energy = 0.;
                    reference = 0.;
                    expected = 0.;
                    for ( int e = 0; e < mesh.GetNBE(); ++e )
                    {
                        auto* transform = mesh.GetBdrFaceTransformations( e );
                        const auto& fe = *space.GetFE( transform->Elem1No );
                        mfem::Array<int> dofs;
                        space.GetElementVDofs( transform->Elem1No, dofs );
                        mfem::Vector local, localDirection, shape( fe.GetDof() ), position( dimension );
                        u.GetSubVector( dofs, local );
                        direction.GetSubVector( dofs, localDirection );
                        const int order = transform->Elem1->OrderW() + 2 * fe.GetOrder() +
                                          ( fe.Space() == mfem::FunctionSpace::Pk ? 1 : 0 );
                        const auto& rule = custom ? customRule : mfem::IntRules.Get( transform->GetGeometryType(), order );
                        for ( int q = 0; q < rule.GetNPoints(); ++q )
                        {
                            const auto& ip = rule.IntPoint( q );
                            transform->SetAllIntPoints( &ip );
                            fe.CalcShape( transform->GetElement1IntPoint(), shape );
                            transform->Transform( ip, position );
                            mfem::Vector dx( dimension );
                            dx = 0.;
                            for ( int c = 0; c < dimension; ++c )
                                for ( int j = 0; j < fe.GetDof(); ++j )
                                {
                                    position( c ) += shape( j ) * local( c * fe.GetDof() + j );
                                    dx( c ) += shape( j ) * localDirection( c * fe.GetDof() + j );
                                }
                            plugin::SignedDistanceEvaluation evaluation( dimension );
                            obstacle.Evaluate( position, evaluation );
                            const auto gap = std::min( mfem::real_t{ 0. }, evaluation.Gap );
                            const auto weight = penalty * ip.weight * transform->Weight();
                            energy += .5 * gap * gap * weight;
                            if ( evaluation.Gap > 0. )
                            {
                                continue;
                            }
                            const auto dg = evaluation.Gradient * dx;
                            mfem::Vector dn( dimension );
                            evaluation.Hessian.Mult( dx, dn );
                            for ( int c = 0; c < dimension; ++c )
                                for ( int j = 0; j < fe.GetDof(); ++j )
                                {
                                    const int encoded = dofs[c * fe.GetDof() + j];
                                    const int dof = encoded < 0 ? -1 - encoded : encoded;
                                    const auto w = weight * shape( j ) * ( encoded < 0 ? -1. : 1. );
                                    reference( dof ) += w * gap * evaluation.Gradient( c );
                                    expected( dof ) += w * ( dg * evaluation.Gradient( c ) + gap * dn( c ) );
                                }
                        }
                    }
                    EXPECT_NEAR( boundary.GetEnergy( u ), energy, kAssemblyTolerance * ( 1. + energy ) );
                    ExpectVectorNear( residual, reference, kAssemblyTolerance );
                    ExpectVectorNear( action, expected, kAssemblyTolerance );
                }
                mfem::Array<int> essentialMarker( mesh.bdr_attributes.Max() ), essential;
                essentialMarker = 0;
                essentialMarker[0] = 1;
                space.GetEssentialTrueDofs( essentialMarker, essential );
                // Independently apply row/column elimination to the verified
                // unconstrained operator, including identity on essential rows.
                mfem::Vector actual( u.Size() ), expectedResidual( u.Size() ), expectedAction( u.Size() );
                mfem::Vector freeDirection( direction );
                for ( const int dof : essential )
                {
                    freeDirection( dof ) = 0.;
                }
                boundary.Mult( u, expectedResidual );
                boundary.GetGradient( u ).Mult( freeDirection, expectedAction );
                for ( const int dof : essential )
                {
                    expectedResidual( dof ) = 0.;
                    expectedAction( dof ) = direction( dof );
                }
                boundary.SetEssentialTrueDofs( essential );
                boundary.Mult( u, actual );
                ExpectVectorNear( actual, expectedResidual, kAssemblyTolerance );
                boundary.GetGradient( u ).Mult( direction, actual );
                ExpectVectorNear( actual, expectedAction, kAssemblyTolerance );
            }
}

TEST( PenaltyContact, DISABLED_BoundaryAssemblyBenchmark )
{
    mfem::Mesh mesh = mfem::Mesh::MakeCartesian3D( 3, 3, 3, mfem::Element::HEXAHEDRON );
    mfem::H1_FECollection collection( 4, 3 );
    mfem::FiniteElementSpace space( &mesh, &collection, 3, mfem::Ordering::byVDIM );
    plugin::RigidSphereObstacle obstacle( MakeVector( { -2., -2., -2. } ), 6. );
    mfem::NonlinearForm boundary( &space );
    boundary.AddBoundaryIntegrator( new plugin::FrictionlessPenaltyContactIntegrator( obstacle, 17. ) );
    mfem::Vector u( space.GetTrueVSize() ), residual( u.Size() );
    u = 0.;
    for ( int sample = -1; sample < 5; ++sample )
    {
        const auto start = std::chrono::steady_clock::now();
        for ( int repeat = 0; repeat < 20; ++repeat )
        {
            boundary.Mult( u, residual );
            boundary.GetGradient( u );
        }
        const auto seconds = std::chrono::duration<double>( std::chrono::steady_clock::now() - start ).count();
        if ( sample >= 0 )
        {
            std::cout << "penalty boundary sample=" << sample << " ms/pair=" << seconds * 50.
                      << " residual_norm=" << residual.Norml2()
                      << " nnz=" << dynamic_cast<mfem::SparseMatrix&>( boundary.GetGradient( u ) ).NumNonZeroElems() << '\n';
        }
    }
}

TEST( SemismoothContact, DISABLED_TraceAssemblyBenchmark )
{
    mfem::Mesh mesh = mfem::Mesh::MakeCartesian3D( 3, 3, 3, mfem::Element::HEXAHEDRON );
    mfem::H1_FECollection collection( 4, 3 );
    mfem::FiniteElementSpace space( &mesh, &collection, 3, mfem::Ordering::byVDIM );
    mfem::NonlinearForm primal( &space );
    mfem::IdentityOperator extraction( space.GetTrueVSize() );
    plugin::RigidSphereObstacle obstacle( MakeVector( { -2., -2., -2. } ), 6. );
    mfem::Array<int> attributes( mesh.bdr_attributes ), essential;
    plugin::BoundaryMultiplierSpace multipliers( mesh, attributes );
    plugin::SemismoothRigidContactOperator contact( primal, extraction, space, obstacle, multipliers, essential, 1. / 17. );
    mfem::Vector unknown( contact.Width() ), residual;
    unknown = 0.;
    for ( int i = space.GetTrueVSize(); i < unknown.Size(); ++i )
    {
        unknown( i ) = 100.;
    }
    for ( int sample = -1; sample < 5; ++sample )
    {
        const auto start = std::chrono::steady_clock::now();
        for ( int repeat = 0; repeat < 20; ++repeat )
        {
            contact.Mult( unknown, residual );
            contact.GetGradient( unknown );
        }
        const auto seconds = std::chrono::duration<double>( std::chrono::steady_clock::now() - start ).count();
        if ( sample >= 0 )
        {
            std::cout << "assembly sample=" << sample << " ms/pair=" << seconds * 50.
                      << " residual_norm=" << residual.Norml2() << '\n';
        }
    }
}

TEST( SemismoothContact, CurvedMixedJacobianMatchesDirectionalDifferenceWithExtraPrimalField )
{
    mfem::Mesh mesh = mfem::Mesh::MakeCartesian2D( 1, 1, mfem::Element::QUADRILATERAL, true, 1., 1. );
    mfem::H1_FECollection collection( 2, mesh.Dimension() );
    mfem::FiniteElementSpace space( &mesh, &collection, mesh.Dimension(), mfem::Ordering::byNODES );
    constexpr int extraFieldSize = 3;
    const int primalSize = space.GetTrueVSize() + extraFieldSize;
    mfem::Vector diagonal( primalSize );
    for ( int i = 0; i < diagonal.Size(); i++ )
    {
        diagonal( i ) = .3 + .01 * i;
    }
    DiagonalNonlinearOperator primal( diagonal );
    LeadingIdentityOperator displacementExtraction( space.GetTrueVSize(), primalSize );
    const plugin::RigidSphereObstacle obstacle( MakeVector( { .5, -.4 } ), 1.2 );
    const mfem::Array<int> contactAttributes = ContactAttributes( 1 );
    mfem::Array<int> essentialDofs;
    plugin::BoundaryMultiplierSpace multipliers( mesh, contactAttributes );
    plugin::SemismoothRigidContactOperator contact( primal, displacementExtraction, space, obstacle, multipliers,
                                                    essentialDofs, 1. / 31., .17, 4.2 );

    mfem::Vector unknown( contact.Width() );
    mfem::Vector direction( contact.Width() );
    for ( int i = 0; i < unknown.Size(); i++ )
    {
        unknown( i ) = .004 * static_cast<mfem::real_t>( ( i * 3 ) % 7 - 3 );
        direction( i ) = static_cast<mfem::real_t>( ( i * 7 ) % 13 - 6 );
    }
    unknown( contact.GetBlockOffsets()[1] ) = .4;
    direction /= direction.Norml2();

    mfem::Vector analyticalDerivative( contact.Height() );
    contact.GetGradient( unknown ).Mult( direction, analyticalDerivative );
    mfem::Vector plus( unknown );
    mfem::Vector minus( unknown );
    plus.Add( kFiniteDifferenceStep, direction );
    minus.Add( -kFiniteDifferenceStep, direction );
    mfem::Vector plusResidual;
    mfem::Vector minusResidual;
    contact.Mult( plus, plusResidual );
    contact.Mult( minus, minusResidual );
    mfem::Vector numericalDerivative( plusResidual );
    numericalDerivative -= minusResidual;
    numericalDerivative /= 2. * kFiniteDifferenceStep;

    analyticalDerivative -= numericalDerivative;
    EXPECT_LE( analyticalDerivative.Norml2(), kDerivativeTolerance * ( 1. + numericalDerivative.Norml2() ) );
}

TEST( SemismoothContact, SupportsThreeDimensionalPlaneContact )
{
    constexpr mfem::real_t width = 1.3;
    constexpr mfem::real_t depth = .7;
    constexpr mfem::real_t penetration = .09;
    constexpr mfem::real_t multiplierValue = .4;
    constexpr mfem::real_t gamma = 1. / 19.;
    mfem::Mesh mesh = mfem::Mesh::MakeCartesian3D( 1, 1, 1, mfem::Element::HEXAHEDRON, width, depth, 1. );
    mfem::H1_FECollection collection( 1, mesh.Dimension() );
    mfem::FiniteElementSpace space( &mesh, &collection, mesh.Dimension(), mfem::Ordering::byNODES );
    mfem::Vector diagonal( space.GetTrueVSize() );
    diagonal = 0.;
    DiagonalNonlinearOperator primal( diagonal );
    mfem::IdentityOperator displacementExtraction( space.GetTrueVSize() );
    const plugin::RigidPlaneObstacle obstacle( MakeVector( { 0., 0., 0. } ), MakeVector( { 0., 0., 1. } ) );
    const mfem::Array<int> contactAttributes = ContactAttributes( 1 );
    mfem::Array<int> essentialDofs;
    plugin::BoundaryMultiplierSpace multipliers( mesh, contactAttributes );
    plugin::SemismoothRigidContactOperator contact( primal, displacementExtraction, space, obstacle, multipliers,
                                                    essentialDofs, gamma );

    mfem::Vector unknown( contact.Width() );
    unknown = 0.;
    SetUniformDisplacement( space, unknown, 2, -penetration );
    unknown( contact.GetBlockOffsets()[1] ) = multiplierValue;
    mfem::Vector residual;
    contact.Mult( unknown, residual );

    std::array<mfem::real_t, 3> resultant{ 0., 0., 0. };
    for ( int component = 0; component < mesh.Dimension(); component++ )
    {
        for ( int degreeOfFreedom = 0; degreeOfFreedom < space.GetNDofs(); degreeOfFreedom++ )
        {
            resultant[component] += residual( space.DofToVDof( degreeOfFreedom, component ) );
        }
    }
    const mfem::real_t area = width * depth;
    EXPECT_NEAR( resultant[0], 0., kAssemblyTolerance );
    EXPECT_NEAR( resultant[1], 0., kAssemblyTolerance );
    EXPECT_NEAR( resultant[2], -area * ( multiplierValue + penetration / gamma ),
                 kAssemblyTolerance * ( 1. + area * ( multiplierValue + penetration / gamma ) ) );
    EXPECT_NEAR( residual( contact.GetBlockOffsets()[1] ), area * penetration, kAssemblyTolerance * ( 1. + area * penetration ) );

    mfem::Vector direction( contact.Width() );
    for ( int i = 0; i < direction.Size(); i++ )
    {
        direction( i ) = static_cast<mfem::real_t>( ( i * 5 ) % 11 - 5 );
    }
    direction /= direction.Norml2();
    mfem::Vector analyticalDerivative( contact.Height() );
    contact.GetGradient( unknown ).Mult( direction, analyticalDerivative );
    mfem::Vector plus( unknown );
    mfem::Vector minus( unknown );
    plus.Add( kFiniteDifferenceStep, direction );
    minus.Add( -kFiniteDifferenceStep, direction );
    mfem::Vector plusResidual;
    mfem::Vector minusResidual;
    contact.Mult( plus, plusResidual );
    contact.Mult( minus, minusResidual );
    mfem::Vector numericalDerivative( plusResidual );
    numericalDerivative -= minusResidual;
    numericalDerivative /= 2. * kFiniteDifferenceStep;
    analyticalDerivative -= numericalDerivative;
    EXPECT_LE( analyticalDerivative.Norml2(), kDerivativeTolerance * ( 1. + numericalDerivative.Norml2() ) );
}

TEST( SemismoothContact, EssentialDisplacementDofsAreRemovedFromContactVariations )
{
    constexpr mfem::real_t penetration = .08;
    mfem::Mesh mesh = mfem::Mesh::MakeCartesian2D( 1, 1, mfem::Element::QUADRILATERAL );
    mfem::H1_FECollection collection( 1, mesh.Dimension() );
    mfem::FiniteElementSpace space( &mesh, &collection, mesh.Dimension() );
    mfem::Vector diagonal( space.GetTrueVSize() );
    diagonal = 0.;
    DiagonalNonlinearOperator primal( diagonal );
    mfem::IdentityOperator displacementExtraction( space.GetTrueVSize() );
    const plugin::RigidPlaneObstacle obstacle( MakeVector( { 0., 0. } ), MakeVector( { 0., 1. } ) );
    const mfem::Array<int> contactAttributes = ContactAttributes( 1 );
    mfem::Array<int> essentialDofs( 1 );
    essentialDofs[0] = space.DofToVDof( 0, 1 );
    plugin::BoundaryMultiplierSpace multipliers( mesh, contactAttributes );
    plugin::SemismoothRigidContactOperator contact( primal, displacementExtraction, space, obstacle, multipliers,
                                                    essentialDofs, 1. / 23. );

    mfem::Vector unknown( contact.Width() );
    unknown = 0.;
    SetUniformDisplacement( space, unknown, 1, -penetration );
    unknown( contact.GetBlockOffsets()[1] ) = .3;
    mfem::Vector residual;
    contact.Mult( unknown, residual );
    EXPECT_NEAR( residual( essentialDofs[0] ), 0., kAssemblyTolerance );
    EXPECT_NEAR( residual( contact.GetBlockOffsets()[1] ), penetration, kAssemblyTolerance );

    mfem::Vector direction( contact.Width() );
    direction = 0.;
    direction( essentialDofs[0] ) = 1.;
    mfem::Vector tangentAction( contact.Height() );
    contact.GetGradient( unknown ).Mult( direction, tangentAction );
    EXPECT_LE( tangentAction.Norml2(), kAssemblyTolerance );
}

TEST( SemismoothContact, RejectsFullyActiveFaceWithoutFreeNormalVariation )
{
    GTEST_FLAG_SET( death_test_style, "threadsafe" );
    mfem::Mesh mesh = mfem::Mesh::MakeCartesian2D( 1, 1, mfem::Element::QUADRILATERAL );
    mfem::H1_FECollection collection( 1, mesh.Dimension() );
    mfem::FiniteElementSpace space( &mesh, &collection, mesh.Dimension() );
    mfem::Vector diagonal( space.GetTrueVSize() );
    diagonal = 0.;
    DiagonalNonlinearOperator primal( diagonal );
    mfem::IdentityOperator displacementExtraction( space.GetTrueVSize() );
    const plugin::RigidPlaneObstacle obstacle( MakeVector( { 0., 0. } ), MakeVector( { 0., 1. } ) );
    const mfem::Array<int> contactAttributes = ContactAttributes( 1 );
    mfem::Array<int> essentialDofs( space.GetNDofs() );
    for ( int i = 0; i < space.GetNDofs(); i++ )
    {
        essentialDofs[i] = space.DofToVDof( i, 1 );
    }
    plugin::BoundaryMultiplierSpace multipliers( mesh, contactAttributes );
    plugin::SemismoothRigidContactOperator contact( primal, displacementExtraction, space, obstacle, multipliers,
                                                    essentialDofs, 1. / 23. );
    mfem::Vector unknown( contact.Width() );
    unknown = 0.;
    SetUniformDisplacement( space, unknown, 1, -.08 );
    unknown( contact.GetBlockOffsets()[1] ) = .3;

    EXPECT_DEATH( (void)contact.GetGradient( unknown ), "needs a free normal displacement variation" );
}

TEST( SemismoothContact, RejectsEssentialDofsThatDifferFromPrimalForm )
{
    GTEST_FLAG_SET( death_test_style, "threadsafe" );
    mfem::Mesh mesh = mfem::Mesh::MakeCartesian2D( 1, 1, mfem::Element::QUADRILATERAL );
    mfem::H1_FECollection collection( 1, mesh.Dimension() );
    mfem::FiniteElementSpace space( &mesh, &collection, mesh.Dimension() );
    mfem::NonlinearForm primal( &space );
    mfem::Array<int> primalEssentialDofs( 1 );
    primalEssentialDofs[0] = 0;
    primal.SetEssentialTrueDofs( primalEssentialDofs );
    mfem::IdentityOperator displacementExtraction( space.GetTrueVSize() );
    const plugin::RigidPlaneObstacle obstacle( MakeVector( { 0., 0. } ), MakeVector( { 0., 1. } ) );
    const mfem::Array<int> contactAttributes = ContactAttributes( 1 );
    mfem::Array<int> contactEssentialDofs( 1 );
    contactEssentialDofs[0] = 1;
    plugin::BoundaryMultiplierSpace multipliers( mesh, contactAttributes );

    EXPECT_DEATH( (void)plugin::SemismoothRigidContactOperator( primal, displacementExtraction, space, obstacle,
                                                                multipliers, contactEssentialDofs, 1. / 23. ),
                  "essential displacement true DOFs must match" );
}

TEST( SemismoothContact, EqualityUsesTheActiveOneSidedLinearization )
{
    constexpr mfem::real_t separation = .1;
    constexpr mfem::real_t gamma = 1. / 10.;
    mfem::Mesh mesh = mfem::Mesh::MakeCartesian2D( 1, 1, mfem::Element::QUADRILATERAL );
    mfem::H1_FECollection collection( 1, mesh.Dimension() );
    mfem::FiniteElementSpace space( &mesh, &collection, mesh.Dimension(), mfem::Ordering::byVDIM );
    mfem::Vector diagonal( space.GetTrueVSize() );
    diagonal = 0.;
    DiagonalNonlinearOperator primal( diagonal );
    mfem::IdentityOperator displacementExtraction( space.GetTrueVSize() );
    const plugin::RigidPlaneObstacle obstacle( MakeVector( { 0., 0. } ), MakeVector( { 0., 1. } ) );
    const mfem::Array<int> contactAttributes = ContactAttributes( 1 );
    mfem::Array<int> essentialDofs;
    plugin::BoundaryMultiplierSpace multipliers( mesh, contactAttributes );
    plugin::SemismoothRigidContactOperator contact( primal, displacementExtraction, space, obstacle, multipliers,
                                                    essentialDofs, gamma );

    mfem::Vector unknown( contact.Width() );
    unknown = 0.;
    SetUniformDisplacement( space, unknown, 1, separation );
    unknown( contact.GetBlockOffsets()[1] ) = separation / gamma;
    mfem::Vector direction( contact.Width() );
    direction = 0.;
    SetUniformDisplacement( space, direction, 1, -1. );
    mfem::Vector analyticalDerivative( contact.Height() );
    contact.GetGradient( unknown ).Mult( direction, analyticalDerivative );

    mfem::Vector baseResidual;
    contact.Mult( unknown, baseResidual );
    mfem::Vector activeUnknown( unknown );
    activeUnknown.Add( kFiniteDifferenceStep, direction );
    mfem::Vector activeResidual;
    contact.Mult( activeUnknown, activeResidual );
    activeResidual -= baseResidual;
    activeResidual /= kFiniteDifferenceStep;

    analyticalDerivative -= activeResidual;
    EXPECT_LE( analyticalDerivative.Norml2(), kDerivativeTolerance * ( 1. + activeResidual.Norml2() ) );
}

TEST( SemismoothContact, BoundaryP0JumpStabilizationCouplesAdjacentFaces )
{
    constexpr mfem::real_t gamma = 1. / 10.;
    constexpr mfem::real_t delta = 2.;
    mfem::Mesh mesh = mfem::Mesh::MakeCartesian2D( 2, 1, mfem::Element::QUADRILATERAL );
    mfem::H1_FECollection collection( 1, mesh.Dimension() );
    mfem::FiniteElementSpace space( &mesh, &collection, mesh.Dimension() );
    mfem::Vector diagonal( space.GetTrueVSize() );
    diagonal = 0.;
    DiagonalNonlinearOperator primal( diagonal );
    mfem::IdentityOperator displacementExtraction( space.GetTrueVSize() );
    const plugin::RigidPlaneObstacle obstacle( MakeVector( { 0., 0. } ), MakeVector( { 0., 1. } ) );
    const mfem::Array<int> contactAttributes = ContactAttributes( 1 );
    mfem::Array<int> essentialDofs;
    plugin::BoundaryMultiplierSpace multipliers( mesh, contactAttributes );
    plugin::SemismoothRigidContactOperator contact( primal, displacementExtraction, space, obstacle, multipliers,
                                                    essentialDofs, gamma, 1., 1., delta );

    mfem::Vector unknown( contact.Width() );
    unknown = 0.;
    SetUniformDisplacement( space, unknown, 1, 1. );
    unknown( contact.GetBlockOffsets()[1] ) = .2;
    unknown( contact.GetBlockOffsets()[1] + 1 ) = .6;
    mfem::Vector residual;
    contact.Mult( unknown, residual );
    EXPECT_NEAR( residual( contact.GetBlockOffsets()[1] ) + residual( contact.GetBlockOffsets()[1] + 1 ), -.04, kAssemblyTolerance );

    mfem::Vector direction( contact.Width() );
    direction = 0.;
    direction( contact.GetBlockOffsets()[1] ) = 1.;
    direction( contact.GetBlockOffsets()[1] + 1 ) = -1.;
    mfem::Vector tangentAction( contact.Height() );
    contact.GetGradient( unknown ).Mult( direction, tangentAction );
    EXPECT_NEAR( tangentAction( contact.GetBlockOffsets()[1] ), -.25, kAssemblyTolerance );
    EXPECT_NEAR( tangentAction( contact.GetBlockOffsets()[1] + 1 ), .25, kAssemblyTolerance );
}

TEST( SemismoothContact, EmptyRuleRetainsJumpsAndRestoresPointAssemblyAndDiagnostics )
{
    constexpr mfem::real_t gamma = .125;
    constexpr mfem::real_t delta = 2.;
    constexpr mfem::real_t multiplierScale = 5.;
    // Two faces of length 1/2 share one vertex of measure 1.
    constexpr mfem::real_t scaledJump = multiplierScale * delta * gamma * .5;
    for ( const auto ordering : { mfem::Ordering::byNODES, mfem::Ordering::byVDIM } )
    {
        SCOPED_TRACE( ::testing::Message() << "ordering=" << ordering );
        mfem::Mesh mesh = mfem::Mesh::MakeCartesian2D( 2, 1, mfem::Element::QUADRILATERAL );
        mfem::H1_FECollection collection( 1, 2 );
        mfem::FiniteElementSpace space( &mesh, &collection, 2, ordering );
        mfem::Vector diagonal( space.GetTrueVSize() );
        diagonal = 0.;
        DiagonalNonlinearOperator primal( diagonal );
        mfem::IdentityOperator extraction( space.GetTrueVSize() );
        const plugin::RigidPlaneObstacle obstacle( MakeVector( { 0., 0. } ), MakeVector( { 0., 1. } ) );
        plugin::BoundaryMultiplierSpace multipliers( mesh, ContactAttributes( 1 ) );
        mfem::Array<int> essential;
        mfem::IntegrationRule emptyRule;
        plugin::SemismoothRigidContactOperator contact( primal, extraction, space, obstacle, multipliers, essential,
                                                        gamma, 3., multiplierScale, delta );
        const int offset = contact.GetBlockOffsets()[1];
        ASSERT_EQ( contact.GetContactSubMesh().GetNE(), 2 );
        ASSERT_EQ( contact.Width() - offset, 2 );
        ASSERT_EQ( emptyRule.GetNPoints(), 0 );
        mfem::Vector unknown( contact.Width() );
        unknown = 0.;
        unknown( offset ) = .5;
        unknown( offset + 1 ) = 1.5;
        const auto sampleJacobian = [&]()
        {
            mfem::DenseMatrix matrix( contact.Height(), contact.Width() );
            mfem::Vector direction( contact.Width() ), action( contact.Height() );
            auto& jacobian = contact.GetGradient( unknown );
            for ( int column = 0; column < contact.Width(); ++column )
            {
                direction = 0.;
                direction( column ) = 1.;
                jacobian.Mult( direction, action );
                for ( int row = 0; row < contact.Height(); ++row )
                {
                    matrix( row, column ) = action( row );
                }
            }
            return matrix;
        };

        // Clear active uu/uL/Lu terms and inactive LL mass on successive visits.
        for ( const mfem::real_t gap : { mfem::real_t{ -.25 }, mfem::real_t{ 1. } } )
        {
            SCOPED_TRACE( ::testing::Message() << "gap=" << gap );
            SetUniformDisplacement( space, unknown, 1, gap );
            mfem::Vector referenceResidual;
            contact.Mult( unknown, referenceResidual );
            const mfem::DenseMatrix referenceJacobian = sampleJacobian();
            const auto referenceDiagnostics = contact.ComputeContactDiagnostics( unknown );
            ASSERT_GT( referenceDiagnostics.QuadraturePointCount, 0 );
            EXPECT_EQ( referenceDiagnostics.ActiveQuadraturePointCount, gap < 0. ? referenceDiagnostics.QuadraturePointCount : 0 );

            contact.SetIntegrationRule( &emptyRule );
            mfem::Vector residual( referenceResidual ), expectedResidual( contact.Height() );
            expectedResidual = 0.;
            expectedResidual( offset ) = scaledJump;
            expectedResidual( offset + 1 ) = -scaledJump;
            contact.Mult( unknown, residual );
            ExpectVectorNear( residual, expectedResidual, kAssemblyTolerance );
            mfem::DenseMatrix expectedJacobian( contact.Height() );
            expectedJacobian = 0.;
            expectedJacobian( offset, offset ) = expectedJacobian( offset + 1, offset + 1 ) = -scaledJump;
            expectedJacobian( offset, offset + 1 ) = expectedJacobian( offset + 1, offset ) = scaledJump;
            mfem::DenseMatrix emptyJacobianError = sampleJacobian();
            emptyJacobianError -= expectedJacobian;
            EXPECT_LE( emptyJacobianError.FNorm(), kAssemblyTolerance * ( 1. + expectedJacobian.FNorm() ) );
            const auto emptyDiagnostics = contact.ComputeContactDiagnostics( unknown );
            EXPECT_EQ( emptyDiagnostics.QuadraturePointCount, 0 );
            EXPECT_EQ( emptyDiagnostics.ActiveQuadraturePointCount, 0 );
            EXPECT_EQ( emptyDiagnostics.MinimumGap, std::numeric_limits<mfem::real_t>::infinity() );
            EXPECT_EQ( emptyDiagnostics.MinimumMultiplier, std::numeric_limits<mfem::real_t>::infinity() );
            EXPECT_EQ( emptyDiagnostics.MaximumMultiplier, -std::numeric_limits<mfem::real_t>::infinity() );
            EXPECT_EQ( emptyDiagnostics.MaximumPenetration, 0. );
            EXPECT_EQ( emptyDiagnostics.MaximumPressure, 0. );
            EXPECT_EQ( emptyDiagnostics.MaximumComplementarityResidual, 0. );
            EXPECT_EQ( emptyDiagnostics.Resultant.Size(), 2 );
            EXPECT_EQ( emptyDiagnostics.Resultant.Norml2(), 0. );

            contact.SetIntegrationRule( nullptr );
            contact.Mult( unknown, residual );
            ExpectVectorNear( residual, referenceResidual, kAssemblyTolerance );
            mfem::DenseMatrix restoredJacobianError = sampleJacobian();
            restoredJacobianError -= referenceJacobian;
            EXPECT_LE( restoredJacobianError.FNorm(), kAssemblyTolerance * ( 1. + referenceJacobian.FNorm() ) );
            const auto restoredDiagnostics = contact.ComputeContactDiagnostics( unknown );
            EXPECT_EQ( restoredDiagnostics.QuadraturePointCount, referenceDiagnostics.QuadraturePointCount );
            EXPECT_EQ( restoredDiagnostics.ActiveQuadraturePointCount, referenceDiagnostics.ActiveQuadraturePointCount );
            EXPECT_NEAR( restoredDiagnostics.MinimumGap, referenceDiagnostics.MinimumGap, kAssemblyTolerance );
            EXPECT_NEAR( restoredDiagnostics.MaximumPenetration, referenceDiagnostics.MaximumPenetration, kAssemblyTolerance );
            EXPECT_NEAR( restoredDiagnostics.MinimumMultiplier, referenceDiagnostics.MinimumMultiplier, kAssemblyTolerance );
            EXPECT_NEAR( restoredDiagnostics.MaximumMultiplier, referenceDiagnostics.MaximumMultiplier, kAssemblyTolerance );
            EXPECT_NEAR( restoredDiagnostics.MaximumPressure, referenceDiagnostics.MaximumPressure, kAssemblyTolerance );
            EXPECT_NEAR( restoredDiagnostics.MaximumComplementarityResidual,
                         referenceDiagnostics.MaximumComplementarityResidual, kAssemblyTolerance );
            ExpectVectorNear( restoredDiagnostics.Resultant, referenceDiagnostics.Resultant, kAssemblyTolerance );
        }
    }
}

TEST( SemismoothContact, GammaAndDeltaMatchIntegratedTwoFaceReference )
{
    // Independent hand integration on two straight faces of length 1/2:
    // use the constant normal displacement and the two P0 multipliers as
    // trial/test directions. Partition of unity gives every displacement
    // integral exactly, without using production quadrature or shape helpers.
    constexpr mfem::real_t faceLength = .5;
    constexpr mfem::real_t primalScale = 3.;
    constexpr mfem::real_t multiplierScale = 5.;
    constexpr std::array<mfem::real_t, 2> lambda = { .5, 1.5 };
    mfem::Mesh mesh = mfem::Mesh::MakeCartesian2D( 2, 1, mfem::Element::QUADRILATERAL );
    mfem::H1_FECollection collection( 1, mesh.Dimension() );
    mfem::FiniteElementSpace space( &mesh, &collection, mesh.Dimension() );
    mfem::NonlinearForm primal( &space );
    mfem::IdentityOperator extraction( space.GetTrueVSize() );
    const plugin::RigidPlaneObstacle obstacle( MakeVector( { 0., 0. } ), MakeVector( { 0., 1. } ) );
    const mfem::Array<int> attributes = ContactAttributes( 1 );
    mfem::Array<int> essential;
    plugin::BoundaryMultiplierSpace multipliers( mesh, attributes );
    for ( const mfem::real_t gamma : { mfem::real_t{ .125 }, mfem::real_t{ .5 } } )
        for ( const mfem::real_t delta : { mfem::real_t{ 0. }, mfem::real_t{ 2. } } )
        {
            plugin::SemismoothRigidContactOperator contact( primal, extraction, space, obstacle, multipliers, essential,
                                                            gamma, primalScale, multiplierScale, delta );
            // Reuse the operator through activation, release, and reactivation:
            // neither global blocks nor per-element scratch may retain old terms.
            for ( const mfem::real_t gap : { mfem::real_t{ -.25 }, mfem::real_t{ 1. }, mfem::real_t{ -.25 } } )
            {
                SCOPED_TRACE( ::testing::Message() << "gamma=" << gamma << " delta=" << delta << " gap=" << gap );
                const int offset = contact.GetBlockOffsets()[1];
                ASSERT_EQ( contact.Width() - offset, 2 );
                mfem::Vector unknown( contact.Width() ), residual;
                unknown = 0.;
                SetUniformDisplacement( space, unknown, 1, gap );
                unknown( offset ) = lambda[0];
                unknown( offset + 1 ) = lambda[1];
                contact.Mult( unknown, residual );

                std::array<mfem::Vector, 3> basis;
                for ( auto& direction : basis )
                {
                    direction.SetSize( contact.Width() );
                    direction = 0.;
                }
                SetUniformDisplacement( space, basis[0], 1, 1. );
                basis[1]( offset ) = 1.;
                basis[2]( offset + 1 ) = 1.;

                // The negative gap cases are fully active; gap=1 is inactive
                // for both prescribed multipliers at both compliance values.
                // A single interior vertex has measure 1 and h = faceLength.
                const bool active = gap < 0.;
                const mfem::real_t jump = delta * gamma * faceLength;
                const mfem::real_t normalResidual =
                    active ? -primalScale * faceLength * ( lambda[0] + lambda[1] - 2. * gap / gamma ) : 0.;
                EXPECT_NEAR( basis[0] * residual, normalResidual, kAssemblyTolerance * ( 1. + std::abs( normalResidual ) ) );
                for ( int i = 0; i < 2; ++i )
                {
                    const mfem::real_t moment = faceLength * ( active ? -gap : -gamma * lambda[i] );
                    const mfem::real_t expected = multiplierScale * ( moment - jump * ( lambda[i] - lambda[1 - i] ) );
                    EXPECT_NEAR( basis[i + 1] * residual, expected, kAssemblyTolerance * ( 1. + std::abs( expected ) ) );
                }

                mfem::DenseMatrix reference( 3 );
                reference = 0.;
                reference( 0, 0 ) = active ? primalScale / gamma : 0.;
                for ( int i = 1; i < 3; ++i )
                {
                    reference( 0, i ) = active ? -primalScale * faceLength : 0.;
                    reference( i, 0 ) = active ? -multiplierScale * faceLength : 0.;
                    reference( i, i ) = -multiplierScale * ( jump + ( active ? 0. : gamma * faceLength ) );
                }
                reference( 1, 2 ) = reference( 2, 1 ) = multiplierScale * jump;
                auto& jacobian = contact.GetGradient( unknown );
                mfem::Vector action( contact.Height() );
                for ( int column = 0; column < 3; ++column )
                {
                    jacobian.Mult( basis[column], action );
                    for ( int row = 0; row < 3; ++row )
                    {
                        EXPECT_NEAR( basis[row] * action, reference( row, column ),
                                     kAssemblyTolerance * ( 1. + std::abs( reference( row, column ) ) ) );
                    }
                }
            }
        }
}

TEST( SemismoothContact, CurvedThreeDimensionalJumpUsesIntegratedInterfaceMeasure )
{
    constexpr mfem::real_t gamma = 1. / 10.;
    constexpr mfem::real_t delta = 2.;
    constexpr mfem::real_t multiplier1 = .2;
    constexpr mfem::real_t multiplier2 = .6;
    mfem::Mesh mesh = mfem::Mesh::MakeCartesian3D( 2, 1, 1, mfem::Element::HEXAHEDRON, 1., 1., 1. );
    mesh.SetCurvature( 2, false, 3, mfem::Ordering::byVDIM );
    mfem::GridFunction* nodes = mesh.GetNodes();
    ASSERT_NE( nodes, nullptr );
    mfem::FiniteElementSpace* nodeSpace = nodes->FESpace();
    ASSERT_NE( nodeSpace, nullptr );
    for ( int dof = 0; dof < nodeSpace->GetNDofs(); dof++ )
    {
        const mfem::real_t y = ( *nodes )( nodeSpace->DofToVDof( dof, 1 ) );
        ( *nodes )( nodeSpace->DofToVDof( dof, 2 ) ) += .5 * y * ( 1. - y );
    }

    mfem::H1_FECollection collection( 1, mesh.Dimension() );
    mfem::FiniteElementSpace space( &mesh, &collection, mesh.Dimension() );
    mfem::Vector diagonal( space.GetTrueVSize() );
    diagonal = 0.;
    DiagonalNonlinearOperator primal( diagonal );
    mfem::IdentityOperator displacementExtraction( space.GetTrueVSize() );
    const plugin::RigidPlaneObstacle obstacle( MakeVector( { 0., 0., -1. } ), MakeVector( { 0., 0., 1. } ) );
    const mfem::Array<int> contactAttributes = ContactAttributes( 1 );
    mfem::Array<int> essentialDofs;
    plugin::BoundaryMultiplierSpace multipliers( mesh, contactAttributes );
    plugin::SemismoothRigidContactOperator unstabilized( primal, displacementExtraction, space, obstacle, multipliers,
                                                         essentialDofs, gamma, 1., 1., 0. );
    plugin::SemismoothRigidContactOperator stabilized( primal, displacementExtraction, space, obstacle, multipliers,
                                                       essentialDofs, gamma, 1., 1., delta );

    auto& contactMesh = const_cast<mfem::SubMesh&>( stabilized.GetContactSubMesh() );
    int interiorFace = -1;
    int element1 = -1;
    int element2 = -1;
    for ( int face = 0; face < contactMesh.GetNumFaces(); face++ )
    {
        contactMesh.GetFaceElements( face, &element1, &element2 );
        if ( element2 >= 0 )
        {
            interiorFace = face;
            break;
        }
    }
    ASSERT_GE( interiorFace, 0 );
    mfem::FaceElementTransformations* faceTransformation = contactMesh.GetFaceElementTransformations( interiorFace );
    ASSERT_NE( faceTransformation, nullptr );
    const mfem::IntegrationRule& highOrderRule = mfem::IntRules.Get( faceTransformation->GetGeometryType(), 10 );
    const mfem::IntegrationRule& constantRule = mfem::IntRules.Get( faceTransformation->GetGeometryType(), 0 );
    auto integrateMeasure = [&]( const mfem::IntegrationRule& rule )
    {
        mfem::real_t measure = 0.;
        for ( int point = 0; point < rule.GetNPoints(); point++ )
        {
            const mfem::IntegrationPoint& integrationPoint = rule.IntPoint( point );
            faceTransformation->SetAllIntPoints( &integrationPoint );
            measure += integrationPoint.weight * faceTransformation->Weight();
        }
        return measure;
    };
    const mfem::real_t interfaceMeasure = integrateMeasure( highOrderRule );
    EXPECT_GT( std::abs( interfaceMeasure - integrateMeasure( constantRule ) ), 1e-3 );
    const mfem::real_t elementSize =
        .5 * ( std::sqrt( contactMesh.GetElementVolume( element1 ) ) + std::sqrt( contactMesh.GetElementVolume( element2 ) ) );
    const mfem::real_t jumpCoefficient = delta * gamma * elementSize * interfaceMeasure;
    const mfem::real_t interfaceTolerance = kSinglePrecision ? 2e-4f : 2e-8;

    mfem::Array<int> element1Dofs;
    mfem::Array<int> element2Dofs;
    stabilized.GetMultiplierSpace().GetElementDofs( element1, element1Dofs );
    stabilized.GetMultiplierSpace().GetElementDofs( element2, element2Dofs );
    ASSERT_EQ( element1Dofs.Size(), 1 );
    ASSERT_EQ( element2Dofs.Size(), 1 );
    const int multiplierDof1 = element1Dofs[0] >= 0 ? element1Dofs[0] : -1 - element1Dofs[0];
    const int multiplierDof2 = element2Dofs[0] >= 0 ? element2Dofs[0] : -1 - element2Dofs[0];
    mfem::Vector unknown( stabilized.Width() );
    unknown = 0.;
    unknown( stabilized.GetBlockOffsets()[1] + multiplierDof1 ) = multiplier1;
    unknown( stabilized.GetBlockOffsets()[1] + multiplierDof2 ) = multiplier2;
    mfem::Vector stabilizedResidual;
    mfem::Vector unstabilizedResidual;
    stabilized.Mult( unknown, stabilizedResidual );
    unstabilized.Mult( unknown, unstabilizedResidual );
    stabilizedResidual -= unstabilizedResidual;

    EXPECT_NEAR( stabilizedResidual( stabilized.GetBlockOffsets()[1] + multiplierDof1 ),
                 -jumpCoefficient * ( multiplier1 - multiplier2 ), interfaceTolerance * ( 1. + jumpCoefficient ) );
    EXPECT_NEAR( stabilizedResidual( stabilized.GetBlockOffsets()[1] + multiplierDof2 ),
                 jumpCoefficient * ( multiplier1 - multiplier2 ), interfaceTolerance * ( 1. + jumpCoefficient ) );
}

namespace
{
void CheckMixedJacobianAndTranspose( plugin::SemismoothRigidContactOperator& contact, const mfem::Vector& unknown )
{
    const auto& offsets = contact.GetBlockOffsets();
    // Isolate each input field so errors in different blocks cannot cancel.
    for ( int columnBlock = 0; columnBlock < 2; ++columnBlock )
    {
        mfem::Vector direction( unknown.Size() ), action( unknown.Size() ), plus( unknown ), minus( unknown ), rp, rm;
        direction = 0.;
        for ( int i = offsets[columnBlock]; i < offsets[columnBlock + 1]; ++i )
        {
            direction( i ) = std::cos( .7 + i );
        }
        direction /= direction.Norml2();
        contact.GetGradient( unknown ).Mult( direction, action );
        plus.Add( kFiniteDifferenceStep, direction );
        minus.Add( -kFiniteDifferenceStep, direction );
        contact.Mult( plus, rp );
        contact.Mult( minus, rm );
        rp -= rm;
        rp /= 2. * kFiniteDifferenceStep;
        for ( int rowBlock = 0; rowBlock < 2; ++rowBlock )
        {
            SCOPED_TRACE( ::testing::Message() << "block(" << rowBlock << ',' << columnBlock << ')' );
            mfem::Vector actual( action.GetData() + offsets[rowBlock], offsets[rowBlock + 1] - offsets[rowBlock] );
            mfem::Vector expected( rp.GetData() + offsets[rowBlock], actual.Size() );
            ExpectVectorNear( actual, expected, kDerivativeTolerance );
        }
    }

    // Independent transpose oracle: differentiate residual columns, then transpose
    // that numerical matrix. Unequal row scales make a mistaken Mult call fail.
    mfem::DenseMatrix numerical( unknown.Size() );
    for ( int column = 0; column < unknown.Size(); ++column )
    {
        mfem::Vector plus( unknown ), minus( unknown ), rp, rm;
        plus( column ) += kFiniteDifferenceStep;
        minus( column ) -= kFiniteDifferenceStep;
        contact.Mult( plus, rp );
        contact.Mult( minus, rm );
        for ( int row = 0; row < unknown.Size(); ++row )
        {
            numerical( row, column ) = ( rp( row ) - rm( row ) ) / ( 2. * kFiniteDifferenceStep );
        }
    }
    for ( int rowBlock = 0; rowBlock < 2; ++rowBlock )
    {
        mfem::Vector test( unknown.Size() ), actual( unknown.Size() ), expected( unknown.Size() );
        test = 0.;
        for ( int i = offsets[rowBlock]; i < offsets[rowBlock + 1]; ++i )
        {
            test( i ) = std::sin( .3 + i );
        }
        test /= test.Norml2();
        contact.GetGradient( unknown ).MultTranspose( test, actual );
        numerical.MultTranspose( test, expected );
        ExpectVectorNear( actual, expected, kDerivativeTolerance );
    }
}
} // namespace

TEST( SemismoothContact, BoundaryP1PlaneAndSphereMixedBlocksTransposeAndAssemblyReset )
{
    for ( const auto ordering : { mfem::Ordering::byNODES, mfem::Ordering::byVDIM } )
        for ( const bool curved : { false, true } )
        {
            SCOPED_TRACE( ::testing::Message() << "ordering=" << ordering << " sphere=" << curved );
            mfem::Mesh mesh = mfem::Mesh::MakeCartesian2D( 2, 1, mfem::Element::QUADRILATERAL );
            mfem::H1_FECollection collection( 2, 2 );
            mfem::FiniteElementSpace space( &mesh, &collection, 2, ordering );
            mfem::NonlinearForm primal( &space );
            mfem::IdentityOperator extraction( space.GetTrueVSize() );
            plugin::RigidPlaneObstacle plane( MakeVector( { 0., 0. } ), MakeVector( { 0., 1. } ) );
            plugin::RigidSphereObstacle sphere( MakeVector( { .5, -.4 } ), 1.2 );
            const plugin::RigidObstacle& obstacle = curved ? static_cast<const plugin::RigidObstacle&>( sphere ) : plane;
            const auto attributes = ContactAttributes( 1 );
            plugin::BoundaryMultiplierSpace multipliers( mesh, attributes, std::make_unique<mfem::L2_FECollection>( 1, 1 ) );
            mfem::Array<int> essential;
            mfem::IntegrationRule rule( 2 );
            rule.IntPoint( 0 ).Set1w( .13, .37 );
            rule.IntPoint( 1 ).Set1w( .61, .63 );
            plugin::SemismoothRigidContactOperator contact( primal, extraction, space, obstacle, multipliers, essential,
                                                            .125, .17, 4.2, .7 );
            contact.SetIntegrationRule( &rule );
            mfem::FunctionCoefficient affine( []( const mfem::Vector& x ) { return 20. * ( x( 0 ) - .53 ); } );
            mfem::GridFunction lambda( &multipliers.GetSpace() );
            lambda.ProjectCoefficient( affine );
            mfem::Vector firstResidual, firstAction( contact.Height() ), direction( contact.Width() );
            for ( int i = 0; i < direction.Size(); ++i )
            {
                direction( i ) = std::cos( i + .4 );
            }
            // Release and reactivation must clear all four blocks, including the
            // inactive multiplier mass and the active curved-obstacle Hessian.
            for ( const int phase : { 1, -1, 0, 1 } )
            {
                SCOPED_TRACE( ::testing::Message() << "phase=" << phase );
                mfem::BlockVector unknown( contact.GetBlockOffsets() );
                unknown = 0.;
                SetUniformDisplacement( space, unknown, 1, -.03 );
                unknown.GetBlock( 1 ) = lambda;
                unknown.GetBlock( 1 ) += 40. * phase;
                for ( int e = 0; e < multipliers.GetMesh().GetNE(); ++e )
                {
                    auto* transform = mesh.GetBdrFaceTransformations( multipliers.GetParentBoundaryElement( e ) );
                    for ( int q = 0; q < rule.GetNPoints(); ++q )
                    {
                        mfem::Vector x( 2 );
                        transform->Transform( rule.IntPoint( q ), x );
                        const auto lambdaAtPoint = 20. * ( x( 0 ) - .53 ) + 40. * phase;
                        x( 1 ) -= .03;
                        plugin::SignedDistanceEvaluation evaluation( 2 );
                        obstacle.Evaluate( x, evaluation );
                        EXPECT_GT( std::abs( lambdaAtPoint - evaluation.Gap / .125 ), .1 );
                    }
                }
                const auto diagnostics = contact.ComputeContactDiagnostics( unknown );
                if ( phase == 1 )
                {
                    EXPECT_EQ( diagnostics.ActiveQuadraturePointCount, diagnostics.QuadraturePointCount );
                }
                else if ( phase == -1 )
                {
                    EXPECT_EQ( diagnostics.ActiveQuadraturePointCount, 0 );
                }
                else
                {
                    EXPECT_GT( diagnostics.ActiveQuadraturePointCount, 0 );
                    EXPECT_LT( diagnostics.ActiveQuadraturePointCount, diagnostics.QuadraturePointCount );
                }
                mfem::Vector residual, action( contact.Height() );
                contact.Mult( unknown, residual );
                contact.GetGradient( unknown ).Mult( direction, action );
                if ( phase == 1 )
                {
                    if ( firstResidual.Size() == 0 )
                    {
                        firstResidual = residual;
                        firstAction = action;
                    }
                    else
                    {
                        ExpectVectorNear( residual, firstResidual, kAssemblyTolerance );
                        ExpectVectorNear( action, firstAction, kAssemblyTolerance );
                    }
                }
                CheckMixedJacobianAndTranspose( contact, unknown );
            }
        }
}

TEST( SemismoothContact, BoundaryP1EssentialMaskMatchesIndependentRowAndColumnElimination )
{
    for ( const auto ordering : { mfem::Ordering::byNODES, mfem::Ordering::byVDIM } )
    {
        mfem::Mesh mesh = mfem::Mesh::MakeCartesian2D( 2, 1, mfem::Element::QUADRILATERAL );
        mfem::H1_FECollection collection( 2, 2 );
        mfem::FiniteElementSpace space( &mesh, &collection, 2, ordering );
        mfem::Vector diagonal( space.GetTrueVSize() );
        diagonal = 0.;
        DiagonalNonlinearOperator primal( diagonal );
        mfem::IdentityOperator extraction( space.GetTrueVSize() );
        plugin::RigidSphereObstacle obstacle( MakeVector( { .5, -.4 } ), 1.2 );
        plugin::BoundaryMultiplierSpace multipliers( mesh, ContactAttributes( 1 ),
                                                     std::make_unique<mfem::L2_FECollection>( 1, 1 ) );
        mfem::Array<int> trace, essential( 2 ), none;
        space.GetBdrElementDofs( FindBoundaryElement( mesh, 1 ), trace );
        essential[0] = space.DofToVDof( trace[0], 0 );
        essential[1] = space.DofToVDof( trace[0], 1 );
        plugin::SemismoothRigidContactOperator freeContact( primal, extraction, space, obstacle, multipliers, none,
                                                            .125, .17, 4.2, .7 );
        plugin::SemismoothRigidContactOperator maskedContact( primal, extraction, space, obstacle, multipliers,
                                                              essential, .125, .17, 4.2, .7 );
        mfem::FunctionCoefficient affine( []( const mfem::Vector& x ) { return 20. * ( x( 0 ) - .53 ); } );
        mfem::GridFunction lambda( &multipliers.GetSpace() );
        lambda.ProjectCoefficient( affine );
        for ( const int phase : { 1, -1, 0 } )
        {
            mfem::BlockVector unknown( freeContact.GetBlockOffsets() );
            unknown = 0.;
            SetUniformDisplacement( space, unknown, 1, -.03 );
            unknown.GetBlock( 1 ) = lambda;
            unknown.GetBlock( 1 ) += 40. * phase;
            mfem::Vector actual, expected;
            freeContact.Mult( unknown, expected );
            maskedContact.Mult( unknown, actual );
            for ( const int dof : essential )
            {
                expected( dof ) = 0.;
            }
            ExpectVectorNear( actual, expected, kAssemblyTolerance );
            auto& freeJacobian = freeContact.GetGradient( unknown );
            auto& maskedJacobian = maskedContact.GetGradient( unknown );
            for ( const bool transpose : { false, true } )
                for ( int column = 0; column < unknown.Size(); ++column )
                {
                    mfem::Vector unit( unknown.Size() ), freeUnit( unknown.Size() );
                    unit = 0.;
                    unit( column ) = 1.;
                    freeUnit = unit;
                    for ( const int dof : essential )
                    {
                        freeUnit( dof ) = 0.;
                    }
                    if ( transpose )
                    {
                        freeJacobian.MultTranspose( freeUnit, expected );
                        maskedJacobian.MultTranspose( unit, actual );
                    }
                    else
                    {
                        freeJacobian.Mult( freeUnit, expected );
                        maskedJacobian.Mult( unit, actual );
                    }
                    for ( const int dof : essential )
                    {
                        expected( dof ) = 0.;
                    }
                    ExpectVectorNear( actual, expected, kAssemblyTolerance );
                }
        }
    }
}

TEST( SemismoothContact, BoundaryP1AffineMultiplierMomentsArePreservedUnderRefinement )
{
    // A manufactured assembly check for this flat P2/P1 pair, not an inf-sup
    // claim. Continuous affine multipliers have zero stabilization jump.
    constexpr mfem::real_t gamma = .125, primalScale = .17, multiplierScale = 4.2;
    for ( const int nx : { 1, 2, 4 } )
    {
        mfem::Mesh mesh = mfem::Mesh::MakeCartesian2D( nx, 1, mfem::Element::QUADRILATERAL );
        mfem::H1_FECollection collection( 2, 2 );
        mfem::FiniteElementSpace space( &mesh, &collection, 2 );
        mfem::NonlinearForm primal( &space );
        mfem::IdentityOperator extraction( space.GetTrueVSize() );
        plugin::RigidPlaneObstacle obstacle( MakeVector( { 0., 0. } ), MakeVector( { 0., 1. } ) );
        plugin::BoundaryMultiplierSpace multipliers( mesh, ContactAttributes( 1 ),
                                                     std::make_unique<mfem::L2_FECollection>( 1, 1 ) );
        mfem::Array<int> essential;
        plugin::SemismoothRigidContactOperator contact( primal, extraction, space, obstacle, multipliers, essential,
                                                        gamma, primalScale, multiplierScale, .7 );
        mfem::FunctionCoefficient affine( []( const mfem::Vector& x ) { return .4 + .6 * x( 0 ); } );
        mfem::FunctionCoefficient coordinate( []( const mfem::Vector& x ) { return x( 0 ); } );
        mfem::GridFunction lambda( &multipliers.GetSpace() ), mu( &multipliers.GetSpace() ), v( &space );
        lambda.ProjectCoefficient( affine );
        mu.ProjectCoefficient( coordinate );
        mfem::VectorFunctionCoefficient normalMoment( 2,
                                                      []( const mfem::Vector& x, mfem::Vector& value )
                                                      {
                                                          value( 0 ) = 0.;
                                                          value( 1 ) = x( 0 );
                                                      } );
        v.ProjectCoefficient( normalMoment );
        for ( const auto gap : { mfem::real_t{ -.08 }, mfem::real_t{ 1. } } )
        {
            mfem::BlockVector unknown( contact.GetBlockOffsets() );
            unknown = 0.;
            SetUniformDisplacement( space, unknown, 1, gap );
            unknown.GetBlock( 1 ) = lambda;
            mfem::Vector residual;
            contact.Mult( unknown, residual );
            mfem::Vector ru( residual.GetData(), space.GetTrueVSize() );
            mfem::Vector rl( residual.GetData() + space.GetTrueVSize(), lambda.Size() );
            const bool active = gap < 0.;
            const auto expectedForceMoment = active ? -primalScale * ( .4 / 2. + .6 / 3. - gap / ( 2. * gamma ) ) : 0.;
            EXPECT_NEAR( v * ru, expectedForceMoment, kAssemblyTolerance );
            EXPECT_NEAR( rl.Sum(), multiplierScale * ( active ? -gap : -gamma * ( .4 + .6 / 2. ) ), kAssemblyTolerance );
            EXPECT_NEAR( mu * rl, multiplierScale * ( active ? -gap / 2. : -gamma * ( .4 / 2. + .6 / 3. ) ), kAssemblyTolerance );
        }
    }
}

TEST( SemismoothContact, BoundaryP1Q1JumpUsesTracesAndCurvedInterfaceProducts )
{
    constexpr mfem::real_t gamma = .125, delta = 2., multiplierScale = 4.2;
    const mfem::real_t interfaceTolerance = kSinglePrecision ? 2e-4f : 2e-8;
    for ( const int dimension : { 2, 3 } )
    {
        SCOPED_TRACE( ::testing::Message() << "dimension=" << dimension );
        mfem::Mesh mesh = dimension == 2 ? mfem::Mesh::MakeCartesian2D( 2, 1, mfem::Element::QUADRILATERAL )
                                         : mfem::Mesh::MakeCartesian3D( 2, 1, 1, mfem::Element::HEXAHEDRON );
        if ( dimension == 3 )
        {
            mesh.SetCurvature( 2 );
            mfem::VectorFunctionCoefficient warp( 3,
                                                  []( const mfem::Vector& x, mfem::Vector& y )
                                                  {
                                                      y = x;
                                                      y( 2 ) += .5 * x( 1 ) * ( 1. - x( 1 ) );
                                                  } );
            mesh.Transform( warp );
        }
        mfem::H1_FECollection collection( 2, dimension );
        mfem::FiniteElementSpace space( &mesh, &collection, dimension );
        mfem::NonlinearForm primal( &space );
        mfem::IdentityOperator extraction( space.GetTrueVSize() );
        mfem::Vector point( dimension ), normal( dimension );
        point = 0.;
        point( dimension - 1 ) = -1.;
        normal = 0.;
        normal( dimension - 1 ) = 1.;
        plugin::RigidPlaneObstacle obstacle( point, normal );
        plugin::BoundaryMultiplierSpace multipliers( mesh, ContactAttributes( 1 ),
                                                     std::make_unique<mfem::L2_FECollection>( 1, dimension - 1 ) );
        mfem::Array<int> essential;
        plugin::SemismoothRigidContactOperator unstabilized( primal, extraction, space, obstacle, multipliers,
                                                             essential, gamma, .17, multiplierScale, 0. );
        plugin::SemismoothRigidContactOperator stabilized( primal, extraction, space, obstacle, multipliers, essential,
                                                           gamma, .17, multiplierScale, delta );
        auto& contactMesh = multipliers.GetMesh();
        auto& multiplierSpace = multipliers.GetSpace();
        const int size = multiplierSpace.GetTrueVSize();
        ASSERT_EQ( contactMesh.GetNE(), 2 );
        ASSERT_EQ( size, dimension == 2 ? 4 : 8 );

        // Independently integrate S_ij = delta*gamma*h * integral [M_i][M_j].
        // MFEM Loc1/Loc2 provide both traces at the SAME physical interface
        // point. No production multiplier shape/map or stabilization helper is used.
        const auto integrateJump = [&]( const int order )
        {
            mfem::DenseMatrix matrix( size );
            matrix = 0.;
            for ( int face = 0; face < contactMesh.GetNumFaces(); ++face )
            {
                int e1, e2;
                contactMesh.GetFaceElements( face, &e1, &e2 );
                if ( e2 < 0 )
                {
                    continue;
                }
                const mfem::real_t exponent = 1. / ( dimension - 1 );
                const auto h = .5 * ( std::pow( contactMesh.GetElementVolume( e1 ), exponent ) +
                                      std::pow( contactMesh.GetElementVolume( e2 ), exponent ) );
                auto* transform = contactMesh.GetFaceElementTransformations( face );
                const auto& rule = mfem::IntRules.Get( transform->GetGeometryType(), order );
                mfem::Array<int> dofs1, dofs2;
                multiplierSpace.GetElementDofs( e1, dofs1 );
                multiplierSpace.GetElementDofs( e2, dofs2 );
                mfem::Vector shape1( dofs1.Size() ), shape2( dofs2.Size() ), jump( size );
                for ( int q = 0; q < rule.GetNPoints(); ++q )
                {
                    const auto& ip = rule.IntPoint( q );
                    transform->SetAllIntPoints( &ip );
                    multiplierSpace.GetFE( e1 )->CalcShape( transform->GetElement1IntPoint(), shape1 );
                    multiplierSpace.GetFE( e2 )->CalcShape( transform->GetElement2IntPoint(), shape2 );
                    jump = 0.;
                    jump.AddElementVector( dofs1, shape1 );
                    jump.AddElementVector( dofs2, -1., shape2 );
                    // In 2D the interface is a vertex with counting measure 1.
                    const auto weight = delta * gamma * h * ip.weight * ( dimension == 2 ? 1. : transform->Weight() );
                    for ( int i = 0; i < size; ++i )
                        for ( int j = 0; j < size; ++j )
                        {
                            matrix( i, j ) += weight * jump( i ) * jump( j );
                        }
                }
            }
            return matrix;
        };
        const mfem::DenseMatrix reference = integrateJump( 30 );
        mfem::DenseMatrix quadratureError = integrateJump( 20 );
        quadratureError -= reference;
        EXPECT_LE( quadratureError.FNorm(), kAssemblyTolerance );
        if ( dimension == 3 )
        {
            mfem::DenseMatrix midpointError = integrateJump( 0 );
            midpointError -= reference;
            EXPECT_GT( midpointError.FNorm(), 1e-3 );
        }

        for ( const int field : { 0, 1, 2 } )
        {
            SCOPED_TRACE( ::testing::Message() << "field=" << field );
            mfem::FunctionCoefficient coefficient(
                [dimension, field]( const mfem::Vector& x ) -> mfem::real_t
                {
                    const mfem::real_t y = dimension == 3 ? x( 1 ) : 0.;
                    if ( field == 0 )
                    {
                        return mfem::real_t{ .6 };
                    }
                    return .2 + .6 * x( 0 ) + .7 * y + ( field == 2 && x( 0 ) > .5 ? .4 + .3 * y : 0. );
                } );
            mfem::GridFunction lambda( &multiplierSpace );
            lambda.ProjectCoefficient( coefficient );
            mfem::BlockVector unknown( stabilized.GetBlockOffsets() );
            unknown = 0.;
            unknown.GetBlock( 1 ) = lambda;
            mfem::Vector expected( size ), rs, ru;
            reference.Mult( lambda, expected );
            expected *= -multiplierScale;
            stabilized.Mult( unknown, rs );
            unstabilized.Mult( unknown, ru );
            rs -= ru;
            mfem::Vector difference( rs.GetData() + space.GetTrueVSize(), size );
            ExpectVectorNear( difference, expected, interfaceTolerance );
            mfem::Vector primalDifference( rs.GetData(), space.GetTrueVSize() );
            EXPECT_LE( primalDifference.Norml2(), kAssemblyTolerance );
            if ( field < 2 )
            {
                // Gauss-Legendre coefficients of a continuous affine field differ
                // across adjacent elements; comparing coefficients would penalize it.
                EXPECT_LE( difference.Norml2(), interfaceTolerance );
            }
            else
            {
                EXPECT_GT( difference.Norml2(), .01 );
                EXPECT_LT( lambda * difference, -.01 );
                EXPECT_NEAR( difference.Sum(), 0., interfaceTolerance );
            }

            auto& js = stabilized.GetGradient( unknown );
            auto& ju = unstabilized.GetGradient( unknown );
            for ( int column = 0; column < size; ++column )
            {
                mfem::Vector unit( unknown.Size() ), as( unknown.Size() ), au( unknown.Size() );
                unit = 0.;
                unit( space.GetTrueVSize() + column ) = 1.;
                js.Mult( unit, as );
                ju.Mult( unit, au );
                as -= au;
                for ( int row = 0; row < size; ++row )
                {
                    EXPECT_NEAR( as( space.GetTrueVSize() + row ), -multiplierScale * reference( row, column ), interfaceTolerance );
                }
                // The jump derivative is linear and independent of activity.
                mfem::Vector plus( unknown ), minus( unknown ), sp, sm, up, um;
                plus.Add( kFiniteDifferenceStep, unit );
                minus.Add( -kFiniteDifferenceStep, unit );
                stabilized.Mult( plus, sp );
                stabilized.Mult( minus, sm );
                unstabilized.Mult( plus, up );
                unstabilized.Mult( minus, um );
                sp -= up;
                sp -= sm;
                sp += um;
                sp /= 2. * kFiniteDifferenceStep;
                ExpectVectorNear( as, sp, kDerivativeTolerance );
            }
        }
    }
}

TEST( BoundaryMultiplierSpace, RejectsUnsupportedCurvedParentGeometry )
{
    GTEST_FLAG_SET( death_test_style, "threadsafe" );
    {
        mfem::Mesh root = mfem::Mesh::MakeCartesian2D( 1, 1, mfem::Element::QUADRILATERAL );
        root.SetCurvature( 2, true );
        const mfem::Array<int> attributes( root.bdr_attributes );
        // Reject discontinuous geometry before MFEM's boundary transfer aborts.
        EXPECT_DEATH( (void)plugin::BoundaryMultiplierSpace( root, attributes ), "Gauss-Lobatto H1" );
    }
    {
        mfem::Mesh root = mfem::Mesh::MakeCartesian2D( 2, 1, mfem::Element::QUADRILATERAL );
        root.GetElement( 0 )->SetAttribute( 2 );
        root.SetAttributes();
        root.SetCurvature( 2 );
        mfem::Array<int> domainAttributes( 1 );
        domainAttributes[0] = 2;
        mfem::SubMesh child = mfem::SubMesh::CreateFromDomain( root, domainAttributes );
        ASSERT_EQ( child.GetNE(), 1 );
        ASSERT_NE( child.GetNodes(), nullptr );
        const mfem::Array<int> attributes( child.bdr_attributes );
        // The first MFEM extraction succeeds; the unsupported operation is a
        // further boundary extraction from this curved SubMesh parent.
        EXPECT_DEATH( (void)plugin::BoundaryMultiplierSpace( child, attributes ), "curved SubMesh" );
    }
}

TEST( SemismoothContact, HigherOrderMultiplierQuadratureAndPositiveBasis )
{
    // Assembly/quadrature coverage for an experimental pair, not a stability claim.
    constexpr mfem::real_t gamma = .125, primalScale = .17, multiplierScale = 4.2;
    for ( const int basisType : { mfem::BasisType::GaussLegendre, mfem::BasisType::Positive } )
    {
        SCOPED_TRACE( ::testing::Message() << "basis=" << basisType );
        mfem::Mesh mesh = mfem::Mesh::MakeCartesian2D( 1, 1, mfem::Element::QUADRILATERAL );
        mfem::H1_FECollection collection( 1, 2 );
        mfem::FiniteElementSpace space( &mesh, &collection, 2 );
        mfem::NonlinearForm primal( &space );
        mfem::IdentityOperator extraction( space.GetTrueVSize() );
        plugin::RigidPlaneObstacle obstacle( MakeVector( { 0., 0. } ), MakeVector( { 0., 1. } ) );
        plugin::BoundaryMultiplierSpace multipliers( mesh, ContactAttributes( 1 ),
                                                     std::make_unique<mfem::L2_FECollection>( 2, 1, basisType ) );
        mfem::Array<int> essential;
        plugin::SemismoothRigidContactOperator contact( primal, extraction, space, obstacle, multipliers, essential,
                                                        gamma, primalScale, multiplierScale, 2. );
        ASSERT_EQ( multipliers.GetMesh().GetNE(), 1 );
        ASSERT_EQ( multipliers.GetSpace().GetTrueVSize(), 3 );
        mfem::FunctionCoefficient square( []( const mfem::Vector& x ) { return x( 0 ) * x( 0 ); } );
        mfem::FunctionCoefficient coordinate( []( const mfem::Vector& x ) { return x( 0 ); } );
        mfem::ConstantCoefficient one( 1. );
        mfem::GridFunction lambda( &multipliers.GetSpace() ), linearTest( &multipliers.GetSpace() ),
            constantTest( &multipliers.GetSpace() );
        // PositiveFiniteElement::Project samples values (a monotone approximation,
        // not an exact projection). Use element L2 projection for Bernstein DOFs:
        // coefficients must represent x^2, not be assumed to equal nodal values.
        lambda.ProjectCoefficientElementL2( square );
        linearTest.ProjectCoefficientElementL2( coordinate );
        constantTest.ProjectCoefficientElementL2( one );
        for ( const auto t : { mfem::real_t{ .13 }, mfem::real_t{ .61 }, mfem::real_t{ .89 } } )
        {
            mfem::IntegrationPoint ip;
            ip.Set1w( t, 1. );
            mfem::Vector x( 2 );
            multipliers.GetMesh().GetElementTransformation( 0 )->Transform( ip, x );
            EXPECT_NEAR( lambda.GetValue( 0, ip ), x( 0 ) * x( 0 ), kAssemblyTolerance );
        }
        mfem::Vector normalTest( space.GetTrueVSize() );
        normalTest = 0.;
        SetUniformDisplacement( space, normalTest, 1, 1. );
        for ( const auto gap : { mfem::real_t{ 1. }, mfem::real_t{ -.1 } } )
        {
            SCOPED_TRACE( ::testing::Message() << "gap=" << gap );
            mfem::BlockVector unknown( contact.GetBlockOffsets() );
            unknown = 0.;
            SetUniformDisplacement( space, unknown, 1, gap );
            unknown.GetBlock( 1 ) = lambda;
            const bool active = gap < 0.;
            const auto diagnostics = contact.ComputeContactDiagnostics( unknown );
            EXPECT_EQ( diagnostics.ActiveQuadraturePointCount, active ? diagnostics.QuadraturePointCount : 0 );
            mfem::Vector residual;
            contact.Mult( unknown, residual );
            mfem::Vector ru( residual.GetData(), space.GetTrueVSize() );
            mfem::Vector rl( residual.GetData() + space.GetTrueVSize(), lambda.Size() );
            const auto force = active ? -primalScale * ( 1. / 3. - gap / gamma ) : 0.;
            EXPECT_NEAR( normalTest * ru, force, kAssemblyTolerance );
            EXPECT_NEAR( constantTest * rl, multiplierScale * ( active ? -gap : -gamma / 3. ), kAssemblyTolerance );
            EXPECT_NEAR( linearTest * rl, multiplierScale * ( active ? -gap / 2. : -gamma / 4. ), kAssemblyTolerance );
            // Inactive lambda^T r_lambda = -scale*gamma*integral_0^1 x^4 dx.
            // Displacement-only order 2 uses two Gauss points and misses 1/5.
            // Leave the contact integration rule at its default throughout.
            EXPECT_NEAR( lambda * rl, multiplierScale * ( active ? -gap / 3. : -gamma / 5. ), kAssemblyTolerance );
            CheckMixedJacobianAndTranspose( contact, unknown );
        }
    }
}

TEST( SemismoothContact, HigherOrderPairCanRetainMultiplierNullMode )
{
    constexpr mfem::real_t gamma = .125, delta = 2.;
    mfem::Mesh mesh = mfem::Mesh::MakeCartesian2D( 2, 1, mfem::Element::QUADRILATERAL );
    mfem::H1_FECollection collection( 1, 2 );
    mfem::FiniteElementSpace space( &mesh, &collection, 2 );
    mfem::NonlinearForm primal( &space );
    mfem::IdentityOperator extraction( space.GetTrueVSize() );
    plugin::RigidPlaneObstacle obstacle( MakeVector( { 0., 0. } ), MakeVector( { 0., 1. } ) );
    plugin::BoundaryMultiplierSpace multipliers( mesh, ContactAttributes( 1 ), std::make_unique<mfem::L2_FECollection>( 4, 1 ) );
    mfem::Array<int> essential;
    plugin::SemismoothRigidContactOperator contact( primal, extraction, space, obstacle, multipliers, essential, gamma,
                                                    .17, 4.2, delta );
    ASSERT_EQ( multipliers.GetMesh().GetNE(), 2 );
    ASSERT_EQ( multipliers.GetSpace().GetTrueVSize(), 10 );
    mfem::ConstantCoefficient one( 1. );
    mfem::GridFunction lambda( &multipliers.GetSpace() ), bubble( &multipliers.GetSpace() );
    lambda.ProjectCoefficient( one );
    // Analytic null-mode construction on the left edge, s = 2*x in [0,1]:
    // b = s - 6*s^2 + 10*s^3 - 5*s^4, b(0) = b(1) = 0,
    // integral b ds = 1/2 - 2 + 5/2 - 1 = 0,
    // integral s*b ds = 1/3 - 3/2 + 2 - 5/6 = 0.
    // Thus b has zero jump and is orthogonal to every linear displacement
    // trace. With all points active the multiplier mass term is zero, too.
    mfem::FunctionCoefficient polynomial(
        []( const mfem::Vector& x ) -> mfem::real_t
        {
            if ( x( 0 ) > .5 )
            {
                return 0.;
            }
            const mfem::real_t s = 2. * x( 0 );
            const mfem::real_t t = s * ( 1. - s );
            return t * ( 1. - 5. * t );
        } );
    bubble.ProjectCoefficient( polynomial );
    mfem::BlockVector unknown( contact.GetBlockOffsets() ), direction( contact.GetBlockOffsets() );
    unknown = 0.;
    SetUniformDisplacement( space, unknown, 1, -.1 );
    unknown.GetBlock( 1 ) = lambda;
    direction = 0.;
    direction.GetBlock( 1 ) = bubble;
    ASSERT_GT( direction.Norml2(), .01 );
    direction /= direction.Norml2();
    const auto diagnostics = contact.ComputeContactDiagnostics( unknown );
    ASSERT_GT( diagnostics.QuadraturePointCount, 0 );
    ASSERT_EQ( diagnostics.ActiveQuadraturePointCount, diagnostics.QuadraturePointCount );
    mfem::Vector action( contact.Height() );
    auto& jacobian = contact.GetGradient( unknown );
    jacobian.Mult( direction, action );
    EXPECT_LE( action.Norml2(), kAssemblyTolerance );
    // Rule out a vacuous zero operator: a constant multiplier variation couples
    // to the normal displacement even though its stabilization jump is zero.
    direction.GetBlock( 1 ) = lambda;
    jacobian.Mult( direction, action );
    EXPECT_GT( action.Norml2(), .01 );
    // This is an experimental unstable pair despite delta > 0, not a solver
    // convergence test or a claim that configured higher orders are stable.
}

namespace
{
class CountingNonlinearPrimal final : public mfem::Operator
{
public:
    explicit CountingNonlinearPrimal( const int size ) : mfem::Operator( size )
    {
    }

    void Mult( const mfem::Vector& input, mfem::Vector& output ) const override
    {
        output.SetSize( Height() );
        for ( int i = 0; i < Height(); ++i )
        {
            output( i ) = input( i ) + input( i ) * input( i ) * input( i );
        }
    }

    mfem::Operator& GetGradient( const mfem::Vector& input ) const override
    {
        ++GradientCalls;
        auto gradient = std::make_unique<mfem::DenseMatrix>( Height() );
        *gradient = 0.;
        for ( int i = 0; i < Height(); ++i )
        {
            ( *gradient )( i, i ) = 1. + 3. * input( i ) * input( i );
        }
        // Keep every old object alive: a cached old pointer must yield a wrong
        // derivative deterministically, rather than a use-after-free accident.
        mGradients.push_back( std::move( gradient ) );
        return *mGradients.back();
    }

    mutable int GradientCalls = 0;

private:
    mutable std::vector<std::unique_ptr<mfem::DenseMatrix>> mGradients;
};
} // namespace

TEST( SemismoothContact, RefreshesReplacingPrimalGradientOnlyWhenRequested )
{
    constexpr mfem::real_t primalScale = 2.5;
    mfem::Mesh mesh = mfem::Mesh::MakeCartesian2D( 1, 1, mfem::Element::QUADRILATERAL );
    mfem::H1_FECollection collection( 1, mesh.Dimension() );
    mfem::FiniteElementSpace space( &mesh, &collection, mesh.Dimension(), mfem::Ordering::byVDIM );
    CountingNonlinearPrimal primal( space.GetTrueVSize() );
    mfem::IdentityOperator extraction( space.GetTrueVSize() );
    // All trial displacements are small relative to this strictly positive gap.
    const plugin::RigidPlaneObstacle obstacle( MakeVector( { 0., -10. } ), MakeVector( { 0., 1. } ) );
    const auto attributes = ContactAttributes( 1 );
    plugin::BoundaryMultiplierSpace multipliers( mesh, attributes );
    mfem::Array<int> essential;
    plugin::SemismoothRigidContactOperator contact( primal, extraction, space, obstacle, multipliers, essential, .125,
                                                    primalScale, 1., 0. );
    EXPECT_EQ( primal.GradientCalls, 0 );

    mfem::Vector direction( contact.Width() );
    direction = 0.;
    for ( int i = 0; i < primal.Width(); ++i )
    {
        direction( i ) = ( i % 2 == 0 ? 1. : -1. ) * ( 1. + .1 * i );
    }

    for ( int state = 0; state < 2; ++state )
    {
        SCOPED_TRACE( ::testing::Message() << "state=" << state );
        mfem::Vector unknown( contact.Width() ), expected( contact.Height() );
        unknown = 0.; // In particular lambda = 0, so contact stays inactive.
        expected = 0.;
        for ( int i = 0; i < primal.Width(); ++i )
        {
            unknown( i ) = ( state == 0 ? .02 : .07 ) * ( i + 1 );
            expected( i ) = primalScale * ( 1. + 3. * unknown( i ) * unknown( i ) ) * direction( i );
        }
        const auto diagnostics = contact.ComputeContactDiagnostics( unknown );
        ASSERT_GT( diagnostics.QuadraturePointCount, 0 );
        ASSERT_EQ( diagnostics.ActiveQuadraturePointCount, 0 );
        ASSERT_GT( diagnostics.MinimumGap, 9. );

        mfem::Vector residual( contact.Height() );
        contact.Mult( unknown, residual );
        EXPECT_EQ( primal.GradientCalls, state );
        // Obtain a fresh reference each time; no old Jacobian reference is used
        // after GetGradient, and no concrete wrapper/block type is assumed.
        auto& jacobian = contact.GetGradient( unknown );
        EXPECT_EQ( primal.GradientCalls, state + 1 );
        mfem::Vector action( contact.Height() ), transposeAction( contact.Width() );
        for ( int application = 0; application < 3; ++application )
        {
            action = 17.;
            transposeAction = -19.;
            jacobian.Mult( direction, action );
            jacobian.MultTranspose( direction, transposeAction );
            // The primal Jacobian is diagonal; both mixed blocks vanish on
            // the inactive branch and the multiplier direction is zero.
            ExpectVectorNear( action, expected, kAssemblyTolerance );
            ExpectVectorNear( transposeAction, expected, kAssemblyTolerance );
            EXPECT_EQ( primal.GradientCalls, state + 1 );
        }

        mfem::Vector otherTrial( unknown );
        for ( int i = 0; i < primal.Width(); ++i )
        {
            otherTrial( i ) *= .5;
        }
        contact.Mult( otherTrial, residual );
        EXPECT_EQ( primal.GradientCalls, state + 1 );
        // Residual evaluation must not refresh the previously requested tangent.
        // This test primal's Mult deliberately leaves its gradient objects intact.
        jacobian.Mult( direction, action );
        jacobian.MultTranspose( direction, transposeAction );
        ExpectVectorNear( action, expected, kAssemblyTolerance );
        ExpectVectorNear( transposeAction, expected, kAssemblyTolerance );
        EXPECT_EQ( primal.GradientCalls, state + 1 );
    }
}
