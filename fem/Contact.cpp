#include "Contact.h"
#include "typeDef.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace plugin
{
namespace
{
void VerifyDimension( const int dimension )
{
    MFEM_VERIFY( dimension == 2 || dimension == 3, "Rigid contact supports only two- and three-dimensional problems." );
}

void VerifyFiniteVector( const mfem::Vector& vector, const char* message )
{
    for ( int i = 0; i < vector.Size(); i++ )
    {
        MFEM_VERIFY( std::isfinite( vector( i ) ), message );
    }
}
} // namespace

SignedDistanceEvaluation::SignedDistanceEvaluation( const int dimension )
{
    SetSize( dimension );
}

void SignedDistanceEvaluation::SetSize( const int dimension )
{
    MFEM_VERIFY( dimension >= 0, "A signed-distance evaluation cannot have negative dimension." );
    if ( Gradient.Size() != dimension )
    {
        Gradient.SetSize( dimension );
    }
    if ( Hessian.Height() != dimension || Hessian.Width() != dimension )
    {
        Hessian.SetSize( dimension );
    }
}

RigidPlaneObstacle::RigidPlaneObstacle( const mfem::Vector& point, const mfem::Vector& admissibleNormal )
    : mPoint( point ), mNormal( admissibleNormal )
{
    VerifyDimension( mPoint.Size() );
    MFEM_VERIFY( mNormal.Size() == mPoint.Size(), "A rigid plane point and normal must have the same dimension." );
    VerifyFiniteVector( mPoint, "A rigid plane point must be finite." );
    VerifyFiniteVector( mNormal, "A rigid plane normal must be finite." );
    const mfem::real_t normalMagnitude = mNormal.Norml2();
    MFEM_VERIFY( std::isfinite( normalMagnitude ) && normalMagnitude > 0.,
                 "A rigid plane normal must have positive finite magnitude." );
    mNormal /= normalMagnitude;
}

int RigidPlaneObstacle::Dimension() const noexcept
{
    return mPoint.Size();
}

void RigidPlaneObstacle::Evaluate( const mfem::Vector& position, SignedDistanceEvaluation& evaluation ) const
{
    MFEM_VERIFY( position.Size() == Dimension(), "A rigid plane can only evaluate positions of its own dimension." );
    evaluation.SetSize( Dimension() );
    evaluation.Gap = 0.;
    for ( int component = 0; component < Dimension(); component++ )
    {
        evaluation.Gap += ( position( component ) - mPoint( component ) ) * mNormal( component );
    }
    evaluation.Gradient = mNormal;
    evaluation.Hessian = 0.;
}

RigidSphereObstacle::RigidSphereObstacle( const mfem::Vector& center, const mfem::real_t radius )
    : mCenter( center ), mRadius( radius )
{
    VerifyDimension( mCenter.Size() );
    VerifyFiniteVector( mCenter, "A rigid sphere center must be finite." );
    MFEM_VERIFY( std::isfinite( mRadius ) && mRadius > 0., "A rigid sphere radius must be positive and finite." );
}

int RigidSphereObstacle::Dimension() const noexcept
{
    return mCenter.Size();
}

void RigidSphereObstacle::Evaluate( const mfem::Vector& position, SignedDistanceEvaluation& evaluation ) const
{
    MFEM_VERIFY( position.Size() == Dimension(), "A rigid sphere can only evaluate positions of its own dimension." );
    evaluation.SetSize( Dimension() );
    evaluation.Gradient = position;
    evaluation.Gradient -= mCenter;
    const mfem::real_t distance = evaluation.Gradient.Norml2();
    MFEM_VERIFY( std::isfinite( distance ) && distance > 0.,
                 "The signed distance to a circle or sphere is not differentiable at its center." );

    evaluation.Gap = distance - mRadius;
    evaluation.Gradient /= distance;
    for ( int row = 0; row < Dimension(); row++ )
    {
        for ( int column = 0; column < Dimension(); column++ )
        {
            const mfem::real_t identity = row == column ? 1. : 0.;
            evaluation.Hessian( row, column ) = ( identity - evaluation.Gradient( row ) * evaluation.Gradient( column ) ) / distance;
        }
    }
}

FrictionlessPenaltyContactIntegrator::FrictionlessPenaltyContactIntegrator( const RigidObstacle& obstacle, const mfem::real_t penalty )
    : mObstacle( obstacle ), mPenalty( penalty ), mEvaluation( obstacle.Dimension() )
{
    VerifyDimension( mObstacle.Dimension() );
    MFEM_VERIFY( std::isfinite( mPenalty ) && mPenalty > 0.,
                 "The normal contact penalty must be positive and finite." );
}

void FrictionlessPenaltyContactIntegrator::AssembleFaceVector( const mfem::FiniteElement& element1,
                                                               const mfem::FiniteElement& element2,
                                                               mfem::FaceElementTransformations& transformation,
                                                               const mfem::Vector& elementDisplacement,
                                                               mfem::Vector& elementResidual )
{
    (void)element2;
    VerifyFaceInput( element1, transformation, elementDisplacement );
    const int degreeOfFreedomCount = element1.GetDof();
    const int dimension = transformation.GetSpaceDim();
    SetScratchSize( degreeOfFreedomCount, dimension );
    elementResidual.SetSize( degreeOfFreedomCount * dimension );
    elementResidual = 0.;
    Eigen::Map<Eigen::MatrixXr> residual( elementResidual.GetData(), degreeOfFreedomCount, dimension );

    const mfem::IntegrationRule& integrationRule = SelectIntegrationRule( element1, transformation );
    for ( int point = 0; point < integrationRule.GetNPoints(); point++ )
    {
        const mfem::IntegrationPoint& integrationPoint = integrationRule.IntPoint( point );
        EvaluateContactPoint( element1, transformation, integrationPoint, elementDisplacement );
        if ( mEvaluation.Gap >= 0. )
        {
            continue;
        }

        const mfem::real_t scale = integrationPoint.weight * transformation.Weight() * mPenalty * mEvaluation.Gap;
        const Eigen::Map<const Eigen::VectorXr> shape( mShape.GetData(), degreeOfFreedomCount );
        const Eigen::Map<const Eigen::VectorXr> normal( mEvaluation.Gradient.GetData(), dimension );
        residual.noalias() += scale * shape * normal.transpose();
    }
}

void FrictionlessPenaltyContactIntegrator::AssembleFaceGrad( const mfem::FiniteElement& element1,
                                                             const mfem::FiniteElement& element2,
                                                             mfem::FaceElementTransformations& transformation,
                                                             const mfem::Vector& elementDisplacement,
                                                             mfem::DenseMatrix& elementTangent )
{
    (void)element2;
    VerifyFaceInput( element1, transformation, elementDisplacement );
    const int degreeOfFreedomCount = element1.GetDof();
    const int dimension = transformation.GetSpaceDim();
    SetScratchSize( degreeOfFreedomCount, dimension );
    elementTangent.SetSize( degreeOfFreedomCount * dimension );
    elementTangent = 0.;
    Eigen::Map<Eigen::MatrixXr> tangent( elementTangent.GetData(), degreeOfFreedomCount * dimension, degreeOfFreedomCount * dimension );

    const mfem::IntegrationRule& integrationRule = SelectIntegrationRule( element1, transformation );
    for ( int point = 0; point < integrationRule.GetNPoints(); point++ )
    {
        const mfem::IntegrationPoint& integrationPoint = integrationRule.IntPoint( point );
        EvaluateContactPoint( element1, transformation, integrationPoint, elementDisplacement );
        if ( mEvaluation.Gap > 0. )
        {
            continue;
        }

        const mfem::real_t scale = integrationPoint.weight * transformation.Weight() * mPenalty;
        const Eigen::Map<const Eigen::VectorXr> shape( mShape.GetData(), degreeOfFreedomCount );
        for ( int testComponent = 0; testComponent < dimension; testComponent++ )
        {
            for ( int trialComponent = 0; trialComponent < dimension; trialComponent++ )
            {
                const mfem::real_t spatialTangent = mEvaluation.Gradient( testComponent ) * mEvaluation.Gradient( trialComponent ) +
                                                    mEvaluation.Gap * mEvaluation.Hessian( testComponent, trialComponent );
                auto tangentBlock = tangent.block( testComponent * degreeOfFreedomCount, trialComponent * degreeOfFreedomCount,
                                                   degreeOfFreedomCount, degreeOfFreedomCount );
                tangentBlock.noalias() += ( scale * spatialTangent ) * shape * shape.transpose();
            }
        }
    }
}

void FrictionlessPenaltyContactIntegrator::VerifyFaceInput( const mfem::FiniteElement& element,
                                                            const mfem::FaceElementTransformations& transformation,
                                                            const mfem::Vector& elementDisplacement ) const
{
    const int dimension = transformation.GetSpaceDim();
    VerifyDimension( dimension );
    MFEM_VERIFY( transformation.Elem2No < 0, "Rigid-obstacle contact must be assembled on an exterior boundary face." );
    MFEM_VERIFY( transformation.Elem1 != nullptr, "Rigid-obstacle contact requires an adjacent volume element." );
    MFEM_VERIFY( element.GetDim() == dimension,
                 "Rigid-obstacle contact does not support embedded or reduced-dimensional elements." );
    MFEM_VERIFY( element.GetRangeType() == mfem::FiniteElement::SCALAR && element.GetMapType() == mfem::FiniteElement::VALUE,
                 "Rigid-obstacle contact requires a value-mapped scalar element replicated by spatial dimension." );
    MFEM_VERIFY( mObstacle.Dimension() == dimension,
                 "The rigid obstacle and displacement field dimensions must match." );
    MFEM_VERIFY( elementDisplacement.Size() == element.GetDof() * dimension,
                 "The contact element vector must contain one displacement value per component and scalar DOF." );
    VerifyFiniteVector( elementDisplacement, "The contact element displacement must be finite." );
}

const mfem::IntegrationRule& FrictionlessPenaltyContactIntegrator::SelectIntegrationRule(
    const mfem::FiniteElement& element, const mfem::FaceElementTransformations& transformation ) const
{
    if ( IntRule != nullptr )
    {
        return *IntRule;
    }

    int order = transformation.Elem1->OrderW() + 2 * element.GetOrder();
    if ( element.Space() == mfem::FunctionSpace::Pk )
    {
        order++;
    }
    return mfem::IntRules.Get( transformation.GetGeometryType(), order );
}

void FrictionlessPenaltyContactIntegrator::SetScratchSize( const int degreeOfFreedomCount, const int dimension )
{
    if ( mShape.Size() != degreeOfFreedomCount )
    {
        mShape.SetSize( degreeOfFreedomCount );
    }
    if ( mReferencePosition.Size() != dimension || mCurrentPosition.Size() != dimension )
    {
        mReferencePosition.SetSize( dimension );
        mCurrentPosition.SetSize( dimension );
    }
}

void FrictionlessPenaltyContactIntegrator::EvaluateContactPoint( const mfem::FiniteElement& element,
                                                                 mfem::FaceElementTransformations& transformation,
                                                                 const mfem::IntegrationPoint& integrationPoint,
                                                                 const mfem::Vector& elementDisplacement )
{
    const int degreeOfFreedomCount = element.GetDof();
    const int dimension = transformation.GetSpaceDim();

    transformation.SetAllIntPoints( &integrationPoint );
    element.CalcShape( transformation.GetElement1IntPoint(), mShape );
    transformation.Transform( integrationPoint, mReferencePosition );
    const Eigen::Map<const Eigen::MatrixXr> displacement( elementDisplacement.GetData(), degreeOfFreedomCount, dimension );
    const Eigen::Map<const Eigen::VectorXr> shape( mShape.GetData(), degreeOfFreedomCount );
    Eigen::Map<Eigen::VectorXr> currentPosition( mCurrentPosition.GetData(), dimension );
    const Eigen::Map<const Eigen::VectorXr> referencePosition( mReferencePosition.GetData(), dimension );
    currentPosition = referencePosition;
    currentPosition.noalias() += displacement.transpose() * shape;

    mObstacle.Evaluate( mCurrentPosition, mEvaluation );
    VerifySignedDistanceEvaluation();
}

void FrictionlessPenaltyContactIntegrator::VerifySignedDistanceEvaluation() const
{
    const int dimension = mObstacle.Dimension();
    MFEM_VERIFY( mEvaluation.Gradient.Size() == dimension && mEvaluation.Hessian.Height() == dimension &&
                     mEvaluation.Hessian.Width() == dimension,
                 "A rigid obstacle returned derivatives with the wrong dimension." );
    MFEM_VERIFY( std::isfinite( mEvaluation.Gap ), "A rigid obstacle returned a non-finite gap." );
    VerifyFiniteVector( mEvaluation.Gradient, "A rigid obstacle returned a non-finite signed-distance gradient." );

    mfem::real_t hessianScale = 1.;
    for ( int row = 0; row < dimension; row++ )
    {
        for ( int column = 0; column < dimension; column++ )
        {
            MFEM_VERIFY( std::isfinite( mEvaluation.Hessian( row, column ) ),
                         "A rigid obstacle returned a non-finite signed-distance Hessian." );
            hessianScale = std::max( hessianScale, std::abs( mEvaluation.Hessian( row, column ) ) );
        }
    }

    const mfem::real_t tolerance = 1000. * std::numeric_limits<mfem::real_t>::epsilon();
    MFEM_VERIFY( std::abs( mEvaluation.Gradient.Norml2() - 1. ) <= tolerance,
                 "A rigid obstacle gradient must be a unit signed-distance gradient." );
    mfem::real_t normalCurvatureSquared = 0.;
    for ( int row = 0; row < dimension; row++ )
    {
        mfem::real_t normalCurvature = 0.;
        for ( int column = 0; column < dimension; column++ )
        {
            MFEM_VERIFY( std::abs( mEvaluation.Hessian( row, column ) - mEvaluation.Hessian( column, row ) ) <= tolerance * hessianScale,
                         "A rigid obstacle signed-distance Hessian must be symmetric." );
            normalCurvature += mEvaluation.Hessian( row, column ) * mEvaluation.Gradient( column );
        }
        normalCurvatureSquared += normalCurvature * normalCurvature;
    }
    MFEM_VERIFY( std::sqrt( normalCurvatureSquared ) <= tolerance * hessianScale,
                 "A rigid obstacle Hessian must annihilate its signed-distance gradient." );
}
} // namespace plugin
