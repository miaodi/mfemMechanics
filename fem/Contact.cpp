#include "Contact.h"
#include "OperatorSum.h"
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

void VerifySignedDistanceEvaluation( const RigidObstacle& obstacle, const SignedDistanceEvaluation& evaluation )
{
    const int dimension = obstacle.Dimension();
    MFEM_VERIFY( evaluation.Gradient.Size() == dimension && evaluation.Hessian.Height() == dimension &&
                     evaluation.Hessian.Width() == dimension,
                 "A rigid obstacle returned derivatives with the wrong dimension." );
    MFEM_VERIFY( std::isfinite( evaluation.Gap ), "A rigid obstacle returned a non-finite gap." );
    VerifyFiniteVector( evaluation.Gradient, "A rigid obstacle returned a non-finite signed-distance gradient." );

    mfem::real_t hessianScale = 1.;
    for ( int row = 0; row < dimension; row++ )
    {
        for ( int column = 0; column < dimension; column++ )
        {
            MFEM_VERIFY( std::isfinite( evaluation.Hessian( row, column ) ),
                         "A rigid obstacle returned a non-finite signed-distance Hessian." );
            hessianScale = std::max( hessianScale, std::abs( evaluation.Hessian( row, column ) ) );
        }
    }

    const mfem::real_t tolerance = 1000. * std::numeric_limits<mfem::real_t>::epsilon();
    MFEM_VERIFY( std::abs( evaluation.Gradient.Norml2() - 1. ) <= tolerance,
                 "A rigid obstacle gradient must be a unit signed-distance gradient." );
    mfem::real_t normalCurvatureSquared = 0.;
    for ( int row = 0; row < dimension; row++ )
    {
        mfem::real_t normalCurvature = 0.;
        for ( int column = 0; column < dimension; column++ )
        {
            MFEM_VERIFY( std::abs( evaluation.Hessian( row, column ) - evaluation.Hessian( column, row ) ) <= tolerance * hessianScale,
                         "A rigid obstacle signed-distance Hessian must be symmetric." );
            normalCurvature += evaluation.Hessian( row, column ) * evaluation.Gradient( column );
        }
        normalCurvatureSquared += normalCurvature * normalCurvature;
    }
    MFEM_VERIFY( std::sqrt( normalCurvatureSquared ) <= tolerance * hessianScale,
                 "A rigid obstacle Hessian must annihilate its signed-distance gradient." );
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

void FrictionlessPenaltyContactIntegrator::AssembleElementVector( const mfem::FiniteElement& element1,
                                                                  mfem::ElementTransformation& transformation,
                                                                  const mfem::Vector& elementDisplacement,
                                                                  mfem::Vector& elementResidual )
{
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

void FrictionlessPenaltyContactIntegrator::AssembleElementGrad( const mfem::FiniteElement& element1,
                                                                mfem::ElementTransformation& transformation,
                                                                const mfem::Vector& elementDisplacement,
                                                                mfem::DenseMatrix& elementTangent )
{
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

mfem::real_t FrictionlessPenaltyContactIntegrator::GetElementEnergy( const mfem::FiniteElement& element,
                                                                     mfem::ElementTransformation& transformation,
                                                                     const mfem::Vector& displacement )
{
    VerifyFaceInput( element, transformation, displacement );
    SetScratchSize( element.GetDof(), transformation.GetSpaceDim() );
    const auto& rule = SelectIntegrationRule( element, transformation );
    mfem::real_t energy = 0.;
    for ( int q = 0; q < rule.GetNPoints(); ++q )
    {
        const auto& ip = rule.IntPoint( q );
        EvaluateContactPoint( element, transformation, ip, displacement );
        const auto gap = std::min( mfem::real_t{ 0. }, mEvaluation.Gap );
        energy += .5 * mPenalty * gap * gap * ip.weight * transformation.Weight();
    }
    return energy;
}

void FrictionlessPenaltyContactIntegrator::VerifyFaceInput( const mfem::FiniteElement& element,
                                                            const mfem::ElementTransformation& transformation,
                                                            const mfem::Vector& displacement ) const
{
    const int dimension = transformation.GetSpaceDim();
    VerifyDimension( dimension );
    MFEM_VERIFY(
        transformation.mesh != nullptr && transformation.ElementType == mfem::ElementTransformation::BDR_ELEMENT,
        "Penalty boundary assembly requires a mesh-owned boundary element transformation; use AddBoundaryIntegrator." );
    MFEM_VERIFY( transformation.ElementNo >= 0 && transformation.ElementNo < transformation.mesh->GetNBE(),
                 "The penalty boundary transformation has an invalid boundary element index." );
    int adjacent, second;
    transformation.mesh->GetFaceElements( transformation.mesh->GetBdrElementFaceIndex( transformation.ElementNo ),
                                          &adjacent, &second );
    MFEM_VERIFY( adjacent >= 0 && second < 0,
                 "Rigid-obstacle contact must be assembled on an exterior boundary face." );
    MFEM_VERIFY( transformation.mesh->Dimension() == dimension && element.GetDim() == dimension - 1,
                 "Penalty boundary assembly requires a codimension-one trace of a full-dimensional mesh." );
    MFEM_VERIFY( element.GetRangeType() == mfem::FiniteElement::SCALAR && element.GetMapType() == mfem::FiniteElement::VALUE,
                 "Penalty boundary assembly requires value-mapped scalar H1 traces." );
    MFEM_VERIFY( mObstacle.Dimension() == dimension && displacement.Size() == element.GetDof() * dimension,
                 "The obstacle and boundary displacement dimensions must match." );
    VerifyFiniteVector( displacement, "The contact boundary displacement must be finite." );
}

const mfem::IntegrationRule& FrictionlessPenaltyContactIntegrator::SelectIntegrationRule( const mfem::FiniteElement& element,
                                                                                          const mfem::ElementTransformation& transformation )
{
    const auto& mesh = *transformation.mesh;
    int adjacent, info, face, orientation;
    mesh.GetBdrElementAdjacentElement( transformation.ElementNo, adjacent, info );
    mesh.GetBdrElementFace( transformation.ElementNo, &face, &orientation );
    const auto geometry = element.GetGeomType();
    const auto* rule = IntRule;
    if ( rule == nullptr )
    {
        // Preserve the shipped volume-face rule, including the volume measure
        // order and Pk increment. A triangular wedge trace is NOT a Pk volume.
        mesh.GetElementTransformation( adjacent, &mVolumeTransformation );
        const auto volumeGeometry = mesh.GetElementBaseGeometry( adjacent );
        const bool pk = volumeGeometry == mfem::Geometry::TRIANGLE || volumeGeometry == mfem::Geometry::TETRAHEDRON;
        const int order = mVolumeTransformation.OrderW() + 2 * element.GetOrder() + ( pk ? 1 : 0 );
        rule = &mfem::IntRules.Get( geometry, order );
    }
    mBoundaryRule.SetSize( rule->GetNPoints() );
    const int inverse = mfem::Geometry::GetInverseOrientation( geometry, orientation );
    for ( int q = 0; q < rule->GetNPoints(); ++q )
    {
        mBoundaryRule.IntPoint( q ) = mfem::Mesh::TransformBdrElementToFace( geometry, inverse, rule->IntPoint( q ) );
        mBoundaryRule.IntPoint( q ).weight = rule->IntPoint( q ).weight;
    }
    return mBoundaryRule;
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
                                                                 mfem::ElementTransformation& transformation,
                                                                 const mfem::IntegrationPoint& integrationPoint,
                                                                 const mfem::Vector& displacement )
{
    transformation.SetIntPoint( &integrationPoint );
    element.CalcShape( integrationPoint, mShape );
    transformation.Transform( integrationPoint, mReferencePosition );
    EvaluateDisplacedPoint( element.GetDof(), transformation.GetSpaceDim(), displacement );
}

void FrictionlessPenaltyContactIntegrator::EvaluateDisplacedPoint( const int degreeOfFreedomCount,
                                                                   const int dimension,
                                                                   const mfem::Vector& elementDisplacement )
{
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
    plugin::VerifySignedDistanceEvaluation( mObstacle, mEvaluation );
}

namespace
{
int DecodeDof( const int dof )
{
    return dof >= 0 ? dof : -1 - dof;
}
} // namespace

SemismoothRigidContactOperator::SemismoothRigidContactOperator( mfem::Operator& primalOperator,
                                                                const mfem::Operator& primalToDisplacement,
                                                                mfem::FiniteElementSpace& displacementSpace,
                                                                const RigidObstacle& obstacle,
                                                                BoundaryMultiplierSpace& boundaryMultiplier,
                                                                const mfem::Array<int>& essentialDisplacementTrueDofs,
                                                                const mfem::real_t gamma,
                                                                const mfem::real_t primalResidualScale,
                                                                const mfem::real_t multiplierResidualScale,
                                                                const mfem::real_t delta )
    : mfem::Operator( 0 ),
      mPrimalOperator( primalOperator ),
      mPrimalToDisplacement( primalToDisplacement ),
      mDisplacementSpace( displacementSpace ),
      mObstacle( obstacle ),
      mBoundaryMultiplier( boundaryMultiplier ),
      mGamma( gamma ),
      mPrimalResidualScale( primalResidualScale ),
      mMultiplierResidualScale( multiplierResidualScale ),
      mDelta( delta ),
      mContactSubMesh( boundaryMultiplier.GetMesh() ),
      mMultiplierSpace( boundaryMultiplier.GetSpace() ),
      mEvaluation( obstacle.Dimension() )
{
    VerifyInput( essentialDisplacementTrueDofs );
    VerifyContactTraceHasFreeDisplacementDofs();
    BuildIntegrationRules( nullptr );
    BuildMultiplierJumpStabilization();
    // Allocate zero-valued matrices; GetGradient() fills their entries in place.
    BuildSparsity();
    mDisplacement.SetSize( mDisplacementSpace.GetTrueVSize() );
    mPrimalContactResidual.SetSize( mPrimalOperator.Height() );

    // Lazy E^T*K_uu*E, where E extracts displacement; RAPOperator takes R^T, A, P.
    mDisplacementContactJacobian =
        std::make_unique<mfem::RAPOperator>( mPrimalToDisplacement, *mDisplacementJacobian, mPrimalToDisplacement );
    mPrimalDisplacementCoupling = std::make_unique<mfem::TransposeOperator>( mPrimalToDisplacement );
    // Lazy mixed blocks: E^T*K_uL and K_Lu*E.
    mPrimalMultiplierCoupling = std::make_unique<mfem::ProductOperator>(
        mPrimalDisplacementCoupling.get(), mDisplacementMultiplierJacobian.get(), false, false );
    mMultiplierPrimalCoupling =
        std::make_unique<mfem::ProductOperator>( mMultiplierDisplacementJacobian.get(), &mPrimalToDisplacement, false, false );

    mBlockOffsets.SetSize( 3 );
    mBlockOffsets[0] = 0;
    mBlockOffsets[1] = mPrimalOperator.Width();
    mBlockOffsets[2] = mBlockOffsets[1] + mMultiplierSpace.GetTrueVSize();
    mJacobian = std::make_unique<mfem::BlockOperator>( mBlockOffsets );
    // GetGradient() installs the primal-plus-contact sum in block (0,0).
    mJacobian->SetBlock( 0, 1, mPrimalMultiplierCoupling.get() );
    mJacobian->SetBlock( 1, 0, mMultiplierPrimalCoupling.get() );
    mJacobian->SetBlock( 1, 1, mMultiplierJacobian.get() );
    height = width = mBlockOffsets.Last();
}

SemismoothRigidContactOperator::~SemismoothRigidContactOperator() = default;

// Evaluate F([y,lambda]) = [R_y,R_lambda]; do not solve or update the unknowns.
// Here y contains the primal fields and u = E*y, E = mPrimalToDisplacement.
// For displacement-only primal physics with matching DOFs, y = u and E = I.
void SemismoothRigidContactOperator::Mult( const mfem::Vector& unknown, mfem::Vector& residual ) const
{
    VerifyUnknown( unknown );
    // Borrow the two input blocks without copying. MFEM's view constructor
    // requires mutable pointers, but these views are only read below.
    mfem::Vector primal( const_cast<mfem::real_t*>( unknown.GetData() ), mPrimalOperator.Width() );
    mfem::Vector multiplier( const_cast<mfem::real_t*>( unknown.GetData() ) + mBlockOffsets[1], mMultiplierSpace.GetTrueVSize() );

    // Writable block views: assembly fills the caller's residual directly.
    residual.SetSize( Height() );
    mfem::Vector primalResidual( residual.GetData(), mPrimalOperator.Height() );
    mfem::Vector multiplierResidual( residual.GetData() + mBlockOffsets[1], mMultiplierSpace.GetTrueVSize() );

    // Start with the scaled bulk/primal residual, without this contact term.
    mPrimalOperator.Mult( primal, primalResidual );
    primalResidual *= mPrimalResidualScale;

    // Extract u, then evaluate contact at the same current (u,lambda).
    // AssembleContact fills the displacement contact residual and the full
    // multiplier residual (complementarity minus jump stabilization), with
    // each row already scaled. false requests residuals only, not a Jacobian.
    mPrimalToDisplacement.Mult( primal, mDisplacement );
    AssembleContact( mDisplacement, multiplier, &mDisplacementResidual, &multiplierResidual, false );

    // Pull contact forces back to primal coordinates:
    // R_y = s_u*R_primal(y) + E^T*R_contact(u,lambda), with R_contact scaled.
    // R_lambda was written directly into the second output block above.
    mPrimalToDisplacement.MultTranspose( mDisplacementResidual, mPrimalContactResidual );
    primalResidual += mPrimalContactResidual;
}

mfem::Operator& SemismoothRigidContactOperator::GetGradient( const mfem::Vector& unknown ) const
{
    using operator_algebra::op;

    VerifyUnknown( unknown );
    mfem::Vector primal( const_cast<mfem::real_t*>( unknown.GetData() ), mPrimalOperator.Width() );
    mfem::Vector multiplier( const_cast<mfem::real_t*>( unknown.GetData() ) + mBlockOffsets[1], mMultiplierSpace.GetTrueVSize() );
    mPrimalToDisplacement.Mult( primal, mDisplacement );
    AssembleContact( mDisplacement, multiplier, nullptr, nullptr, true );
    // The primal form may replace its gradient object on each evaluation.
    // Contact is already row-scaled; form a complete sum for this linearization.
    auto& primalGradient = mPrimalOperator.GetGradient( primal );
    mPrimalJacobian = mPrimalResidualScale * op( primalGradient ) + op( *mDisplacementContactJacobian );
    mJacobian->SetBlock( 0, 0, mPrimalJacobian.get() );
    return *mJacobian;
}

void SemismoothRigidContactOperator::SetIntegrationRule( const mfem::IntegrationRule* integrationRule )
{
    BuildIntegrationRules( integrationRule );
}

void SemismoothRigidContactOperator::GetMultiplier( const mfem::Vector& unknown, mfem::Vector& multiplier ) const
{
    VerifyUnknown( unknown );
    multiplier.SetSize( mMultiplierSpace.GetTrueVSize() );
    for ( int i = 0; i < multiplier.Size(); i++ )
    {
        multiplier( i ) = unknown( mBlockOffsets[1] + i );
    }
}

void SemismoothRigidContactOperator::VerifyInput( const mfem::Array<int>& essentialDisplacementTrueDofs )
{
    mfem::Mesh* mesh = mDisplacementSpace.GetMesh();
    MFEM_VERIFY( mesh != nullptr, "Semismooth contact requires a displacement mesh." );
    MFEM_VERIFY( mfem::Device::IsDisabled(), "Semismooth contact currently supports only MFEM's default CPU backend." );
    MFEM_VERIFY( !mesh->Nonconforming(), "Semismooth contact currently requires a conforming displacement mesh." );
#ifdef MFEM_USE_MPI
    MFEM_VERIFY( dynamic_cast<mfem::ParFiniteElementSpace*>( &mDisplacementSpace ) == nullptr,
                 "Semismooth contact currently supports only serial finite-element spaces." );
#endif
    MFEM_VERIFY( mContactSubMesh.GetParent() == mesh,
                 "The boundary multiplier parent must be the exact displacement mesh." );
    const int dimension = mesh->Dimension();
    VerifyDimension( dimension );
    MFEM_VERIFY( mesh->SpaceDimension() == dimension,
                 "Semismooth contact does not support embedded displacement meshes." );
    MFEM_VERIFY( mObstacle.Dimension() == dimension,
                 "The rigid obstacle and displacement field dimensions must match." );
    MFEM_VERIFY( mDisplacementSpace.GetVDim() == dimension,
                 "Semismooth contact requires one displacement component per spatial dimension." );
    MFEM_VERIFY( mDisplacementSpace.GetTrueVSize() == mDisplacementSpace.GetVSize(),
                 "Semismooth contact currently requires a conforming serial displacement space without constraints." );
    MFEM_VERIFY( mPrimalOperator.Height() == mPrimalOperator.Width(), "The primal nonlinear operator must be square." );
    MFEM_VERIFY( mPrimalToDisplacement.Width() == mPrimalOperator.Width() &&
                     mPrimalToDisplacement.Height() == mDisplacementSpace.GetTrueVSize(),
                 "The primal-to-displacement operator has incompatible dimensions." );
    MFEM_VERIFY( std::isfinite( mGamma ) && mGamma > 0.,
                 "The semismooth contact compliance gamma must be positive and finite." );
    MFEM_VERIFY( std::isfinite( mfem::real_t{ 1. } / mGamma ),
                 "The semismooth contact compliance gamma must have a finite reciprocal." );
    MFEM_VERIFY( std::isfinite( mPrimalResidualScale ) && mPrimalResidualScale > 0. &&
                     std::isfinite( mMultiplierResidualScale ) && mMultiplierResidualScale > 0.,
                 "Semismooth contact residual scales must be positive and finite." );
    MFEM_VERIFY( std::isfinite( mDelta ) && mDelta >= 0.,
                 "Semismooth contact multiplier jump weight delta must be finite and nonnegative." );
    MFEM_VERIFY( mContactSubMesh.GetNE() > 0, "The selected contact boundary is empty." );
    MFEM_VERIFY( mMultiplierSpace.GetTrueVSize() == mMultiplierSpace.GetVSize(),
                 "Semismooth contact currently requires an unconstrained boundary L2 multiplier space." );

    mEssentialMarker.SetSize( mDisplacementSpace.GetTrueVSize() );
    mEssentialMarker = 0;
    for ( int i = 0; i < essentialDisplacementTrueDofs.Size(); i++ )
    {
        const int dof = essentialDisplacementTrueDofs[i];
        MFEM_VERIFY( dof >= 0 && dof < mEssentialMarker.Size(),
                     "An essential displacement true DOF is outside the displacement space." );
        mEssentialMarker[dof] = 1;
    }

    const auto* primalForm = dynamic_cast<const mfem::NonlinearForm*>( &mPrimalOperator );
    if ( primalForm != nullptr && dynamic_cast<const mfem::IdentityOperator*>( &mPrimalToDisplacement ) != nullptr )
    {
        mfem::Array<int> primalEssentialMarker( mEssentialMarker.Size() );
        primalEssentialMarker = 0;
        for ( const int dof : primalForm->GetEssentialTrueDofs() )
        {
            MFEM_VERIFY( dof >= 0 && dof < primalEssentialMarker.Size(),
                         "The primal form contains an invalid essential true DOF." );
            primalEssentialMarker[dof] = 1;
        }
        for ( int dof = 0; dof < mEssentialMarker.Size(); dof++ )
        {
            MFEM_VERIFY( primalEssentialMarker[dof] == mEssentialMarker[dof],
                         "The contact and primal form essential displacement true DOFs must match." );
        }
    }
}

void SemismoothRigidContactOperator::VerifyContactTraceHasFreeDisplacementDofs()
{
    mfem::Array<int> traceDofs;
    for ( int contactElement = 0; contactElement < mContactSubMesh.GetNE(); contactElement++ )
    {
        mfem::DofTransformation* transformation =
            mDisplacementSpace.GetBdrElementVDofs( mBoundaryMultiplier.GetParentBoundaryElement( contactElement ), traceDofs );
        MFEM_VERIFY( transformation == nullptr,
                     "Semismooth contact currently supports boundary traces without DOF transformations." );
        const bool hasFreeDof = std::any_of( traceDofs.begin(), traceDofs.end(), [this]( const int dof )
                                             { return mEssentialMarker[DecodeDof( dof )] == 0; } );
        MFEM_VERIFY( hasFreeDof, "Every semismooth contact face needs at least one free displacement trace DOF." );
    }
}

void SemismoothRigidContactOperator::BuildMultiplierJumpStabilization()
{
    // Build and cache the multiplier-jump bilinear form, adapted from preprint
    // HTML Section 2, (9) (not journal numbering; full citation in Contact.h):
    // s(lambda,mu) = sum_F delta*gamma*h_F * integral_F [lambda]*[mu] ds.
    // F is an interior interface of the reference contact submesh, counted
    // once, not a volume interface; [lambda] = lambda_e - lambda_f, with
    // the same orientation for [mu]. Evaluate both neighboring multiplier bases
    // at the same physical point: J = [M_e^T, -M_f^T],
    // S_F = delta*gamma*h_F * integral_F J^T*J ds.
    // Cached DOF lists use the same concatenated element ordering as J.
    // Our h_F averages measure(e)^(1/d_c) and measure(f)^(1/d_c), d_c = d-1.
    // measure(F) is 1 at a vertex in 2D, or integrated edge length in 3D.
    // Thus s(lambda,mu) = mu^T S lambda with S positive semidefinite and
    // constants unpenalized. The coefficients are basis coefficients, not
    // necessarily nodal values (MFEM's positive/Bernstein basis is supported).
    mMultiplierJumps.clear();
    for ( int face = 0; face < mContactSubMesh.GetNumFaces(); face++ )
    {
        int element1 = -1;
        int element2 = -1;
        mContactSubMesh.GetFaceElements( face, &element1, &element2 );
        if ( element2 < 0 )
        {
            continue;
        }

        const mfem::real_t inverseDimension = 1. / static_cast<mfem::real_t>( mContactSubMesh.Dimension() );
        const mfem::real_t elementSize = .5 * ( std::pow( mContactSubMesh.GetElementVolume( element1 ), inverseDimension ) +
                                                std::pow( mContactSubMesh.GetElementVolume( element2 ), inverseDimension ) );
        MFEM_VERIFY( std::isfinite( elementSize ) && elementSize > 0.,
                     "Semismooth contact multiplier stabilization requires positive finite mesh measures." );
        const mfem::real_t coefficient = mDelta * mGamma * elementSize;
        MFEM_VERIFY( std::isfinite( coefficient ), "The multiplier stabilization coefficient must be finite." );

        mfem::Array<int> dofs1, dofs2;
        mBoundaryMultiplier.GetElementDofs( element1, dofs1 );
        mBoundaryMultiplier.GetElementDofs( element2, dofs2 );
        const auto& finiteElement1 = *mMultiplierSpace.GetFE( element1 );
        const auto& finiteElement2 = *mMultiplierSpace.GetFE( element2 );
        mMultiplierJumps.emplace_back();
        auto& jump = mMultiplierJumps.back();
        jump.Dofs = dofs1;
        jump.Dofs.Append( dofs2 );
        jump.Matrix.SetSize( jump.Dofs.Size() );
        jump.Matrix = 0.;
        mfem::Vector shape1( dofs1.Size() ), shape2( dofs2.Size() ), jumpShape( jump.Dofs.Size() );
        Eigen::Map<Eigen::MatrixXr> matrix( jump.Matrix.GetData(), jump.Dofs.Size(), jump.Dofs.Size() );
        const Eigen::Map<const Eigen::VectorXr> shape( jumpShape.GetData(), jump.Dofs.Size() );

        // GetElementVolume above uses mesh transformation scratch; acquire the
        // interface transformations afterwards and retain them for this loop.
        auto* transformation = mContactSubMesh.GetFaceElementTransformations( face );
        MFEM_VERIFY( transformation != nullptr && transformation->Elem1No == element1 && transformation->Elem2No == element2,
                     "Multiplier jump stabilization requires an interior contact-submesh face." );
        const bool vertexInterface = mContactSubMesh.Dimension() == 1;
        const int geometryOrder =
            vertexInterface ? 0
                            : std::max( transformation->OrderW(),
                                        std::min( transformation->Elem1->OrderW(), transformation->Elem2->OrderW() ) );
        // Account for both interface geometry and multiplier basis products.
        // Curved metrics are nonpolynomial; check sensitivity to quadrature order.
        const int order = 2 * geometryOrder + 2 + 2 * std::max( finiteElement1.GetOrder(), finiteElement2.GetOrder() );
        const auto& rule = mfem::IntRules.Get( transformation->GetGeometryType(), order );
        mfem::real_t measure = 0.;
        for ( int point = 0; point < rule.GetNPoints(); ++point )
        {
            const auto& integrationPoint = rule.IntPoint( point );
            transformation->SetAllIntPoints( &integrationPoint );
            // MFEM mesh.cpp:GetFaceElementTransformations constructs Loc1/Loc2
            // using each neighbor's local-face orientation. These are SUBMESH
            // element points at the same interface position, so evaluate the
            // local FEs directly, not the parent-boundary CalcShape API.
            finiteElement1.CalcShape( transformation->GetElement1IntPoint(), shape1 );
            finiteElement2.CalcShape( transformation->GetElement2IntPoint(), shape2 );
            for ( int i = 0; i < dofs1.Size(); ++i )
            {
                jumpShape( i ) = shape1( i );
            }
            for ( int i = 0; i < dofs2.Size(); ++i )
            {
                jumpShape( dofs1.Size() + i ) = -shape2( i );
            }
            // A 2D-body interface is a vertex with measure one. A 3D-body
            // interface uses the actual reference curved-edge metric, not a chord.
            const mfem::real_t metric = vertexInterface ? 1. : transformation->Weight();
            MFEM_VERIFY( std::isfinite( metric ) && metric > 0.,
                         "Multiplier stabilization requires a positive finite interface metric." );
            const mfem::real_t weight = integrationPoint.weight * metric;
            measure += weight;
            matrix.noalias() += ( coefficient * weight ) * shape * shape.transpose();
        }
        MFEM_VERIFY( std::isfinite( measure ) && measure > 0. && matrix.allFinite(),
                     "Multiplier stabilization requires positive finite interface measure and finite contributions." );
    }
}

void SemismoothRigidContactOperator::BuildSparsity()
{
    const int displacementSize = mDisplacementSpace.GetTrueVSize();
    const int multiplierSize = mMultiplierSpace.GetTrueVSize();
    mDisplacementJacobian = std::make_unique<mfem::SparseMatrix>( displacementSize, displacementSize );
    mDisplacementMultiplierJacobian = std::make_unique<mfem::SparseMatrix>( displacementSize, multiplierSize );
    mMultiplierDisplacementJacobian = std::make_unique<mfem::SparseMatrix>( multiplierSize, displacementSize );
    mMultiplierJacobian = std::make_unique<mfem::SparseMatrix>( multiplierSize, multiplierSize );

    for ( int contactElement = 0; contactElement < mContactSubMesh.GetNE(); contactElement++ )
    {
        // Contact uses x = X + u on the boundary, not volume gradients.
        // Trace-zero volume DOFs must not enter any contact sparsity block.
        mfem::DofTransformation* transformation = mDisplacementSpace.GetBdrElementVDofs(
            mBoundaryMultiplier.GetParentBoundaryElement( contactElement ), mElementVectorDofs );
        MFEM_VERIFY( transformation == nullptr,
                     "Semismooth contact currently supports boundary traces without DOF transformations." );
        mBoundaryMultiplier.GetElementDofs( contactElement, mElementMultiplierDofs );
        const int multiplierDofCount = mElementMultiplierDofs.Size();
        mElementJacobian.SetSize( mElementVectorDofs.Size() );
        mElementJacobian = 0.;
        mElementDisplacementMultiplierJacobian.SetSize( mElementVectorDofs.Size(), multiplierDofCount );
        mElementDisplacementMultiplierJacobian = 0.;
        mElementMultiplierDisplacementJacobian.SetSize( multiplierDofCount, mElementVectorDofs.Size() );
        mElementMultiplierDisplacementJacobian = 0.;
        mElementMultiplierJacobian.SetSize( multiplierDofCount );
        mElementMultiplierJacobian = 0.;
        mDisplacementJacobian->AddSubMatrix( mElementVectorDofs, mElementVectorDofs, mElementJacobian, 0 );
        mDisplacementMultiplierJacobian->AddSubMatrix( mElementVectorDofs, mElementMultiplierDofs,
                                                       mElementDisplacementMultiplierJacobian, 0 );
        mMultiplierDisplacementJacobian->AddSubMatrix( mElementMultiplierDofs, mElementVectorDofs,
                                                       mElementMultiplierDisplacementJacobian, 0 );
        mMultiplierJacobian->AddSubMatrix( mElementMultiplierDofs, mElementMultiplierDofs, mElementMultiplierJacobian, 0 );
    }
    mfem::DenseMatrix jumpMatrix;
    for ( const MultiplierJump& jump : mMultiplierJumps )
    {
        jumpMatrix.SetSize( jump.Dofs.Size() );
        jumpMatrix = 0.;
        mMultiplierJacobian->AddSubMatrix( jump.Dofs, jump.Dofs, jumpMatrix, 0 );
    }
    mDisplacementJacobian->Finalize( 0 );
    mDisplacementMultiplierJacobian->Finalize( 0 );
    mMultiplierDisplacementJacobian->Finalize( 0 );
    mMultiplierJacobian->Finalize( 0 );
}

void SemismoothRigidContactOperator::VerifyUnknown( const mfem::Vector& unknown ) const
{
    MFEM_VERIFY( unknown.Size() == Height(), "The monolithic contact unknown has the wrong size." );
    VerifyFiniteVector( unknown, "The monolithic contact unknown must be finite." );
}

void SemismoothRigidContactOperator::BuildIntegrationRules( const mfem::IntegrationRule* integrationRule )
{
    mIntegrationRules.assign( mContactSubMesh.GetNE(), integrationRule );
    if ( integrationRule != nullptr )
    {
        return;
    }

    for ( int contactElement = 0; contactElement < mContactSubMesh.GetNE(); ++contactElement )
    {
        const int boundaryElement = mBoundaryMultiplier.GetParentBoundaryElement( contactElement );
        auto* transformation = mDisplacementSpace.GetMesh()->GetBdrFaceTransformations( boundaryElement );
        MFEM_VERIFY( transformation != nullptr && transformation->Elem1 != nullptr && transformation->Elem2No < 0,
                     "A semismooth contact element must map to an exterior parent boundary face." );
        const auto& element = *mDisplacementSpace.GetFE( transformation->Elem1No );
        const auto& multiplierElement = *mMultiplierSpace.GetFE( contactElement );
        // Products N*N, N*M, and M*M share this rule. Preserve the volume-based
        // geometry and Pk increment of the P0 policy while accounting for L2 degree.
        int order = transformation->Elem1->OrderW() + 2 * std::max( element.GetOrder(), multiplierElement.GetOrder() );
        if ( element.Space() == mfem::FunctionSpace::Pk )
        {
            order++;
        }
        mIntegrationRules[contactElement] = &mfem::IntRules.Get( transformation->GetGeometryType(), order );
    }
}

template <typename Visitor>
void SemismoothRigidContactOperator::VisitElementQuadraturePoints( const int contactElement,
                                                                   const mfem::Vector& displacement,
                                                                   const mfem::Vector& multiplier,
                                                                   const mfem::real_t inverseGamma,
                                                                   Visitor&& visitor ) const
{
    const int dimension = mObstacle.Dimension();
    const auto& integrationRule = *mIntegrationRules[contactElement];
    const int boundaryElement = mBoundaryMultiplier.GetParentBoundaryElement( contactElement );
    const mfem::FiniteElement& element = *mDisplacementSpace.GetBE( boundaryElement );
    const auto geometry = element.GetGeomType();
    int face = -1;
    int orientation = -1;
    mDisplacementSpace.GetMesh()->GetBdrElementFace( boundaryElement, &face, &orientation );
    const int inverseOrientation = mfem::Geometry::GetInverseOrientation( geometry, orientation );
    mfem::ElementTransformation* boundaryTransformation = mDisplacementSpace.GetBdrElementTransformation( boundaryElement );
    MFEM_VERIFY( element.GetDim() == dimension - 1 && element.GetRangeType() == mfem::FiniteElement::SCALAR &&
                     element.GetMapType() == mfem::FiniteElement::VALUE,
                 "Semismooth contact requires value-mapped scalar displacement elements." );
    mfem::DofTransformation* transformation = mDisplacementSpace.GetBdrElementVDofs( boundaryElement, mElementVectorDofs );
    MFEM_VERIFY( transformation == nullptr,
                 "Semismooth contact currently supports boundary traces without DOF transformations." );
    displacement.GetSubVector( mElementVectorDofs, mElementDisplacement );
    const int degreeOfFreedomCount = element.GetDof();
    MFEM_VERIFY( mElementDisplacement.Size() == degreeOfFreedomCount * dimension,
                 "The local displacement vector has an incompatible size." );

    mDisplacementShape.SetSize( degreeOfFreedomCount );
    mReferencePosition.SetSize( dimension );
    mCurrentPosition.SetSize( dimension );

    // Prepare both DOF maps even for an empty rule; assembly scatters before
    // another element can replace them.
    mBoundaryMultiplier.GetElementDofs( contactElement, mElementMultiplierDofs );
    multiplier.GetSubVector( mElementMultiplierDofs, mElementMultiplier );
    mMultiplierShape.SetSize( mElementMultiplierDofs.Size() );

    // These views remain valid while point evaluations overwrite the fixed-size buffers.
    const Eigen::Map<const Eigen::MatrixXr> uDofs( mElementDisplacement.GetData(), degreeOfFreedomCount, dimension );
    const Eigen::Map<const Eigen::VectorXr> uShape( mDisplacementShape.GetData(), degreeOfFreedomCount );
    const Eigen::Map<const Eigen::VectorXr> referencePosition( mReferencePosition.GetData(), dimension );
    Eigen::Map<Eigen::VectorXr> currentPosition( mCurrentPosition.GetData(), dimension );

    for ( int point = 0; point < integrationRule.GetNPoints(); point++ )
    {
        const mfem::IntegrationPoint& integrationPoint = integrationRule.IntPoint( point );
        const mfem::IntegrationPoint boundaryPoint =
            mfem::Mesh::TransformBdrElementToFace( geometry, inverseOrientation, integrationPoint );
        boundaryTransformation->SetIntPoint( &boundaryPoint );
        element.CalcShape( boundaryPoint, mDisplacementShape );
        boundaryTransformation->Transform( boundaryPoint, mReferencePosition );
        // Both fields use the ORIGINAL parent boundary reference point.
        // CalcShape must not receive the mesh-face integrationPoint above.
        mBoundaryMultiplier.CalcShape( contactElement, boundaryPoint, mMultiplierShape );
        const mfem::real_t lambda = mMultiplierShape * mElementMultiplier;

        // Interpolate displacement u from its trace DOFs; the generic
        // primal fields have already been mapped through the extraction.
        currentPosition = referencePosition;
        currentPosition.noalias() += uDofs.transpose() * uShape;

        mObstacle.Evaluate( mCurrentPosition, mEvaluation );
        VerifySignedDistanceEvaluation( mObstacle, mEvaluation );
        // Positive gap is admissible, positive lambda is compression:
        // u_source = -gap, lambda_source = -lambda in the cited preprint.
        const mfem::real_t augmentedMultiplier = lambda - inverseGamma * mEvaluation.Gap;
        const mfem::real_t pressure = std::max( mfem::real_t{ 0. }, augmentedMultiplier );
        const bool active = augmentedMultiplier >= 0.;
        const mfem::real_t weight = integrationPoint.weight * boundaryTransformation->Weight();
        const ContactPointData contactPoint{ mDisplacementShape, mMultiplierShape, mEvaluation, lambda,
                                             pressure,           weight,           active };
        visitor( contactPoint );
    }
}

void SemismoothRigidContactOperator::MaskEssentialElementEntries( const bool assembleResidual, const bool assembleJacobian ) const
{
    for ( int localDof = 0; localDof < mElementVectorDofs.Size(); localDof++ )
    {
        const int globalDof = DecodeDof( mElementVectorDofs[localDof] );
        if ( mEssentialMarker[globalDof] == 0 )
        {
            continue;
        }
        if ( assembleResidual )
        {
            mElementResidual( localDof ) = 0.;
        }
        if ( assembleJacobian )
        {
            mElementConstantNormalCoupling( localDof ) = 0.;
            for ( int multiplierDof = 0; multiplierDof < mElementMultiplierDofs.Size(); ++multiplierDof )
            {
                mElementDisplacementMultiplierJacobian( localDof, multiplierDof ) = 0.;
                mElementMultiplierDisplacementJacobian( multiplierDof, localDof ) = 0.;
            }
            for ( int column = 0; column < mElementJacobian.Width(); column++ )
            {
                mElementJacobian( localDof, column ) = 0.;
                mElementJacobian( column, localDof ) = 0.;
            }
        }
    }
}

void SemismoothRigidContactOperator::AssembleContact( const mfem::Vector& displacement,
                                                      const mfem::Vector& multiplier,
                                                      mfem::Vector* displacementResidual,
                                                      mfem::Vector* multiplierResidual,
                                                      const bool assembleJacobian ) const
{
    // Gamma is fixed: evaluate 1/gamma outside the quadrature/component loops.
    const mfem::real_t inverseGamma = 1. / mGamma;
    const bool assembleResidual = displacementResidual != nullptr;
    MFEM_ASSERT( assembleResidual == ( multiplierResidual != nullptr ),
                 "Both contact residual blocks must be assembled together." );
    if ( assembleResidual )
    {
        displacementResidual->SetSize( mDisplacementSpace.GetTrueVSize() );
        *displacementResidual = 0.;
        multiplierResidual->SetSize( mMultiplierSpace.GetTrueVSize() );
        *multiplierResidual = 0.;
    }
    if ( assembleJacobian )
    {
        *mDisplacementJacobian = 0.;
        *mDisplacementMultiplierJacobian = 0.;
        *mMultiplierDisplacementJacobian = 0.;
        *mMultiplierJacobian = 0.;
    }

    for ( int contactElement = 0; contactElement < mContactSubMesh.GetNE(); contactElement++ )
    {
        ResetElementAssembly( contactElement, assembleResidual, assembleJacobian );
        ContactElementActivity activity{};
        VisitElementQuadraturePoints( contactElement, displacement, multiplier, inverseGamma,
                                      [&]( const ContactPointData& point )
                                      {
                                          activity.AccumulatePointActivity( point.Pressure, point.Active );
                                          if ( assembleResidual )
                                          {
                                              AccumulateContactPointResidual( point );
                                          }
                                          if ( assembleJacobian )
                                          {
                                              AccumulateContactPointJacobian( point, inverseGamma );
                                          }
                                      } );
        ScatterContactElement( activity, displacementResidual, multiplierResidual, assembleJacobian );
    }
    AssembleMultiplierJumpStabilization( multiplier, multiplierResidual, assembleJacobian );
}

void SemismoothRigidContactOperator::AccumulateContactPointResidual( const ContactPointData& point ) const
{
    // Adapt preprint HTML Section 2, (3)-(5), (12), not journal numbering;
    // citation/signs in Contact.h; derivation in docs/boundary-multiplier-space.md.
    // With p = max(0,lambda-gap/gamma), n = grad(gap), the contact functional is
    // gamma*(<p,p>_Gamma - <lambda,lambda>_Gamma)/2 - s(lambda,lambda)/2.
    // For displacement test v and multiplier test mu:
    // r_u(v) = -<p,n dot v>_Gamma,
    // r_lambda(mu) = <gamma*(p-lambda),mu>_Gamma - s(lambda,mu).
    // <.,.>_Gamma is the reference-boundary integral, sampled with Weight
    // including the metric once. Apply row scales here; assemble jumps separately.
    const int degreeOfFreedomCount = point.DisplacementShape.Size();
    const int dimension = point.Distance.Gradient.Size();
    // vShape contains the scalar factors of displacement tests v;
    // each vector test is vShape(i) times a Cartesian unit vector.
    const Eigen::Map<const Eigen::VectorXr> vShape( point.DisplacementShape.GetData(), degreeOfFreedomCount );
    const Eigen::Map<const Eigen::VectorXr> normal( point.Distance.Gradient.GetData(), dimension );
    if ( point.Pressure > 0. )
    {
        Eigen::Map<Eigen::MatrixXr> elementResidual( mElementResidual.GetData(), degreeOfFreedomCount, dimension );
        elementResidual.noalias() -= ( mPrimalResidualScale * point.Weight * point.Pressure ) * vShape * normal.transpose();
    }
    // c = gamma*(p-lambda), p = max(0,lambda-gap/gamma), is algebraically
    // -gap when active and -gamma*lambda otherwise. Evaluate those branches
    // directly to avoid cancellation when p rounds to lambda at large gamma.
    const mfem::real_t complementarity = point.Active ? -point.Distance.Gap : -mGamma * point.Lambda;
    mElementMultiplierResidual.Add( mMultiplierResidualScale * point.Weight * complementarity, point.MultiplierShape );
}

void SemismoothRigidContactOperator::AccumulateContactPointJacobian( const ContactPointData& point, const mfem::real_t inverseGamma ) const
{
    // Contact-point Jacobian only: bulk physics and multiplier jump stabilization
    // are assembled separately. In K_ul etc., l denotes lambda; the first index
    // identifies the residual row and the second the differentiated unknown.
    // Displacement DOFs are component-major: localIndex(a,c) = a + c*nDOFs.
    // Our signed-distance linearization: dg = n dot du, n = grad(gap),
    // dn = H*du, H = Hessian(gap), dp = chi*(dlambda-dg/gamma).
    // chi = 1 on the active branch (also at equality), else 0.
    // The boundary bilinear forms therefore use (chi/gamma)*n*n^T
    // - p*H in uu. With B = [N*n_1; ...; N*n_d] and multiplier shape M,
    // K_ul += -w_u*chi*B*M^T, K_lu += -w_lambda*chi*M*B^T,
    // K_ll += w_lambda*gamma*(chi-1)*M*M^T. Here each w includes its
    // positive row scale and the same reference quadrature weight once.
    // Galerkin trial factors represent du/dlambda, paired with the same tests.
    // The curved -p*H term is derived here, not quoted from the
    // scalar preprint. Reference weights have no u derivative.
    const int degreeOfFreedomCount = point.DisplacementShape.Size();
    const int dimension = point.Distance.Gradient.Size();
    const int multiplierDofCount = point.MultiplierShape.Size();
    const Eigen::Map<const Eigen::VectorXr> vShape( point.DisplacementShape.GetData(), degreeOfFreedomCount );
    const Eigen::Map<const Eigen::VectorXr> multiplierShape( point.MultiplierShape.GetData(), multiplierDofCount );
    const Eigen::Map<const Eigen::VectorXr> normal( point.Distance.Gradient.GetData(), dimension );
    // N: displacement test (v) and trial (u) shapes share the Galerkin basis.
    const auto& uShape = vShape;
    if ( point.Active )
    {
        // K_ul = dR_u/dlambda: displacement rows, multiplier columns.
        Eigen::Map<Eigen::MatrixXr> displacementMultiplier( mElementDisplacementMultiplierJacobian.GetData(),
                                                            degreeOfFreedomCount * dimension, multiplierDofCount );
        // K_lu = dR_lambda/du: multiplier rows, displacement columns.
        Eigen::Map<Eigen::MatrixXr> multiplierDisplacement( mElementMultiplierDisplacementJacobian.GetData(),
                                                            multiplierDofCount, degreeOfFreedomCount * dimension );
        // Unscaled mixed contributions are transposes; unequal row scales break that relation.
        for ( int component = 0; component < dimension; component++ )
        {
            displacementMultiplier
                .block( component * degreeOfFreedomCount, 0, degreeOfFreedomCount, multiplierDofCount )
                .noalias() -=
                ( mPrimalResidualScale * point.Weight * normal( component ) ) * vShape * multiplierShape.transpose();
            multiplierDisplacement
                .block( 0, component * degreeOfFreedomCount, multiplierDofCount, degreeOfFreedomCount )
                .noalias() -=
                ( mMultiplierResidualScale * point.Weight * normal( component ) ) * multiplierShape * uShape.transpose();
        }
        // Diagnostic dR_u for delta_lambda = 1, not another Jacobian block.
        // After essential-DOF masking, tests for a free constant-mode normal coupling.
        // Test the constant multiplier mode directly, without assuming that
        // coefficients are nodal or that summing basis columns represents it.
        Eigen::Map<Eigen::MatrixXr> constantCoupling( mElementConstantNormalCoupling.GetData(), degreeOfFreedomCount, dimension );
        constantCoupling.noalias() -= ( mPrimalResidualScale * point.Weight ) * vShape * normal.transpose();
        // Contact K_uu = dR_u/du, with nDOFs-by-nDOFs component blocks.
        Eigen::Map<Eigen::MatrixXr> elementJacobian( mElementJacobian.GetData(), degreeOfFreedomCount * dimension,
                                                     degreeOfFreedomCount * dimension );
        // Active spatial tangent A = n*n^T/gamma - p*H, H = Hessian(gap):
        // n*n^T/gamma accounts for pressure changes; -p*H for normal-direction changes.
        // H = 0 for a plane. The curvature term is multiplied by pressure p.
        for ( int testComponent = 0; testComponent < dimension; testComponent++ )
        {
            for ( int trialComponent = 0; trialComponent < dimension; trialComponent++ )
            {
                // primalScale * A(testComponent, trialComponent), before quadrature weight.
                const mfem::real_t spatialTangent =
                    mPrimalResidualScale *
                    ( inverseGamma * point.Distance.Gradient( testComponent ) * point.Distance.Gradient( trialComponent ) -
                      point.Pressure * point.Distance.Hessian( testComponent, trialComponent ) );
                // K_uu(c,d) += primalScale * weight * A(c,d) * N*N^T for all DOF pairs.
                auto tangentBlock = elementJacobian.block( testComponent * degreeOfFreedomCount, trialComponent * degreeOfFreedomCount,
                                                           degreeOfFreedomCount, degreeOfFreedomCount );
                tangentBlock.noalias() += ( point.Weight * spatialTangent ) * vShape * uShape.transpose();
            }
        }
    }
    else
    {
        // K_ll = dR_lambda/dlambda += multiplierScale * weight * gamma * (chi-1) * M*M^T.
        // Here chi = 0: negative scaled multiplier mass. At chi = 1 this point term
        // vanishes, but the separately assembled jump stabilization can still contribute.
        Eigen::Map<Eigen::MatrixXr> multiplierJacobian( mElementMultiplierJacobian.GetData(), multiplierDofCount, multiplierDofCount );
        multiplierJacobian.noalias() -=
            ( mMultiplierResidualScale * point.Weight * mGamma ) * multiplierShape * multiplierShape.transpose();
    }
}

void SemismoothRigidContactOperator::ScatterContactElement( const ContactElementActivity& activity,
                                                            mfem::Vector* displacementResidual,
                                                            mfem::Vector* multiplierResidual,
                                                            const bool assembleJacobian ) const
{
    const bool assembleResidual = displacementResidual != nullptr;
    const mfem::real_t unmaskedCouplingNorm =
        assembleJacobian && activity.HasActivePoints() ? mElementConstantNormalCoupling.Norml2() : 0.;
    MaskEssentialElementEntries( assembleResidual, assembleJacobian );
    if ( assembleJacobian && activity.IsFullyActive() )
    {
        // Preserve the P0 free-normal rejection using its constant-mode test.
        // This restrictive local check is NOT a full-rank/inf-sup test for higher L2
        // orders. Do not reject all locally deficient modes: jumps may control them.
        const mfem::real_t minimumRelativeCoupling = std::sqrt( std::numeric_limits<mfem::real_t>::epsilon() ) * unmaskedCouplingNorm;
        MFEM_VERIFY( std::isfinite( unmaskedCouplingNorm ) && unmaskedCouplingNorm > 0. &&
                         mElementConstantNormalCoupling.Norml2() > minimumRelativeCoupling,
                     "A fully active semismooth contact face needs a free normal displacement variation." );
    }
    if ( assembleResidual )
    {
        if ( activity.HasPressure() )
        {
            displacementResidual->AddElementVector( mElementVectorDofs, mElementResidual );
        }
        multiplierResidual->AddElementVector( mElementMultiplierDofs, mElementMultiplierResidual );
    }
    if ( assembleJacobian )
    {
        if ( activity.HasActivePoints() )
        {
            mDisplacementJacobian->AddSubMatrix( mElementVectorDofs, mElementVectorDofs, mElementJacobian );
            mDisplacementMultiplierJacobian->AddSubMatrix( mElementVectorDofs, mElementMultiplierDofs,
                                                           mElementDisplacementMultiplierJacobian );
            mMultiplierDisplacementJacobian->AddSubMatrix( mElementMultiplierDofs, mElementVectorDofs,
                                                           mElementMultiplierDisplacementJacobian );
        }
        mMultiplierJacobian->AddSubMatrix( mElementMultiplierDofs, mElementMultiplierDofs, mElementMultiplierJacobian );
    }
}

void SemismoothRigidContactOperator::AssembleMultiplierJumpStabilization( const mfem::Vector& multiplier,
                                                                          mfem::Vector* multiplierResidual,
                                                                          const bool assembleJacobian ) const
{
    // The -s(lambda,lambda)/2 saddle term gives -S lambda and Jacobian -S,
    // not a positive body stiffness (preprint HTML (12), adapted signs above).
    // Each cached interface block is the unscaled integral of J^T*J with
    // J=[M_e^T,-M_f^T]. It leaves the constant multiplier function intact.
    // Apply the multiplier row scale to both contributions, exactly once.
    for ( const MultiplierJump& jump : mMultiplierJumps )
    {
        for ( int row = 0; row < jump.Dofs.Size(); ++row )
        {
            mfem::real_t residual = 0.;
            for ( int column = 0; column < jump.Dofs.Size(); ++column )
            {
                const mfem::real_t entry = -mMultiplierResidualScale * jump.Matrix( row, column );
                if ( multiplierResidual != nullptr )
                {
                    residual += entry * multiplier( jump.Dofs[column] );
                }
                if ( assembleJacobian )
                {
                    mMultiplierJacobian->Add( jump.Dofs[row], jump.Dofs[column], entry );
                }
            }
            if ( multiplierResidual != nullptr )
            {
                ( *multiplierResidual )( jump.Dofs[row] ) += residual;
            }
        }
    }
}

void SemismoothRigidContactOperator::ResetElementAssembly( const int contactElement, const bool assembleResidual, const bool assembleJacobian ) const
{
    const int boundaryElement = mBoundaryMultiplier.GetParentBoundaryElement( contactElement );
    const int vectorDofCount = mDisplacementSpace.GetBE( boundaryElement )->GetDof() * mDisplacementSpace.GetVDim();
    const int multiplierDofCount = mMultiplierSpace.GetFE( contactElement )->GetDof();
    if ( assembleResidual )
    {
        mElementResidual.SetSize( vectorDofCount );
        mElementResidual = 0.;
        mElementMultiplierResidual.SetSize( multiplierDofCount );
        mElementMultiplierResidual = 0.;
    }
    if ( assembleJacobian )
    {
        mElementJacobian.SetSize( vectorDofCount );
        mElementJacobian = 0.;
        mElementDisplacementMultiplierJacobian.SetSize( vectorDofCount, multiplierDofCount );
        mElementDisplacementMultiplierJacobian = 0.;
        mElementMultiplierDisplacementJacobian.SetSize( multiplierDofCount, vectorDofCount );
        mElementMultiplierDisplacementJacobian = 0.;
        mElementMultiplierJacobian.SetSize( multiplierDofCount );
        mElementMultiplierJacobian = 0.;
        mElementConstantNormalCoupling.SetSize( vectorDofCount );
        mElementConstantNormalCoupling = 0.;
    }
}

void SemismoothRigidContactOperator::GetChildOperators( std::vector<const mfem::Operator*>& children ) const
{
    children.push_back( &mPrimalOperator );
}

const mfem::Array<int>& SemismoothRigidContactOperator::GetBlockOffsets() const noexcept
{
    return mBlockOffsets;
}

const mfem::SubMesh& SemismoothRigidContactOperator::GetContactSubMesh() const noexcept
{
    return mBoundaryMultiplier.GetMesh();
}

const mfem::FiniteElementSpace& SemismoothRigidContactOperator::GetMultiplierSpace() const noexcept
{
    return mBoundaryMultiplier.GetSpace();
}

SemismoothContactDiagnostics SemismoothRigidContactOperator::ComputeContactDiagnostics( const mfem::Vector& unknown ) const
{
    VerifyUnknown( unknown );
    mfem::Vector primal( const_cast<mfem::real_t*>( unknown.GetData() ), mPrimalOperator.Width() );
    mfem::Vector multiplier( const_cast<mfem::real_t*>( unknown.GetData() ) + mBlockOffsets[1], mMultiplierSpace.GetTrueVSize() );
    mPrimalToDisplacement.Mult( primal, mDisplacement );

    SemismoothContactDiagnostics diagnostics( mObstacle.Dimension() );
    const mfem::real_t inverseGamma = 1. / mGamma;
    for ( int contactElement = 0; contactElement < mContactSubMesh.GetNE(); contactElement++ )
    {
        VisitElementQuadraturePoints(
            contactElement, mDisplacement, multiplier, inverseGamma,
            [&diagnostics, gamma = mGamma]( const ContactPointData& point )
            {
                diagnostics.MinimumGap = std::min( diagnostics.MinimumGap, point.Distance.Gap );
                diagnostics.MaximumPenetration = std::max( diagnostics.MaximumPenetration, -point.Distance.Gap );
                diagnostics.MinimumMultiplier = std::min( diagnostics.MinimumMultiplier, point.Lambda );
                diagnostics.MaximumMultiplier = std::max( diagnostics.MaximumMultiplier, point.Lambda );
                diagnostics.MaximumPressure = std::max( diagnostics.MaximumPressure, point.Pressure );
                // Same cancellation-free c = gamma*(p-lambda) branches as the residual.
                const mfem::real_t complementarity = point.Active ? -point.Distance.Gap : -gamma * point.Lambda;
                diagnostics.MaximumComplementarityResidual =
                    std::max( diagnostics.MaximumComplementarityResidual, std::abs( complementarity ) );
                diagnostics.Resultant.Add( point.Weight * point.Pressure, point.Distance.Gradient );
                diagnostics.ActiveQuadraturePointCount += point.Active ? 1 : 0;
                diagnostics.QuadraturePointCount++;
            } );
    }
    return diagnostics;
}
} // namespace plugin
