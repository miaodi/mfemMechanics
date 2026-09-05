#pragma once

#include <mfem.hpp>

namespace plugin
{
/** @brief Value and first two spatial derivatives of a signed-distance function. */
struct SignedDistanceEvaluation
{
    explicit SignedDistanceEvaluation( int dimension = 0 );

    void SetSize( int dimension );

    mfem::real_t Gap{ 0. };
    mfem::Vector Gradient;
    mfem::DenseMatrix Hessian;
};

/** @brief Smooth signed distance to an analytic rigid obstacle.

    The admissible region has nonnegative gap. Evaluate() must return a coherent
    signed distance, its unit gradient, and its symmetric Hessian. The obstacle
    must outlive every contact integrator that borrows it. */
class RigidObstacle
{
public:
    virtual ~RigidObstacle() = default;

    [[nodiscard]] virtual int Dimension() const noexcept = 0;
    virtual void Evaluate( const mfem::Vector& position, SignedDistanceEvaluation& evaluation ) const = 0;
};

/** @brief Rigid plane with signed distance (x - point) dot admissibleNormal. */
class RigidPlaneObstacle final : public RigidObstacle
{
public:
    RigidPlaneObstacle( const mfem::Vector& point, const mfem::Vector& admissibleNormal );

    [[nodiscard]] int Dimension() const noexcept override;
    void Evaluate( const mfem::Vector& position, SignedDistanceEvaluation& evaluation ) const override;

private:
    mfem::Vector mPoint;
    mfem::Vector mNormal;
};

/** @brief Circle in 2D or sphere in 3D whose exterior is admissible. */
class RigidSphereObstacle final : public RigidObstacle
{
public:
    RigidSphereObstacle( const mfem::Vector& center, mfem::real_t radius );

    [[nodiscard]] int Dimension() const noexcept override;
    void Evaluate( const mfem::Vector& position, SignedDistanceEvaluation& evaluation ) const override;

private:
    mfem::Vector mCenter;
    mfem::real_t mRadius;
};

/** @brief Frictionless penalty contact integrated over the initial boundary. */
class FrictionlessPenaltyContactIntegrator final : public mfem::NonlinearFormIntegrator
{
public:
    FrictionlessPenaltyContactIntegrator( const RigidObstacle& obstacle, mfem::real_t penalty );
    FrictionlessPenaltyContactIntegrator( RigidObstacle&& obstacle, mfem::real_t penalty ) = delete;

    void AssembleFaceVector( const mfem::FiniteElement& element1,
                             const mfem::FiniteElement& element2,
                             mfem::FaceElementTransformations& transformation,
                             const mfem::Vector& elementDisplacement,
                             mfem::Vector& elementResidual ) override;

    void AssembleFaceGrad( const mfem::FiniteElement& element1,
                           const mfem::FiniteElement& element2,
                           mfem::FaceElementTransformations& transformation,
                           const mfem::Vector& elementDisplacement,
                           mfem::DenseMatrix& elementTangent ) override;

private:
    void VerifyFaceInput( const mfem::FiniteElement& element,
                          const mfem::FaceElementTransformations& transformation,
                          const mfem::Vector& elementDisplacement ) const;
    const mfem::IntegrationRule& SelectIntegrationRule( const mfem::FiniteElement& element,
                                                        const mfem::FaceElementTransformations& transformation ) const;
    void SetScratchSize( int degreeOfFreedomCount, int dimension );
    void EvaluateContactPoint( const mfem::FiniteElement& element,
                               mfem::FaceElementTransformations& transformation,
                               const mfem::IntegrationPoint& integrationPoint,
                               const mfem::Vector& elementDisplacement );
    void VerifySignedDistanceEvaluation() const;

    const RigidObstacle& mObstacle;
    mfem::real_t mPenalty;
    mfem::Vector mShape;
    mfem::Vector mReferencePosition;
    mfem::Vector mCurrentPosition;
    SignedDistanceEvaluation mEvaluation;
};
} // namespace plugin
