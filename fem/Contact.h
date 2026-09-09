#pragma once

#include "BoundaryMultiplierSpace.h"
#include "Solvers.h"

#include <limits>
#include <memory>
#include <mfem.hpp>
#include <vector>

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

/** @brief Frictionless penalty contact integrated over the initial boundary.

    Register with NonlinearForm::AddBoundaryIntegrator for H1 displacement: local
    vectors/matrices contain boundary DOFs only. Custom rules use mesh-face
    coordinates. The obstacle and custom rule are borrowed; scratch is not
    reentrant. Boundary entry points require a mesh-owned boundary transformation
    and standard H1 traces (the trace and adjacent volume have the same order). */
class FrictionlessPenaltyContactIntegrator final : public mfem::NonlinearFormIntegrator
{
public:
    FrictionlessPenaltyContactIntegrator( const RigidObstacle& obstacle, mfem::real_t penalty );
    FrictionlessPenaltyContactIntegrator( RigidObstacle&& obstacle, mfem::real_t penalty ) = delete;

    void AssembleElementVector( const mfem::FiniteElement& element,
                                mfem::ElementTransformation& transformation,
                                const mfem::Vector& displacement,
                                mfem::Vector& residual ) override;
    void AssembleElementGrad( const mfem::FiniteElement& element,
                              mfem::ElementTransformation& transformation,
                              const mfem::Vector& displacement,
                              mfem::DenseMatrix& tangent ) override;
    mfem::real_t GetElementEnergy( const mfem::FiniteElement& element,
                                   mfem::ElementTransformation& transformation,
                                   const mfem::Vector& displacement ) override;

private:
    void VerifyFaceInput( const mfem::FiniteElement& element,
                          const mfem::ElementTransformation& transformation,
                          const mfem::Vector& displacement ) const;
    const mfem::IntegrationRule& SelectIntegrationRule( const mfem::FiniteElement& element,
                                                        const mfem::ElementTransformation& transformation );
    void EvaluateContactPoint( const mfem::FiniteElement& element,
                               mfem::ElementTransformation& transformation,
                               const mfem::IntegrationPoint& point,
                               const mfem::Vector& displacement );
    void EvaluateDisplacedPoint( int degreeOfFreedomCount, int dimension, const mfem::Vector& displacement );
    void SetScratchSize( int degreeOfFreedomCount, int dimension );
    void VerifySignedDistanceEvaluation() const;

    const RigidObstacle& mObstacle;
    mfem::real_t mPenalty;
    mfem::Vector mShape;
    mfem::Vector mReferencePosition;
    mfem::Vector mCurrentPosition;
    SignedDistanceEvaluation mEvaluation;
    mfem::IntegrationRule mBoundaryRule;
    mfem::IsoparametricTransformation mVolumeTransformation;
};

/** Diagnostics sample the assembly quadrature, not coefficient extrema or a
    certified pointwise bound between quadrature points. */
struct SemismoothContactDiagnostics
{
    explicit SemismoothContactDiagnostics( int dimension = 0 ) : Resultant( dimension )
    {
        Resultant = 0.;
    }

    mfem::real_t MinimumGap{ std::numeric_limits<mfem::real_t>::infinity() };
    mfem::real_t MaximumPenetration{ 0. };
    mfem::real_t MinimumMultiplier{ std::numeric_limits<mfem::real_t>::infinity() };
    mfem::real_t MaximumMultiplier{ -std::numeric_limits<mfem::real_t>::infinity() };
    mfem::real_t MaximumPressure{ 0. };
    mfem::real_t MaximumComplementarityResidual{ 0. };
    mfem::Vector Resultant;
    int ActiveQuadraturePointCount{ 0 };
    int QuadraturePointCount{ 0 };
};

/** @brief Monolithic semismooth contact operator with configurable boundary L2 multipliers.

    Formulation adapted from Erik Burman, Peter Hansbo, and Mats G. Larson,
    "Augmented Lagrangian finite element methods for contact problems",
    arXiv:1609.03326v1 (2016), https://arxiv.org/abs/1609.03326v1.
    Verified preprint HTML Section 2: (3)-(5) contact map and functional, (9) jump
    stabilization, (12) saddle form; these are NOT journal equation numbers.
    The parameter gamma is the source's compliance (length/stress), equal to
    1/kappa for augmentation stiffness kappa. Signs correspond to u_source = -gap
    and lambda_source = -lambda (positive gap outside the obstacle, positive
    multiplier in compression). The curved
    signed-distance derivative and the concrete L2 discretization below are
    repository adaptations, not a stability result from that scalar model.
    See docs/boundary-multiplier-space.md for equations and source correspondence,
    docs/semismooth-rigid-contact-formulation.tex for the detailed derivation,
    and docs/frictionless-penalty-contact.md for the implementation conventions.

    The unknown is ordered as [primal true dofs, multiplier true dofs]. The
    primal-to-displacement operator extracts displacement true dofs from an
    arbitrary primal field vector, allowing the primal operator itself to be a
    BlockNonlinearForm. Positive residual scales multiply the two block rows
    without changing their zero; the Jacobian has the same row scaling. Unequal
    scales generally destroy symmetry and change the residual stopping norm.
    The dimensionless factor delta controls the boundary L2 multiplier
    jump term delta*gamma*h. The primal operator, extraction operator,
    displacement space, obstacle, and BoundaryMultiplierSpace are borrowed
    and must outlive this operator. The displacement space must be serial and
    conforming, and only MFEM's default CPU backend is supported. Essential
    displacement true DOFs must match the primal constraints, and every contact
    face must retain a free displacement trace DOF and, when fully active, a
    free constant-mode normal variation. This restrictive local check does not establish
    higher-order multiplier rank or inf-sup stability; stabilization can control
    additional modes. Reconstruct the operator after mesh/geometry/space changes.
    A gradient reference remains valid only until either
    this operator or the borrowed primal operator evaluates another gradient. */
class SemismoothRigidContactOperator final : public mfem::Operator, public CompositeNonlinearOperator
{
public:
    /** @brief Couple a primal residual to displacement--multiplier contact.

        @param primalOperator Borrowed square residual operator on primal true
        DOFs; it may contain fields besides displacement (e.g. a BlockNonlinearForm).
        @param primalToDisplacement Borrowed linear extraction from primal true
        DOFs to displacement true DOFs. Its transpose inserts contact forces into
        the primal residual; it is not the displacement test function v.
        @param displacementSpace Borrowed serial conforming H1 vector space for
        displacement u (length), with one component per spatial dimension.
        @param obstacle Borrowed signed-distance geometry evaluated at X + u;
        gap is positive in the admissible region and has units of length.
        @param boundaryMultiplier Borrowed boundary mesh and scalar VALUE-mapped
        discontinuous L2 space on the displacement mesh; must outlive this operator.
        The multiplier lambda is positive in compression and has units of stress.
        @param essentialDisplacementTrueDofs Constrained displacement true DOF
        indices, matching the primal constraints; contact variations there vanish.
        @param gamma Positive finite augmentation compliance (length/stress),
        with finite reciprocal. Sets p = max(0, lambda - gap/gamma); smaller gamma
        increases the active normal stiffness 1/gamma. This replaces the former
        positional penalty kappa: migrate callers by passing gamma = 1/kappa.
        @param primalResidualScale Positive finite weight on the entire primal
        residual row (including contact) and its Jacobian row. The unscaled contact
        coefficients have units stress times reference boundary measure.
        @param multiplierResidualScale Positive finite weight on the multiplier
        residual row and its Jacobian row, including jump stabilization. Before
        scaling, r_lambda(mu) = integral gamma*(p-lambda)*mu dA0 - s(lambda,mu);
        its coefficients have units length times reference boundary measure.
        Row weights may be inverse reference residuals for nondimensionalization;
        they change the stopping norm, not the roots, and unequal weights generally
        destroy Jacobian symmetry. Diagnostics remain unscaled.
        @param delta Finite nonnegative dimensionless multiplier-jump weight:
        s(lambda,mu) = sum_F delta*gamma*h_F*integral_F [lambda]*[mu] ds.
        Zero disables this stabilization; positive values penalize jumps between
        adjacent contact faces without penalizing a constant multiplier. Here h_F has
        units of length and averages the sizes of the adjacent contact faces.

        The contact unknown fields are u (within primal) and lambda; v and mu
        denote displacement and multiplier test functions, not extra unknowns. */
    SemismoothRigidContactOperator( mfem::Operator& primalOperator,
                                    const mfem::Operator& primalToDisplacement,
                                    mfem::FiniteElementSpace& displacementSpace,
                                    const RigidObstacle& obstacle,
                                    BoundaryMultiplierSpace& boundaryMultiplier,
                                    const mfem::Array<int>& essentialDisplacementTrueDofs,
                                    mfem::real_t gamma,
                                    mfem::real_t primalResidualScale = 1.,
                                    mfem::real_t multiplierResidualScale = 1.,
                                    mfem::real_t delta = 1. );
    SemismoothRigidContactOperator( mfem::Operator& primalOperator,
                                    mfem::Operator&& primalToDisplacement,
                                    mfem::FiniteElementSpace& displacementSpace,
                                    const RigidObstacle& obstacle,
                                    BoundaryMultiplierSpace& boundaryMultiplier,
                                    const mfem::Array<int>& essentialDisplacementTrueDofs,
                                    mfem::real_t gamma,
                                    mfem::real_t primalResidualScale = 1.,
                                    mfem::real_t multiplierResidualScale = 1.,
                                    mfem::real_t delta = 1. ) = delete;
    SemismoothRigidContactOperator( mfem::Operator& primalOperator,
                                    const mfem::Operator&& primalToDisplacement,
                                    mfem::FiniteElementSpace& displacementSpace,
                                    const RigidObstacle& obstacle,
                                    BoundaryMultiplierSpace& boundaryMultiplier,
                                    const mfem::Array<int>& essentialDisplacementTrueDofs,
                                    mfem::real_t gamma,
                                    mfem::real_t primalResidualScale = 1.,
                                    mfem::real_t multiplierResidualScale = 1.,
                                    mfem::real_t delta = 1. ) = delete;
    SemismoothRigidContactOperator( mfem::Operator& primalOperator,
                                    const mfem::Operator& primalToDisplacement,
                                    mfem::FiniteElementSpace& displacementSpace,
                                    RigidObstacle&& obstacle,
                                    BoundaryMultiplierSpace& boundaryMultiplier,
                                    const mfem::Array<int>& essentialDisplacementTrueDofs,
                                    mfem::real_t gamma,
                                    mfem::real_t primalResidualScale = 1.,
                                    mfem::real_t multiplierResidualScale = 1.,
                                    mfem::real_t delta = 1. ) = delete;
    SemismoothRigidContactOperator( mfem::Operator& primalOperator,
                                    const mfem::Operator& primalToDisplacement,
                                    mfem::FiniteElementSpace& displacementSpace,
                                    const RigidObstacle&& obstacle,
                                    BoundaryMultiplierSpace& boundaryMultiplier,
                                    const mfem::Array<int>& essentialDisplacementTrueDofs,
                                    mfem::real_t gamma,
                                    mfem::real_t primalResidualScale = 1.,
                                    mfem::real_t multiplierResidualScale = 1.,
                                    mfem::real_t delta = 1. ) = delete;
    ~SemismoothRigidContactOperator() override;

    SemismoothRigidContactOperator( const SemismoothRigidContactOperator& ) = delete;
    SemismoothRigidContactOperator& operator=( const SemismoothRigidContactOperator& ) = delete;
    SemismoothRigidContactOperator( SemismoothRigidContactOperator&& ) = delete;
    SemismoothRigidContactOperator& operator=( SemismoothRigidContactOperator&& ) = delete;

    void Mult( const mfem::Vector& unknown, mfem::Vector& residual ) const override;
    mfem::Operator& GetGradient( const mfem::Vector& unknown ) const override;
    void GetChildOperators( std::vector<const mfem::Operator*>& children ) const override;

    /** Borrow a rule in parent mesh-FACE reference coordinates; nullptr restores
        the degree-aware default. An empty rule retains the existing behavior:
        no contact-point terms or samples, but jump stabilization still applies.
        The rule must outlive its use. Refreshes the cached per-element point
        rules without changing the cached jump rule. */
    void SetIntegrationRule( const mfem::IntegrationRule* integrationRule );
    [[nodiscard]] const mfem::Array<int>& GetBlockOffsets() const noexcept;
    [[nodiscard]] const mfem::SubMesh& GetContactSubMesh() const noexcept;
    [[nodiscard]] const mfem::FiniteElementSpace& GetMultiplierSpace() const noexcept;
    void GetMultiplier( const mfem::Vector& unknown, mfem::Vector& multiplier ) const;
    [[nodiscard]] SemismoothContactDiagnostics ComputeContactDiagnostics( const mfem::Vector& unknown ) const;

private:
    struct MultiplierJump
    {
        mfem::Array<int> Dofs;
        mfem::DenseMatrix Matrix;
    };

    // Summarizes one element's sampled contact state, not committed history.
    // Used to skip zero displacement residual/Jacobian blocks during scattering
    // and to trigger the fully active-face free-normal-coupling check.
    // At lambda - gap/gamma = 0, pressure is zero but the active-side tangent
    // is selected, so pressure and activity must be tracked separately.
    class ContactElementActivity
    {
    public:
        void AccumulatePointActivity( mfem::real_t pressure, bool active )
        {
            mHasPressure = mHasPressure || pressure > 0.;
            mHasActivePoint = mHasActivePoint || active;
            mHasInactivePoint = mHasInactivePoint || !active;
        }

        // Any positive pressure: the displacement contact residual may be nonzero.
        [[nodiscard]] bool HasPressure() const noexcept
        {
            return mHasPressure;
        }

        // Any active point: contact K_uu, K_ul, and K_lu may be nonzero.
        [[nodiscard]] bool HasActivePoints() const noexcept
        {
            return mHasActivePoint;
        }

        // All sampled points are active; an empty quadrature rule is not fully active.
        [[nodiscard]] bool IsFullyActive() const noexcept
        {
            return mHasActivePoint && !mHasInactivePoint;
        }

    private:
        bool mHasPressure{ false };
        bool mHasActivePoint{ false };
        bool mHasInactivePoint{ false };
    };

    // Borrowed point data is valid only during the synchronous visitor callback.
    struct ContactPointData
    {
        const mfem::Vector& DisplacementShape;
        const mfem::Vector& MultiplierShape;
        const SignedDistanceEvaluation& Distance;
        mfem::real_t Lambda;
        mfem::real_t Pressure;
        mfem::real_t Weight;
        bool Active;
    };

    void VerifyInput( const mfem::Array<int>& essentialDisplacementTrueDofs );
    void VerifyContactTraceHasFreeDisplacementDofs();
    void BuildIntegrationRules( const mfem::IntegrationRule* integrationRule );
    void BuildMultiplierJumpStabilization();
    void BuildSparsity();
    void VerifyUnknown( const mfem::Vector& unknown ) const;

    template <typename Visitor>
    void VisitElementQuadraturePoints( int contactElement,
                                       const mfem::Vector& displacement,
                                       const mfem::Vector& multiplier,
                                       mfem::real_t inverseGamma,
                                       Visitor&& visitor ) const;

    void AssembleContact( const mfem::Vector& displacement,
                          const mfem::Vector& multiplier,
                          mfem::Vector* displacementResidual,
                          mfem::Vector* multiplierResidual,
                          bool assembleJacobian ) const;
    void ResetElementAssembly( int contactElement, bool assembleResidual, bool assembleJacobian ) const;
    void AccumulateContactPointResidual( const ContactPointData& point ) const;
    void AccumulateContactPointJacobian( const ContactPointData& point, mfem::real_t inverseGamma ) const;
    void ScatterContactElement( const ContactElementActivity& activity,
                                mfem::Vector* displacementResidual,
                                mfem::Vector* multiplierResidual,
                                bool assembleJacobian ) const;
    void MaskEssentialElementEntries( bool assembleResidual, bool assembleJacobian ) const;
    void AssembleMultiplierJumpStabilization( const mfem::Vector& multiplier, mfem::Vector* multiplierResidual, bool assembleJacobian ) const;

    // Borrowed dependencies must outlive this operator and its Jacobian wrappers.
    mfem::Operator& mPrimalOperator;
    const mfem::Operator& mPrimalToDisplacement;
    mfem::FiniteElementSpace& mDisplacementSpace;
    const RigidObstacle& mObstacle;
    BoundaryMultiplierSpace& mBoundaryMultiplier;
    mfem::real_t mGamma;
    mfem::real_t mPrimalResidualScale;
    mfem::real_t mMultiplierResidualScale;
    mfem::real_t mDelta;

    // Aliases of the borrowed boundary object, not independently owned spaces.
    // Declaration order keeps block offsets alive until after mJacobian.
    mfem::SubMesh& mContactSubMesh;
    mfem::FiniteElementSpace& mMultiplierSpace;
    mfem::Array<int> mBlockOffsets;
    mfem::Array<int> mEssentialMarker;
    std::vector<MultiplierJump> mMultiplierJumps;
    // Borrowed from mfem::IntRules or the caller; indexed by contact element.
    std::vector<const mfem::IntegrationRule*> mIntegrationRules;

    // Owning handles, ordered so borrowing wrappers are destroyed before their
    // operands. The block/product operators do not own the blocks they reference.
    std::unique_ptr<mfem::SparseMatrix> mDisplacementJacobian;
    std::unique_ptr<mfem::SparseMatrix> mDisplacementMultiplierJacobian;
    std::unique_ptr<mfem::SparseMatrix> mMultiplierDisplacementJacobian;
    std::unique_ptr<mfem::SparseMatrix> mMultiplierJacobian;
    std::unique_ptr<mfem::RAPOperator> mDisplacementContactJacobian;
    std::unique_ptr<mfem::TransposeOperator> mPrimalDisplacementCoupling;
    std::unique_ptr<mfem::ProductOperator> mPrimalMultiplierCoupling;
    std::unique_ptr<mfem::ProductOperator> mMultiplierPrimalCoupling;
    mutable std::unique_ptr<mfem::Operator> mPrimalJacobian;
    std::unique_ptr<mfem::BlockOperator> mJacobian;

    // Reused by const evaluations; this operator is not reentrant or thread-safe.
    mutable mfem::Vector mDisplacement;
    mutable mfem::Vector mDisplacementResidual;
    mutable mfem::Vector mPrimalContactResidual;
    mutable mfem::Vector mElementDisplacement;
    mutable mfem::Vector mElementMultiplier;
    mutable mfem::Vector mDisplacementShape;
    mutable mfem::Vector mMultiplierShape;
    mutable mfem::Vector mReferencePosition;
    mutable mfem::Vector mCurrentPosition;
    mutable SignedDistanceEvaluation mEvaluation;
    mutable mfem::Array<int> mElementVectorDofs;
    mutable mfem::Array<int> mElementMultiplierDofs;
    mutable mfem::Vector mElementResidual;
    mutable mfem::Vector mElementMultiplierResidual;
    mutable mfem::Vector mElementConstantNormalCoupling;
    mutable mfem::DenseMatrix mElementJacobian;
    mutable mfem::DenseMatrix mElementDisplacementMultiplierJacobian;
    mutable mfem::DenseMatrix mElementMultiplierDisplacementJacobian;
    mutable mfem::DenseMatrix mElementMultiplierJacobian;
};
} // namespace plugin
