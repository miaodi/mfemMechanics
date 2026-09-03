#pragma once

#include <Eigen/Dense>
#include <mfem.hpp>

#include "FEMPlugin.h"
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>
#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>
#include <functional>

namespace plugin
{
struct ExponentialCZMConst
{
    void update_phi()
    {
        phi_n = std::exp( 1. ) * sigma_max * delta_n;
        phi_t = std::sqrt( std::exp( 1. ) / 2 ) * tau_max * delta_t;
    }
    mfem::real_t sigma_max{ 0 };
    mfem::real_t tau_max{ 0 };
    mfem::real_t delta_n{ 0 };
    mfem::real_t delta_t{ 0 };
    mfem::real_t phi_n{ 0 };
    mfem::real_t phi_t{ 0 };
};

struct CZMEvaluation
{
    Eigen::VectorXr traction;
    Eigen::MatrixXr tangent;
};

struct CZMHistoryState
{
    mfem::real_t maximum_normal_opening{ 0. };
    mfem::real_t maximum_tangential_opening{ 0. };
    mfem::real_t normal_unloading_stiffness{ 0. };
    mfem::real_t tangential_unloading_stiffness{ 0. };
    bool has_normal_history{ false };
    bool has_tangential_history{ false };

    mfem::real_t normal_opening{ 0. };
    mfem::real_t tangential_opening_1{ 0. };
    mfem::real_t tangential_opening_2{ 0. };
};

class CZMHistory
{
public:
    const CZMHistoryState& CommittedState() const
    {
        return mCommitted;
    }

    const CZMHistoryState& TrialState() const
    {
        return mTrial;
    }

    void BeginStep() noexcept;
    void CommitStep() noexcept;
    void RollbackStep() noexcept;

    CZMEvaluation EvaluateTrial( const Eigen::VectorXr& local_separation,
                                 const CZMEvaluation& envelope,
                                 const mfem::real_t normal_length,
                                 const mfem::real_t tangential_length,
                                 const mfem::real_t normal_damping,
                                 const mfem::real_t tangential_damping,
                                 const mfem::real_t delta_lambda );

private:
    CZMHistoryState mCommitted;
    CZMHistoryState mTrial;
};

using CZMHistoryPointStorage = IntegrationPointStorage<NoIntegrationPointState, CZMHistory>;

CZMEvaluation EvaluateExponentialCZMEnvelope( const ExponentialCZMConst& law, const Eigen::VectorXr& local_separation );

CZMEvaluation EvaluateExponentialCZMEnvelopeAutodiff( const ExponentialCZMConst& law, const Eigen::VectorXr& local_separation );

CZMEvaluation EvaluateIrreversibleExponentialCZM( const ExponentialCZMConst& law,
                                                  const Eigen::VectorXr& local_separation,
                                                  CZMHistory& history,
                                                  const mfem::real_t normal_damping = 0.,
                                                  const mfem::real_t tangential_damping = 0.,
                                                  const mfem::real_t delta_lambda = 0. );

class CZMIntegrator : public StepAwareNonlinearFormIntegrator
{
public:
    /// Geometry-only construction for cohesive laws with no irreversible state.
    explicit CZMIntegrator( IntegrationPointStorageBase& pointStorage );
    /// Stateful construction; use separate storage for each irreversible cohesive integrator.
    explicit CZMIntegrator( CZMHistoryPointStorage& pointStorage );
    CZMIntegrator( const CZMIntegrator& ) = delete;
    CZMIntegrator& operator=( const CZMIntegrator& ) = delete;

    virtual void AssembleFaceVector( const mfem::FiniteElement& el1,
                                     const mfem::FiniteElement& el2,
                                     mfem::FaceElementTransformations& Tr,
                                     const mfem::Vector& elfun,
                                     mfem::Vector& elvect ) override;

    virtual void AssembleFaceGrad( const mfem::FiniteElement& el1,
                                   const mfem::FiniteElement& el2,
                                   mfem::FaceElementTransformations& Tr,
                                   const mfem::Vector& elfun,
                                   mfem::DenseMatrix& elmat ) override;

    virtual void matrixB( const int dof1,
                          const int dof2,
                          const mfem::Vector& shape1,
                          const mfem::Vector& shape2,
                          const mfem::DenseMatrix& gshape1,
                          const mfem::DenseMatrix& gshape2,
                          const int dim );

    virtual void Traction( const Eigen::VectorXr& Delta, const int gauss, const int dim, Eigen::VectorXr& T ) const = 0;

    virtual void TractionStiffTangent( const Eigen::VectorXr& Delta, const int gauss, const int dim, Eigen::MatrixXr& H ) const = 0;

    // TODO: should be pure virtual
    virtual void EvalCZMLaw( mfem::ElementTransformation& Tr, const mfem::IntegrationPoint& ip )
    {
    }

    void SetDamping( const mfem::real_t normal, const mfem::real_t tangential );

    virtual bool SupportsDamping() const
    {
        return true;
    }

    virtual void BeginStep() noexcept override;
    virtual void CommitStep() noexcept override;
    virtual void RollbackStep() noexcept override;
    bool CanCommitStep() const noexcept override
    {
        return !mStepRejected;
    }

protected:
    CZMHistory& GetHistory( const int gauss ) const;
    CZMEvaluation EvaluateIrreversibleLocalLaw( const ExponentialCZMConst& law,
                                                const Eigen::VectorXr& local_separation,
                                                const int gauss ) const;

    IntegrationPointStorageBase& mPointStorage;
    mfem::Vector shape1, shape2;

    Eigen::MatrixXr mB;
    Eigen::VectorXr u;

    mfem::real_t xi_n{ 0. };
    mfem::real_t xi_t{ 0. };

private:
    template <typename Visitor>
    void VisitHistory( Visitor&& visitor )
    {
        if ( mHistoryPointStorage == nullptr )
        {
            return;
        }

        mHistoryPointStorage->VisitFaceStates( [&visitor]( CZMHistory& history ) { visitor( history ); } );
    }

    CZMHistoryPointStorage* mHistoryPointStorage{ nullptr };
    int mStepDepth{ 0 };
    bool mStepRejected{ false };
};

class LinearCZMIntegrator : public CZMIntegrator
{
public:
    LinearCZMIntegrator( IntegrationPointStorageBase& pointStorage ) : CZMIntegrator( pointStorage )
    {
    }

    LinearCZMIntegrator( IntegrationPointStorageBase& pointStorage,
                         const mfem::real_t sigmaMax,
                         const mfem::real_t tauMax,
                         const mfem::real_t deltaN,
                         const mfem::real_t deltaT,
                         const mfem::real_t phiN,
                         const mfem::real_t phiT )
        : CZMIntegrator( pointStorage )
    {
        mPhiN = std::exp( 1. ) * sigmaMax * deltaN;
        mPhiT = std::sqrt( std::exp( 1. ) / 2 ) * tauMax * deltaT;
        mDeltaNMax = 2 * phiN / sigmaMax;
        mDeltaTMax = 2 * phiT / tauMax;
        mDeltaN = deltaN;
        mDeltaT = deltaT;
        mSigmaMax = sigmaMax;
        mTauMax = tauMax;
    }

    virtual void Traction( const Eigen::VectorXr& Delta, const int gauss, const int dim, Eigen::VectorXr& T ) const;

    virtual void TractionStiffTangent( const Eigen::VectorXr& Delta, const int gauss, const int dim, Eigen::MatrixXr& H ) const;

protected:
    mfem::real_t mDeltaNMax{ 0. };
    mfem::real_t mDeltaTMax{ 0. };
    mfem::real_t mPhiN{ 0. };
    mfem::real_t mPhiT{ 0. };
    mfem::real_t mDeltaN{ 0. };
    mfem::real_t mDeltaT{ 0. };
    mfem::real_t mSigmaMax{ 0. };
    mfem::real_t mTauMax{ 0. };
};

class ExponentialCZMIntegrator : public CZMIntegrator
{
public:
    ExponentialCZMIntegrator( CZMHistoryPointStorage& pointStorage,
                              mfem::Coefficient& sigmaMax,
                              mfem::Coefficient& tauMax,
                              mfem::Coefficient& deltaN,
                              mfem::Coefficient& deltaT )
        : CZMIntegrator( pointStorage ), mSigmaMax( &sigmaMax ), mTauMax( &tauMax ), mDeltaN( &deltaN ), mDeltaT( &deltaT )
    {
    }

    virtual void EvalCZMLaw( mfem::ElementTransformation& Tr, const mfem::IntegrationPoint& ip ) override;

    virtual void Traction( const Eigen::VectorXr& Delta, const int gauss, const int dim, Eigen::VectorXr& T ) const;

    virtual void TractionStiffTangent( const Eigen::VectorXr& Delta, const int gauss, const int dim, Eigen::MatrixXr& H ) const;

protected:
    mfem::Coefficient* mSigmaMax{ nullptr };
    mfem::Coefficient* mTauMax{ nullptr };
    mfem::Coefficient* mDeltaN{ nullptr };
    mfem::Coefficient* mDeltaT{ nullptr };
    ExponentialCZMConst mCZMLawConst;
};

class ADCZMIntegrator : public CZMIntegrator
{
public:
    /// Geometry-only construction used by reversible autodiff laws.
    ADCZMIntegrator( IntegrationPointStorageBase& pointStorage ) : CZMIntegrator( pointStorage )
    {
    }

    virtual void Traction( const Eigen::VectorXr& Delta, const int gauss, const int dim, Eigen::VectorXr& T ) const;

    virtual void TractionStiffTangent( const Eigen::VectorXr& Delta, const int gauss, const int dim, Eigen::MatrixXr& H ) const;

protected:
    ADCZMIntegrator( CZMHistoryPointStorage& pointStorage ) : CZMIntegrator( pointStorage )
    {
    }

    void EvaluatePotential( const Eigen::VectorXr& Delta, const int gauss, Eigen::VectorXr& traction, Eigen::MatrixXr& tangent ) const;

    std::function<autodiff::dual2nd( const autodiff::VectorXdual2nd&, const int )> potential;
};

class ExponentialADCZMIntegrator : public ADCZMIntegrator
{
public:
    ExponentialADCZMIntegrator( CZMHistoryPointStorage& pointStorage,
                                mfem::Coefficient& sigmaMax,
                                mfem::Coefficient& tauMax,
                                mfem::Coefficient& deltaN,
                                mfem::Coefficient& deltaT );

    virtual void EvalCZMLaw( mfem::ElementTransformation& Tr, const mfem::IntegrationPoint& ip ) override;

    virtual void Traction( const Eigen::VectorXr& Delta, const int gauss, const int dim, Eigen::VectorXr& T ) const override;

    virtual void TractionStiffTangent( const Eigen::VectorXr& Delta, const int gauss, const int dim, Eigen::MatrixXr& H ) const override;

protected:
    ExponentialADCZMIntegrator( IntegrationPointStorageBase& pointStorage,
                                mfem::Coefficient& sigmaMax,
                                mfem::Coefficient& tauMax,
                                mfem::Coefficient& deltaN,
                                mfem::Coefficient& deltaT );

    void InitializePotential();

    mfem::Coefficient* mSigmaMax{ nullptr };
    mfem::Coefficient* mTauMax{ nullptr };
    mfem::Coefficient* mDeltaN{ nullptr };
    mfem::Coefficient* mDeltaT{ nullptr };

    ExponentialCZMConst mCZMLawConst;
};

// class OrtizIrreversibleADCZMIntegrator : public ADCZMIntegrator
// {
// public:
//     OrtizIrreversibleADCZMIntegrator( CZMHistoryPointStorage& pointStorage );

// protected:
//     mfem::real_t mBeta{ .2 };
//     mfem::real_t mDeltaC{ 0. };
//     mfem::real_t mSgimaC{ 0. };
// };

class ExponentialRotADCZMIntegrator : public ExponentialADCZMIntegrator
{
public:
    ExponentialRotADCZMIntegrator( IntegrationPointStorageBase& pointStorage,
                                   mfem::Coefficient& sigmaMax,
                                   mfem::Coefficient& tauMax,
                                   mfem::Coefficient& deltaN,
                                   mfem::Coefficient& deltaT );

    // The rotating law uses an eight-component generalized input in 2D. It
    // intentionally remains reversible until a consistent generalized-history
    // formulation is available.
    virtual bool SupportsDamping() const override
    {
        return false;
    }

    virtual void Traction( const Eigen::VectorXr& Delta, const int gauss, const int dim, Eigen::VectorXr& T ) const override;

    virtual void TractionStiffTangent( const Eigen::VectorXr& Delta, const int gauss, const int dim, Eigen::MatrixXr& H ) const override;

    virtual void matrixB( const int dof1,
                          const int dof2,
                          const mfem::Vector& shape1,
                          const mfem::Vector& shape2,
                          const mfem::DenseMatrix& gshape1,
                          const mfem::DenseMatrix& gshape2,
                          const int dim );
};

void DeltaToTNMat( const mfem::DenseMatrix& Jacobian, Eigen::MatrixXr& DeltaToTN );
} // namespace plugin
