#pragma once

#include <Eigen/Dense>
#include <mfem.hpp>

#include "FEMPlugin.h"
#include <array>
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>
#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>
#include <functional>
#include <string>

namespace plugin
{
class Memorize;

struct ExponentialCZMConst
{
    void update_phi()
    {
        phi_n = std::exp( 1. ) * sigma_max * delta_n;
        phi_t = std::sqrt( std::exp( 1. ) / 2 ) * tau_max * delta_t;
    }
    double sigma_max{0};
    double tau_max{0};
    double delta_n{0};
    double delta_t{0};
    double phi_n{0};
    double phi_t{0};
};

struct CZMEvaluation
{
    Eigen::VectorXd traction;
    Eigen::MatrixXd tangent;
};

struct CZMHistoryState
{
    double maximum_normal_opening{ 0. };
    double maximum_tangential_opening{ 0. };
    double normal_unloading_stiffness{ 0. };
    double tangential_unloading_stiffness{ 0. };
    bool has_normal_history{ false };
    bool has_tangential_history{ false };

    double normal_opening{ 0. };
    double tangential_opening_1{ 0. };
    double tangential_opening_2{ 0. };
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

    void BeginStep();
    void CommitStep();
    void RollbackStep();
    void RevertStep();

    CZMEvaluation EvaluateTrial( const Eigen::VectorXd& local_separation,
                                 const CZMEvaluation& envelope,
                                 const double normal_length,
                                 const double tangential_length,
                                 const double normal_damping,
                                 const double tangential_damping,
                                 const double delta_lambda );

private:
    CZMHistoryState mCommitted;
    CZMHistoryState mTrial;
    std::array<CZMHistoryState, 20> mCommittedHistory;
    std::size_t mCommittedHistorySize{ 0 };
};

CZMEvaluation EvaluateExponentialCZMEnvelope( const ExponentialCZMConst& law, const Eigen::VectorXd& local_separation );

CZMEvaluation EvaluateExponentialCZMEnvelopeAutodiff( const ExponentialCZMConst& law, const Eigen::VectorXd& local_separation );

CZMEvaluation EvaluateIrreversibleExponentialCZM( const ExponentialCZMConst& law,
                                                  const Eigen::VectorXd& local_separation,
                                                  CZMHistory& history,
                                                  const double normal_damping = 0.,
                                                  const double tangential_damping = 0.,
                                                  const double delta_lambda = 0. );

class CZMIntegrator : public NonlinearFormIntegratorLambda
{
public:
    CZMIntegrator( Memorize& memo );
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

    virtual void Traction( const Eigen::VectorXd& Delta, const int gauss, const int dim, Eigen::VectorXd& T ) const = 0;

    virtual void TractionStiffTangent( const Eigen::VectorXd& Delta, const int gauss, const int dim, Eigen::MatrixXd& H ) const = 0;

    // TODO: should be pure virtual
    virtual void EvalCZMLaw( mfem::ElementTransformation& Tr, const mfem::IntegrationPoint& ip )
    {
    }

    void SetDamping( const double normal, const double tangential );

    virtual bool SupportsDamping() const
    {
        return true;
    }

    virtual void BeginStep() override;
    virtual void CommitStep() override;
    virtual void RollbackStep() override;
    virtual void RevertStep() override;

protected:
    CZMHistory& GetHistory( const int gauss ) const;
    CZMEvaluation EvaluateLocalLaw( const ExponentialCZMConst& law, const Eigen::VectorXd& local_separation, const int gauss ) const;

    Memorize& mMemo;
    mfem::Vector shape1, shape2;

    Eigen::MatrixXd mB;
    Eigen::VectorXd u;

    double xi_n{ 0. };
    double xi_t{ 0. };

private:
    template <typename Visitor>
    void VisitHistory( Visitor&& visitor )
    {
        mMemo.VisitFacePointData(
            [this, &visitor]( util::AnyMap& point_data )
            {
                auto history = point_data.get_val<CZMHistory>( mStateKey );
                if ( history )
                {
                    visitor( history->get() );
                }
            } );
    }

    std::string mStateKey;
    int mStepDepth{ 0 };
    bool mStepRejected{ false };
};

class LinearCZMIntegrator : public CZMIntegrator
{
public:
    LinearCZMIntegrator( Memorize& memo ) : CZMIntegrator( memo )
    {
    }

    LinearCZMIntegrator( Memorize& memo, const double sigmaMax, const double tauMax, const double deltaN, const double deltaT, const double phiN, const double phiT )
        : CZMIntegrator( memo )
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

    virtual void Traction( const Eigen::VectorXd& Delta, const int gauss, const int dim, Eigen::VectorXd& T ) const;

    virtual void TractionStiffTangent( const Eigen::VectorXd& Delta, const int gauss, const int dim, Eigen::MatrixXd& H ) const;

protected:
    double mDeltaNMax{0.};
    double mDeltaTMax{0.};
    double mPhiN{0.};
    double mPhiT{0.};
    double mDeltaN{0.};
    double mDeltaT{0.};
    double mSigmaMax{0.};
    double mTauMax{0.};
};

class ExponentialCZMIntegrator : public CZMIntegrator
{
public:
    ExponentialCZMIntegrator( Memorize& memo, mfem::Coefficient& sigmaMax, mfem::Coefficient& tauMax, mfem::Coefficient& deltaN, mfem::Coefficient& deltaT )
        : CZMIntegrator( memo ), mSigmaMax( &sigmaMax ), mTauMax( &tauMax ), mDeltaN( &deltaN ), mDeltaT( &deltaT )
    {
    }

    virtual void EvalCZMLaw( mfem::ElementTransformation& Tr, const mfem::IntegrationPoint& ip ) override;

    virtual void Traction( const Eigen::VectorXd& Delta, const int gauss, const int dim, Eigen::VectorXd& T ) const;

    virtual void TractionStiffTangent( const Eigen::VectorXd& Delta, const int gauss, const int dim, Eigen::MatrixXd& H ) const;

protected:
    mfem::Coefficient* mSigmaMax{nullptr};
    mfem::Coefficient* mTauMax{nullptr};
    mfem::Coefficient* mDeltaN{nullptr};
    mfem::Coefficient* mDeltaT{nullptr};
    ExponentialCZMConst mCZMLawConst;
};

class ADCZMIntegrator : public CZMIntegrator
{
public:
    ADCZMIntegrator( Memorize& memo ) : CZMIntegrator( memo )
    {
    }

    virtual void Traction( const Eigen::VectorXd& Delta, const int gauss, const int dim, Eigen::VectorXd& T ) const;

    virtual void TractionStiffTangent( const Eigen::VectorXd& Delta, const int gauss, const int dim, Eigen::MatrixXd& H ) const;

protected:
    void EvaluatePotential( const Eigen::VectorXd& Delta, const int gauss, Eigen::VectorXd& traction, Eigen::MatrixXd& tangent ) const;

    std::function<autodiff::dual2nd( const autodiff::VectorXdual2nd&, const int )> potential;
};

class ExponentialADCZMIntegrator : public ADCZMIntegrator
{
public:
    ExponentialADCZMIntegrator( Memorize& memo,
                                mfem::Coefficient& sigmaMax,
                                mfem::Coefficient& tauMax,
                                mfem::Coefficient& deltaN,
                                mfem::Coefficient& deltaT );

    virtual void EvalCZMLaw( mfem::ElementTransformation& Tr, const mfem::IntegrationPoint& ip ) override;

    virtual void Traction( const Eigen::VectorXd& Delta, const int gauss, const int dim, Eigen::VectorXd& T ) const override;

    virtual void TractionStiffTangent( const Eigen::VectorXd& Delta, const int gauss, const int dim, Eigen::MatrixXd& H ) const override;

protected:
    mfem::Coefficient* mSigmaMax{nullptr};
    mfem::Coefficient* mTauMax{nullptr};
    mfem::Coefficient* mDeltaN{nullptr};
    mfem::Coefficient* mDeltaT{nullptr};

    ExponentialCZMConst mCZMLawConst;
};

// class OrtizIrreversibleADCZMIntegrator : public ADCZMIntegrator
// {
// public:
//     OrtizIrreversibleADCZMIntegrator( Memorize& memo );

// protected:
//     double mBeta{ .2 };
//     double mDeltaC{ 0. };
//     double mSgimaC{ 0. };
// };

class ExponentialRotADCZMIntegrator : public ExponentialADCZMIntegrator
{
public:
    ExponentialRotADCZMIntegrator( Memorize& memo,
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

    virtual void Traction( const Eigen::VectorXd& Delta, const int gauss, const int dim, Eigen::VectorXd& T ) const override;

    virtual void TractionStiffTangent( const Eigen::VectorXd& Delta, const int gauss, const int dim, Eigen::MatrixXd& H ) const override;

    virtual void matrixB( const int dof1,
                          const int dof2,
                          const mfem::Vector& shape1,
                          const mfem::Vector& shape2,
                          const mfem::DenseMatrix& gshape1,
                          const mfem::DenseMatrix& gshape2,
                          const int dim );
};

void DeltaToTNMat( const mfem::DenseMatrix& Jacobian, Eigen::MatrixXd& DeltaToTN );
} // namespace plugin
