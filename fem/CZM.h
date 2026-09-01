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
class CZMIntegrator : public NonlinearFormIntegratorLambda
{
protected:
    struct PointData
    {
        PointData( const double delta_n, const double delta_t1, const double delta_t2 )
            : delta_n_prev( delta_n ), delta_t1_prev( delta_t1 ), delta_t2_prev( delta_t2 )
        {
        }
        double delta_n_prev{0.};
        double delta_t1_prev{0.};
        double delta_t2_prev{0.};
    };

public:
    CZMIntegrator( Memorize& memo ) : NonlinearFormIntegratorLambda(), mMemo{memo}
    {
    }

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

protected:
    void Update( const int gauss, const double delta_n, const double delta_t1, const double delta_t2 = 0. ) const;

protected:
    Memorize& mMemo;
    mfem::Vector shape1, shape2;

    Eigen::MatrixXd mB;
    Eigen::VectorXd u;
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

    double xi_n{0};
    double xi_t{0};
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

protected:
    mfem::Coefficient* mSigmaMax{nullptr};
    mfem::Coefficient* mTauMax{nullptr};
    mfem::Coefficient* mDeltaN{nullptr};
    mfem::Coefficient* mDeltaT{nullptr};

    ExponentialCZMConst mCZMLawConst;

    double xi_n{0};
    double xi_t{0};
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