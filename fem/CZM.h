#pragma once

#include <Eigen/Dense>
#include <mfem.hpp>

// autodiff include
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>
#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>
#include <functional>

namespace plugin
{
class Memorize;

class CZMIntegrator : public mfem::NonlinearFormIntegrator
{
public:
    CZMIntegrator( Memorize& memo ) : mfem::NonlinearFormIntegrator(), mMemo{ memo }
    {
    }

    CZMIntegrator( Memorize& memo, const double sigmaMax, const double tauMax, const double deltaN, const double deltaT )
        : mfem::NonlinearFormIntegrator(), mMemo{ memo }, mSigmaMax{ sigmaMax }, mTauMax{ tauMax }, mDeltaN{ deltaN }, mDeltaT{ deltaT }
    {
        mPhiN = std::exp( 1. ) * mSigmaMax * mDeltaN;
        mPhiT = std::sqrt( std::exp( 1. ) / 2 ) * mTauMax * mDeltaT;
    }

    CZMIntegrator( Memorize& memo, const double sigmaMax, const double tauMax, const double deltaN, const double deltaT, const double phiN, const double phiT )
        : mfem::NonlinearFormIntegrator(),
          mMemo{ memo },
          mSigmaMax{ sigmaMax },
          mTauMax{ tauMax },
          mDeltaN{ deltaN },
          mDeltaT{ deltaT },
          mPhiN{ phiN },
          mPhiT{ phiT }
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

    virtual void Traction( const Eigen::VectorXd& Delta, const mfem::DenseMatrix& Jacobian, const int dim, Eigen::VectorXd& T ) const = 0;

    virtual void TractionStiffTangent( const Eigen::VectorXd& Delta,
                                       const mfem::DenseMatrix& Jacobian,
                                       const int dim,
                                       Eigen::MatrixXd& H ) const = 0;

protected:
    Memorize& mMemo;
    double mSigmaMax{ 0. };
    double mTauMax{ 0. };
    double mDeltaN{ 0. };
    double mDeltaT{ 0. };
    double mPhiN{ 0. };
    double mPhiT{ 0. };
    mfem::Vector shape1, shape2;

    Eigen::MatrixXd mB;
    Eigen::VectorXd u;
};

class ADCZMIntegrator : public CZMIntegrator
{
public:
    ADCZMIntegrator( Memorize& memo, const double sigmaMax, const double tauMax, const double deltaN, const double deltaT, const double phiN, const double phiT )
        : CZMIntegrator( memo, sigmaMax, tauMax, deltaN, deltaT, phiN, phiT )
    {
    }

    virtual void Traction( const Eigen::VectorXd& Delta, const mfem::DenseMatrix& Jacobian, const int dim, Eigen::VectorXd& T ) const override;

    virtual void TractionStiffTangent( const Eigen::VectorXd& Delta,
                                       const mfem::DenseMatrix& Jacobian,
                                       const int dim,
                                       Eigen::MatrixXd& H ) const override;

protected:
    std::function<autodiff::dual2nd( const autodiff::VectorXdual2nd&, const autodiff::VectorXdual2nd& )> potential;
};

class ExponentialCZMIntegrator : public CZMIntegrator
{
public:
    ExponentialCZMIntegrator( Memorize& memo, const double sigmaMax, const double tauMax, const double deltaN, const double deltaT )
        : CZMIntegrator(
              memo, sigmaMax, tauMax, deltaN, deltaT, std::exp( 1. ) * sigmaMax * deltaN, std::sqrt( std::exp( 1. ) / 2 ) * tauMax * deltaT )
    {
    }

    virtual void Traction( const Eigen::VectorXd& Delta, const mfem::DenseMatrix& Jacobian, const int dim, Eigen::VectorXd& T ) const override;

    virtual void TractionStiffTangent( const Eigen::VectorXd& Delta,
                                       const mfem::DenseMatrix& Jacobian,
                                       const int dim,
                                       Eigen::MatrixXd& H ) const override;

protected:
    void DeltaToTNMat( const mfem::DenseMatrix& Jacobian, const int dim, Eigen::MatrixXd& DeltaToTN ) const;
};

class ExponentialADCZMIntegrator : public ADCZMIntegrator
{
public:
    ExponentialADCZMIntegrator( Memorize& memo, const double sigmaMax, const double tauMax, const double deltaN, const double deltaT )
        : ADCZMIntegrator(
              memo, sigmaMax, tauMax, deltaN, deltaT, std::exp( 1. ) * sigmaMax * deltaN, std::sqrt( std::exp( 1. ) / 2 ) * tauMax * deltaT )
    {
        // x: diffX, diffY
        // p: deltaT, deltaN, phiT, phiN, dA1x, dA1y, dA2x, dA2y
        potential = []( const autodiff::VectorXdual2nd& x, const autodiff::VectorXdual2nd& p )
        {
            const autodiff::dual2nd& deltaT = p( 0 );
            const autodiff::dual2nd& deltaN = p( 1 );
            const autodiff::dual2nd& phiT = p( 2 );
            const autodiff::dual2nd& phiN = p( 3 );

            Eigen::Map<const autodiff::VectorXdual2nd> dA1( p.data() + 4, 2 );
            Eigen::Map<const autodiff::VectorXdual2nd> dA2( p.data() + 6, 2 );
            const autodiff::dual2nd q = phiT / phiN;
            const autodiff::dual2nd r = 0.;

            autodiff::VectorXdual2nd directionT = dA1;
            directionT.normalize();

            static Eigen::Rotation2Dd rot( EIGEN_PI / 2 );
            autodiff::VectorXdual2nd directionN = rot.toRotationMatrix() * directionT;
            const autodiff::dual2nd DeltaT = directionT.dot( x );
            const autodiff::dual2nd DeltaN = directionN.dot( x );

            autodiff::dual2nd res = phiN + phiN * autodiff::detail::exp( -DeltaN / deltaN ) *
                                               ( ( autodiff::dual2nd( 1. ) - r + DeltaN / deltaN ) *
                                                     ( autodiff::dual2nd( 1. ) - q ) / ( r - autodiff::dual2nd( 1. ) ) -
                                                 ( q + ( r - q ) / ( r - autodiff::dual2nd( 1. ) ) * DeltaN / deltaN ) *
                                                     autodiff::detail::exp( -DeltaT * DeltaT / deltaT / deltaT ) );
            return res;
        };
    }

    // virtual void matrixB( const int dof1,
    //                       const int dof2,
    //                       const mfem::Vector& shape1,
    //                       const mfem::Vector& shape2,
    //                       const mfem::DenseMatrix& gshape1,
    //                       const mfem::DenseMatrix& gshape2,
    //                       const int dim );
};

class ExponentialRotADCZMIntegrator : public ADCZMIntegrator
{
public:
    ExponentialRotADCZMIntegrator( Memorize& memo, const double sigmaMax, const double tauMax, const double deltaN, const double deltaT )
        : ADCZMIntegrator(
              memo, sigmaMax, tauMax, deltaN, deltaT, std::exp( 1. ) * sigmaMax * deltaN, std::sqrt( std::exp( 1. ) / 2 ) * tauMax * deltaT )
    {
        // x: u1x, u1y, u2x, u2y, du1x, du1y, du2x, du2y
        // p: deltaT, deltaN, phiT, phiN, dA1x, dA1y, dA2x, dA2y
        potential = []( const autodiff::VectorXdual2nd& x, const autodiff::VectorXdual2nd& p )
        {
            Eigen::Map<const autodiff::VectorXdual2nd> U1( x.data(), 2 );
            Eigen::Map<const autodiff::VectorXdual2nd> U2( x.data() + 2, 2 );
            Eigen::Map<const autodiff::VectorXdual2nd> dU1( x.data() + 4, 2 );
            Eigen::Map<const autodiff::VectorXdual2nd> dU2( x.data() + 6, 2 );

            const autodiff::dual2nd& deltaT = p( 0 );
            const autodiff::dual2nd& deltaN = p( 1 );
            const autodiff::dual2nd& phiT = p( 2 );
            const autodiff::dual2nd& phiN = p( 3 );

            Eigen::Map<const autodiff::VectorXdual2nd> dA1( p.data() + 4, 2 );
            Eigen::Map<const autodiff::VectorXdual2nd> dA2( p.data() + 6, 2 );
            const autodiff::dual2nd q = phiT / phiN;
            const autodiff::dual2nd r = 0.;

            autodiff::VectorXdual2nd diff = U1 - U2;
            autodiff::VectorXdual2nd directionT = dA1 + dA2 + dU1 + dU2;
            directionT.normalize();

            static Eigen::Rotation2Dd rot( EIGEN_PI / 2 );
            autodiff::VectorXdual2nd directionN = rot.toRotationMatrix() * directionT;
            const autodiff::dual2nd DeltaT = directionT.dot( diff );
            const autodiff::dual2nd DeltaN = directionN.dot( diff );

            autodiff::dual2nd res = phiN + phiN * autodiff::detail::exp( -DeltaN / deltaN ) *
                                               ( ( autodiff::dual2nd( 1. ) - r + DeltaN / deltaN ) *
                                                     ( autodiff::dual2nd( 1. ) - q ) / ( r - autodiff::dual2nd( 1. ) ) -
                                                 ( q + ( r - q ) / ( r - autodiff::dual2nd( 1. ) ) * DeltaN / deltaN ) *
                                                     autodiff::detail::exp( -DeltaT * DeltaT / deltaT / deltaT ) );
            return res;
        };
    }

    virtual void matrixB( const int dof1,
                          const int dof2,
                          const mfem::Vector& shape1,
                          const mfem::Vector& shape2,
                          const mfem::DenseMatrix& gshape1,
                          const mfem::DenseMatrix& gshape2,
                          const int dim );
};

// class CZMIntegrator : public mfem::NonlinearFormIntegrator
// {
// public:
//     CZMIntegrator( Memorize& memo ) : mfem::NonlinearFormIntegrator(), mMemo{ memo }
//     {
//     }

//     CZMIntegrator( Memorize& memo, const double sigmaMax, const double tauMax, const double deltaN, const double deltaT )
//         : mfem::NonlinearFormIntegrator(), mMemo{ memo }, mSigmaMax{ sigmaMax }, mTauMax{ tauMax }, mDeltaN{ deltaN }, mDeltaT{ deltaT }
//     {
//         mPhiN = std::exp( 1. ) * mSigmaMax * mDeltaN;
//         mPhiT = std::sqrt( std::exp( 1. ) / 2 ) * mTauMax * mDeltaT;
//     }

//     CZMIntegrator( Memorize& memo, const double sigmaMax, const double tauMax, const double deltaN, const double deltaT, const double phiN, const double phiT )
//         : mfem::NonlinearFormIntegrator(),
//           mMemo{ memo },
//           mSigmaMax{ sigmaMax },
//           mTauMax{ tauMax },
//           mDeltaN{ deltaN },
//           mDeltaT{ deltaT },
//           mPhiN{ phiN },
//           mPhiT{ phiT }
//     {
//     }

//     virtual void AssembleFaceVector( const mfem::FiniteElement& el1,
//                                      const mfem::FiniteElement& el2,
//                                      mfem::FaceElementTransformations& Tr,
//                                      const mfem::Vector& elfun,
//                                      mfem::Vector& elvect ) override;

//     virtual void AssembleFaceGrad( const mfem::FiniteElement& el1,
//                                    const mfem::FiniteElement& el2,
//                                    mfem::FaceElementTransformations& Tr,
//                                    const mfem::Vector& elfun,
//                                    mfem::DenseMatrix& elmat ) override;

//     void matrixB( const int dof1,
//                   const int dof2,
//                   const mfem::Vector& shape1,
//                   const mfem::Vector& shape2,
//                   const mfem::DenseMatrix& gshape1,
//                   const mfem::DenseMatrix& gshape2,
//                   const int dim )
//     {
//         if ( dim == 2 )
//         {
//             mB.resize( 8, 2 * ( dof1 + dof2 ) );
//             mB.setZero();

//             for ( int i = 0; i < dof1; i++ )
//             {
//                 for ( int j = 0; j < dim; j++ )
//                 {
//                     mB( j, i + j * dof1 ) = shape1( i );
//                 }
//             }

//             for ( int i = 0; i < dof2; i++ )
//             {
//                 for ( int j = 0; j < dim; j++ )
//                 {
//                     mB( 2 + j, i + j * dof2 + dim * dof1 ) = shape2( i );
//                 }
//             }

//             for ( int i = 0; i < dof1; i++ )
//             {
//                 for ( int j = 0; j < dim; j++ )
//                 {
//                     mB( 4 + j, i + j * dof1 ) = gshape1( i, 0 );
//                 }
//             }

//             for ( int i = 0; i < dof2; i++ )
//             {
//                 for ( int j = 0; j < dim; j++ )
//                 {
//                     mB( 6 + j, i + j * dof2 + dim * dof1 ) = gshape2( i, 0 );
//                 }
//             }
//         }
//         else if ( dim == 3 )
//         {
//             std::cout << "not implemented!\n";
//         }
//     }

//     void DeltaToTNMat( const mfem::DenseMatrix&, const int dim, Eigen::MatrixXd& DeltaToTN ) const;

//     virtual void Traction( const Eigen::VectorXd& Delta, const mfem::DenseMatrix& Jacobian, const int dim, Eigen::VectorXd& T ) const;

//     virtual void TractionStiffTangent( const Eigen::VectorXd& Delta, const mfem::DenseMatrix& Jacobian, const int dim, Eigen::MatrixXd& H ) const;

//     // x: diffX, diffY
//     // p: deltaT, deltaN, phiT, phiN, dA1x, dA1y, dA2x, dA2y
//     static autodiff::dual2nd f( const autodiff::VectorXdual2nd& x, const autodiff::VectorXdual2nd& p )
//     {
//         autodiff::dual2nd res =
//             p( 2 ) + p( 2 ) * autodiff::detail::exp( -x( 1 ) / p( 1 ) ) *
//                          ( ( autodiff::dual2nd( 1. ) - p( 3 ) + x( 1 ) / p( 1 ) ) *
//                                ( autodiff::dual2nd( 1. ) - p( 4 ) ) / ( p( 3 ) - autodiff::dual2nd( 1. ) ) -
//                            ( p( 4 ) + ( p( 3 ) - p( 4 ) ) / ( p( 3 ) - autodiff::dual2nd( 1. ) ) * x( 1 ) / p( 1 ) ) *
//                                autodiff::detail::exp( -x( 0 ) * x( 0 ) / p( 0 ) / p( 0 ) ) );
//         return res;
//     }

//     // x: u1x, u1y, u2x, u2y, du1x, du1y, du2x, du2y
//     // p: deltaT, deltaN, phiT, phiN, dA1x, dA1y, dA2x, dA2y
//     static autodiff::dual2nd fLarge( const autodiff::VectorXdual2nd& x, const autodiff::VectorXdual2nd& p )
//     {
//         Eigen::Map<const autodiff::VectorXdual2nd> U1( x.data(), 2 );
//         Eigen::Map<const autodiff::VectorXdual2nd> U2( x.data() + 2, 2 );
//         Eigen::Map<const autodiff::VectorXdual2nd> dU1( x.data() + 4, 2 );
//         Eigen::Map<const autodiff::VectorXdual2nd> dU2( x.data() + 6, 2 );

//         const autodiff::dual2nd& deltaT = p( 0 );
//         const autodiff::dual2nd& deltaN = p( 1 );
//         const autodiff::dual2nd& phiT = p( 2 );
//         const autodiff::dual2nd& phiN = p( 3 );

//         Eigen::Map<const autodiff::VectorXdual2nd> dA1( p.data() + 4, 2 );
//         Eigen::Map<const autodiff::VectorXdual2nd> dA2( p.data() + 6, 2 );
//         const autodiff::dual2nd q = phiT / phiN;
//         const autodiff::dual2nd r = 0.;

//         autodiff::VectorXdual2nd diff = U1 - U2;
//         autodiff::VectorXdual2nd directionT = dA1 + dA2 + dU1 + dU2;
//         directionT.normalize();

//         static Eigen::Rotation2Dd rot( EIGEN_PI / 2 );
//         autodiff::VectorXdual2nd directionN = rot.toRotationMatrix() * directionT;
//         const autodiff::dual2nd DeltaT = directionT.dot( diff );
//         const autodiff::dual2nd DeltaN = directionN.dot( diff );

//         autodiff::dual2nd res = phiN + phiN * autodiff::detail::exp( -DeltaN / deltaN ) *
//                                            ( ( autodiff::dual2nd( 1. ) - r + DeltaN / deltaN ) *
//                                                  ( autodiff::dual2nd( 1. ) - q ) / ( r - autodiff::dual2nd( 1. ) ) -
//                                              ( q + ( r - q ) / ( r - autodiff::dual2nd( 1. ) ) * DeltaN / deltaN ) *
//                                                  autodiff::detail::exp( -DeltaT * DeltaT / deltaT / deltaT ) );
//         return res;
//     }

// protected:
//     Memorize& mMemo;
//     double mSigmaMax{ 0. };
//     double mTauMax{ 0. };
//     double mDeltaN{ 0. };
//     double mDeltaT{ 0. };
//     double mPhiN{ 0. };
//     double mPhiT{ 0. };
//     mfem::Vector shape1, shape2;

//     Eigen::MatrixXd mB;
//     Eigen::VectorXd u;
// };

class LinearCZMIntegrator : public CZMIntegrator
{
public:
    LinearCZMIntegrator( Memorize& memo ) : CZMIntegrator( memo )
    {
    }

    LinearCZMIntegrator( Memorize& memo, const double sigmaMax, const double tauMax, const double deltaN, const double deltaT, const double phiN, const double phiT )
        : CZMIntegrator( memo, sigmaMax, tauMax, deltaN, deltaT, phiN, phiT )
    {
        mDeltaNMax = 2 * phiN / sigmaMax;
        mDeltaTMax = 2 * phiT / tauMax;
    }

    virtual void Traction( const Eigen::VectorXd& Delta, const mfem::DenseMatrix& Jacobian, const int dim, Eigen::VectorXd& T ) const;

    virtual void TractionStiffTangent( const Eigen::VectorXd& Delta, const mfem::DenseMatrix& Jacobian, const int dim, Eigen::MatrixXd& H ) const;

protected:
    double mDeltaNMax{ 0. };
    double mDeltaTMax{ 0. };
};
} // namespace plugin