#include "CZM.h"
#include "FEMPlugin.h"
#include "Solvers.h"

namespace plugin
{
void CZMIntegrator::AssembleFaceVector( const mfem::FiniteElement& el1,
                                        const mfem::FiniteElement& el2,
                                        mfem::FaceElementTransformations& Tr,
                                        const mfem::Vector& elfun,
                                        mfem::Vector& elvect )
{
    int vdim = Tr.GetSpaceDim();
    int dof1 = el1.GetDof();
    int dof2 = el2.GetDof();
    int dof = dof1 + dof2;
    MFEM_ASSERT( Tr.Elem2No >= 0, "CZMIntegrator is an internal bdr integrator" );
    elvect.SetSize( dof * vdim );
    elvect = 0.0;
    Eigen::Map<Eigen::VectorXd> eigenVec( elvect.GetData(), elvect.Size() );
    Eigen::Map<const Eigen::VectorXd> u( elfun.GetData(), elfun.Size() );

    const mfem::IntegrationRule* ir = IntRule;
    if ( ir == NULL )
    {
        int intorder = 2 * el1.GetOrder();
        ir = &mfem::IntRules.Get( Tr.GetGeometryType(), intorder );
    }

    mMemo.InitializeFace( el1, el2, Tr, *ir );

    for ( int i = 0; i < ir->GetNPoints(); i++ )
    {
        // // Set the integration point in the face and the neighboring element
        // const mfem::IntegrationPoint& ip = ir->IntPoint( i );
        // Tr.SetAllIntPoints( &ip );
        // mfem::Vector phy;
        // Tr.Transform( ip, phy );
        // if ( std::abs( phy( 1 ) ) > 1e-10 )
        //     continue;

        const mfem::Vector& shape1 = mMemo.GetFace1Shape( i );
        const mfem::Vector& shape2 = mMemo.GetFace2Shape( i );

        const mfem::DenseMatrix& gshape1 = mMemo.GetFace1GShape( i );
        const mfem::DenseMatrix& gshape2 = mMemo.GetFace2GShape( i );
        matrixB( dof1, dof2, shape1, shape2, gshape1, gshape2, vdim );
        Eigen::VectorXd Delta = mB * u;
        Eigen::VectorXd T;
        Traction( Delta, mMemo.GetFaceJacobian( i ), vdim, T );
        eigenVec += mB.transpose() * T * mMemo.GetFaceWeight( i );
    }
}

void CZMIntegrator::AssembleFaceGrad( const mfem::FiniteElement& el1,
                                      const mfem::FiniteElement& el2,
                                      mfem::FaceElementTransformations& Tr,
                                      const mfem::Vector& elfun,
                                      mfem::DenseMatrix& elmat )
{
    int vdim = Tr.GetSpaceDim();
    int dof1 = el1.GetDof();
    int dof2 = el2.GetDof();
    int dof = dof1 + dof2;
    MFEM_ASSERT( Tr.Elem2No >= 0, "CZMIntegrator is an internal bdr integrator" );

    elmat.SetSize( dof * vdim );
    elmat = 0.0;
    Eigen::Map<Eigen::MatrixXd> eigenMat( elmat.Data(), dof * vdim, dof * vdim );
    Eigen::Map<const Eigen::VectorXd> u( elfun.GetData(), elfun.Size() );

    const mfem::IntegrationRule* ir = IntRule;
    if ( ir == NULL )
    {
        int intorder = 2 * el1.GetOrder();
        ir = &mfem::IntRules.Get( Tr.GetGeometryType(), intorder );
    }
    mMemo.InitializeFace( el1, el2, Tr, *ir );
    for ( int i = 0; i < ir->GetNPoints(); i++ )
    {
        // // Set the integration point in the face and the neighboring element
        // const mfem::IntegrationPoint& ip = ir->IntPoint( i );
        // Tr.SetAllIntPoints( &ip );
        // mfem::Vector phy;
        // Tr.Transform( ip, phy );
        // if ( std::abs( phy( 1 ) ) > 1e-10 )
        //     continue;

        const mfem::Vector& shape1 = mMemo.GetFace1Shape( i );
        const mfem::Vector& shape2 = mMemo.GetFace2Shape( i );

        const mfem::DenseMatrix& gshape1 = mMemo.GetFace1GShape( i );
        const mfem::DenseMatrix& gshape2 = mMemo.GetFace2GShape( i );
        matrixB( dof1, dof2, shape1, shape2, gshape1, gshape2, vdim );
        Eigen::VectorXd Delta = mB * u;

        Eigen::MatrixXd H;
        TractionStiffTangent( Delta, mMemo.GetFaceJacobian( i ), vdim, H );
        eigenMat += mB.transpose() * H * mB * mMemo.GetFaceWeight( i );
    }
}

void CZMIntegrator::matrixB( const int dof1,
                             const int dof2,
                             const mfem::Vector& shape1,
                             const mfem::Vector& shape2,
                             const mfem::DenseMatrix& gshape1,
                             const mfem::DenseMatrix& gshape2,
                             const int dim )
{
    mB.resize( dim, dim * ( dof1 + dof2 ) );
    mB.setZero();

    for ( int i = 0; i < dof1; i++ )
    {
        for ( int j = 0; j < dim; j++ )
        {
            mB( j, i + j * dof1 ) = shape1( i );
        }
    }
    for ( int i = 0; i < dof2; i++ )
    {
        for ( int j = 0; j < dim; j++ )
        {
            mB( j, i + j * dof2 + dim * dof1 ) = -shape2( i );
        }
    }
}

void LinearCZMIntegrator::Traction( const Eigen::VectorXd& Delta, const mfem::DenseMatrix& Jacobian, const int dim, Eigen::VectorXd& T ) const
{
    if ( dim == 2 )
    {
        T.resize( 2 );
        // Tt
        if ( std::abs( Delta( 0 ) ) <= mDeltaT )
        {
            T( 0 ) = mTauMax * Delta( 0 ) / mDeltaT;
        }
        else if ( mDeltaT < Delta( 0 ) && Delta( 0 ) <= mDeltaTMax )
        {
            T( 0 ) = mTauMax * ( mDeltaTMax - Delta( 0 ) ) / ( mDeltaTMax - mDeltaT );
        }
        else if ( -mDeltaT > Delta( 0 ) && Delta( 0 ) >= -mDeltaTMax )
        {
            T( 0 ) = -mTauMax * ( mDeltaTMax + Delta( 0 ) ) / ( mDeltaTMax - mDeltaT );
        }
        else
        {
            T( 0 ) = 0;
        }
        // Tn
        if ( Delta( 1 ) <= mDeltaN )
        {
            T( 1 ) = mSigmaMax * Delta( 1 ) / mDeltaN;
        }
        else if ( mDeltaN < Delta( 1 ) && Delta( 1 ) <= mDeltaNMax )
        {
            T( 1 ) = mSigmaMax * ( mDeltaNMax - Delta( 1 ) ) / ( mDeltaNMax - mDeltaN );
        }
        else
        {
            T( 1 ) = 0;
        }
    }
}

void LinearCZMIntegrator::TractionStiffTangent( const Eigen::VectorXd& Delta,
                                                const mfem::DenseMatrix& Jacobian,
                                                const int dim,
                                                Eigen::MatrixXd& H ) const
{
    if ( dim == 2 )
    {
        H.resize( 2, 2 );
        H( 1, 0 ) = H( 0, 1 ) = 0.;
        // Tt
        if ( std::abs( Delta( 0 ) ) <= mDeltaT )
        {
            H( 0, 0 ) = mTauMax / mDeltaT;
        }
        else if ( mDeltaT < Delta( 0 ) && Delta( 0 ) <= mDeltaTMax )
        {
            H( 0, 0 ) = -mTauMax / ( mDeltaTMax - mDeltaT );
        }
        else if ( -mDeltaT > Delta( 0 ) && Delta( 0 ) >= -mDeltaTMax )
        {
            H( 0, 0 ) = -mTauMax / ( mDeltaTMax - mDeltaT );
        }
        else
        {
            H( 0, 0 ) = 0;
        }
        // Tn
        if ( Delta( 1 ) <= mDeltaN )
        {
            H( 1, 1 ) = mSigmaMax / mDeltaN;
        }
        else if ( mDeltaN < Delta( 1 ) && Delta( 1 ) <= mDeltaNMax )
        {
            H( 1, 1 ) = -mSigmaMax / ( mDeltaNMax - mDeltaN );
        }
        else
        {
            H( 1, 1 ) = 0;
        }
    }
}

void ExponentialCZMIntegrator::DeltaToTNMat( const mfem::DenseMatrix& Jacobian, const int dim, Eigen::MatrixXd& DeltaToTN ) const
{
    DeltaToTN.resize( dim, dim );
    if ( dim == 2 )
    {
        Eigen::Map<const Eigen::Matrix<double, 2, 1>> Jac( Jacobian.Data() );
        static Eigen::Rotation2Dd rot( EIGEN_PI / 2 );
        DeltaToTN.col( 0 ) = Jac;
        DeltaToTN.col( 0 ).normalize();
        DeltaToTN.col( 1 ) = rot.toRotationMatrix() * DeltaToTN.col( 0 );
    }
    else if ( dim == 3 )
    {
        Eigen::Map<const Eigen::Matrix<double, 3, 2>> Jac( Jacobian.Data() );
        DeltaToTN.col( 0 ) = Jac.col( 0 );
        DeltaToTN.col( 0 ).normalize();
        DeltaToTN.col( 2 ) = Jac.col( 1 ).cross( Jac.col( 0 ) );
        DeltaToTN.col( 2 ).normalize();
        Eigen::Map<const Eigen::Matrix<double, 3, 3>> DeltaToTN33( DeltaToTN.data() );
        DeltaToTN.col( 1 ) = DeltaToTN33.col( 2 ).cross( DeltaToTN33.col( 0 ) );
    }
}

void ExponentialCZMIntegrator::Traction( const Eigen::VectorXd& Delta, const mfem::DenseMatrix& Jacobian, const int dim, Eigen::VectorXd& T ) const
{
    double q = mPhiT / mPhiN;
    double r = 0.;
    Eigen::MatrixXd DeltaToTN;
    DeltaToTNMat( Jacobian, dim, DeltaToTN );
    Eigen::VectorXd DeltaRot = DeltaToTN.transpose() * Delta;
    if ( dim == 2 )
    {
        T.resize( 2 );
        // Tt
        T( 0 ) = 2 * DeltaRot( 0 ) * exp( -DeltaRot( 1 ) / mDeltaN - DeltaRot( 0 ) * DeltaRot( 0 ) / mDeltaT / mDeltaT ) *
                 mPhiN * ( q + DeltaRot( 1 ) * ( r - q ) / mDeltaN / ( r - 1 ) ) / mDeltaT / mDeltaT;
        // Tn
        T( 1 ) = mPhiN / mDeltaN * exp( -DeltaRot( 1 ) / mDeltaN ) *
                 ( DeltaRot( 1 ) / mDeltaN * exp( -DeltaRot( 0 ) * DeltaRot( 0 ) / mDeltaT / mDeltaT ) +
                   ( 1 - q ) / ( r - 1 ) * ( 1 - exp( -DeltaRot( 0 ) * DeltaRot( 0 ) / mDeltaT / mDeltaT ) ) *
                       ( r - DeltaRot( 1 ) / mDeltaN ) );
    }
    else if ( dim == 3 )
    {
        T.resize( 3 );
        // Tt1
        T( 0 ) = ( 2 * DeltaRot( 0 ) *
                   exp( -( DeltaRot( 2 ) / mDeltaN ) - ( pow( DeltaRot( 0 ), 2 ) + pow( DeltaRot( 1 ), 2 ) ) / pow( mDeltaT, 2 ) ) *
                   mPhiN * ( q + ( DeltaRot( 2 ) * ( r - q ) ) / ( mDeltaN * ( r - 1 ) ) ) ) /
                 pow( mDeltaT, 2 );
        // Tt2
        T( 1 ) = ( 2 * DeltaRot( 1 ) *
                   exp( -( DeltaRot( 2 ) / mDeltaN ) - ( pow( DeltaRot( 0 ), 2 ) + pow( DeltaRot( 1 ), 2 ) ) / pow( mDeltaT, 2 ) ) *
                   mPhiN * ( q + ( DeltaRot( 2 ) * ( r - q ) ) / ( mDeltaN * ( r - 1 ) ) ) ) /
                 pow( mDeltaT, 2 );
        // Tn
        T( 2 ) =
            ( exp( -( DeltaRot( 2 ) / mDeltaN ) - ( pow( DeltaRot( 0 ), 2 ) + pow( DeltaRot( 1 ), 2 ) ) / pow( mDeltaT, 2 ) ) * mPhiN *
              ( -( mDeltaN * ( -1 + exp( ( pow( DeltaRot( 0 ), 2 ) + pow( DeltaRot( 1 ), 2 ) ) / pow( mDeltaT, 2 ) ) ) * ( -1 + q ) * r ) +
                DeltaRot( 2 ) *
                    ( exp( ( pow( DeltaRot( 0 ), 2 ) + pow( DeltaRot( 1 ), 2 ) ) / pow( mDeltaT, 2 ) ) * ( -1 + q ) - q + r ) ) ) /
            ( pow( mDeltaN, 2 ) * ( -1 + r ) );
    }
    T = DeltaToTN * T;
}

void ExponentialCZMIntegrator::TractionStiffTangent( const Eigen::VectorXd& Delta,
                                                     const mfem::DenseMatrix& Jacobian,
                                                     const int dim,
                                                     Eigen::MatrixXd& H ) const
{
    double q = mPhiT / mPhiN;
    double r = 0.;
    Eigen::MatrixXd DeltaToTN;
    DeltaToTNMat( Jacobian, dim, DeltaToTN );
    Eigen::VectorXd DeltaRot = DeltaToTN.transpose() * Delta;
    if ( dim == 2 )
    {
        H.resize( 2, 2 );
        // Ttt
        H( 0, 0 ) = 2 * ( std::pow( mDeltaT, 2 ) - 2 * std::pow( DeltaRot( 0 ), 2 ) ) *
                    exp( -DeltaRot( 1 ) / mDeltaN - std::pow( DeltaRot( 0 ), 2 ) / std::pow( mDeltaT, 2 ) ) * mPhiN *
                    ( mDeltaN * q * ( r - 1 ) + DeltaRot( 1 ) * ( r - q ) ) / mDeltaN / std::pow( mDeltaT, 4 ) / ( r - 1 );
        // Tnn
        H( 1, 1 ) =
            exp( -DeltaRot( 1 ) / mDeltaN - std::pow( DeltaRot( 0 ), 2 ) / std::pow( mDeltaT, 2 ) ) * mPhiN *
            ( mDeltaN * ( 2 * r - q - q * r + exp( DeltaRot( 0 ) * DeltaRot( 0 ) / mDeltaT / mDeltaT ) * ( q - 1 ) * ( r + 1 ) ) -
              DeltaRot( 1 ) * ( exp( DeltaRot( 0 ) * DeltaRot( 0 ) / mDeltaT / mDeltaT ) * ( q - 1 ) - q + r ) ) /
            std::pow( mDeltaN, 3 ) / ( r - 1 );
        // Tnt
        H( 0, 1 ) = 2 * DeltaRot( 0 ) *
                    exp( -DeltaRot( 1 ) / mDeltaN - std::pow( DeltaRot( 0 ), 2 ) / std::pow( mDeltaT, 2 ) ) * mPhiN *
                    ( DeltaRot( 1 ) * ( q - r ) - mDeltaN * ( q - 1 ) * r ) / std::pow( mDeltaN * mDeltaT, 2 ) / ( r - 1 );
    }
    else if ( dim == 3 )
    {
        H.resize( 3, 3 );
        // Tt1t1
        H( 0, 0 ) =
            ( 2 * ( pow( mDeltaT, 2 ) - 2 * pow( DeltaRot( 0 ), 2 ) ) *
              exp( -( DeltaRot( 2 ) / mDeltaN ) - ( pow( DeltaRot( 0 ), 2 ) + pow( DeltaRot( 1 ), 2 ) ) / pow( mDeltaT, 2 ) ) *
              mPhiN * ( mDeltaN * q * ( -1 + r ) + DeltaRot( 2 ) * ( -q + r ) ) ) /
            ( mDeltaN * pow( mDeltaT, 4 ) * ( -1 + r ) );
        // Tt2t2
        H( 1, 1 ) =
            ( 2 * ( pow( mDeltaT, 2 ) - 2 * pow( DeltaRot( 1 ), 2 ) ) *
              exp( -( DeltaRot( 2 ) / mDeltaN ) - ( pow( DeltaRot( 0 ), 2 ) + pow( DeltaRot( 1 ), 2 ) ) / pow( mDeltaT, 2 ) ) *
              mPhiN * ( mDeltaN * q * ( -1 + r ) + DeltaRot( 2 ) * ( -q + r ) ) ) /
            ( mDeltaN * pow( mDeltaT, 4 ) * ( -1 + r ) );
        // Tnn
        H( 2, 2 ) =
            ( exp( -( DeltaRot( 2 ) / mDeltaN ) - ( pow( DeltaRot( 0 ), 2 ) + pow( DeltaRot( 1 ), 2 ) ) / pow( mDeltaT, 2 ) ) * mPhiN *
              ( -( DeltaRot( 2 ) *
                   ( exp( ( pow( DeltaRot( 0 ), 2 ) + pow( DeltaRot( 1 ), 2 ) ) / pow( mDeltaT, 2 ) ) * ( -1 + q ) - q + r ) ) +
                mDeltaN * ( -q + 2 * r - q * r +
                            exp( ( pow( DeltaRot( 0 ), 2 ) + pow( DeltaRot( 1 ), 2 ) ) / pow( mDeltaT, 2 ) ) *
                                ( -1 + q ) * ( 1 + r ) ) ) ) /
            ( pow( mDeltaN, 3 ) * ( -1 + r ) );
        // Tt1t2
        H( 0, 1 ) =
            ( -4 * DeltaRot( 0 ) * DeltaRot( 1 ) *
              exp( -( DeltaRot( 2 ) / mDeltaN ) - ( pow( DeltaRot( 0 ), 2 ) + pow( DeltaRot( 1 ), 2 ) ) / pow( mDeltaT, 2 ) ) *
              mPhiN * ( q + ( DeltaRot( 2 ) * ( -q + r ) ) / ( mDeltaN * ( -1 + r ) ) ) ) /
            pow( mDeltaT, 4 );
        // Tt1n
        H( 0, 2 ) =
            ( 2 * DeltaRot( 0 ) *
              exp( -( DeltaRot( 2 ) / mDeltaN ) - ( pow( DeltaRot( 0 ), 2 ) + pow( DeltaRot( 1 ), 2 ) ) / pow( mDeltaT, 2 ) ) *
              mPhiN * ( DeltaRot( 2 ) * ( q - r ) - mDeltaN * ( -1 + q ) * r ) ) /
            ( pow( mDeltaN, 2 ) * pow( mDeltaT, 2 ) * ( -1 + r ) );
        // Tt2n
        H( 1, 2 ) =
            ( 2 * DeltaRot( 1 ) *
              exp( -( DeltaRot( 2 ) / mDeltaN ) - ( pow( DeltaRot( 0 ), 2 ) + pow( DeltaRot( 1 ), 2 ) ) / pow( mDeltaT, 2 ) ) *
              mPhiN * ( DeltaRot( 2 ) * ( q - r ) - mDeltaN * ( -1 + q ) * r ) ) /
            ( pow( mDeltaN, 2 ) * pow( mDeltaT, 2 ) * ( -1 + r ) );
    }
    H = DeltaToTN * H * DeltaToTN.transpose();
}

void ADCZMIntegrator::Traction( const Eigen::VectorXd& Delta, const mfem::DenseMatrix& Jacobian, const int dim, Eigen::VectorXd& T ) const
{
    if ( dim == 2 )
    {
        autodiff::VectorXdual2nd delta( Delta );
        autodiff::VectorXdual2nd params = Parameters( Jacobian );
        autodiff::dual2nd u;
        T = autodiff::gradient( potential, autodiff::wrt( delta ), autodiff::at( delta, params ), u );
    }
    else if ( dim == 3 )
    {
        mfem::mfem_error(
            "ADCZMIntegrator::Traction\n"
            "   is not implemented for this class." );
    }
}

void ADCZMIntegrator::TractionStiffTangent( const Eigen::VectorXd& Delta,
                                            const mfem::DenseMatrix& Jacobian,
                                            const int dim,
                                            Eigen::MatrixXd& H ) const
{
    if ( dim == 2 )
    {
        autodiff::VectorXdual2nd delta( Delta );
        autodiff::VectorXdual2nd params = Parameters( Jacobian );
        autodiff::dual2nd u;
        autodiff::VectorXdual g;
        H = autodiff::hessian( potential, autodiff::wrt( delta ), autodiff::at( delta, params ), u, g );
    }
    else if ( dim == 3 )
    {
        mfem::mfem_error(
            "ADCZMIntegrator::Traction\n"
            "   is not implemented for this class." );
    }
}

autodiff::VectorXdual2nd ExponentialADCZMIntegrator::Parameters( const mfem::DenseMatrix& Jacobian ) const
{
    autodiff::VectorXdual2nd params( 8 );
    params << mDeltaT, mDeltaN, mPhiN, mPhiT, Jacobian( 0, 0 ), Jacobian( 1, 0 ), Jacobian( 0, 0 ), Jacobian( 1, 0 );
    return params;
}

OrtizIrreversibleADCZMIntegrator::OrtizIrreversibleADCZMIntegrator( Memorize& memo ) : ADCZMIntegrator( memo )
{
    // x: diffX, diffY
    // p: deltaT, deltaN, phiT, phiN, dA1x, dA1y, dA2x, dA2y
    potential = [this]( const autodiff::VectorXdual2nd& x, const autodiff::VectorXdual2nd& p )
    {
        Eigen::Map<const autodiff::VectorXdual2nd> dA1( p.data() + 4, 2 );
        Eigen::Map<const autodiff::VectorXdual2nd> dA2( p.data() + 6, 2 );

        autodiff::VectorXdual2nd directionT = dA1;
        directionT.normalize();

        static Eigen::Rotation2Dd rot( EIGEN_PI / 2 );
        autodiff::VectorXdual2nd directionN = rot.toRotationMatrix() * directionT;
        const autodiff::dual2nd DeltaT = directionT.dot( x );
        const autodiff::dual2nd DeltaN = directionN.dot( x );

        const autodiff::dual2nd delta = autodiff::detail::sqrt( mBeta * mBeta * DeltaT * DeltaT + DeltaN * DeltaN );

        autodiff::dual2nd res = autodiff::detail::exp( 1. ) * mSgimaC * mDeltaC *
                                ( 1 - ( 1 + delta / mDeltaC ) * autodiff::detail::exp( -delta / mDeltaC ) );
        return res;
    };
}

void OrtizIrreversibleADCZMIntegrator::UpdateMemo( const int ind )
{
    auto& czm_gps = mMemo.GetCZMPointStorage( ind );
    if ( czm_gps.Lambda < czm_gps.CurLambda )
    {
        czm_gps.Lambda = czm_gps.CurLambda;
    }
}

ExponentialADCZMIntegrator::ExponentialADCZMIntegrator(
    Memorize& memo, const double sigmaMax, const double tauMax, const double deltaN, const double deltaT )
    : ADCZMIntegrator( memo ),
      mSigmaMax{ sigmaMax },
      mTauMax{ tauMax },
      mDeltaN{ deltaN },
      mDeltaT{ deltaT },
      mPhiN{ std::exp( 1. ) * sigmaMax * deltaN },
      mPhiT{ std::sqrt( std::exp( 1. ) / 2 ) * tauMax * deltaT }
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

ExponentialRotADCZMIntegrator::ExponentialRotADCZMIntegrator(
    Memorize& memo, const double sigmaMax, const double tauMax, const double deltaN, const double deltaT )
    : ExponentialADCZMIntegrator( memo, sigmaMax, tauMax, deltaN, deltaT )
{
    // x: u1x, u1y, u2x, u2y, du1x, du1y, du2x, du2y
    // p: deltaT, deltaN, phiT, phiN, dA1x, dA1y, dA2x, dA2y
    potential = [this]( const autodiff::VectorXdual2nd& x, const autodiff::VectorXdual2nd& p )
    {
        Eigen::Map<const autodiff::VectorXdual2nd> U1( x.data(), 2 );
        Eigen::Map<const autodiff::VectorXdual2nd> U2( x.data() + 2, 2 );
        Eigen::Map<const autodiff::VectorXdual2nd> dU1( x.data() + 4, 2 );
        Eigen::Map<const autodiff::VectorXdual2nd> dU2( x.data() + 6, 2 );

        Eigen::Map<const autodiff::VectorXdual2nd> dA1( p.data() + 4, 2 );
        Eigen::Map<const autodiff::VectorXdual2nd> dA2( p.data() + 6, 2 );
        const double q = mPhiT / mPhiN;

        const double r = 0.;

        autodiff::VectorXdual2nd diff = U1 - U2;
        autodiff::VectorXdual2nd directionT = dA1 + dA2 + dU1 + dU2;
        directionT.normalize();

        static Eigen::Rotation2Dd rot( EIGEN_PI / 2 );
        autodiff::VectorXdual2nd directionN = rot.toRotationMatrix() * directionT;
        const autodiff::dual2nd DeltaT = directionT.dot( diff );
        const autodiff::dual2nd DeltaN = directionN.dot( diff );

        autodiff::dual2nd res = mPhiN + mPhiN * autodiff::detail::exp( -DeltaN / mDeltaN ) *
                                            ( ( 1. - r + DeltaN / mDeltaN ) * ( 1. - q ) / ( r - 1. ) -
                                              ( q + ( r - q ) / ( r - 1. ) * DeltaN / mDeltaN ) *
                                                  autodiff::detail::exp( -DeltaT * DeltaT / mDeltaT / mDeltaT ) );
        if ( DeltaN < 0 )
            res += 1e17 * DeltaN * DeltaN;
        return res;
    };
}

void ExponentialRotADCZMIntegrator::matrixB( const int dof1,
                                             const int dof2,
                                             const mfem::Vector& shape1,
                                             const mfem::Vector& shape2,
                                             const mfem::DenseMatrix& gshape1,
                                             const mfem::DenseMatrix& gshape2,
                                             const int dim )
{
    if ( dim == 2 )
    {
        mB.resize( 8, 2 * ( dof1 + dof2 ) );
        mB.setZero();

        for ( int i = 0; i < dof1; i++ )
        {
            for ( int j = 0; j < dim; j++ )
            {
                mB( j, i + j * dof1 ) = shape1( i );
            }
        }

        for ( int i = 0; i < dof2; i++ )
        {
            for ( int j = 0; j < dim; j++ )
            {
                mB( 2 + j, i + j * dof2 + dim * dof1 ) = shape2( i );
            }
        }

        for ( int i = 0; i < dof1; i++ )
        {
            for ( int j = 0; j < dim; j++ )
            {
                mB( 4 + j, i + j * dof1 ) = gshape1( i, 0 );
            }
        }

        for ( int i = 0; i < dof2; i++ )
        {
            for ( int j = 0; j < dim; j++ )
            {
                mB( 6 + j, i + j * dof2 + dim * dof1 ) = gshape2( i, 0 );
            }
        }
    }
    else if ( dim == 3 )
    {
        std::cout << "not implemented!\n";
    }
}
} // namespace plugin