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
    if ( mIterAux == nullptr )
    {
        mfem::mfem_error( "IterAux is not provided yet.\n" );
    }
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
        // Set the integration point in the face and the neighboring element
        const mfem::IntegrationPoint& ip = ir->IntPoint( i );
        Tr.SetAllIntPoints( &ip );
        EvalCZMLaw( Tr, ip );

        const mfem::Vector& shape1 = mMemo.GetFace1Shape( i );
        const mfem::Vector& shape2 = mMemo.GetFace2Shape( i );

        const mfem::DenseMatrix& gshape1 = mMemo.GetFace1GShape( i );
        const mfem::DenseMatrix& gshape2 = mMemo.GetFace2GShape( i );
        matrixB( dof1, dof2, shape1, shape2, gshape1, gshape2, vdim );
        Eigen::VectorXd Delta = mB * u;
        Eigen::VectorXd T;
        Traction( Delta, i, vdim, T );
        eigenVec += mB.transpose() * T * mMemo.GetFaceWeight( i );
    }
}

void CZMIntegrator::AssembleFaceGrad( const mfem::FiniteElement& el1,
                                      const mfem::FiniteElement& el2,
                                      mfem::FaceElementTransformations& Tr,
                                      const mfem::Vector& elfun,
                                      mfem::DenseMatrix& elmat )
{
    if ( mIterAux == nullptr )
    {
        mfem::mfem_error( "IterAux is not provided yet.\n" );
    }
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
        // Set the integration point in the face and the neighboring element
        const mfem::IntegrationPoint& ip = ir->IntPoint( i );
        Tr.SetAllIntPoints( &ip );
        EvalCZMLaw( Tr, ip );

        const mfem::Vector& shape1 = mMemo.GetFace1Shape( i );
        const mfem::Vector& shape2 = mMemo.GetFace2Shape( i );

        const mfem::DenseMatrix& gshape1 = mMemo.GetFace1GShape( i );
        const mfem::DenseMatrix& gshape2 = mMemo.GetFace2GShape( i );
        matrixB( dof1, dof2, shape1, shape2, gshape1, gshape2, vdim );
        Eigen::VectorXd Delta = mB * u;

        Eigen::MatrixXd H;
        TractionStiffTangent( Delta, i, vdim, H );
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

void CZMIntegrator::Update( const int gauss, const double delta_n, const double delta_t1, const double delta_t2 ) const
{
    auto& pd = mMemo.GetFacePointData( gauss );
    // historical strain energy+ for KKT condition
    if ( !pd.get_val<PointData>( "delta" ).has_value() )
        pd.set_val<PointData>( "delta", std::move( PointData( delta_n, delta_t1, delta_t2 ) ) );
    else
    {
        auto& delta_data = pd.get_val<PointData>( "delta" ).value().get();
        delta_data.delta_n_prev = delta_n;
        delta_data.delta_t1_prev = delta_t1;
        delta_data.delta_t2_prev = delta_t2;
    }
}

void LinearCZMIntegrator::Traction( const Eigen::VectorXd& Delta, const int i, const int dim, Eigen::VectorXd& T ) const
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

void LinearCZMIntegrator::TractionStiffTangent( const Eigen::VectorXd& Delta, const int i, const int dim, Eigen::MatrixXd& H ) const
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

void ExponentialCZMIntegrator::EvalCZMLaw( mfem::ElementTransformation& Tr, const mfem::IntegrationPoint& ip )
{
    mCZMLawConst.sigma_max = mSigmaMax->Eval( Tr, ip );
    mCZMLawConst.tau_max = mTauMax->Eval( Tr, ip );
    mCZMLawConst.delta_n = mDeltaN->Eval( Tr, ip );
    mCZMLawConst.delta_t = mDeltaT->Eval( Tr, ip );
    mCZMLawConst.update_phi();
}

void ExponentialCZMIntegrator::Traction( const Eigen::VectorXd& Delta, const int i, const int dim, Eigen::VectorXd& T ) const
{
    double q = mCZMLawConst.phi_t / mCZMLawConst.phi_n;
    double r = 0.;
    Eigen::MatrixXd DeltaToTN;
    DeltaToTNMat( mMemo.GetFaceJacobian( i ), DeltaToTN );
    Eigen::VectorXd DeltaRot = DeltaToTN.transpose() * Delta;

    const auto& pd = this->mMemo.GetFacePointData( i );

    if ( mIterAux->IterNumber() == 0 )
    {
        Update( i, DeltaRot( 0 ), DeltaRot( 1 ), dim == 3 ? DeltaRot( 2 ) : 0. );
    }
    const auto& delta_data = pd.get_val<PointData>( "delta" ).value().get();
    if ( dim == 2 )
    {
        T.resize( 2 );
        // Tt
        T( 0 ) = 2 * DeltaRot( 0 ) *
                     exp( -DeltaRot( 1 ) / mCZMLawConst.delta_n -
                          DeltaRot( 0 ) * DeltaRot( 0 ) / mCZMLawConst.delta_t / mCZMLawConst.delta_t ) *
                     mCZMLawConst.phi_n * ( q + DeltaRot( 1 ) * ( r - q ) / mCZMLawConst.delta_n / ( r - 1 ) ) /
                     mCZMLawConst.delta_t / mCZMLawConst.delta_t +
                 2 * xi_t * ( DeltaRot( 0 ) - delta_data.delta_t1_prev ) / mCZMLawConst.delta_t / mIterAux->GetDeltaLambda();
        // Tn
        T( 1 ) = mCZMLawConst.phi_n / mCZMLawConst.delta_n * exp( -DeltaRot( 1 ) / mCZMLawConst.delta_n ) *
                     ( DeltaRot( 1 ) / mCZMLawConst.delta_n *
                           exp( -DeltaRot( 0 ) * DeltaRot( 0 ) / mCZMLawConst.delta_t / mCZMLawConst.delta_t ) +
                       ( 1 - q ) / ( r - 1 ) *
                           ( 1 - exp( -DeltaRot( 0 ) * DeltaRot( 0 ) / mCZMLawConst.delta_t / mCZMLawConst.delta_t ) ) *
                           ( r - DeltaRot( 1 ) / mCZMLawConst.delta_n ) ) +
                 2 * xi_n * ( DeltaRot( 1 ) - delta_data.delta_n_prev ) / mCZMLawConst.delta_n / mIterAux->GetDeltaLambda();
    }
    else if ( dim == 3 )
    {
        T.resize( 3 );
        // Tt1
        T( 0 ) = ( 2 * DeltaRot( 0 ) *
                   exp( -( DeltaRot( 2 ) / mCZMLawConst.delta_n ) -
                        ( pow( DeltaRot( 0 ), 2 ) + pow( DeltaRot( 1 ), 2 ) ) / pow( mCZMLawConst.delta_t, 2 ) ) *
                   mCZMLawConst.phi_n * ( q + ( DeltaRot( 2 ) * ( r - q ) ) / ( mCZMLawConst.delta_n * ( r - 1 ) ) ) ) /
                 pow( mCZMLawConst.delta_t, 2 );
        // Tt2
        T( 1 ) = ( 2 * DeltaRot( 1 ) *
                   exp( -( DeltaRot( 2 ) / mCZMLawConst.delta_n ) -
                        ( pow( DeltaRot( 0 ), 2 ) + pow( DeltaRot( 1 ), 2 ) ) / pow( mCZMLawConst.delta_t, 2 ) ) *
                   mCZMLawConst.phi_n * ( q + ( DeltaRot( 2 ) * ( r - q ) ) / ( mCZMLawConst.delta_n * ( r - 1 ) ) ) ) /
                 pow( mCZMLawConst.delta_t, 2 );
        // Tn
        T( 2 ) =
            ( exp( -( DeltaRot( 2 ) / mCZMLawConst.delta_n ) -
                   ( pow( DeltaRot( 0 ), 2 ) + pow( DeltaRot( 1 ), 2 ) ) / pow( mCZMLawConst.delta_t, 2 ) ) *
              mCZMLawConst.phi_n *
              ( -( mCZMLawConst.delta_n *
                   ( -1 + exp( ( pow( DeltaRot( 0 ), 2 ) + pow( DeltaRot( 1 ), 2 ) ) / pow( mCZMLawConst.delta_t, 2 ) ) ) *
                   ( -1 + q ) * r ) +
                DeltaRot( 2 ) *
                    ( exp( ( pow( DeltaRot( 0 ), 2 ) + pow( DeltaRot( 1 ), 2 ) ) / pow( mCZMLawConst.delta_t, 2 ) ) * ( -1 + q ) -
                      q + r ) ) ) /
            ( pow( mCZMLawConst.delta_n, 2 ) * ( -1 + r ) );
    }
    T = DeltaToTN * T;
}

void ExponentialCZMIntegrator::TractionStiffTangent( const Eigen::VectorXd& Delta, const int i, const int dim, Eigen::MatrixXd& H ) const
{
    double q = mCZMLawConst.phi_t / mCZMLawConst.phi_n;
    double r = 0.;
    Eigen::MatrixXd DeltaToTN;
    DeltaToTNMat( mMemo.GetFaceJacobian( i ), DeltaToTN );
    Eigen::VectorXd DeltaRot = DeltaToTN.transpose() * Delta;

    if ( mIterAux->IterNumber() == 0 )
    {
        Update( i, DeltaRot( 0 ), DeltaRot( 1 ), dim == 3 ? DeltaRot( 2 ) : 0. );
    }

    if ( dim == 2 )
    {
        H.resize( 2, 2 );
        // Ttt
        H( 0, 0 ) = 2 * ( std::pow( mCZMLawConst.delta_t, 2 ) - 2 * std::pow( DeltaRot( 0 ), 2 ) ) *
                        exp( -DeltaRot( 1 ) / mCZMLawConst.delta_n -
                             std::pow( DeltaRot( 0 ), 2 ) / std::pow( mCZMLawConst.delta_t, 2 ) ) *
                        mCZMLawConst.phi_n * ( mCZMLawConst.delta_n * q * ( r - 1 ) + DeltaRot( 1 ) * ( r - q ) ) /
                        mCZMLawConst.delta_n / std::pow( mCZMLawConst.delta_t, 4 ) / ( r - 1 ) +
                    2 * xi_t / mCZMLawConst.delta_t / mIterAux->GetDeltaLambda();
        // Tnn
        H( 1, 1 ) =
            exp( -DeltaRot( 1 ) / mCZMLawConst.delta_n - std::pow( DeltaRot( 0 ), 2 ) / std::pow( mCZMLawConst.delta_t, 2 ) ) *
                mCZMLawConst.phi_n *
                ( mCZMLawConst.delta_n * ( 2 * r - q - q * r +
                                           exp( DeltaRot( 0 ) * DeltaRot( 0 ) / mCZMLawConst.delta_t / mCZMLawConst.delta_t ) *
                                               ( q - 1 ) * ( r + 1 ) ) -
                  DeltaRot( 1 ) *
                      ( exp( DeltaRot( 0 ) * DeltaRot( 0 ) / mCZMLawConst.delta_t / mCZMLawConst.delta_t ) * ( q - 1 ) - q + r ) ) /
                std::pow( mCZMLawConst.delta_n, 3 ) / ( r - 1 ) +
            2 * xi_n / mCZMLawConst.delta_n / mIterAux->GetDeltaLambda();
        // Tnt
        H( 0, 1 ) =
            2 * DeltaRot( 0 ) *
            exp( -DeltaRot( 1 ) / mCZMLawConst.delta_n - std::pow( DeltaRot( 0 ), 2 ) / std::pow( mCZMLawConst.delta_t, 2 ) ) *
            mCZMLawConst.phi_n * ( DeltaRot( 1 ) * ( q - r ) - mCZMLawConst.delta_n * ( q - 1 ) * r ) /
            std::pow( mCZMLawConst.delta_n * mCZMLawConst.delta_t, 2 ) / ( r - 1 );
    }
    else if ( dim == 3 )
    {
        H.resize( 3, 3 );
        // Tt1t1
        H( 0, 0 ) = ( 2 * ( pow( mCZMLawConst.delta_t, 2 ) - 2 * pow( DeltaRot( 0 ), 2 ) ) *
                      exp( -( DeltaRot( 2 ) / mCZMLawConst.delta_n ) -
                           ( pow( DeltaRot( 0 ), 2 ) + pow( DeltaRot( 1 ), 2 ) ) / pow( mCZMLawConst.delta_t, 2 ) ) *
                      mCZMLawConst.phi_n * ( mCZMLawConst.delta_n * q * ( -1 + r ) + DeltaRot( 2 ) * ( -q + r ) ) ) /
                    ( mCZMLawConst.delta_n * pow( mCZMLawConst.delta_t, 4 ) * ( -1 + r ) );
        // Tt2t2
        H( 1, 1 ) = ( 2 * ( pow( mCZMLawConst.delta_t, 2 ) - 2 * pow( DeltaRot( 1 ), 2 ) ) *
                      exp( -( DeltaRot( 2 ) / mCZMLawConst.delta_n ) -
                           ( pow( DeltaRot( 0 ), 2 ) + pow( DeltaRot( 1 ), 2 ) ) / pow( mCZMLawConst.delta_t, 2 ) ) *
                      mCZMLawConst.phi_n * ( mCZMLawConst.delta_n * q * ( -1 + r ) + DeltaRot( 2 ) * ( -q + r ) ) ) /
                    ( mCZMLawConst.delta_n * pow( mCZMLawConst.delta_t, 4 ) * ( -1 + r ) );
        // Tnn
        H( 2, 2 ) =
            ( exp( -( DeltaRot( 2 ) / mCZMLawConst.delta_n ) -
                   ( pow( DeltaRot( 0 ), 2 ) + pow( DeltaRot( 1 ), 2 ) ) / pow( mCZMLawConst.delta_t, 2 ) ) *
              mCZMLawConst.phi_n *
              ( -( DeltaRot( 2 ) *
                   ( exp( ( pow( DeltaRot( 0 ), 2 ) + pow( DeltaRot( 1 ), 2 ) ) / pow( mCZMLawConst.delta_t, 2 ) ) * ( -1 + q ) - q + r ) ) +
                mCZMLawConst.delta_n * ( -q + 2 * r - q * r +
                                         exp( ( pow( DeltaRot( 0 ), 2 ) + pow( DeltaRot( 1 ), 2 ) ) / pow( mCZMLawConst.delta_t, 2 ) ) *
                                             ( -1 + q ) * ( 1 + r ) ) ) ) /
            ( pow( mCZMLawConst.delta_n, 3 ) * ( -1 + r ) );
        // Tt1t2
        H( 0, 1 ) = ( -4 * DeltaRot( 0 ) * DeltaRot( 1 ) *
                      exp( -( DeltaRot( 2 ) / mCZMLawConst.delta_n ) -
                           ( pow( DeltaRot( 0 ), 2 ) + pow( DeltaRot( 1 ), 2 ) ) / pow( mCZMLawConst.delta_t, 2 ) ) *
                      mCZMLawConst.phi_n * ( q + ( DeltaRot( 2 ) * ( -q + r ) ) / ( mCZMLawConst.delta_n * ( -1 + r ) ) ) ) /
                    pow( mCZMLawConst.delta_t, 4 );
        // Tt1n
        H( 0, 2 ) = ( 2 * DeltaRot( 0 ) *
                      exp( -( DeltaRot( 2 ) / mCZMLawConst.delta_n ) -
                           ( pow( DeltaRot( 0 ), 2 ) + pow( DeltaRot( 1 ), 2 ) ) / pow( mCZMLawConst.delta_t, 2 ) ) *
                      mCZMLawConst.phi_n * ( DeltaRot( 2 ) * ( q - r ) - mCZMLawConst.delta_n * ( -1 + q ) * r ) ) /
                    ( pow( mCZMLawConst.delta_n, 2 ) * pow( mCZMLawConst.delta_t, 2 ) * ( -1 + r ) );
        // Tt2n
        H( 1, 2 ) = ( 2 * DeltaRot( 1 ) *
                      exp( -( DeltaRot( 2 ) / mCZMLawConst.delta_n ) -
                           ( pow( DeltaRot( 0 ), 2 ) + pow( DeltaRot( 1 ), 2 ) ) / pow( mCZMLawConst.delta_t, 2 ) ) *
                      mCZMLawConst.phi_n * ( DeltaRot( 2 ) * ( q - r ) - mCZMLawConst.delta_n * ( -1 + q ) * r ) ) /
                    ( pow( mCZMLawConst.delta_n, 2 ) * pow( mCZMLawConst.delta_t, 2 ) * ( -1 + r ) );
    }
    H = DeltaToTN * H * DeltaToTN.transpose();
}

void ADCZMIntegrator::Traction( const Eigen::VectorXd& Delta, const int i, const int dim, Eigen::VectorXd& T ) const
{
    autodiff::VectorXdual2nd delta( Delta );
    autodiff::dual2nd u;
    T = autodiff::gradient( potential, autodiff::wrt( delta ), autodiff::at( delta, i ), u );
}

void ADCZMIntegrator::TractionStiffTangent( const Eigen::VectorXd& Delta, const int i, const int dim, Eigen::MatrixXd& H ) const
{
    autodiff::VectorXdual2nd delta( Delta );
    autodiff::dual2nd u;
    autodiff::VectorXdual g;
    H = autodiff::hessian( potential, autodiff::wrt( delta ), autodiff::at( delta, i ), u, g );
}

void ExponentialADCZMIntegrator::EvalCZMLaw( mfem::ElementTransformation& Tr, const mfem::IntegrationPoint& ip )
{
    mCZMLawConst.sigma_max = mSigmaMax->Eval( Tr, ip );
    mCZMLawConst.tau_max = mTauMax->Eval( Tr, ip );
    mCZMLawConst.delta_n = mDeltaN->Eval( Tr, ip );
    mCZMLawConst.delta_t = mDeltaT->Eval( Tr, ip );
    mCZMLawConst.update_phi();
}

ExponentialADCZMIntegrator::ExponentialADCZMIntegrator(
    Memorize& memo, mfem::Coefficient& sigmaMax, mfem::Coefficient& tauMax, mfem::Coefficient& deltaN, mfem::Coefficient& deltaT )
    : ADCZMIntegrator( memo ), mSigmaMax{&sigmaMax}, mTauMax{&tauMax}, mDeltaN{&deltaN}, mDeltaT{&deltaT}
{
    // x: diffX, diffY
    potential = [this]( const autodiff::VectorXdual2nd& x, const int i ) {
        const auto& Jacobian = this->mMemo.GetFaceJacobian( i );
        const int dim = Jacobian.Height();
        const auto& pd = this->mMemo.GetFacePointData( i );

        const double q = mCZMLawConst.phi_t / mCZMLawConst.phi_n;
        const double r = 0.;
        Eigen::MatrixXd DeltaToTN;
        DeltaToTNMat( Jacobian, DeltaToTN );

        if ( dim == 2 )
        {
            const autodiff::dual2nd DeltaT = DeltaToTN.col( 0 ).dot( x );
            const autodiff::dual2nd DeltaN = DeltaToTN.col( 1 ).dot( x );

            if ( mIterAux->IterNumber() == 0 )
            {
                Update( i, autodiff::detail::val( DeltaN ), autodiff::detail::val( DeltaT ) );
            }

            const auto& delta_data = pd.get_val<PointData>( "delta" ).value().get();

            autodiff::dual2nd res =
                mCZMLawConst.phi_n +
                mCZMLawConst.phi_n * autodiff::detail::exp( -DeltaN / mCZMLawConst.delta_n ) *
                    ( ( 1. - r + DeltaN / mCZMLawConst.delta_n ) * ( 1. - q ) / ( r - 1. ) -
                      ( q + ( r - q ) / ( r - 1. ) * DeltaN / mCZMLawConst.delta_n ) *
                          autodiff::detail::exp( -DeltaT * DeltaT / mCZMLawConst.delta_t / mCZMLawConst.delta_t ) ) +
                xi_n * ( DeltaN - delta_data.delta_n_prev ) * ( DeltaN - delta_data.delta_n_prev ) /
                    mCZMLawConst.delta_n / mIterAux->GetDeltaLambda() +
                xi_t * ( DeltaT - delta_data.delta_t1_prev ) * ( DeltaT - delta_data.delta_t1_prev ) /
                    mCZMLawConst.delta_t / mIterAux->GetDeltaLambda();
            return res;
        }
        else
        {
            const autodiff::dual2nd DeltaT1 = DeltaToTN.col( 0 ).dot( x );
            const autodiff::dual2nd DeltaT2 = DeltaToTN.col( 1 ).dot( x );
            const autodiff::dual2nd DeltaN = DeltaToTN.col( 2 ).dot( x );

            if ( mIterAux->IterNumber() == 0 )
            {
                Update( i, autodiff::detail::val( DeltaN ), autodiff::detail::val( DeltaT1 ), autodiff::detail::val( DeltaT2 ) );
            }

            const auto& delta_data = pd.get_val<PointData>( "delta" ).value().get();

            autodiff::dual2nd res = mCZMLawConst.phi_n +
                                    mCZMLawConst.phi_n * autodiff::detail::exp( -DeltaN / mCZMLawConst.delta_n ) *
                                        ( ( autodiff::dual2nd( 1. ) - r + DeltaN / mCZMLawConst.delta_n ) *
                                              ( autodiff::dual2nd( 1. ) - q ) / ( r - autodiff::dual2nd( 1. ) ) -
                                          ( q + ( r - q ) / ( r - autodiff::dual2nd( 1. ) ) * DeltaN / mCZMLawConst.delta_n ) *
                                              autodiff::detail::exp( -( DeltaT1 * DeltaT1 + DeltaT2 * DeltaT2 ) /
                                                                     mCZMLawConst.delta_t / mCZMLawConst.delta_t ) ) +
                                    xi_n * ( DeltaN - delta_data.delta_n_prev ) * ( DeltaN - delta_data.delta_n_prev ) /
                                        mCZMLawConst.delta_n / mIterAux->GetDeltaLambda() +
                                    xi_t * ( DeltaT1 - delta_data.delta_t1_prev ) * ( DeltaT1 - delta_data.delta_t1_prev ) /
                                        mCZMLawConst.delta_t / mIterAux->GetDeltaLambda() +
                                    xi_t * ( DeltaT2 - delta_data.delta_t2_prev ) * ( DeltaT2 - delta_data.delta_t2_prev ) /
                                        mCZMLawConst.delta_t / mIterAux->GetDeltaLambda();
            return res;
        }
    };
}

ExponentialRotADCZMIntegrator::ExponentialRotADCZMIntegrator(
    Memorize& memo, mfem::Coefficient& sigmaMax, mfem::Coefficient& tauMax, mfem::Coefficient& deltaN, mfem::Coefficient& deltaT )
    : ExponentialADCZMIntegrator( memo, sigmaMax, tauMax, deltaN, deltaT )
{
    // x: u1x, u1y, u2x, u2y, du1x, du1y, du2x, du2y
    potential = [this]( const autodiff::VectorXdual2nd& x, const int i ) {
        Eigen::Map<const autodiff::VectorXdual2nd> U1( x.data(), 2 );
        Eigen::Map<const autodiff::VectorXdual2nd> U2( x.data() + 2, 2 );
        Eigen::Map<const autodiff::VectorXdual2nd> dU1( x.data() + 4, 2 );
        Eigen::Map<const autodiff::VectorXdual2nd> dU2( x.data() + 6, 2 );
        const auto& Jacobian = this->mMemo.GetFaceJacobian( i );
        const auto& pd = this->mMemo.GetFacePointData( i );

        autodiff::VectorXdual2nd dA1( 2 );
        dA1 << Jacobian( 0, 0 ), Jacobian( 1, 0 );
        const double q = mCZMLawConst.phi_t / mCZMLawConst.phi_n;
        const double r = 0.;
        autodiff::VectorXdual2nd diff = U1 - U2;
        autodiff::VectorXdual2nd directionT = dA1 + dA1 + dU1 + dU2;
        directionT.normalize();

        static Eigen::Rotation2Dd rot( EIGEN_PI / 2 );
        autodiff::VectorXdual2nd directionN = rot.toRotationMatrix() * directionT;
        const autodiff::dual2nd DeltaT = directionT.dot( diff );
        const autodiff::dual2nd DeltaN = directionN.dot( diff );
        if ( mIterAux->IterNumber() == 0 )
        {
            Update( i, autodiff::detail::val( DeltaN ), autodiff::detail::val( DeltaT ) );
        }
        const auto& delta_data = pd.get_val<PointData>( "delta" ).value().get();

        autodiff::dual2nd res =
            mCZMLawConst.phi_n +
            mCZMLawConst.phi_n * autodiff::detail::exp( -DeltaN / mCZMLawConst.delta_n ) *
                ( ( 1. - r + DeltaN / mCZMLawConst.delta_n ) * ( 1. - q ) / ( r - 1. ) -
                  ( q + ( r - q ) / ( r - 1. ) * DeltaN / mCZMLawConst.delta_n ) *
                      autodiff::detail::exp( -DeltaT * DeltaT / mCZMLawConst.delta_t / mCZMLawConst.delta_t ) ) +
            xi_n * ( DeltaN - delta_data.delta_n_prev ) * ( DeltaN - delta_data.delta_n_prev ) / mCZMLawConst.delta_n /
                mIterAux->GetDeltaLambda() +
            xi_t * ( DeltaT - delta_data.delta_t1_prev ) * ( DeltaT - delta_data.delta_t1_prev ) /
                mCZMLawConst.delta_t / mIterAux->GetDeltaLambda();
        // autodiff::dual2nd res = mCZMLawConst.phi_n +
        //                         mCZMLawConst.phi_n * autodiff::detail::exp( -DeltaN / mCZMLawConst.delta_n ) *
        //                             ( ( 1. - r + DeltaN / mCZMLawConst.delta_n ) * ( 1. - q ) / ( r - 1. ) -
        //                               ( q + ( r - q ) / ( r - 1. ) * DeltaN / mCZMLawConst.delta_n ) *
        //                                   autodiff::detail::exp( -DeltaT * DeltaT / mCZMLawConst.delta_t / mCZMLawConst.delta_t ) );

        if ( DeltaN < 0 )
            res += 1e20 * DeltaN * DeltaN;
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

void DeltaToTNMat( const mfem::DenseMatrix& Jacobian, Eigen::MatrixXd& DeltaToTN )
{
    int dim = Jacobian.Height();
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
} // namespace plugin