#include "FEMPlugin.h"
#include "util.h"
#include <Eigen/Dense>
#include <iostream>
#include <unsupported/Eigen/KroneckerProduct>

namespace plugin
{
Eigen::MatrixXd mapper( const int dim, const int dof )
{
    Eigen::MatrixXd res( dim * dof, dim * dof );
    res.setZero();
    for ( int i = 0; i < dim; i++ )
    {
        for ( int j = 0; j < dof; j++ )
        {
            res( dim * j + i, i * dof + j ) = 1;
        }
    }
    return res;
}

void smallDeformMatrixB( const int dof, const int dim, const Eigen::MatrixXd& gshape, Eigen::Matrix<double, 6, Eigen::Dynamic>& B )
{
    if ( dim == 2 )
    {
        for ( int i = 0; i < dof; i++ )
        {
            B( 0, i + 0 * dof ) = gshape( i, 0 );
            B( 1, i + 1 * dof ) = gshape( i, 1 );

            B( 3, i + 0 * dof ) = gshape( i, 1 );
            B( 3, i + 1 * dof ) = gshape( i, 0 );
        }
    }
    else if ( dim == 3 )
    {
        for ( int i = 0; i < dof; i++ )
        {
            B( 0, i + 0 * dof ) = gshape( i, 0 );
            B( 1, i + 1 * dof ) = gshape( i, 1 );
            B( 2, i + 2 * dof ) = gshape( i, 2 );

            B( 3, i + 0 * dof ) = gshape( i, 1 );
            B( 3, i + 1 * dof ) = gshape( i, 0 );

            B( 4, i + 1 * dof ) = gshape( i, 2 );
            B( 4, i + 2 * dof ) = gshape( i, 1 );

            B( 5, i + 0 * dof ) = gshape( i, 2 );
            B( 5, i + 2 * dof ) = gshape( i, 0 );
        }
    }
    else
    {
        MFEM_WARNING( "It is not for 1D analysis." );
    }
}

Memorize::Memorize( mfem::Mesh* m ) : mEleStorage( m->GetNE() ), mFaceStorage( m->GetNumFaces() )
{
}

void Memorize::InitializeElement( const mfem::FiniteElement& el, mfem::ElementTransformation& Trans, const mfem::IntegrationRule& ir )
{
    mElementNo = Trans.ElementNo;
    if ( mEleStorage[mElementNo] != nullptr )
        return;

    const int dim = el.GetDim();
    const int numOfNodes = el.GetDof();
    const int numOfGauss = ir.GetNPoints();
    mDShape1.SetSize( numOfNodes, dim );
    mEleStorage[mElementNo] = std::make_unique<std::vector<GaussPointStorage>>( numOfGauss );
    for ( int i = 0; i < numOfGauss; i++ )
    {
        ( *mEleStorage[mElementNo] )[i].GShape.resize( numOfNodes, dim );
        const mfem::IntegrationPoint& ip = ir.IntPoint( i );
        Trans.SetIntPoint( &ip );
        el.CalcDShape( ip, mDShape1 );
        mGShape1.UseExternalData( ( *mEleStorage[mElementNo] )[i].GShape.data(), numOfNodes, dim );
        Mult( mDShape1, Trans.InverseJacobian(), mGShape1 );

        ( *mEleStorage[mElementNo] )[i].DetdXdXi = Trans.Weight();
    }
}

void Memorize::InitializeFace( const mfem::FiniteElement& el1,
                               const mfem::FiniteElement& el2,
                               mfem::FaceElementTransformations& Trans,
                               const mfem::IntegrationRule& ir )
{
    mElementNo = Trans.ElementNo;
    if ( mFaceStorage[mElementNo] != nullptr )
        return;

    const int dim = el1.GetDim();
    const int numOfNodes1 = el1.GetDof();
    mDShape1.SetSize( numOfNodes1, dim );
    mGShape1.SetSize( numOfNodes1, dim );
    const int numOfNodes2 = el2.GetDof();
    mDShape2.SetSize( numOfNodes2, dim );
    mGShape2.SetSize( numOfNodes2, dim );
    const int numOfGauss = ir.GetNPoints();
    mFaceStorage[mElementNo] = std::make_unique<std::vector<CZMGaussPointStorage>>( numOfGauss );
    for ( int i = 0; i < numOfGauss; i++ )
    {
        ( *mFaceStorage[mElementNo] )[i].Shape1.SetSize( el1.GetDof() );
        ( *mFaceStorage[mElementNo] )[i].Shape2.SetSize( el2.GetDof() );
        const mfem::IntegrationPoint& ip = ir.IntPoint( i );
        Trans.SetAllIntPoints( &ip );
        const mfem::IntegrationPoint& eip1 = Trans.GetElement1IntPoint();
        const mfem::IntegrationPoint& eip2 = Trans.GetElement2IntPoint();
        el1.CalcShape( eip1, ( *mFaceStorage[mElementNo] )[i].Shape1 );
        el2.CalcShape( eip2, ( *mFaceStorage[mElementNo] )[i].Shape2 );

        ( *mFaceStorage[mElementNo] )[i].Weight = ip.weight * Trans.Weight();
        ( *mFaceStorage[mElementNo] )[i].Jacobian = Trans.Jacobian();

        el1.CalcDShape( eip1, mDShape1 );
        el2.CalcDShape( eip2, mDShape2 );
        auto& Trans1 = Trans.GetElement1Transformation();
        auto& Trans2 = Trans.GetElement2Transformation();
        Trans1.SetIntPoint( &eip1 );
        Trans2.SetIntPoint( &eip2 );

        Mult( mDShape1, Trans1.InverseJacobian(), mGShape1 );
        Mult( mDShape2, Trans2.InverseJacobian(), mGShape2 );

        ( *mFaceStorage[mElementNo] )[i].GShapeFace1.SetSize( numOfNodes1, Trans.GetDimension() );
        ( *mFaceStorage[mElementNo] )[i].GShapeFace2.SetSize( numOfNodes2, Trans.GetDimension() );
        Mult( mGShape1, ( *mFaceStorage[mElementNo] )[i].Jacobian, ( *mFaceStorage[mElementNo] )[i].GShapeFace1 );
        Mult( mGShape2, ( *mFaceStorage[mElementNo] )[i].Jacobian, ( *mFaceStorage[mElementNo] )[i].GShapeFace2 );
    }
}

const mfem::Vector& Memorize::GetFace1Shape( const int gauss ) const
{
    return ( *mFaceStorage[mElementNo] )[gauss].Shape1;
}

const mfem::Vector& Memorize::GetFace2Shape( const int gauss ) const
{
    return ( *mFaceStorage[mElementNo] )[gauss].Shape2;
}

const mfem::DenseMatrix& Memorize::GetFace1GShape( const int gauss ) const
{
    return ( *mFaceStorage[mElementNo] )[gauss].GShapeFace1;
}

const mfem::DenseMatrix& Memorize::GetFace2GShape( const int gauss ) const
{
    return ( *mFaceStorage[mElementNo] )[gauss].GShapeFace2;
}

const Eigen::MatrixXd& Memorize::GetdNdX( const int gauss ) const
{
    return ( *mEleStorage[mElementNo] )[gauss].GShape;
}

double Memorize::GetDetdXdXi( const int gauss ) const
{
    return ( *mEleStorage[mElementNo] )[gauss].DetdXdXi;
}

const mfem::DenseMatrix& Memorize::GetFaceJacobian( const int gauss ) const
{
    return ( *mFaceStorage[mElementNo] )[gauss].Jacobian;
}

double Memorize::GetFaceWeight( const int gauss ) const
{
    return ( *mFaceStorage[mElementNo] )[gauss].Weight;
}

void ElasticityIntegrator::AssembleElementMatrix( const mfem::FiniteElement& el, mfem::ElementTransformation& Trans, mfem::DenseMatrix& elmat )
{
    int dof = el.GetDof();
    int dim = el.GetDim();
    double w{ 0 };

    MFEM_ASSERT( dim == Trans.GetSpaceDim(), "" );

    mDShape.SetSize( dof, dim );
    mGShape.SetSize( dof, dim );

    elmat.SetSize( dof * dim );

    Eigen::Map<Eigen::MatrixXd> eigenMat( elmat.Data(), dof * dim, dof * dim );
    Eigen::Matrix<double, 6, Eigen::Dynamic> B( 6, dof * dim );
    B.setZero();

    const mfem::IntegrationRule* ir = IntRule;
    if ( ir == NULL )
    {
        int order = 2 * Trans.OrderGrad( &el ); // correct order?
        ir = &mfem::IntRules.Get( el.GetGeomType(), order );
    }

    elmat = 0.0;

    for ( int i = 0; i < ir->GetNPoints(); i++ )
    {
        const mfem::IntegrationPoint& ip = ir->IntPoint( i );

        Trans.SetIntPoint( &ip );

        mMaterialModel->at( Trans, ip );
        mMaterialModel->updateRefModuli();

        el.CalcDShape( ip, mDShape );

        w = ip.weight * Trans.Weight();
        Mult( mDShape, Trans.InverseJacobian(), mGShape );

        matrixB( dof, dim, mGShape, B );
        eigenMat += w * B.transpose() * mMaterialModel->getRefModuli() * B;
    }
}

void ElasticityIntegrator::matrixB( const int dof,
                                    const int dim,
                                    const mfem::DenseMatrix& gshape,
                                    Eigen::Matrix<double, 6, Eigen::Dynamic>& B ) const
{
    smallDeformMatrixB( dof, dim, Eigen::Map<Eigen::MatrixXd>( gshape.Data(), gshape.Height(), gshape.Width() ), B );
}

void NonlinearElasticityIntegrator::AssembleElementGrad( const mfem::FiniteElement& el,
                                                         mfem::ElementTransformation& Ttr,
                                                         const mfem::Vector& elfun,
                                                         mfem::DenseMatrix& elmat )
{
    double w;
    int dof = el.GetDof(), dim = el.GetDim();

    mGeomStiff.resize( dof, dof );

    Eigen::Map<const Eigen::MatrixXd> u( elfun.GetData(), dof, dim );
    elmat.SetSize( dof * dim );
    elmat = 0.0;

    Eigen::Map<Eigen::MatrixXd> eigenMat( elmat.Data(), dof * dim, dof * dim );

    const Eigen::Matrix3d identity = Eigen::Matrix3d::Identity();

    const mfem::IntegrationRule* ir = IntRule;
    if ( !ir )
    {
        ir = &( mfem::IntRules.Get( el.GetGeomType(), 2 * el.GetOrder() + 1 ) ); // <---
    }
    mMemo.InitializeElement( el, Ttr, *ir );
    for ( int i = 0; i < ir->GetNPoints(); i++ )
    {
        const mfem::IntegrationPoint& ip = ir->IntPoint( i );
        Ttr.SetIntPoint( &ip );
        const Eigen::MatrixXd& gShape = mMemo.GetdNdX( i );
        mdxdX.setZero();
        mdxdX.block( 0, 0, dim, dim ) = u.transpose() * gShape;
        mdxdX += identity;
        matrixB( dof, dim, gShape );

        mMaterialModel->at( Ttr, ip );
        mMaterialModel->setDeformationGradient( mdxdX );
        mMaterialModel->updateRefModuli();

        w = ip.weight * mMemo.GetDetdXdXi( i );
        if ( !onlyGeomStiff() )
            eigenMat += w * mB.transpose() * mMaterialModel->getRefModuli() * mB;
        if ( isNonlinear() || onlyGeomStiff() )
        {
            mGeomStiff =
                ( w * gShape * mMaterialModel->getPK2StressTensor().block( 0, 0, dim, dim ) * gShape.transpose() ).eval();
            for ( int j = 0; j < dim; j++ )
            {
                eigenMat.block( j * dof, j * dof, dof, dof ) += mGeomStiff;
            }
        }
    }
}

void NonlinearElasticityIntegrator::AssembleElementVector( const mfem::FiniteElement& el,
                                                           mfem::ElementTransformation& Ttr,
                                                           const mfem::Vector& elfun,
                                                           mfem::Vector& elvect )
{
    double w;
    int dof = el.GetDof(), dim = el.GetDim();

    Eigen::Map<const Eigen::MatrixXd> u( elfun.GetData(), dof, dim );

    elvect.SetSize( dof * dim );
    elvect = 0.0;
    Eigen::Map<Eigen::VectorXd> eigenVec( elvect.GetData(), dof * dim );

    const mfem::IntegrationRule* ir = IntRule;
    if ( !ir )
    {
        ir = &( mfem::IntRules.Get( el.GetGeomType(), 2 * el.GetOrder() + 1 ) ); // <---
    }

    const Eigen::Matrix3d identity = Eigen::Matrix3d::Identity();
    mMemo.InitializeElement( el, Ttr, *ir );
    for ( int i = 0; i < ir->GetNPoints(); i++ )
    {
        const mfem::IntegrationPoint& ip = ir->IntPoint( i );
        Ttr.SetIntPoint( &ip );
        const Eigen::MatrixXd& gShape = mMemo.GetdNdX( i );

        mdxdX.setZero();
        mdxdX.block( 0, 0, dim, dim ) = u.transpose() * gShape;
        mdxdX += identity;
        matrixB( dof, dim, gShape );

        mMaterialModel->at( Ttr, ip );
        mMaterialModel->setDeformationGradient( mdxdX );
        mMaterialModel->updateRefModuli();

        w = ip.weight * mMemo.GetDetdXdXi( i );
        eigenVec += w * ( mB.transpose() * mMaterialModel->getPK2StressVector() );
    }
    // std::cout<<"Rhs:\n";
    // std::cout<<eigenVec<<std::endl;
}

void NonlinearElasticityIntegrator::matrixB( const int dof, const int dim, const Eigen::MatrixXd& gshape )
{
    mB.resize( 6, dof * dim );
    mB.setZero();
    if ( isNonlinear() )
    {
        if ( dim == 2 )
        {
            for ( int i = 0; i < dof; i++ )
            {
                mB( 0, i + 0 * dof ) = gshape( i, 0 ) * mdxdX( 0, 0 );
                mB( 0, i + 1 * dof ) = gshape( i, 0 ) * mdxdX( 1, 0 );

                mB( 1, i + 0 * dof ) = gshape( i, 1 ) * mdxdX( 0, 1 );
                mB( 1, i + 1 * dof ) = gshape( i, 1 ) * mdxdX( 1, 1 );

                mB( 3, i + 0 * dof ) = gshape( i, 1 ) * mdxdX( 0, 0 ) + gshape( i, 0 ) * mdxdX( 0, 1 );
                mB( 3, i + 1 * dof ) = gshape( i, 1 ) * mdxdX( 1, 0 ) + gshape( i, 0 ) * mdxdX( 1, 1 );
            }
        }
        else if ( dim == 3 )
        {
            for ( int i = 0; i < dof; i++ )
            {
                mB( 0, i + 0 * dof ) = gshape( i, 0 ) * mdxdX( 0, 0 );
                mB( 0, i + 1 * dof ) = gshape( i, 0 ) * mdxdX( 1, 0 );
                mB( 0, i + 2 * dof ) = gshape( i, 0 ) * mdxdX( 2, 0 );

                mB( 1, i + 0 * dof ) = gshape( i, 1 ) * mdxdX( 0, 1 );
                mB( 1, i + 1 * dof ) = gshape( i, 1 ) * mdxdX( 1, 1 );
                mB( 1, i + 2 * dof ) = gshape( i, 1 ) * mdxdX( 2, 1 );

                mB( 2, i + 0 * dof ) = gshape( i, 2 ) * mdxdX( 0, 2 );
                mB( 2, i + 1 * dof ) = gshape( i, 2 ) * mdxdX( 1, 2 );
                mB( 2, i + 2 * dof ) = gshape( i, 2 ) * mdxdX( 2, 2 );

                mB( 3, i + 0 * dof ) = gshape( i, 1 ) * mdxdX( 0, 0 ) + gshape( i, 0 ) * mdxdX( 0, 1 );
                mB( 3, i + 1 * dof ) = gshape( i, 1 ) * mdxdX( 1, 0 ) + gshape( i, 0 ) * mdxdX( 1, 1 );
                mB( 3, i + 2 * dof ) = gshape( i, 1 ) * mdxdX( 2, 0 ) + gshape( i, 0 ) * mdxdX( 2, 1 );

                mB( 4, i + 0 * dof ) = gshape( i, 2 ) * mdxdX( 0, 1 ) + gshape( i, 1 ) * mdxdX( 0, 2 );
                mB( 4, i + 1 * dof ) = gshape( i, 2 ) * mdxdX( 1, 1 ) + gshape( i, 1 ) * mdxdX( 1, 2 );
                mB( 4, i + 2 * dof ) = gshape( i, 2 ) * mdxdX( 2, 1 ) + gshape( i, 1 ) * mdxdX( 2, 2 );

                mB( 5, i + 0 * dof ) = gshape( i, 0 ) * mdxdX( 0, 2 ) + gshape( i, 2 ) * mdxdX( 0, 0 );
                mB( 5, i + 1 * dof ) = gshape( i, 0 ) * mdxdX( 1, 2 ) + gshape( i, 2 ) * mdxdX( 1, 0 );
                mB( 5, i + 2 * dof ) = gshape( i, 0 ) * mdxdX( 2, 2 ) + gshape( i, 2 ) * mdxdX( 2, 0 );
            }
        }
        else
        {
            MFEM_WARNING( "It is not for 1D analysis." );
        }
    }
    else
    {
        smallDeformMatrixB( dof, dim, gshape, mB );
    }
}

void NonlinearVectorBoundaryLFIntegrator::AssembleFaceVector( const mfem::FiniteElement& el1,
                                                              const mfem::FiniteElement& el2,
                                                              mfem::FaceElementTransformations& Tr,
                                                              const mfem::Vector& elfun,
                                                              mfem::Vector& elvect )
{
    int vdim = Tr.GetSpaceDim();
    int dof = el1.GetDof();

    shape.SetSize( dof );
    vec.SetSize( vdim );

    elvect.SetSize( dof * vdim );
    elvect = 0.0;

    const mfem::IntegrationRule* ir = IntRule;
    if ( ir == NULL )
    {
        int intorder = 2 * el1.GetOrder();
        ir = &mfem::IntRules.Get( Tr.GetGeometryType(), intorder );
    }

    for ( int i = 0; i < ir->GetNPoints(); i++ )
    {
        const mfem::IntegrationPoint& ip = ir->IntPoint( i );

        // Set the integration point in the face and the neighboring element
        Tr.SetAllIntPoints( &ip );

        // Access the neighboring element's integration point
        const mfem::IntegrationPoint& eip = Tr.GetElement1IntPoint();

        // Use Tr transformation in case Q depends on boundary attribute
        Q.Eval( vec, Tr, ip );
        vec *= Tr.Weight() * ip.weight * GetLambda();
        // vec.Print();
        // std::cout << GetLambda() << " " << Tr.Attribute << std::endl;
        el1.CalcShape( eip, shape );

        for ( int k = 0; k < vdim; k++ )
        {
            for ( int s = 0; s < dof; s++ )
            {
                // move r.h.s to l.h.s, hence minus
                elvect( dof * k + s ) -= vec( k ) * shape( s );
            }
        }
    }
}

void NonlinearVectorBoundaryLFIntegrator::AssembleFaceGrad( const mfem::FiniteElement& el1,
                                                            const mfem::FiniteElement& el2,
                                                            mfem::FaceElementTransformations& Tr,
                                                            const mfem::Vector& elfun,
                                                            mfem::DenseMatrix& elmat )
{
    int vdim = Tr.GetSpaceDim();
    int dof = el1.GetDof();

    shape.SetSize( dof );
    vec.SetSize( vdim );

    elmat.SetSize( dof * vdim );
    elmat = 0.0;
}

void NonlinearPressureIntegrator::AssembleFaceVector( const mfem::FiniteElement& el1,
                                                      const mfem::FiniteElement& el2,
                                                      mfem::FaceElementTransformations& Tr,
                                                      const mfem::Vector& elfun,
                                                      mfem::Vector& elvect )
{
    int vdim = Tr.GetSpaceDim();
    int dof = el1.GetDof();
    MFEM_ASSERT( vdim == 2, "NonlinearPressureIntegrator only support 2D elements" );

    shape.SetSize( dof );
    mDShape.SetSize( dof, vdim );
    mGShape.SetSize( dof, vdim );
    mdxdX.resize( vdim, vdim );

    elvect.SetSize( dof * vdim );
    elvect = 0.0;
    Eigen::Map<Eigen::VectorXd> eigenVec( elvect.GetData(), elvect.Size() );
    Eigen::Map<const Eigen::MatrixXd> u( elfun.GetData(), dof, vdim );

    const mfem::IntegrationRule* ir = IntRule;
    if ( ir == NULL )
    {
        int intorder = 2 * el1.GetOrder();
        ir = &mfem::IntRules.Get( Tr.GetGeometryType(), intorder );
    }

    Eigen::Rotation2Dd r( EIGEN_PI / 2 );

    const Eigen::MatrixXd identity = Eigen::MatrixXd::Identity( vdim, vdim );

    auto& Ttr = Tr.GetElement1Transformation();

    for ( int i = 0; i < ir->GetNPoints(); i++ )
    {
        const mfem::IntegrationPoint& ip = ir->IntPoint( i );

        // Set the integration point in the face and the neighboring element
        Tr.SetAllIntPoints( &ip );

        // Access the neighboring element's integration point
        const mfem::IntegrationPoint& eip = Tr.GetElement1IntPoint();

        // Use Tr transformation in case Q depends on boundary attribute
        const double val = Q.Eval( Tr, ip ) * GetLambda();
        // vec *= Tr.Weight() * ip.weight;
        el1.CalcShape( eip, shape );
        el1.CalcDShape( eip, mDShape );
        Mult( mDShape, Ttr.InverseJacobian(), mGShape );
        mdxdX = u.transpose() * Eigen::Map<const Eigen::MatrixXd>( mGShape.Data(), dof, vdim ) + identity;

        Eigen::Map<const Eigen::MatrixXd> vec( shape.GetData(), 1, dof );

        Eigen::Map<const Eigen::MatrixXd> Jac( Tr.Jacobian().Data(), Tr.Jacobian().NumRows(), Tr.Jacobian().NumCols() );

        Eigen::VectorXd dxdxi = mdxdX * Jac;

        eigenVec -= Eigen::kroneckerProduct( identity, vec ).transpose() * r.toRotationMatrix() * dxdxi.normalized() *
                    dxdxi.norm() * ip.weight * val;
    }
}

void NonlinearPressureIntegrator::AssembleFaceGrad( const mfem::FiniteElement& el1,
                                                    const mfem::FiniteElement& el2,
                                                    mfem::FaceElementTransformations& Tr,
                                                    const mfem::Vector& elfun,
                                                    mfem::DenseMatrix& elmat )
{
    int vdim = Tr.GetSpaceDim();
    int dof = el1.GetDof();
    MFEM_ASSERT( vdim == 2, "NonlinearPressureIntegrator only support 2D elements" );

    shape.SetSize( dof );
    mDShape.SetSize( dof, vdim );
    mGShape.SetSize( dof, vdim );
    mdxdX.resize( vdim, vdim );

    elmat.SetSize( dof * vdim );
    elmat = 0.0;
    Eigen::Map<Eigen::MatrixXd> eigenMat( elmat.Data(), dof * vdim, dof * vdim );
    Eigen::Map<const Eigen::MatrixXd> u( elfun.GetData(), dof, vdim );

    Eigen::VectorXd dxdxi;
    Eigen::MatrixXd deltau;

    const mfem::IntegrationRule* ir = IntRule;
    if ( ir == NULL )
    {
        int intorder = 2 * el1.GetOrder();
        ir = &mfem::IntRules.Get( Tr.GetGeometryType(), intorder );
    }

    Eigen::Rotation2Dd r( EIGEN_PI / 2 );

    auto& Ttr = Tr.GetElement1Transformation();

    const Eigen::MatrixXd identity = Eigen::MatrixXd::Identity( vdim, vdim );

    for ( int i = 0; i < ir->GetNPoints(); i++ )
    {
        const mfem::IntegrationPoint& ip = ir->IntPoint( i );

        // Set the integration point in the face and the neighboring element
        Tr.SetAllIntPoints( &ip );

        // Access the neighboring element's integration point
        const mfem::IntegrationPoint& eip = Tr.GetElement1IntPoint();

        // Use Tr transformation in case Q depends on boundary attribute
        const double val = Q.Eval( Tr, ip ) * GetLambda();
        // vec *= Tr.Weight() * ip.weight;
        el1.CalcShape( eip, shape );
        el1.CalcDShape( eip, mDShape );
        Mult( mDShape, Ttr.InverseJacobian(), mGShape );
        Eigen::Map<const Eigen::MatrixXd> eigenGShape( mGShape.Data(), dof, vdim );
        mdxdX = u.transpose() * eigenGShape + identity;

        Eigen::Map<const Eigen::MatrixXd> vec( shape.GetData(), 1, dof );

        Eigen::Map<const Eigen::MatrixXd> Jac( Tr.Jacobian().Data(), Tr.Jacobian().NumRows(), Tr.Jacobian().NumCols() );

        dxdxi = mdxdX * Jac;
        deltau = Eigen::kroneckerProduct( identity, eigenGShape * Jac ).transpose();
        mB = deltau / std::pow( dxdxi.dot( dxdxi ), .5 ) -
             dxdxi * ( dxdxi.transpose() * deltau ) / std::pow( dxdxi.dot( dxdxi ), 1.5 );
        eigenMat -= ip.weight * val * Eigen::kroneckerProduct( identity, vec ).transpose() * r.toRotationMatrix() *
                    ( mB * dxdxi.norm() + dxdxi.normalized() * ( dxdxi.transpose() * deltau ) / dxdxi.norm() );
    }
}

void NonlinearCompositeSolidShellIntegrator::AssembleElementGrad( const mfem::FiniteElement& el,
                                                                  mfem::ElementTransformation& Ttr,
                                                                  const mfem::Vector& elfun,
                                                                  mfem::DenseMatrix& elmat )
{
    double w = 0;
    int dof = el.GetDof(), dim = el.GetDim();

    MFEM_ASSERT( dim == 3 && dof == 8, "NonlinearCompositeSolidShellIntegrator only support linearHex elements" );

    Eigen::Map<const Eigen::MatrixXd> u( elfun.GetData(), dof, dim );
    elmat.SetSize( dof * dim );
    elmat = 0.0;

    Eigen::Map<Eigen::MatrixXd> eigenMat( elmat.Data(), dof * dim, dof * dim );

    mfem::DenseMatrix mat;
    mfem::IntegrationPoint ip;

    // from [-1, 1] to [0, 1]
    auto convert = []( double& x ) { x = ( x + 1 ) / 2.; };
    double pt[3];
    pt[0] = .5, pt[1] = .5, pt[2] = .5;
    Ttr.SetIntPoint( &ip );
    mMaterialModel->at( Ttr, ip );
    mMaterialModel->updateRefModuli();
    mStiffModuli = mMaterialModel->getRefModuli();
    const Eigen::Matrix3d orthonormalBasis = Eigen::Matrix3d::Identity();

    auto preprocessColl = [&]( Eigen::Matrix<double, 3, 3>& g, Eigen::Matrix<double, 8, 3>& DShape )
    {
        mat.UseExternalData( mGCovariant.data(), mGCovariant.rows(), mGCovariant.cols() );
        mat = Ttr.Jacobian();
        mat.UseExternalData( DShape.data(), DShape.rows(), DShape.cols() );
        el.CalcDShape( ip, mat );
        g = mGCovariant + u.transpose() * DShape;
    };

    // prepare collocation points
    {
        // point A
        pt[0] = -1, pt[1] = 0, pt[2] = 0;
        convert( pt[0] ), convert( pt[1] ), convert( pt[2] );
        ip.Set3( pt[0], pt[1], pt[2] );
        preprocessColl( mgA, mDShapeA );
        // point B
        pt[0] = 0, pt[1] = -1, pt[2] = 0;
        convert( pt[0] ), convert( pt[1] ), convert( pt[2] );
        ip.Set3( pt[0], pt[1], pt[2] );
        preprocessColl( mgB, mDShapeB );
        // point C
        pt[0] = 1, pt[1] = 0, pt[2] = 0;
        convert( pt[0] ), convert( pt[1] ), convert( pt[2] );
        ip.Set3( pt[0], pt[1], pt[2] );
        preprocessColl( mgC, mDShapeC );
        // point D
        pt[0] = 0, pt[1] = 1, pt[2] = 0;
        convert( pt[0] ), convert( pt[1] ), convert( pt[2] );
        ip.Set3( pt[0], pt[1], pt[2] );
        preprocessColl( mgD, mDShapeD );

        // point A1
        pt[0] = -1, pt[1] = -1, pt[2] = 0;
        convert( pt[0] ), convert( pt[1] ), convert( pt[2] );
        ip.Set3( pt[0], pt[1], pt[2] );
        preprocessColl( mgA1, mDShapeA1 );
        // point A2
        pt[0] = 1, pt[1] = -1, pt[2] = 0;
        convert( pt[0] ), convert( pt[1] ), convert( pt[2] );
        ip.Set3( pt[0], pt[1], pt[2] );
        preprocessColl( mgA2, mDShapeA2 );
        // point A3
        pt[0] = 1, pt[1] = 1, pt[2] = 0;
        convert( pt[0] ), convert( pt[1] ), convert( pt[2] );
        ip.Set3( pt[0], pt[1], pt[2] );
        preprocessColl( mgA3, mDShapeA3 );
        // point A4
        pt[0] = -1, pt[1] = 1, pt[2] = 0;
        convert( pt[0] ), convert( pt[1] ), convert( pt[2] );
        ip.Set3( pt[0], pt[1], pt[2] );
        preprocessColl( mgA4, mDShapeA4 );

        std::cout << mDShapeB << std::endl << std::endl;
        std::cout << mDShapeA4 << std::endl;

        mfem::Vector v;
        v.SetSize( 8 );
        ip.Set3( 0, 0, 0 );
        el.CalcShape( ip, v );
        v.Print();
        ip.Set3( 1, 0, 0 );
        el.CalcShape( ip, v );
        v.Print();
        ip.Set3( 1, 1, 0 );
        el.CalcShape( ip, v );
        v.Print();
        ip.Set3( 0, 1, 0 );
        el.CalcShape( ip, v );
        v.Print();
        ip.Set3( 0, 0, 1 );
        el.CalcShape( ip, v );
        v.Print();
        ip.Set3( 1, 0, 1 );
        el.CalcShape( ip, v );
        v.Print();
        ip.Set3( 1, 1, 1 );
        el.CalcShape( ip, v );
        v.Print();
        ip.Set3( 0, 1, 1 );
        el.CalcShape( ip, v );
        v.Print();
    }

    const mfem::IntegrationRule* ir = IntRule;
    if ( !ir )
    {
        ir = &( mfem::IntRules.Get( el.GetGeomType(), 2 * el.GetOrder() + 1 ) ); // <---
    }

    for ( int i = 0; i < ir->GetNPoints(); i++ )
    {
        const mfem::IntegrationPoint& ip = ir->IntPoint( i );
        Ttr.SetIntPoint( &ip );

        preprocessColl( mg, mDShape );
        matrixB( dof, dim, ip );

        mMaterialModel->at( Ttr, ip );
        mMaterialModel->updateRefModuli();

        w = ip.weight * Ttr.Weight();
        mGContravariant = mGCovariant.inverse();
        Eigen::Matrix3d T = orthonormalBasis.transpose() * mGContravariant;
        mTransform = util::TransformationVoigtForm( T );
        matrixB( dof, dim, ip );
        // mGeomStiff =
        //     ( w * mGShapeEig * mMaterialModel->getPK2StressTensor().block( 0, 0, dim, dim ) * mGShapeEig.transpose() ).eval();
        eigenMat += w * mB.transpose() * ( mTransform.transpose() * mStiffModuli * mTransform ) * mB;
        // for ( int j = 0; j < dim; j++ )
        // {
        //     eigenMat.block( j * dof, j * dof, dof, dof ) += mGeomStiff;
        // }
    }
}

void NonlinearCompositeSolidShellIntegrator::matrixB( const int dof, const int dim, const mfem::IntegrationPoint& ip )
{
    // from [0, 1] to [-1, 1]
    auto convert = []( double& x ) { x = 2 * x - 1; };
    double pt[3];
    ip.Get( pt, 3 );
    convert( pt[0] );
    convert( pt[1] );
    convert( pt[2] );
    for ( int i = 0; i < dof; i++ )
    {
        mB( 0, i + 0 * dof ) = mDShape( i, 0 ) * mg( 0, 0 );
        mB( 0, i + 1 * dof ) = mDShape( i, 0 ) * mg( 1, 0 );
        mB( 0, i + 2 * dof ) = mDShape( i, 0 ) * mg( 2, 0 );

        mB( 1, i + 0 * dof ) = mDShape( i, 1 ) * mg( 0, 1 );
        mB( 1, i + 1 * dof ) = mDShape( i, 1 ) * mg( 1, 1 );
        mB( 1, i + 2 * dof ) = mDShape( i, 1 ) * mg( 2, 1 );

        mB( 2, i + 0 * dof ) = 1. / 4 *
                               ( ( 1 - 1 * pt[0] ) * ( 1 - 1 * pt[1] ) * mgA1( 0, 2 ) * mDShapeA1( i, 2 ) +
                                 ( 1 + 1 * pt[0] ) * ( 1 - 1 * pt[1] ) * mgA2( 0, 2 ) * mDShapeA2( i, 2 ) +
                                 ( 1 + 1 * pt[0] ) * ( 1 + 1 * pt[1] ) * mgA3( 0, 2 ) * mDShapeA3( i, 2 ) +
                                 ( 1 - 1 * pt[0] ) * ( 1 + 1 * pt[1] ) * mgA4( 0, 2 ) * mDShapeA4( i, 2 ) );
        mB( 2, i + 1 * dof ) = 1. / 4 *
                               ( ( 1 - 1 * pt[0] ) * ( 1 - 1 * pt[1] ) * mgA1( 1, 2 ) * mDShapeA1( i, 2 ) +
                                 ( 1 + 1 * pt[0] ) * ( 1 - 1 * pt[1] ) * mgA2( 1, 2 ) * mDShapeA2( i, 2 ) +
                                 ( 1 + 1 * pt[0] ) * ( 1 + 1 * pt[1] ) * mgA3( 1, 2 ) * mDShapeA3( i, 2 ) +
                                 ( 1 - 1 * pt[0] ) * ( 1 + 1 * pt[1] ) * mgA4( 1, 2 ) * mDShapeA4( i, 2 ) );
        mB( 2, i + 2 * dof ) = 1. / 4 *
                               ( ( 1 - 1 * pt[0] ) * ( 1 - 1 * pt[1] ) * mgA1( 2, 2 ) * mDShapeA1( i, 2 ) +
                                 ( 1 + 1 * pt[0] ) * ( 1 - 1 * pt[1] ) * mgA2( 2, 2 ) * mDShapeA2( i, 2 ) +
                                 ( 1 + 1 * pt[0] ) * ( 1 + 1 * pt[1] ) * mgA3( 2, 2 ) * mDShapeA3( i, 2 ) +
                                 ( 1 - 1 * pt[0] ) * ( 1 + 1 * pt[1] ) * mgA4( 2, 2 ) * mDShapeA4( i, 2 ) );

        mB( 3, i + 0 * dof ) = mDShape( i, 1 ) * mg( 0, 0 ) + mDShape( i, 0 ) * mg( 0, 1 );
        mB( 3, i + 1 * dof ) = mDShape( i, 1 ) * mg( 1, 0 ) + mDShape( i, 0 ) * mg( 1, 1 );
        mB( 3, i + 2 * dof ) = mDShape( i, 1 ) * mg( 2, 0 ) + mDShape( i, 0 ) * mg( 2, 1 );

        mB( 4, i + 0 * dof ) = 1. / 2 *
                               ( ( 1 - pt[0] ) * ( mgB( 0, 1 ) * mDShapeB( i, 2 ) + mgB( 0, 2 ) * mDShapeB( i, 1 ) ) +
                                 ( 1 + pt[0] ) * ( mgD( 0, 1 ) * mDShapeD( i, 2 ) + mgD( 0, 2 ) * mDShapeD( i, 1 ) ) );
        mB( 4, i + 1 * dof ) = 1. / 2 *
                               ( ( 1 - pt[0] ) * ( mgB( 1, 1 ) * mDShapeB( i, 2 ) + mgB( 1, 2 ) * mDShapeB( i, 1 ) ) +
                                 ( 1 + pt[0] ) * ( mgD( 1, 1 ) * mDShapeD( i, 2 ) + mgD( 1, 2 ) * mDShapeD( i, 1 ) ) );
        mB( 4, i + 2 * dof ) = 1. / 2 *
                               ( ( 1 - pt[0] ) * ( mgB( 2, 1 ) * mDShapeB( i, 2 ) + mgB( 2, 2 ) * mDShapeB( i, 1 ) ) +
                                 ( 1 + pt[0] ) * ( mgD( 2, 1 ) * mDShapeD( i, 2 ) + mgD( 2, 2 ) * mDShapeD( i, 1 ) ) );

        mB( 5, i + 0 * dof ) = 1. / 2 *
                               ( ( 1 - pt[1] ) * ( mgA( 0, 0 ) * mDShapeA( i, 2 ) + mgA( 0, 2 ) * mDShapeA( i, 0 ) ) +
                                 ( 1 + pt[1] ) * ( mgC( 0, 0 ) * mDShapeC( i, 2 ) + mgC( 0, 2 ) * mDShapeC( i, 0 ) ) );
        mB( 5, i + 1 * dof ) = 1. / 2 *
                               ( ( 1 - pt[1] ) * ( mgA( 1, 0 ) * mDShapeA( i, 2 ) + mgA( 1, 2 ) * mDShapeA( i, 0 ) ) +
                                 ( 1 + pt[1] ) * ( mgC( 1, 0 ) * mDShapeC( i, 2 ) + mgC( 1, 2 ) * mDShapeC( i, 0 ) ) );
        mB( 5, i + 2 * dof ) = 1. / 2 *
                               ( ( 1 - pt[1] ) * ( mgA( 2, 0 ) * mDShapeA( i, 2 ) + mgA( 2, 2 ) * mDShapeA( i, 0 ) ) +
                                 ( 1 + pt[1] ) * ( mgC( 2, 0 ) * mDShapeC( i, 2 ) + mgC( 2, 2 ) * mDShapeC( i, 0 ) ) );
    }
}


void NonlinearDirichletPenaltyIntegrator::AssembleFaceVector( const mfem::FiniteElement& el1,
                                                              const mfem::FiniteElement& el2,
                                                              mfem::FaceElementTransformations& Tr,
                                                              const mfem::Vector& elfun,
                                                              mfem::Vector& elvect )
{
    int vdim = Tr.GetSpaceDim();
    int dof = el1.GetDof();

    shape.SetSize( dof );
    dispEval.SetSize( vdim );
    penalEval.SetSize( vdim );

    elvect.SetSize( dof * vdim );
    elvect = 0.0;

    Eigen::Map<const Eigen::VectorXd> u( elfun.GetData(), elfun.Size() );
    Eigen::Map<Eigen::VectorXd> eigenVec( elvect.GetData(), elvect.Size() );
    Eigen::Map<const Eigen::VectorXd> dispEvalEigen( dispEval.GetData(), dispEval.Size() );
    Eigen::Map<const Eigen::VectorXd> penalEvalEigen( penalEval.GetData(), penalEval.Size() );

    const mfem::IntegrationRule* ir = IntRule;
    if ( ir == NULL )
    {
        int intorder = 2 * el1.GetOrder();
        ir = &mfem::IntRules.Get( Tr.GetGeometryType(), intorder );
    }

    for ( int i = 0; i < ir->GetNPoints(); i++ )
    {
        const mfem::IntegrationPoint& ip = ir->IntPoint( i );

        // Set the integration point in the face and the neighboring element
        Tr.SetAllIntPoints( &ip );

        // Access the neighboring element's integration point
        const mfem::IntegrationPoint& eip = Tr.GetElement1IntPoint();

        // Use Tr transformation in case Q depends on boundary attribute
        Q.Eval( dispEval, Tr, ip );
        H.Eval( penalEval, Tr, ip );
        dispEval *= GetLambda();

        el1.CalcShape( eip, shape );

        matrixB( dof, vdim );

        mU = mB * u;
        eigenVec += mB.transpose() * penalEvalEigen.asDiagonal() * ( mU - dispEvalEigen ) * Tr.Weight() * ip.weight;
    }
}

void NonlinearDirichletPenaltyIntegrator::AssembleFaceGrad( const mfem::FiniteElement& el1,
                                                            const mfem::FiniteElement& el2,
                                                            mfem::FaceElementTransformations& Tr,
                                                            const mfem::Vector& elfun,
                                                            mfem::DenseMatrix& elmat )
{
    int vdim = Tr.GetSpaceDim();
    int dof = el1.GetDof();

    shape.SetSize( dof );
    penalEval.SetSize( vdim );

    elmat.SetSize( dof * vdim );
    elmat = 0.0;

    Eigen::Map<Eigen::MatrixXd> eigenMat( elmat.Data(), dof * vdim, dof * vdim );
    Eigen::Map<const Eigen::VectorXd> penalEvalEigen( penalEval.GetData(), penalEval.Size() );

    const mfem::IntegrationRule* ir = IntRule;
    if ( ir == NULL )
    {
        int intorder = 2 * el1.GetOrder();
        ir = &mfem::IntRules.Get( Tr.GetGeometryType(), intorder );
    }

    for ( int i = 0; i < ir->GetNPoints(); i++ )
    {
        const mfem::IntegrationPoint& ip = ir->IntPoint( i );
        // Set the integration point in the face and the neighboring element
        Tr.SetAllIntPoints( &ip );
        H.Eval( penalEval, Tr, ip );
        // Access the neighboring element's integration point
        const mfem::IntegrationPoint& eip = Tr.GetElement1IntPoint();
        el1.CalcShape( eip, shape );
        matrixB( dof, vdim );
        eigenMat += mB.transpose() * penalEvalEigen.asDiagonal() * mB * ip.weight * Tr.Weight();
    }
}

void NonlinearInternalPenaltyIntegrator::AssembleFaceVector( const mfem::FiniteElement& el1,
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

    shape1.SetSize( dof1 );
    shape2.SetSize( dof2 );
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

    for ( int i = 0; i < ir->GetNPoints(); i++ )
    {
        const mfem::IntegrationPoint& ip = ir->IntPoint( i );

        // Set the integration point in the face and the neighboring element
        Tr.SetAllIntPoints( &ip );
        mfem::Vector phy;
        Tr.Transform( ip, phy );
        if ( std::abs( phy( 1 ) ) <= 1e-10 )
            continue;

        // Access the neighboring element's integration point
        const mfem::IntegrationPoint& eip1 = Tr.GetElement1IntPoint();
        const mfem::IntegrationPoint& eip2 = Tr.GetElement2IntPoint();

        el1.CalcShape( eip1, shape1 );
        el2.CalcShape( eip2, shape2 );

        matrixB( dof1, dof2, vdim );
        Eigen::VectorXd Delta = mB * u;
        eigenVec += p * mB.transpose() * Delta * ip.weight * Tr.Weight();
    }
}

void NonlinearInternalPenaltyIntegrator::AssembleFaceGrad( const mfem::FiniteElement& el1,
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
    shape1.SetSize( dof1 );
    shape2.SetSize( dof2 );

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

    for ( int i = 0; i < ir->GetNPoints(); i++ )
    {
        const mfem::IntegrationPoint& ip = ir->IntPoint( i );

        // Set the integration point in the face and the neighboring element
        Tr.SetAllIntPoints( &ip );
        mfem::Vector phy;
        Tr.Transform( ip, phy );
        if ( std::abs( phy( 1 ) ) <= 1e-10 )
            continue;

        // Access the neighboring element's integration point
        const mfem::IntegrationPoint& eip1 = Tr.GetElement1IntPoint();
        const mfem::IntegrationPoint& eip2 = Tr.GetElement2IntPoint();
        el1.CalcShape( eip1, shape1 );
        el2.CalcShape( eip2, shape2 );

        matrixB( dof1, dof2, vdim );

        eigenMat += p * mB.transpose() * mB * ip.weight * Tr.Weight();
    }
}
} // namespace plugin