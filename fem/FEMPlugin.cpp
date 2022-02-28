#include "FEMPlugin.h"
#include <Eigen/Dense>
#include <iostream>

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
using namespace mfem;
std::vector<std::unique_ptr<IsoparametricTransformation>> ElasticityIntegrator::refEleTransVec;

void ElasticityIntegrator::resizeRefEleTransVec( const size_t size )
{
    refEleTransVec.resize( size );
}

void ElasticityIntegrator::AssembleElementMatrix( const FiniteElement& el, ElementTransformation& Trans, DenseMatrix& elmat )
{
    int dof = el.GetDof();
    int dim = el.GetDim();
    double w{ 0 };
    const int eleNum = Trans.ElementNo;

    MFEM_ASSERT( dim == Trans.GetSpaceDim(), "" );
    MFEM_ASSERT( (size_t)eleNum < ElasticityIntegrator::refEleTransVec.size(),
                 "ElasticityIntegrator::refEleTransVec has not been "
                 "initiated yet." );
    if ( ElasticityIntegrator::refEleTransVec[eleNum] == nullptr )
        ElasticityIntegrator::refEleTransVec[eleNum] =
            std::make_unique<IsoparametricTransformation>( dynamic_cast<IsoparametricTransformation&>( Trans ) );

    mDShape.SetSize( dof, dim );
    mGShape.SetSize( dof, dim );

    elmat.SetSize( dof * dim );

    mdxdX.setZero();

    Eigen::Map<Eigen::MatrixXd> eigenMat( elmat.Data(), dof * dim, dof * dim );
    Eigen::Matrix<double, 6, Eigen::Dynamic> B( 6, dof * dim );

    const IntegrationRule* ir = IntRule;
    if ( ir == NULL )
    {
        int order = 2 * Trans.OrderGrad( &el ); // correct order?
        ir = &IntRules.Get( el.GetGeomType(), order );
    }

    elmat = 0.0;

    for ( int i = 0; i < ir->GetNPoints(); i++ )
    {
        const IntegrationPoint& ip = ir->IntPoint( i );

        Trans.SetIntPoint( &ip );
        updateDeformationGradient( dim, *ElasticityIntegrator::refEleTransVec[eleNum], Trans, ip );

        mMaterialModel->at( Trans, ip );
        mMaterialModel->setDeformationGradient( mdxdX );
        mMaterialModel->updateRefModuli();
        mMaterialModel->updateCurModuli();

        el.CalcDShape( ip, mDShape );

        w = ip.weight * Trans.Weight();
        Mult( mDShape, Trans.InverseJacobian(), mGShape );

        matrixB( dof, dim, mGShape, B );
        eigenMat += w * B.transpose() * mMaterialModel->getCurModuli() * B;
    }
}

void ElasticityIntegrator::matrixB( const int dof,
                                    const int dim,
                                    const mfem::DenseMatrix& gshape,
                                    Eigen::Matrix<double, 6, Eigen::Dynamic>& B ) const
{
    B.setZero();
    if ( dim == 2 )
    {
        for ( int i = 0; i < dof; i++ )
        {
            B( 0, i ) = gshape( i, 0 );
            B( 1, i + dof ) = gshape( i, 1 );

            B( 3, i ) = gshape( i, 1 );
            B( 3, i + dof ) = gshape( i, 0 );
        }
    }
    else if ( dim == 3 )
    {
        for ( int i = 0; i < dof; i++ )
        {
            B( 0, i ) = gshape( i, 0 );
            B( 1, i + dof ) = gshape( i, 1 );
            B( 2, i + 2 * dof ) = gshape( i, 2 );

            B( 3, i ) = gshape( i, 1 );
            B( 3, i + dof ) = gshape( i, 0 );

            B( 4, i ) = gshape( i, 2 );
            B( 4, i + 2 * dof ) = gshape( i, 0 );

            B( 5, i + dof ) = gshape( i, 2 );
            B( 5, i + 2 * dof ) = gshape( i, 1 );
        }
    }
    else
    {
        MFEM_WARNING( "It is not for 1D analysis." );
    }
}

void ElasticityIntegrator::updateDeformationGradient( const int dim, ElementTransformation& ref, ElementTransformation& cur, const IntegrationPoint& ip )
{
    ref.SetIntPoint( &ip );
    cur.SetIntPoint( &ip );
    const auto& refInvJac = ref.InverseJacobian();
    const auto& curJac = cur.Jacobian();

    Eigen::Map<const Eigen::MatrixXd> refInvJacEig( refInvJac.Data(), dim, dim );
    Eigen::Map<const Eigen::MatrixXd> curJacEig( curJac.Data(), dim, dim );
    mdxdX.setZero();
    // std::cout << "curJacEig: \n";
    // std::cout << curJacEig << std::endl;
    // std::cout << "refInvJacEig: \n";
    // std::cout << refInvJacEig << std::endl;
    mdxdX.block( 0, 0, dim, dim ) = curJacEig * refInvJacEig;
    if ( dim == 2 )
    {
        mdxdX( 2, 2 ) = 1;
    }
}

ElasticityIntegrator::~ElasticityIntegrator()
{
    if ( ElasticityIntegrator::refEleTransVec.size() )
    {
        ElasticityIntegrator::refEleTransVec.clear();
    }
}

void NonlinearElasticityIntegrator::AssembleElementGrad( const FiniteElement& el, ElementTransformation& Ttr, const Vector& elfun, DenseMatrix& elmat )
{
    double w;
    int dof = el.GetDof(), dim = el.GetDim();

    mDShape.SetSize( dof, dim );
    mGShape.SetSize( dof, dim );
    mGeomStiff.resize( dof, dof );

    Eigen::Map<const Eigen::MatrixXd> curCoords( elfun.GetData(), dof, dim );
    elmat.SetSize( dof * dim );
    elmat = 0.0;

    Eigen::Map<Eigen::MatrixXd> eigenMat( elmat.Data(), dof * dim, dof * dim );

    const IntegrationRule* ir = IntRule;
    if ( !ir )
    {
        ir = &( IntRules.Get( el.GetGeomType(), 2 * el.GetOrder() + 3 ) ); // <---
    }
    for ( int i = 0; i < ir->GetNPoints(); i++ )
    {
        const IntegrationPoint& ip = ir->IntPoint( i );
        Ttr.SetIntPoint( &ip );

        el.CalcDShape( ip, mDShape );
        Mult( mDShape, Ttr.InverseJacobian(), mGShape );

        Eigen::Map<const Eigen::MatrixXd> mGShapeEig( mGShape.Data(), dof, dim );

        mdxdX.setZero();
        mdxdX.block( 0, 0, dim, dim ) = curCoords.transpose() * mGShapeEig;
        if ( dim == 2 )
        {
            mdxdX( 2, 2 ) = 1;
        }

        matrixB( dof, dim, mGShape );

        mMaterialModel->at( Ttr, ip );
        mMaterialModel->setDeformationGradient( mdxdX );
        mMaterialModel->updateRefModuli();

        w = ip.weight * Ttr.Weight();

        mGeomStiff =
            (w * mGShapeEig * mMaterialModel->getPK2StressTensor().block( 0, 0, dim, dim ) * mGShapeEig.transpose()).eval();
        eigenMat += w * mB.transpose() * mMaterialModel->getRefModuli() * mB;
        for ( int j = 0; j < dim; j++ )
        {
            eigenMat.block( j * dof, j * dof, dof, dof ) += mGeomStiff;
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

    mDShape.SetSize( dof, dim );
    mGShape.SetSize( dof, dim );

    Eigen::Map<const Eigen::MatrixXd> curCoords( elfun.GetData(), dof, dim );

    elvect.SetSize( dof * dim );
    elvect = 0.0;
    Eigen::Map<Eigen::VectorXd> eigenVec( elvect.GetData(), dof * dim );

    const IntegrationRule* ir = IntRule;
    if ( !ir )
    {
        ir = &( IntRules.Get( el.GetGeomType(), 2 * el.GetOrder() + 3 ) ); // <---
    }
    for ( int i = 0; i < ir->GetNPoints(); i++ )
    {
        const IntegrationPoint& ip = ir->IntPoint( i );
        Ttr.SetIntPoint( &ip );

        el.CalcDShape( ip, mDShape );
        Mult( mDShape, Ttr.InverseJacobian(), mGShape );

        Eigen::Map<const Eigen::MatrixXd> mGShapeEig( mGShape.Data(), dof, dim );

        mdxdX.setZero();
        mdxdX.block( 0, 0, dim, dim ) = curCoords.transpose() * mGShapeEig;
        if ( dim == 2 )
        {
            mdxdX( 2, 2 ) = 1;
        }

        matrixB( dof, dim, mGShape );

        mMaterialModel->at( Ttr, ip );
        mMaterialModel->setDeformationGradient( mdxdX );
        mMaterialModel->updateRefModuli();

        w = ip.weight * Ttr.Weight();
        eigenVec += w * ( mB.transpose() * mMaterialModel->getPK2StressVector() );
    }
}

void NonlinearElasticityIntegrator::matrixB( const int dof, const int dim, const mfem::DenseMatrix& gshape )
{
    mB.resize( 6, dof * dim );
    mB.setZero();
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

void NonlinearVectorBoundaryLFIntegrator::AssembleFaceVector( const mfem::FiniteElement& el1,
                                                              const mfem::FiniteElement& el2,
                                                              mfem::FaceElementTransformations& Tr,
                                                              const mfem::Vector& elfun,
                                                              mfem::Vector& elvect )
{
    int vdim = Q.GetVDim();
    int dof = el1.GetDof();

    shape.SetSize( dof );
    vec.SetSize( vdim );

    elvect.SetSize( dof * vdim );
    elvect = 0.0;

    const IntegrationRule* ir = IntRule;
    if ( ir == NULL )
    {
        int intorder = 2 * el1.GetOrder();
        ir = &IntRules.Get( Tr.GetGeometryType(), intorder );
    }

    for ( int i = 0; i < ir->GetNPoints(); i++ )
    {
        const IntegrationPoint& ip = ir->IntPoint( i );

        // Set the integration point in the face and the neighboring element
        Tr.SetAllIntPoints( &ip );

        // Access the neighboring element's integration point
        const IntegrationPoint& eip = Tr.GetElement1IntPoint();

        // Use Tr transformation in case Q depends on boundary attribute
        Q.Eval( vec, Tr, ip );
        // vec.Print();
        vec *= Tr.Weight() * ip.weight;
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
    int vdim = Q.GetVDim();
    int dof = el1.GetDof();

    shape.SetSize( dof );
    vec.SetSize( vdim );

    elmat.SetSize( dof * vdim );
    elmat = 0.0;
}
} // namespace plugin