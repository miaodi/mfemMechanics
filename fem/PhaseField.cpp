#include "PhaseField.h"
#include "Solvers.h"

namespace plugin
{

void PhaseFieldIntegrator::UpdateH( const int gauss, double& H )
{
    auto& pd = mMemo.GetBodyPointData( gauss );
    // historical strain energy+ for KKT condition
    if ( mIterAux->IterNumber() == 0 )
    {
        if ( mIterAux->StepNumber() == 0 )
        {
            if ( !pd.get_val<PointData>( "H" ).has_value() )
                pd.set_val<PointData>( "H", std::move( PointData() ) );
        }
        else
        {
            auto& H_data = pd.get_val<PointData>( "H" ).value().get();
            if ( mIterAux->Convergence() )
            {
                if ( mIterAux->StepNumber() > H_data.success_step )
                {
                    H_data.H_bac = H_data.H;
                    H_data.success_step = mIterAux->StepNumber();
                }
            }
            else
            {
                H_data.H = H_data.H_bac;
            }
        }
    }

    auto& H_data = pd.get_val<PointData>( "H" ).value().get();
    if ( H < H_data.H )
    {
        H = H_data.H;
    }
    else
    {
        H_data.H = H;
    }
}

void PhaseFieldIntegrator::AssembleElementVector( const mfem::Array<const mfem::FiniteElement*>& el,
                                                  mfem::ElementTransformation& Tr,
                                                  const mfem::Array<const mfem::Vector*>& elfun,
                                                  const mfem::Array<mfem::Vector*>& elvec )
{
    double w;
    int dof_u = el[0]->GetDof(), dim = el[0]->GetDim();
    int dof_p = el[1]->GetDof();

    mDShape.SetSize( dof_p, dim );
    mGShape.SetSize( dof_p, dim );
    shape.SetSize( dof_p );

    const double gc = mMaterialModel->getGc();
    const double k = mMaterialModel->getK();
    const double l0 = mMaterialModel->getL0();

    // mGeomStiff.resize( dof_u, dof_u );

    Eigen::Map<const Eigen::MatrixXd> u( elfun[0]->GetData(), dof_u, dim );
    Eigen::Map<const Eigen::VectorXd> p( elfun[1]->GetData(), dof_p );

    elvec[0]->SetSize( dof_u * dim );
    elvec[1]->SetSize( dof_p );

    *elvec[0] = 0.0;
    *elvec[1] = 0.0;

    Eigen::Map<Eigen::VectorXd> eigenVec0( elvec[0]->GetData(), dof_u * dim );
    Eigen::Map<Eigen::VectorXd> eigenVec1( elvec[1]->GetData(), dof_p );

    const mfem::IntegrationRule* ir = &( mfem::IntRules.Get( el[0]->GetGeomType(), 2 * el[0]->GetOrder() + 1 ) ); // <---

    const Eigen::Matrix3d identity = Eigen::Matrix3d::Identity();
    mMemo.InitializeElement( *el[0], Tr, *ir );
    for ( int i = 0; i < ir->GetNPoints(); i++ )
    {
        const mfem::IntegrationPoint& ip = ir->IntPoint( i );
        Tr.SetIntPoint( &ip );
        const Eigen::MatrixXd& gShape = mMemo.GetdNdX( i );
        mdxdX.setZero();
        mdxdX.block( 0, 0, dim, dim ) = u.transpose() * gShape;
        mdxdX += identity;

        // data from phase field element

        el[1]->CalcShape( ip, shape );
        el[1]->CalcDShape( ip, mDShape );
        Mult( mDShape, Tr.InverseJacobian(), mGShape );
        Eigen::Map<const Eigen::MatrixXd> eigenGShape( mGShape.Data(), dof_p, dim );
        Eigen::Map<const Eigen::VectorXd> eigenShape( shape.GetData(), dof_p );

        smallDeformMatrixB( dof_u, dim, gShape, mB );

        double pVal = p.dot( eigenShape );

        Eigen::MatrixXd pGrad = p.transpose() * eigenGShape;
        // std::cout<<pGrad<<std::endl<<std::endl;

        mMaterialModel->at( Tr, ip );
        mMaterialModel->setDeformationGradient( mdxdX );
        mMaterialModel->setPhaseField( pVal );

        // mMaterialModel->updateRefModuli();

        w = ip.weight * mMemo.GetDetdXdXi( i );
        eigenVec0 += w * ( mB.transpose() * mMaterialModel->getPK2StressVector() );
        double H = mMaterialModel->getPsiPos();
        UpdateH( i, H );

        eigenVec1 += w * ( -( 2 * ( 1 - k ) * H * ( 1 - pVal ) * eigenShape ) +
                           gc * ( l0 * eigenGShape * pGrad.transpose() + 1 / l0 * pVal * eigenShape ) );
    }
}

/// Assemble the local gradient matrix
void PhaseFieldIntegrator::AssembleElementGrad( const mfem::Array<const mfem::FiniteElement*>& el,
                                                mfem::ElementTransformation& Tr,
                                                const mfem::Array<const mfem::Vector*>& elfun,
                                                const mfem::Array2D<mfem::DenseMatrix*>& elmats )
{
    double w;
    int dof_u = el[0]->GetDof(), dim = el[0]->GetDim();
    int dof_p = el[1]->GetDof();

    mDShape.SetSize( dof_p, dim );
    mGShape.SetSize( dof_p, dim );
    shape.SetSize( dof_p );

    const double gc = mMaterialModel->getGc();
    const double k = mMaterialModel->getK();
    const double l0 = mMaterialModel->getL0();

    // mGeomStiff.resize( dof_u, dof_u );

    Eigen::Map<const Eigen::MatrixXd> u( elfun[0]->GetData(), dof_u, dim );
    Eigen::Map<const Eigen::VectorXd> p( elfun[1]->GetData(), dof_p );

    elmats( 0, 0 )->SetSize( dof_u * dim, dof_u * dim );
    elmats( 0, 1 )->SetSize( dof_u * dim, dof_p );
    elmats( 1, 0 )->SetSize( dof_p, dof_u * dim );
    elmats( 1, 1 )->SetSize( dof_p, dof_p );

    *elmats( 0, 0 ) = 0.0;
    *elmats( 0, 1 ) = 0.0;
    *elmats( 1, 0 ) = 0.0;
    *elmats( 1, 1 ) = 0.0;

    Eigen::Map<Eigen::MatrixXd> eigenMat00( elmats( 0, 0 )->Data(), dof_u * dim, dof_u * dim );

    Eigen::Map<Eigen::MatrixXd> eigenMat11( elmats( 1, 1 )->Data(), dof_p, dof_p );

    const mfem::IntegrationRule* ir = &( mfem::IntRules.Get( el[0]->GetGeomType(), 2 * el[0]->GetOrder() + 1 ) ); // <---

    const Eigen::Matrix3d identity = Eigen::Matrix3d::Identity();
    mMemo.InitializeElement( *el[0], Tr, *ir );
    for ( int i = 0; i < ir->GetNPoints(); i++ )
    {
        const mfem::IntegrationPoint& ip = ir->IntPoint( i );
        Tr.SetIntPoint( &ip );
        const Eigen::MatrixXd& gShape = mMemo.GetdNdX( i );
        mdxdX.setZero();
        mdxdX.block( 0, 0, dim, dim ) = u.transpose() * gShape;
        mdxdX += identity;

        // data from phase field element

        el[1]->CalcShape( ip, shape );
        el[1]->CalcDShape( ip, mDShape );
        Mult( mDShape, Tr.InverseJacobian(), mGShape );
        Eigen::Map<const Eigen::MatrixXd> eigenGShape( mGShape.Data(), dof_p, dim );
        Eigen::Map<const Eigen::VectorXd> eigenShape( shape.GetData(), dof_p );

        smallDeformMatrixB( dof_u, dim, gShape, mB );

        double pVal = p.dot( eigenShape );

        mMaterialModel->at( Tr, ip );
        mMaterialModel->setDeformationGradient( mdxdX );
        mMaterialModel->setPhaseField( pVal );

        mMaterialModel->updateRefModuli();

        w = ip.weight * mMemo.GetDetdXdXi( i );
        // if ( !onlyGeomStiff() )
        eigenMat00 += w * mB.transpose() * mMaterialModel->getRefModuli() * mB;

        double H = mMaterialModel->getPsiPos();
        UpdateH( i, H );

        eigenMat11 += w * ( gc * l0 * eigenGShape * eigenGShape.transpose() +
                            ( gc / l0 + 2 * ( 1 - k ) * H ) * eigenShape * eigenShape.transpose() );
    }
    // std::cout << eigenMat00 << std::endl << std::endl;
}

void BlockNonlinearDirichletPenaltyIntegrator::AssembleFaceVector( const mfem::Array<const mfem::FiniteElement*>& el1,
                                                                   const mfem::Array<const mfem::FiniteElement*>& el2,
                                                                   mfem::FaceElementTransformations& Tr,
                                                                   const mfem::Array<const mfem::Vector*>& elfun,
                                                                   const mfem::Array<mfem::Vector*>& elvec )
{
    int dof_u = el1[0]->GetDof(), dim = el1[0]->GetDim();
    int dof_p = el1[1]->GetDof();

    elvec[0]->SetSize( dof_u * dim );
    elvec[1]->SetSize( dof_p );

    *elvec[0] = 0.0;
    *elvec[1] = 0.0;

    mIntegrator.AssembleFaceVector( *el1[0], *el1[1], Tr, *elfun[0], *elvec[0] );
}

/// Assemble the local gradient matrix
void BlockNonlinearDirichletPenaltyIntegrator::AssembleFaceGrad( const mfem::Array<const mfem::FiniteElement*>& el1,
                                                                 const mfem::Array<const mfem::FiniteElement*>& el2,
                                                                 mfem::FaceElementTransformations& Tr,
                                                                 const mfem::Array<const mfem::Vector*>& elfun,
                                                                 const mfem::Array2D<mfem::DenseMatrix*>& elmats )
{
    int dof_u = el1[0]->GetDof(), dim = el1[0]->GetDim();
    int dof_p = el1[1]->GetDof();
    // mGeomStiff.resize( dof_u, dof_u );

    Eigen::Map<const Eigen::MatrixXd> u( elfun[0]->GetData(), dof_u, dim );
    Eigen::Map<const Eigen::VectorXd> p( elfun[1]->GetData(), dof_p );

    elmats( 0, 0 )->SetSize( dof_u * dim, dof_u * dim );
    elmats( 0, 1 )->SetSize( dof_u * dim, dof_p );
    elmats( 1, 0 )->SetSize( dof_p, dof_u * dim );
    elmats( 1, 1 )->SetSize( dof_p, dof_p );

    *elmats( 0, 0 ) = 0.0;
    *elmats( 0, 1 ) = 0.0;
    *elmats( 1, 0 ) = 0.0;
    *elmats( 1, 1 ) = 0.0;

    mIntegrator.AssembleFaceGrad( *el1[0], *el1[1], Tr, *elfun[0], *elmats( 0, 0 ) );
}
} // namespace plugin