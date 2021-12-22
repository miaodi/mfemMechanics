#include "PostProc.h"

namespace plugin
{
StressCoefficient::StressCoefficient( int d, ElasticMaterial& mat )
    : mfem::Coefficient(), u( NULL ), materialModel( &mat ), si( 0 ), sj( 0 ), dim( d )
{
    grad.SetSize( dim );
}

double StressCoefficient::Eval( mfem::ElementTransformation& T, const mfem::IntegrationPoint& ip )
{
    MFEM_ASSERT( u != NULL, "displacement field is not set" );
    u->GetVectorGradient( T, grad );

    Eigen::Map<Eigen::MatrixXd> dudX( grad.Data(), dim, dim );
    F.setZero();
    F.block( 0, 0, dim, dim ) = dudX;
    F += Eigen::Matrix3d::Identity();

    materialModel->at( T, ip );
    materialModel->setDeformationGradient( F );
    materialModel->updateRefModuli();
    auto tensor = materialModel->getCauchyStressTensor();
    return tensor( si, sj );
}
} // namespace plugin