#include "PostProc.h"

namespace plugin
{
StressCoefficient::StressCoefficient( int d, ElasticMaterial& mat )
    : mfem::VectorCoefficient( 7 ), u( NULL ), materialModel( &mat ), dim( d )
{
    grad.SetSize( dim );
}

void StressCoefficient::Eval( mfem::Vector& V, mfem::ElementTransformation& T, const mfem::IntegrationPoint& ip )
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
    auto vector = materialModel->getCauchyStressVector();

    V( 0 ) = vector( 0 );
    V( 1 ) = vector( 1 );
    V( 2 ) = vector( 2 );
    V( 3 ) = vector( 3 );
    V( 4 ) = vector( 4 );
    V( 5 ) = vector( 5 );
    V( 6 ) = std::sqrt( 3. / 2 * vector.dot( vector ) );
}
} // namespace plugin