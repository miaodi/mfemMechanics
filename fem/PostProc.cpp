#include "PostProc.h"

namespace plugin
{
ParaView2DVectorCoefficient::ParaView2DVectorCoefficient( const mfem::GridFunction& field )
    : mfem::VectorCoefficient( 3 ), mField( field ), mPlanarValue( 2 )
{
    MFEM_VERIFY( field.FESpace() != nullptr && field.VectorDim() == 2,
                 "ParaView planar-vector output requires a two-component GridFunction." );
}

void ParaView2DVectorCoefficient::Eval( mfem::Vector& value, mfem::ElementTransformation& transformation, const mfem::IntegrationPoint& integrationPoint )
{
    mField.GetVectorValue( transformation, integrationPoint, mPlanarValue );
    MFEM_ASSERT( mPlanarValue.Size() == 2, "The planar GridFunction returned an unexpected vector dimension." );
    value.SetSize( 3 );
    value( 0 ) = mPlanarValue( 0 );
    value( 1 ) = mPlanarValue( 1 );
    value( 2 ) = 0.;
}

StressCoefficient::StressCoefficient( int d, ElasticMaterial& mat )
    : mfem::VectorCoefficient( 7 ), u( NULL ), materialModel( &mat ), dim( d )
{
    grad.SetSize( dim );
}

void StressCoefficient::Eval( mfem::Vector& V, mfem::ElementTransformation& T, const mfem::IntegrationPoint& ip )
{
    MFEM_ASSERT( u != NULL, "displacement field is not set" );
    u->GetVectorGradient( T, grad );

    Eigen::Map<Eigen::MatrixXr> dudX( grad.Data(), dim, dim );
    F.setZero();
    F.block( 0, 0, dim, dim ) = dudX;
    F += Eigen::Matrix3r::Identity();

    materialModel->at( T, ip );
    materialModel->setLoadFactor( loadFactor );
    materialModel->setDeformationGradient( F );
    if ( !stressFreeDeformations.Empty() )
    {
        if ( materialModel->isSmallDeformation() )
        {
            MFEM_VERIFY( materialModel->SupportsMechanicalStrainInput(),
                         "Small-strain stress-free deformations require a strain-based material response." );
            mechanicalStrain = materialModel->getGreenLagrangeStrainTensor();
            mechanicalStrain -= stressFreeDeformations.EvalSmallStrain( T, ip, loadFactor );
            materialModel->setMechanicalStrain( mechanicalStrain );
        }
        else
        {
            stressFreeF = stressFreeDeformations.EvalDeformationGradient( T, ip, loadFactor );
            if ( dim < 3 )
            {
                MFEM_VERIFY(
                    stressFreeF.topRightCorner( dim, 3 - dim ).isZero() && stressFreeF.bottomLeftCorner( 3 - dim, dim ).isZero(),
                    "A reduced-dimensional stress-free deformation cannot couple active and out-of-plane directions." );
            }
            elasticF.noalias() = F * stressFreeF.inverse();
            materialModel->setDeformationGradient( elasticF );
        }
    }
    materialModel->updateRefModuli();
    auto vector = materialModel->getCauchyStressVector();

    V( 0 ) = vector( 0 );
    V( 1 ) = vector( 1 );
    V( 2 ) = vector( 2 );
    V( 3 ) = vector( 3 );
    V( 4 ) = vector( 4 );
    V( 5 ) = vector( 5 );
    V( 6 ) = std::sqrt( 1. / 2 *
                        ( std::pow( V( 0 ) - V( 1 ), 2 ) + std::pow( V( 1 ) - V( 2 ), 2 ) + std::pow( V( 2 ) - V( 0 ), 2 ) +
                          6 * ( std::pow( V( 3 ), 2 ), std::pow( V( 4 ), 2 ), std::pow( V( 5 ), 2 ) ) ) ); // Von mises srtess
}
} // namespace plugin
