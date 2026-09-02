#include "StressFreeDeformation.h"

#include <cmath>

namespace plugin
{
Eigen::Matrix3r StressFreeDeformationModel::EvalSmallStrain( mfem::ElementTransformation& transformation,
                                                             const mfem::IntegrationPoint& integrationPoint,
                                                             const mfem::real_t loadFactor )
{
    Eigen::Matrix3r result = Eigen::Matrix3r::Zero();
    for ( auto contribution : mContributions )
    {
        const Eigen::Matrix3r value = contribution.get().EvalSmallStrain( transformation, integrationPoint, loadFactor );
        MFEM_VERIFY( value.allFinite(), "A stress-free strain contains a non-finite value." );
        MFEM_VERIFY( value.isApprox( value.transpose() ), "A stress-free strain must be symmetric." );
        result += value;
    }
    return result;
}

Eigen::Matrix3r StressFreeDeformationModel::EvalDeformationGradient( mfem::ElementTransformation& transformation,
                                                                     const mfem::IntegrationPoint& integrationPoint,
                                                                     const mfem::real_t loadFactor )
{
    Eigen::Matrix3r result = Eigen::Matrix3r::Identity();
    for ( auto contribution : mContributions )
    {
        const Eigen::Matrix3r value = contribution.get().EvalDeformationGradient( transformation, integrationPoint, loadFactor );
        MFEM_VERIFY( value.allFinite(), "A stress-free deformation contains a non-finite value." );
        MFEM_VERIFY( value.determinant() > 0., "A stress-free deformation must preserve orientation." );
        result = ( result * value ).eval();
    }
    return result;
}

mfem::real_t IsotropicThermalExpansion::EvalThermalLogStrain( mfem::ElementTransformation& transformation,
                                                              const mfem::IntegrationPoint& integrationPoint,
                                                              const mfem::real_t loadFactor )
{
    const mfem::real_t expansion = mThermalExpansion.Eval( transformation, integrationPoint );
    const mfem::real_t temperature = mTemperature.Eval( transformation, integrationPoint );
    const mfem::real_t referenceTemperature = mReferenceTemperature.Eval( transformation, integrationPoint );

    MFEM_VERIFY( std::isfinite( expansion ) && std::isfinite( temperature ) && std::isfinite( referenceTemperature ) &&
                     std::isfinite( loadFactor ),
                 "Thermal-expansion inputs must be finite." );

    return expansion * ( temperature - referenceTemperature ) * loadFactor;
}

Eigen::Matrix3r IsotropicThermalExpansion::EvalSmallStrain( mfem::ElementTransformation& transformation,
                                                            const mfem::IntegrationPoint& integrationPoint,
                                                            const mfem::real_t loadFactor )
{
    return EvalThermalLogStrain( transformation, integrationPoint, loadFactor ) * Eigen::Matrix3r::Identity();
}

Eigen::Matrix3r IsotropicThermalExpansion::EvalDeformationGradient( mfem::ElementTransformation& transformation,
                                                                    const mfem::IntegrationPoint& integrationPoint,
                                                                    const mfem::real_t loadFactor )
{
    const mfem::real_t thermalStretch = std::exp( EvalThermalLogStrain( transformation, integrationPoint, loadFactor ) );
    MFEM_VERIFY( std::isfinite( thermalStretch ) && thermalStretch > 0.,
                 "The isotropic thermal stretch must be finite and positive." );
    return thermalStretch * Eigen::Matrix3r::Identity();
}
} // namespace plugin
