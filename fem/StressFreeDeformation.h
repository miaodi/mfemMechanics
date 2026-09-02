#pragma once

#include "typeDef.h"
#include <functional>
#include <mfem.hpp>
#include <vector>

namespace plugin
{
/** A prescribed local mapping from the initial reference configuration to a
 * stress-free configuration.
 *
 * The field supplies an additive eigenstrain for small-deformation integrators
 * and a stress-free deformation gradient for finite-deformation integrators.
 * The latter use the multiplicative split F = F_e F_0 and evaluate the
 * constitutive law with F_e = F F_0^{-1}.
 *
 * @see L. Vujosevic and V. A. Lubarda, "Finite-strain thermoelasticity
 * based on multiplicative decomposition of deformation gradient,"
 * Theoretical and Applied Mechanics 28-29 (2002), equations 37, 41, 45, 50,
 * and 59, https://doi.org/10.2298/TAM0229379V.
 */
class StressFreeDeformation
{
public:
    virtual ~StressFreeDeformation() = default;

    virtual Eigen::Matrix3r EvalSmallStrain( mfem::ElementTransformation& transformation,
                                             const mfem::IntegrationPoint& integrationPoint,
                                             mfem::real_t loadFactor ) = 0;

    virtual Eigen::Matrix3r EvalDeformationGradient( mfem::ElementTransformation& transformation,
                                                     const mfem::IntegrationPoint& integrationPoint,
                                                     mfem::real_t loadFactor ) = 0;
};

/** Non-owning multiplicative composition of stress-free deformations.
 *
 * Contributions are multiplied in insertion order. They must outlive this
 * model.
 */
class StressFreeDeformationModel
{
public:
    void Add( StressFreeDeformation& deformation )
    {
        mContributions.emplace_back( deformation );
    }

    void Clear()
    {
        mContributions.clear();
    }

    bool Empty() const noexcept
    {
        return mContributions.empty();
    }

    Eigen::Matrix3r EvalSmallStrain( mfem::ElementTransformation& transformation,
                                     const mfem::IntegrationPoint& integrationPoint,
                                     mfem::real_t loadFactor );

    Eigen::Matrix3r EvalDeformationGradient( mfem::ElementTransformation& transformation,
                                             const mfem::IntegrationPoint& integrationPoint,
                                             mfem::real_t loadFactor );

private:
    std::vector<std::reference_wrapper<StressFreeDeformation>> mContributions;
};

/** Isotropic thermal expansion.
 *
 * Small strain uses epsilon_theta = alpha (T - T_ref) I. Finite strain uses
 * F_theta = exp(alpha (T - T_ref)) I, corresponding to a constant
 * instantaneous coefficient of thermal expansion. The temperature change is
 * ramped by loadFactor. Coefficients are referenced, not owned, and must
 * outlive this object.
 */
class IsotropicThermalExpansion final : public StressFreeDeformation
{
public:
    IsotropicThermalExpansion( mfem::Coefficient& thermalExpansion, mfem::Coefficient& temperature, mfem::Coefficient& referenceTemperature )
        : mThermalExpansion{ thermalExpansion }, mTemperature{ temperature }, mReferenceTemperature{ referenceTemperature }
    {
    }

    Eigen::Matrix3r EvalSmallStrain( mfem::ElementTransformation& transformation,
                                     const mfem::IntegrationPoint& integrationPoint,
                                     mfem::real_t loadFactor ) override;

    Eigen::Matrix3r EvalDeformationGradient( mfem::ElementTransformation& transformation,
                                             const mfem::IntegrationPoint& integrationPoint,
                                             mfem::real_t loadFactor ) override;

private:
    mfem::real_t EvalThermalLogStrain( mfem::ElementTransformation& transformation,
                                       const mfem::IntegrationPoint& integrationPoint,
                                       mfem::real_t loadFactor );

    mfem::Coefficient& mThermalExpansion;
    mfem::Coefficient& mTemperature;
    mfem::Coefficient& mReferenceTemperature;
};
} // namespace plugin
