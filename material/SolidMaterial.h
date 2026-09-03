#pragma once

#include "typeDef.h"

#include <Eigen/Dense>
#include <mfem.hpp>

namespace plugin
{
enum class SolidKinematics
{
    SmallStrain,
    FiniteStrain
};

struct MaterialPointContext
{
    mfem::ElementTransformation& Transformation;
    const mfem::IntegrationPoint& IntegrationPoint;
    mfem::real_t LoadFactor;
};

struct SmallStrainMaterialPoint
{
    const Eigen::Matrix3r& MechanicalStrain;
    MaterialPointContext Context;
};

struct FiniteStrainMaterialPoint
{
    const Eigen::Matrix3r& DeformationGradient;
    MaterialPointContext Context;
};

/// Small strain uses Cauchy stress and d(sigma)/d(epsilon); finite strain uses
/// second Piola--Kirchhoff stress and d(S)/d(E), both in repository Voigt form.
struct SolidMaterialResponse
{
    Eigen::Matrix3r Stress{ Eigen::Matrix3r::Zero() };
    Eigen::Matrix6r ConsistentTangent{ Eigen::Matrix6r::Zero() };
};
} // namespace plugin
