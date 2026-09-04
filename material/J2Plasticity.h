#pragma once

#include "MaterialPointState.h"
#include "SolidMaterial.h"
#include "typeDef.h"

#include <Eigen/Dense>
#include <mfem.hpp>

class J2PlasticityMaterial;

namespace plugin
{
struct LinearIsotropicHardening
{
    mfem::real_t InitialYieldStress{ 0. };
    mfem::real_t HardeningModulus{ 0. };
};

struct LinearKinematicHardening
{
    /// Prager law: beta_dot = (2/3) Modulus * plastic_strain_dot.
    mfem::real_t Modulus{ 0. };
};

struct J2PlasticityParameters
{
    mfem::real_t YoungsModulus{ 0. };
    mfem::real_t PoissonRatio{ 0. };
    LinearIsotropicHardening Hardening;
    LinearKinematicHardening KinematicHardening;
};

struct J2PlasticityState
{
    Eigen::Matrix3r PlasticStrain{ Eigen::Matrix3r::Zero() };
    Eigen::Matrix3r BackStress{ Eigen::Matrix3r::Zero() };
    mfem::real_t EquivalentPlasticStrain{ 0. };
};

class J2PlasticityHistory
{
public:
    const J2PlasticityState& CommittedState() const noexcept;
    const J2PlasticityState& TrialState() const noexcept;

    void SetTrialState( const J2PlasticityState& state );
    void BeginStep() noexcept;
    void CommitStep() noexcept;
    void RollbackStep() noexcept;

private:
    J2PlasticityState mCommitted;
    J2PlasticityState mTrial;
};

enum class J2PlasticityBranch
{
    Elastic,
    Plastic
};

struct J2PlasticityResponse : SolidMaterialResponse
{
    J2PlasticityState TrialState;
    mfem::real_t PlasticIncrement{ 0. };
    J2PlasticityBranch Branch{ J2PlasticityBranch::Elastic };
};

J2PlasticityResponse EvaluateJ2Plasticity( const Eigen::Matrix3r& mechanicalStrain,
                                           const J2PlasticityState& committedState,
                                           const J2PlasticityParameters& parameters );

template <>
struct MaterialPointTraits<J2PlasticityMaterial>
{
    using State = J2PlasticityHistory;
};

} // namespace plugin

/// Spatially varying parameters for infinitesimal, rate-independent J2 plasticity.
class J2PlasticityMaterial
{
public:
    static constexpr plugin::SolidKinematics Kinematics = plugin::SolidKinematics::SmallStrain;

    J2PlasticityMaterial( mfem::Coefficient& youngsModulus,
                          mfem::Coefficient& poissonRatio,
                          mfem::Coefficient& initialYieldStress,
                          mfem::Coefficient& hardeningModulus );
    J2PlasticityMaterial( mfem::Coefficient& youngsModulus,
                          mfem::Coefficient& poissonRatio,
                          mfem::Coefficient& initialYieldStress,
                          mfem::Coefficient& hardeningModulus,
                          mfem::Coefficient& kinematicHardeningModulus );

    plugin::J2PlasticityResponse Evaluate( const plugin::SmallStrainMaterialPoint& materialPoint,
                                           const plugin::J2PlasticityState& committedState ) const;

private:
    mfem::Coefficient& mYoungsModulus;
    mfem::Coefficient& mPoissonRatio;
    mfem::Coefficient& mInitialYieldStress;
    mfem::Coefficient& mHardeningModulus;
    mfem::Coefficient* mKinematicHardeningModulus{ nullptr };
};
