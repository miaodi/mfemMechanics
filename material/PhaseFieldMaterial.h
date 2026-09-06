#pragma once

#include "Material.h"
#include <Eigen/Dense>
#include <optional>

struct PhaseFieldFractureParameters
{
    mfem::real_t criticalEnergyReleaseRate{ 2700. };
    mfem::real_t lengthScale{ 0.015e-3 };
    mfem::real_t residualStiffness{ 1e-9 };
};

class PhaseFieldElasticMaterial : public ElasticMaterial
{
public:
    enum class StrainEnergySplit
    {
        MieheSpectral,
        AmorVolumetricDeviatoric,
        Isotropic
    };

    PhaseFieldElasticMaterial( mfem::Coefficient& E,
                               mfem::Coefficient& nu,
                               StrainEnergySplit split = StrainEnergySplit::MieheSpectral,
                               PhaseFieldFractureParameters parameters = {} );
    PhaseFieldElasticMaterial( const PhaseFieldElasticMaterial& ) = delete;
    PhaseFieldElasticMaterial& operator=( const PhaseFieldElasticMaterial& ) = delete;
    PhaseFieldElasticMaterial( PhaseFieldElasticMaterial&& ) = delete;
    PhaseFieldElasticMaterial& operator=( PhaseFieldElasticMaterial&& ) = delete;

    virtual void updateRefModuli() override;

    mfem::real_t E() const
    {
        MFEM_ASSERT( mEleTrans && mIntgP, "ElementTransformation or IntegrationPoint is not set" );
        return mE->Eval( *mEleTrans, *mIntgP );
    }

    mfem::real_t Nu() const
    {
        MFEM_ASSERT( mEleTrans && mIntgP, "ElementTransformation or IntegrationPoint is not set" );
        return mNu->Eval( *mEleTrans, *mIntgP );
    }

    virtual const Eigen::Vector6r& getPK2StressVector() const override;

    void setPhaseField( mfem::real_t value );

    mfem::real_t getPsiPos() const;
    Eigen::Vector6r getPositiveStressVector() const;
    Eigen::Vector6r getPhaseStressDerivative() const;

    struct Response
    {
        mfem::real_t positiveEnergy;
        Eigen::Vector6r positiveStress;
        Eigen::Vector6r stress;
        Eigen::Vector6r phaseStressDerivative;
        // Engaged exactly when EvaluateResponse(true) is requested. Maps
        // engineering strain increments to unscaled Voigt stress increments.
        std::optional<Eigen::Matrix6r> tangent;
    };

    /// Owning snapshot of the current borrowed material point; no persistent cache.
    /// false skips all tangent work and returns a disengaged tangent.
    Response EvaluateResponse( bool computeTangent ) const;

    mfem::real_t getGc() const noexcept
    {
        return mParameters.criticalEnergyReleaseRate;
    }

    mfem::real_t getK() const noexcept
    {
        return mParameters.residualStiffness;
    }

    mfem::real_t getL0() const noexcept
    {
        return mParameters.lengthScale;
    }

    StrainEnergySplit GetStrainEnergySplit() const noexcept
    {
        return mStrainEnergySplit;
    }

    bool SupportsMechanicalStrainInput() const noexcept override
    {
        return true;
    }

private:
    struct ElasticConstants
    {
        mfem::real_t lambda;
        mfem::real_t mu;
        mfem::real_t bulkModulus;
    };

    static void EvaluateIsotropic( const Eigen::Matrix3r& strain, const ElasticConstants& constants, mfem::real_t degradation, Response& response );
    static void EvaluateAmor( const Eigen::Matrix3r& strain, const ElasticConstants& constants, mfem::real_t degradation, Response& response );
    static void EvaluateSpectral( const Eigen::Matrix3r& strain, const ElasticConstants& constants, mfem::real_t degradation, Response& response );
    ElasticConstants GetElasticConstants() const;
    mfem::real_t Degradation() const noexcept;
    mfem::real_t DegradationDerivative() const noexcept;

    mfem::Coefficient* mE{ nullptr };
    mfem::Coefficient* mNu{ nullptr };

    mfem::real_t mPhi{ 0. };
    StrainEnergySplit mStrainEnergySplit;
    PhaseFieldFractureParameters mParameters;
};

namespace plugin
{
/// Irreversible positive-energy history at one material point.
class PhaseFieldHistory
{
public:
    mfem::real_t EvaluateTrial( mfem::real_t positiveEnergy );
    void BeginStep() noexcept;
    void CommitStep() noexcept;
    void RollbackStep() noexcept;

    mfem::real_t CommittedValue() const noexcept
    {
        return mCommitted;
    }

    mfem::real_t TrialValue() const noexcept
    {
        return mTrial;
    }

private:
    mfem::real_t mCommitted{ 0. };
    mfem::real_t mTrial{ 0. };
};

template <>
struct MaterialPointTraits<PhaseFieldElasticMaterial>
{
    using State = PhaseFieldHistory;
};

using PhaseFieldIntegrationPointState = MaterialPointStateBundle<PhaseFieldElasticMaterial>;
} // namespace plugin
