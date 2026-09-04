#include "J2Plasticity.h"

#include "util.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace
{
const Eigen::Matrix6r kDeviatoricProjector = []
{
    Eigen::Matrix6r projector = Eigen::Matrix6r::Zero();
    projector.topLeftCorner<3, 3>().setConstant( -1. / 3. );
    projector( 0, 0 ) = 2. / 3.;
    projector( 1, 1 ) = 2. / 3.;
    projector( 2, 2 ) = 2. / 3.;
    projector( 3, 3 ) = .5;
    projector( 4, 4 ) = .5;
    projector( 5, 5 ) = .5;
    return projector;
}();

const Eigen::Matrix6r kVolumetricProjector = []
{
    Eigen::Vector6r direction;
    direction << 1., 1., 1., 0., 0., 0.;
    return Eigen::Matrix6r( direction * direction.transpose() );
}();

const Eigen::Matrix3r kIdentity = Eigen::Matrix3r::Identity();

struct HardeningEvaluation
{
    mfem::real_t YieldStress{ 0. };
    mfem::real_t Slope{ 0. };
};

HardeningEvaluation EvaluateHardening( const plugin::LinearIsotropicHardening& hardening,
                                       const mfem::real_t equivalentPlasticStrain )
{
    MFEM_VERIFY( std::isfinite( hardening.InitialYieldStress ) && hardening.InitialYieldStress >= 0.,
                 "J2 initial yield stress must be finite and nonnegative." );
    MFEM_VERIFY( std::isfinite( hardening.HardeningModulus ) && hardening.HardeningModulus >= 0.,
                 "J2 isotropic hardening modulus must be finite and nonnegative." );
    MFEM_VERIFY( std::isfinite( equivalentPlasticStrain ) && equivalentPlasticStrain >= 0.,
                 "J2 equivalent plastic strain must be finite and nonnegative." );

    const mfem::real_t yieldStress =
        hardening.InitialYieldStress + hardening.HardeningModulus * equivalentPlasticStrain;
    MFEM_VERIFY( std::isfinite( yieldStress ), "J2 hardening law produced a nonfinite yield stress." );
    return { yieldStress, hardening.HardeningModulus };
}

Eigen::Matrix6r ElasticTangent( const mfem::real_t bulkModulus, const mfem::real_t shearModulus )
{
    return bulkModulus * kVolumetricProjector + 2. * shearModulus * kDeviatoricProjector;
}

mfem::real_t J2EquivalentStress( const Eigen::Matrix3r& stress )
{
    MFEM_VERIFY( stress.allFinite(), "J2 equivalent stress requires a finite stress tensor." );
    const Eigen::Matrix3r deviatoricStress = stress - stress.trace() / 3. * kIdentity;
    const mfem::real_t equivalentStressSquared = 1.5 * deviatoricStress.squaredNorm();
    MFEM_VERIFY( std::isfinite( equivalentStressSquared ), "J2 equivalent stress overflowed." );
    return std::sqrt( std::max( equivalentStressSquared, mfem::real_t{ 0. } ) );
}

void VerifyParameters( const plugin::J2PlasticityParameters& parameters )
{
    MFEM_VERIFY( std::isfinite( parameters.YoungsModulus ) && parameters.YoungsModulus > 0.,
                 "J2 Young's modulus must be finite and positive." );
    MFEM_VERIFY( std::isfinite( parameters.PoissonRatio ) && parameters.PoissonRatio > -1. &&
                     parameters.PoissonRatio < .5,
                 "J2 Poisson ratio must be finite and lie in (-1, 0.5)." );
    EvaluateHardening( parameters.Hardening, 0. );
    MFEM_VERIFY( std::isfinite( parameters.KinematicHardening.Modulus ) && parameters.KinematicHardening.Modulus >= 0.,
                 "J2 kinematic hardening modulus must be finite and nonnegative." );
}

void VerifyState( const plugin::J2PlasticityState& state )
{
    MFEM_VERIFY( state.PlasticStrain.allFinite() && state.BackStress.allFinite() &&
                     std::isfinite( state.EquivalentPlasticStrain ),
                 "J2 plastic state must be finite." );
    MFEM_VERIFY( state.EquivalentPlasticStrain >= 0., "J2 equivalent plastic strain must be nonnegative." );

    const mfem::real_t plasticStrainTolerance =
        64. * std::numeric_limits<mfem::real_t>::epsilon() * ( 1. + state.PlasticStrain.norm() );
    MFEM_VERIFY( ( state.PlasticStrain - state.PlasticStrain.transpose() ).norm() <= plasticStrainTolerance,
                 "J2 plastic strain must be symmetric." );
    MFEM_VERIFY( std::abs( state.PlasticStrain.trace() ) <= plasticStrainTolerance,
                 "J2 plastic strain must be deviatoric." );

    const mfem::real_t backStressTolerance =
        64. * std::numeric_limits<mfem::real_t>::epsilon() * ( 1. + state.BackStress.norm() );
    MFEM_VERIFY( ( state.BackStress - state.BackStress.transpose() ).norm() <= backStressTolerance,
                 "J2 backstress must be symmetric." );
    MFEM_VERIFY( std::abs( state.BackStress.trace() ) <= backStressTolerance, "J2 backstress must be deviatoric." );
}
} // namespace

namespace plugin
{
const J2PlasticityState& J2PlasticityHistory::CommittedState() const noexcept
{
    return mCommitted;
}

const J2PlasticityState& J2PlasticityHistory::TrialState() const noexcept
{
    return mTrial;
}

void J2PlasticityHistory::SetTrialState( const J2PlasticityState& state )
{
    VerifyState( state );
    mTrial = state;
}

void J2PlasticityHistory::BeginStep() noexcept
{
    mTrial = mCommitted;
}

void J2PlasticityHistory::CommitStep() noexcept
{
    mCommitted = mTrial;
}

void J2PlasticityHistory::RollbackStep() noexcept
{
    mTrial = mCommitted;
}

J2PlasticityResponse EvaluateJ2Plasticity( const Eigen::Matrix3r& mechanicalStrain,
                                           const J2PlasticityState& committedState,
                                           const J2PlasticityParameters& parameters )
{
    MFEM_VERIFY( mechanicalStrain.allFinite(), "J2 mechanical strain must be finite." );
    const mfem::real_t symmetryTolerance =
        64. * std::numeric_limits<mfem::real_t>::epsilon() * ( 1. + mechanicalStrain.norm() );
    MFEM_VERIFY( ( mechanicalStrain - mechanicalStrain.transpose() ).norm() <= symmetryTolerance,
                 "J2 mechanical strain must be symmetric." );
    VerifyParameters( parameters );
    VerifyState( committedState );

    const mfem::real_t shearModulus = parameters.YoungsModulus / ( 2. * ( 1. + parameters.PoissonRatio ) );
    const mfem::real_t bulkModulus = parameters.YoungsModulus / ( 3. * ( 1. - 2. * parameters.PoissonRatio ) );
    MFEM_VERIFY( std::isfinite( shearModulus ) && std::isfinite( bulkModulus ), "J2 elastic moduli must be finite." );
    const Eigen::Matrix3r trialElasticStrain = mechanicalStrain - committedState.PlasticStrain;
    const mfem::real_t pressure = bulkModulus * trialElasticStrain.trace();
    const Eigen::Matrix3r trialDeviatoricStress =
        2. * shearModulus * ( trialElasticStrain - trialElasticStrain.trace() / 3. * kIdentity );
    const Eigen::Matrix3r relativeTrialStress = trialDeviatoricStress - committedState.BackStress;
    const mfem::real_t trialEquivalentStress = J2EquivalentStress( relativeTrialStress );
    const auto hardening = EvaluateHardening( parameters.Hardening, committedState.EquivalentPlasticStrain );
    const mfem::real_t yieldFunction = trialEquivalentStress - hardening.YieldStress;
    const mfem::real_t yieldScale =
        std::max( { mfem::real_t{ 1. }, trialEquivalentStress, hardening.YieldStress } );
    const mfem::real_t yieldTolerance = 64. * std::numeric_limits<mfem::real_t>::epsilon() * yieldScale;

    J2PlasticityResponse response;
    response.TrialState = committedState;
    response.Stress = pressure * kIdentity + trialDeviatoricStress;
    response.ConsistentTangent = ElasticTangent( bulkModulus, shearModulus );
    MFEM_VERIFY( response.Stress.allFinite() && response.ConsistentTangent.allFinite(),
                 "J2 elastic predictor produced a nonfinite response." );
    if ( yieldFunction <= yieldTolerance )
    {
        return response;
    }

    MFEM_VERIFY( trialEquivalentStress > 0., "A plastic J2 return requires nonzero trial deviatoric stress." );
    const mfem::real_t denominator = 3. * shearModulus + hardening.Slope + parameters.KinematicHardening.Modulus;
    MFEM_VERIFY( std::isfinite( denominator ) && denominator > 0.,
                 "J2 consistency denominator must be finite and positive." );

    // Backward-Euler consistency is scalar and closed-form for combined linear hardening.
    response.PlasticIncrement = yieldFunction / denominator;
    const Eigen::Matrix3r flowDirection = 1.5 * relativeTrialStress / trialEquivalentStress;
    response.TrialState.PlasticStrain = committedState.PlasticStrain + response.PlasticIncrement * flowDirection;
    response.TrialState.BackStress = committedState.BackStress + ( 2. / 3. ) * parameters.KinematicHardening.Modulus *
                                                                     response.PlasticIncrement * flowDirection;
    response.TrialState.EquivalentPlasticStrain = committedState.EquivalentPlasticStrain + response.PlasticIncrement;

    const mfem::real_t stressRadialScale = 1. - 3. * shearModulus * response.PlasticIncrement / trialEquivalentStress;
    response.Stress =
        pressure * kIdentity + trialDeviatoricStress - 2. * shearModulus * response.PlasticIncrement * flowDirection;

    const Eigen::Matrix3r unitTrialDirection = relativeTrialStress / relativeTrialStress.norm();
    const Eigen::Vector6r directionVoigt = util::Voigt<mfem::real_t, mfem::real_t>( unitTrialDirection, false );
    Eigen::Matrix6r deviatoricPart = ElasticTangent( 0., shearModulus );
    deviatoricPart *= stressRadialScale;
    const mfem::real_t directionalCoefficient =
        6. * shearModulus * shearModulus * ( 1. / denominator - response.PlasticIncrement / trialEquivalentStress );
    // This is the derivative of the discrete radial return in engineering-Voigt form.
    response.ConsistentTangent = bulkModulus * kVolumetricProjector + deviatoricPart -
                                 directionalCoefficient * directionVoigt * directionVoigt.transpose();
    response.Branch = J2PlasticityBranch::Plastic;

    VerifyState( response.TrialState );
    MFEM_VERIFY( response.Stress.allFinite() && response.ConsistentTangent.allFinite(),
                 "J2 return mapping produced a nonfinite response." );
    return response;
}
} // namespace plugin

J2PlasticityMaterial::J2PlasticityMaterial( mfem::Coefficient& youngsModulus,
                                            mfem::Coefficient& poissonRatio,
                                            mfem::Coefficient& initialYieldStress,
                                            mfem::Coefficient& hardeningModulus )
    : mYoungsModulus( youngsModulus ),
      mPoissonRatio( poissonRatio ),
      mInitialYieldStress( initialYieldStress ),
      mHardeningModulus( hardeningModulus )
{
}

J2PlasticityMaterial::J2PlasticityMaterial( mfem::Coefficient& youngsModulus,
                                            mfem::Coefficient& poissonRatio,
                                            mfem::Coefficient& initialYieldStress,
                                            mfem::Coefficient& hardeningModulus,
                                            mfem::Coefficient& kinematicHardeningModulus )
    : J2PlasticityMaterial( youngsModulus, poissonRatio, initialYieldStress, hardeningModulus )
{
    mKinematicHardeningModulus = &kinematicHardeningModulus;
}

plugin::J2PlasticityResponse J2PlasticityMaterial::Evaluate( const plugin::SmallStrainMaterialPoint& materialPoint,
                                                             const plugin::J2PlasticityState& committedState ) const
{
    auto& transformation = materialPoint.Context.Transformation;
    const auto& integrationPoint = materialPoint.Context.IntegrationPoint;
    plugin::J2PlasticityParameters parameters;
    parameters.YoungsModulus = mYoungsModulus.Eval( transformation, integrationPoint );
    parameters.PoissonRatio = mPoissonRatio.Eval( transformation, integrationPoint );
    parameters.Hardening.InitialYieldStress = mInitialYieldStress.Eval( transformation, integrationPoint );
    parameters.Hardening.HardeningModulus = mHardeningModulus.Eval( transformation, integrationPoint );
    if ( mKinematicHardeningModulus != nullptr )
    {
        parameters.KinematicHardening.Modulus =
            mKinematicHardeningModulus->Eval( transformation, integrationPoint );
    }
    return plugin::EvaluateJ2Plasticity( materialPoint.MechanicalStrain, committedState, parameters );
}
