#include "PhaseFieldMaterial.h"
#include "util.h"
#include <algorithm>
#include <cmath>

PhaseFieldElasticMaterial::PhaseFieldElasticMaterial( mfem::Coefficient& E,
                                                      mfem::Coefficient& nu,
                                                      const StrainEnergySplit split,
                                                      const PhaseFieldFractureParameters parameters )
    : ElasticMaterial(), mE( &E ), mNu( &nu ), mStrainEnergySplit( split ), mParameters( parameters )
{
    MFEM_VERIFY( std::isfinite( mParameters.criticalEnergyReleaseRate ) && mParameters.criticalEnergyReleaseRate > 0.,
                 "Phase-field critical energy release rate must be finite and positive." );
    MFEM_VERIFY( std::isfinite( mParameters.lengthScale ) && mParameters.lengthScale > 0.,
                 "Phase-field length scale must be finite and positive." );
    MFEM_VERIFY( std::isfinite( mParameters.residualStiffness ) && mParameters.residualStiffness >= 0. &&
                     mParameters.residualStiffness < 1.,
                 "Phase-field residual stiffness must be finite and in [0, 1)." );
    setLargeDeformation( false );
}

void PhaseFieldElasticMaterial::setPhaseField( const mfem::real_t value )
{
    MFEM_VERIFY( std::isfinite( value ), "Phase field must be finite." );
    mPhi = value;
}

PhaseFieldElasticMaterial::ElasticConstants PhaseFieldElasticMaterial::GetElasticConstants() const
{
    MFEM_VERIFY( isSmallDeformation(), "Phase-field elasticity supports only small-strain kinematics." );
    const mfem::real_t youngsModulus = E();
    const mfem::real_t poissonRatio = Nu();
    MFEM_VERIFY( std::isfinite( youngsModulus ) && youngsModulus > 0., "Young's modulus must be finite and positive." );
    MFEM_VERIFY( std::isfinite( poissonRatio ) && poissonRatio > -1. && poissonRatio < .5,
                 "Poisson's ratio must be finite and in (-1, 0.5)." );

    const mfem::real_t mu = youngsModulus / ( 2. * ( 1. + poissonRatio ) );
    const mfem::real_t lambda = poissonRatio * youngsModulus / ( ( 1. + poissonRatio ) * ( 1. - 2. * poissonRatio ) );
    return { lambda, mu, lambda + 2. * mu / 3. };
}

mfem::real_t PhaseFieldElasticMaterial::Degradation() const noexcept
{
    const mfem::real_t intactFraction = 1. - mPhi;
    return ( 1. - mParameters.residualStiffness ) * intactFraction * intactFraction + mParameters.residualStiffness;
}

mfem::real_t PhaseFieldElasticMaterial::DegradationDerivative() const noexcept
{
    return -2. * ( 1. - mParameters.residualStiffness ) * ( 1. - mPhi );
}

PhaseFieldElasticMaterial::Response PhaseFieldElasticMaterial::EvaluateResponse( const bool computeTangent ) const
{
    Eigen::Vector6r strainVector;
    getGreenLagrangeStrainVector( strainVector );
    MFEM_VERIFY( strainVector.allFinite(), "Phase-field strain must be finite." );
    const Eigen::Matrix3r strain = util::InverseVoigt( strainVector, true );
    const auto constants = GetElasticConstants();
    const mfem::real_t degradation = Degradation();
    Response response;
    if ( computeTangent )
    {
        response.tangent.emplace();
    }
    switch ( mStrainEnergySplit )
    {
    case StrainEnergySplit::Isotropic:
        EvaluateIsotropic( strain, constants, degradation, response );
        break;
    case StrainEnergySplit::AmorVolumetricDeviatoric:
        EvaluateAmor( strain, constants, degradation, response );
        break;
    case StrainEnergySplit::MieheSpectral:
        EvaluateSpectral( strain, constants, degradation, response );
        break;
    default:
        MFEM_ABORT( "Unsupported phase-field strain-energy split." );
    }
    response.phaseStressDerivative = DegradationDerivative() * response.positiveStress;
    return response;
}

void PhaseFieldElasticMaterial::EvaluateIsotropic( const Eigen::Matrix3r& strain,
                                                   const ElasticConstants& constants,
                                                   const mfem::real_t degradation,
                                                   Response& response )
{
    const auto [lambda, mu, bulkModulus] = constants;
    const mfem::real_t trace = strain.trace();
    response.positiveEnergy = lambda / 2. * trace * trace + mu * strain.squaredNorm();
    const Eigen::Matrix3r positiveStress = lambda * trace * Eigen::Matrix3r::Identity() + 2. * mu * strain;
    response.positiveStress = util::Voigt<mfem::real_t, mfem::real_t>( positiveStress, false );
    response.stress = degradation * response.positiveStress;
    if ( response.tangent )
    {
        for ( int column = 0; column < 6; column++ )
        {
            const Eigen::Vector6r basis = Eigen::Vector6r::Unit( column );
            const Eigen::Matrix3r increment = util::InverseVoigt( basis, true );
            const Eigen::Matrix3r stressIncrement =
                degradation * ( lambda * increment.trace() * Eigen::Matrix3r::Identity() + 2. * mu * increment );
            response.tangent->col( column ) = util::Voigt<mfem::real_t, mfem::real_t>( stressIncrement, false );
        }
    }
}

void PhaseFieldElasticMaterial::EvaluateAmor( const Eigen::Matrix3r& strain,
                                              const ElasticConstants& constants,
                                              const mfem::real_t degradation,
                                              Response& response )
{
    const auto [lambda, mu, bulkModulus] = constants;
    const mfem::real_t trace = strain.trace();
    const mfem::real_t positiveTrace = std::max( trace, mfem::real_t( 0. ) );
    const mfem::real_t negativeTrace = std::min( trace, mfem::real_t( 0. ) );
    const mfem::real_t traceSlope = trace > 0. ? 1. : ( trace < 0. ? 0. : .5 );
    const Eigen::Matrix3r deviatoricStrain = strain - trace / 3. * Eigen::Matrix3r::Identity();
    response.positiveEnergy = bulkModulus / 2. * positiveTrace * positiveTrace + mu * deviatoricStrain.squaredNorm();
    const Eigen::Matrix3r positiveStress = bulkModulus * positiveTrace * Eigen::Matrix3r::Identity() + 2. * mu * deviatoricStrain;
    const Eigen::Matrix3r negativeStress = bulkModulus * negativeTrace * Eigen::Matrix3r::Identity();
    response.positiveStress = util::Voigt<mfem::real_t, mfem::real_t>( positiveStress, false );
    const Eigen::Matrix3r stress = degradation * positiveStress + negativeStress;
    response.stress = util::Voigt<mfem::real_t, mfem::real_t>( stress, false );
    if ( response.tangent )
    {
        // Elementary energy derivatives, with engineering-Voigt basis increments.
        // Amor uses complementary centered slopes at zero trace: unlike the
        // former AD branches (both zero), these recover intact bulk stiffness.
        for ( int column = 0; column < 6; column++ )
        {
            const Eigen::Vector6r basis = Eigen::Vector6r::Unit( column );
            const Eigen::Matrix3r increment = util::InverseVoigt( basis, true );
            const Eigen::Matrix3r deviatoricIncrement = increment - increment.trace() / 3. * Eigen::Matrix3r::Identity();
            const Eigen::Matrix3r stressIncrement = bulkModulus * ( degradation * traceSlope + ( 1. - traceSlope ) ) *
                                                        increment.trace() * Eigen::Matrix3r::Identity() +
                                                    2. * mu * degradation * deviatoricIncrement;
            response.tangent->col( column ) = util::Voigt<mfem::real_t, mfem::real_t>( stressIncrement, false );
        }
    }
}

void PhaseFieldElasticMaterial::EvaluateSpectral( const Eigen::Matrix3r& strain,
                                                  const ElasticConstants& constants,
                                                  const mfem::real_t degradation,
                                                  Response& response )
{
    const auto [lambda, mu, bulkModulus] = constants;
    const mfem::real_t trace = strain.trace();
    const mfem::real_t positiveTrace = std::max( trace, mfem::real_t( 0. ) );
    const mfem::real_t negativeTrace = std::min( trace, mfem::real_t( 0. ) );
    const mfem::real_t traceSlope = trace > 0. ? 1. : ( trace < 0. ? 0. : .5 );
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3r> eigen( strain );
    MFEM_VERIFY( eigen.info() == Eigen::Success, "Phase-field strain eigendecomposition failed." );
    const Eigen::Vector3r values = eigen.eigenvalues();
    const Eigen::Matrix3r vectors = eigen.eigenvectors();
    const Eigen::Vector3r positiveValues = values.cwiseMax( 0. );
    const Eigen::Matrix3r positiveStrain = vectors * positiveValues.asDiagonal() * vectors.transpose();
    const Eigen::Matrix3r positiveStress = lambda * positiveTrace * Eigen::Matrix3r::Identity() + 2. * mu * positiveStrain;
    const Eigen::Matrix3r negativeStrain = vectors * values.cwiseMin( 0. ).asDiagonal() * vectors.transpose();
    const Eigen::Matrix3r stress =
        degradation * positiveStress + lambda * negativeTrace * Eigen::Matrix3r::Identity() + 2. * mu * negativeStrain;
    response.positiveEnergy = lambda / 2. * positiveTrace * positiveTrace + mu * positiveValues.squaredNorm();
    response.positiveStress = util::Voigt<mfem::real_t, mfem::real_t>( positiveStress, false );
    response.stress = util::Voigt<mfem::real_t, mfem::real_t>( stress, false );
    if ( !response.tangent )
    {
        return;
    }

    // Frechet derivative of max(strain, 0): divided differences in the
    // eigenbasis avoid differentiating nonunique eigenvectors. Equal-sign
    // pairs have exact slopes 0 or 1 even for repeated eigenvalues.
    // At zero choose the centered generalized slope 1/2. See docs/phase-field-fracture.md.
    Eigen::Matrix3r slopes;
    for ( int i = 0; i < 3; i++ )
    {
        for ( int j = 0; j < 3; j++ )
        {
            if ( values( i ) == 0. && values( j ) == 0. )
            {
                slopes( i, j ) = .5;
            }
            else if ( values( i ) >= 0. && values( j ) >= 0. )
            {
                slopes( i, j ) = 1.;
            }
            else if ( values( i ) <= 0. && values( j ) <= 0. )
            {
                slopes( i, j ) = 0.;
            }
            else
            {
                slopes( i, j ) = ( positiveValues( i ) - positiveValues( j ) ) / ( values( i ) - values( j ) );
            }
        }
    }
    // Direct weighting preserves tiny residual stiffness in pure tension.
    const Eigen::Matrix3r weightedSlopes = degradation * slopes + ( Eigen::Matrix3r::Ones() - slopes );
    for ( int column = 0; column < 6; column++ )
    {
        const Eigen::Vector6r basis = Eigen::Vector6r::Unit( column );
        const Eigen::Matrix3r increment = util::InverseVoigt( basis, true );
        const Eigen::Matrix3r localIncrement = vectors.transpose() * increment * vectors;
        const Eigen::Matrix3r stressIncrement =
            lambda * ( degradation * traceSlope + ( 1. - traceSlope ) ) * increment.trace() * Eigen::Matrix3r::Identity() +
            2. * mu * vectors * weightedSlopes.cwiseProduct( localIncrement ) * vectors.transpose();
        response.tangent->col( column ) = util::Voigt<mfem::real_t, mfem::real_t>( stressIncrement, false );
    }
}

void PhaseFieldElasticMaterial::updateRefModuli()
{
    mRefModuli = *EvaluateResponse( true ).tangent;
}

const Eigen::Vector6r& PhaseFieldElasticMaterial::getPK2StressVector() const
{
    mStressVec = EvaluateResponse( false ).stress;
    return mStressVec;
}

mfem::real_t PhaseFieldElasticMaterial::getPsiPos() const
{
    return EvaluateResponse( false ).positiveEnergy;
}

Eigen::Vector6r PhaseFieldElasticMaterial::getPositiveStressVector() const
{
    return EvaluateResponse( false ).positiveStress;
}

Eigen::Vector6r PhaseFieldElasticMaterial::getPhaseStressDerivative() const
{
    return EvaluateResponse( false ).phaseStressDerivative;
}

mfem::real_t plugin::PhaseFieldHistory::EvaluateTrial( const mfem::real_t positiveEnergy )
{
    MFEM_VERIFY( std::isfinite( positiveEnergy ), "Phase-field history requires a finite positive-strain energy." );
    // H_trial = max(H_committed, psi_plus(current trial)). Do not accumulate
    // maxima over Newton iterates that may later be rejected.
    mTrial = std::max( mCommitted, positiveEnergy );
    return mTrial;
}

void plugin::PhaseFieldHistory::BeginStep() noexcept
{
    mTrial = mCommitted;
}

void plugin::PhaseFieldHistory::CommitStep() noexcept
{
    mCommitted = mTrial;
}

void plugin::PhaseFieldHistory::RollbackStep() noexcept
{
    mTrial = mCommitted;
}
