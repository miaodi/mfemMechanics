#include "PhaseFieldMaterial.h"
#include "util.h"
#include <SymmetricEigensolver3x3.hpp>

PhaseFieldElasticMaterial::PhaseFieldElasticMaterial( mfem::Coefficient& E, mfem::Coefficient& nu, StrainEnergyType set )
    : ElasticMaterial(), mE( &E ), mNu( &nu ), mSET( set )
{
    setLargeDeformation( false );
    mStrainEnergyFunc = StrainEnergyFactory( mSET );
    mParams.resize( 10 );
}

std::function<autodiff::dual2nd( const autodiff::Vector6dual2nd&, const Eigen::VectorXd& )> PhaseFieldElasticMaterial::StrainEnergyFactory(
    const PhaseFieldElasticMaterial::StrainEnergyType set ) const
{
    switch ( set )
    {
    case PhaseFieldElasticMaterial::StrainEnergyType::Amor:
    {
        // zhou2018phase
        return [this]( const autodiff::Vector6dual2nd& strainVec, const Eigen::VectorXd& params ) {
            using T = typename std::decay_t<decltype( strainVec( 0 ) )>;

            auto curlyBracPos = []( const T& val ) { return val > 0 ? val : static_cast<T>( 0 ); };
            auto curlyBracNeg = []( const T& val ) { return val < 0 ? val : static_cast<T>( 0 ); };

            const auto strainTensor = util::InverseVoigt( strainVec, true );
            const double phi = params[1];
            const double Nu = this->Nu();
            const double E = this->E();

            const double mu = E / ( 2. * ( 1. + Nu ) );
            const double lambda = ( Nu * E ) / ( ( 1 + Nu ) * ( 1. - 2. * Nu ) );

            auto [strainPos, strainNeg] = util::StrainSplit( strainTensor );
            const T psiPos = mLambda / 2 * autodiff::detail::pow( curlyBracPos( strainTensor.trace() ), 2 ) +
                             mu * strainPos.squaredNorm();
            if ( params[0] == 0 )
                return psiPos;
            const T psiNeg = lambda / 2 * autodiff::detail::pow( curlyBracNeg( strainTensor.trace() ), 2 ) +
                             mu * strainNeg.squaredNorm();
            if ( params[0] == 1 )
                return psiNeg;
            const T res = ( ( 1 - mK ) * std::pow( 1 - phi, 2 ) + mK ) * psiPos + psiNeg;
            if ( params[0] == 2 )
                return res;
            return T( 0. );
        };
    }
    case StrainEnergyType::IsotropicLinearElastic:
    {
        return [this]( const autodiff::Vector6dual2nd& strainVec, const Eigen::VectorXd& params ) {
            const auto strainTensor = util::InverseVoigt( strainVec, true );
            const double Nu = this->Nu();
            const double E = this->E();

            const double mu = E / ( 2. * ( 1. + Nu ) );
            const double lambda = ( Nu * E ) / ( ( 1 + Nu ) * ( 1. - 2. * Nu ) );

            return lambda / 2 * strainTensor.trace() * strainTensor.trace() + mu * strainTensor.squaredNorm();
        };
    }
    default:
        return [this]( const autodiff::Vector6dual2nd& strainVec, const Eigen::VectorXd& params ) {
            return autodiff::dual2nd( 0 );
        };
    }
}

void PhaseFieldElasticMaterial::updateRefModuli()
{
    getGreenLagrangeStrainVector( mStrainVecDual );
    autodiff::dual2nd u;
    autodiff::VectorXdual g;

    mParams[0] = 2;
    mParams[1] = mPhi;

    mRefModuli = autodiff::hessian( mStrainEnergyFunc, autodiff::wrt( mStrainVecDual ),
                                    autodiff::at( mStrainVecDual, mParams ), u, g );
}

const Eigen::Vector6d& PhaseFieldElasticMaterial::getPK2StressVector() const
{
    getGreenLagrangeStrainVector( mStrainVecDual );
    autodiff::dual2nd u;

    mParams[0] = 2;
    mParams[1] = mPhi;

    mStressVec =
        autodiff::gradient( mStrainEnergyFunc, autodiff::wrt( mStrainVecDual ), autodiff::at( mStrainVecDual, mParams ), u );
    return mStressVec;
}

double PhaseFieldElasticMaterial::getPsiPos() const
{
    getGreenLagrangeStrainVector( mStrainVecDual );

    mParams[0] = 0;
    mParams[1] = mPhi;

    return autodiff::detail::val( mStrainEnergyFunc( mStrainVecDual, mParams ) );
}