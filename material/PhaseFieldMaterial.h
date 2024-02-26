#pragma once

#include "Material.h"
#include <Eigen/Dense>
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>
#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>
#include <functional>

class PhaseFieldElasticMaterial : public ElasticMaterial
{
public:
    enum class StrainEnergyType
    {
        Amor,
        Borden
    };
    PhaseFieldElasticMaterial( mfem::Coefficient& E, mfem::Coefficient& nu, StrainEnergyType set = StrainEnergyType::Amor );

    virtual void updateRefModuli() override;

    double E() const
    {
        MFEM_ASSERT( mEleTrans && mIntgP, "ElementTransformation or IntegrationPoint is not set" );
        return mE->Eval( *mEleTrans, *mIntgP );
    }

    double Nu() const
    {
        MFEM_ASSERT( mEleTrans && mIntgP, "ElementTransformation or IntegrationPoint is not set" );
        return mNu->Eval( *mEleTrans, *mIntgP );
    }

    virtual const Eigen::Vector6d& getPK2StressVector() const override;

protected:
    std::function<autodiff::dual2nd( const autodiff::Vector6dual2nd&, const Eigen::VectorXd& )> StrainEnergyFactory( const StrainEnergyType set ) const;

protected:
    mfem::Coefficient* mE{ nullptr };
    mfem::Coefficient* mNu{ nullptr };

    double mK{ 0. };    // prevent from 0 elasticity
    double mGc{ 1e-3 }; // Grifﬁth-type critical energy release rate
    StrainEnergyType mSET;

    std::function<autodiff::dual2nd( const autodiff::Vector6dual2nd&, const Eigen::VectorXd& )> mStrainEnergyFunc;

    // params[0]: select strain energy: 0 positive, 1 negative, 2 total
    // params[1]: phase field phi
    mutable Eigen::VectorXd mParams;

    // strain cache
    mutable autodiff::Vector6dual2nd mStrainVecDual;
};