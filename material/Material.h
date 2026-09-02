#pragma once
#include "mfem.hpp"
#include "typeDef.h"
#include "util.h"
#include <Eigen/Dense>

class ElasticMaterial
{
public:
    ElasticMaterial();

    virtual Eigen::Matrix3r getGreenLagrangeStrainTensor() const;

    template <typename T>
    void getGreenLagrangeStrainVector( Eigen::Vector<T, 6>& strainVec ) const
    {
        return util::Voigt<mfem::real_t, T>( getGreenLagrangeStrainTensor(), true, strainVec );
    }

    virtual const Eigen::Vector6r& getGreenLagrangeStrainVector() const;

    bool isSmallDeformation() const
    {
        return mSmallDeformation;
    }

    virtual Eigen::Matrix3r getPK2StressTensor() const;

    template <typename T>
    void getPK2StressVector( Eigen::Vector<T, 6>& stressVec ) const
    {
        return util::Voigt<mfem::real_t, T>( getPK2StressTensor(), false, stressVec );
    }

    virtual const Eigen::Vector6r& getPK2StressVector() const;

    Eigen::Matrix3r getCauchyStressTensor() const;

    Eigen::Vector6r getCauchyStressVector() const;

    virtual void updateRefModuli() = 0;

    void updateCurModuli();

    const Eigen::Matrix6r& getRefModuli() const
    {
        return mRefModuli;
    }

    const Eigen::Matrix6r& getCurModuli() const
    {
        return mCurModuli;
    }

    void at( mfem::ElementTransformation& eltran, const mfem::IntegrationPoint& p )
    {
        mEleTrans = &eltran;
        mIntgP = &p;
    }

    void setDeformationGradient( const Eigen::Matrix<mfem::real_t, 3, 3>& F )
    {
        mdxdX = &F;
        mHasMechanicalStrain = false;
    }

    void setMechanicalStrain( const Eigen::Matrix3r& strain )
    {
        MFEM_VERIFY( SupportsMechanicalStrainInput(),
                     "This material does not support an additive mechanical-strain input." );
        mMechanicalStrain = strain;
        mHasMechanicalStrain = true;
    }

    void setLargeDeformation( const bool flg )
    {
        mSmallDeformation = !flg;
    }

    void setLoadFactor( const mfem::real_t loadFactor )
    {
        mLoadFactor = loadFactor;
    }

    virtual bool SupportsMechanicalStrainInput() const noexcept
    {
        return false;
    }

    Eigen::Vector6r getIntrinsicPK2StressVector() const;

    void setIntrinsicStress( mfem::VectorCoefficient* intrinsicStress )
    {
        mIntrinsicStress = intrinsicStress;
    }

protected:
    // moduli in reference configuration
    Eigen::Matrix6r mRefModuli;

    // moduli in current configuration
    Eigen::Matrix6r mCurModuli;
    const Eigen::Matrix3r* mdxdX{ nullptr };
    Eigen::Matrix3r mMechanicalStrain;
    bool mHasMechanicalStrain{ false };
    bool mSmallDeformation{ true };

    mfem::ElementTransformation* mEleTrans{ nullptr };
    const mfem::IntegrationPoint* mIntgP{ nullptr };
    mfem::real_t mLoadFactor{ 0 };
    // intrinsic stress
    mfem::VectorCoefficient* mIntrinsicStress{ nullptr };

    // strain cache
    mutable Eigen::Vector6r mStrainVec;
    mutable Eigen::Vector6r mStressVec;
};

class IsotropicElasticMaterial : public ElasticMaterial
{
public:
    IsotropicElasticMaterial( mfem::Coefficient& E, mfem::Coefficient& nu ) : ElasticMaterial(), mE( &E ), mNu( &nu )
    {
    }

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

    virtual void updateRefModuli() override;

    virtual const Eigen::Vector6r& getPK2StressVector() const;

    bool SupportsMechanicalStrainInput() const noexcept override
    {
        return true;
    }

protected:
    mfem::Coefficient* mE{ nullptr };
    mfem::Coefficient* mNu{ nullptr };
};
