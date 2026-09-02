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

    bool isSamllDeformation() const
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
    }

    void setLargeDeformation( const bool flg )
    {
        mSmallDeformation = !flg;
    }

    void setLambda( const double l )
    {
        mLambda = l;
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
    bool mSmallDeformation{ true };

    mfem::ElementTransformation* mEleTrans{ nullptr };
    const mfem::IntegrationPoint* mIntgP{ nullptr };
    double mLambda{ 0 };
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

    virtual void updateRefModuli() override;

    virtual const Eigen::Vector6r& getPK2StressVector() const;

protected:
    mfem::Coefficient* mE{ nullptr };
    mfem::Coefficient* mNu{ nullptr };
};

class IsotropicElasticThermalMaterial : public IsotropicElasticMaterial
{
public:
    IsotropicElasticThermalMaterial( mfem::Coefficient& E, mfem::Coefficient& nu, mfem::Coefficient& cte )
        : IsotropicElasticMaterial( E, nu ), mCTE( &cte )
    {
    }

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

    void setInitialTemp( const double t0 )
    {
        mT0 = t0;
    }
    void setFinalTemp( const double tf )
    {
        mTF = tf;
    }

    double CTE() const
    {
        MFEM_ASSERT( mEleTrans && mIntgP, "ElementTransformation or IntegrationPoint is not set" );
        return mCTE->Eval( *mEleTrans, *mIntgP );
    }

    virtual Eigen::Matrix3r getGreenLagrangeStrainTensor() const;

protected:
    mfem::Coefficient* mCTE{ nullptr };
    double mT0{ 0 };
    double mTF{ 0 };
};
