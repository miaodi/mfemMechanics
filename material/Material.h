#pragma once
#include "mfem.hpp"
#include "typeDef.h"
#include <Eigen/Dense>

namespace util
{
Eigen::Vector6d Voigt( const Eigen::Matrix3d& tensor, const bool isStrain );

Eigen::Matrix3d InverseVoigt( const Eigen::Vector6d& vector, const bool isStrain );

short Voigt( const short i, const short pos );

void symmetricIdentityTensor( const Eigen::Matrix3d& C, Eigen::Matrix6d& CC );

void tensorProduct( const Eigen::Matrix3d& A, const Eigen::Matrix3d& B, Eigen::Matrix6d& CC );
} // namespace util

class ElasticMaterial
{
public:
    ElasticMaterial();

    virtual Eigen::Matrix3d getGreenLagrangeStrainTensor() const;

    virtual Eigen::Vector6d getGreenLagrangeStrainVector() const;

    bool isSamllDeformation() const
    {
        return mSmallDeformation;
    }

    virtual Eigen::Matrix3d getPK2StressTensor() const;

    virtual Eigen::Vector6d getPK2StressVector() const;

    Eigen::Matrix3d getCauchyStressTensor() const;

    Eigen::Vector6d getCauchyStressVector() const;

    virtual void updateRefModuli() = 0;

    void updateCurModuli();

    const Eigen::Matrix6d& getRefModuli() const
    {
        return mRefModuli;
    }

    const Eigen::Matrix6d& getCurModuli() const
    {
        return mCurModuli;
    }

    void at( mfem::ElementTransformation& eltran, const mfem::IntegrationPoint& p )
    {
        mEleTrans = &eltran;
        mIntgP = &p;
    }

    void setDeformationGradient( const Eigen::Matrix<double, 3, 3>& F )
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

protected:
    // moduli in reference configuration
    Eigen::Matrix6d mRefModuli;

    // moduli in current configuration
    Eigen::Matrix6d mCurModuli;
    const Eigen::Matrix3d* mdxdX{ nullptr };
    bool mSmallDeformation{ true };

    mfem::ElementTransformation* mEleTrans{ nullptr };
    const mfem::IntegrationPoint* mIntgP{ nullptr };
    double mLambda{ 0 };
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

    virtual Eigen::Vector6d getPK2StressVector() const;

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

    virtual Eigen::Matrix3d getGreenLagrangeStrainTensor() const;

protected:
    mfem::Coefficient* mCTE{ nullptr };
    double mT0{ 0 };
    double mTF{ 0 };
};