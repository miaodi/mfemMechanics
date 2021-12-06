#pragma once
#include "mfem.hpp"
#include <Eigen/Dense>

namespace util
{
Eigen::Vector<double, 6> Voigt( const Eigen::Matrix3d& tensor, const bool isStrain );

Eigen::Matrix3d InverseVoigt( const Eigen::Vector<double, 6>& vector, const bool isStrain );

} // namespace util

class ElasticMaterial
{
public:
    ElasticMaterial();

    Eigen::Matrix3d getGreenLagrangeStrainTensor() const;

    Eigen::Vector<double, 6> getGreenLagrangeStrainVector() const;

    bool isSamllDeformation() const
    {
        return mSmallDeformation;
    }

    Eigen::Matrix3d getPK2StressTensor() const;

    Eigen::Vector<double, 6> getPK2StressVector() const;

    Eigen::Matrix3d getCauchyStressTensor() const;

    Eigen::Vector<double, 6> getCauchyStressVector() const;

    virtual void updateRefModuli() = 0;

    virtual void updateCurModuli() = 0;

    const Eigen::Matrix<double, 6, 6>& getRefModuli() const
    {
        return mRefModuli;
    }

    const Eigen::Matrix<double, 6, 6>& getCurModuli() const
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

    void setLargeDeformation()
    {
        mSmallDeformation = false;
    }

protected:
    // moduli in reference configuration
    Eigen::Matrix<double, 6, 6> mRefModuli;

    // moduli in current configuration
    Eigen::Matrix<double, 6, 6> mCurModuli;
    const Eigen::Matrix3d* mdxdX{ nullptr };
    bool mSmallDeformation{ true };

    mfem::ElementTransformation* mEleTrans{ nullptr };
    const mfem::IntegrationPoint* mIntgP{ nullptr };
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

    void updateRefModuli() override;

    void updateCurModuli() override;

protected:
    mfem::Coefficient* mE{ nullptr };
    mfem::Coefficient* mNu{ nullptr };
};