#pragma once

#include "Material.h"

enum class NeoHookeanType
{
    Poly1, // mu/2*(I_1-3-2*ln(J))+lambda/2*(J-1)^2
    Poly2,
    Ln, // mu/2*(I_1-3-2*ln(J))+lambda/2*(lnJ)^2
};

class NeoHookeanMaterial : public ElasticMaterial
{
public:
    NeoHookeanMaterial( mfem::Coefficient& m, mfem::Coefficient& l, NeoHookeanType nh = NeoHookeanType::Poly1 )
        : ElasticMaterial(), mMu( &m ), mLambda( &l ), mNH( nh )
    {
        setLargeDeformation();
    }

    virtual Eigen::Matrix3d getPK2StressTensor() const;

    double Mu() const
    {
        MFEM_ASSERT( mEleTrans && mIntgP, "ElementTransformation or IntegrationPoint is not set" );
        return mMu->Eval( *mEleTrans, *mIntgP );
    }

    double Lambda() const
    {
        MFEM_ASSERT( mEleTrans && mIntgP, "ElementTransformation or IntegrationPoint is not set" );
        return mLambda->Eval( *mEleTrans, *mIntgP );
    }

    virtual void updateRefModuli();

protected:
    mfem::Coefficient* mMu{ nullptr };
    mfem::Coefficient* mLambda{ nullptr };

    Eigen::Matrix6d mTempModuli;
    NeoHookeanType mNH;
};