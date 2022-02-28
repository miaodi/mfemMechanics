#pragma once

#include "Material.h"

// mu/2*(I_1-3-2*ln(J))+lambda/2*(J-1)^2
class NeoHookeanMaterial : public ElasticMaterial
{
public:
    NeoHookeanMaterial( mfem::Coefficient& m, mfem::Coefficient& l ) : ElasticMaterial(), mMu( &m ), mLambda( &l )
    {
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
};