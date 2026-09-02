#include "NeoHookeanMaterial.h"
#include "util.h"

Eigen::Matrix3r NeoHookeanMaterial::getPK2StressTensor() const
{
    const Eigen::Matrix3r C = ( *mdxdX ).transpose() * ( *mdxdX );
    const Eigen::Matrix3r CInv = C.inverse();
    const auto mu = Mu();
    const auto lambda = Lambda();
    const auto J = ( *mdxdX ).determinant();
    switch ( mNH )
    {
    case NeoHookeanType::Poly1:
    {
        return mu * ( Eigen::Matrix3r::Identity() - CInv ) + lambda * ( J - 1 ) * J * CInv;
    }
    case NeoHookeanType::Ln:
    {
        return mu * ( Eigen::Matrix3r::Identity() - CInv ) + lambda * log( J ) * CInv;
    }
    case NeoHookeanType::Poly2:

    default:
        MFEM_ABORT( "Not implemented" );
        return Eigen::Matrix3r();
    }
}

void NeoHookeanMaterial::updateRefModuli()
{
    const Eigen::Matrix3r C = ( *mdxdX ).transpose() * ( *mdxdX );
    const Eigen::Matrix3r CInv = C.inverse();
    const auto mu = Mu();
    const auto lambda = Lambda();
    const auto J = ( *mdxdX ).determinant();

    util::tensorProduct( CInv, CInv, mRefModuli );

    util::symmetricIdentityTensor( CInv, mTempModuli );

    switch ( mNH )
    {
    case NeoHookeanType::Poly1:
    {
        mRefModuli *= lambda * ( 2 * J - 1 ) * J;
        mRefModuli += 2 * ( mu - lambda * ( J - 1 ) * J ) * mTempModuli;
        break;
    }
    case NeoHookeanType::Ln:
    {
        mRefModuli *= lambda;
        mRefModuli += 2 * ( mu - lambda * log( J ) ) * mTempModuli;
        break;
    }
    case NeoHookeanType::Poly2:
    default:
        MFEM_ABORT( "Not implemented" );
    }
}