#include "NeoHookeanMaterial.h"

Eigen::Matrix3d NeoHookeanMaterial::getPK2StressTensor() const
{
    Eigen::Matrix3d C = ( *mdxdX ).transpose() * ( *mdxdX );
    Eigen::Matrix3d CInv = C.inverse();
    auto mu = Mu();
    auto lambda = Lambda();
    auto J = ( *mdxdX ).determinant();
    Eigen::Matrix3d S = ( lambda * ( J * J - J ) - mu ) * CInv + mu * Eigen::Matrix3d::Identity();
    return S;
}

void NeoHookeanMaterial::updateRefModuli()
{
    Eigen::Matrix3d C = ( *mdxdX ).transpose() * ( *mdxdX );
    Eigen::Matrix3d CInv = C.inverse();
    auto mu = Mu();
    auto lambda = Lambda();
    auto J = ( *mdxdX ).determinant();

    mRefModuli.setZero();

    // tensorProduct();
}