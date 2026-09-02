#include "Material.h"
#include "util.h"

ElasticMaterial::ElasticMaterial() : mRefModuli(), mCurModuli()
{
}

Eigen::Matrix3r ElasticMaterial::getGreenLagrangeStrainTensor() const
{
    if ( mHasMechanicalStrain )
    {
        return mMechanicalStrain;
    }

    MFEM_VERIFY( mdxdX != nullptr, "The deformation gradient has not been set." );
    if ( isSmallDeformation() )
    {
        Eigen::Matrix3r dudX = *mdxdX - Eigen::Matrix3r::Identity();
        return .5 * ( dudX + dudX.transpose() );
    }
    else
    {
        return .5 * ( ( *mdxdX ).transpose() * ( *mdxdX ) - Eigen::Matrix3r::Identity() );
    }
}

const Eigen::Vector6r& ElasticMaterial::getGreenLagrangeStrainVector() const
{
    getGreenLagrangeStrainVector<mfem::real_t>( mStrainVec );
    return mStrainVec;
}

Eigen::Matrix3r ElasticMaterial::getPK2StressTensor() const
{
    return util::InverseVoigt( getPK2StressVector(), false );
}

const Eigen::Vector6r& ElasticMaterial::getPK2StressVector() const
{
    getPK2StressVector<mfem::real_t>( mStressVec );
    return mStressVec;
}

Eigen::Matrix3r ElasticMaterial::getCauchyStressTensor() const
{
    return 1. / mdxdX->determinant() * ( *mdxdX * getPK2StressTensor() * mdxdX->transpose() );
}

Eigen::Vector6r ElasticMaterial::getCauchyStressVector() const
{
    return util::Voigt<mfem::real_t, mfem::real_t>( getCauchyStressTensor(), false );
}

void ElasticMaterial::updateCurModuli()
{
    const Eigen::Matrix3r& F = *mdxdX;
    static const Eigen::Matrix<int, 3, 3> indexMap{ { 0, 3, 5 }, { 3, 1, 4 }, { 5, 4, 2 } };
    static const Eigen::Matrix<int, 2, 6> inverseMap{ { 0, 1, 2, 0, 1, 0 }, { 0, 1, 2, 1, 2, 2 } };
    const mfem::real_t determinant = F.determinant();
    for ( int i = 0; i < 6; i++ )
    {
        for ( int j = 0; j < 6; j++ )
        {
            mfem::real_t term = 0;

            for ( int ii = 0; ii < 3; ii++ )
            {
                for ( int ij = 0; ij < 3; ij++ )
                {
                    for ( int ji = 0; ji < 3; ji++ )
                    {
                        for ( int jj = 0; jj < 3; jj++ )
                        {
                            term += F( inverseMap( 0, i ), ii ) * F( inverseMap( 1, i ), ij ) * F( inverseMap( 0, j ), ji ) *
                                    F( inverseMap( 1, j ), jj ) * mRefModuli( indexMap( ii, ij ), indexMap( ji, jj ) );
                        }
                    }
                }
            }

            mCurModuli( i, j ) = 1. / determinant * term;
        }
    }
}

Eigen::Vector6r ElasticMaterial::getIntrinsicPK2StressVector() const
{
    Eigen::Vector6r stressVector;
    stressVector.setZero();
    mfem::Vector vec( stressVector.data(), stressVector.size() );
    if ( mIntrinsicStress )
        mIntrinsicStress->Eval( vec, *mEleTrans, *mIntgP );
    return stressVector * mLoadFactor;
}

void IsotropicElasticMaterial::updateRefModuli()
{
    const mfem::real_t Nu = this->Nu();
    const mfem::real_t E = this->E();

    const mfem::real_t mu = E / ( 2. * ( 1. + Nu ) );
    const mfem::real_t lambda = ( Nu * E ) / ( ( 1 + Nu ) * ( 1. - 2. * Nu ) );
    const mfem::real_t dilatation = lambda + 2. * mu;

    mRefModuli.setZero();

    mRefModuli( 0, 0 ) = dilatation;
    mRefModuli( 1, 1 ) = dilatation;
    mRefModuli( 2, 2 ) = dilatation;

    mRefModuli( 0, 1 ) = lambda;
    mRefModuli( 0, 2 ) = lambda;
    mRefModuli( 1, 0 ) = lambda;
    mRefModuli( 1, 2 ) = lambda;
    mRefModuli( 2, 0 ) = lambda;
    mRefModuli( 2, 1 ) = lambda;

    mRefModuli( 3, 3 ) = mu;
    mRefModuli( 4, 4 ) = mu;
    mRefModuli( 5, 5 ) = mu;
}

const Eigen::Vector6r& IsotropicElasticMaterial::getPK2StressVector() const
{
    if ( mIntrinsicStress )
        mStressVec = getRefModuli() * getGreenLagrangeStrainVector() + getIntrinsicPK2StressVector();
    else
        mStressVec = getRefModuli() * getGreenLagrangeStrainVector();
    return mStressVec;
}
