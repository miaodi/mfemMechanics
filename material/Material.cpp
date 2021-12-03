#include "Material.h"

namespace util
{
Eigen::Vector<double, 6> Voigt( const Eigen::Matrix3d& tensor, const bool isStrain )
{
    if ( isStrain )
    {
        return Eigen::Vector<double, 6>{ { tensor( 0, 0 ), tensor( 1, 1 ), tensor( 2, 2 ), 2 * tensor( 0, 1 ),
                                           2 * tensor( 0, 2 ), 2 * tensor( 1, 2 ) } };
    }
    else
    {
        return Eigen::Vector<double, 6>{
            { tensor( 0, 0 ), tensor( 1, 1 ), tensor( 2, 2 ), tensor( 0, 1 ), tensor( 0, 2 ), tensor( 1, 2 ) } };
    }
}

Eigen::Matrix3d InverseVoigt( const Eigen::Vector<double, 6>& vector, const bool isStrain )
{
    if ( isStrain )
    {
        return Eigen::Matrix3d{ { vector( 0 ), vector( 3 ) / 2, vector( 4 ) / 2 },
                                { vector( 3 ) / 2, vector( 1 ), vector( 5 ) / 2 },
                                { vector( 4 ) / 2, vector( 5 ) / 2, vector( 2 ) } };
    }
    else
    {
        return Eigen::Matrix3d{ { vector( 0 ), vector( 3 ), vector( 4 ) },
                                { vector( 3 ), vector( 1 ), vector( 5 ) },
                                { vector( 4 ), vector( 5 ), vector( 2 ) } };
    }
}

} // namespace util

ElasticMaterial::ElasticMaterial() : mRefModuli(), mCurModuli()
{
}

Eigen::Matrix3d ElasticMaterial::getGreenLagrangeStrainTensor() const
{
    Eigen::Matrix3d dudX = *mdxdX - Eigen::Matrix3d::Identity();
    if ( isSamllDeformation() )
    {
        return .5 * ( dudX + dudX.transpose() );
    }
    else
    {
        return .5 * ( dudX.transpose() * dudX - Eigen::Matrix3d::Identity() );
    }
}

Eigen::Vector<double, 6> ElasticMaterial::getGreenLagrangeStrainVector() const
{
    return util::Voigt( getGreenLagrangeStrainTensor(), true );
}

Eigen::Vector<double, 6> ElasticMaterial::getPK2StressVector() const
{
    return getRefModuli() * getGreenLagrangeStrainVector();
}

Eigen::Matrix3d ElasticMaterial::getPK2StressTensor() const
{
    return util::InverseVoigt( getPK2StressVector(), false );
}

Eigen::Matrix3d ElasticMaterial::getCauchyStressTensor() const
{
    return 1. / mdxdX->determinant() * ( *mdxdX * getPK2StressTensor() * mdxdX->transpose() );
}

Eigen::Vector<double, 6> ElasticMaterial::getCauchyStressVector() const
{
    return util::Voigt( getCauchyStressTensor(), false );
}

void IsotropicElasticMaterial::updateRefModuli()
{
    const double Nu = this->Nu();
    const double E = this->E();

    const double mu = E / ( 2. * ( 1. + Nu ) );
    const double lambda = ( Nu * E ) / ( ( 1 + Nu ) * ( 1. - 2. * Nu ) );
    const double dilatation = lambda + 2. * mu;

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

void IsotropicElasticMaterial::updateCurModuli()
{
    const Eigen::Matrix3d& F = *mdxdX;
    static const Eigen::Matrix<int, 3, 3> indexMap{ { 0, 5, 4 }, { 5, 1, 3 }, { 4, 3, 2 } };
    static const Eigen::Matrix<int, 2, 6> inverseMap{ { 0, 1, 2, 0, 0, 1 }, { 0, 1, 2, 1, 2, 2 } };
    const double determinant = F.determinant();
    for ( int i = 0; i < 6; i++ )
    {
        for ( int j = 0; j < 6; j++ )
        {
            double term = 0;

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