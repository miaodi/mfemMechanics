#include "Material.h"

namespace util
{
Eigen::Vector6d Voigt( const Eigen::Matrix3d& tensor, const bool isStrain )
{
    if ( isStrain )
    {
        return Eigen::Vector6d{ { tensor( Voigt( 0, 0 ), Voigt( 0, 1 ) ), tensor( Voigt( 1, 0 ), Voigt( 1, 1 ) ),
                                  tensor( Voigt( 2, 0 ), Voigt( 2, 1 ) ), 2 * tensor( Voigt( 3, 0 ), Voigt( 3, 1 ) ),
                                  2 * tensor( Voigt( 4, 0 ), Voigt( 4, 1 ) ), 2 * tensor( Voigt( 5, 0 ), Voigt( 5, 1 ) ) } };
    }
    else
    {
        return Eigen::Vector6d{ { tensor( Voigt( 0, 0 ), Voigt( 0, 1 ) ), tensor( Voigt( 1, 0 ), Voigt( 1, 1 ) ),
                                  tensor( Voigt( 2, 0 ), Voigt( 2, 1 ) ), tensor( Voigt( 3, 0 ), Voigt( 3, 1 ) ),
                                  tensor( Voigt( 4, 0 ), Voigt( 4, 1 ) ), tensor( Voigt( 5, 0 ), Voigt( 5, 1 ) ) } };
    }
}

Eigen::Matrix3d InverseVoigt( const Eigen::Vector6d& vector, const bool isStrain )
{
    if ( isStrain )
    {
        return Eigen::Matrix3d{ { vector( 0 ), vector( 3 ) / 2, vector( 5 ) / 2 },
                                { vector( 3 ) / 2, vector( 1 ), vector( 4 ) / 2 },
                                { vector( 6 ) / 2, vector( 4 ) / 2, vector( 2 ) } };
    }
    else
    {
        return Eigen::Matrix3d{ { vector( 0 ), vector( 3 ), vector( 5 ) },
                                { vector( 3 ), vector( 1 ), vector( 4 ) },
                                { vector( 5 ), vector( 4 ), vector( 2 ) } };
    }
}

short Voigt( const short i, const short pos )
{
    bool even = pos % 2;
    switch ( i )
    {
    case 0:
        return 0;
    case 1:
        return 1;
    case 2:
        return 2;
    case 3:
        return even ? 0 : 1;
    case 4:
        return even ? 1 : 2;
    case 5:
        return even ? 0 : 2;
    default:
        // TODO: add warning
        return -1;
    }
}

void symmetricIdentityTensor( const Eigen::Matrix3d& C, Eigen::Matrix6d& CC )
{
    CC.setZero();

    for ( short i = 0; i < 6; ++i )
        for ( short j = 0; j < 6; ++j )
            CC( i, j ) = .5 * ( C( Voigt( i, 0 ), Voigt( j, 2 ) ) * C( Voigt( i, 1 ), Voigt( j, 3 ) ) +
                                C( Voigt( i, 0 ), Voigt( j, 3 ) ) * C( Voigt( i, 1 ), Voigt( j, 2 ) ) );
}

void tensorProduct( const Eigen::Matrix3d& A, const Eigen::Matrix3d& B, Eigen::Matrix6d& CC )
{
    CC.setZero();

    for ( short i = 0; i < 6; ++i )
        for ( short j = 0; j < 6; ++j )
            CC( i, j ) = A( Voigt( i, 0 ), Voigt( i, 1 ) ) * B( Voigt( j, 2 ), Voigt( j, 3 ) );
}

} // namespace util

ElasticMaterial::ElasticMaterial() : mRefModuli(), mCurModuli()
{
}

Eigen::Matrix3d ElasticMaterial::getGreenLagrangeStrainTensor() const
{
    if ( isSamllDeformation() )
    {
        Eigen::Matrix3d dudX = *mdxdX - Eigen::Matrix3d::Identity();
        return .5 * ( dudX + dudX.transpose() );
    }
    else
    {
        return .5 * ( ( *mdxdX ).transpose() * ( *mdxdX ) - Eigen::Matrix3d::Identity() );
    }
}

Eigen::Vector6d ElasticMaterial::getGreenLagrangeStrainVector() const
{
    return util::Voigt( getGreenLagrangeStrainTensor(), true );
}

Eigen::Matrix3d ElasticMaterial::getPK2StressTensor() const
{
    return util::InverseVoigt( getPK2StressVector(), false );
}

Eigen::Vector6d ElasticMaterial::getPK2StressVector() const
{
    return util::Voigt( getPK2StressTensor(), false );
}

Eigen::Matrix3d ElasticMaterial::getCauchyStressTensor() const
{
    return 1. / mdxdX->determinant() * ( *mdxdX * getPK2StressTensor() * mdxdX->transpose() );
}

Eigen::Vector6d ElasticMaterial::getCauchyStressVector() const
{
    return util::Voigt( getCauchyStressTensor(), false );
}

void ElasticMaterial::updateCurModuli()
{
    const Eigen::Matrix3d& F = *mdxdX;
    static const Eigen::Matrix<int, 3, 3> indexMap{ { 0, 3, 5 }, { 3, 1, 4 }, { 5, 4, 2 } };
    static const Eigen::Matrix<int, 2, 6> inverseMap{ { 0, 1, 2, 0, 1, 0 }, { 0, 1, 2, 1, 2, 2 } };
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

Eigen::Vector6d IsotropicElasticMaterial::getPK2StressVector() const
{
    return getRefModuli() * getGreenLagrangeStrainVector();
}