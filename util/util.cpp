#include "util.h"

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