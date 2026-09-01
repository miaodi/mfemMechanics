#include "util.h"
#include <vector>

namespace util
{

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

Eigen::Matrix6d TransformationVoigtForm( const Eigen::Matrix3d& t )
{
    Eigen::Matrix6d T;
    T( 0, 0 ) = t( 0, 0 ) * t( 0, 0 ), T( 0, 1 ) = t( 0, 1 ) * t( 0, 1 ), T( 0, 2 ) = t( 0, 2 ) * t( 0, 2 ),
            T( 0, 3 ) = t( 0, 0 ) * t( 0, 1 ), T( 0, 4 ) = t( 0, 1 ) * t( 0, 2 ), T( 0, 5 ) = t( 0, 0 ) * t( 0, 2 );
    T( 1, 0 ) = t( 1, 0 ) * t( 1, 0 ), T( 1, 1 ) = t( 1, 1 ) * t( 1, 1 ), T( 1, 2 ) = t( 1, 2 ) * t( 1, 2 ),
            T( 1, 3 ) = t( 1, 0 ) * t( 1, 1 ), T( 1, 4 ) = t( 1, 1 ) * t( 1, 2 ), T( 1, 5 ) = t( 1, 0 ) * t( 1, 2 );
    T( 2, 0 ) = t( 2, 0 ) * t( 2, 0 ), T( 2, 1 ) = t( 2, 1 ) * t( 2, 1 ), T( 2, 2 ) = t( 2, 2 ) * t( 2, 2 ),
            T( 2, 3 ) = t( 2, 0 ) * t( 2, 1 ), T( 2, 4 ) = t( 2, 1 ) * t( 2, 2 ), T( 2, 5 ) = t( 2, 0 ) * t( 2, 2 );

    T( 3, 0 ) = 2 * t( 0, 0 ) * t( 1, 0 ), T( 3, 1 ) = 2 * t( 0, 1 ) * t( 1, 1 ), T( 3, 2 ) = 2 * t( 0, 2 ) * t( 1, 2 ),
            T( 3, 3 ) = t( 0, 0 ) * t( 1, 1 ) + t( 0, 1 ) * t( 1, 0 ),
            T( 3, 4 ) = t( 0, 1 ) * t( 1, 2 ) + t( 0, 2 ) * t( 1, 1 ),
            T( 3, 5 ) = t( 0, 0 ) * t( 1, 2 ) + t( 0, 2 ) * t( 1, 0 );

    T( 4, 0 ) = 2 * t( 1, 0 ) * t( 2, 0 ), T( 4, 1 ) = 2 * t( 1, 1 ) * t( 2, 1 ), T( 4, 2 ) = 2 * t( 1, 2 ) * t( 2, 2 ),
            T( 4, 3 ) = t( 1, 0 ) * t( 2, 1 ) + t( 1, 1 ) * t( 2, 0 ),
            T( 4, 4 ) = t( 1, 1 ) * t( 2, 2 ) + t( 1, 2 ) * t( 2, 1 ),
            T( 4, 5 ) = t( 1, 0 ) * t( 2, 2 ) + t( 1, 2 ) * t( 2, 0 );

    T( 5, 0 ) = 2 * t( 0, 0 ) * t( 2, 0 ), T( 5, 1 ) = 2 * t( 0, 1 ) * t( 2, 1 ), T( 5, 2 ) = 2 * t( 0, 2 ) * t( 2, 2 ),
            T( 5, 3 ) = t( 0, 0 ) * t( 2, 1 ) + t( 0, 1 ) * t( 2, 0 ),
            T( 5, 4 ) = t( 0, 1 ) * t( 2, 2 ) + t( 0, 2 ) * t( 2, 1 ),
            T( 5, 5 ) = t( 0, 0 ) * t( 2, 2 ) + t( 0, 2 ) * t( 2, 0 );
    return T;
}

double ConvergenceRate( const double cur, const double prev, const double prevprev )
{
    return ( std::log( cur ) - std::log( prev ) ) / ( std::log( prev ) - std::log( prevprev ) );
}

// Welzl's algorithm
double SmallestCircle( const mfem::IntegrationRule& nodes, const int dim )
{
    static auto dist = []( const mfem::IntegrationPoint& a, const mfem::IntegrationPoint& b )
    { return std::sqrt( std::pow( a.x - b.x, 2 ) + std::pow( a.y - b.y, 2 ) + std::pow( a.z - b.z, 2 ) ); };
    struct Circle
    {
        mfem::IntegrationPoint center;
        double radius;
        Circle( const mfem::IntegrationPoint& c, const double r ) : center( c ), radius( r )
        {
        }
        Circle( const double cx, const double cy, const double r ) : radius( r )
        {
            center.x = cx;
            center.y = cy;
            center.z = 0;
        }

        bool isInside( const mfem::IntegrationPoint& p ) const
        {
            return dist( center, p ) <= radius;
        }
    };

    if ( dim == 3 )
        MFEM_ABORT( "smallest bound sphere is not implemented yet." );

    const int size = nodes.GetNPoints();
    if ( size <= 1 )
        return 0;
    if ( size == 2 )
        return dist( nodes.IntPoint( 0 ), nodes.IntPoint( 1 ) );
    auto formCircle = []( const mfem::IntegrationPoint& a, const mfem::IntegrationPoint& b, const mfem::IntegrationPoint& c )
    {
        Eigen::Matrix3d m;
        m << 2 * a.x, 2 * a.y, 1, 2 * b.x, 2 * b.y, 1, 2 * c.x, 2 * c.y, 1;
        Eigen::Vector3d rhs;
        rhs << a.x * a.x + a.y * a.y, b.x * b.x + b.y * b.y, c.x * c.x + c.y * c.y;
        Eigen::Vector3d sol = m.fullPivLu().solve( rhs );
        const double r = std::sqrt( sol( 2 ) + sol( 0 ) * sol( 0 ) + sol( 1 ) * sol( 1 ) );
        return Circle( sol( 0 ), sol( 1 ), r );
    };
    std::vector<int> index;
    std::function<Circle( const mfem::IntegrationRule&, std::vector<int>&, int )> recursive_helper =
        [&recursive_helper, &formCircle]( const mfem::IntegrationRule& nodes, std::vector<int>& index, int pos )
    {
        if ( nodes.GetNPoints() - pos == 3 )
            return formCircle( nodes.IntPoint( pos + 0 ), nodes.IntPoint( pos + 1 ), nodes.IntPoint( pos + 2 ) );
        if ( index.size() == 3 )
            return formCircle( nodes.IntPoint( index[0] ), nodes.IntPoint( index[1] ), nodes.IntPoint( index[2] ) );

        auto d = recursive_helper( nodes, index, pos + 1 );
        if ( d.isInside( nodes.IntPoint( pos ) ) )
            return d;
        index.push_back( pos );
        return recursive_helper( nodes, index, pos + 1 );
    };
    return recursive_helper( nodes, index, 0 ).radius * 2;
}
} // namespace util