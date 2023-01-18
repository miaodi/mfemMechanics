
#include "Plugin.h"
#include "mfem.hpp"
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>
#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>
#include <fstream>
#include <iostream>

using VectorXr = std::vector<autodiff::real>;

using namespace std;
using namespace mfem;

class GeneralResidualMonitor : public IterativeSolverMonitor
{
public:
    GeneralResidualMonitor( const std::string& prefix_, int print_lvl ) : prefix( prefix_ )
    {
        print_level = print_lvl;
    }

    virtual void MonitorResidual( int it, double norm, const Vector& r, bool final );

private:
    const std::string prefix;
    int print_level;
    mutable double norm0;
};

void ReferenceConfiguration( const Vector& x, Vector& y )
{
    // Set the reference, stress free, configuration
    y = x;
}

void GeneralResidualMonitor::MonitorResidual( int it, double norm, const Vector& r, bool final )
{
    if ( print_level == 1 || ( print_level == 3 && ( final || it == 0 ) ) )
    {
        mfem::out << prefix << " iteration " << setw( 2 ) << it << " : ||r|| = " << norm;
        if ( it > 0 )
        {
            mfem::out << ",  ||r||/||r_0|| = " << norm / norm0;
        }
        else
        {
            norm0 = norm;
        }
        mfem::out << '\n';
    }
}
autodiff::dual2nd f( const autodiff::ArrayXdual2nd& x, const autodiff::ArrayXdual2nd& p )
{
    autodiff::dual2nd res =
        p( 2 ) + p( 2 ) * autodiff::detail::exp( -x( 1 ) / p( 1 ) ) *
                     ( ( autodiff::dual2nd( 1. ) - p( 3 ) + x( 1 ) / p( 1 ) ) * ( autodiff::dual2nd( 1. ) - p( 4 ) ) /
                           ( p( 3 ) - autodiff::dual2nd( 1. ) ) -
                       ( p( 4 ) + ( p( 3 ) - p( 4 ) ) / ( p( 3 ) - autodiff::dual2nd( 1. ) ) * x( 1 ) / p( 1 ) ) *
                           autodiff::detail::exp( -x( 0 ) * x( 0 ) / p( 0 ) / p( 0 ) ) );
    return res;
}

autodiff::dual2nd ff( const autodiff::dual2nd& DeltaT,
                      const autodiff::dual2nd& DeltaN,
                      const autodiff::dual2nd& deltaT,
                      const autodiff::dual2nd& deltaN,
                      const autodiff::dual2nd& PhiN,
                      const autodiff::dual2nd& r,
                      const autodiff::dual2nd& q )
{
    autodiff::dual2nd res = PhiN + PhiN * autodiff::detail::exp( -DeltaN / deltaN ) *
                                       ( ( autodiff::dual2nd( 1. ) - r + DeltaN / deltaN ) *
                                             ( autodiff::dual2nd( 1. ) - q ) / ( r - autodiff::dual2nd( 1. ) ) -
                                         ( q + ( r - q ) / ( r - autodiff::dual2nd( 1. ) ) * DeltaN / deltaN ) *
                                             autodiff::detail::exp( -DeltaT * DeltaT / deltaT * deltaT ) );
    return res;
}
int main( int argc, char* argv[] )
{
    // 1. Parse command-line options.
    const char* mesh_file = "../../data/tensile.mesh";
    int order = 1;
    bool static_cond = false;
    bool visualization = 1;
    int refineLvl = 0;
    int localRefineLvl = 0;

    OptionsParser args( argc, argv );
    args.AddOption( &mesh_file, "-m", "--mesh", "Mesh file to use." );
    args.AddOption( &order, "-o", "--order", "Finite element order (polynomial degree)." );
    args.AddOption( &static_cond, "-sc", "--static-condensation", "-no-sc", "--no-static-condensation",
                    "Enable static condensation." );
    args.AddOption( &visualization, "-vis", "--visualization", "-no-vis", "--no-visualization",
                    "Enable or disable GLVis visualization." );
    args.AddOption( &refineLvl, "-r", "--refine-level", "Finite element refine level." );
    args.AddOption( &localRefineLvl, "-lr", "--local-refine-level", "Finite element local refine level." );
    args.Parse();
    if ( !args.Good() )
    {
        args.PrintUsage( cout );
        return 1;
    }
    args.PrintOptions( cout );
    cout << static_cond << endl;

    // 2. Read the mesh from the given mesh file. We can handle triangular,
    //    quadrilateral, tetrahedral or hexahedral elements with the same code.
    Mesh* mesh = new Mesh( mesh_file, 1, 1 );
    int dim = mesh->Dimension();

    // 3. Select the order of the finite element discretization space. For NURBS
    //    meshes, we increase the order by degree elevation.
    if ( mesh->NURBSext )
    {
        mesh->DegreeElevate( order, order );
    }

    //  4. Mesh refinement
    for ( int k = 0; k < refineLvl; k++ )
        mesh->UniformRefinement();

    for ( int k = 0; k < localRefineLvl; k++ )
    {
        int ne = mesh->GetNE();
        auto eles = mesh->GetElementsArray();
        Array<Refinement> refinements;
        for ( int i = 0; i < ne; i++ )
        {
            double* node{ nullptr };
            for ( int j = 0; j < eles[i]->GetNVertices(); j++ )
            {
                const int vi = eles[i]->GetVertices()[j];
                node = mesh->GetVertex( vi );
                if ( std::abs( node[1] ) < 1e-10 && node[0] + 1e-10 > 0 )
                {
                    refinements.Append( i );
                    break;
                }
            }
        }
        mesh->GeneralRefinement( refinements );
    }
    ofstream file;
    file.open( "refined.vtk" );
    mesh->PrintVTK( file );
    file.close();

    // 5. Define a finite element space on the mesh. Here we use vector finite
    //    elements, i.e. dim copies of a scalar finite element space. The vector
    //    dimension is specified by the last argument of the FiniteElementSpace
    //    constructor. For NURBS meshes, we use the (degree elevated) NURBS space
    //    associated with the mesh nodes.
    FiniteElementCollection* fec;
    FiniteElementSpace* fespace;
    if ( mesh->NURBSext )
    {
        fec = NULL;
        fespace = mesh->GetNodes()->FESpace();
    }
    else
    {
        fec = new H1_FECollection( order, dim, mfem::BasisType::GaussLobatto );
        fespace = new FiniteElementSpace( mesh, fec, dim );
    }
    cout << "Number of vertices: " << fespace->GetNV() << endl;
    cout << "Number of faces: " << mesh->GetNumFaces() << endl;
    cout << "Number of faces (with ghost): " << mesh->GetNumFacesWithGhost() << endl;
    cout << "Number of finite element unknowns: " << fespace->GetTrueVSize() << endl << "Assembling: " << endl;
    mfem::Array<int> vdofs;

    autodiff::ArrayXdual2nd x( 2 ); // the input vector x with 5 variables
    x << 1, 2;                      // x = [1, 2, 3, 4, 5]

    autodiff::ArrayXdual2nd p( 5 ); // the input parameter vector p with 3 variables
    p << 1, 2, 3, 4, 5;             // p = [1, 2, 3]

    autodiff::dual2nd u; // the output scalar u = f(x, p, q) evaluated together with gradient below

    autodiff::VectorXdual g; // gradient of f(x) evaluated together with Hessian below

    Eigen::MatrixXd H = hessian( f, wrt( x ), at( x, p ), u,
                                 g ); // evaluate the function value u, its gradient vector g, and its Hessian matrix H with respect to (x, p, q)
    std::cout << "u = " << u << std::endl;   // print the evaluated output u
    std::cout << "g =\n" << g << std::endl;  // print the evaluated gradient vector g = [du/dx, du/dp, du/dq]
    std::cout << "H = \n" << H << std::endl; // print the evaluated Hessian matrix H = d²u/d[x, p, q]²

    autodiff::dual2nd DeltaT = 1.;
    autodiff::dual2nd DeltaN = 2.;
    autodiff::dual2nd deltaT = 1.;
    autodiff::dual2nd deltaN = 2.0;
    autodiff::dual2nd PhiN = 3.0;
    autodiff::dual2nd r = 4.0;
    autodiff::dual2nd q = 5.0;

    auto [u0, ux, uxy] = autodiff::derivatives( ff, autodiff::wrt( DeltaT, DeltaN ), autodiff::at( DeltaT, DeltaN, deltaT, deltaN, PhiN, r, q ) );

    std::cout << "u0 = " << u0 << std::endl;       // print the evaluated value of u = f(x, y, z)
    std::cout << "ux = " << ux << std::endl;       // print the evaluated derivative du/dx
    std::cout << "uxy = " << uxy << std::endl;     // print the evaluated derivative d²u/dxdy
    return 0;
}
