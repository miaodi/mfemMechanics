
#include "Plugin.h"
#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

int main( int argc, char* argv[] )
{
    // 1. Parse command-line options.
    const char* mesh_file = "../data/square-nurbs.mesh";
    int order = 1;
    bool static_cond = false;
    bool visualization = 1;
    int ref_levels = 0;

    OptionsParser args( argc, argv );
    args.AddOption( &mesh_file, "-m", "--mesh", "Mesh file to use." );
    args.AddOption( &order, "-o", "--order", "Finite element order (polynomial degree)." );
    args.AddOption( &static_cond, "-sc", "--static-condensation", "-no-sc", "--no-static-condensation",
                    "Enable static condensation." );
    args.AddOption( &visualization, "-vis", "--visualization", "-no-vis", "--no-visualization",
                    "Enable or disable GLVis visualization." );
    args.AddOption( &ref_levels, "-r", "--refine", "Number of times to refine the mesh uniformly." );
    args.Parse();
    if ( !args.Good() )
    {
        args.PrintUsage( cout );
        return 1;
    }
    args.PrintOptions( cout );

    // 2. Read the mesh from the given mesh file. We can handle triangular,
    //    quadrilateral, tetrahedral or hexahedral elements with the same code.
    Mesh* mesh = new Mesh( mesh_file, 1, 1 );
    int dim = mesh->Dimension();
    cout << "dim: " << dim << endl;

    // 3. Select the order of the finite element discretization space. For NURBS
    //    meshes, we increase the order by degree elevation.
    if ( mesh->NURBSext )
    {
        mesh->DegreeElevate( order, order );

        for ( int l = 0; l < ref_levels; l++ )
        {
            mesh->UniformRefinement();
        }
    }
    else
    {
        return 1;
    }

    FiniteElementCollection* fec;
    FiniteElementSpace* fespace;

    fec = new NURBSFECollection( order );
    fespace = new FiniteElementSpace( mesh, fec );

    cout << "Number of finite element unknowns: " << fespace->GetTrueVSize() << endl << "Assembling: " << endl;
    cout << "Number of finite element unknowns: " << mesh->NURBSext->GetGNV() << endl << "Assembling: " << endl;

    for ( int i = 0; i < mesh->GetNV(); i++ )
    {
        cout << "index: " << i << " x: " << mesh->GetVertex( i )[0] << " y: " << mesh->GetVertex( i )[1] << endl;
    }
    ofstream mesh_ofs( "../data/square-nurbs-refine.mesh" );
    mesh_ofs.precision( 8 );
    mesh->Print( mesh_ofs );
    mesh_ofs.close();
    return 0;
}
