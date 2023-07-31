//                                MFEM Example 0
//
// Compile with: make ex0
//
// Sample runs:  ex0
//               ex0 -m ../data/fichera.mesh
//               ex0 -m ../data/square-disc.mesh -o 2
//
// Description: This example code demonstrates the most basic usage of MFEM to
//              define a simple finite element discretization of the Laplace
//              problem -Delta u = 1 with zero Dirichlet boundary conditions.
//              General 2D/3D mesh files and finite element polynomial degrees
//              can be specified by command line options.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

int main( int argc, char* argv[] )
{
    // 1. Parse command line options.
    const char* mesh_file = "../../data/mesh.msh";
    int order = 1;

    OptionsParser args( argc, argv );
    args.AddOption( &mesh_file, "-m", "--mesh", "Mesh file to use." );
    args.AddOption( &order, "-o", "--order", "Finite element polynomial degree" );
    args.ParseCheck();

    // 2. Read the mesh from the given mesh file, and refine once uniformly.
    Mesh mesh( mesh_file );

    // 3. Define a finite element space on the mesh. Here we use H1 continuous
    //    high-order Lagrange finite elements of the given order.
    H1_FECollection fec( order, mesh.Dimension() );
    FiniteElementSpace fespace( &mesh, &fec );

    // 4. Extract the list of all the boundary DOFs. These will be marked as
    //    Dirichlet in order to enforce zero boundary conditions.
    Array<int> ess_bdr( fespace.GetMesh()->bdr_attributes.Max() );
    ess_bdr = 0;
    ess_bdr[10] = 0;
    ess_bdr[11] = 1; // boundary attribute 1 (index 10) is fixed

    // 5. Define the solution x as a finite element grid function in fespace. Set
    //    the initial guess to zero, which also sets the boundary conditions.
    GridFunction u_gf( &fespace );

    // ================================================================= //
    // 8. Set u initially to zero.
    ConstantCoefficient zero( 0.0 );
    u_gf.ProjectCoefficient( zero );
    Vector u, b;
    u_gf.GetTrueDofs( u );
    u_gf.GetTrueDofs( b );
    // ================================================================= //

    // ================================================================= //
    // 8.b Mark Essential true DOFs for bottom and top walls (will treat
    // as inhomogeneous Dirichlet). Fill boundary of u to be 10.0.
    ConstantCoefficient five( 5.0 );
    u_gf.ProjectBdrCoefficient( five, ess_bdr );
    u_gf.GetTrueDofs( u );

    ess_bdr[10] = 1;
    Array<int> ess_tdof_list;
    fespace.GetEssentialTrueDofs( ess_bdr, ess_tdof_list );

    // 7. Set up the bilinear form a(.,.) corresponding to the -Delta operator.
    BilinearForm a( &fespace );
    ConstantCoefficient Kappa( 1. );
    a.AddDomainIntegrator( new DiffusionIntegrator( Kappa ) );
    a.Assemble();

    // 8. Form the linear system A X = B. This includes eliminating boundary
    //    conditions, applying AMR constraints, and other transformations.
    SparseMatrix A;
    Vector B, X;
    a.FormLinearSystem( ess_tdof_list, u, b, A, X, B );


    // 9. Solve the system using PCG with symmetric Gauss-Seidel preconditioner.
    GSSmoother M( A );
    PCG( A, M, B, X, 1, 200, 1e-12, 0.0 );

    // 10. Recover the solution x as a grid function and save to file. The output
    //     can be viewed using GLVis as follows: "glvis -m mesh.mesh -g sol.gf"
    a.RecoverFEMSolution( X, b, u );
    u.Print();
    // x.Save( "sol.gf" );
    // mesh.Save( "mesh.mesh" );

    return 0;
}
