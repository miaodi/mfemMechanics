
#include "FEMPlugin.h"
#include "Material.h"
#include "PostProc.h"
#include "mfem.hpp"
#include <fstream>
#include <iostream>

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

void ReferenceConfiguration( const Vector& x, Vector& y )
{
    // Set the reference, stress free, configuration
    y = x;
}

void InitialDeformation( const Vector& x, Vector& y )
{
    // Set the initial configuration. Having this different from the reference
    // configuration can help convergence
    y = x;
    y[1] = x[1] + .05 * x[0];
}

int main( int argc, char* argv[] )
{
    // 1. Parse command-line options.
    const char* mesh_file = "../beam.mesh";
    int order = 1;
    bool static_cond = false;
    bool visualization = 1;

    OptionsParser args( argc, argv );
    args.AddOption( &mesh_file, "-m", "--mesh", "Mesh file to use." );
    args.AddOption( &order, "-o", "--order", "Finite element order (polynomial degree)." );
    args.AddOption( &static_cond, "-sc", "--static-condensation", "-no-sc", "--no-static-condensation",
                    "Enable static condensation." );
    args.AddOption( &visualization, "-vis", "--visualization", "-no-vis", "--no-visualization",
                    "Enable or disable GLVis visualization." );
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

    for ( int k = 0; k < 7; k++ )
    {
        int ne = mesh->GetNE();
        Array<Refinement> refinements;
        for ( int i = 0; i < ne; i++ )
        {
            refinements.Append( Refinement( i, 1 ) );
        }
        mesh->GeneralRefinement( refinements );
    }
    for ( int k = 0; k < 3; k++ )
    {
        int ne = mesh->GetNE();
        Array<Refinement> refinements;
        for ( int i = 0; i < ne; i++ )
        {
            refinements.Append( Refinement( i, 2 ) );
        }
        mesh->GeneralRefinement( refinements );
    }

    // for ( int i = 0; i < 2; i++ )
    // {
    //     mesh->UniformRefinement();
    // }

    // // 4. Refine the mesh to increase the resolution. In this example we do
    // //    'ref_levels' of uniform refinement. We choose 'ref_levels' to be the
    // //    largest number that gives a final mesh with no more than 5,000
    // //    elements.
    // {
    //     int ref_levels = (int)floor( log( 500. / mesh->GetNE() ) / log( 2. ) / dim );
    //     for ( int l = 0; l < ref_levels; l++ )
    //     {
    //         mesh->UniformRefinement();
    //     }
    // }
    // mesh->UniformRefinement();

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
        fec = new H1_FECollection( order, dim );
        fespace = new FiniteElementSpace( mesh, fec, dim );
    }
    cout << "Number of finite element unknowns: " << fespace->GetTrueVSize() << endl << "Assembling: " << flush;

    // 6. Determine the list of true (i.e. conforming) essential boundary dofs.
    //    In this example, the boundary conditions are defined by marking only
    //    boundary attribute 1 from the mesh as essential and converting it to a
    //    list of true dofs.
    Array<int> ess_tdof_list, ess_bdr( mesh->bdr_attributes.Max() );
    ess_bdr = 0;
    ess_bdr[0] = 1;
    fespace->GetEssentialTrueDofs( ess_bdr, ess_tdof_list );

    // 7. Set up the linear form b(.) which corresponds to the right-hand side of
    //    the FEM linear system. In this case, b_i equals the boundary integral
    //    of f*phi_i where f represents a "pull down" force on the Neumann part
    //    of the boundary and phi_i are the basis functions in the finite element
    //    fespace. The force is defined by the VectorArrayCoefficient object f,
    //    which is a vector of Coefficient objects. The fact that f is non-zero
    //    on boundary attribute 2 is indicated by the use of piece-wise constants
    //    coefficient for its last component.
    VectorArrayCoefficient f( dim );
        f.Set( 0, new ConstantCoefficient( 0.0 ) );
        f.Set( 1, new ConstantCoefficient( 0.0 ) );
    {
        Vector pull_force( mesh->bdr_attributes.Max() );
        pull_force = 0.0;
        pull_force( 1 ) = 1.0e-4;
        f.Set( 2, new PWConstCoefficient( pull_force ) );
    }

    LinearForm* b = new LinearForm( fespace );
    b->AddBoundaryIntegrator( new VectorBoundaryLFIntegrator( f ) );
    cout << "r.h.s. ... " << flush;
    b->Assemble();

    // 8. Define the solution vector x as a finite element grid function
    //    corresponding to fespace. Initialize x with initial guess of zero,
    //    which satisfies the boundary conditions.
    GridFunction x( fespace );
    x = 0.0;

    // 9. Set up the bilinear form a(.,.) on the finite element space
    //    corresponding to the linear elasticity integrator with piece-wise
    //    constants coefficient lambda and mu.
    Vector lambda( mesh->attributes.Max() );
    lambda = 1.0;
    PWConstCoefficient lambda_func( lambda );
    Vector mu( mesh->attributes.Max() );
    mu = 1.0;
    PWConstCoefficient mu_func( mu );

    BilinearForm* a = new BilinearForm( fespace );
    a->AddDomainIntegrator( new ElasticityIntegrator( lambda_func, mu_func ) );

    // 10. Assemble the bilinear form and the corresponding linear system,
    //     applying any necessary transformations such as: eliminating boundary
    //     conditions, applying conforming constraints for non-conforming AMR,
    //     static condensation, etc.
    cout << "matrix ... " << flush;
    if ( static_cond )
    {
        a->EnableStaticCondensation();
    }
    a->Assemble();

    SparseMatrix A;
    Vector B, X;
    a->FormLinearSystem( ess_tdof_list, x, *b, A, X, B );
    cout << "done." << endl;

    cout << "Size of linear system: " << A.Height() << endl;

    // 11. Define a simple symmetric Gauss-Seidel preconditioner and use it to
    //     solve the system Ax=b with PCG.
    GSSmoother M( A );
    PCG( A, M, B, X, 1, 500, 1e-8, 0.0 );

    // 12. Recover the solution as a finite element grid function.
    a->RecoverFEMSolution( X, *b, x );

    // 15. Save data in the ParaView format
    // Visualize the stress components.
    const char* c = "xyz";
    ParaViewDataCollection paraview_dc( "test1", mesh );
    paraview_dc.SetPrefixPath( "ParaView" );
    paraview_dc.SetLevelsOfDetail( order );
    paraview_dc.SetCycle( 0 );
    paraview_dc.SetDataFormat( VTKFormat::BINARY );
    paraview_dc.SetHighOrderOutput( true );
    paraview_dc.SetTime( 0.0 ); // set the time
    paraview_dc.RegisterField( "Displace", &x );
    paraview_dc.Save();

    if ( fec )
    {
        delete fespace;
        delete fec;
    }
    delete mesh;

    return 0;
}
