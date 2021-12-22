
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
    const char* mesh_file = "/home/dimiao/repo/mfem/data/beam-hex.mesh";
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

    bool small = true;

    // 2. Read the mesh from the given mesh file. We can handle triangular,
    //    quadrilateral, tetrahedral or hexahedral elements with the same code.
    Mesh* mesh = new Mesh( mesh_file, 1, 1 );
    int dim = mesh->Dimension();

    if ( mesh->attributes.Max() < 2 || mesh->bdr_attributes.Max() < 2 )
    {
        cerr << "\nInput mesh should have at least two materials and "
             << "two boundary attributes! (See schematic in ex2.cpp)\n"
             << endl;
        return 3;
    }

    // 3. Select the order of the finite element discretization space. For NURBS
    //    meshes, we increase the order by degree elevation.
    if ( mesh->NURBSext )
    {
        mesh->DegreeElevate( order, order );
    }

    // 4. Refine the mesh to increase the resolution. In this example we do
    //    'ref_levels' of uniform refinement. We choose 'ref_levels' to be the
    //    largest number that gives a final mesh with no more than 5,000
    //    elements.
    {
        int ref_levels = (int)floor( log( 20000. / mesh->GetNE() ) / log( 2. ) / dim );
        for ( int l = 0; l < ref_levels; l++ )
        {
            mesh->UniformRefinement();
        }
    }

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
        fespace = new FiniteElementSpace( mesh, fec, dim, Ordering::byVDIM );
    }
    cout << "Number of finite element unknowns: " << fespace->GetTrueVSize() << endl << "Assembling: " << flush;

    // 6. Determine the list of true (i.e. conforming) essential boundary dofs.
    //    In this example, the boundary conditions are defined by marking only
    //    boundary attribute 1 from the mesh as essential and converting it to a
    //    list of true dofs.
    Array<int> ess_tdof_list, ess_bdr( mesh->bdr_attributes.Max() );
    ess_bdr = 0;
    ess_bdr[0] = 1;

    ess_bdr[1] = 1;
    fespace->GetEssentialTrueDofs( ess_bdr, ess_tdof_list );
    // 8. Define the solution vector x as a finite element grid function
    //    corresponding to fespace. Initialize x with initial guess of zero,
    //    which satisfies the boundary conditions.
    GridFunction x_gf( fespace );
    GridFunction x_ref( fespace );
    GridFunction x_def( fespace );

    VectorFunctionCoefficient deform( dim, InitialDeformation );
    VectorFunctionCoefficient refconfig( dim, ReferenceConfiguration );

    x_gf.ProjectCoefficient( refconfig );
    x_ref.ProjectCoefficient( refconfig );

    Array<int> bdr2( mesh->bdr_attributes.Max() );
    bdr2 = 0;
    bdr2[1] = 1;
    GridFunction bcgf( fespace );
    Vector vec( 3 );
    vec( 1 ) = .2;
    VectorConstantCoefficient vcc( vec );
    bcgf.ProjectBdrCoefficient( vcc, bdr2 );
    Vector Nu( mesh->attributes.Max() );
    Nu = .33;
    PWConstCoefficient nu_func( Nu );

    Vector E( mesh->attributes.Max() );
    E = 2.5e8;
    PWConstCoefficient E_func( E );

    IsotropicElasticMaterial iem( E_func, nu_func );
    iem.setLargeDeformation();
    if ( small )
    {
        for ( int i = 0; i < 30; i++ )
        {
            x_def += bcgf;
        }

        BilinearForm* a = new BilinearForm( fespace );
        auto ei = new plugin::ElasticityIntegrator( iem );
        ei->resizeRefEleTransVec( mesh->GetNE() );
        a->AddDomainIntegrator( ei );

        a->Assemble();

        LinearForm* b = new LinearForm( fespace );
        cout << "r.h.s. ... " << flush;
        b->Assemble();

        SparseMatrix A;
        Vector B, X;
        a->FormLinearSystem( ess_tdof_list, x_def, *b, A, X, B );

        GSSmoother M( A );
        PCG( A, M, B, X, 1, 500, 1e-8, 0.0 );

        // 12. Recover the solution as a finite element grid function.
        a->RecoverFEMSolution( X, *b, x_def );
    }
    else
    {
        auto intg = new plugin::NonlinearElasticityIntegrator( iem );

        NonlinearForm* nlf = new NonlinearForm( fespace );
        nlf->AddDomainIntegrator( intg );
        nlf->SetEssentialBC( ess_bdr );

        GeneralResidualMonitor newton_monitor( "Newton", 1 );
        GeneralResidualMonitor j_monitor( "GMRES", 3 );

        // Set up the Jacobian solver
        auto* j_gmres = new CGSolver();
        j_gmres->iterative_mode = false;
        j_gmres->SetRelTol( 1e-12 );
        j_gmres->SetAbsTol( 1e-12 );
        j_gmres->SetMaxIter( 2000 );
        j_gmres->SetPrintLevel( -1 );
        // j_gmres->SetMonitor( j_monitor );

        auto newton_solver = new NewtonSolver();

        // Set the newton solve parameters
        newton_solver->iterative_mode = true;
        newton_solver->SetSolver( *j_gmres );
        newton_solver->SetOperator( *nlf );
        newton_solver->SetPrintLevel( -1 );
        newton_solver->SetMonitor( newton_monitor );
        newton_solver->SetRelTol( 1e-8 );
        newton_solver->SetAbsTol( 1e-10 );
        newton_solver->SetMaxIter( 50 );
        for ( int i = 0; i < 30; i++ )
        {
            x_gf += bcgf;

            Vector zero;
            newton_solver->Mult( zero, x_gf );
        }
        // MFEM_VERIFY( newton_solver->GetConverged(), "Newton Solver did not converge." );
        subtract( x_gf, x_ref, x_def );
    }

    FiniteElementSpace scalar_space( mesh, fec );

    plugin::StressCoefficient stress_c( dim, iem );
    stress_c.SetDisplacement( x_def );

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
    paraview_dc.RegisterField( "Displace", &x_def );
    for ( int i = 0; i < dim; i++ )
    {
        for ( int j = 0; j < dim; j++ )
        {
            stress_c.SetComponent( i, j );
            auto stress = new GridFunction( &scalar_space );
            stress->ProjectCoefficient( stress_c );
            string x( 1, c[i] );
            string y( 1, c[j] );
            string name = "S" + x + y;

            paraview_dc.RegisterField( name, stress );
        }
    }
    paraview_dc.Save();
    if ( fec )
    {
        delete fespace;
        delete fec;
    }
    delete mesh;

    return 0;
}
