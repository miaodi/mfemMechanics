
#include "FEMPlugin.h"
#include "Material.h"
#include "NeoHookeanMaterial.h"
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

int main( int argc, char* argv[] )
{
    // 1. Parse command-line options.
    const char* mesh_file = "../gmshBeam.msh";
    int order = 1;
    bool static_cond = false;
    bool visualization = 1;
    int refineLvl = 0;

    OptionsParser args( argc, argv );
    args.AddOption( &mesh_file, "-m", "--mesh", "Mesh file to use." );
    args.AddOption( &order, "-o", "--order", "Finite element order (polynomial degree)." );
    args.AddOption( &static_cond, "-sc", "--static-condensation", "-no-sc", "--no-static-condensation",
                    "Enable static condensation." );
    args.AddOption( &visualization, "-vis", "--visualization", "-no-vis", "--no-visualization",
                    "Enable or disable GLVis visualization." );
    args.AddOption( &refineLvl, "-r", "--refine-level", "Finite element refine level." );
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

    // for ( int k = 0; k < 7; k++ )
    // {
    //     int ne = mesh->GetNE();
    //     Array<Refinement> refinements;
    //     for ( int i = 0; i < ne; i++ )
    //     {
    //         refinements.Append( Refinement( i, 1 ) );
    //     }
    //     mesh->GeneralRefinement( refinements );
    // }
    // for ( int k = 0; k < 3; k++ )
    // {
    //     int ne = mesh->GetNE();
    //     Array<Refinement> refinements;
    //     for ( int i = 0; i < ne; i++ )
    //     {
    //         refinements.Append( Refinement( i, 2 ) );
    //     }
    //     mesh->GeneralRefinement( refinements );
    // }

    // for ( int i = 0; i < 2; i++ )
    // {
    //     mesh->UniformRefinement();
    // }

    // // 4. Refine the mesh to increase the resolution. In this example we do
    // //    'ref_levels' of uniform refinement. We choose 'ref_levels' to be the
    // //    largest number that gives a final mesh with no more than 5,000
    // //    elements.
    {
        for ( int l = 0; l < refineLvl; l++ )
        {
            mesh->UniformRefinement();
        }
    }
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
    fespace->GetEssentialTrueDofs( ess_bdr, ess_tdof_list );

    // 8. Define the solution vector x as a finite element grid function
    //    corresponding to fespace. Initialize x with initial guess of zero,
    //    which satisfies the boundary conditions.
    GridFunction x_gf( fespace );
    GridFunction x_ref( fespace );
    GridFunction x_def( fespace );

    VectorFunctionCoefficient refconfig( dim, ReferenceConfiguration );

    x_gf.ProjectCoefficient( refconfig );
    x_ref.ProjectCoefficient( refconfig );

    VectorArrayCoefficient f( dim );
    for ( int i = 0; i < dim; i++ )
    {
        f.Set( i, new ConstantCoefficient( 0.0 ) );
    }

    // Vector Nu( mesh->attributes.Max() );
    // Nu = .0;
    // PWConstCoefficient nu_func( Nu );

    // Vector E( mesh->attributes.Max() );
    // E = 1.2e6;
    // PWConstCoefficient E_func( E );

    // IsotropicElasticMaterial iem( E_func, nu_func );
    // iem.setLargeDeformation();
    double E = 1.2e6;

    double nu = 0;

    Vector Mu( mesh->attributes.Max() );
    // Mu = 1.61148e6;
    Mu = E / 2 / ( 1 + nu );

    PWConstCoefficient mu_func( Mu );

    Vector Lambda( mesh->attributes.Max() );
    // Lambda = 499.92568e6;
    Lambda = E * nu / ( 1 + nu ) / ( 1 - 2 * nu );
    PWConstCoefficient lambda_func( Lambda );

    NeoHookeanMaterial nh( mu_func, lambda_func, NeoHookeanType::Ln );

    plugin::Memorize mm( mesh );

    auto intg = new plugin::NonlinearElasticityIntegrator( nh, mm );

    NonlinearForm* nlf = new NonlinearForm( fespace );
    // {
    //     nlf->AddDomainIntegrator( new HyperelasticNLFIntegrator( new NeoHookeanModel( 1.5e6, 10e9 ) ) );
    // }
    nlf->AddDomainIntegrator( intg );
    nlf->SetEssentialTrueDofs( ess_tdof_list );
    // Vector r;
    // r.SetSize(nlf->Height());
    // nlf->Mult(x_gf, r);

    GeneralResidualMonitor newton_monitor( "Newton", 1 );
    GeneralResidualMonitor j_monitor( "GMRES", 3 );

    // Set up the Jacobian solver
    auto j_gmres = new UMFPackSolver();

    auto newton_solver = new NewtonSolver();

    // Set the newton solve parameters
    newton_solver->iterative_mode = true;
    newton_solver->SetSolver( *j_gmres );
    newton_solver->SetOperator( *nlf );
    newton_solver->SetPrintLevel( -1 );
    newton_solver->SetMonitor( newton_monitor );
    newton_solver->SetRelTol( 1e-6 );
    newton_solver->SetAbsTol( 1e-7 );
    newton_solver->SetMaxIter( 20 );

    nlf->AddBdrFaceIntegrator( new plugin::NonlinearVectorBoundaryLFIntegrator( f ) );

    // 15. Save data in the ParaView format
    ParaViewDataCollection paraview_dc( "Beam", mesh );
    paraview_dc.SetPrefixPath( "ParaView" );
    paraview_dc.SetLevelsOfDetail( order );
    paraview_dc.SetCycle( 0 );
    paraview_dc.SetDataFormat( VTKFormat::BINARY );
    paraview_dc.SetHighOrderOutput( true );
    paraview_dc.SetTime( 0.0 ); // set the time
    paraview_dc.RegisterField( "Displace", &x_def );

    paraview_dc.Save();
    for ( int i = 1; i <= 10; i++ )
    {
        Vector pull_force( mesh->bdr_attributes.Max() );
        pull_force = 0.0;
        pull_force( 1 ) = 4 * i;
        f.Set( 2, new PWConstCoefficient( pull_force ) );
        Vector zero;
        newton_solver->Mult( zero, x_gf );
        // MFEM_VERIFY( newton_solver->GetConverged(), "Newton Solver did not converge." );
        subtract( x_gf, x_ref, x_def );
        paraview_dc.SetTime( i ); // set the time
        paraview_dc.SetCycle( i );
        paraview_dc.Save();
    }

    if ( fec )
    {
        delete fespace;
        delete fec;
    }
    delete newton_solver;
    delete mesh;

    return 0;
}
