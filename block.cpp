
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
    const char* mesh_file = "../block.mesh";
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
    Array<int> ess_tdof_list, temp_list, ess_bdr( mesh->bdr_attributes.Max() );
    ess_bdr = 0;
    ess_bdr[0] = 1;
    fespace->GetEssentialTrueDofs( ess_bdr, temp_list, 2 );
    ess_tdof_list.Append( temp_list );

    ess_bdr = 0;
    ess_bdr[1] = 1;
    fespace->GetEssentialTrueDofs( ess_bdr, temp_list, 1 );
    ess_tdof_list.Append( temp_list );

    ess_bdr = 0;
    ess_bdr[2] = 1;
    fespace->GetEssentialTrueDofs( ess_bdr, temp_list, 0 );
    ess_tdof_list.Append( temp_list );

    ess_bdr = 0;
    ess_bdr[3] = 1;
    fespace->GetEssentialTrueDofs( ess_bdr, temp_list, 0 );
    ess_tdof_list.Append( temp_list );

    ess_bdr = 0;
    ess_bdr[3] = 1;
    fespace->GetEssentialTrueDofs( ess_bdr, temp_list, 1 );
    ess_tdof_list.Append( temp_list );

    ess_bdr = 0;
    ess_bdr[5] = 1;
    fespace->GetEssentialTrueDofs( ess_bdr, temp_list, 0 );
    ess_tdof_list.Append( temp_list );

    ess_bdr = 0;
    ess_bdr[5] = 1;
    fespace->GetEssentialTrueDofs( ess_bdr, temp_list, 1 );
    ess_tdof_list.Append( temp_list );

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

    Vector Mu( mesh->attributes.Max() );
    Mu = 1.61148e6;

    PWConstCoefficient mu_func( Mu );

    Vector Lambda( mesh->attributes.Max() );
    Lambda = 499.92568e6;
    PWConstCoefficient lambda_func( Lambda );

    NeoHookeanMaterial nh( mu_func, lambda_func, NeoHookeanType::Poly1 );

    auto intg = new plugin::NonlinearElasticityIntegrator( nh );

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
    // Set up the Jacobian solver
    auto* j_gmres = new KLUSolver();

    auto newton_solver = new NewtonSolver();

    // Set the newton solve parameters
    newton_solver->iterative_mode = true;
    newton_solver->SetSolver( *j_gmres );
    newton_solver->SetOperator( *nlf );
    newton_solver->SetPrintLevel( -1 );
    newton_solver->SetMonitor( newton_monitor );
    newton_solver->SetRelTol( 1e-7 );
    newton_solver->SetAbsTol( 1e-8 );
    newton_solver->SetMaxIter( 20 );

    nlf->AddBdrFaceIntegrator( new plugin::NonlinearVectorBoundaryLFIntegrator( f ) );
    int steps = 10;
    for ( int i = 1; i <= steps; i++ )
    {
        Vector push_force( mesh->bdr_attributes.Max() );
        push_force = 0.0;
        push_force( 5 ) = -3e6 / steps * i;
        f.Set( 2, new PWConstCoefficient( push_force ) );
        Vector zero;
        newton_solver->Mult( zero, x_gf );
    }
    // MFEM_VERIFY( newton_solver->GetConverged(), "Newton Solver did not converge." );
    subtract( x_gf, x_ref, x_def );

    // 15. Save data in the ParaView format
    ParaViewDataCollection paraview_dc( "test", mesh );
    paraview_dc.SetPrefixPath( "ParaView" );
    paraview_dc.SetLevelsOfDetail( order );
    paraview_dc.SetCycle( 0 );
    paraview_dc.SetDataFormat( VTKFormat::BINARY );
    paraview_dc.SetHighOrderOutput( true );
    paraview_dc.SetTime( 0.0 ); // set the time
    paraview_dc.RegisterField( "Displace", &x_def );
    paraview_dc.Save();
    if ( fec )
    {
        delete fespace;
        delete fec;
    }
    delete newton_solver;
    delete mesh;

    return 0;
}
