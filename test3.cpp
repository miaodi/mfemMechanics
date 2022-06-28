
#include "FEMPlugin.h"
#include "Material.h"
#include "NeoHookeanMaterial.h"
#include "Solvers.h"
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
    const char* mesh_file = "/home/miaodi/repo/mfem_test/3DBeamBuckle.msh";
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
    for ( int l = 0; l < refineLvl; l++ )
    {
        mesh->UniformRefinement();
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
        fespace = new FiniteElementSpace( mesh, fec, dim );
    }
    cout << "Number of finite element unknowns: " << fespace->GetTrueVSize() << endl << "Assembling: " << flush;

    
    // 6. Determine the list of true (i.e. conforming) essential boundary dofs.
    //    In this example, the boundary conditions are defined by marking only
    //    boundary attribute 1 from the mesh as essential and converting it to a
    //    list of true dofs.
    Array<int> ess_tdof_list, ess_bdr( mesh->bdr_attributes.Max() ), temp_list;
    ess_bdr = 0;
    ess_bdr[28] = 1; // left
    fespace->GetEssentialTrueDofs( ess_bdr, temp_list, 0 );
    ess_tdof_list.Append( temp_list );

    ess_bdr = 0;
    ess_bdr[29] = 1; // right
    fespace->GetEssentialTrueDofs( ess_bdr, temp_list, 0 );
    ess_tdof_list.Append( temp_list );

    ess_bdr = 0;
    ess_bdr[27] = 1; // bottom
    fespace->GetEssentialTrueDofs( ess_bdr, temp_list, 1 );
    ess_tdof_list.Append( temp_list );
    fespace->GetEssentialTrueDofs( ess_bdr, temp_list, 2 );
    ess_tdof_list.Append( temp_list );

    printf( "Mesh is %i dimensional.\n", dim );
    printf( "Number of mesh attributes: %i\n", mesh->attributes.Size() );
    printf( "Number of boundary attributes: %i\n", mesh->bdr_attributes.Size() );
    printf( "Max of mesh attributes: %i\n", mesh->attributes.Max() );
    printf( "Max of boundary attributes: %i\n", mesh->bdr_attributes.Max() );

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

    Vector Nu( mesh->attributes.Max() );
    Nu = .3;
    PWConstCoefficient nu_func( Nu );

    Vector E( mesh->attributes.Max() );
    E = 12.8e9;
    PWConstCoefficient E_func( E );

    Vector CTE( mesh->attributes.Max() );
    CTE = 23.1e-6;
    PWConstCoefficient CTE_func( CTE );

    IsotropicElasticThermalMaterial ietm( E_func, nu_func, CTE_func );
    ietm.setInitialTemp( 0 );
    ietm.setFinalTemp( 2000 );

    plugin::Memorize mm( mesh );

    auto intg = new plugin::NonlinearElasticityIntegrator( ietm, mm );
    NonlinearForm* nlf = new NonlinearForm( fespace );
    intg->setNonlinear( true );
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
    auto j_gmres = new KLUSolver();

    auto newton_solver = new plugin::Crisfield();

    // Set the newton solve parameters
    newton_solver->iterative_mode = true;
    newton_solver->SetSolver( *j_gmres );
    newton_solver->SetOperator( *nlf );
    newton_solver->SetPrintLevel( -1 );
    newton_solver->SetMonitor( newton_monitor );
    newton_solver->SetRelTol( 1e-6 );
    newton_solver->SetAbsTol( 1e-10 );
    newton_solver->SetMaxIter( 6 );
    newton_solver->SetDelta( .01 );
    newton_solver->SetMaxDelta( 10 );
    newton_solver->SetMinDelta( 1e-5 );
    newton_solver->SetPhi( .0 );
    newton_solver->SetMaxStep( 10000 );

    Vector zero;
    newton_solver->Mult( zero, x_gf );

    // MFEM_VERIFY( newton_solver->GetConverged(), "Newton Solver did not converge." );
    subtract( x_gf, x_ref, x_def );

    // 15. Save data in the ParaView format
    ParaViewDataCollection paraview_dc( "beamBuckle", mesh );
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
