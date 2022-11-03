
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
    const char* mesh_file = "../../data/clamp_plate.msh";
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

    // 6. Determine the list of true (i.e. conforming) essential boundary dofs.
    //    In this example, the boundary conditions are defined by marking only
    //    boundary attribute 1 from the mesh as essential and converting it to a
    //    list of true dofs.
    Array<int> ess_tdof_list, ess_bdr( mesh->bdr_attributes.Max() ), temp_list;
    ess_bdr = 0;
    ess_bdr[20] = 1; // left
    fespace->GetEssentialTrueDofs( ess_bdr, temp_list );
    ess_tdof_list.Append( temp_list );

    ess_bdr = 0;
    ess_bdr[21] = 1; // right
    fespace->GetEssentialTrueDofs( ess_bdr, temp_list );
    ess_tdof_list.Append( temp_list );

    ess_bdr = 0;
    ess_bdr[22] = 1; // front
    fespace->GetEssentialTrueDofs( ess_bdr, temp_list );
    ess_tdof_list.Append( temp_list );

    ess_bdr = 0;
    ess_bdr[23] = 1; // back
    fespace->GetEssentialTrueDofs( ess_bdr, temp_list );
    ess_tdof_list.Append( temp_list );

    printf( "Mesh is %i dimensional.\n", dim );
    printf( "Number of mesh attributes: %i\n", mesh->attributes.Size() );
    printf( "Number of mesh elements: %i\n", mesh->GetNE() );
    printf( "Number of mesh vertices: %i\n", mesh->GetNV() );
    printf( "Number of boundary attributes: %i\n", mesh->bdr_attributes.Size() );
    printf( "Max of mesh attributes: %i\n", mesh->attributes.Max() );
    printf( "Max of boundary attributes: %i\n", mesh->bdr_attributes.Max() );

    Vector Nu( mesh->attributes.Max() );
    Nu = .28;
    PWConstCoefficient nu_func( Nu );

    Vector E( mesh->attributes.Max() );
    E = 2.1e11;
    PWConstCoefficient E_func( E );

    IsotropicElasticMaterial iem( E_func, nu_func );

    plugin::Memorize mm( mesh );
    // auto intg = new plugin::NonlinearCompositeSolidShellIntegrator( iem );
    
    auto intg = new plugin::NonlinearElasticityIntegrator( iem,mm );
    NonlinearForm* nlf = new NonlinearForm( fespace );
    nlf->AddDomainIntegrator( intg );
    nlf->SetEssentialTrueDofs( ess_tdof_list );

    GeneralResidualMonitor newton_monitor( "Newton", 1 );
    GeneralResidualMonitor j_monitor( "GMRES", 3 );

    // Set up the Jacobian solver
    auto j_gmres = new UMFPackSolver();

    auto newton_solver = new plugin::MultiNewtonAdaptive();

    // Set the newton solve parameters
    newton_solver->iterative_mode = true;
    newton_solver->SetSolver( *j_gmres );
    newton_solver->SetOperator( *nlf );
    newton_solver->SetPrintLevel( -1 );
    newton_solver->SetMonitor( newton_monitor );
    newton_solver->SetRelTol( 1e-5 );
    // newton_solver->SetAbsTol( 1e11 );
    newton_solver->SetMaxIter( 1 );
    newton_solver->SetPrintLevel( 0 );
    newton_solver->SetDelta( 1 );
    newton_solver->SetMaxStep( 1 );
    GridFunction X( fespace );
    X = 0.;

    VectorArrayCoefficient f1( dim );
    for ( int i = 0; i < dim; i++ )
    {
        f1.Set( i, new ConstantCoefficient( 0.0 ) );
    }
    Vector bottom_force( mesh->bdr_attributes.Max() );
    bottom_force = .0;
    bottom_force( 24 ) = -800;
    f1.Set( 2, new PWConstCoefficient( bottom_force ) );
    nlf->AddBdrFaceIntegrator( new plugin::NonlinearVectorBoundaryLFIntegrator( f1 ) );

    ParaViewDataCollection paraview_dc( "plate", mesh );
    paraview_dc.SetPrefixPath( "ParaView" );
    paraview_dc.SetLevelsOfDetail( order );
    paraview_dc.SetCycle( 0 );
    paraview_dc.SetFormat( 1 );
    paraview_dc.SetDataFormat( VTKFormat::BINARY );
    paraview_dc.SetTime( 0.0 ); // set the time
    paraview_dc.RegisterField( "Displace", &X );
    newton_solver->SetDataCollection( &paraview_dc );
    paraview_dc.Save();

    Vector zero;
    newton_solver->Mult( zero, X );
    if ( fec )
    {
        delete fespace;
        delete fec;
    }
    delete newton_solver;
    delete mesh;

    return 0;
}
