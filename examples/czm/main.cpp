
#include "Plugin.h"
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
        fec = new L2_FECollection( order, dim );
        fespace = new FiniteElementSpace( mesh, fec, dim );
    }
    cout << "Number of finite element unknowns: " << fespace->GetTrueVSize() << endl << "Assembling: " << flush;
    mfem::Array<int> vdofs;
    fespace->GetElementVDofs( 0, vdofs );
    cout << "vdofs: " << vdofs.Size() << endl;
    vdofs.Print();
    fespace->GetElementVDofs( 1, vdofs );
    vdofs.Print();

    // 6. Determine the list of true (i.e. conforming) essential boundary dofs.
    //    In this example, the boundary conditions are defined by marking only
    //    boundary attribute 1 from the mesh as essential and converting it to a
    //    list of true dofs.
    Array<int> ess_tdof_list, ess_bdr( mesh->bdr_attributes.Max() );
    ess_bdr = 0;
    ess_bdr[0] = 1;
    ess_bdr[1] = 1;
    fespace->GetEssentialTrueDofs( ess_bdr, ess_tdof_list );

    printf( "Mesh is %i dimensional.\n", dim );
    printf( "Number of mesh attributes: %i\n", mesh->attributes.Size() );
    printf( "Number of boundary attributes: %i\n", mesh->bdr_attributes.Size() );

    // 8. Define the solution vector x as a finite element grid function
    //    corresponding to fespace. Initialize x with initial guess of zero,
    //    which satisfies the boundary conditions.
    Vector Nu( mesh->attributes.Max() );
    Nu = .3;
    PWConstCoefficient nu_func( Nu );

    Vector E( mesh->attributes.Max() );
    E = 210e9;
    PWConstCoefficient E_func( E );

    IsotropicElasticMaterial iem( E_func, nu_func );

    plugin::Memorize mm( mesh );

    auto intg = new plugin::NonlinearElasticityIntegrator( iem, mm );

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

    auto newton_solver = new plugin::MultiNewtonAdaptive();

    // Set the newton solve parameters
    newton_solver->iterative_mode = true;
    newton_solver->SetSolver( *j_gmres );
    newton_solver->SetOperator( *nlf );
    newton_solver->SetPrintLevel( -1 );
    newton_solver->SetMonitor( newton_monitor );
    newton_solver->SetRelTol( 1e-7 );
    newton_solver->SetAbsTol( 1e-8 );
    newton_solver->SetMaxIter( 2 );
    newton_solver->SetDelta( .1 );
    newton_solver->SetMaxStep( 1 );

    plugin::CZMIntegrator czm_intg( 324e6, 755.4e6, 352.3, 352.3, 4e-7, 4e-7 );
    nlf->AddInteriorFaceIntegrator( &czm_intg );

    Vector zero;

    GridFunction u( fespace );
    u = 0.;

    Array<int> bdr2( mesh->bdr_attributes.Max() );
    bdr2 = 0;
    bdr2[1] = 1;

    Vector vec( 2 );
    vec( 1 ) = .01;
    VectorConstantCoefficient vcc( vec );
    u.ProjectBdrCoefficient( vcc, bdr2 );
    // newton_solver->Mult( zero, u );

    // // 15. Save data in the ParaView format
    // ParaViewDataCollection paraview_dc( "test2", mesh );
    // paraview_dc.SetPrefixPath( "ParaView" );
    // paraview_dc.SetLevelsOfDetail( order );
    // paraview_dc.SetCycle( 0 );
    // paraview_dc.SetDataFormat( VTKFormat::BINARY );
    // paraview_dc.SetHighOrderOutput( true );
    // paraview_dc.SetTime( 0.0 ); // set the time
    // paraview_dc.RegisterField( "Displace", &u );
    // paraview_dc.Save();
    // if ( fec )
    // {
    //     delete fespace;
    //     delete fec;
    // }
    // delete newton_solver;
    // delete mesh;

    return 0;
}
