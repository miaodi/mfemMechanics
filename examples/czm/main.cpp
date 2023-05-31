
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

int main( int argc, char* argv[] )
{
    // 1. Parse command-line options.
    const char* mesh_file = "../../data/crack_square2d.msh";
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
                if ( node[1] > -0.00000001 && node[1] < 0.0008 && node[0] > -0.00000001 && node[0] < .0006 )
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
        fec = new DG_FECollection( order, dim, mfem::BasisType::GaussLobatto );
        fespace = new FiniteElementSpace( mesh, fec, dim );
    }
    cout << "Number of vertices: " << fespace->GetNV() << endl;
    cout << "Number of finite element unknowns: " << fespace->GetTrueVSize() << endl << "Assembling: " << endl;
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
    VectorArrayCoefficient d( dim );
    for ( int i = 0; i < dim; i++ )
    {
        d.Set( i, new ConstantCoefficient( 0.0 ) );
    }
    Vector topDisp( mesh->bdr_attributes.Max() );
    topDisp = .0;
    topDisp( 10 ) = 2e-3;
    topDisp( 11 ) = 0;
    d.Set( 0, new PWConstCoefficient( topDisp ) );

    Vector activeBC( mesh->bdr_attributes.Max() );
    activeBC = 0.0;
    activeBC( 10 ) = 1e17;
    activeBC( 11 ) = 1e17;
    VectorArrayCoefficient hevi( dim );
    for ( int i = 0; i < dim; i++ )
    {
        hevi.Set( i, new PWConstCoefficient( activeBC ) );
    }

    printf( "Mesh is %i dimensional.\n", dim );
    printf( "Number of mesh attributes: %i\n", mesh->attributes.Size() );
    printf( "Number of boundary attributes: %i\n", mesh->bdr_attributes.Size() );

    // 8. Define the solution vector x as a finite element grid function
    //    corresponding to fespace. Initialize x with initial guess of zero,
    //    which satisfies the boundary conditions.
    Vector Nu( mesh->attributes.Max() );
    Nu = .0;
    PWConstCoefficient nu_func( Nu );

    Vector E( mesh->attributes.Max() );
    E = 324e7;
    PWConstCoefficient E_func( E );

    IsotropicElasticMaterial iem( E_func, nu_func );

    plugin::Memorize mm( mesh );

    auto intg = new plugin::NonlinearElasticityIntegrator( iem, mm );
    intg->setNonlinear( false );

    NonlinearForm* nlf = new NonlinearForm( fespace );
    nlf->AddDomainIntegrator( intg );

    nlf->AddBdrFaceIntegrator( new plugin::NonlinearDirichletPenaltyIntegrator( d, hevi ) );

    GeneralResidualMonitor newton_monitor( "Newton", 1 );
    GeneralResidualMonitor j_monitor( "GMRES", 3 );

    // Set up the Jacobian solver
    auto j_gmres = new UMFPackSolver();

    auto newton_solver = new plugin::ArcLengthLinearize();

    // Set the newton solve parameters
    newton_solver->iterative_mode = true;
    newton_solver->SetSolver( *j_gmres );
    newton_solver->SetOperator( *nlf );
    newton_solver->SetPrintLevel( -1 );
    newton_solver->SetMonitor( newton_monitor );
    newton_solver->SetRelTol( 1e-7 );
    newton_solver->SetAbsTol( 1e-14 );
    newton_solver->SetMaxIter( 13 );
    newton_solver->SetPrintLevel( 0 );
    newton_solver->SetDelta( .0001 );
    newton_solver->SetPhi( 0 );
    newton_solver->SetMaxDelta( 1e-2 );
    newton_solver->SetMinDelta( 1e-14 );
    newton_solver->SetMaxStep( 10000 );

    // nlf->AddInteriorFaceIntegrator( new plugin::NonlinearInternalPenaltyIntegrator( 1e14 ) );
    auto czm_intg = new plugin::ExponentialRotADCZMIntegrator( mm, 324E6, 755.4E6, 1E-5, 1E-5 );
    nlf->AddInteriorFaceIntegrator( czm_intg );
    mfem::IntegrationRules GLIntRules( 0, mfem::Quadrature1D::GaussLobatto );
    czm_intg->SetIntRule( &GLIntRules.Get( mfem::Geometry::SQUARE, -1 ) );
    // nlf->AddInteriorFaceIntegrator( new plugin::LinearCZMIntegrator( .257E-3, 1E-6, 48E-6, 324E7 ) );

    Vector zero;

    GridFunction u( fespace );
    u = 0.;
    // VectorArrayCoefficient f( dim );
    // for ( int i = 0; i < dim - 1; i++ )
    // {
    //     f.Set( i, new ConstantCoefficient( 0.0 ) );
    // }
    // {
    //     Vector pull_force( mesh->bdr_attributes.Max() );
    //     pull_force = 0.0;
    //     pull_force( 13 ) = -0.e8;
    //     pull_force( 14 ) = 1.e8;
    //     f.Set( dim - 1, new PWConstCoefficient( pull_force ) );
    // }
    // nlf->AddBdrFaceIntegrator( new plugin::NonlinearVectorBoundaryLFIntegrator( f ) );

    // 15. Save data in the ParaView format
    ParaViewDataCollection paraview_dc( "czm_square_test", mesh );
    paraview_dc.SetPrefixPath( "ParaView" );
    paraview_dc.SetLevelsOfDetail( order );
    paraview_dc.SetCycle( 0 );
    paraview_dc.SetDataFormat( VTKFormat::BINARY );
    paraview_dc.SetHighOrderOutput( true );
    paraview_dc.SetTime( 0.0 ); // set the time
    paraview_dc.RegisterField( "Displace", &u );

    auto stress_fec = new H1_FECollection( order, dim );

    GridFunction ue( fespace );
    ue = 0.;
    auto stress_fespace = new FiniteElementSpace( mesh, stress_fec, 7 );
    GridFunction stress_grid( stress_fespace );
    plugin::StressCoefficient sc( dim, iem );
    sc.SetDisplacement( ue );
    paraview_dc.RegisterField( "Stress", &stress_grid );
    stress_grid.ProjectCoefficient( sc );
    paraview_dc.Save();

    std::function<void( int, int, double )> func = [&paraview_dc, &stress_grid, &sc]( int step, int count, double time )
    {
        if ( step % 2 == 0 )
        {
            paraview_dc.SetCycle( count );
            paraview_dc.SetTime( count );
            stress_grid.ProjectCoefficient( sc );
            paraview_dc.Save();
        }
    };

    // adaptive refinement
    plugin::CriticalVMRefiner refiner( sc );
    refiner.SetCriticalVM( 1e7 );
    refiner.SetCriticalH( 1e-2 );

    std::function<bool( const Vector& )> func2 =
        [&ue, mesh, &refiner, fespace, &u, nlf, &mm, stress_fespace, &stress_grid]( const Vector& du )
    {
        add( u, du, ue );
        refiner.Apply( *mesh );
        if ( refiner.Stop() )
        {
            return true;
        }
        fespace->Update();
        stress_fespace->Update();
        u.Update();
        nlf->Update();
        ue.Update();
        stress_grid.Update();
        mm.Reset( mesh );
        return false;
    };

    newton_solver->SetDataCollectionFunc( func );
    newton_solver->SetAMRFunc( func2 );
    newton_solver->Mult( zero, u );

    return 0;
}
