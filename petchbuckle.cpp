
#include "Plugin.h"
#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

class GeneralResidualMonitor : public IterativeSolverMonitor
{
public:
    GeneralResidualMonitor( MPI_Comm comm, const std::string& prefix_, int print_lvl ) : prefix( prefix_ )
    {
#ifndef MFEM_USE_MPI
        print_level = print_lvl;
#else
        int rank;
        MPI_Comm_rank( comm, &rank );
        if ( rank == 0 )
        {
            print_level = print_lvl;
        }
        else
        {
            print_level = -1;
        }
#endif
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
    // 1. Initialize MPI.
    int num_procs, myid;
    MPI_Init( &argc, &argv );
    MPI_Comm_size( MPI_COMM_WORLD, &num_procs );
    MPI_Comm_rank( MPI_COMM_WORLD, &myid );

    // 1. Parse command-line options.
    const char* mesh_file = "/home/miaodi/repo/mfem_test/3DBeamBuckle.msh";
    int order = 1;
    bool static_cond = false;
    bool visualization = 1;
    int ser_ref_levels = -1, par_ref_levels = -1;
    double f_temp = 100.;
    const char* petscrc_file = "";

    OptionsParser args( argc, argv );
    args.AddOption( &mesh_file, "-m", "--mesh", "Mesh file to use." );
    args.AddOption( &order, "-o", "--order", "Finite element order (polynomial degree)." );
    args.AddOption( &static_cond, "-sc", "--static-condensation", "-no-sc", "--no-static-condensation",
                    "Enable static condensation." );
    args.AddOption( &visualization, "-vis", "--visualization", "-no-vis", "--no-visualization",
                    "Enable or disable GLVis visualization." );
    args.AddOption( &ser_ref_levels, "-rs", "--refine-serial",
                    "Number of times to refine the mesh uniformly in serial." );
    args.AddOption( &par_ref_levels, "-rp", "--refine-parallel",
                    "Number of times to refine the mesh uniformly in parallel." );
    args.AddOption( &petscrc_file, "-petscopts", "--petscopts", "PetscOptions file to use." );
    args.AddOption( &f_temp, "-ft", "--ftemp", "Final temperature." );
    args.Parse();
    if ( !args.Good() )
    {
        if ( myid == 0 )
        {
            args.PrintUsage( cout );
        }
        MPI_Finalize();
        return 1;
    }
    if ( myid == 0 )
    {
        args.PrintOptions( cout );
    }

    MFEMInitializePetsc( NULL, NULL, petscrc_file, NULL );

    // 2. Read the mesh from the given mesh file. We can handle triangular,
    //    quadrilateral, tetrahedral or hexahedral elements with the same code.
    Mesh* mesh = new Mesh( mesh_file, 1, 1 );
    int dim = mesh->Dimension();

    if ( mesh->bdr_attributes.Max() < 2 )
    {
        if ( myid == 0 )
            cerr << "\nInput mesh should have at least "
                 << "two boundary attributes! (See schematic in ex2.cpp)\n"
                 << endl;
        MPI_Finalize();
        return 3;
    }

    // {
    //     int ref_levels = ser_ref_levels >= 0 ? ser_ref_levels : (int)floor( log( 1000. / mesh->GetNE() ) / log( 2. )
    //     / dim ); for ( int l = 0; l < ref_levels; l++ )
    //     {
    //         mesh->UniformRefinement();
    //     }
    // }

    // 6. Define a parallel mesh by a partitioning of the serial mesh. Refine
    //    this mesh further in parallel to increase the resolution. Once the
    //    parallel mesh is defined, the serial mesh can be deleted.
    ParMesh* pmesh = new ParMesh( MPI_COMM_WORLD, *mesh );
    delete mesh;
    {
        for ( int l = 0; l < par_ref_levels; l++ )
        {
            pmesh->UniformRefinement();
        }
    }

    // 5. Define a finite element space on the mesh. Here we use vector finite
    //    elements, i.e. dim copies of a scalar finite element space. The vector
    //    dimension is specified by the last argument of the FiniteElementSpace
    //    constructor. For NURBS meshes, we use the (degree elevated) NURBS space
    //    associated with the mesh nodes.
    FiniteElementCollection* fec;
    ParFiniteElementSpace* fespace;
    fec = new H1_FECollection( order, dim );
    fespace = new ParFiniteElementSpace( pmesh, fec, dim, Ordering::byVDIM );
    HYPRE_BigInt size = fespace->GlobalTrueVSize();
    if ( myid == 0 )
    {
        cout << "Number of finite element unknowns: " << size << endl << "Assembling: " << endl;
    }

    // 6. Determine the list of true (i.e. conforming) essential boundary dofs.
    //    In this example, the boundary conditions are defined by marking only
    //    boundary attribute 1 from the mesh as essential and converting it to a
    //    list of true dofs.
    Array<int> ess_tdof_list, ess_bdr( pmesh->bdr_attributes.Max() ), temp_list;
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

    if ( myid == 0 )
    {
        printf( "Mesh is %i dimensional.\n", dim );
        printf( "Number of mesh attributes: %i\n", pmesh->attributes.Size() );
        printf( "Number of boundary attributes: %i\n", pmesh->bdr_attributes.Size() );
        printf( "Max of mesh attributes: %i\n", pmesh->attributes.Max() );
        printf( "Max of boundary attributes: %i\n", pmesh->bdr_attributes.Max() );
    }
    // 8. Define the solution vector x as a finite element grid function
    //    corresponding to fespace. Initialize x with initial guess of zero,
    //    which satisfies the boundary conditions.
    ParGridFunction x_gf( fespace );
    ParGridFunction x_ref( fespace );
    ParGridFunction x_def( fespace );

    VectorFunctionCoefficient refconfig( dim, ReferenceConfiguration );

    x_gf.ProjectCoefficient( refconfig );
    x_ref.ProjectCoefficient( refconfig );

    Vector Nu( pmesh->attributes.Max() );
    Nu = .3;
    PWConstCoefficient nu_func( Nu );

    Vector E( pmesh->attributes.Max() );
    E = 12.8e9;
    PWConstCoefficient E_func( E );

    Vector CTE( pmesh->attributes.Max() );
    CTE = 23.1e-6;
    PWConstCoefficient CTE_func( CTE );

    IsotropicElasticThermalMaterial ietm( E_func, nu_func, CTE_func );
    ietm.setInitialTemp( 0 );
    ietm.setFinalTemp( f_temp );

    plugin::Memorize mm( pmesh );

    auto intg = new plugin::NonlinearElasticityIntegrator( ietm, mm );
    intg->setNonlinear( true );
    auto* nlf = new ParNonlinearForm( fespace );
    nlf->AddDomainIntegrator( intg );
    nlf->SetEssentialTrueDofs( ess_tdof_list );
    nlf->SetGradientType( Operator::Type::PETSC_MATAIJ );

    GeneralResidualMonitor newton_monitor( fespace->GetComm(), "Newton", 1 );
    GeneralResidualMonitor j_monitor( fespace->GetComm(), "GMRES", 3 );

    // {
    // Vector r( fespace->GetTrueVSize() );
    // Vector q( fespace->GetTrueVSize() );
    // Vector X( fespace->GetTrueVSize() );
    // plugin::SetLambdaToIntegrators( nlf, 1. + 0 );
    // x_gf.ParallelProject( X );
    // nlf->Mult( X, q );
    // plugin::SetLambdaToIntegrators( nlf, 0 );
    // nlf->Mult( X, r );
    // // q.Print();
    // mfem::out << "x: " << InnerProduct( fespace->GetComm(), X, X ) << " q: " << InnerProduct( fespace->GetComm(), q, q )
    //           << " r: " << InnerProduct( fespace->GetComm(), r, r ) << "\n";
    // }

    // Set up the Jacobian solver
    PetscLinearSolver* petsc = new PetscLinearSolver( fespace->GetComm() );

    auto newton_solver = new plugin::Crisfield( fespace->GetComm() );

    // Set the newton solve parameters
    newton_solver->iterative_mode = true;
    newton_solver->SetSolver( *petsc );
    newton_solver->SetOperator( *nlf );
    newton_solver->SetPrintLevel( -1 );
    newton_solver->SetMonitor( newton_monitor );
    newton_solver->SetRelTol( 1e-6 );
    newton_solver->SetAbsTol( 1e-10 );
    newton_solver->SetMaxIter( 6 );
    newton_solver->SetDelta( .1 );
    newton_solver->SetMaxDelta( 25 );
    newton_solver->SetMinDelta( 1e-5 );
    newton_solver->SetPhi( .0 );
    newton_solver->SetMaxStep( 10000 );

    Vector X( fespace->GetTrueVSize() );
    x_gf.ParallelProject( X );
    Vector zero;
    newton_solver->Mult( zero, X );
    x_gf.Distribute( X );
    subtract( x_gf, x_ref, x_def );

    // 15. Save data in the ParaView format
    ParaViewDataCollection paraview_dc( "buckling1", pmesh );
    paraview_dc.SetPrefixPath( "ParaView" );
    paraview_dc.SetLevelsOfDetail( order );
    paraview_dc.SetCycle( 0 );
    paraview_dc.SetDataFormat( VTKFormat::BINARY );
    paraview_dc.SetTime( 0.0 ); // set the time
    paraview_dc.RegisterField( "Displace", &x_def );
    paraview_dc.Save();
    if ( fec )
    {
        delete fespace;
        delete fec;
    }
    delete newton_solver;
    delete pmesh;


    MFEMFinalizePetsc();

    MPI_Finalize();
    return 0;
}
