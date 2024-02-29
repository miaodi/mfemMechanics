
#include "Plugin.h"
#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <unistd.h>

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
        if ( it > 1 )
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
    // 1. Initialize MPI.
    int num_procs, myid;
    MPI_Init( &argc, &argv );
    MPI_Comm_size( MPI_COMM_WORLD, &num_procs );
    MPI_Comm_rank( MPI_COMM_WORLD, &myid );

    // 1. Parse command-line options.
    const char* mesh_file = "../../data/DCB.msh";
    int order = 1;
    bool static_cond = false;
    bool visualization = 1;
    int ser_ref_levels = -1, par_ref_levels = -1;

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
    args.Parse();
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

    // 2. Read the mesh from the given mesh file. We can handle triangular,
    //    quadrilateral, tetrahedral or hexahedral elements with the same code.
    Mesh* mesh = new Mesh( mesh_file, 1, 1 );
    int dim = mesh->Dimension();

    // 3. Select the order of the finite element discretization space. For NURBS
    //    meshes, we increase the order by degree elevation.
    if ( mesh->bdr_attributes.Max() < 2 )
    {
        if ( myid == 0 )
            cerr << "\nInput mesh should have at least "
                 << "two boundary attributes! (See schematic in ex2.cpp)\n"
                 << endl;
        MPI_Finalize();
        return 3;
    }

    {
        int ref_levels = ser_ref_levels >= 0 ? ser_ref_levels : (int)floor( log( 1000. / mesh->GetNE() ) / log( 2. ) / dim );
        for ( int l = 0; l < ref_levels; l++ )
        {
            mesh->UniformRefinement();
        }
    }
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

    if ( myid == 0 )
    {
        printf( "Mesh is %i dimensional.\n", dim );
        printf( "Number of mesh attributes: %i\n", pmesh->attributes.Size() );
        printf( "Number of boundary attributes: %i\n", pmesh->bdr_attributes.Size() );
        printf( "Max of boundary attributes: %i\n", pmesh->bdr_attributes.Max() );
    }

    // 5. Define a finite element space on the mesh. Here we use vector finite
    //    elements, i.e. dim copies of a scalar finite element space. The vector
    //    dimension is specified by the last argument of the FiniteElementSpace
    //    constructor. For NURBS meshes, we use the (degree elevated) NURBS space
    //    associated with the mesh nodes.
    FiniteElementCollection* fec;
    ParFiniteElementSpace* fespace;
    fec = new DG_FECollection( order, dim, mfem::BasisType::GaussLobatto );
    fespace = new ParFiniteElementSpace( pmesh, fec, dim, Ordering::byVDIM );
    HYPRE_BigInt size = fespace->GlobalTrueVSize();

    if ( myid == 0 )
    {
        cout << "Number of vertices: " << fespace->GetNV() << endl;
        cout << "Number of finite element unknowns: " << fespace->GetTrueVSize() << endl << "Assembling: " << endl;
    }

    // 6. Determine the list of true (i.e. conforming) essential boundary dofs.
    //    In this example, the boundary conditions are defined by marking only
    //    boundary attribute 1 from the mesh as essential and converting it to a
    //    list of true dofs.
    VectorArrayCoefficient d( dim );
    for ( int i = 0; i < dim; i++ )
    {
        d.Set( i, new ConstantCoefficient( 0.0 ) );
    }

    Vector topDisp( pmesh->bdr_attributes.Max() );
    topDisp = .0;
    topDisp( 11 ) = 4e-2;
    topDisp( 12 ) = -4e-2;
    d.Set( 1, new PWConstCoefficient( topDisp ) );

    Vector activeBCX( pmesh->bdr_attributes.Max() );
    activeBCX = 0.0;
    activeBCX( 10 ) = 1e15;
    Vector activeBCY( pmesh->bdr_attributes.Max() );
    activeBCY = 0.0;
    activeBCY( 10 ) = 1e15;
    activeBCY( 11 ) = 1e15;
    activeBCY( 12 ) = 1e15;

    VectorArrayCoefficient hevi( dim );
    hevi.Set( 0, new PWConstCoefficient( activeBCX ) );
    hevi.Set( 1, new PWConstCoefficient( activeBCY ) );
    // 8. Define the solution vector x as a finite element grid function
    //    corresponding to fespace. Initialize x with initial guess of zero,
    //    which satisfies the boundary conditions.
    Vector Nu( pmesh->attributes.Max() );
    Nu = .0;
    PWConstCoefficient nu_func( Nu );

    Vector E( pmesh->attributes.Max() );
    E = 324E7;
    PWConstCoefficient E_func( E );

    IsotropicElasticMaterial iem( E_func, nu_func );

    plugin::Memorize mm( pmesh );

    auto intg = new plugin::NonlinearElasticityIntegrator( iem, mm );
    intg->setNonlinear( false );

    ParNonlinearForm* nlf = new ParNonlinearForm( fespace );
    nlf->AddDomainIntegrator( intg );
    nlf->AddBdrFaceIntegrator( new plugin::NonlinearDirichletPenaltyIntegrator( d, hevi ) );

    GeneralResidualMonitor newton_monitor( fespace->GetComm(), "Newton", 1 );

    // Set up the Jacobian solver
    mfem::Solver* lin_solver{ nullptr };
    
    // {
    //     auto cg  = new mfem::CGSolver( MPI_COMM_WORLD );
    //     lin_solver = cg;
    //     // gmres->SetPrintLevel( -1 );
    //     cg->SetRelTol( 1e-11 );
    //     cg->SetMaxIter( 200000 );
    //     cg->SetPrintLevel(0);

    //     // mfem::HypreBoomerAMG* prec = new mfem::HypreBoomerAMG();
    //     // prec->SetPrintLevel(0);
    //     // cg->SetPreconditioner( *prec );
    // }
    {
        auto mumps = new mfem::MUMPSSolver( MPI_COMM_WORLD );
        mumps->SetMatrixSymType( MUMPSSolver::MatType::SYMMETRIC_INDEFINITE );
        // mumps->SetReorderingStrategy( MUMPSSolver::ReorderingStrategy::PARMETIS );
        mumps->SetPrintLevel( -1 );
        lin_solver = mumps;
    }

    auto newton_solver = new plugin::Crisfield( fespace->GetComm() );
    // Set the newton solve parameters
    newton_solver->iterative_mode = true;
    newton_solver->SetSolver( *lin_solver );
    newton_solver->SetOperator( *nlf );
    newton_solver->SetPrintLevel( -1 );
    newton_solver->SetMonitor( newton_monitor );
    newton_solver->SetRelTol( 1e-8 );
    newton_solver->SetAbsTol( 0 );
    newton_solver->SetMaxIter( 100 );
    newton_solver->SetPrintLevel( 0 );
    newton_solver->SetDelta( .00001 );
    newton_solver->SetPhi( 10 );
    newton_solver->SetMaxDelta( 1e-6 );
    newton_solver->SetMinDelta( 1e-12 );
    newton_solver->SetMaxStep( 200000 );

    nlf->AddInteriorFaceIntegrator( new plugin::NonlinearInternalPenaltyIntegrator( 1e15 ) );
    nlf->AddInteriorFaceIntegrator( new plugin::ExponentialCZMIntegrator( mm, 324E5, 755.4E5, 4E-7, 4E-7 ) );
    // nlf->AddInteriorFaceIntegrator( new plugin::LinearCZMIntegrator( .257E-3, 1E-6, 48E-6, 324E7 ) );

    Vector zero;

    GridFunction u( fespace );
    u = 0.;

    // nlf->AddBdrFaceIntegrator( new plugin::NonlinearVectorBoundaryLFIntegrator( f ) );
    // 15. Save data in the ParaView format
    // ParaViewDataCollection paraview_dc( "czm2Dp", pmesh );
    // paraview_dc.SetPrefixPath( "ParaView" );
    // paraview_dc.SetLevelsOfDetail( order );
    // paraview_dc.SetCycle( 0 );
    // paraview_dc.SetDataFormat( VTKFormat::BINARY );
    // paraview_dc.SetHighOrderOutput( true );
    // paraview_dc.SetTime( 0.0 ); // set the time
    // paraview_dc.RegisterField( "Displace", &u );
    // newton_solver->SetDataCollection( &paraview_dc );
    // paraview_dc.Save();

    newton_solver->Mult( zero, u );
    MPI_Finalize();
    return 0;
}
