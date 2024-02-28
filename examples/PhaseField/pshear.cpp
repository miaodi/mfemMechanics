
#include "PhaseField.h"
#include "Plugin.h"
#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <omp.h>

using namespace std;
using namespace mfem;

double crack_curve_y( const double x )
{
    return -1.69104E-6 - 2.65179 * x + 6455.51 * std::pow( x, 2 ) - 9.11803E6 * std::pow( x, 3 );
}

int main( int argc, char* argv[] )
{
    // 1. Initialize MPI.
    int num_procs, myid;
    Mpi::Init( argc, argv );
    MPI_Comm_size( MPI_COMM_WORLD, &num_procs );
    MPI_Comm_rank( MPI_COMM_WORLD, &myid );
    Hypre::Init();

    // 1. Parse command-line options.
    const char* mesh_file = "../../data/crack_square2d_hex.msh";
    int order = 1;
    bool static_cond = false;
    int ser_ref_levels = -1, par_ref_levels = -1;
    int localRefineLvl = 0;
    const char* petscrc_file = "../../data/petscSetting";

    OptionsParser args( argc, argv );
    args.AddOption( &mesh_file, "-m", "--mesh", "Mesh file to use." );
    args.AddOption( &order, "-o", "--order", "Finite element order (polynomial degree)." );
    args.AddOption( &static_cond, "-sc", "--static-condensation", "-no-sc", "--no-static-condensation",
                    "Enable static condensation." );
    args.AddOption( &ser_ref_levels, "-rs", "--refine-serial",
                    "Number of times to refine the mesh uniformly in serial." );
    args.AddOption( &par_ref_levels, "-rp", "--refine-parallel",
                    "Number of times to refine the mesh uniformly in parallel." );
    args.AddOption( &petscrc_file, "-petscopts", "--petscopts", "PetscOptions file to use." );
    args.AddOption( &localRefineLvl, "-lr", "--local-refine-level", "Finite element local refine level." );
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

    // MFEMInitializePetsc( NULL, NULL, petscrc_file, NULL );

    //  4. Mesh refinement
    for ( int lev = 0; lev < ser_ref_levels; lev++ )
    {
        mesh->UniformRefinement();
    }

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
                if ( node[0] >= -.00005 && node[1] < .00005 &&
                     std::abs( node[1] - crack_curve_y( node[0] ) ) < .00008 * std::pow( 1 + 250 * std::abs( node[1] ), 4 ) )
                {
                    refinements.Append( i );
                    break;
                }
            }
        }
        mesh->GeneralRefinement( refinements );
    }
    // ofstream file;
    // file.open( "refined.vtk" );
    // mesh->PrintVTK( file );
    // file.close();
    // return 0;

    ParMesh* pmesh = new ParMesh( MPI_COMM_WORLD, *mesh );
    delete mesh;
    {
        for ( int l = 0; l < par_ref_levels; l++ )
        {
            pmesh->UniformRefinement();
        }
    }

    // 5. Define the finite element spaces for displacement and pressure
    //    (Taylor-Hood elements). By default, the displacement (u/x) is a second
    //    order vector field, while the pressure (p) is a linear scalar function.
    H1_FECollection lin_coll( order, dim );

    ParFiniteElementSpace R_space( pmesh, &lin_coll, dim, Ordering::byVDIM );
    ParFiniteElementSpace W_space( pmesh, &lin_coll );

    Array<ParFiniteElementSpace*> spaces( 2 );
    spaces[0] = &R_space;
    spaces[1] = &W_space;

    HYPRE_BigInt glob_R_size = R_space.GlobalTrueVSize();
    HYPRE_BigInt glob_W_size = W_space.GlobalTrueVSize();

    // 9. Print the mesh statistics
    if ( myid == 0 )
    {
        std::cout << "***********************************************************\n";
        std::cout << "dim(u) = " << glob_R_size << "\n";
        std::cout << "dim(p) = " << glob_W_size << "\n";
        std::cout << "dim(u+p) = " << glob_R_size + glob_W_size << "\n";
        std::cout << "***********************************************************\n";
    }

    VectorArrayCoefficient d( dim );
    Vector topDisp( R_space.GetMesh()->bdr_attributes.Max() );
    topDisp = .0;
    topDisp( 10 ) = 0;
    topDisp( 11 ) = 1e-4;
    d.Set( 0, new PWConstCoefficient( topDisp ) );

    Vector activeBC( R_space.GetMesh()->bdr_attributes.Max() );
    activeBC = 0.0;
    activeBC( 10 ) = 1e17;
    activeBC( 11 ) = 1e17;
    VectorArrayCoefficient hevi( dim );
    for ( int i = 0; i < dim; i++ )
    {
        hevi.Set( i, new PWConstCoefficient( activeBC ) );
    }

    VectorArrayCoefficient d2( dim );
    Vector sideDisp( R_space.GetMesh()->bdr_attributes.Max() );
    sideDisp = .0;
    d.Set( 1, new PWConstCoefficient( sideDisp ) );

    Vector activeBC2( R_space.GetMesh()->bdr_attributes.Max() );
    activeBC2 = 0.0;
    activeBC2( 12 ) = 1e17;
    activeBC2( 13 ) = 1e17;
    activeBC2( 14 ) = 1e17;
    VectorArrayCoefficient hevi2( dim );
    hevi2.Set( 1, new PWConstCoefficient( activeBC ) );

    //  Define the block structure of the solution vector (u then p)
    Array<int> block_trueOffsets( 3 );
    block_trueOffsets[0] = 0;
    block_trueOffsets[1] = R_space.GetTrueVSize();
    block_trueOffsets[2] = W_space.GetTrueVSize();
    block_trueOffsets.PartialSum();

    BlockVector xp( block_trueOffsets );

    xp = 0.;

    //    Define grid functions for the current configuration, reference
    //    configuration, final deformation, and pressure
    ParGridFunction x_gf( &R_space );
    ParGridFunction p_gf( &W_space );

    if ( myid == 0 )
    {
        printf( "Mesh is %i dimensional.\n", dim );
        printf( "Number of mesh attributes: %i\n", pmesh->attributes.Size() );
        printf( "Number of boundary attributes: %i\n", pmesh->bdr_attributes.Size() );
    }

    //    Define the solution vector x as a finite element grid function
    //    corresponding to fespace. Initialize x with initial guess of zero,
    //    which satisfies the boundary conditions.
    Vector Nu( pmesh->attributes.Max() );
    Nu = .3;
    PWConstCoefficient nu_func( Nu );

    Vector E( pmesh->attributes.Max() );
    E = 210E9;
    PWConstCoefficient E_func( E );

    PhaseFieldElasticMaterial iem( E_func, nu_func, PhaseFieldElasticMaterial::StrainEnergyType::Amor );

    plugin::Memorize mm( pmesh );

    auto intg = new plugin::PhaseFieldIntegrator( iem, mm );
    // intg->setNonlinear( true );

    auto* nlf = new mfem::ParBlockNonlinearForm( spaces );
    nlf->AddDomainIntegrator( intg );
    nlf->AddBdrFaceIntegrator( new plugin::BlockNonlinearDirichletPenaltyIntegrator( d, hevi ) );
    nlf->AddBdrFaceIntegrator( new plugin::BlockNonlinearDirichletPenaltyIntegrator( d2, hevi2 ) );
    nlf->SetGradientType( Operator::Type::Hypre_ParCSR );

    // Set up the Jacobian solver
    // PetscLinearSolver* petsc = new PetscLinearSolver( MPI_COMM_WORLD );

    mfem::Solver* lin_solver{ nullptr };
    // {
    //     auto cg  = new mfem::CGSolver( MPI_COMM_WORLD );
    //     lin_solver = cg;
    //     // gmres->SetPrintLevel( -1 );
    //     cg->SetRelTol( 1e-11 );
    //     cg->SetMaxIter( 200000 );
    //     cg->SetPrintLevel(0);

    //     mfem::HypreBoomerAMG* prec = new mfem::HypreBoomerAMG();
    //     prec->SetPrintLevel(0);
    //     cg->SetPreconditioner( *prec );
    // }
    {
        auto mumps = new mfem::MUMPSSolver( MPI_COMM_WORLD );
        mumps->SetPrintLevel( 0 );
        mumps->SetMatrixSymType( MUMPSSolver::MatType::UNSYMMETRIC );
        lin_solver = mumps;
    }

    auto newton_solver = new plugin::MultiNewtonAdaptive( MPI_COMM_WORLD );
    intg->SetIterAux( newton_solver );

    // Set the newton solve parameters
    newton_solver->iterative_mode = true;
    newton_solver->SetSolver( *lin_solver );
    newton_solver->SetOperator( *nlf );
    newton_solver->SetPrintLevel( -1 );
    newton_solver->SetRelTol( 1e-7 );
    newton_solver->SetAbsTol( 0 );
    newton_solver->SetMaxIter( 8 );
    newton_solver->SetPrintLevel( 0 );
    newton_solver->SetDelta( 1e-5 );
    newton_solver->SetMaxStep( 1000000 );
    newton_solver->SetMaxDelta( 5e-5 );
    newton_solver->SetMinDelta( 1e-14 );
    std::string outPutName = "p_phase_field_square_shear_hex_test_rp=" + std::to_string( par_ref_levels );

    ParaViewDataCollection paraview_dc( outPutName, pmesh );
    paraview_dc.SetPrefixPath( "ParaView" );
    paraview_dc.SetLevelsOfDetail( order );
    paraview_dc.SetCycle( 0 );
    paraview_dc.SetDataFormat( VTKFormat::BINARY );
    paraview_dc.SetHighOrderOutput( true );
    paraview_dc.SetTime( 0.0 ); // set the time
    paraview_dc.RegisterField( "Displace", &x_gf );
    paraview_dc.RegisterField( "PhaseField", &p_gf );

    auto stress_fec = new DG_FECollection( order, dim );
    auto stress_fespace = new ParFiniteElementSpace( pmesh, stress_fec, 7 );
    ParGridFunction stress_grid( stress_fespace );
    plugin::StressCoefficient sc( dim, iem );
    sc.SetDisplacement( x_gf );
    paraview_dc.RegisterField( "Stress", &stress_grid );
    stress_grid.ProjectCoefficient( sc );
    paraview_dc.Save();

    std::function<void( int, int, double )> func =
        [&paraview_dc, &stress_grid, &sc, &x_gf, &p_gf, &xp]( int step, int count, double time )
    {
        static int local_counter = 0;
        if ( count % 10 == 0 )
        {
            x_gf.Distribute( xp.GetBlock( 0 ) );
            p_gf.Distribute( xp.GetBlock( 1 ) );
            paraview_dc.SetCycle( local_counter++ );
            paraview_dc.SetTime( time );
            stress_grid.ProjectCoefficient( sc );
            paraview_dc.Save();
        }
    };

    newton_solver->SetDataCollectionFunc( func );

    Vector zero;

    newton_solver->Mult( zero, xp );

    // MFEMFinalizePetsc();
    return 0;
}
