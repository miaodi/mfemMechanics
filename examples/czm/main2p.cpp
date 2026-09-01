
#include "Plugin.h"
#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <unistd.h>

using namespace std;
using namespace mfem;

int main( int argc, char* argv[] )
{
    // 1. Initialize MPI.
    Mpi::Init( argc, argv );
    int num_procs = Mpi::WorldSize();
    int myid = Mpi::WorldRank();
    Hypre::Init();

    // 1. Parse command-line options.
    const char* mesh_file = "../../data/crack_square3d.msh";
    int order = 1;
    bool static_cond = false;
    bool visualization = 1;
    int ser_ref_levels = -1, par_ref_levels = -1;
    const char* petscrc_file = "../../data/petscSetting";
    std::string problem_type = "tensile";
    int localRefineLvl = 0;

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
    MFEMInitializePetsc( NULL, NULL, petscrc_file, NULL );

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
    for ( int k = 0; k < localRefineLvl; k++ )
    {
        int ne = mesh->GetNE();
        auto eles = mesh->GetElementsArray();
        Array<Refinement> refinements;
        for ( int i = 0; i < ne; i++ )
        {
            double* node{nullptr};
            for ( int j = 0; j < eles[i]->GetNVertices(); j++ )
            {
                const int vi = eles[i]->GetVertices()[j];
                node = mesh->GetVertex( vi );
                if ( std::abs( std::sqrt( std::pow( node[0] - .00025, 2 ) + std::pow( node[1], 2 ) ) - .0001 ) < .000005 )
                {
                    refinements.Append( i );
                    break;
                }
            }
        }
        mesh->GeneralRefinement( refinements );
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
        cout << "Number of finite element unknowns: " << size << endl << "Assembling: " << endl;
    }

    // 8. Define the solution vector x as a finite element grid function
    //    corresponding to fespace. Initialize x with initial guess of zero,
    //    which satisfies the boundary conditions.
    Vector Nu( pmesh->attributes.Max() );
    Nu = .0;
    PWConstCoefficient nu_func( Nu );

    Vector E( pmesh->attributes.Max() );
    E = 324E9;
    PWConstCoefficient E_func( E );

    IsotropicElasticMaterial iem( E_func, nu_func );

    plugin::Memorize mm( pmesh );

    auto intg = new plugin::NonlinearElasticityIntegrator( iem, mm );
    // intg->setNonlinear( true );

    ParNonlinearForm* nlf = new ParNonlinearForm( fespace );
    nlf->AddDomainIntegrator( intg );

    if ( problem_type == "shear" )
    {
        static VectorArrayCoefficient d( dim );
        for ( int i = 0; i < dim; i++ )
        {
            d.Set( i, new ConstantCoefficient( 0.0 ) );
        }
        Vector topDisp( pmesh->bdr_attributes.Max() );
        topDisp = .0;
        topDisp( 11 ) = 2e-5;
        d.Set( 0, new PWConstCoefficient( topDisp ) );
        Vector sideDisp( pmesh->bdr_attributes.Max() );
        sideDisp = .0;
        d.Set( 1, new PWConstCoefficient( sideDisp ) );
        Vector activeBCX( pmesh->bdr_attributes.Max() );
        activeBCX = 0.0;
        activeBCX( 10 ) = 1e16;
        activeBCX( 11 ) = 1e16;
        Vector activeBCY( pmesh->bdr_attributes.Max() );
        activeBCY = 0.0;
        activeBCY( 12 ) = 1e16;
        activeBCY( 13 ) = 1e16;
        activeBCY( 14 ) = 1e16;
        static VectorArrayCoefficient hevi( dim );
        hevi.Set( 0, new PWConstCoefficient( activeBCX ) );
        hevi.Set( 1, new PWConstCoefficient( activeBCY ) );
        nlf->AddBdrFaceIntegrator( new plugin::NonlinearDirichletPenaltyIntegrator( d, hevi ) );
    }
    else if ( problem_type == "tensile" )
    {
        static VectorArrayCoefficient d( dim );
        for ( int i = 0; i < dim; i++ )
        {
            d.Set( i, new ConstantCoefficient( 0.0 ) );
        }
        Vector topDisp( pmesh->bdr_attributes.Max() );
        topDisp = .0;
        topDisp( 11 ) = 2e-5;
        d.Set( 1, new PWConstCoefficient( topDisp ) );
        Vector activeBC( pmesh->bdr_attributes.Max() );
        activeBC = 0.0;
        activeBC( 10 ) = 1e16;
        activeBC( 11 ) = 1e16;
        static VectorArrayCoefficient hevi( dim );
        for ( int i = 0; i < dim; i++ )
            hevi.Set( i, new PWConstCoefficient( activeBC ) );
        nlf->AddBdrFaceIntegrator( new plugin::NonlinearDirichletPenaltyIntegrator( d, hevi ) );
    }

    mfem::FunctionCoefficient sigma_max{[]( const mfem::Vector& x ) {
        if ( std::sqrt( std::pow( x( 0 ) - .00025, 2 ) + std::pow( x( 1 ), 2 ) ) <= .0001 )
        {
            return 324E6 * 5;
        }
        else
        {
            return 324E6;
        }
    }};
    mfem::FunctionCoefficient tau_max{[]( const mfem::Vector& x ) {
        if ( std::sqrt( std::pow( x( 0 ) - .00025, 2 ) + std::pow( x( 1 ), 2 ) ) <= .0001 )
        {
            return 755.4E6 * 5;
        }
        else
        {
            return 755.4E6;
        }
    }};
    mfem::ConstantCoefficient delta_n{4E-7};
    mfem::ConstantCoefficient delta_t{4E-7};

    auto czm_intg = new plugin::ExponentialADCZMIntegrator( mm, sigma_max, tau_max, delta_n, delta_t );
    nlf->AddInteriorFaceIntegrator( czm_intg );
    // mfem::IntegrationRules GLIntRules( 0, mfem::Quadrature1D::GaussLobatto );
    // czm_intg->SetIntRule( &GLIntRules.Get( mfem::Geometry::SEGMENT, -1 ) );

    // Set up the Jacobian solver
    mfem::Solver* lin_solver{nullptr};

    // {
    //     auto cg  = new mfem::CGSolver( MPI_COMM_WORLD );
    //     lin_solver = cg;
    //     cg->SetRelTol( 1e-11 );
    //     cg->SetMaxIter( 200000 );
    //     cg->SetPrintLevel(1);

    //     mfem::HypreBoomerAMG* prec = new mfem::HypreBoomerAMG();
    //     prec->SetSystemsOptions( dim );
    //     prec->SetPrintLevel(0);
    //     cg->SetPreconditioner( *prec );
    // }
    // {
    //     auto gmres  = new mfem::GMRESSolver( MPI_COMM_WORLD );
    //     lin_solver = gmres;
    //     // gmres->SetPrintLevel( -1 );
    //     gmres->SetRelTol( 1e-13 );
    //     gmres->SetMaxIter( 2000 );
    //     gmres->SetKDim( 50 );
    //     gmres->SetPrintLevel(1);

    //     // mfem::HypreBoomerAMG* prec = new mfem::HypreBoomerAMG();
    //     // prec->SetSystemsOptions( dim );
    //     // prec->SetPrintLevel(0);
    //     // gmres->SetPreconditioner( *prec );
    // }
    {
        auto mumps = new mfem::MUMPSSolver( MPI_COMM_WORLD );
        mumps->SetMatrixSymType( MUMPSSolver::MatType::UNSYMMETRIC );
        // mumps->SetReorderingStrategy( MUMPSSolver::ReorderingStrategy::PARMETIS );
        mumps->SetPrintLevel( -1 );
        mumps->SetDetComp( false );
        lin_solver = mumps;
    }
    // {
    //     nlf->SetGradientType( Operator::Type::PETSC_MATAIJ );
    //     PetscLinearSolver* petsc = new PetscLinearSolver( fespace->GetComm() );
    //     lin_solver = petsc;
    // }

    auto newton_solver = new plugin::MultiNewtonAdaptive<plugin::NewtonLineSearch>( fespace->GetComm() );
    // Set the newton solve parameters
    newton_solver->iterative_mode = true;
    newton_solver->SetSolver( *lin_solver );
    newton_solver->SetOperator( *nlf );
    newton_solver->SetPrintLevel( -1 );
    newton_solver->SetRelTol( 1e-5 );
    newton_solver->SetAbsTol( 0 );
    newton_solver->SetMaxIter( 10 );
    newton_solver->SetPrintLevel( 0 );
    newton_solver->SetDelta( 1e-5 );
    newton_solver->SetMaxDelta( 1e-2 );
    newton_solver->SetMinDelta( 1e-14 );
    newton_solver->SetMaxStep( 100000 );
    // newton_solver->SetAdaptiveL( true );

    Vector zero;

    // ParGridFunction u( fespace );
    Vector u( fespace->GetTrueVSize() );
    u = 0.;

    std::string outPutName = "czm_parallel_rp=" + std::to_string( par_ref_levels );

    ParaViewDataCollection paraview_dc( outPutName, pmesh );
    paraview_dc.SetPrefixPath( "ParaView" );
    paraview_dc.SetCycle( 0 );
    paraview_dc.SetDataFormat( VTKFormat::BINARY );
    paraview_dc.SetHighOrderOutput( true );
    paraview_dc.SetTime( 0.0 ); // set the time

    auto stress_fec = new DG_FECollection( order - 1, dim );
    auto stress_fespace = new ParFiniteElementSpace( pmesh, stress_fec, 7 );
    ParGridFunction stress_grid( stress_fespace );
    plugin::StressCoefficient sc( dim, iem );
    ParGridFunction x_gf( fespace );
    sc.SetDisplacement( x_gf );
    stress_grid.ProjectCoefficient( sc );
    paraview_dc.RegisterField( "Displace", &x_gf );
    paraview_dc.RegisterField( "Stress", &stress_grid );
    stress_grid.ProjectCoefficient( sc );
    paraview_dc.Save();

    std::function<void( int, int, double )> func = [&paraview_dc, &stress_grid, &sc, &x_gf, &u]( int step, int count, double time ) {
        static int local_counter = 0;
        if ( count % 5 == 0 )
        {
            x_gf.Distribute( u );
            paraview_dc.SetCycle( local_counter++ );
            paraview_dc.SetTime( time );
            stress_grid.ProjectCoefficient( sc );
            paraview_dc.Save();
        }
    };
    newton_solver->SetDataCollectionFunc( func );

    // newton_solver->SetLUpdateFunc( []( bool converged, int final_iter, double lambda, double& L ) {
    //     double max_delta = 0;
    //     if ( lambda < .2 )
    //     {
    //         max_delta = 1e-3;
    //     }
    //     else if ( lambda < .3 )
    //     {
    //         max_delta = 1e-4;
    //     }
    //     else if ( lambda < .31 )
    //     {
    //         max_delta = 1e-5;
    //     }
    //     else if ( lambda < .325 )
    //     {
    //         max_delta = 1e-6;
    //     }
    //     else
    //     {
    //         max_delta = 1e-8;
    //     }
    //     if ( converged )
    //     {
    //         L *= 1.2;
    //         L = std::min( max_delta, L );
    //     }
    //     else
    //     {
    //         L /= 2;
    //     }
    // } );
    newton_solver->Mult( zero, u );
    MPI_Finalize();
    return 0;
}
