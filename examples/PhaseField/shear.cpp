
#include "PhaseField.h"
#include "Plugin.h"
#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <omp.h>

using namespace std;
using namespace mfem;

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
                if ( node[1] > -0.00001 && node[1] < 0.0008 && node[0] > -0.00001 && node[0] < .0006 )
                {
                    refinements.Append( i );
                    break;
                }
            }
        }
        mesh->GeneralRefinement( refinements );
    }

    // 5. Define the finite element spaces for displacement and pressure
    //    (Taylor-Hood elements). By default, the displacement (u/x) is a second
    //    order vector field, while the pressure (p) is a linear scalar function.
    H1_FECollection lin_coll( order, dim );

    FiniteElementSpace R_space( mesh, &lin_coll, dim, Ordering::byVDIM );
    FiniteElementSpace W_space( mesh, &lin_coll );

    Array<FiniteElementSpace*> spaces( 2 );
    spaces[0] = &R_space;
    spaces[1] = &W_space;

    int R_size = R_space.GetTrueVSize();
    int W_size = W_space.GetTrueVSize();

    // Print the mesh statistics
    std::cout << "***********************************************************\n";
    std::cout << "dim(u) = " << R_size << "\n";
    std::cout << "dim(p) = " << W_size << "\n";
    std::cout << "dim(u+p) = " << R_size + W_size << "\n";
    std::cout << "***********************************************************\n";

    VectorArrayCoefficient d( dim );
    Vector topDisp( R_space.GetMesh()->bdr_attributes.Max() );
    topDisp = .0;
    topDisp( 10 ) = 1e-4;
    topDisp( 11 ) = 0;
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
    GridFunction x_gf( &R_space );
    GridFunction p_gf( &W_space );

    x_gf.MakeTRef( &R_space, xp.GetBlock( 0 ), 0 );
    p_gf.MakeTRef( &W_space, xp.GetBlock( 1 ), 0 );

    printf( "Mesh is %i dimensional.\n", dim );
    printf( "Number of mesh attributes: %i\n", mesh->attributes.Size() );
    printf( "Number of boundary attributes: %i\n", mesh->bdr_attributes.Size() );

    //    Define the solution vector x as a finite element grid function
    //    corresponding to fespace. Initialize x with initial guess of zero,
    //    which satisfies the boundary conditions.
    Vector Nu( mesh->attributes.Max() );
    Nu = .3;
    PWConstCoefficient nu_func( Nu );

    Vector E( mesh->attributes.Max() );
    E = 210E9;
    PWConstCoefficient E_func( E );

    PhaseFieldElasticMaterial iem( E_func, nu_func, PhaseFieldElasticMaterial::StrainEnergyType::Amor );

    plugin::Memorize mm( mesh );

    auto intg = new plugin::PhaseFieldIntegrator( iem, mm );
    // intg->setNonlinear( true );

    auto* nlf = new mfem::BlockNonlinearForm( spaces );
    nlf->AddDomainIntegrator( intg );
    nlf->AddBdrFaceIntegrator( new plugin::BlockNonlinearDirichletPenaltyIntegrator( d, hevi ) );

    // Set up the Jacobian solver
    omp_set_num_threads( 10 );
    auto j_gmres = new UMFPackSolver();
    auto newton_solver = new plugin::MultiNewtonAdaptive<plugin::NewtonForPhaseField>();
    intg->SetIterAux( newton_solver );

    // Set the newton solve parameters
    newton_solver->iterative_mode = true;
    newton_solver->SetSolver( *j_gmres );
    newton_solver->SetOperator( *nlf );
    newton_solver->SetPrintLevel( -1 );
    newton_solver->SetRelTol( 1e-7 );
    newton_solver->SetAbsTol( 0 );
    newton_solver->SetMaxIter( 10 );
    newton_solver->SetPrintLevel( 0 );
    newton_solver->SetDelta( 2e-6 );
    newton_solver->SetMaxStep( 10000000 );
    newton_solver->SetMaxDelta( 1e-4 );
    newton_solver->SetMinDelta( 1e-14 );

    ParaViewDataCollection paraview_dc( "phase_field_square_shear_test", mesh );
    paraview_dc.SetPrefixPath( "ParaView" );
    paraview_dc.SetLevelsOfDetail( order );
    paraview_dc.SetCycle( 0 );
    paraview_dc.SetDataFormat( VTKFormat::BINARY );
    paraview_dc.SetHighOrderOutput( true );
    paraview_dc.SetTime( 0.0 ); // set the time
    paraview_dc.RegisterField( "Displace", &x_gf );
    paraview_dc.RegisterField( "PhaseField", &p_gf );

    auto stress_fec = new H1_FECollection( order, dim );

    // GridFunction ue( fespace );
    // ue = 0.;
    auto stress_fespace = new FiniteElementSpace( mesh, stress_fec, 7 );
    GridFunction stress_grid( stress_fespace );
    plugin::StressCoefficient sc( dim, iem );
    sc.SetDisplacement( x_gf );
    paraview_dc.RegisterField( "Stress", &stress_grid );
    stress_grid.ProjectCoefficient( sc );
    paraview_dc.Save();

    std::function<void( int, int, double )> func = [&paraview_dc, &stress_grid, &sc]( int step, int count, double time )
    {
        if ( step % 20 == 0 )
        {
            paraview_dc.SetCycle( count );
            paraview_dc.SetTime( count );
            stress_grid.ProjectCoefficient( sc );
            paraview_dc.Save();
        }
    };

    newton_solver->SetDataCollectionFunc( func );

    Vector zero;

    newton_solver->Mult( zero, xp );
    return 0;
}
