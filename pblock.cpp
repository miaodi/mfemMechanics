
#include "FEMPlugin.h"
#include "Material.h"
#include "NeoHookeanMaterial.h"
#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

class QuasiNewtonSolver : public NewtonSolver
{
public:
    QuasiNewtonSolver( MPI_Comm comm_ ) : NewtonSolver( comm_ )
    {
    }
    virtual void Mult( const Vector& b, Vector& x ) const;
};

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

void InitialDeformation( const Vector& x, Vector& y )
{
    // Set the initial configuration. Having this different from the reference
    // configuration can help convergence
    y = x;
    y[1] = x[1] + .05 * x[0];
}

int main( int argc, char* argv[] )
{
    // 1. Initialize MPI.
    int num_procs, myid;
    MPI_Init( &argc, &argv );
    MPI_Comm_size( MPI_COMM_WORLD, &num_procs );
    MPI_Comm_rank( MPI_COMM_WORLD, &myid );

    // 1. Parse command-line options.
    const char* mesh_file = "../block.mesh";
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

    {
        for ( int l = 0; l < ser_ref_levels; l++ )
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
        cout << "Number of finite element unknowns: " << size << endl << "Assembling: " << flush;
    }

    // 6. Determine the list of true (i.e. conforming) essential boundary dofs.
    //    In this example, the boundary conditions are defined by marking only
    //    boundary attribute 1 from the mesh as essential and converting it to a
    //    list of true dofs.
    Array<int> ess_tdof_list, temp_list, ess_bdr( pmesh->bdr_attributes.Max() );
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
    ParGridFunction x_gf( fespace );
    ParGridFunction x_ref( fespace );
    ParGridFunction x_def( fespace );

    VectorFunctionCoefficient deform( dim, InitialDeformation );
    VectorFunctionCoefficient refconfig( dim, ReferenceConfiguration );

    x_gf.ProjectCoefficient( refconfig );
    x_ref.ProjectCoefficient( refconfig );

    // Vector Nu( pmesh->attributes.Max() );
    // Nu = .3;
    // PWConstCoefficient nu_func( Nu );

    // Vector E( pmesh->attributes.Max() );
    // E = 2.5e6;
    // PWConstCoefficient E_func( E );

    // IsotropicElasticMaterial iem( E_func, nu_func );
    // iem.setLargeDeformation();

    Vector Mu( pmesh->attributes.Max() );
    Mu = 1.61148e6;

    PWConstCoefficient mu_func( Mu );

    Vector Lambda( pmesh->attributes.Max() );
    Lambda = 499.92568e6;

    PWConstCoefficient lambda_func( Lambda );

    NeoHookeanMaterial nh( mu_func, lambda_func, NeoHookeanType::Poly1 );

    plugin::Memorize mm( pmesh );

    auto intg = new plugin::NonlinearElasticityIntegrator( nh, mm );

    ParNonlinearForm* nlf = new ParNonlinearForm( fespace );
    // {
    //     nlf->AddDomainIntegrator( new HyperelasticNLFIntegrator( new NeoHookeanModel( 1.5e6, 10e9 ) ) );
    // }
    nlf->AddDomainIntegrator( intg );
    nlf->SetEssentialTrueDofs( ess_tdof_list );
    GeneralResidualMonitor newton_monitor( fespace->GetComm(), "Newton", 1 );
    GeneralResidualMonitor j_monitor( fespace->GetComm(), "GMRES", 3 );

    HypreSmoother J_hypreSmoother;
    J_hypreSmoother.SetType( HypreSmoother::l1Jacobi );
    J_hypreSmoother.SetPositiveDiagonal( true );

    MINRESSolver solver( fespace->GetComm() );
    solver.SetRelTol( 1e-12 );
    solver.SetAbsTol( 1e-12 );
    solver.SetMaxIter( 10000 );
    solver.SetPrintLevel( 0 );
    solver.SetPreconditioner( J_hypreSmoother );
    solver.iterative_mode = false;

    // Set the newton solve parameters
    auto newton_solver = new NewtonSolver( fespace->GetComm() );
    newton_solver->iterative_mode = true;
    newton_solver->SetSolver( solver );
    newton_solver->SetOperator( *nlf );
    newton_solver->SetPrintLevel( -1 );
    newton_solver->SetMonitor( newton_monitor );
    newton_solver->SetRelTol( 1e-7 );
    newton_solver->SetAbsTol( 1e-8 );
    newton_solver->SetMaxIter( 20 );

    Vector X( fespace->GetTrueVSize() );
    x_gf.ParallelProject( X );
    int steps = 10;
    VectorArrayCoefficient f( dim );

    nlf->AddBdrFaceIntegrator( new plugin::NonlinearVectorBoundaryLFIntegrator( f ) );
    for ( int i = 0; i < dim; i++ )
    {
        f.Set( i, new ConstantCoefficient( 0.0 ) );
    }

    // 15. Save data in the ParaView format
    ParaViewDataCollection paraview_dc( "test", pmesh );
    paraview_dc.SetPrefixPath( "ParaView" );
    paraview_dc.SetLevelsOfDetail( order );
    paraview_dc.SetDataFormat( VTKFormat::BINARY );
    paraview_dc.SetHighOrderOutput( true );
    paraview_dc.RegisterField( "Displace", &x_def );
    for ( int i = 0; i <= 50; i++ )
    {
        Vector push_force( pmesh->bdr_attributes.Max() );
        push_force = 0.0;
        push_force( 5 ) = -3.e5 * i;
        f.Set( 2, new PWConstCoefficient( push_force ) );
        Vector zero;

        newton_solver->Mult( zero, X );

        x_gf.Distribute( X );
        // // MFEM_VERIFY( newton_solver->GetConverged(), "Newton Solver did not converge." );
        subtract( x_gf, x_ref, x_def );

        paraview_dc.SetTime( i ); // set the time
        paraview_dc.SetCycle( i );

        paraview_dc.Save();
    }

    if ( fec )
    {
        delete fespace;
        delete fec;
    }
    delete newton_solver;
    delete pmesh;

    MPI_Finalize();
    return 0;
}

void QuasiNewtonSolver::Mult( const Vector& b, Vector& x ) const
{
    MFEM_ASSERT( oper != NULL, "the Operator is not set (use SetOperator)." );
    MFEM_ASSERT( prec != NULL, "the Solver is not set (use SetSolver)." );

    int it;
    double norm0, norm, norm_goal;
    const bool have_b = ( b.Size() == Height() );

    if ( !iterative_mode )
    {
        x = 0.0;
    }

    ProcessNewState( x );

    oper->Mult( x, r );
    if ( have_b )
    {
        r -= b;
    }

    norm0 = norm = Norm( r );
    if ( print_options.first_and_last && !print_options.iterations )
    {
        mfem::out << "Newton iteration " << setw( 2 ) << 0 << " : ||r|| = " << norm << "...\n";
    }
    norm_goal = std::max( rel_tol * norm, abs_tol );

    prec->iterative_mode = false;

    // x_{i+1} = x_i - [DF(x_i)]^{-1} [F(x_i)-b]
    for ( it = 0; true; it++ )
    {
        MFEM_ASSERT( IsFinite( norm ), "norm = " << norm );
        if ( print_options.iterations )
        {
            mfem::out << "Newton iteration " << setw( 2 ) << it << " : ||r|| = " << norm;
            if ( it > 0 )
            {
                mfem::out << ", ||r||/||r_0|| = " << norm / norm0;
            }
            mfem::out << '\n';
        }
        Monitor( it, norm, r, x );

        if ( norm <= norm_goal )
        {
            converged = true;
            break;
        }

        if ( it >= max_iter )
        {
            converged = false;
            break;
        }
        if ( it == 0 )
        {
            grad = &oper->GetGradient( x );
            prec->SetOperator( *grad );
        }

        if ( lin_rtol_type )
        {
            AdaptiveLinRtolPreSolve( x, it, norm );
        }

        prec->Mult( r, c ); // c = [DF(x_i)]^{-1} [F(x_i)-b]

        if ( lin_rtol_type )
        {
            AdaptiveLinRtolPostSolve( c, r, it, norm );
        }

        const double c_scale = ComputeScalingFactor( x, b );
        if ( c_scale == 0.0 )
        {
            converged = false;
            break;
        }
        add( x, -c_scale, c, x );

        ProcessNewState( x );

        oper->Mult( x, r );
        if ( have_b )
        {
            r -= b;
        }
        norm = Norm( r );
    }

    final_iter = it;
    final_norm = norm;

    if ( print_options.summary || ( !converged && print_options.warnings ) || print_options.first_and_last )
    {
        mfem::out << "Newton: Number of iterations: " << final_iter << '\n' << "   ||r|| = " << final_norm << '\n';
    }
    if ( print_options.summary || ( !converged && print_options.warnings ) )
    {
        mfem::out << "Newton: No convergence!\n";
    }
}