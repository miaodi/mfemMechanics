//                                MFEM Example 16
//
// Compile with: make ex16
//
// Sample runs:  ex16
//               ex16 -m ../data/inline-tri.mesh
//               ex16 -m ../data/disc-nurbs.mesh -tf 2
//               ex16 -s 1 -a 0.0 -k 1.0
//               ex16 -s 2 -a 1.0 -k 0.0
//               ex16 -s 3 -a 0.5 -k 0.5 -o 4
//               ex16 -s 14 -dt 1.0e-4 -tf 4.0e-2 -vs 40
//               ex16 -m ../data/fichera-q2.mesh
//               ex16 -m ../data/fichera-mixed.mesh
//               ex16 -m ../data/escher.mesh
//               ex16 -m ../data/beam-tet.mesh -tf 10 -dt 0.1
//               ex16 -m ../data/amr-quad.mesh -o 4 -r 0
//               ex16 -m ../data/amr-hex.mesh -o 2 -r 0
//
// Description:  This example solves a time dependent nonlinear heat equation
//               problem of the form du/dt = C(u), with a non-linear diffusion
//               operator C(u) = \nabla \cdot (\kappa + \alpha u) \nabla u.
//
//               The example demonstrates the use of nonlinear operators (the
//               class ConductionOperator defining C(u)), as well as their
//               implicit time integration. Note that implementing the method
//               ConductionOperator::ImplicitSolve is the only requirement for
//               high-order implicit (SDIRK) time integration. In this example,
//               the diffusion operator is linearized by evaluating with the
//               lagged solution from the previous timestep, so there is only
//               a linear solve.
//
//               We recommend viewing examples 2, 9 and 10 before viewing this
//               example.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

/** After spatial discretization, the conduction model can be written as:
 *
 *     du/dt = M^{-1}(-Ku)
 *
 *  where u is the vector representing the temperature, M is the mass matrix,
 *  and K is the diffusion operator with diffusivity depending on u:
 *  (\kappa + \alpha u).
 *
 *  Class ConductionOperator represents the right-hand side of the above ODE.
 */
class ConductionOperator : public TimeDependentOperator
{
protected:
    FiniteElementSpace& fespace;
    Array<int> ess_tdof_list; // this list remains empty for pure Neumann b.c.

    BilinearForm* M;
    BilinearForm* K;
    LinearForm* b;

    SparseMatrix Mmat, Kmat;
    SparseMatrix* T; // T = M + dt K
    double current_dt;

    CGSolver M_solver; // Krylov solver for inverting the mass matrix M
    DSmoother M_prec;  // Preconditioner for the mass matrix M

    CGSolver T_solver; // Implicit solver for T = M + dt K
    DSmoother T_prec;  // Preconditioner for the implicit solver

    mutable Vector z; // auxiliary vector

public:
    ConductionOperator( FiniteElementSpace& f );

    virtual void Mult( const Vector& u, Vector& du_dt ) const;
    /** Solve the Backward-Euler equation: k = f(u + dt*k, t), for the unknown k.
        This is the only requirement for high-order SDIRK implicit integration.*/
    virtual void ImplicitSolve( const double dt, const Vector& u, Vector& k );

    virtual ~ConductionOperator();
};

double InitialTemperature( const Vector& x );

int main( int argc, char* argv[] )
{
    // 1. Parse command-line options.
    const char* mesh_file = "../../data/thermal.msh";
    int ref_levels = 0;
    int order = 1;
    double t_final = 0.5;
    double dt = 1.0e-2;
    bool paraview = true;
    int vis_steps = 5;

    int precision = 8;
    cout.precision( precision );

    OptionsParser args( argc, argv );
    args.AddOption( &mesh_file, "-m", "--mesh", "Mesh file to use." );
    args.AddOption( &ref_levels, "-r", "--refine", "Number of times to refine the mesh uniformly." );
    args.AddOption( &order, "-o", "--order", "Order (degree) of the finite elements." );
    args.AddOption( &t_final, "-tf", "--t-final", "Final time; start time is 0." );
    args.AddOption( &dt, "-dt", "--time-step", "Time step." );
   args.AddOption(&paraview, "-paraview", "--paraview-datafiles", "-no-paraview",
                  "--no-paraview-datafiles",
                  "Save data files for ParaView (paraview.org) visualization.");
    args.AddOption( &vis_steps, "-vs", "--visualization-steps", "Visualize every n-th timestep." );
    args.Parse();
    if ( !args.Good() )
    {
        args.PrintUsage( cout );
        return 1;
    }
    args.PrintOptions( cout );

    // 2. Read the mesh from the given mesh file. We can handle triangular,
    //    quadrilateral, tetrahedral and hexahedral meshes with the same code.
    Mesh* mesh = new Mesh( mesh_file, 1, 1 );
    int dim = mesh->Dimension();

    // 3. Define the ODE solver used for time integration. Several implicit
    //    singly diagonal implicit Runge-Kutta (SDIRK) methods, as well as
    //    explicit Runge-Kutta methods are available.
    ODESolver* ode_solver = new BackwardEulerSolver;

    // 4. Refine the mesh to increase the resolution. In this example we do
    //    'ref_levels' of uniform refinement, where 'ref_levels' is a
    //    command-line parameter.
    for ( int lev = 0; lev < ref_levels; lev++ )
    {
        mesh->UniformRefinement();
    }

    // 5. Define the vector finite element space representing the current and the
    //    initial temperature, u_ref.
    H1_FECollection fe_coll( order, dim );
    FiniteElementSpace fespace( mesh, &fe_coll );

    int fe_size = fespace.GetTrueVSize();
    cout << "Number of temperature unknowns: " << fe_size << endl;

    GridFunction u_gf( &fespace );

    // 6. Set the initial conditions for u. All boundaries are considered
    //    natural.
    FunctionCoefficient u_0( InitialTemperature );
    u_gf.ProjectCoefficient( u_0 );
    Vector u;
    u_gf.GetTrueDofs( u );

    // 7. Initialize the conduction operator and the visualization.
    ConductionOperator oper( fespace );

    u_gf.SetFromTrueDofs( u );
    {
        ofstream omesh( "ex16.mesh" );
        omesh.precision( precision );
        mesh->Print( omesh );
        ofstream osol( "ex16-init.gf" );
        osol.precision( precision );
        u_gf.Save( osol );
    }

   ParaViewDataCollection *pd = NULL;
   if (paraview)
   {
      pd = new ParaViewDataCollection("thermal", mesh);
      pd->SetPrefixPath("ParaView");
      pd->RegisterField("temperature", &u_gf);
      pd->SetLevelsOfDetail(order);
      pd->SetDataFormat(VTKFormat::BINARY);
      pd->SetHighOrderOutput(true);
      pd->SetCycle(0);
      pd->SetTime(0.0);
      pd->Save();
   }

    // 8. Perform time-integration (looping over the time iterations, ti, with a
    //    time-step dt).
    ode_solver->Init( oper );
    double t = 0.0;

    bool last_step = false;
    for ( int ti = 1; !last_step; ti++ )
    {
        if ( t + dt >= t_final - dt / 2 )
        {
            last_step = true;
        }

        ode_solver->Step( u, t, dt );

        if ( last_step || ( ti % vis_steps ) == 0 )
        {
            cout << "step " << ti << ", t = " << t << endl;

            u_gf.SetFromTrueDofs( u );
            if (paraview)
            {
                pd->SetCycle(ti);
                pd->SetTime(t);
                pd->Save();
            }
        }
    }
    std::cout<<u_gf.Max();;

    // 9. Save the final solution. This output can be viewed later using GLVis:
    //    "glvis -m ex16.mesh -g ex16-final.gf".
    {
        ofstream osol( "ex16-final.gf" );
        osol.precision( precision );
        u_gf.Save( osol );
    }

    // 10. Free the used memory.
    delete ode_solver;
    delete mesh;

    return 0;
}

ConductionOperator::ConductionOperator( FiniteElementSpace& f)
    : TimeDependentOperator( f.GetTrueVSize(), 0.0 ), fespace( f ), M( NULL ), K( NULL ), T( NULL ), current_dt( 0.0 ), z( height )
{
    Array<int> ess_bdr( fespace.GetMesh()->bdr_attributes.Max() );
    ess_bdr = 0;
    ess_bdr[10] = 1; // boundary attribute 1 (index 10) is fixed

    Array<int> ess_tdof_list;
    fespace.GetEssentialTrueDofs( ess_bdr, ess_tdof_list );

    const double rel_tol = 1e-8;

    M = new BilinearForm( &fespace );
    IntegrationRules irs( 0, Quadrature1D::GaussLobatto );
    auto ir = irs.Get( Geometry::SQUARE, 1 );
    ConstantCoefficient C( 1. );
    M->AddDomainIntegrator( new MassIntegrator( C ) );
    M->Assemble();
    M->FormSystemMatrix( ess_tdof_list, Mmat );
    Mmat.PrintInfo( mfem::out );
    ofstream myfile;
    myfile.open ("m.txt");
    Mmat.PrintMatlab(myfile);
    myfile.close();

    K = new BilinearForm( &fespace );
    ConstantCoefficient Kappa( 1. );
    K->AddDomainIntegrator( new DiffusionIntegrator( Kappa ) );
    K->Assemble();
    K->FormSystemMatrix( ess_tdof_list, Kmat );
    Kmat.PrintInfo( mfem::out );
    myfile.open ("k.txt");
    Kmat.PrintMatlab(myfile);
    myfile.close();

    M_solver.iterative_mode = false;
    M_solver.SetRelTol( rel_tol );
    M_solver.SetAbsTol( 0.0 );
    M_solver.SetMaxIter( 300 );
    M_solver.SetPrintLevel( 0 );
    M_solver.SetPreconditioner( M_prec );
    M_solver.SetOperator( Mmat );
    // Mmat.PrintMatlab()

    T_solver.iterative_mode = false;
    T_solver.SetRelTol( rel_tol );
    T_solver.SetAbsTol( 0.0 );
    T_solver.SetMaxIter( 1000 );
    T_solver.SetPrintLevel( 0 );
    T_solver.SetPreconditioner( T_prec );

    Vector power( fespace.GetMesh()->bdr_attributes.Max() );
    {
        b = new LinearForm( &fespace );
        power = .0;
        power( 11 ) = 1;
        // power( 12 ) = 5;
        PWConstCoefficient powerCoef( power );
        b->AddBdrFaceIntegrator( new BoundaryLFIntegrator( powerCoef ) );
        b->Assemble();
        b->Print(mfem::out, 200);
        delete b;
    }

    {
        b = new LinearForm( &fespace );
        power = .0;
        // power( 11 ) = 10;
        power( 12 ) = 1;
        PWConstCoefficient powerCoef( power );
        b->AddBdrFaceIntegrator( new BoundaryLFIntegrator( powerCoef ) );
        b->Assemble();
        b->Print(mfem::out, 200);
        delete b;
    }
    
    {
        b = new LinearForm( &fespace );
        power = .0;
        power( 11 ) = 10;
        power( 12 ) = 5;
        PWConstCoefficient powerCoef( power );
        b->AddBdrFaceIntegrator( new BoundaryLFIntegrator( powerCoef ) );
        b->Assemble();
        b->Print(mfem::out, 200);
    }
}

void ConductionOperator::Mult( const Vector& u, Vector& du_dt ) const
{
    // Compute:
    //    du_dt = M^{-1}*-Ku
    // for du_dt, where K is linearized by using u from the previous timestep
    Kmat.Mult( u, z );
    z.Neg(); // z = -z
    z += *b;
    M_solver.Mult( z, du_dt );
}

void ConductionOperator::ImplicitSolve( const double dt, const Vector& u, Vector& du_dt )
{
    // Solve the equation:
    //    du_dt = M^{-1}*[-K(u + dt*du_dt)]
    // for du_dt, where K is linearized by using u from the previous timestep
    if ( !T )
    {
        T = Add( 1.0, Mmat, dt, Kmat );
        current_dt = dt;
        T_solver.SetOperator( *T );
    }
    MFEM_VERIFY( dt == current_dt, "" ); // SDIRK methods use the same dt
    Kmat.Mult( u, z );
    z.Neg();
    z += *b;
    T_solver.Mult( z, du_dt );
}

ConductionOperator::~ConductionOperator()
{
    delete T;
    delete M;
    delete K;
    delete b;
}

double InitialTemperature( const Vector& x )
{
    return 0.;
}
