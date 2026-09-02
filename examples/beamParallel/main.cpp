/**
 * @file
 * @brief Parallel finite-strain cantilever-beam example.
 *
 * This executable exercises the nonlinear elasticity integrator, integration-
 * point storage, adaptive load stepping, and a distributed direct solve. The
 * 10 x 1 x 0.1 beam is clamped on boundary attribute 1 (x = 0) and loaded by a
 * uniform +z traction on boundary attribute 2 (x = 10). The load factor is
 * advanced from zero to one, and every accepted state is written as a ParaView
 * time series.
 */

#include "Plugin.h"
#include "mfem.hpp"

#include <iomanip>
#include <iostream>
#include <string>

namespace
{
constexpr int fixed_boundary_attribute = 1;
constexpr int loaded_boundary_attribute = 2;
constexpr int traction_component = 2;
constexpr mfem::real_t youngs_modulus = 1.2e6;
constexpr mfem::real_t poisson_ratio = 0.0;
constexpr mfem::real_t applied_traction = 40.0;

class RankZeroResidualMonitor final : public mfem::IterativeSolverMonitor
{
public:
    RankZeroResidualMonitor( MPI_Comm comm, const std::string& prefix, int requested_print_level ) : prefix_( prefix )
    {
        int rank = 0;
        MPI_Comm_rank( comm, &rank );
        print_level_ = rank == 0 ? requested_print_level : -1;
    }

    void MonitorResidual( int iteration,
                          mfem::real_t norm,
                          const mfem::Vector&,
                          bool final ) override
    {
        const bool should_print = print_level_ == 1 || ( print_level_ == 3 && ( final || iteration == 0 ) );
        if ( !should_print )
        {
            return;
        }

        mfem::out << prefix_ << " iteration " << std::setw( 2 ) << iteration << " : ||r|| = " << norm;
        if ( iteration == 0 )
        {
            initial_norm_ = norm;
        }
        else if ( initial_norm_ > 0.0 )
        {
            mfem::out << ",  ||r||/||r_0|| = " << norm / initial_norm_;
        }
        mfem::out << '\n';
    }

private:
    std::string prefix_;
    int print_level_ = -1;
    mfem::real_t initial_norm_ = 0.0;
};

int RunBeamExample( int argc, char* argv[], MPI_Comm comm )
{
    int rank = 0;
    MPI_Comm_rank( comm, &rank );

    const char* mesh_file = "../../../data/gmshBeam.msh";
    const char* output_directory = "ParaView";
    int order = 1;
    int serial_refinement_levels = 0;
    int parallel_refinement_levels = 0;

    mfem::OptionsParser args( argc, argv );
    args.AddOption( &mesh_file, "-m", "--mesh", "Beam mesh file." );
    args.AddOption( &order, "-o", "--order", "H1 finite element polynomial order." );
    args.AddOption( &serial_refinement_levels,
                    "-rs",
                    "--refine-serial",
                    "Number of uniform refinements before mesh partitioning." );
    args.AddOption( &parallel_refinement_levels,
                    "-rp",
                    "--refine-parallel",
                    "Number of uniform refinements after mesh partitioning." );
    args.AddOption( &output_directory,
                    "-od",
                    "--output-directory",
                    "Root directory for the ParaView time series." );
    args.Parse();
    if ( !args.Good() )
    {
        if ( rank == 0 )
        {
            args.PrintUsage( std::cout );
        }
        return 1;
    }
    if ( rank == 0 )
    {
        args.PrintOptions( std::cout );
    }

    mfem::Mesh serial_mesh( mesh_file, 1, 1 );
    const int dimension = serial_mesh.Dimension();
    const bool has_required_boundaries = serial_mesh.bdr_attributes.Size() > 0 &&
                                         serial_mesh.bdr_attributes.Max() >= loaded_boundary_attribute;
    if ( dimension != 3 || !has_required_boundaries )
    {
        if ( rank == 0 )
        {
            std::cerr << "beamParallel requires a three-dimensional mesh with boundary attributes "
                      << fixed_boundary_attribute << " and " << loaded_boundary_attribute << ".\n";
        }
        return 2;
    }

    for ( int level = 0; level < serial_refinement_levels; ++level )
    {
        serial_mesh.UniformRefinement();
    }

    mfem::ParMesh mesh( comm, serial_mesh );
    for ( int level = 0; level < parallel_refinement_levels; ++level )
    {
        mesh.UniformRefinement();
    }

    mfem::H1_FECollection finite_elements( order, dimension );
    mfem::ParFiniteElementSpace space( &mesh, &finite_elements, dimension, mfem::Ordering::byVDIM );
    const HYPRE_BigInt global_true_dofs = space.GlobalTrueVSize();
    if ( rank == 0 )
    {
        std::cout << "Number of finite element unknowns: " << global_true_dofs << '\n';
    }

    mfem::Array<int> essential_boundary( mesh.bdr_attributes.Max() );
    essential_boundary = 0;
    essential_boundary[fixed_boundary_attribute - 1] = 1;

    mfem::ConstantCoefficient elastic_modulus( youngs_modulus );
    mfem::ConstantCoefficient transverse_contraction( poisson_ratio );
    IsotropicElasticMaterial material( elastic_modulus, transverse_contraction );
    material.setLargeDeformation( true );

    mfem::Vector traction_by_boundary( mesh.bdr_attributes.Max() );
    traction_by_boundary = 0.0;
    traction_by_boundary( loaded_boundary_attribute - 1 ) = applied_traction;
    mfem::VectorArrayCoefficient traction( dimension );
    traction.Set( traction_component, new mfem::PWConstCoefficient( traction_by_boundary ) );

    plugin::IntegrationPointStorage<> point_storage( &mesh );
    mfem::ParNonlinearForm residual( &space );
    // ParNonlinearForm owns the integrators registered with it.
    residual.AddDomainIntegrator( new plugin::NonlinearElasticityIntegrator( material, point_storage ) );
    residual.AddBdrFaceIntegrator( new plugin::NonlinearVectorBoundaryLFIntegrator( traction ) );
    residual.SetEssentialBC( essential_boundary );
    residual.SetGradientType( mfem::Operator::Type::Hypre_ParCSR );

    mfem::MUMPSSolver tangent_solver( comm );
    tangent_solver.SetMatrixSymType( mfem::MUMPSSolver::MatType::SYMMETRIC_POSITIVE_DEFINITE );
    tangent_solver.SetPrintLevel( -1 );

    RankZeroResidualMonitor newton_monitor( comm, "Newton", 1 );
    mfem::ParGridFunction displacement( &space );
    displacement = 0.0;
    mfem::Vector true_dofs;
    displacement.GetTrueDofs( true_dofs );

    mfem::ParaViewDataCollection paraview( "beamParallel", &mesh );
    paraview.SetPrefixPath( output_directory );
    paraview.SetLevelsOfDetail( order );
    paraview.SetDataFormat( mfem::VTKFormat::BINARY );
    paraview.SetHighOrderOutput( true );
    paraview.RegisterField( "displacement", &displacement );
    paraview.SetCycle( 0 );
    paraview.SetTime( 0.0 );
    paraview.Save();

    plugin::MultiNewtonAdaptive<plugin::NewtonLineSearch> nonlinear_solver( comm );
    nonlinear_solver.iterative_mode = true;
    nonlinear_solver.SetSolver( tangent_solver );
    nonlinear_solver.SetOperator( residual );
    nonlinear_solver.SetPrintLevel( -1 );
    nonlinear_solver.SetMonitor( newton_monitor );
    nonlinear_solver.SetRelTol( 1e-7 );
    nonlinear_solver.SetAbsTol( 1e-8 );
    nonlinear_solver.SetMaxIter( 6 );
    nonlinear_solver.SetDelta( 0.1 );
    nonlinear_solver.SetMinDelta( 1e-15 );

    int output_cycle = 0;
    nonlinear_solver.SetDataCollectionFunc(
        [&]( int, int, mfem::real_t load_factor )
        {
            displacement.SetFromTrueDofs( true_dofs );
            paraview.SetCycle( ++output_cycle );
            paraview.SetTime( load_factor );
            paraview.Save();
        } );

    // An empty right-hand side selects the zero vector; loading is assembled by
    // the boundary integrator and scaled by the adaptive solver's load factor.
    mfem::Vector zero_rhs;
    nonlinear_solver.Mult( zero_rhs, true_dofs );
    displacement.SetFromTrueDofs( true_dofs );

    if ( rank == 0 )
    {
        std::cout << "ParaView time series: " << output_directory << "/beamParallel/beamParallel.pvd\n";
    }
    return 0;
}
} // namespace

int main( int argc, char* argv[] )
{
    MPI_Init( &argc, &argv );
    const int status = RunBeamExample( argc, argv, MPI_COMM_WORLD );
    MPI_Finalize();
    return status;
}
