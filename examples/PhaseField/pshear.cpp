#include "PhaseField.h"
#include "Plugin.h"
#include "mfem.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <type_traits>

namespace
{
constexpr int kBottomBoundary = 11;
constexpr int kTopBoundary = 12;
constexpr int kRightBoundary = 13;
constexpr int kLeftTopBoundary = 14;
constexpr int kLeftBottomBoundary = 15;

class TimedBoomerAMG final : public mfem::HypreBoomerAMG
{
public:
    explicit TimedBoomerAMG( bool timing ) : mTiming( timing )
    {
    }

    void SetOperator( const mfem::Operator& op ) override
    {
        mSetupSeconds = 0.;
        mfem::HypreBoomerAMG::SetOperator( op );
    }

    using mfem::HypreBoomerAMG::Setup;
    void Setup( const mfem::HypreParVector& rhs, mfem::HypreParVector& solution ) const override
    {
        if ( !mTiming || setup_called )
        {
            mfem::HypreBoomerAMG::Setup( rhs, solution );
            return;
        }
        const double start = MPI_Wtime();
        mfem::HypreBoomerAMG::Setup( rhs, solution );
        mSetupSeconds += MPI_Wtime() - start;
    }

    double SetupSeconds() const
    {
        return mSetupSeconds;
    }

private:
    bool mTiming;
    mutable double mSetupSeconds{ 0. };
};

// Optional diagnostics belong to the example; the nonlinear solver still uses
// the normal MFEM Solver interface and unchanged convergence criteria.
class ReportingGMRESSolver final : public mfem::GMRESSolver
{
public:
    ReportingGMRESSolver( MPI_Comm communicator, const int infoLevel, TimedBoomerAMG& preconditioner, const char* blockName )
        : mfem::GMRESSolver( communicator ),
          mInfoLevel( infoLevel ),
          mCommunicator( communicator ),
          mPreconditioner( preconditioner ),
          mBlockName( blockName )
    {
        MPI_Comm_rank( communicator, &mRank );
    }

    void Mult( const mfem::Vector& rhs, mfem::Vector& solution ) const override
    {
        const double start = mInfoLevel ? MPI_Wtime() : 0.;
        mfem::GMRESSolver::Mult( rhs, solution );
        const double elapsed = mInfoLevel ? MPI_Wtime() - start : 0.;
        if ( mInfoLevel == 0 )
        {
            return;
        }

        // The matrix multiply and global norms must run on every rank.
        mfem::Vector residual( rhs.Size() );
        oper->Mult( solution, residual );
        residual -= rhs;
        const mfem::real_t residualNorm = Norm( residual );
        const mfem::real_t rhsNorm = Norm( rhs );
        const auto relativeNorm = []( const mfem::real_t numerator, const mfem::real_t denominator )
        {
            return denominator > 0. ? numerator / denominator
                                    : ( numerator == 0. ? mfem::real_t( 0. ) : std::numeric_limits<mfem::real_t>::infinity() );
        };
        double timing[3] = { mPreconditioner.SetupSeconds(), elapsed - mPreconditioner.SetupSeconds(), elapsed };
        MPI_Allreduce( MPI_IN_PLACE, timing, 3, MPI_DOUBLE, MPI_MAX, mCommunicator );
        ++mSolveCount;
        if ( mRank == 0 )
        {
            mfem::out << "GMRES solve " << mSolveCount << " [" << mBlockName << "]: total iterations = " << GetNumIterations()
                      << ", relative residual ||b-Ax||/||b|| = " << relativeNorm( residualNorm, rhsNorm )
                      << ", preconditioned relative residual = " << relativeNorm( GetFinalNorm(), GetInitialNorm() )
                      << ", converged = " << ( GetConverged() ? "yes" : "no" ) << ", AMG setup s = " << timing[0]
                      << ", Krylov solve s = " << timing[1] << ", setup+solve s = " << timing[2] << '\n';
        }
    }

private:
    int mInfoLevel;
    MPI_Comm mCommunicator;
    TimedBoomerAMG& mPreconditioner;
    const char* mBlockName;
    int mRank{ 0 };
    mutable int mSolveCount{ 0 };
};

#ifdef MFEM_USE_MUMPS
class ReportingMUMPSSolver final : public mfem::MUMPSSolver
{
public:
    ReportingMUMPSSolver( MPI_Comm communicator, int infoLevel, const char* blockName )
        : mfem::MUMPSSolver( communicator ), mCommunicator( communicator ), mInfoLevel( infoLevel ), mBlockName( blockName )
    {
        // Factor the full assembled block; do not impose SPD or discard a triangle.
        SetMatrixSymType( mfem::MUMPSSolver::UNSYMMETRIC );
        SetReorderingReuse( false );
        SetPrintLevel( 1 );
    }

    void SetOperator( const mfem::Operator& op ) override
    {
        const double start = mInfoLevel ? MPI_Wtime() : 0.;
        mfem::MUMPSSolver::SetOperator( op );
        mSetupSeconds = mInfoLevel ? MPI_Wtime() - start : 0.;
        // Borrowed only for the immediately following Mult diagnostic. The form
        // may replace its gradient on the next assembly; never reuse this pointer.
        mOperator = &op;
    }

    void Mult( const mfem::Vector& rhs, mfem::Vector& solution ) const override
    {
        const double start = mInfoLevel ? MPI_Wtime() : 0.;
        mfem::MUMPSSolver::Mult( rhs, solution );
        const double elapsed = mInfoLevel ? MPI_Wtime() - start : 0.;
        if ( !mInfoLevel )
        {
            return;
        }
        mfem::Vector residual( rhs.Size() );
        mOperator->Mult( solution, residual );
        residual -= rhs;
        const mfem::real_t residualNorm = std::sqrt( mfem::InnerProduct( mCommunicator, residual, residual ) );
        const mfem::real_t rhsNorm = std::sqrt( mfem::InnerProduct( mCommunicator, rhs, rhs ) );
        const mfem::real_t relativeNorm =
            rhsNorm > 0. ? residualNorm / rhsNorm
                         : ( residualNorm == 0. ? mfem::real_t( 0. ) : std::numeric_limits<mfem::real_t>::infinity() );
        double timing[] = { mSetupSeconds, elapsed, mSetupSeconds + elapsed };
        MPI_Allreduce( MPI_IN_PLACE, timing, 3, MPI_DOUBLE, MPI_MAX, mCommunicator );
        int rank;
        MPI_Comm_rank( mCommunicator, &rank );
        ++mSolveCount;
        if ( rank == 0 )
        {
            mfem::out << "MUMPS solve " << mSolveCount << " [" << mBlockName << "]: relative residual ||b-Ax||/||b|| = " << relativeNorm
                      << ", analysis+factorization s = " << timing[0] << ", solve s = " << timing[1]
                      << ", setup+solve s = " << timing[2] << '\n';
        }
    }

private:
    MPI_Comm mCommunicator;
    int mInfoLevel;
    const char* mBlockName;
    const mfem::Operator* mOperator{ nullptr };
    double mSetupSeconds{ 0. };
    mutable int mSolveCount{ 0 };
};
#endif

mfem::real_t CrackCurveY( const mfem::real_t x )
{
    return -1.69104e-6 - 2.65179 * x + 6455.51 * std::pow( x, 2 ) - 9.11803e6 * std::pow( x, 3 );
}

void VerifyMesh( const mfem::Mesh& mesh )
{
    MFEM_VERIFY( mesh.Dimension() == 2 && mesh.SpaceDimension() == 2,
                 "pPhaseField_shear requires a two-dimensional planar mesh." );
    MFEM_VERIFY( mesh.attributes.Size() == 1 && mesh.attributes.Find( 1 ) >= 0,
                 "pPhaseField_shear requires domain attribute 1." );
    MFEM_VERIFY( mesh.bdr_attributes.Find( kBottomBoundary ) >= 0 && mesh.bdr_attributes.Find( kTopBoundary ) >= 0,
                 "pPhaseField_shear requires bottom boundary attribute 11 and top boundary attribute 12." );
    MFEM_VERIFY( mesh.bdr_attributes.Find( kRightBoundary ) >= 0 && mesh.bdr_attributes.Find( kLeftTopBoundary ) >= 0 &&
                     mesh.bdr_attributes.Find( kLeftBottomBoundary ) >= 0,
                 "pPhaseField_shear requires right boundary attribute 13 and left boundary attributes 14 and 15." );
}

mfem::real_t GlobalReduction( mfem::real_t value, const MPI_Op operation, MPI_Comm communicator )
{
    MFEM_VERIFY( MPI_Allreduce( MPI_IN_PLACE, &value, 1, mfem::MPITypeMap<mfem::real_t>::mpi_type, operation, communicator ) == MPI_SUCCESS,
                 "MPI failed to reduce a phase-field diagnostic." );
    return value;
}

void VerifyFiniteSolution( const mfem::Vector& solution, MPI_Comm communicator )
{
    mfem::real_t nonfinite = 0.;
    for ( int i = 0; i < solution.Size(); ++i )
    {
        if ( !mfem::IsFinite( solution( i ) ) )
        {
            nonfinite = 1.;
            break;
        }
    }
    MFEM_VERIFY( GlobalReduction( nonfinite, MPI_MAX, communicator ) == 0.,
                 "The distributed displacement/phase solution contains a non-finite coefficient." );
}

int RunExample( int argc, char* argv[], MPI_Comm communicator )
{
    const int rank = mfem::Mpi::WorldRank();
    const char* meshFile = "data/crack_square2d_quad.msh";
    const char* outputDirectory = "ParaView";
    const char* displacementAMG = "systems";
    const char* linearSolver = "direct";
    mfem::real_t relativeTolerance = std::is_same_v<mfem::real_t, float> ? 1e-4f : 1e-5;
    mfem::real_t displacementAbsoluteTolerance = 1e-2;
    mfem::real_t phaseAbsoluteTolerance = 1e-6;
    mfem::real_t innerRelativeTolerance = std::is_same_v<mfem::real_t, float> ? 1e-5f : 1e-8;
    mfem::real_t innerAbsoluteTolerance = 1e-3;
    int innerNewtonIterations = 30;
    int order = 1;
    int serialRefinementLevels = 3;
    int parallelRefinementLevels = 4;
    int localRefinementLevels = 0;
    int maximumSteps = 100000;
    int maximumSweeps = 1000;
    int outputInterval = 20;
    int infoLevel = 0;
    bool output = true;
    mfem::real_t maximumDisplacement = 1e-4;
    mfem::real_t finalPseudoTime = 1.;
    mfem::real_t initialPseudoTimeStep = 1e-6;
    mfem::real_t maximumPseudoTimeStep = 1e-2;
    mfem::real_t minimumPseudoTimeStep = 1e-14;
    PhaseFieldFractureParameters fractureParameters;

    mfem::OptionsParser args( argc, argv );
    args.AddOption( &displacementAbsoluteTolerance, "-u-atol", "--displacement-absolute-tolerance",
                    "Outer displacement residual absolute tolerance (N/m)." );
    args.AddOption( &phaseAbsoluteTolerance, "-phi-atol", "--phase-absolute-tolerance",
                    "Outer phase residual absolute tolerance (N)." );
    args.AddOption( &innerNewtonIterations, "-u-ni", "--mechanics-newton-iterations",
                    "Maximum inner mechanics Newton corrections at fixed phase." );
    args.AddOption( &innerRelativeTolerance, "-u-rtol", "--mechanics-newton-relative-tolerance",
                    "Inner mechanics relative tolerance (reference frozen per subsolve)." );
    args.AddOption( &innerAbsoluteTolerance, "-u-inner-atol", "--mechanics-newton-absolute-tolerance",
                    "Inner mechanics absolute tolerance (N/m); target cannot be looser than the outer goal." );
    args.AddOption( &relativeTolerance, "-rtol", "--relative-tolerance",
                    "Nonlinear relative tolerance for both blocks." );
    args.AddOption( &linearSolver, "-ls", "--linear-solver",
                    "Block linear solver: direct (default, requires MFEM MUMPS) or gmres (BoomerAMG)." );
    args.AddOption( &displacementAMG, "-uamg", "--displacement-amg",
                    "GMRES-only displacement AMG: systems (default), scalar, elasticity, or elasticity-no-refine." );
    args.AddOption( &meshFile, "-m", "--mesh", "Mesh file to use." );
    args.AddOption( &infoLevel, "-il", "--info-level",
                    "Linear diagnostics: 0 disables summaries (default), 1 prints each solve." );
    args.AddOption( &order, "-o", "--order", "Finite element order." );
    args.AddOption( &serialRefinementLevels, "-rs", "--refine-serial", "Uniform serial refinement levels." );
    args.AddOption( &parallelRefinementLevels, "-rp", "--refine-parallel", "Uniform parallel refinement levels." );
    args.AddOption( &localRefinementLevels, "-lr", "--local-refine-level", "Crack-tip local refinement levels." );
    args.AddOption( &maximumDisplacement, "-disp", "--maximum-displacement", "Final horizontal top displacement." );
    args.AddOption( &fractureParameters.criticalEnergyReleaseRate, "-gc", "--fracture-energy",
                    "Critical energy release rate." );
    args.AddOption( &fractureParameters.lengthScale, "-l", "--length-scale", "AT2 length scale." );
    args.AddOption( &fractureParameters.residualStiffness, "-k", "--residual-stiffness",
                    "Quadratic degradation residual stiffness." );
    args.AddOption( &finalPseudoTime, "-tf", "--final-pseudo-time", "Final continuation coordinate." );
    args.AddOption( &initialPseudoTimeStep, "-dt", "--initial-step", "Initial continuation increment." );
    args.AddOption( &maximumPseudoTimeStep, "-dt-max", "--maximum-step", "Maximum continuation increment." );
    args.AddOption( &minimumPseudoTimeStep, "-dt-min", "--minimum-step", "Minimum continuation increment." );
    args.AddOption( &maximumSteps, "-steps", "--maximum-steps", "Maximum continuation attempts." );
    args.AddOption( &maximumSweeps, "-ni", "--nonlinear-iterations",
                    "Maximum block-Newton sweeps per continuation attempt (not GMRES iterations)." );
    args.AddOption( &outputInterval, "-oi", "--output-interval", "Accepted steps between ParaView writes." );
    args.AddOption( &outputDirectory, "-od", "--output-directory", "ParaView output directory." );
    args.AddOption( &output, "-vis", "--visualization", "-no-vis", "--no-visualization",
                    "Enable ParaView and force-curve output." );
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

    const std::string amgMode( displacementAMG );
    const std::string solverMode( linearSolver );
    MFEM_VERIFY( std::isfinite( relativeTolerance ) && relativeTolerance > 0. && relativeTolerance < 1.,
                 "Relative tolerance must be finite and in (0, 1)." );
    MFEM_VERIFY( std::isfinite( displacementAbsoluteTolerance ) && displacementAbsoluteTolerance >= 0. &&
                     std::isfinite( phaseAbsoluteTolerance ) && phaseAbsoluteTolerance >= 0.,
                 "Outer block absolute tolerances must be finite and nonnegative." );
    MFEM_VERIFY( innerNewtonIterations > 0 && std::isfinite( innerRelativeTolerance ) && innerRelativeTolerance >= 0. &&
                     innerRelativeTolerance < 1. && std::isfinite( innerAbsoluteTolerance ) && innerAbsoluteTolerance >= 0.,
                 "Inner Newton requires a positive budget, 0<=rtol<1, and a finite nonnegative atol." );
    MFEM_VERIFY( solverMode == "direct" || solverMode == "gmres", "Linear solver must be direct or gmres." );
#ifndef MFEM_USE_MUMPS
    MFEM_VERIFY(
        solverMode != "direct",
        "Direct solves require MFEM built with MFEM_USE_MUMPS=ON. Rebuild MFEM with MUMPS or select -ls gmres." );
#endif
    MFEM_VERIFY(
        amgMode == "scalar" || amgMode == "systems" || amgMode == "elasticity" || amgMode == "elasticity-no-refine",
        "Displacement AMG must be scalar, systems, elasticity, or elasticity-no-refine." );
    MFEM_VERIFY( infoLevel == 0 || infoLevel == 1, "Info level must be 0 (default) or 1 (linear solver summaries)." );
    MFEM_VERIFY( order >= 1 && serialRefinementLevels >= 0 && parallelRefinementLevels >= 0 && localRefinementLevels >= 0,
                 "Finite element order must be positive and refinement levels must be nonnegative." );
    MFEM_VERIFY( maximumSteps > 0 && outputInterval > 0, "Step and output intervals must be positive." );
    MFEM_VERIFY( maximumSweeps > 0, "The maximum number of block-Newton sweeps must be positive." );
    MFEM_VERIFY( std::isfinite( maximumDisplacement ) && maximumDisplacement >= 0.,
                 "Maximum displacement must be finite and nonnegative." );
    MFEM_VERIFY( std::isfinite( finalPseudoTime ) && finalPseudoTime > 0.,
                 "Final pseudo-time must be finite and positive." );

    MFEM_VERIFY( std::isfinite( minimumPseudoTimeStep ) && std::isfinite( initialPseudoTimeStep ) &&
                     std::isfinite( maximumPseudoTimeStep ) && minimumPseudoTimeStep > 0. && minimumPseudoTimeStep <= initialPseudoTimeStep &&
                     initialPseudoTimeStep <= maximumPseudoTimeStep && minimumPseudoTimeStep < maximumPseudoTimeStep,
                 "Continuation increments must be finite and satisfy 0 < minimum <= initial <= maximum, with minimum < "
                 "maximum." );

    mfem::Mesh serialMesh( meshFile, 1, 1 );
    VerifyMesh( serialMesh );
    for ( int level = 0; level < serialRefinementLevels; level++ )
    {
        serialMesh.UniformRefinement();
    }
    for ( int level = 0; level < localRefinementLevels; level++ )
    {
        mfem::Array<mfem::Refinement> refinements;
        const auto elements = serialMesh.GetElementsArray();
        for ( int element = 0; element < serialMesh.GetNE(); element++ )
        {
            for ( int vertex = 0; vertex < elements[element]->GetNVertices(); vertex++ )
            {
                const double* coordinate = serialMesh.GetVertex( elements[element]->GetVertices()[vertex] );
                if ( coordinate[0] >= -5e-5 && coordinate[1] < 5e-5 &&
                     std::abs( coordinate[1] - CrackCurveY( coordinate[0] ) ) <
                         8e-5 * std::pow( 1. + 250. * std::abs( coordinate[1] ), 4 ) )
                {
                    refinements.Append( element );
                    break;
                }
            }
        }
        serialMesh.GeneralRefinement( refinements );
    }

    mfem::ParMesh mesh( communicator, serialMesh );
    for ( int level = 0; level < parallelRefinementLevels; level++ )
    {
        mesh.UniformRefinement();
    }

    const int dimension = mesh.Dimension();
    mfem::H1_FECollection collection( order, dimension );
    mfem::ParFiniteElementSpace displacementSpace( &mesh, &collection, dimension, mfem::Ordering::byVDIM );
    mfem::ParFiniteElementSpace phaseSpace( &mesh, &collection );
    mfem::Array<mfem::ParFiniteElementSpace*> spaces( 2 );
    spaces[0] = &displacementSpace;
    spaces[1] = &phaseSpace;
    const HYPRE_BigInt globalDisplacementDofs = displacementSpace.GlobalTrueVSize();
    const HYPRE_BigInt globalPhaseDofs = phaseSpace.GlobalTrueVSize();
    if ( rank == 0 )
    {
        std::cout << "Global displacement true DOFs: " << globalDisplacementDofs << '\n'
                  << "Global phase-field true DOFs: " << globalPhaseDofs << '\n';
    }

    mfem::Array<int> displacementBoundaryMarker( mesh.bdr_attributes.Max() );
    displacementBoundaryMarker = 0;
    displacementBoundaryMarker[kBottomBoundary - 1] = 1;
    displacementBoundaryMarker[kTopBoundary - 1] = 1;
    mfem::Array<int> sideBoundaryMarker( mesh.bdr_attributes.Max() );
    sideBoundaryMarker = 0;
    sideBoundaryMarker[kRightBoundary - 1] = 1;
    sideBoundaryMarker[kLeftTopBoundary - 1] = 1;
    sideBoundaryMarker[kLeftBottomBoundary - 1] = 1;
    mfem::Array<int> topBoundaryMarker( mesh.bdr_attributes.Max() );
    topBoundaryMarker = 0;
    topBoundaryMarker[kTopBoundary - 1] = 1;
    mfem::Array<int> constrainedDisplacementDofs;
    mfem::Array<int> loadedHorizontalDofs;
    displacementSpace.GetEssentialTrueDofs( displacementBoundaryMarker, constrainedDisplacementDofs );
    // Borden et al. (2012), Section 4.1, Fig. 5(a): outer side rollers
    // restrain vertical motion only. Crack faces remain traction-free.
    mfem::Array<int> sideVerticalDofs;
    displacementSpace.GetEssentialTrueDofs( sideBoundaryMarker, sideVerticalDofs, 1 );
    constrainedDisplacementDofs.Append( sideVerticalDofs );
    constrainedDisplacementDofs.Sort();
    constrainedDisplacementDofs.Unique();
    displacementSpace.GetEssentialTrueDofs( topBoundaryMarker, loadedHorizontalDofs, 0 );
    int globalLoadedDofs = 0;
    const int localLoadedDofs = loadedHorizontalDofs.Size();
    MFEM_VERIFY( MPI_Allreduce( &localLoadedDofs, &globalLoadedDofs, 1, MPI_INT, MPI_SUM, communicator ) == MPI_SUCCESS,
                 "MPI failed to count loaded boundary DOFs." );
    MFEM_VERIFY( globalLoadedDofs > 0, "Top boundary attribute 12 has no horizontal true DOFs." );

    mfem::Array<int> blockOffsets( 3 );
    blockOffsets[0] = 0;
    blockOffsets[1] = displacementSpace.GetTrueVSize();
    blockOffsets[2] = phaseSpace.GetTrueVSize();
    blockOffsets.PartialSum();
    mfem::BlockVector solution( blockOffsets );
    solution = 0.;
    mfem::ParGridFunction displacement( &displacementSpace );
    mfem::ParGridFunction phaseField( &phaseSpace );
    displacement = 0.;
    phaseField = 0.;

    mfem::ConstantCoefficient youngsModulus( 210e9 );
    mfem::ConstantCoefficient poissonRatio( .3 );
    PhaseFieldElasticMaterial material(
        youngsModulus, poissonRatio, PhaseFieldElasticMaterial::StrainEnergySplit::MieheSpectral, fractureParameters );
    plugin::PhaseFieldPointStorage pointStorage( &mesh );

    mfem::ParBlockNonlinearForm residual( spaces );
    residual.AddDomainIntegrator( new plugin::PhaseFieldIntegrator<plugin::PhaseFieldPointStorage>( material, pointStorage ) );
    mfem::Array<int> constrainedPhaseDofs;
    mfem::Array<mfem::Array<int>*> essentialTrueDofs( 2 );
    essentialTrueDofs[0] = &constrainedDisplacementDofs;
    essentialTrueDofs[1] = &constrainedPhaseDofs;
    mfem::Array<mfem::Vector*> essentialRightHandSides( 2 );
    essentialRightHandSides = nullptr;
    residual.SetEssentialTrueDofs( essentialTrueDofs, essentialRightHandSides );
    residual.SetGradientType( mfem::Operator::Type::Hypre_ParCSR );

    mfem::ParBlockNonlinearForm internalResidual( spaces );
    auto* reactionIntegrator = new plugin::PhaseFieldIntegrator<plugin::PhaseFieldPointStorage>( material, pointStorage );
    internalResidual.AddDomainIntegrator( reactionIntegrator );

    // Each field has its own hierarchy and physics-specific configuration.
    // SetOperator still rebuilds AMG for each new tangent: no stale hierarchy reuse.
    TimedBoomerAMG displacementPreconditioner( infoLevel != 0 );
    TimedBoomerAMG phasePreconditioner( infoLevel != 0 );
    if ( solverMode == "gmres" && amgMode == "systems" )
    {
        displacementPreconditioner.SetSystemsOptions( dimension );
    }
    else if ( solverMode == "gmres" && ( amgMode == "elasticity" || amgMode == "elasticity-no-refine" ) )
    {
        displacementPreconditioner.SetElasticityOptions( &displacementSpace, amgMode == "elasticity" );
    }
    displacementPreconditioner.SetPrintLevel( 0 );
    phasePreconditioner.SetPrintLevel( 0 );
    ReportingGMRESSolver displacementSolver( communicator, infoLevel, displacementPreconditioner, "displacement" );
    ReportingGMRESSolver phaseSolver( communicator, infoLevel, phasePreconditioner, "phase" );
    displacementSolver.SetPreconditioner( displacementPreconditioner );
    phaseSolver.SetPreconditioner( phasePreconditioner );
    for ( auto* solver : { &displacementSolver, &phaseSolver } )
    {
        solver->SetRelTol( std::is_same_v<mfem::real_t, float> ? 1e-5f : 1e-10 );
        solver->SetMaxIter( 2000 );
        solver->SetKDim( 50 );
        solver->SetPrintLevel( 0 );
    }

    // The nonlinear solver borrows these solvers. All solvers outlive it and
    // are destroyed before the forms/spaces and before MPI finalization.
    mfem::Solver* displacementBlockSolver = &displacementSolver;
    mfem::Solver* phaseBlockSolver = &phaseSolver;
#ifdef MFEM_USE_MUMPS
    std::unique_ptr<ReportingMUMPSSolver> displacementDirectSolver;
    std::unique_ptr<ReportingMUMPSSolver> phaseDirectSolver;
    if ( solverMode == "direct" )
    {
        displacementDirectSolver = std::make_unique<ReportingMUMPSSolver>( communicator, infoLevel, "displacement" );
        phaseDirectSolver = std::make_unique<ReportingMUMPSSolver>( communicator, infoLevel, "phase" );
        displacementBlockSolver = displacementDirectSolver.get();
        phaseBlockSolver = phaseDirectSolver.get();
    }
#endif

    plugin::MultiNewtonAdaptive<plugin::NewtonForPhaseField> nonlinearSolver( communicator );
    nonlinearSolver.iterative_mode = true;
    nonlinearSolver.SetBlockSolvers( *displacementBlockSolver, *phaseBlockSolver );
    nonlinearSolver.SetOperator( residual );
    nonlinearSolver.SetRelTol( relativeTolerance );
    const mfem::real_t precisionTolerance = mfem::real_t( 100 ) * std::numeric_limits<mfem::real_t>::epsilon();
    // Distinct physical residual units require distinct absolute tolerances.
    nonlinearSolver.SetAbsTol( 0. );
    nonlinearSolver.SetBlockAbsTol( displacementAbsoluteTolerance, phaseAbsoluteTolerance );
    nonlinearSolver.SetMechanicsNewton( innerNewtonIterations, innerRelativeTolerance, innerAbsoluteTolerance );
    nonlinearSolver.SetMaxIter( maximumSweeps );
    nonlinearSolver.SetPseudoTimeInterval( 0., finalPseudoTime );
    nonlinearSolver.SetDelta( initialPseudoTimeStep );
    nonlinearSolver.SetMaxDelta( maximumPseudoTimeStep );
    nonlinearSolver.SetMinDelta( minimumPseudoTimeStep );
    nonlinearSolver.SetMaxStep( maximumSteps );
    nonlinearSolver.SetTrialStateFunc(
        [blockOffsets, constrainedDisplacementDofs, loadedHorizontalDofs, maximumDisplacement, finalPseudoTime](
            const mfem::real_t pseudoTime, mfem::Vector& state )
        {
            mfem::Vector displacementBlock( state.GetData() + blockOffsets[0], blockOffsets[1] - blockOffsets[0] );
            displacementBlock.SetSubVector( constrainedDisplacementDofs, 0. );
            displacementBlock.SetSubVector( loadedHorizontalDofs, maximumDisplacement * pseudoTime / finalPseudoTime );
        } );
    reactionIntegrator->SetStepContext( &nonlinearSolver );

    mfem::DG_FECollection stressCollection( order, dimension );
    mfem::ParFiniteElementSpace stressSpace( &mesh, &stressCollection, 7 );
    mfem::ParGridFunction stress( &stressSpace );
    plugin::StressCoefficient stressCoefficient( dimension, material );
    stressCoefficient.SetDisplacement( displacement );
    stressCoefficient.SetPhaseField( phaseField );
    plugin::ParaView2DVectorCoefficient paraviewDisplacement( displacement );

    std::unique_ptr<mfem::ParaViewDataCollection> paraview;
    std::ofstream forceCurve;
    if ( output )
    {
        paraview = std::make_unique<mfem::ParaViewDataCollection>( "p_phase_field_square_shear", &mesh );
        paraview->SetPrefixPath( outputDirectory );
        paraview->SetLevelsOfDetail( order );
        paraview->SetDataFormat( mfem::VTKFormat::BINARY );
        paraview->SetHighOrderOutput( true );
        paraview->RegisterVCoeffField( "displacement", &paraviewDisplacement );
        paraview->RegisterField( "phase_field", &phaseField );
        paraview->RegisterField( "stress", &stress );
        stress.ProjectCoefficient( stressCoefficient );
        paraview->SetCycle( 0 );
        paraview->SetTime( 0. );
        paraview->Save();
        if ( rank == 0 )
        {
            forceCurve.open( "p_phase_field_force.csv" );
            MFEM_VERIFY( forceCurve, "Could not open p_phase_field_force.csv for writing." );
            forceCurve << "displacement,reaction_force\n0,0\n";
        }
    }

    mfem::BlockVector unconstrainedResidual( blockOffsets );
    int acceptedSteps = 0;
    nonlinearSolver.SetDataCollectionFunc(
        [&]( const int, const int, const mfem::real_t pseudoTime )
        {
            acceptedSteps++;
            VerifyFiniteSolution( solution, communicator );
            displacement.SetFromTrueDofs( solution.GetBlock( 0 ) );
            phaseField.SetFromTrueDofs( solution.GetBlock( 1 ) );
            if ( output )
            {
                internalResidual.Mult( solution, unconstrainedResidual );
                mfem::real_t localReaction = 0.;
                for ( int i = 0; i < loadedHorizontalDofs.Size(); i++ )
                {
                    localReaction += unconstrainedResidual.GetBlock( 0 )( loadedHorizontalDofs[i] );
                }
                const mfem::real_t reaction = GlobalReduction( localReaction, MPI_SUM, communicator );
                if ( rank == 0 )
                {
                    forceCurve << maximumDisplacement * pseudoTime / finalPseudoTime << ',' << reaction << std::endl;
                }
            }
            if ( paraview && ( acceptedSteps % outputInterval == 0 || pseudoTime == finalPseudoTime ) )
            {
                stress.ProjectCoefficient( stressCoefficient );
                paraview->SetCycle( acceptedSteps );
                paraview->SetTime( pseudoTime );
                paraview->Save();
            }
        } );

    mfem::Vector zeroRightHandSide;
    nonlinearSolver.Mult( zeroRightHandSide, solution );
    const mfem::real_t targetTolerance = precisionTolerance * ( 1. + std::abs( finalPseudoTime ) );
    MFEM_VERIFY( nonlinearSolver.GetConverged() && std::abs( nonlinearSolver.GetCurLambda() - finalPseudoTime ) <= targetTolerance,
                 "pPhaseField_shear did not reach the requested final pseudo-time." );
    displacement.SetFromTrueDofs( solution.GetBlock( 0 ) );
    phaseField.SetFromTrueDofs( solution.GetBlock( 1 ) );
    VerifyFiniteSolution( solution, communicator );
    if ( rank == 0 )
    {
        mfem::out << "Completed pseudo-time " << nonlinearSolver.GetCurLambda()
                  << ", accepted steps = " << acceptedSteps << std::endl;
    }
    return 0;
}
} // namespace

int main( int argc, char* argv[] )
{
    mfem::Mpi::Init( argc, argv );
    mfem::Hypre::Init();
    return RunExample( argc, argv, MPI_COMM_WORLD );
}
