#include "J2Plasticity.h"
#include "PostProc.h"
#include "SolidMechanicsIntegrator.h"
#include "Solvers.h"
#include "StressFreeDeformation.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mfem.hpp>
#include <sstream>

namespace
{
constexpr int kCopperAttribute = 1;
constexpr int kBottomBoundary = 11;
constexpr int kSideBoundary = 12;
constexpr int kDishedTopBoundary = 13;

mfem::real_t GlobalMaximum( mfem::real_t value, MPI_Comm communicator )
{
    const int status = MPI_Allreduce( MPI_IN_PLACE, &value, 1, mfem::MPITypeMap<mfem::real_t>::mpi_type, MPI_MAX, communicator );
    MFEM_VERIFY( status == MPI_SUCCESS, "MPI failed to reduce the global maximum." );
    return value;
}

mfem::real_t MeasureMaximumTopDisplacement( const mfem::Vector& trueDisplacement, const mfem::Array<int>& topVerticalDofs, MPI_Comm communicator )
{
    mfem::real_t localMaximum = std::numeric_limits<mfem::real_t>::lowest();
    for ( int i = 0; i < topVerticalDofs.Size(); i++ )
    {
        const mfem::real_t value = trueDisplacement( topVerticalDofs[i] );
        localMaximum = std::max( localMaximum, value );
    }

    mfem::real_t globalMaximum = 0.;
    int globalCount = 0;
    const int localCount = topVerticalDofs.Size();
    MFEM_VERIFY( MPI_Allreduce( &localMaximum, &globalMaximum, 1, mfem::MPITypeMap<mfem::real_t>::mpi_type, MPI_MAX,
                                communicator ) == MPI_SUCCESS,
                 "MPI failed to reduce the top displacement." );
    MFEM_VERIFY( MPI_Allreduce( &localCount, &globalCount, 1, MPI_INT, MPI_SUM, communicator ) == MPI_SUCCESS,
                 "MPI failed to count the top displacement DOFs." );
    MFEM_VERIFY( globalCount > 0, "Boundary attribute 13 has no vertical true DOFs." );
    return globalMaximum;
}

template <typename PointStorage>
mfem::real_t MaximumCommittedPlasticStrain( const PointStorage& pointStorage, MPI_Comm communicator )
{
    mfem::real_t localMaximum = 0.;
    pointStorage.VisitElementPoints(
        [&localMaximum]( const int, const int, const auto& point )
        {
            localMaximum = std::max(
                localMaximum, point.State.template Get<J2PlasticityMaterial>().CommittedState().EquivalentPlasticStrain );
        } );
    return GlobalMaximum( localMaximum, communicator );
}

void VerifyMesh( const mfem::Mesh& mesh )
{
    MFEM_VERIFY( mesh.Dimension() == 2 && mesh.SpaceDimension() == 2,
                 "pCuProtrusion requires a two-dimensional planar mesh." );
    MFEM_VERIFY( mesh.attributes.Size() == 1 && mesh.attributes.Find( kCopperAttribute ) >= 0,
                 "pCuProtrusion requires only copper domain attribute 1." );
    MFEM_VERIFY( mesh.bdr_attributes.Size() == 3 && mesh.bdr_attributes.Find( kBottomBoundary ) >= 0 &&
                     mesh.bdr_attributes.Find( kSideBoundary ) >= 0 && mesh.bdr_attributes.Find( kDishedTopBoundary ) >= 0,
                 "pCuProtrusion requires boundary attributes 11 (bottom), 12 (side), and 13 (dished_top)." );
}

int RunExample( int argc, char* argv[], MPI_Comm communicator )
{
    int rank = 0;
    MPI_Comm_rank( communicator, &rank );

    const char* meshFile = "data/cu_dished_2d.msh";
    int order = 1;
    int serialRefinementLevels = 0;
    int parallelRefinementLevels = 0;
    int loadSteps = 20;
    mfem::real_t youngsModulus = 110000.;
    mfem::real_t poissonRatio = .34;
    mfem::real_t initialYieldStress = 70.;
    mfem::real_t hardeningModulus = 1000.;
    mfem::real_t thermalExpansion = 16.5e-6;
    mfem::real_t referenceTemperature = 25.;
    mfem::real_t targetTemperature = 300.;
    bool output = true;
    const char* outputDirectory = "ParaView";

    mfem::OptionsParser options( argc, argv );
    options.AddOption( &meshFile, "-m", "--mesh", "Copper mesh file." );
    options.AddOption( &order, "-o", "--order", "H1 displacement order." );
    options.AddOption( &serialRefinementLevels, "-rs", "--refine-serial", "Uniform refinements before partitioning." );
    options.AddOption( &parallelRefinementLevels, "-rp", "--refine-parallel",
                       "Uniform refinements after partitioning." );
    options.AddOption( &loadSteps, "-steps", "--load-steps", "Nominal number of adaptive thermal-load steps." );
    options.AddOption( &youngsModulus, "-E", "--youngs-modulus", "Young's modulus in MPa." );
    options.AddOption( &poissonRatio, "-nu", "--poisson-ratio", "Poisson ratio." );
    options.AddOption( &initialYieldStress, "-sy", "--yield-stress", "Initial yield stress in MPa." );
    options.AddOption( &hardeningModulus, "-H", "--hardening-modulus", "Linear isotropic hardening modulus in MPa." );
    options.AddOption( &thermalExpansion, "-cte", "--thermal-expansion", "Constant CTE in 1/K." );
    options.AddOption( &referenceTemperature, "-Tref", "--reference-temperature", "Stress-free reference temperature." );
    options.AddOption( &targetTemperature, "-Ttarget", "--target-temperature", "Final prescribed temperature." );
    options.AddOption( &output, "-output", "--output", "-no-output", "--no-output", "Write parallel ParaView output." );
    options.AddOption( &outputDirectory, "-odir", "--output-directory", "ParaView output prefix." );
    options.Parse();
    if ( !options.Good() )
    {
        if ( rank == 0 )
        {
            options.PrintUsage( std::cout );
        }
        return 1;
    }
    if ( rank == 0 )
    {
        options.PrintOptions( std::cout );
    }

    MFEM_VERIFY( order > 0 && loadSteps > 0 && loadSteps <= std::numeric_limits<int>::max() / 100,
                 "Order and load steps must be positive, and load steps must not overflow the retry limit." );
    MFEM_VERIFY( serialRefinementLevels >= 0 && parallelRefinementLevels >= 0,
                 "Serial and parallel refinement levels must be nonnegative." );
    MFEM_VERIFY( std::isfinite( youngsModulus ) && youngsModulus > 0., "Young's modulus must be finite and positive." );
    MFEM_VERIFY( std::isfinite( poissonRatio ) && poissonRatio > -1. && poissonRatio < .5,
                 "Poisson ratio must be finite and lie in (-1, 0.5)." );
    MFEM_VERIFY( std::isfinite( initialYieldStress ) && initialYieldStress >= 0. && std::isfinite( hardeningModulus ) &&
                     hardeningModulus >= 0.,
                 "Yield stress and hardening modulus must be finite and nonnegative." );
    MFEM_VERIFY( std::isfinite( thermalExpansion ) && thermalExpansion > 0.,
                 "Copper CTE must be finite and positive." );
    MFEM_VERIFY( std::isfinite( referenceTemperature ) && std::isfinite( targetTemperature ) && targetTemperature > referenceTemperature,
                 "The prescribed heating interval must be finite and increasing." );

    mfem::Mesh serialMesh( meshFile, 1, 1 );
    VerifyMesh( serialMesh );
    mfem::Vector coordinateMinimum;
    mfem::Vector coordinateMaximum;
    serialMesh.GetBoundingBox( coordinateMinimum, coordinateMaximum );
    coordinateMaximum -= coordinateMinimum;
    const mfem::real_t characteristicLength = coordinateMaximum.Normlinf();
    MFEM_VERIFY( std::isfinite( characteristicLength ) && characteristicLength > 0.,
                 "The copper mesh must have a finite, positive characteristic length." );
    for ( int level = 0; level < serialRefinementLevels; level++ )
    {
        serialMesh.UniformRefinement();
    }

    mfem::ParMesh mesh( communicator, serialMesh );
    for ( int level = 0; level < parallelRefinementLevels; level++ )
    {
        mesh.UniformRefinement();
    }

    mfem::H1_FECollection displacementCollection( order, mesh.Dimension() );
    mfem::ParFiniteElementSpace displacementSpace( &mesh, &displacementCollection, mesh.Dimension(), mfem::Ordering::byVDIM );
    mfem::L2_FECollection plasticityCollection( 0, mesh.Dimension() );
    mfem::ParFiniteElementSpace plasticitySpace( &mesh, &plasticityCollection );
    const HYPRE_BigInt globalDisplacementDofs = displacementSpace.GlobalTrueVSize();
    if ( rank == 0 )
    {
        std::cout << "Global displacement true DOFs: " << globalDisplacementDofs << '\n';
    }

    mfem::Array<int> fixedBoundaryMarker( mesh.bdr_attributes.Max() );
    fixedBoundaryMarker = 0;
    fixedBoundaryMarker[kBottomBoundary - 1] = 1;
    fixedBoundaryMarker[kSideBoundary - 1] = 1;

    mfem::Array<int> topBoundaryMarker( mesh.bdr_attributes.Max() );
    topBoundaryMarker = 0;
    topBoundaryMarker[kDishedTopBoundary - 1] = 1;
    mfem::Array<int> topVerticalDofs;
    displacementSpace.GetEssentialTrueDofs( topBoundaryMarker, topVerticalDofs, 1 );

    mfem::ConstantCoefficient youngsModulusCoefficient( youngsModulus );
    mfem::ConstantCoefficient poissonRatioCoefficient( poissonRatio );
    mfem::ConstantCoefficient initialYieldStressCoefficient( initialYieldStress );
    mfem::ConstantCoefficient hardeningModulusCoefficient( hardeningModulus );
    mfem::ConstantCoefficient thermalExpansionCoefficient( thermalExpansion );
    mfem::ConstantCoefficient referenceTemperatureCoefficient( referenceTemperature );
    mfem::ConstantCoefficient targetTemperatureCoefficient( targetTemperature );
    J2PlasticityMaterial material( youngsModulusCoefficient, poissonRatioCoefficient, initialYieldStressCoefficient,
                                   hardeningModulusCoefficient );
    plugin::IsotropicThermalExpansion thermalExpansionModel( thermalExpansionCoefficient, targetTemperatureCoefficient,
                                                             referenceTemperatureCoefficient );
    plugin::SolidMechanicsPointStorage<J2PlasticityMaterial> pointStorage( &mesh );

    mfem::ParNonlinearForm residual( &displacementSpace );
    auto* integrator = new plugin::SolidMechanicsIntegrator<J2PlasticityMaterial>( material, pointStorage );
    integrator->AddStressFreeDeformation( thermalExpansionModel );
    residual.AddDomainIntegrator( integrator );
    residual.SetEssentialBC( fixedBoundaryMarker );
    residual.SetGradientType( mfem::Operator::Type::Hypre_ParCSR );

    mfem::MUMPSSolver tangentSolver( communicator );
    tangentSolver.SetMatrixSymType( mfem::MUMPSSolver::MatType::SYMMETRIC_INDEFINITE );
    tangentSolver.SetPrintLevel( -1 );

    const mfem::real_t precisionTolerance = 100. * std::numeric_limits<mfem::real_t>::epsilon();
    const mfem::real_t maximumPseudoTimeStep = 1. / loadSteps;
    const mfem::real_t minimumPseudoTimeStep = std::max( static_cast<mfem::real_t>( 1e-8 ), precisionTolerance );
    MFEM_VERIFY( minimumPseudoTimeStep < maximumPseudoTimeStep,
                 "Load steps are too fine for the precision-dependent minimum step." );
    const mfem::real_t thermalStrain = thermalExpansion * ( targetTemperature - referenceTemperature );
    const mfem::real_t residualScale = youngsModulus * std::abs( thermalStrain ) * characteristicLength;

    plugin::MultiNewtonAdaptive<plugin::NewtonLineSearch> nonlinearSolver( communicator );
    nonlinearSolver.iterative_mode = true;
    nonlinearSolver.SetSolver( tangentSolver );
    nonlinearSolver.SetOperator( residual );
    nonlinearSolver.SetRelTol( std::max( static_cast<mfem::real_t>( 1e-9 ), precisionTolerance ) );
    nonlinearSolver.SetAbsTol( std::max( std::numeric_limits<mfem::real_t>::min(), precisionTolerance * residualScale ) );
    nonlinearSolver.SetMaxIter( 12 );
    nonlinearSolver.SetLineSearch( true );
    nonlinearSolver.SetPrintLevel( -1 );
    nonlinearSolver.SetPseudoTimeInterval( 0., 1. );
    nonlinearSolver.SetDelta( maximumPseudoTimeStep );
    nonlinearSolver.SetMaxDelta( maximumPseudoTimeStep );
    nonlinearSolver.SetMinDelta( minimumPseudoTimeStep );
    nonlinearSolver.SetMaxStep( 100 * loadSteps );

    mfem::ParGridFunction displacement( &displacementSpace );
    mfem::ParGridFunction equivalentPlasticStrain( &plasticitySpace );
    displacement = 0.;
    equivalentPlasticStrain = 0.;
    plugin::ParaView2DVectorCoefficient paraviewDisplacement( displacement );
    mfem::Vector trueDisplacement;
    displacement.GetTrueDofs( trueDisplacement );

    std::unique_ptr<mfem::ParaViewDataCollection> paraview;
    if ( output )
    {
        paraview = std::make_unique<mfem::ParaViewDataCollection>( "pCuProtrusion", &mesh );
        paraview->SetPrefixPath( outputDirectory );
        paraview->SetLevelsOfDetail( order );
        paraview->SetDataFormat( mfem::VTKFormat::BINARY );
        paraview->SetHighOrderOutput( true );
        paraview->RegisterVCoeffField( "displacement", &paraviewDisplacement );
        paraview->RegisterField( "equivalent_plastic_strain", &equivalentPlasticStrain );
        paraview->SetCycle( 0 );
        paraview->SetTime( referenceTemperature );
        paraview->Save();
    }

    int outputCycle = 0;
    mfem::real_t finalMaximumTopDisplacement = 0.;
    mfem::real_t finalMaximumPlasticStrain = 0.;
    nonlinearSolver.SetDataCollectionFunc(
        [&]( const int, const int, const mfem::real_t loadFactor )
        {
            displacement.SetFromTrueDofs( trueDisplacement );
            plugin::ProjectCommittedEquivalentPlasticStrain( pointStorage, equivalentPlasticStrain );
            finalMaximumTopDisplacement = MeasureMaximumTopDisplacement( trueDisplacement, topVerticalDofs, communicator );
            finalMaximumPlasticStrain = MaximumCommittedPlasticStrain( pointStorage, communicator );
            const mfem::real_t temperature = referenceTemperature + loadFactor * ( targetTemperature - referenceTemperature );
            if ( rank == 0 )
            {
                std::ostringstream status;
                status << std::fixed << std::setprecision( 3 ) << "Accepted: load " << loadFactor << " | temperature "
                       << temperature << std::scientific << std::setprecision( 6 ) << " | maximum top u_y "
                       << finalMaximumTopDisplacement << " | max eq. plastic strain " << finalMaximumPlasticStrain;
                std::cout << status.str() << '\n';
            }
            if ( paraview )
            {
                paraview->SetCycle( ++outputCycle );
                paraview->SetTime( temperature );
                paraview->Save();
            }
        } );

    mfem::Vector zeroRightHandSide;
    nonlinearSolver.Mult( zeroRightHandSide, trueDisplacement );
    MFEM_VERIFY( nonlinearSolver.GetConverged(), "The thermal J2 solve did not reach the target temperature." );

    const mfem::real_t plasticStrainTolerance = precisionTolerance * ( 1. + std::abs( thermalStrain ) );
    const mfem::real_t displacementTolerance = precisionTolerance * characteristicLength;
    const bool plasticResponseActive = finalMaximumPlasticStrain > plasticStrainTolerance;
    const bool upwardProtrusion = finalMaximumTopDisplacement > displacementTolerance;
    if ( rank == 0 )
    {
        std::cout << "Thermal J2 response: " << ( plasticResponseActive ? "active" : "elastic" )
                  << "; protrusion response: " << ( upwardProtrusion ? "upward" : "not upward" ) << '\n';
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
