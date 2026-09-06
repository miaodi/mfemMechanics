#include "PhaseField.h"
#include "Plugin.h"
#include "mfem.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <type_traits>

namespace
{
constexpr int kBottomBoundary = 11;
constexpr int kTopBoundary = 12;

void VerifyMesh( const mfem::Mesh& mesh )
{
    MFEM_VERIFY( mesh.Dimension() == 2 && mesh.SpaceDimension() == 2,
                 "PhaseField_shear requires a two-dimensional planar mesh." );
    MFEM_VERIFY( mesh.attributes.Size() == 1 && mesh.attributes.Find( 1 ) >= 0,
                 "PhaseField_shear requires domain attribute 1." );
    MFEM_VERIFY( mesh.bdr_attributes.Find( kBottomBoundary ) >= 0 && mesh.bdr_attributes.Find( kTopBoundary ) >= 0,
                 "PhaseField_shear requires bottom boundary attribute 11 and top boundary attribute 12." );
}

void VerifyPhaseBounds( const mfem::GridFunction& phaseField )
{
    const mfem::real_t tolerance = std::is_same_v<mfem::real_t, float> ? 1e-4f : 1e-8;
    MFEM_VERIFY( mfem::IsFinite( phaseField.Min() ) && mfem::IsFinite( phaseField.Max() ),
                 "The phase-field solution contains a non-finite value." );
    MFEM_VERIFY( phaseField.Min() >= -tolerance && phaseField.Max() <= 1. + tolerance,
                 "The phase-field solution left its admissible interval [0, 1]." );
}
} // namespace

int main( int argc, char* argv[] )
{
    const char* meshFile = "data/crack_square2d.msh";
    const char* outputDirectory = "ParaView";
    int order = 1;
    int refinementLevels = 4;
    int localRefinementLevels = 0;
    int maximumSteps = 100000;
    int outputInterval = 20;
    bool output = true;
    mfem::real_t maximumDisplacement = 1e-4;
    mfem::real_t finalPseudoTime = 1.;
    mfem::real_t initialPseudoTimeStep = 2e-6;
    mfem::real_t maximumPseudoTimeStep = 1e-4;
    mfem::real_t minimumPseudoTimeStep = 1e-14;
    PhaseFieldFractureParameters fractureParameters;

    mfem::OptionsParser args( argc, argv );
    args.AddOption( &meshFile, "-m", "--mesh", "Mesh file to use." );
    args.AddOption( &order, "-o", "--order", "Finite element order." );
    args.AddOption( &refinementLevels, "-r", "--refine-level", "Uniform refinement levels." );
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
    args.AddOption( &outputInterval, "-oi", "--output-interval", "Accepted steps between ParaView writes." );
    args.AddOption( &outputDirectory, "-od", "--output-directory", "ParaView output directory." );
    args.AddOption( &output, "-vis", "--visualization", "-no-vis", "--no-visualization",
                    "Enable ParaView and force-curve output." );
    args.Parse();
    if ( !args.Good() )
    {
        args.PrintUsage( std::cout );
        return 1;
    }
    args.PrintOptions( std::cout );

    MFEM_VERIFY( order >= 1 && refinementLevels >= 0 && localRefinementLevels >= 0,
                 "Finite element order must be positive and refinement levels must be nonnegative." );
    MFEM_VERIFY( maximumSteps > 0 && outputInterval > 0, "Step and output intervals must be positive." );
    MFEM_VERIFY( std::isfinite( maximumDisplacement ) && maximumDisplacement >= 0.,
                 "Maximum displacement must be finite and nonnegative." );
    MFEM_VERIFY( std::isfinite( finalPseudoTime ) && finalPseudoTime > 0.,
                 "Final pseudo-time must be finite and positive." );

    MFEM_VERIFY( std::isfinite( minimumPseudoTimeStep ) && std::isfinite( initialPseudoTimeStep ) &&
                     std::isfinite( maximumPseudoTimeStep ) && minimumPseudoTimeStep > 0. && minimumPseudoTimeStep <= initialPseudoTimeStep &&
                     initialPseudoTimeStep <= maximumPseudoTimeStep && minimumPseudoTimeStep < maximumPseudoTimeStep,
                 "Continuation increments must be finite and satisfy 0 < minimum <= initial <= maximum, with minimum < "
                 "maximum." );

    mfem::Mesh mesh( meshFile, 1, 1 );
    VerifyMesh( mesh );
    if ( mesh.NURBSext )
    {
        mesh.DegreeElevate( order, order );
    }
    for ( int level = 0; level < refinementLevels; level++ )
    {
        mesh.UniformRefinement();
    }
    for ( int level = 0; level < localRefinementLevels; level++ )
    {
        mfem::Array<mfem::Refinement> refinements;
        const auto elements = mesh.GetElementsArray();
        for ( int element = 0; element < mesh.GetNE(); element++ )
        {
            for ( int vertex = 0; vertex < elements[element]->GetNVertices(); vertex++ )
            {
                const double* coordinate = mesh.GetVertex( elements[element]->GetVertices()[vertex] );
                if ( coordinate[1] > -1e-5 && coordinate[1] < 8e-4 && coordinate[0] > -1e-5 && coordinate[0] < 6e-4 )
                {
                    refinements.Append( element );
                    break;
                }
            }
        }
        mesh.GeneralRefinement( refinements );
    }

    const int dimension = mesh.Dimension();
    mfem::H1_FECollection collection( order, dimension );
    mfem::FiniteElementSpace displacementSpace( &mesh, &collection, dimension, mfem::Ordering::byVDIM );
    mfem::FiniteElementSpace phaseSpace( &mesh, &collection );
    mfem::Array<mfem::FiniteElementSpace*> spaces( 2 );
    spaces[0] = &displacementSpace;
    spaces[1] = &phaseSpace;

    std::cout << "Displacement true DOFs: " << displacementSpace.GetTrueVSize() << '\n'
              << "Phase-field true DOFs: " << phaseSpace.GetTrueVSize() << '\n';

    mfem::Array<int> displacementBoundaryMarker( mesh.bdr_attributes.Max() );
    displacementBoundaryMarker = 0;
    displacementBoundaryMarker[kBottomBoundary - 1] = 1;
    displacementBoundaryMarker[kTopBoundary - 1] = 1;
    mfem::Array<int> phaseBoundaryMarker( mesh.bdr_attributes.Max() );
    phaseBoundaryMarker = 0;
    mfem::Array<int> topBoundaryMarker( mesh.bdr_attributes.Max() );
    topBoundaryMarker = 0;
    topBoundaryMarker[kTopBoundary - 1] = 1;
    mfem::Array<int> constrainedDisplacementDofs;
    mfem::Array<int> loadedHorizontalDofs;
    displacementSpace.GetEssentialTrueDofs( displacementBoundaryMarker, constrainedDisplacementDofs );
    displacementSpace.GetEssentialTrueDofs( topBoundaryMarker, loadedHorizontalDofs, 0 );
    MFEM_VERIFY( loadedHorizontalDofs.Size() > 0, "Top boundary attribute 12 has no horizontal true DOFs." );

    mfem::Array<int> blockOffsets( 3 );
    blockOffsets[0] = 0;
    blockOffsets[1] = displacementSpace.GetTrueVSize();
    blockOffsets[2] = phaseSpace.GetTrueVSize();
    blockOffsets.PartialSum();
    mfem::BlockVector solution( blockOffsets );
    solution = 0.;
    mfem::GridFunction displacement( &displacementSpace );
    mfem::GridFunction phaseField( &phaseSpace );
    displacement = 0.;
    phaseField = 0.;

    mfem::ConstantCoefficient youngsModulus( 210e9 );
    mfem::ConstantCoefficient poissonRatio( .3 );
    PhaseFieldElasticMaterial material(
        youngsModulus, poissonRatio, PhaseFieldElasticMaterial::StrainEnergySplit::MieheSpectral, fractureParameters );
    plugin::PhaseFieldPointStorage pointStorage( &mesh );

    mfem::BlockNonlinearForm residual( spaces );
    residual.AddDomainIntegrator( new plugin::PhaseFieldIntegrator<plugin::PhaseFieldPointStorage>( material, pointStorage ) );
    mfem::Array<mfem::Array<int>*> essentialBoundaryMarkers( 2 );
    essentialBoundaryMarkers[0] = &displacementBoundaryMarker;
    essentialBoundaryMarkers[1] = &phaseBoundaryMarker;
    mfem::Array<mfem::Vector*> essentialRightHandSides( 2 );
    essentialRightHandSides = nullptr;
    residual.SetEssentialBC( essentialBoundaryMarkers, essentialRightHandSides );

    mfem::BlockNonlinearForm internalResidual( spaces );
    auto* reactionIntegrator = new plugin::PhaseFieldIntegrator<plugin::PhaseFieldPointStorage>( material, pointStorage );
    internalResidual.AddDomainIntegrator( reactionIntegrator );

    mfem::UMFPackSolver tangentSolver;
    plugin::MultiNewtonAdaptive<plugin::NewtonForPhaseField> nonlinearSolver;
    nonlinearSolver.iterative_mode = true;
    nonlinearSolver.SetSolver( tangentSolver );
    nonlinearSolver.SetOperator( residual );
    nonlinearSolver.SetRelTol( std::is_same_v<mfem::real_t, float> ? 1e-4f : 1e-7 );
    const mfem::real_t precisionTolerance = mfem::real_t( 100 ) * std::numeric_limits<mfem::real_t>::epsilon();
    nonlinearSolver.SetAbsTol( std::max(
        std::numeric_limits<mfem::real_t>::min(),
        precisionTolerance * youngsModulus.constant * std::max( maximumDisplacement, mfem::real_t( 1e-12 ) ) ) );
    nonlinearSolver.SetMaxIter( 25 );
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

    mfem::H1_FECollection stressCollection( order, dimension );
    mfem::FiniteElementSpace stressSpace( &mesh, &stressCollection, 7 );
    mfem::GridFunction stress( &stressSpace );
    plugin::StressCoefficient stressCoefficient( dimension, material );
    stressCoefficient.SetDisplacement( displacement );
    stressCoefficient.SetPhaseField( phaseField );
    plugin::ParaView2DVectorCoefficient paraviewDisplacement( displacement );

    std::unique_ptr<mfem::ParaViewDataCollection> paraview;
    std::ofstream forceCurve;
    if ( output )
    {
        paraview = std::make_unique<mfem::ParaViewDataCollection>( "phase_field_square_shear", &mesh );
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

        forceCurve.open( "phase_field_force.csv" );
        MFEM_VERIFY( forceCurve, "Could not open phase_field_force.csv for writing." );
        forceCurve << "displacement,reaction_force\n0,0\n";
    }

    mfem::BlockVector unconstrainedResidual( blockOffsets );
    int acceptedSteps = 0;
    nonlinearSolver.SetDataCollectionFunc(
        [&]( const int, const int, const mfem::real_t pseudoTime )
        {
            acceptedSteps++;
            displacement.SetFromTrueDofs( solution.GetBlock( 0 ) );
            phaseField.SetFromTrueDofs( solution.GetBlock( 1 ) );
            VerifyPhaseBounds( phaseField );
            if ( forceCurve.is_open() )
            {
                internalResidual.Mult( solution, unconstrainedResidual );
                mfem::real_t reaction = 0.;
                for ( int i = 0; i < loadedHorizontalDofs.Size(); i++ )
                {
                    reaction += unconstrainedResidual.GetBlock( 0 )( loadedHorizontalDofs[i] );
                }
                forceCurve << maximumDisplacement * pseudoTime / finalPseudoTime << ',' << reaction << '\n';
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
                 "PhaseField_shear did not reach the requested final pseudo-time." );
    displacement.SetFromTrueDofs( solution.GetBlock( 0 ) );
    phaseField.SetFromTrueDofs( solution.GetBlock( 1 ) );
    VerifyPhaseBounds( phaseField );
    return 0;
}
