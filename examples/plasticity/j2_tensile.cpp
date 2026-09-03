#include "J2PlasticityIntegrator.h"
#include "PostProc.h"
#include "Solvers.h"

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
// Displacement-controlled, rate-independent plane-strain tensile test:
//   boundary 1: u_x = u_y = 0;
//   boundary 2: prescribed u_y, with u_x free;
//   boundaries 3 and 4: natural (traction-free) conditions.
// The continuation parameter is pseudo-time, not physical time. Stage 1 uses
// tau in [0, 1] to load from zero to maximumDisplacement; stage 2 uses tau in
// [1, 2] to unload to unloadFraction * maximumDisplacement.
mfem::real_t AverageTrueDofs( const mfem::Vector& values, const mfem::Array<int>& dofs )
{
    MFEM_VERIFY( dofs.Size() > 0, "The loaded boundary has no vertical true DOFs." );
    mfem::real_t sum = 0.;
    for ( int i = 0; i < dofs.Size(); i++ )
    {
        sum += values( dofs[i] );
    }
    return sum / dofs.Size();
}

int RunExample( int argc, char* argv[] )
{
    const char* meshFile = "data/simple_bar.msh";
    int order = 1;
    int refinementLevels = 0;
    int localRefinementLevels = 0;
    int loadSteps = 20;
    mfem::real_t youngsModulus = 210000.;
    mfem::real_t poissonRatio = .3;
    mfem::real_t initialYieldStress = 250.;
    mfem::real_t hardeningModulus = 1000.;
    mfem::real_t maximumDisplacement = .5;
    mfem::real_t unloadFraction = .75;
    bool output = true;
    const char* outputDirectory = "ParaView";

    mfem::OptionsParser options( argc, argv );
    options.AddOption( &meshFile, "-m", "--mesh", "Mesh file to use." );
    options.AddOption( &order, "-o", "--order", "H1 displacement order." );
    options.AddOption( &refinementLevels, "-r", "--refine-level", "Uniform mesh refinement levels." );
    options.AddOption( &localRefinementLevels, "-lr", "--local-refine-level",
                       "Local refinement levels along the horizontal centerline." );
    options.AddOption( &loadSteps, "-steps", "--load-steps", "Nominal number of adaptive load steps." );
    options.AddOption( &youngsModulus, "-E", "--youngs-modulus", "Young's modulus." );
    options.AddOption( &poissonRatio, "-nu", "--poisson-ratio", "Poisson ratio." );
    options.AddOption( &initialYieldStress, "-yield", "--yield-stress", "Initial uniaxial yield stress." );
    options.AddOption( &hardeningModulus, "-H", "--hardening-modulus", "Linear isotropic hardening modulus." );
    options.AddOption( &maximumDisplacement, "-disp", "--maximum-displacement", "Maximum prescribed top displacement." );
    options.AddOption( &unloadFraction, "-unload", "--unload-fraction",
                       "Final prescribed displacement divided by its maximum." );
    options.AddOption( &output, "-output", "--output", "-no-output", "--no-output", "Write ParaView output." );
    options.AddOption( &outputDirectory, "-odir", "--output-directory", "ParaView output prefix." );
    options.Parse();
    if ( !options.Good() )
    {
        options.PrintUsage( std::cout );
        return 1;
    }
    options.PrintOptions( std::cout );

    MFEM_VERIFY( order > 0 && loadSteps > 0, "Order and load steps must be positive." );
    MFEM_VERIFY( refinementLevels >= 0 && localRefinementLevels >= 0,
                 "Uniform and local refinement levels must be nonnegative." );
    MFEM_VERIFY( maximumDisplacement > 0., "Maximum prescribed displacement must be positive." );
    MFEM_VERIFY( unloadFraction >= 0. && unloadFraction <= 1., "The unload fraction must lie in [0, 1]." );

    constexpr int bottomBoundary = 1;
    constexpr int topBoundary = 2;
    mfem::Mesh mesh( meshFile );
    MFEM_VERIFY( mesh.Dimension() == 2, "The J2 tensile example requires a two-dimensional mesh." );
    MFEM_VERIFY( mesh.bdr_attributes.Find( bottomBoundary ) >= 0 && mesh.bdr_attributes.Find( topBoundary ) >= 0,
                 "The J2 tensile mesh requires bottom and top boundary attributes 1 and 2." );

    mfem::Vector coordinateMinimum;
    mfem::Vector coordinateMaximum;
    mesh.GetBoundingBox( coordinateMinimum, coordinateMaximum );
    const mfem::real_t coordinateScale = std::max( coordinateMinimum.Normlinf(), coordinateMaximum.Normlinf() );
    const mfem::real_t centerlineY = .5 * ( coordinateMinimum( 1 ) + coordinateMaximum( 1 ) );
    const mfem::real_t centerlineTolerance = 100. * std::numeric_limits<mfem::real_t>::epsilon() *
                                             std::max( coordinateScale, std::numeric_limits<mfem::real_t>::min() );

    for ( int level = 0; level < refinementLevels; level++ )
    {
        mesh.UniformRefinement();
    }
    for ( int level = 0; level < localRefinementLevels; level++ )
    {
        mfem::Array<mfem::Refinement> refinements;
        for ( int elementNumber = 0; elementNumber < mesh.GetNE(); elementNumber++ )
        {
            const mfem::Element* element = mesh.GetElement( elementNumber );
            for ( int vertexNumber = 0; vertexNumber < element->GetNVertices(); vertexNumber++ )
            {
                const mfem::real_t* vertex = mesh.GetVertex( element->GetVertices()[vertexNumber] );
                if ( std::abs( vertex[1] - centerlineY ) <= centerlineTolerance )
                {
                    refinements.Append( elementNumber );
                    break;
                }
            }
        }
        MFEM_VERIFY( refinements.Size() > 0, "Local refinement found no elements along the horizontal midline." );
        mesh.GeneralRefinement( refinements );
    }

    mfem::H1_FECollection displacementCollection( order, mesh.Dimension() );
    mfem::FiniteElementSpace displacementSpace( &mesh, &displacementCollection, mesh.Dimension(), mfem::Ordering::byVDIM );
    mfem::L2_FECollection plasticityCollection( 0, mesh.Dimension() );
    mfem::FiniteElementSpace plasticitySpace( &mesh, &plasticityCollection );

    mfem::Array<int> marker( mesh.bdr_attributes.Max() );
    mfem::Array<int> componentDofs;
    mfem::Array<int> essentialTrueDofs;
    // Clamp both displacement components on the bottom boundary.
    marker = 0;
    marker[bottomBoundary - 1] = 1;
    for ( int component = 0; component < mesh.Dimension(); component++ )
    {
        displacementSpace.GetEssentialTrueDofs( marker, componentDofs, component );
        essentialTrueDofs.Append( componentDofs );
    }

    marker = 0;
    marker[topBoundary - 1] = 1;
    mfem::Array<int> topVerticalDofs;
    // Prescribe only vertical motion at the top, allowing Poisson contraction in x.
    displacementSpace.GetEssentialTrueDofs( marker, topVerticalDofs, 1 );
    essentialTrueDofs.Append( topVerticalDofs );

    mfem::ConstantCoefficient youngsModulusCoefficient( youngsModulus );
    mfem::ConstantCoefficient poissonRatioCoefficient( poissonRatio );
    mfem::ConstantCoefficient initialYieldStressCoefficient( initialYieldStress );
    mfem::ConstantCoefficient hardeningModulusCoefficient( hardeningModulus );
    J2PlasticityMaterial material( youngsModulusCoefficient, poissonRatioCoefficient, initialYieldStressCoefficient,
                                   hardeningModulusCoefficient );
    plugin::J2PlasticityPointStorage pointStorage( &mesh );

    mfem::NonlinearForm residual( &displacementSpace );
    auto* plasticityIntegrator = new plugin::J2PlasticityIntegrator<plugin::J2PlasticityPointStorage>( material, pointStorage );
    residual.AddDomainIntegrator( plasticityIntegrator );
    residual.SetEssentialTrueDofs( essentialTrueDofs );

    const mfem::real_t precisionTolerance = 100. * std::numeric_limits<mfem::real_t>::epsilon();
#ifdef MFEM_USE_SUITESPARSE
    // Factor each assembled serial tangent directly when UMFPACK is available.
    mfem::UMFPackSolver tangentSolver;
    tangentSolver.SetPrintLevel( 0 );
#else
    // Associative J2 has a symmetric consistent tangent, so use a symmetric Krylov fallback.
    mfem::DSmoother diagonalPreconditioner;
    mfem::CGSolver tangentSolver;
    tangentSolver.SetPreconditioner( diagonalPreconditioner );
    tangentSolver.SetRelTol( std::max( static_cast<mfem::real_t>( 1e-12 ), precisionTolerance ) );
    tangentSolver.SetAbsTol( std::max( static_cast<mfem::real_t>( 1e-14 ), precisionTolerance ) );
    tangentSolver.SetMaxIter( 500 );
    tangentSolver.SetPrintLevel( -1 );
#endif

    const mfem::real_t maximumPseudoTimeStep = 1. / loadSteps;
    const mfem::real_t minimumPseudoTimeStep = std::max( static_cast<mfem::real_t>( 1e-8 ), precisionTolerance );
    plugin::MultiNewtonAdaptive<plugin::NewtonLineSearch> nonlinearSolver;
    const mfem::real_t residualScale = std::max( mfem::real_t{ 1. }, youngsModulus * std::abs( maximumDisplacement ) );
    nonlinearSolver.iterative_mode = true;
    nonlinearSolver.SetSolver( tangentSolver );
    nonlinearSolver.SetOperator( residual );
    nonlinearSolver.SetRelTol( std::max( static_cast<mfem::real_t>( 1e-9 ), precisionTolerance ) );
    nonlinearSolver.SetAbsTol( std::max( static_cast<mfem::real_t>( 1e-8 ), precisionTolerance * residualScale ) );
    nonlinearSolver.SetMaxIter( 10 );
    nonlinearSolver.SetLineSearch( true );
    nonlinearSolver.SetPrintLevel( -1 );
    nonlinearSolver.SetMaxDelta( maximumPseudoTimeStep );
    nonlinearSolver.SetMinDelta( minimumPseudoTimeStep );
    nonlinearSolver.SetMaxStep( 100 * loadSteps );

    mfem::GridFunction displacement( &displacementSpace );
    mfem::GridFunction equivalentPlasticStrain( &plasticitySpace );
    displacement = 0.;
    equivalentPlasticStrain = 0.;
    mfem::Vector trueDisplacement;
    displacement.GetTrueDofs( trueDisplacement );

    std::unique_ptr<mfem::ParaViewDataCollection> paraview;
    if ( output )
    {
        paraview = std::make_unique<mfem::ParaViewDataCollection>( "j2_tensile", &mesh );
        paraview->SetPrefixPath( outputDirectory );
        paraview->SetLevelsOfDetail( order );
        paraview->SetDataFormat( mfem::VTKFormat::BINARY );
        paraview->SetHighOrderOutput( true );
        paraview->RegisterField( "displacement", &displacement );
        paraview->RegisterField( "equivalent_plastic_strain", &equivalentPlasticStrain );
        paraview->SetCycle( 0 );
        paraview->SetTime( 0. );
        paraview->Save();
    }

    int outputCycle = 0;
    const auto writeOutput = [&]( const int stage, const mfem::real_t stageFactor, const mfem::real_t prescribedDisplacement )
    {
        const mfem::real_t pseudoTime = static_cast<mfem::real_t>( stage - 1 ) + stageFactor;
        displacement.SetFromTrueDofs( trueDisplacement );
        plugin::ProjectCommittedEquivalentPlasticStrain( pointStorage, equivalentPlasticStrain );
        const mfem::real_t averageTopDisplacement = AverageTrueDofs( trueDisplacement, topVerticalDofs );
        MFEM_VERIFY( std::abs( averageTopDisplacement - prescribedDisplacement ) <=
                         precisionTolerance * ( 1. + std::abs( prescribedDisplacement ) ),
                     "The top boundary does not satisfy its prescribed vertical displacement." );
        std::ostringstream status;
        status << std::fixed << std::setprecision( 6 ) << "Accepted: stage " << stage << " | factor " << stageFactor
               << " | pseudo-time " << pseudoTime << std::scientific << std::setprecision( 6 ) << " | top u_y "
               << prescribedDisplacement << " | max eq. plastic strain " << equivalentPlasticStrain.Max();
        std::cout << status.str() << '\n';
        if ( paraview )
        {
            paraview->SetCycle( ++outputCycle );
            paraview->SetTime( pseudoTime );
            paraview->Save();
        }
    };

    mfem::Vector zeroRightHandSide;
    const auto solveStage = [&]( const int stage, const mfem::real_t initialValue, const mfem::real_t targetValue )
    {
        const mfem::real_t initialPseudoTime = static_cast<mfem::real_t>( stage - 1 );
        const mfem::real_t finalPseudoTime = static_cast<mfem::real_t>( stage );
        nonlinearSolver.SetPseudoTimeInterval( initialPseudoTime, finalPseudoTime );
        nonlinearSolver.SetDelta( maximumPseudoTimeStep );
        nonlinearSolver.SetTrialStateFunc(
            [&, initialPseudoTime, initialValue, targetValue]( const mfem::real_t pseudoTime, mfem::Vector& trialDisplacement )
            {
                const mfem::real_t stageFactor = pseudoTime - initialPseudoTime;
                const mfem::real_t prescribedDisplacement = ( 1. - stageFactor ) * initialValue + stageFactor * targetValue;
                // MFEM eliminates essential rows and columns, so install the trial
                // Dirichlet values before Newton and leave them fixed during it.
                for ( int i = 0; i < topVerticalDofs.Size(); i++ )
                {
                    trialDisplacement( topVerticalDofs[i] ) = prescribedDisplacement;
                }
            } );
        nonlinearSolver.SetDataCollectionFunc(
            [&, stage, initialPseudoTime, initialValue, targetValue]( int, int, const mfem::real_t pseudoTime )
            {
                const mfem::real_t stageFactor = pseudoTime - initialPseudoTime;
                const mfem::real_t prescribedDisplacement = ( 1. - stageFactor ) * initialValue + stageFactor * targetValue;
                // Newton has converged and committed the quadrature history before this callback.
                writeOutput( stage, stageFactor, prescribedDisplacement );
            } );
        nonlinearSolver.Mult( zeroRightHandSide, trueDisplacement );
        MFEM_VERIFY( nonlinearSolver.GetConverged(), "The J2 tensile solve did not reach its pseudo-time target." );
    };

    solveStage( 1, 0., maximumDisplacement );
    const mfem::real_t loadedPlasticStrain = equivalentPlasticStrain.Max();
    const mfem::Vector loadedPlasticStrainField( equivalentPlasticStrain );
    solveStage( 2, maximumDisplacement, unloadFraction * maximumDisplacement );
    mfem::Vector plasticStrainChange( equivalentPlasticStrain );
    plasticStrainChange -= loadedPlasticStrainField;
    const mfem::real_t historyTolerance = precisionTolerance * ( 1. + loadedPlasticStrain );
    const bool plasticResponseActive = loadedPlasticStrain > historyTolerance;
    const bool unloadingWasElastic = plasticStrainChange.Normlinf() <= historyTolerance;

    std::cout << "Plastic response: " << ( plasticResponseActive ? "active" : "elastic" )
              << "; unloading response: " << ( unloadingWasElastic ? "elastic" : "plastic" ) << '\n';
    return 0;
}
} // namespace

int main( int argc, char* argv[] )
{
    return RunExample( argc, argv );
}
