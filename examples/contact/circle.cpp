#include "Contact.h"
#include "FEMPlugin.h"
#include "GapSampling.h"
#include "J2Plasticity.h"
#include "Material.h"
#include "PostProc.h"
#include "SolidMechanicsIntegrator.h"
#include "Solvers.h"

#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mfem.hpp>
#include <stdexcept>
#include <string>

namespace
{
void SaveParaView( mfem::ParaViewDataCollection& collection )
{
    collection.Save();
    if ( collection.Error() != mfem::DataCollection::No_Error )
    {
        throw std::runtime_error( "ParaView output failed for " + collection.GetPrefixPath() + collection.GetCollectionName() );
    }
}

mfem::Vector MakeVector( const std::initializer_list<mfem::real_t> values )
{
    mfem::Vector vector( static_cast<int>( values.size() ) );
    int index = 0;
    for ( const mfem::real_t value : values )
    {
        vector( index++ ) = value;
    }
    return vector;
}

struct ContactDiagnostics
{
    explicit ContactDiagnostics( const int dimension ) : Resultant( dimension )
    {
        Resultant = 0.;
    }

    mfem::real_t MinimumGap{ std::numeric_limits<mfem::real_t>::infinity() };
    mfem::real_t MaximumPenetration{ 0. };
    mfem::real_t MinimumPressure{ std::numeric_limits<mfem::real_t>::infinity() };
    mfem::real_t MaximumPressure{ 0. };
    mfem::real_t PressureIntegral{ 0. };
    mfem::real_t PressureMomentX{ 0. };
    mfem::Vector Resultant;
    int QuadraturePointCount{ 0 };
    int ActiveQuadraturePointCount{ 0 };
};

ContactDiagnostics MeasureContact( mfem::Mesh& mesh,
                                   mfem::FiniteElementSpace& displacementSpace,
                                   const mfem::GridFunction& displacement,
                                   const plugin::RigidObstacle& obstacle,
                                   const mfem::real_t penalty,
                                   const int contactBoundaryAttribute,
                                   const mfem::IntegrationRule& integrationRule )
{
    ContactDiagnostics diagnostics( mesh.SpaceDimension() );
    mfem::Array<int> elementDofs;
    mfem::Vector elementDisplacement;
    mfem::Vector shape;
    mfem::Vector referencePosition;
    mfem::Vector currentPosition;
    plugin::SignedDistanceEvaluation evaluation( mesh.SpaceDimension() );

    for ( int boundaryElement = 0; boundaryElement < mesh.GetNBE(); boundaryElement++ )
    {
        if ( mesh.GetBdrAttribute( boundaryElement ) != contactBoundaryAttribute )
        {
            continue;
        }

        auto* transformation = mesh.GetBdrFaceTransformations( boundaryElement );
        MFEM_VERIFY( transformation != nullptr, "The contact boundary must have an adjacent volume element." );
        const auto* element = displacementSpace.GetFE( transformation->Elem1No );
        MFEM_VERIFY( element != nullptr, "The contact boundary has no displacement finite element." );
        displacementSpace.GetElementVDofs( transformation->Elem1No, elementDofs );
        displacement.GetSubVector( elementDofs, elementDisplacement );
        shape.SetSize( element->GetDof() );
        referencePosition.SetSize( mesh.SpaceDimension() );
        currentPosition.SetSize( mesh.SpaceDimension() );

        for ( int point = 0; point < integrationRule.GetNPoints(); point++ )
        {
            const mfem::IntegrationPoint& integrationPoint = integrationRule.IntPoint( point );
            transformation->SetAllIntPoints( &integrationPoint );
            element->CalcShape( transformation->GetElement1IntPoint(), shape );
            transformation->Transform( integrationPoint, referencePosition );
            currentPosition = referencePosition;
            for ( int component = 0; component < mesh.SpaceDimension(); component++ )
            {
                for ( int degreeOfFreedom = 0; degreeOfFreedom < element->GetDof(); degreeOfFreedom++ )
                {
                    currentPosition( component ) +=
                        shape( degreeOfFreedom ) * elementDisplacement( degreeOfFreedom + component * element->GetDof() );
                }
            }

            obstacle.Evaluate( currentPosition, evaluation );
            const mfem::real_t pressure = penalty * std::max( -evaluation.Gap, mfem::real_t{ 0. } );
            const mfem::real_t weight = integrationPoint.weight * transformation->Weight();
            diagnostics.MinimumGap = std::min( diagnostics.MinimumGap, evaluation.Gap );
            diagnostics.MaximumPenetration = std::max( diagnostics.MaximumPenetration, -evaluation.Gap );
            diagnostics.MinimumPressure = std::min( diagnostics.MinimumPressure, pressure );
            diagnostics.MaximumPressure = std::max( diagnostics.MaximumPressure, pressure );
            diagnostics.Resultant.Add( weight * pressure, evaluation.Gradient );
            diagnostics.PressureIntegral += weight * pressure;
            diagnostics.PressureMomentX += weight * pressure * currentPosition( 0 );
            diagnostics.QuadraturePointCount++;
            diagnostics.ActiveQuadraturePointCount += pressure > 0. ? 1 : 0;
        }
    }

    MFEM_VERIFY( diagnostics.QuadraturePointCount > 0, "The selected contact boundary has no quadrature points." );
    return diagnostics;
}

int RunExample( int argc, char* argv[] )
{
    int elementsX = 4;
    int elementsY = 4;
    int refinementLevels = 0;
    int order = 1;
    int loadSteps = 4;
    const char* loadPath = "load";
    const char* materialName = "elastic";
    mfem::real_t yieldStress = 1.;
    mfem::real_t hardeningModulus = 100.;
    // 64 Gauss points/face: benchmark sensitivity at orders 127/255/511 is
    // recorded in docs/frictionless-penalty-contact.md. Not a collision bound.
    int contactIntegrationOrder = 127;
    int gapSampleIntervals = 200;
    mfem::real_t width = 1.;
    mfem::real_t height = 1.;
    mfem::real_t youngsModulus = 1000.;
    mfem::real_t poissonRatio = .3;
    mfem::real_t maximumBottomDisplacement = .02;
    mfem::real_t circleCenterX = .65;
    mfem::real_t circleRadius = .25;
    mfem::real_t initialClearance = .01;
    mfem::real_t penaltyFactor = 10.;
    bool semismoothContact = false;
    bool output = true;
    const char* outputDirectory = "ParaView";

    mfem::OptionsParser options( argc, argv );
    options.AddOption( &elementsX, "-nx", "--elements-x", "Number of elements along the contact boundary." );
    options.AddOption( &elementsY, "-ny", "--elements-y", "Number of elements through the block height." );
    options.AddOption( &refinementLevels, "-r", "--refine-level", "Uniform mesh refinement levels." );
    options.AddOption( &order, "-o", "--order", "H1 displacement order." );
    options.AddOption( &loadSteps, "-steps", "--load-steps",
                       "Number of increments per stage (adaptive subdivision allowed)." );
    options.AddOption( &loadPath, "-lp", "--load-path", "load or load-return; return retains supports." );
    options.AddOption( &materialName, "-mat", "--material", "elastic or j2 (small-strain plane strain)." );
    options.AddOption( &yieldStress, "-sy", "--yield-stress", "Initial J2 yield stress." );
    options.AddOption( &hardeningModulus, "-H", "--hardening-modulus", "J2 isotropic hardening modulus." );
    options.AddOption( &contactIntegrationOrder, "-cq", "--contact-integration-order",
                       "Contact Gauss integration order (circle gap and active-set kink are nonpolynomial)." );
    options.AddOption( &gapSampleIntervals, "-gs", "--gap-sample-intervals",
                       "Independent uniform gap-sampling intervals per contact face, including endpoints." );
    options.AddOption( &width, "-w", "--width", "Block width." );
    options.AddOption( &height, "-hy", "--height", "Block height." );
    options.AddOption( &youngsModulus, "-E", "--youngs-modulus", "Young's modulus." );
    options.AddOption( &poissonRatio, "-nu", "--poisson-ratio", "Poisson ratio." );
    options.AddOption( &maximumBottomDisplacement, "-disp", "--maximum-displacement",
                       "Maximum prescribed upward displacement on the bottom boundary." );
    options.AddOption( &circleCenterX, "-cx", "--circle-center-x", "Horizontal coordinate of the circle center." );
    options.AddOption( &circleRadius, "-R", "--circle-radius", "Rigid-circle radius." );
    options.AddOption( &initialClearance, "-gap", "--initial-clearance",
                       "Initial vertical clearance between the block and circle." );
    options.AddOption(
        &penaltyFactor, "-gamma", "--penalty-factor",
        "Dimensionless alpha in kappa = alpha E / h; the legacy -gamma flag is not paper compliance gamma." );
    options.AddOption( &semismoothContact, "-sm", "--semismooth", "-penalty", "--penalty",
                       "Use the monolithic displacement--boundary-multiplier semismooth formulation." );
    options.AddOption( &output, "-output", "--output", "-no-output", "--no-output", "Write ParaView output." );
    options.AddOption( &outputDirectory, "-odir", "--output-directory", "ParaView output prefix." );
    options.Parse();
    if ( !options.Good() )
    {
        options.PrintUsage( std::cout );
        return 1;
    }
    options.PrintOptions( std::cout );
    const bool returnToZero = std::string( loadPath ) == "load-return";
    const bool useJ2 = std::string( materialName ) == "j2";
    MFEM_VERIFY( returnToZero || std::string( loadPath ) == "load", "Use --load-path load|load-return." );
    MFEM_VERIFY( useJ2 || std::string( materialName ) == "elastic", "Use --material elastic|j2." );
    MFEM_VERIFY( std::isfinite( yieldStress ) && yieldStress > 0. && std::isfinite( hardeningModulus ) && hardeningModulus >= 0.,
                 "Yield stress must be positive and hardening nonnegative, both finite." );
    const mfem::real_t finalTime = returnToZero ? 2. : 1.;
    const mfem::real_t finalBoundaryDisplacement = returnToZero ? 0. : maximumBottomDisplacement;
    const auto prescribedMotion = [&]( const mfem::real_t time )
    { return maximumBottomDisplacement * ( time <= 1. ? time : 2. - time ); };

    MFEM_VERIFY( elementsX > 0 && elementsY > 0 && order > 0 && loadSteps > 0,
                 "Element counts, displacement order, and load steps must be positive." );
    MFEM_VERIFY( refinementLevels >= 0, "Uniform mesh refinement levels must be nonnegative." );
    MFEM_VERIFY( contactIntegrationOrder >= 2 * order,
                 "Contact integration order must be at least twice the displacement order; increase -cq." );
    MFEM_VERIFY( gapSampleIntervals > 0, "Gap sampling intervals must be positive." );
    MFEM_VERIFY( std::isfinite( width ) && std::isfinite( height ) && width > 0. && height > 0.,
                 "Block dimensions must be positive and finite." );
    MFEM_VERIFY( std::isfinite( youngsModulus ) && youngsModulus > 0., "Young's modulus must be positive and finite." );
    MFEM_VERIFY( std::isfinite( poissonRatio ) && poissonRatio > -1. && poissonRatio < .5,
                 "Poisson's ratio must lie strictly between -1 and 0.5." );
    MFEM_VERIFY( std::isfinite( maximumBottomDisplacement ) && maximumBottomDisplacement > 0.,
                 "The maximum bottom displacement must be positive and finite." );
    MFEM_VERIFY( std::isfinite( circleCenterX ), "The circle-center coordinate must be finite." );
    MFEM_VERIFY( std::isfinite( circleRadius ) && circleRadius > 0., "The circle radius must be positive and finite." );
    MFEM_VERIFY( std::isfinite( initialClearance ) && initialClearance >= 0.,
                 "The initial circle clearance must be nonnegative and finite." );
    MFEM_VERIFY( maximumBottomDisplacement > initialClearance,
                 "The prescribed displacement must exceed the initial clearance to activate contact." );
    MFEM_VERIFY( circleCenterX + circleRadius > 0. && circleCenterX - circleRadius < width,
                 "The circle must overlap the block's horizontal span." );
    MFEM_VERIFY( std::isfinite( penaltyFactor ) && penaltyFactor > 0.,
                 "The penalty factor must be positive and finite." );

    constexpr int bottomBoundary = 1;
    constexpr int topBoundary = 3;
    constexpr int leftBoundary = 4;
    mfem::Mesh mesh = mfem::Mesh::MakeCartesian2D( elementsX, elementsY, mfem::Element::QUADRILATERAL, true, width, height );
    mfem::real_t contactFaceSize = width / elementsX;
    for ( int level = 0; level < refinementLevels; level++ )
    {
        mesh.UniformRefinement();
        contactFaceSize *= .5;
    }
    MFEM_VERIFY( mesh.bdr_attributes.Find( bottomBoundary ) >= 0 && mesh.bdr_attributes.Find( topBoundary ) >= 0 &&
                     mesh.bdr_attributes.Find( leftBoundary ) >= 0,
                 "The Cartesian contact mesh is missing a required boundary attribute." );
    MFEM_VERIFY( std::isfinite( contactFaceSize ) && contactFaceSize > 0.,
                 "Uniform refinement produced an invalid contact-face size." );
    int expectedContactFaceCount = 0;
    for ( int boundaryElement = 0; boundaryElement < mesh.GetNBE(); boundaryElement++ )
    {
        expectedContactFaceCount += mesh.GetBdrAttribute( boundaryElement ) == topBoundary ? 1 : 0;
    }
    MFEM_VERIFY( expectedContactFaceCount > 0, "The selected contact boundary is empty." );

    mfem::H1_FECollection displacementCollection( order, mesh.Dimension() );
    mfem::FiniteElementSpace displacementSpace( &mesh, &displacementCollection, mesh.Dimension(), mfem::Ordering::byVDIM );
    mfem::Array<int> contactBoundary( mesh.bdr_attributes.Max() );
    mfem::Array<int> drivenBoundary( mesh.bdr_attributes.Max() );
    mfem::Array<int> leftBoundaryMarker( mesh.bdr_attributes.Max() );
    contactBoundary = 0;
    drivenBoundary = 0;
    leftBoundaryMarker = 0;
    contactBoundary[topBoundary - 1] = 1;
    drivenBoundary[bottomBoundary - 1] = 1;
    leftBoundaryMarker[leftBoundary - 1] = 1;
    mfem::Array<int> bottomVerticalDofs;
    mfem::Array<int> leftHorizontalDofs;
    mfem::Array<int> essentialTrueDofs;
    displacementSpace.GetEssentialTrueDofs( drivenBoundary, bottomVerticalDofs, 1 );
    displacementSpace.GetEssentialTrueDofs( leftBoundaryMarker, leftHorizontalDofs, 0 );
    essentialTrueDofs.Append( bottomVerticalDofs );
    essentialTrueDofs.Append( leftHorizontalDofs );

    const mfem::real_t penalty = penaltyFactor * youngsModulus / contactFaceSize;
    MFEM_VERIFY( std::isfinite( penalty ) && penalty > 0.,
                 "The computed contact stiffness must be positive and finite." );
    mfem::ConstantCoefficient youngsModulusCoefficient( youngsModulus );
    mfem::ConstantCoefficient poissonRatioCoefficient( poissonRatio );
    IsotropicElasticMaterial material( youngsModulusCoefficient, poissonRatioCoefficient );
    plugin::IntegrationPointStorage<> pointStorage( &mesh );
    mfem::ConstantCoefficient yieldCoefficient( yieldStress ), hardeningCoefficient( hardeningModulus );
    J2PlasticityMaterial plasticMaterial( youngsModulusCoefficient, poissonRatioCoefficient, yieldCoefficient, hardeningCoefficient );
    plugin::SolidMechanicsPointStorage<J2PlasticityMaterial> plasticStorage( &mesh );
    mfem::L2_FECollection plasticCollection( 0, mesh.Dimension() );
    mfem::FiniteElementSpace plasticSpace( &mesh, &plasticCollection );
    mfem::GridFunction equivalentPlasticStrain( &plasticSpace );
    equivalentPlasticStrain = 0.;
    const auto updatePlasticOutput = [&]()
    {
        if ( useJ2 )
        {
            plugin::ProjectCommittedEquivalentPlasticStrain( plasticStorage, equivalentPlasticStrain );
        }
    };
    const auto reportResidualShape = [&]( const mfem::Vector& displacement )
    {
        updatePlasticOutput();
        for ( int i = 0; i < leftHorizontalDofs.Size(); ++i )
        {
            MFEM_VERIFY( displacement( leftHorizontalDofs[i] ) == 0., "The left horizontal support moved." );
        }
        std::cout << std::scientific << std::setprecision( 8 )
                  << "Final displacement infinity norm: " << displacement.Normlinf() << '\n'
                  << "Maximum committed equivalent plastic strain (cell average): " << equivalentPlasticStrain.Max() << '\n';
    };
    const mfem::real_t circleCenterY = height + initialClearance + circleRadius;
    const mfem::Vector circleCenter( MakeVector( { circleCenterX, circleCenterY } ) );
    plugin::RigidSphereObstacle obstacle( circleCenter, circleRadius );
    // MFEM owns this stable rule. Hold it fixed throughout all residual/Jacobian
    // evaluations: changing the samples during Newton changes the discrete problem.
    const auto& contactRule = mfem::IntRules.Get( mfem::Geometry::SEGMENT, contactIntegrationOrder );

    mfem::NonlinearForm residual( &displacementSpace );
    if ( useJ2 )
    {
        residual.AddDomainIntegrator( new plugin::SolidMechanicsIntegrator<J2PlasticityMaterial>( plasticMaterial, plasticStorage ) );
    }
    else
    {
        auto* elasticityIntegrator = new plugin::NonlinearElasticityIntegrator( material, pointStorage );
        elasticityIntegrator->setNonlinear( false );
        residual.AddDomainIntegrator( elasticityIntegrator );
    }

    if ( semismoothContact )
    {
        residual.SetEssentialTrueDofs( essentialTrueDofs );

        mfem::Array<int> contactBoundaryAttributes( 1 );
        contactBoundaryAttributes[0] = topBoundary;
        // Default P0 is deliberate: the benchmark checks and diagonal multiplier
        // preconditioner below assume one multiplier DOF per contact face.
        // This owner outlives contactOperator and all borrowed output objects.
        plugin::BoundaryMultiplierSpace boundaryMultiplierSpace( mesh, contactBoundaryAttributes );
        mfem::IdentityOperator primalToDisplacement( displacementSpace.GetTrueVSize() );
        const mfem::real_t primalResidualReference = youngsModulus * maximumBottomDisplacement * width / height;
        const mfem::real_t multiplierResidualReference = maximumBottomDisplacement * width;
        const mfem::real_t primalResidualScale = 1. / primalResidualReference;
        const mfem::real_t multiplierResidualScale = 1. / multiplierResidualReference;
        // Paper gamma is compliance (length/stress), not the dimensionless CLI
        // penalty factor. Delta controls pressure-jump stabilization; zero disables it.
        const mfem::real_t gamma = 1. / penalty;
        constexpr mfem::real_t delta = 1.;
        plugin::SemismoothRigidContactOperator contactOperator( residual, primalToDisplacement, displacementSpace,
                                                                obstacle, boundaryMultiplierSpace, essentialTrueDofs, gamma,
                                                                primalResidualScale, multiplierResidualScale, delta );
        contactOperator.SetIntegrationRule( &contactRule );

        mfem::BlockVector unknown( contactOperator.GetBlockOffsets() );
        unknown = 0.;
        mfem::GridFunction displacement( &displacementSpace );
        displacement = 0.;
        mfem::Vector initialTrueDisplacement;
        displacement.GetTrueDofs( initialTrueDisplacement );
        unknown.GetBlock( 0 ) = initialTrueDisplacement;
        unknown.GetBlock( 1 ) = 0.;

        const plugin::SemismoothContactDiagnostics initialDiagnostics = contactOperator.ComputeContactDiagnostics( unknown );

        // Output borrows the boundary owner, without extending its lifetime.
        mfem::GridFunction boundaryMultiplier( &boundaryMultiplierSpace.GetSpace() );
        boundaryMultiplier = 0.;
        mfem::Vector trueBoundaryMultiplier;
        plugin::ParaView2DVectorCoefficient paraviewDisplacement( displacement );
        std::unique_ptr<mfem::ParaViewDataCollection> bodyParaview;
        std::unique_ptr<mfem::ParaViewDataCollection> multiplierParaview;
        int outputCycle = 0;

        mfem::SparseMatrix primalPreconditionerMatrix;
#ifdef MFEM_USE_SUITESPARSE
        mfem::UMFPackSolver primalBlockSolver;
#else
        mfem::GSSmoother primalBlockSolver( mfem::GSSmoother::SYMMETRIC, 2 );
#endif
        mfem::Array<int> noEssentialDofs;
        mfem::Vector multiplierDiagonal( contactOperator.GetMultiplierSpace().GetTrueVSize() );
        // This is the magnitude of the inactive P0 multiplier block on a
        // uniform contact face and has the same h^2/E Schur-complement scale.
        multiplierDiagonal = multiplierResidualScale * gamma * contactFaceSize;
        mfem::OperatorJacobiSmoother multiplierBlockSolver( multiplierDiagonal, noEssentialDofs );
        mfem::BlockDiagonalPreconditioner blockPreconditioner( contactOperator.GetBlockOffsets() );
        mfem::GMRESSolver tangentSolver;
        plugin::MultiNewtonAdaptive<plugin::NewtonLineSearch> nonlinearSolver;
        nonlinearSolver.SetOperator( contactOperator );

        // The mixed Jacobian remains a matrix-free 2-by-2 BlockOperator. Use a
        // copied elasticity matrix only to precondition its primal block.
        const auto* primalGradient = dynamic_cast<const mfem::SparseMatrix*>( &residual.GetGradient( unknown.GetBlock( 0 ) ) );
        MFEM_VERIFY( primalGradient != nullptr,
                     "The serial semismooth benchmark requires an assembled sparse elasticity Jacobian." );
        primalPreconditionerMatrix = *primalGradient;
        primalPreconditionerMatrix *= primalResidualScale;
        primalBlockSolver.SetOperator( primalPreconditionerMatrix );
#ifdef MFEM_USE_SUITESPARSE
        primalBlockSolver.SetPrintLevel( 0 );
#endif
        primalBlockSolver.iterative_mode = false;
        multiplierBlockSolver.iterative_mode = false;
        blockPreconditioner.SetDiagonalBlock( 0, &primalBlockSolver );
        blockPreconditioner.SetDiagonalBlock( 1, &multiplierBlockSolver );

        const mfem::real_t precisionTolerance = 100. * std::numeric_limits<mfem::real_t>::epsilon();
        tangentSolver.SetPreconditioner( blockPreconditioner );
        tangentSolver.SetRelTol( std::max( static_cast<mfem::real_t>( 1e-12 ), precisionTolerance ) );
        tangentSolver.SetAbsTol( std::max( static_cast<mfem::real_t>( 1e-14 ), precisionTolerance ) );
        tangentSolver.SetMaxIter( std::max( 500, 4 * contactOperator.Height() ) );
        tangentSolver.SetKDim( std::min( 200, contactOperator.Height() ) );
        tangentSolver.SetPrintLevel( -1 );

        const mfem::real_t residualScale = std::sqrt( static_cast<mfem::real_t>( contactOperator.Height() ) );
        const mfem::real_t nonlinearAbsoluteTolerance =
            std::max( static_cast<mfem::real_t>( 1e-12 ), precisionTolerance * residualScale );
        const mfem::real_t nonlinearRelativeTolerance = std::max( static_cast<mfem::real_t>( 1e-10 ), precisionTolerance );
        nonlinearSolver.iterative_mode = true;
        nonlinearSolver.SetSolver( tangentSolver );
        nonlinearSolver.SetRelTol( nonlinearRelativeTolerance );
        nonlinearSolver.SetAbsTol( nonlinearAbsoluteTolerance );
        nonlinearSolver.SetMaxIter( 10 );
        nonlinearSolver.SetPrintLevel( -1 );
        nonlinearSolver.SetDelta( 1. / loadSteps );
        nonlinearSolver.SetMaxDelta( 1. / loadSteps );
        nonlinearSolver.SetMinDelta( std::max( static_cast<mfem::real_t>( 1e-10 ), precisionTolerance ) );
        nonlinearSolver.SetTrialStateFunc(
            [&]( const mfem::real_t pseudoTime, mfem::Vector& trialUnknown )
            {
                mfem::Vector trialDisplacement( trialUnknown.GetData(), contactOperator.GetBlockOffsets()[1] );
                const mfem::real_t prescribedDisplacement = prescribedMotion( pseudoTime );
                for ( int index = 0; index < bottomVerticalDofs.Size(); index++ )
                {
                    trialDisplacement( bottomVerticalDofs[index] ) = prescribedDisplacement;
                }
            } );

        if ( output )
        {
            bodyParaview = std::make_unique<mfem::ParaViewDataCollection>( "semismooth_contact", &mesh );
            bodyParaview->SetPrefixPath( outputDirectory );
            bodyParaview->SetLevelsOfDetail( order );
            bodyParaview->SetDataFormat( mfem::VTKFormat::BINARY );
            bodyParaview->SetHighOrderOutput( true );
            bodyParaview->RegisterVCoeffField( "displacement", &paraviewDisplacement );
            if ( useJ2 )
            {
                bodyParaview->RegisterField( "equivalent_plastic_strain", &equivalentPlasticStrain );
            }
            bodyParaview->SetCycle( 0 );
            bodyParaview->SetTime( 0. );
            SaveParaView( *bodyParaview );

            auto* contactSubMesh = &boundaryMultiplierSpace.GetMesh();
            multiplierParaview = std::make_unique<mfem::ParaViewDataCollection>( "semismooth_contact_multiplier", contactSubMesh );
            multiplierParaview->SetPrefixPath( outputDirectory );
            multiplierParaview->SetDataFormat( mfem::VTKFormat::BINARY );
            multiplierParaview->RegisterField( "boundary_multiplier", &boundaryMultiplier );
            multiplierParaview->SetCycle( 0 );
            multiplierParaview->SetTime( 0. );
            SaveParaView( *multiplierParaview );
        }

        nonlinearSolver.SetDataCollectionFunc(
            [&]( const int, const int, const mfem::real_t pseudoTime )
            {
                displacement.SetFromTrueDofs( unknown.GetBlock( 0 ) );
                contactOperator.GetMultiplier( unknown, trueBoundaryMultiplier );
                boundaryMultiplier.SetFromTrueDofs( trueBoundaryMultiplier );
                updatePlasticOutput();
                if ( !output )
                {
                    return;
                }
                outputCycle++;
                bodyParaview->SetCycle( outputCycle );
                bodyParaview->SetTime( pseudoTime );
                SaveParaView( *bodyParaview );
                multiplierParaview->SetCycle( outputCycle );
                multiplierParaview->SetTime( pseudoTime );
                SaveParaView( *multiplierParaview );
            } );

        mfem::Vector zeroRightHandSide;
        nonlinearSolver.Mult( zeroRightHandSide, unknown );
        MFEM_VERIFY( nonlinearSolver.GetConverged(), "The semismooth contact benchmark did not reach full load." );
        const auto peakDiagnostics = contactOperator.ComputeContactDiagnostics( unknown );
        std::cout << std::scientific << std::setprecision( 8 )
                  << "Peak displacement infinity norm: " << unknown.GetBlock( 0 ).Normlinf() << '\n'
                  << "Peak contact pressure: " << peakDiagnostics.MaximumPressure << '\n';
        // Separate solves stop exactly at reversal; the same operator/storage owns
        // committed history throughout. Adaptive failed trials are rolled back by
        // the solver lifecycle, including through the mixed primal operator.
        if ( returnToZero )
        {
            nonlinearSolver.SetPseudoTimeInterval( 1., 2. );
            nonlinearSolver.SetDelta( 1. / loadSteps );
            nonlinearSolver.Mult( zeroRightHandSide, unknown );
            MFEM_VERIFY( nonlinearSolver.GetConverged(), "Semismooth return stage failed." );
        }
        reportResidualShape( unknown.GetBlock( 0 ) );
        displacement.SetFromTrueDofs( unknown.GetBlock( 0 ) );
        contactOperator.GetMultiplier( unknown, trueBoundaryMultiplier );
        boundaryMultiplier.SetFromTrueDofs( trueBoundaryMultiplier );

        mfem::Vector finalResidual( contactOperator.Height() );
        contactOperator.Mult( unknown, finalResidual );
        const plugin::SemismoothContactDiagnostics diagnostics = contactOperator.ComputeContactDiagnostics( unknown );
        const mfem::real_t sampledMinimumGap =
            contact_example::SampleMinimumGap( mesh, displacement, obstacle, topBoundary, gapSampleIntervals );
        const mfem::real_t epsilon = std::numeric_limits<mfem::real_t>::epsilon();
        const mfem::real_t relativeVerificationTolerance = std::max( static_cast<mfem::real_t>( 1e-8 ), 1000. * epsilon );
        mfem::real_t maximumBottomDisplacementError = 0.;
        for ( int index = 0; index < bottomVerticalDofs.Size(); index++ )
        {
            maximumBottomDisplacementError =
                std::max( maximumBottomDisplacementError,
                          std::abs( unknown.GetBlock( 0 )( bottomVerticalDofs[index] ) - finalBoundaryDisplacement ) );
        }
        const bool reachedFullLoad = std::abs( nonlinearSolver.GetCurrentPseudoTime() - finalTime ) <= relativeVerificationTolerance;
        const mfem::real_t residualVerificationTolerance =
            10. * std::max( nonlinearAbsoluteTolerance, nonlinearRelativeTolerance * residualScale );
        const bool residualConverged = finalResidual.Norml2() <= residualVerificationTolerance;
        const bool bottomDisplacementSatisfied =
            maximumBottomDisplacementError <= relativeVerificationTolerance * ( 1. + maximumBottomDisplacement );
        const bool initiallySeparated = initialDiagnostics.MinimumGap > 0. && initialDiagnostics.MaximumPressure == 0. &&
                                        initialDiagnostics.MinimumMultiplier == 0. && initialDiagnostics.MaximumMultiplier == 0.;
        const bool contactActive = peakDiagnostics.ActiveQuadraturePointCount > 0 &&
                                   peakDiagnostics.MaximumPressure > 0. && peakDiagnostics.MaximumMultiplier > 0.;
        const mfem::real_t multiplierTolerance =
            relativeVerificationTolerance * std::max( mfem::real_t{ 1. }, diagnostics.MaximumMultiplier );
        const bool multiplierAdmissible = std::isfinite( diagnostics.MinimumMultiplier ) &&
                                          std::isfinite( diagnostics.MaximumMultiplier ) &&
                                          diagnostics.MinimumMultiplier >= -multiplierTolerance;
        const mfem::real_t penetrationTolerance = maximumBottomDisplacement + relativeVerificationTolerance * height;
        const bool penetrationControlled = std::isfinite( diagnostics.MaximumPenetration ) &&
                                           diagnostics.MaximumPenetration <= penetrationTolerance &&
                                           -sampledMinimumGap <= penetrationTolerance;
        const bool complementaritySatisfied = std::isfinite( diagnostics.MaximumComplementarityResidual ) &&
                                              diagnostics.MaximumComplementarityResidual <= penetrationTolerance;
        const bool multiplierCountCorrect = trueBoundaryMultiplier.Size() == expectedContactFaceCount;
        const bool contactOpposesMotion = peakDiagnostics.Resultant( 1 ) < 0.;
        const bool verificationPassed = reachedFullLoad && residualConverged && bottomDisplacementSatisfied &&
                                        initiallySeparated && contactActive && multiplierAdmissible && penetrationControlled &&
                                        complementaritySatisfied && multiplierCountCorrect && contactOpposesMotion;

        std::cout << std::scientific << std::setprecision( 8 );
        std::cout << "Semismooth contact augmentation kappa: " << penalty << '\n';
        std::cout << "Semismooth contact compliance gamma: " << gamma << '\n';
        std::cout << "Multiplier jump weight delta: " << delta << '\n';
        std::cout << "Primal residual reference: " << primalResidualReference << '\n';
        std::cout << "Multiplier residual reference: " << multiplierResidualReference << '\n';
        std::cout << "Circle center: [" << circleCenterX << ", " << circleCenterY << "], radius: " << circleRadius << '\n';
        std::cout << "Boundary P0 multiplier dofs: " << trueBoundaryMultiplier.Size() << '\n';
        std::cout << "Expected contact faces: " << expectedContactFaceCount << '\n';
        std::cout << "Initial minimum gap: " << initialDiagnostics.MinimumGap << '\n';
        std::cout << "Final minimum gap: " << diagnostics.MinimumGap << '\n';
        std::cout << "Maximum penetration: " << diagnostics.MaximumPenetration << '\n';
        std::cout << "Independent sampled minimum gap: " << sampledMinimumGap << '\n';
        std::cout << "Independent sampled maximum penetration: " << std::max( mfem::real_t{ 0. }, -sampledMinimumGap ) << '\n';
        std::cout << "Multiplier range: [" << diagnostics.MinimumMultiplier << ", " << diagnostics.MaximumMultiplier << "]\n";
        std::cout << "Maximum contact pressure: " << diagnostics.MaximumPressure << '\n';
        std::cout << "Maximum pointwise contact-map residual: " << diagnostics.MaximumComplementarityResidual << '\n';
        std::cout << "Penetration verification tolerance: " << penetrationTolerance << '\n';
        std::cout << "Active contact quadrature points: " << diagnostics.ActiveQuadraturePointCount << "/"
                  << diagnostics.QuadraturePointCount << '\n';
        std::cout << "Contact resultant: [" << diagnostics.Resultant( 0 ) << ", " << diagnostics.Resultant( 1 ) << "]\n";
        std::cout << "Maximum bottom displacement error: " << maximumBottomDisplacementError << '\n';
        std::cout << "Final mixed residual norm: " << finalResidual.Norml2() << '\n';
        if ( !contactActive && sampledMinimumGap < 0. )
        {
            std::cerr << "Independent gap samples detect penetration without active contact. "
                         "Increase --contact-integration-order and/or refine the mesh; this solve is not verified.\n";
        }

        if ( output )
        {
            // Contact uses the exact analytic circle; this closed polyline is only
            // its independent visualization geometry.
            constexpr int obstacleSegments = 128;
            mfem::Mesh obstacleMesh( 1, obstacleSegments, obstacleSegments, 0, 2 );
            const mfem::real_t twoPi = 2. * std::acos( -1. );
            for ( int vertex = 0; vertex < obstacleSegments; vertex++ )
            {
                const mfem::real_t angle = twoPi * vertex / obstacleSegments;
                obstacleMesh.AddVertex( circleCenterX + circleRadius * std::cos( angle ),
                                        circleCenterY + circleRadius * std::sin( angle ) );
            }
            for ( int segment = 0; segment < obstacleSegments; segment++ )
            {
                obstacleMesh.AddSegment( segment, ( segment + 1 ) % obstacleSegments );
            }
            obstacleMesh.FinalizeTopology( false );
            obstacleMesh.Finalize();
            mfem::ParaViewDataCollection obstacleParaview( "semismooth_contact_obstacle", &obstacleMesh );
            obstacleParaview.SetPrefixPath( outputDirectory );
            obstacleParaview.SetDataFormat( mfem::VTKFormat::BINARY );
            obstacleParaview.SetCycle( 0 );
            obstacleParaview.SetTime( 0. );
            SaveParaView( obstacleParaview );

            std::cout << "ParaView body: " << outputDirectory << "/semismooth_contact/semismooth_contact.pvd\n";
            std::cout << "ParaView multiplier: " << outputDirectory
                      << "/semismooth_contact_multiplier/semismooth_contact_multiplier.pvd\n";
            std::cout << "ParaView obstacle: " << outputDirectory << "/semismooth_contact_obstacle/semismooth_contact_obstacle.pvd\n";
        }

        std::cout << "Semismooth contact verification: " << ( verificationPassed ? "passed" : "failed" ) << '\n';
        return verificationPassed ? 0 : 2;
    }

    auto* penaltyIntegrator = new plugin::FrictionlessPenaltyContactIntegrator( obstacle, penalty );
    penaltyIntegrator->SetIntRule( &contactRule );
    residual.AddBoundaryIntegrator( penaltyIntegrator, contactBoundary );
    residual.SetEssentialTrueDofs( essentialTrueDofs );

    const mfem::real_t precisionTolerance = 100. * std::numeric_limits<mfem::real_t>::epsilon();
#ifdef MFEM_USE_SUITESPARSE
    mfem::UMFPackSolver tangentSolver;
    tangentSolver.SetPrintLevel( 0 );
#else
    mfem::DSmoother diagonalPreconditioner;
    mfem::GMRESSolver tangentSolver;
    tangentSolver.SetPreconditioner( diagonalPreconditioner );
    tangentSolver.SetRelTol( std::max( static_cast<mfem::real_t>( 1e-12 ), precisionTolerance ) );
    tangentSolver.SetAbsTol( std::max( static_cast<mfem::real_t>( 1e-14 ), precisionTolerance ) );
    tangentSolver.SetMaxIter( 500 );
    tangentSolver.SetKDim( 100 );
    tangentSolver.SetPrintLevel( -1 );
#endif

    plugin::MultiNewtonAdaptive<plugin::NewtonLineSearch> nonlinearSolver;
    const mfem::real_t residualScale = std::max( mfem::real_t{ 1. }, youngsModulus * maximumBottomDisplacement * width / height );
    const mfem::real_t nonlinearAbsoluteTolerance =
        std::max( static_cast<mfem::real_t>( 1e-12 ), precisionTolerance * residualScale );
    const mfem::real_t nonlinearRelativeTolerance = std::max( static_cast<mfem::real_t>( 1e-10 ), precisionTolerance );
    nonlinearSolver.iterative_mode = true;
    nonlinearSolver.SetSolver( tangentSolver );
    nonlinearSolver.SetOperator( residual );
    nonlinearSolver.SetRelTol( nonlinearRelativeTolerance );
    nonlinearSolver.SetAbsTol( nonlinearAbsoluteTolerance );
    nonlinearSolver.SetMaxIter( 12 );
    nonlinearSolver.SetPrintLevel( -1 );
    nonlinearSolver.SetDelta( 1. / loadSteps );
    nonlinearSolver.SetMaxDelta( 1. / loadSteps );
    nonlinearSolver.SetMinDelta( std::max( static_cast<mfem::real_t>( 1e-10 ), precisionTolerance ) );
    nonlinearSolver.SetTrialStateFunc(
        [&]( const mfem::real_t pseudoTime, mfem::Vector& trialDisplacement )
        {
            const mfem::real_t prescribedDisplacement = prescribedMotion( pseudoTime );
            for ( int index = 0; index < bottomVerticalDofs.Size(); index++ )
            {
                trialDisplacement( bottomVerticalDofs[index] ) = prescribedDisplacement;
            }
        } );

    mfem::GridFunction displacement( &displacementSpace );
    displacement = 0.;
    plugin::ParaView2DVectorCoefficient paraviewDisplacement( displacement );
    const ContactDiagnostics initialDiagnostics =
        MeasureContact( mesh, displacementSpace, displacement, obstacle, penalty, topBoundary, contactRule );
    mfem::Vector trueDisplacement;
    displacement.GetTrueDofs( trueDisplacement );
    std::unique_ptr<mfem::ParaViewDataCollection> paraview;
    int outputCycle = 0;
    if ( output )
    {
        paraview = std::make_unique<mfem::ParaViewDataCollection>( "penalty_contact", &mesh );
        paraview->SetPrefixPath( outputDirectory );
        paraview->SetLevelsOfDetail( order );
        paraview->SetDataFormat( mfem::VTKFormat::BINARY );
        paraview->SetHighOrderOutput( true );
        paraview->RegisterVCoeffField( "displacement", &paraviewDisplacement );
        if ( useJ2 )
        {
            paraview->RegisterField( "equivalent_plastic_strain", &equivalentPlasticStrain );
        }
        paraview->SetCycle( 0 );
        paraview->SetTime( 0. );
        SaveParaView( *paraview );
    }
    nonlinearSolver.SetDataCollectionFunc(
        [&]( int, int, const mfem::real_t time )
        {
            displacement.SetFromTrueDofs( trueDisplacement );
            updatePlasticOutput();
            if ( paraview )
            {
                paraview->SetCycle( ++outputCycle );
                paraview->SetTime( time );
                SaveParaView( *paraview );
            }
        } );
    mfem::Vector zeroRightHandSide;
    nonlinearSolver.Mult( zeroRightHandSide, trueDisplacement );
    MFEM_VERIFY( nonlinearSolver.GetConverged(), "The penalty contact benchmark did not reach full load." );
    displacement.SetFromTrueDofs( trueDisplacement );
    const auto peakDiagnostics =
        MeasureContact( mesh, displacementSpace, displacement, obstacle, penalty, topBoundary, contactRule );
    std::cout << std::scientific << std::setprecision( 8 )
              << "Peak displacement infinity norm: " << trueDisplacement.Normlinf() << '\n'
              << "Peak contact pressure: " << peakDiagnostics.MaximumPressure << '\n';
    if ( returnToZero )
    {
        nonlinearSolver.SetPseudoTimeInterval( 1., 2. );
        nonlinearSolver.SetDelta( 1. / loadSteps );
        nonlinearSolver.Mult( zeroRightHandSide, trueDisplacement );
        MFEM_VERIFY( nonlinearSolver.GetConverged(), "Penalty return stage failed." );
        displacement.SetFromTrueDofs( trueDisplacement );
    }
    reportResidualShape( trueDisplacement );

    mfem::Vector finalResidual( residual.Height() );
    residual.Mult( trueDisplacement, finalResidual );
    const ContactDiagnostics diagnostics =
        MeasureContact( mesh, displacementSpace, displacement, obstacle, penalty, topBoundary, contactRule );
    const mfem::real_t sampledMinimumGap =
        contact_example::SampleMinimumGap( mesh, displacement, obstacle, topBoundary, gapSampleIntervals );
    const mfem::real_t epsilon = std::numeric_limits<mfem::real_t>::epsilon();
    const mfem::real_t relativeVerificationTolerance = std::max( static_cast<mfem::real_t>( 1e-8 ), 1000. * epsilon );
    mfem::real_t maximumBottomDisplacementError = 0.;
    for ( int index = 0; index < bottomVerticalDofs.Size(); index++ )
    {
        maximumBottomDisplacementError =
            std::max( maximumBottomDisplacementError,
                      std::abs( trueDisplacement( bottomVerticalDofs[index] ) - finalBoundaryDisplacement ) );
    }
    const mfem::real_t pressureCentroidX = peakDiagnostics.PressureIntegral > 0.
                                               ? peakDiagnostics.PressureMomentX / peakDiagnostics.PressureIntegral
                                               : std::numeric_limits<mfem::real_t>::quiet_NaN();
    const bool reachedFullLoad = std::abs( nonlinearSolver.GetCurrentPseudoTime() - finalTime ) <= relativeVerificationTolerance;
    const mfem::real_t residualVerificationTolerance =
        10. * std::max( nonlinearAbsoluteTolerance, nonlinearRelativeTolerance * residualScale );
    const bool residualConverged = finalResidual.Norml2() <= residualVerificationTolerance;
    const bool bottomDisplacementSatisfied =
        maximumBottomDisplacementError <= relativeVerificationTolerance * ( 1. + maximumBottomDisplacement );
    const bool initiallySeparated = initialDiagnostics.MinimumGap > 0. && initialDiagnostics.MaximumPressure == 0.;
    const bool contactActive = peakDiagnostics.ActiveQuadraturePointCount > 0 && peakDiagnostics.MaximumPressure > 0. &&
                               peakDiagnostics.MaximumPenetration > 0.;
    const bool contactOpposesMotion = peakDiagnostics.Resultant( 1 ) < 0.;
    const bool contactLocalizedNearCircle =
        std::isfinite( pressureCentroidX ) &&
        pressureCentroidX >= circleCenterX - circleRadius - relativeVerificationTolerance * width &&
        pressureCentroidX <= circleCenterX + circleRadius + relativeVerificationTolerance * width;
    const bool verificationPassed = reachedFullLoad && residualConverged && bottomDisplacementSatisfied &&
                                    initiallySeparated && contactActive && contactOpposesMotion && contactLocalizedNearCircle;

    std::cout << std::scientific << std::setprecision( 8 );
    std::cout << "Contact penalty kappa: " << penalty << '\n';
    std::cout << "Circle center: [" << circleCenterX << ", " << circleCenterY << "], radius: " << circleRadius << '\n';
    std::cout << "Initial minimum gap: " << initialDiagnostics.MinimumGap << '\n';
    std::cout << "Final minimum gap: " << diagnostics.MinimumGap << '\n';
    std::cout << "Maximum penetration: " << diagnostics.MaximumPenetration << '\n';
    std::cout << "Independent sampled minimum gap: " << sampledMinimumGap << '\n';
    std::cout << "Independent sampled maximum penetration: " << std::max( mfem::real_t{ 0. }, -sampledMinimumGap ) << '\n';
    std::cout << "Pressure range: [" << diagnostics.MinimumPressure << ", " << diagnostics.MaximumPressure << "]\n";
    std::cout << "Active contact quadrature points: " << diagnostics.ActiveQuadraturePointCount << "/"
              << diagnostics.QuadraturePointCount << '\n';
    std::cout << "Peak pressure centroid x: " << pressureCentroidX << '\n';
    std::cout << "Contact resultant: [" << diagnostics.Resultant( 0 ) << ", " << diagnostics.Resultant( 1 ) << "]\n";
    std::cout << "Maximum bottom displacement error: " << maximumBottomDisplacementError << '\n';
    std::cout << "Final residual norm: " << finalResidual.Norml2() << '\n';

    if ( output )
    {
        // Contact uses the exact analytic circle; this closed polyline is only
        // its independent visualization geometry.
        constexpr int obstacleSegments = 128;
        mfem::Mesh obstacleMesh( 1, obstacleSegments, obstacleSegments, 0, 2 );
        const mfem::real_t twoPi = 2. * std::acos( -1. );
        for ( int vertex = 0; vertex < obstacleSegments; vertex++ )
        {
            const mfem::real_t angle = twoPi * vertex / obstacleSegments;
            obstacleMesh.AddVertex( circleCenterX + circleRadius * std::cos( angle ),
                                    circleCenterY + circleRadius * std::sin( angle ) );
        }
        for ( int segment = 0; segment < obstacleSegments; segment++ )
        {
            obstacleMesh.AddSegment( segment, ( segment + 1 ) % obstacleSegments );
        }
        obstacleMesh.FinalizeTopology( false );
        obstacleMesh.Finalize();
        mfem::ParaViewDataCollection obstacleParaview( "penalty_contact_obstacle", &obstacleMesh );
        obstacleParaview.SetPrefixPath( outputDirectory );
        obstacleParaview.SetDataFormat( mfem::VTKFormat::BINARY );
        obstacleParaview.SetCycle( 0 );
        obstacleParaview.SetTime( 0. );
        SaveParaView( obstacleParaview );

        std::cout << "ParaView body: " << outputDirectory << "/penalty_contact/penalty_contact.pvd\n";
        std::cout << "ParaView obstacle: " << outputDirectory << "/penalty_contact_obstacle/penalty_contact_obstacle.pvd\n";
    }

    std::cout << "Penalty contact verification: " << ( verificationPassed ? "passed" : "failed" ) << '\n';
    return verificationPassed ? 0 : 2;
}
} // namespace

int main( int argc, char* argv[] )
{
    try
    {
        return RunExample( argc, argv );
    }
    catch ( const std::exception& error )
    {
        std::cerr << error.what() << '\n';
        return 3;
    }
}
