#include "Contact.h"
#include "FEMPlugin.h"
#include "Material.h"
#include "PostProc.h"
#include "Solvers.h"

#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mfem.hpp>

namespace
{
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
                                   const int contactBoundaryAttribute )
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

        int integrationOrder = transformation->Elem1->OrderW() + 2 * element->GetOrder();
        if ( element->Space() == mfem::FunctionSpace::Pk )
        {
            integrationOrder++;
        }
        const auto& integrationRule = mfem::IntRules.Get( transformation->GetGeometryType(), integrationOrder );
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
    mfem::real_t width = 1.;
    mfem::real_t height = 1.;
    mfem::real_t youngsModulus = 1000.;
    mfem::real_t poissonRatio = .3;
    mfem::real_t maximumBottomDisplacement = .02;
    mfem::real_t circleCenterX = .65;
    mfem::real_t circleRadius = .25;
    mfem::real_t initialClearance = .01;
    mfem::real_t penaltyFactor = 10.;
    bool output = true;
    const char* outputDirectory = "ParaView";

    mfem::OptionsParser options( argc, argv );
    options.AddOption( &elementsX, "-nx", "--elements-x", "Number of elements along the contact boundary." );
    options.AddOption( &elementsY, "-ny", "--elements-y", "Number of elements through the block height." );
    options.AddOption( &refinementLevels, "-r", "--refine-level", "Uniform mesh refinement levels." );
    options.AddOption( &order, "-o", "--order", "H1 displacement order." );
    options.AddOption( &loadSteps, "-steps", "--load-steps", "Number of load increments." );
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
    options.AddOption( &penaltyFactor, "-gamma", "--penalty-factor", "Dimensionless factor in kappa = gamma E / h." );
    options.AddOption( &output, "-output", "--output", "-no-output", "--no-output", "Write ParaView output." );
    options.AddOption( &outputDirectory, "-odir", "--output-directory", "ParaView output prefix." );
    options.Parse();
    if ( !options.Good() )
    {
        options.PrintUsage( std::cout );
        return 1;
    }
    options.PrintOptions( std::cout );

    MFEM_VERIFY( elementsX > 0 && elementsY > 0 && order > 0 && loadSteps > 0,
                 "Element counts, displacement order, and load steps must be positive." );
    MFEM_VERIFY( refinementLevels >= 0, "Uniform mesh refinement levels must be nonnegative." );
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
    mfem::ConstantCoefficient youngsModulusCoefficient( youngsModulus );
    mfem::ConstantCoefficient poissonRatioCoefficient( poissonRatio );
    IsotropicElasticMaterial material( youngsModulusCoefficient, poissonRatioCoefficient );
    plugin::IntegrationPointStorage<> pointStorage( &mesh );
    const mfem::real_t circleCenterY = height + initialClearance + circleRadius;
    const mfem::Vector circleCenter( MakeVector( { circleCenterX, circleCenterY } ) );
    plugin::RigidSphereObstacle obstacle( circleCenter, circleRadius );

    mfem::NonlinearForm residual( &displacementSpace );
    auto* elasticityIntegrator = new plugin::NonlinearElasticityIntegrator( material, pointStorage );
    elasticityIntegrator->setNonlinear( false );
    residual.AddDomainIntegrator( elasticityIntegrator );
    residual.AddBdrFaceIntegrator( new plugin::FrictionlessPenaltyContactIntegrator( obstacle, penalty ), contactBoundary );
    residual.SetEssentialTrueDofs( essentialTrueDofs );

    const mfem::real_t precisionTolerance = 100. * std::numeric_limits<mfem::real_t>::epsilon();
#ifdef MFEM_USE_SUITESPARSE
    mfem::UMFPackSolver tangentSolver;
    tangentSolver.SetPrintLevel( 0 );
#else
    mfem::DSmoother diagonalPreconditioner;
    mfem::CGSolver tangentSolver;
    tangentSolver.SetPreconditioner( diagonalPreconditioner );
    tangentSolver.SetRelTol( std::max( static_cast<mfem::real_t>( 1e-12 ), precisionTolerance ) );
    tangentSolver.SetAbsTol( std::max( static_cast<mfem::real_t>( 1e-14 ), precisionTolerance ) );
    tangentSolver.SetMaxIter( 500 );
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
    nonlinearSolver.SetMaxStep( 10 * loadSteps );
    nonlinearSolver.SetTrialStateFunc(
        [&]( const mfem::real_t pseudoTime, mfem::Vector& trialDisplacement )
        {
            const mfem::real_t prescribedDisplacement = pseudoTime * maximumBottomDisplacement;
            for ( int index = 0; index < bottomVerticalDofs.Size(); index++ )
            {
                trialDisplacement( bottomVerticalDofs[index] ) = prescribedDisplacement;
            }
        } );

    mfem::GridFunction displacement( &displacementSpace );
    displacement = 0.;
    plugin::ParaView2DVectorCoefficient paraviewDisplacement( displacement );
    const ContactDiagnostics initialDiagnostics =
        MeasureContact( mesh, displacementSpace, displacement, obstacle, penalty, topBoundary );
    mfem::Vector trueDisplacement;
    displacement.GetTrueDofs( trueDisplacement );
    mfem::Vector zeroRightHandSide;
    nonlinearSolver.Mult( zeroRightHandSide, trueDisplacement );
    MFEM_VERIFY( nonlinearSolver.GetConverged(), "The penalty contact benchmark did not reach full load." );
    displacement.SetFromTrueDofs( trueDisplacement );

    mfem::Vector finalResidual( residual.Height() );
    residual.Mult( trueDisplacement, finalResidual );
    const ContactDiagnostics diagnostics = MeasureContact( mesh, displacementSpace, displacement, obstacle, penalty, topBoundary );
    const mfem::real_t epsilon = std::numeric_limits<mfem::real_t>::epsilon();
    const mfem::real_t relativeVerificationTolerance = std::max( static_cast<mfem::real_t>( 1e-8 ), 1000. * epsilon );
    mfem::real_t maximumBottomDisplacementError = 0.;
    for ( int index = 0; index < bottomVerticalDofs.Size(); index++ )
    {
        maximumBottomDisplacementError =
            std::max( maximumBottomDisplacementError,
                      std::abs( trueDisplacement( bottomVerticalDofs[index] ) - maximumBottomDisplacement ) );
    }
    const mfem::real_t pressureCentroidX = diagnostics.PressureIntegral > 0.
                                               ? diagnostics.PressureMomentX / diagnostics.PressureIntegral
                                               : std::numeric_limits<mfem::real_t>::quiet_NaN();
    const bool reachedFullLoad = std::abs( nonlinearSolver.GetCurrentPseudoTime() - 1. ) <= relativeVerificationTolerance;
    const mfem::real_t residualVerificationTolerance =
        10. * std::max( nonlinearAbsoluteTolerance, nonlinearRelativeTolerance * residualScale );
    const bool residualConverged = finalResidual.Norml2() <= residualVerificationTolerance;
    const bool bottomDisplacementSatisfied =
        maximumBottomDisplacementError <= relativeVerificationTolerance * ( 1. + maximumBottomDisplacement );
    const bool initiallySeparated = initialDiagnostics.MinimumGap > 0. && initialDiagnostics.MaximumPressure == 0.;
    const bool contactActive = diagnostics.ActiveQuadraturePointCount > 0 && diagnostics.MaximumPressure > 0. &&
                               diagnostics.MaximumPenetration > 0.;
    const bool contactOpposesMotion = diagnostics.Resultant( 1 ) < 0.;
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
    std::cout << "Pressure range: [" << diagnostics.MinimumPressure << ", " << diagnostics.MaximumPressure << "]\n";
    std::cout << "Active contact quadrature points: " << diagnostics.ActiveQuadraturePointCount << "/"
              << diagnostics.QuadraturePointCount << '\n';
    std::cout << "Pressure centroid x: " << pressureCentroidX << '\n';
    std::cout << "Contact resultant: [" << diagnostics.Resultant( 0 ) << ", " << diagnostics.Resultant( 1 ) << "]\n";
    std::cout << "Maximum bottom displacement error: " << maximumBottomDisplacementError << '\n';
    std::cout << "Final residual norm: " << finalResidual.Norml2() << '\n';

    if ( output )
    {
        mfem::ParaViewDataCollection paraview( "penalty_contact", &mesh );
        paraview.SetPrefixPath( outputDirectory );
        paraview.SetLevelsOfDetail( order );
        paraview.SetDataFormat( mfem::VTKFormat::BINARY );
        paraview.SetHighOrderOutput( true );
        paraview.RegisterVCoeffField( "displacement", &paraviewDisplacement );
        paraview.SetCycle( 1 );
        paraview.SetTime( 1. );
        paraview.Save();

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
        obstacleParaview.SetCycle( 1 );
        obstacleParaview.SetTime( 1. );
        obstacleParaview.Save();

        std::cout << "ParaView body: " << outputDirectory << "/penalty_contact/penalty_contact.pvd\n";
        std::cout << "ParaView obstacle: " << outputDirectory << "/penalty_contact_obstacle/penalty_contact_obstacle.pvd\n";
    }

    std::cout << "Penalty contact verification: " << ( verificationPassed ? "passed" : "failed" ) << '\n';
    return verificationPassed ? 0 : 2;
}
} // namespace

int main( int argc, char* argv[] )
{
    return RunExample( argc, argv );
}
