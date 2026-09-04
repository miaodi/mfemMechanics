#include "J2Plasticity.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mfem.hpp>
#include <type_traits>

namespace
{
constexpr bool kSinglePrecision = std::is_same_v<mfem::real_t, float>;

struct MaterialPoint
{
    plugin::J2PlasticityParameters Parameters;
    plugin::J2PlasticityState State;
    mfem::real_t TransverseStrain{ 0. };
};

struct UniaxialResponse
{
    plugin::J2PlasticityResponse MaterialResponse;
    mfem::real_t TransverseStrain{ 0. };
};

struct ReverseYield
{
    bool HasElasticBracket{ false };
    bool Found{ false };
    mfem::real_t Strain{ 0. };
    mfem::real_t Stress{ 0. };
    mfem::real_t LastElasticStrain{ 0. };
    MaterialPoint LastElasticPoint;
};

const char* BranchName( const plugin::J2PlasticityBranch branch )
{
    return branch == plugin::J2PlasticityBranch::Elastic ? "elastic" : "plastic";
}

UniaxialResponse EvaluateUniaxialStress( const mfem::real_t axialStrain, const MaterialPoint& point )
{
    constexpr int maximumIterations = 25;
    const mfem::real_t relativeTolerance = kSinglePrecision ? 5e-5f : 1e-12;
    const mfem::real_t stressTolerance =
        relativeTolerance * std::max( mfem::real_t{ 1. }, point.Parameters.Hardening.InitialYieldStress );
    mfem::real_t transverseStrain = point.TransverseStrain;
    plugin::J2PlasticityResponse response;

    for ( int iteration = 0; iteration < maximumIterations; iteration++ )
    {
        Eigen::Matrix3r strain = Eigen::Matrix3r::Zero();
        strain.diagonal() << axialStrain, transverseStrain, transverseStrain;
        response = plugin::EvaluateJ2Plasticity( strain, point.State, point.Parameters );
        const mfem::real_t transverseStress = .5 * ( response.Stress( 1, 1 ) + response.Stress( 2, 2 ) );
        if ( std::abs( transverseStress ) <= stressTolerance )
        {
            MFEM_VERIFY( std::abs( response.Stress( 1, 1 ) ) <= stressTolerance &&
                             std::abs( response.Stress( 2, 2 ) ) <= stressTolerance,
                         "The uniaxial material-point solve did not eliminate both transverse stresses." );
            return { response, transverseStrain };
        }

        const mfem::real_t transverseTangent =
            .5 * ( response.ConsistentTangent( 1, 1 ) + response.ConsistentTangent( 1, 2 ) +
                   response.ConsistentTangent( 2, 1 ) + response.ConsistentTangent( 2, 2 ) );
        MFEM_VERIFY( std::isfinite( transverseTangent ) && transverseTangent > 0.,
                     "The uniaxial material-point solve requires a finite positive transverse tangent." );
        transverseStrain -= transverseStress / transverseTangent;
        MFEM_VERIFY( std::isfinite( transverseStrain ),
                     "The uniaxial material-point solve produced a nonfinite strain." );
    }

    MFEM_ABORT( "The uniaxial material-point solve did not converge." );
}

void ObserveReverseYield( ReverseYield& reverseYield,
                          const mfem::real_t axialStrain,
                          const MaterialPoint& point,
                          const plugin::J2PlasticityResponse& response )
{
    if ( response.Branch == plugin::J2PlasticityBranch::Elastic )
    {
        reverseYield.HasElasticBracket = true;
        reverseYield.LastElasticStrain = axialStrain;
        reverseYield.LastElasticPoint = point;
        return;
    }
    if ( !reverseYield.HasElasticBracket || reverseYield.Found )
    {
        return;
    }

    mfem::real_t elasticStrain = reverseYield.LastElasticStrain;
    mfem::real_t plasticStrain = axialStrain;
    plugin::J2PlasticityResponse elasticResponse =
        EvaluateUniaxialStress( elasticStrain, reverseYield.LastElasticPoint ).MaterialResponse;
    for ( int iteration = 0; iteration < std::numeric_limits<mfem::real_t>::digits; iteration++ )
    {
        const mfem::real_t midpoint = .5 * ( elasticStrain + plasticStrain );
        const auto midpointResponse =
            EvaluateUniaxialStress( midpoint, reverseYield.LastElasticPoint ).MaterialResponse;
        if ( midpointResponse.Branch == plugin::J2PlasticityBranch::Elastic )
        {
            elasticStrain = midpoint;
            elasticResponse = midpointResponse;
        }
        else
        {
            plasticStrain = midpoint;
        }
    }

    reverseYield.Found = true;
    reverseYield.Strain = elasticStrain;
    reverseYield.Stress = elasticResponse.Stress( 0, 0 );
}

plugin::J2PlasticityResponse Advance( MaterialPoint& point, const mfem::real_t axialStrain )
{
    const auto result = EvaluateUniaxialStress( axialStrain, point );
    point.State = result.MaterialResponse.TrialState;
    point.TransverseStrain = result.TransverseStrain;
    return result.MaterialResponse;
}

void WriteSample( std::ofstream& output,
                  const int step,
                  const char* stage,
                  const mfem::real_t axialStrain,
                  const plugin::J2PlasticityResponse& isotropic,
                  const plugin::J2PlasticityResponse& kinematic )
{
    if ( !output.is_open() )
    {
        return;
    }

    output << step << ',' << stage << ',' << axialStrain << ',' << isotropic.Stress( 0, 0 ) << ','
           << kinematic.Stress( 0, 0 ) << ',' << isotropic.TrialState.EquivalentPlasticStrain << ','
           << kinematic.TrialState.EquivalentPlasticStrain << ',' << kinematic.TrialState.BackStress( 0, 0 ) << ','
           << BranchName( isotropic.Branch ) << ',' << BranchName( kinematic.Branch ) << '\n';
}

int RunExample( int argc, char* argv[] )
{
    mfem::real_t youngsModulus = 210000.;
    mfem::real_t poissonRatio = .3;
    mfem::real_t initialYieldStress = 250.;
    mfem::real_t hardeningModulus = 1000.;
    mfem::real_t strainAmplitude = .01;
    int stepsPerHalfCycle = 100;
    bool outputEnabled = true;
    const char* outputFile = "bauschinger.csv";

    mfem::OptionsParser options( argc, argv );
    options.AddOption( &youngsModulus, "-E", "--youngs-modulus", "Young's modulus." );
    options.AddOption( &poissonRatio, "-nu", "--poisson-ratio", "Poisson ratio." );
    options.AddOption( &initialYieldStress, "-yield", "--yield-stress", "Initial uniaxial yield stress." );
    options.AddOption( &hardeningModulus, "-H", "--hardening-modulus",
                       "Matched isotropic and kinematic hardening modulus." );
    options.AddOption( &strainAmplitude, "-strain", "--strain-amplitude",
                       "Positive tensile and compressive strain amplitude." );
    options.AddOption( &stepsPerHalfCycle, "-steps", "--steps-per-half-cycle", "Increments per strain half-cycle." );
    options.AddOption( &outputEnabled, "-output", "--output", "-no-output", "--no-output", "Write CSV output." );
    options.AddOption( &outputFile, "-f", "--output-file", "CSV output path." );
    options.Parse();
    if ( !options.Good() )
    {
        options.PrintUsage( std::cout );
        return 1;
    }
    options.PrintOptions( std::cout );

    MFEM_VERIFY( std::isfinite( youngsModulus ) && youngsModulus > 0.,
                 "The Bauschinger example requires a finite positive Young's modulus." );
    MFEM_VERIFY( std::isfinite( poissonRatio ) && poissonRatio > -1. && poissonRatio < .5,
                 "The Bauschinger example requires a Poisson ratio in (-1, 0.5)." );
    MFEM_VERIFY( std::isfinite( initialYieldStress ) && initialYieldStress > 0.,
                 "The Bauschinger example requires a finite positive yield stress." );
    MFEM_VERIFY( std::isfinite( hardeningModulus ) && hardeningModulus > 0.,
                 "The Bauschinger example requires a finite positive hardening modulus." );
    MFEM_VERIFY( std::isfinite( strainAmplitude ) && strainAmplitude > 0.,
                 "The strain amplitude must be finite and positive." );
    MFEM_VERIFY( stepsPerHalfCycle > 0, "The number of steps per half-cycle must be positive." );

    MaterialPoint isotropic;
    isotropic.Parameters.YoungsModulus = youngsModulus;
    isotropic.Parameters.PoissonRatio = poissonRatio;
    isotropic.Parameters.Hardening.InitialYieldStress = initialYieldStress;
    isotropic.Parameters.Hardening.HardeningModulus = hardeningModulus;

    MaterialPoint kinematic;
    kinematic.Parameters.YoungsModulus = youngsModulus;
    kinematic.Parameters.PoissonRatio = poissonRatio;
    kinematic.Parameters.Hardening.InitialYieldStress = initialYieldStress;
    kinematic.Parameters.KinematicHardening.Modulus = hardeningModulus;

    std::ofstream output;
    if ( outputEnabled )
    {
        output.open( outputFile );
        MFEM_VERIFY( output.is_open(), "Unable to open the Bauschinger CSV output file." );
        output << std::setprecision( std::numeric_limits<mfem::real_t>::max_digits10 );
        output << "step,stage,axial_strain,isotropic_axial_stress,kinematic_axial_stress,"
                  "isotropic_equivalent_plastic_strain,kinematic_equivalent_plastic_strain,"
                  "kinematic_backstress_xx,isotropic_branch,kinematic_branch\n";
    }

    int step = 0;
    auto isotropicResponse = Advance( isotropic, 0. );
    auto kinematicResponse = Advance( kinematic, 0. );
    WriteSample( output, step, "loading", 0., isotropicResponse, kinematicResponse );

    for ( int increment = 1; increment <= stepsPerHalfCycle; increment++ )
    {
        const mfem::real_t axialStrain = strainAmplitude * increment / stepsPerHalfCycle;
        isotropicResponse = Advance( isotropic, axialStrain );
        kinematicResponse = Advance( kinematic, axialStrain );
        WriteSample( output, ++step, "loading", axialStrain, isotropicResponse, kinematicResponse );
    }

    ReverseYield isotropicReverseYield;
    ReverseYield kinematicReverseYield;
    isotropicReverseYield.HasElasticBracket = true;
    isotropicReverseYield.LastElasticStrain = strainAmplitude;
    isotropicReverseYield.LastElasticPoint = isotropic;
    kinematicReverseYield.HasElasticBracket = true;
    kinematicReverseYield.LastElasticStrain = strainAmplitude;
    kinematicReverseYield.LastElasticPoint = kinematic;
    for ( int increment = 1; increment <= 2 * stepsPerHalfCycle; increment++ )
    {
        const mfem::real_t axialStrain =
            strainAmplitude * ( 1. - static_cast<mfem::real_t>( increment ) / stepsPerHalfCycle );
        isotropicResponse = Advance( isotropic, axialStrain );
        kinematicResponse = Advance( kinematic, axialStrain );
        ObserveReverseYield( isotropicReverseYield, axialStrain, isotropic, isotropicResponse );
        ObserveReverseYield( kinematicReverseYield, axialStrain, kinematic, kinematicResponse );
        WriteSample( output, ++step, "reversal", axialStrain, isotropicResponse, kinematicResponse );
    }

    if ( output.is_open() )
    {
        output.close();
        MFEM_VERIFY( output.good(), "Writing the Bauschinger CSV output failed." );
    }

    MFEM_VERIFY( isotropicReverseYield.Found && kinematicReverseYield.Found,
                 "The strain cycle did not reach reverse yielding for both hardening models." );
    const mfem::real_t stressComparisonTolerance =
        ( kSinglePrecision ? 1e-4f : 1e-10 ) * std::max( mfem::real_t{ 1. }, initialYieldStress );
    const mfem::real_t strainComparisonTolerance =
        128. * std::numeric_limits<mfem::real_t>::epsilon() * std::max( mfem::real_t{ 1. }, strainAmplitude );
    MFEM_VERIFY( kinematicReverseYield.Strain > isotropicReverseYield.Strain + strainComparisonTolerance,
                 "The kinematic model did not begin reverse plastic flow earlier than the isotropic model." );
    MFEM_VERIFY( kinematicReverseYield.Stress > isotropicReverseYield.Stress + stressComparisonTolerance,
                 "The kinematic model did not yield earlier in reverse loading than the isotropic model." );

    std::cout << std::scientific << std::setprecision( 6 )
              << "Bauschinger effect: active | isotropic reverse yield stress " << isotropicReverseYield.Stress
              << " at strain " << isotropicReverseYield.Strain << " | kinematic reverse yield stress "
              << kinematicReverseYield.Stress << " at strain " << kinematicReverseYield.Strain << '\n';
    if ( outputEnabled )
    {
        std::cout << "CSV output: " << outputFile << '\n';
    }
    return 0;
}
} // namespace

int main( int argc, char* argv[] )
{
    return RunExample( argc, argv );
}
