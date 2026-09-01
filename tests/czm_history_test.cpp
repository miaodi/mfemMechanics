#include "CZM.h"
#include <Eigen/Dense>
#include <cmath>
#include <functional>
#include <gtest/gtest.h>
#include <limits>
#include <type_traits>

namespace
{
plugin::ExponentialCZMConst MakeLaw()
{
    plugin::ExponentialCZMConst law;
    law.sigma_max = 4.2;
    law.tau_max = 3.1;
    law.delta_n = .5;
    law.delta_t = .7;
    law.update_phi();
    return law;
}

Eigen::MatrixXd FiniteDifferenceTangent( const std::function<Eigen::VectorXd( const Eigen::VectorXd& )>& traction,
                                         const Eigen::VectorXd& separation )
{
    constexpr double epsilon = 1e-7;
    Eigen::MatrixXd tangent( separation.size(), separation.size() );
    for ( int j = 0; j < separation.size(); j++ )
    {
        Eigen::VectorXd plus = separation;
        Eigen::VectorXd minus = separation;
        plus( j ) += epsilon;
        minus( j ) -= epsilon;
        tangent.col( j ) = ( traction( plus ) - traction( minus ) ) / ( 2. * epsilon );
    }
    return tangent;
}

void ExpectMatricesNear( const Eigen::MatrixXd& actual, const Eigen::MatrixXd& expected, const double tolerance )
{
    ASSERT_EQ( actual.rows(), expected.rows() );
    ASSERT_EQ( actual.cols(), expected.cols() );
    EXPECT_LE( ( actual - expected ).norm(), tolerance * ( 1. + expected.norm() ) );
}

void ExpectStatesEqual( const plugin::CZMHistoryState& actual, const plugin::CZMHistoryState& expected )
{
    EXPECT_DOUBLE_EQ( actual.maximum_normal_opening, expected.maximum_normal_opening );
    EXPECT_DOUBLE_EQ( actual.maximum_tangential_opening, expected.maximum_tangential_opening );
    EXPECT_DOUBLE_EQ( actual.normal_unloading_stiffness, expected.normal_unloading_stiffness );
    EXPECT_DOUBLE_EQ( actual.tangential_unloading_stiffness, expected.tangential_unloading_stiffness );
    EXPECT_EQ( actual.has_normal_history, expected.has_normal_history );
    EXPECT_EQ( actual.has_tangential_history, expected.has_tangential_history );
    EXPECT_DOUBLE_EQ( actual.normal_opening, expected.normal_opening );
    EXPECT_DOUBLE_EQ( actual.tangential_opening_1, expected.tangential_opening_1 );
    EXPECT_DOUBLE_EQ( actual.tangential_opening_2, expected.tangential_opening_2 );
}

static_assert( !std::is_copy_constructible_v<plugin::ExponentialCZMIntegrator> );
} // namespace

TEST( Memorize, EmptyAndResetFaceStorageCanBeVisited )
{
    plugin::Memorize memorize( nullptr );
    int visits = 0;
    memorize.VisitFacePointData( [&visits]( util::AnyMap& ) { visits++; } );
    EXPECT_EQ( visits, 0 );

    memorize.Reset( nullptr );
    memorize.VisitFacePointData( [&visits]( util::AnyMap& ) { visits++; } );
    EXPECT_EQ( visits, 0 );
}

TEST( CZMHistory, ZeroAndCompressionAreFinite )
{
    const auto law = MakeLaw();
    plugin::CZMHistory history;

    Eigen::Vector2d separation = Eigen::Vector2d::Zero();
    auto evaluation = plugin::EvaluateIrreversibleExponentialCZM( law, separation, history );
    EXPECT_TRUE( evaluation.traction.allFinite() );
    EXPECT_TRUE( evaluation.tangent.allFinite() );

    separation << .12, -.25;
    evaluation = plugin::EvaluateIrreversibleExponentialCZM( law, separation, history );
    EXPECT_TRUE( evaluation.traction.allFinite() );
    EXPECT_TRUE( evaluation.tangent.allFinite() );
}

TEST( CZMHistory, MonotonicLoadingCommitsTrialMaxima )
{
    const auto law = MakeLaw();
    plugin::CZMHistory history;
    const Eigen::Vector2d separation( -.14, .2 );

    history.BeginStep();
    plugin::EvaluateIrreversibleExponentialCZM( law, separation, history );

    EXPECT_DOUBLE_EQ( history.CommittedState().maximum_normal_opening, 0. );
    EXPECT_DOUBLE_EQ( history.CommittedState().maximum_tangential_opening, 0. );
    EXPECT_DOUBLE_EQ( history.TrialState().maximum_normal_opening, .2 );
    EXPECT_DOUBLE_EQ( history.TrialState().maximum_tangential_opening, .14 );
    EXPECT_TRUE( std::isfinite( history.TrialState().normal_unloading_stiffness ) );
    EXPECT_TRUE( std::isfinite( history.TrialState().tangential_unloading_stiffness ) );

    history.CommitStep();
    EXPECT_DOUBLE_EQ( history.CommittedState().maximum_normal_opening, .2 );
    EXPECT_DOUBLE_EQ( history.CommittedState().maximum_tangential_opening, .14 );
    EXPECT_DOUBLE_EQ( history.CommittedState().normal_opening, .2 );
    EXPECT_DOUBLE_EQ( history.CommittedState().tangential_opening_1, -.14 );
}

TEST( CZMHistory, UnloadingAndReloadingUseCommittedSecantsWithoutHealing )
{
    const auto law = MakeLaw();
    plugin::CZMHistory history;
    const Eigen::Vector2d maximum_separation( .28, .32 );

    plugin::EvaluateIrreversibleExponentialCZM( law, maximum_separation, history );
    history.CommitStep();
    const auto committed = history.CommittedState();

    const Eigen::Vector2d unloading( -.11, .14 );
    auto evaluation = plugin::EvaluateIrreversibleExponentialCZM( law, unloading, history );
    EXPECT_NEAR( evaluation.traction( 0 ), committed.tangential_unloading_stiffness * unloading( 0 ), 1e-13 );
    EXPECT_NEAR( evaluation.traction( 1 ), committed.normal_unloading_stiffness * unloading( 1 ), 1e-13 );
    EXPECT_DOUBLE_EQ( history.TrialState().maximum_normal_opening, committed.maximum_normal_opening );
    EXPECT_DOUBLE_EQ( history.TrialState().maximum_tangential_opening, committed.maximum_tangential_opening );

    history.CommitStep();
    const Eigen::Vector2d reloading( .2, .25 );
    evaluation = plugin::EvaluateIrreversibleExponentialCZM( law, reloading, history );
    EXPECT_NEAR( evaluation.traction( 0 ), committed.tangential_unloading_stiffness * reloading( 0 ), 1e-13 );
    EXPECT_NEAR( evaluation.traction( 1 ), committed.normal_unloading_stiffness * reloading( 1 ), 1e-13 );
    EXPECT_DOUBLE_EQ( history.TrialState().maximum_normal_opening, committed.maximum_normal_opening );
    EXPECT_DOUBLE_EQ( history.TrialState().maximum_tangential_opening, committed.maximum_tangential_opening );
}

TEST( CZMHistory, NewMaximumRejoinsEnvelope )
{
    const auto law = MakeLaw();
    plugin::CZMHistory history;
    plugin::EvaluateIrreversibleExponentialCZM( law, Eigen::Vector2d( .2, .24 ), history );
    history.CommitStep();

    const Eigen::Vector2d separation( -.31, .38 );
    const auto envelope = plugin::EvaluateExponentialCZMEnvelope( law, separation );
    const auto evaluation = plugin::EvaluateIrreversibleExponentialCZM( law, separation, history );

    ExpectMatricesNear( evaluation.traction, envelope.traction, 1e-13 );
    ExpectMatricesNear( evaluation.tangent, envelope.tangent, 1e-13 );
    EXPECT_DOUBLE_EQ( history.TrialState().maximum_normal_opening, .38 );
    EXPECT_DOUBLE_EQ( history.TrialState().maximum_tangential_opening, .31 );
}

TEST( CZMHistory, RollbackRestoresCommittedHistory )
{
    const auto law = MakeLaw();
    plugin::CZMHistory history;
    plugin::EvaluateIrreversibleExponentialCZM( law, Eigen::Vector2d( .18, .23 ), history );
    history.CommitStep();
    const auto committed = history.CommittedState();

    history.BeginStep();
    plugin::EvaluateIrreversibleExponentialCZM( law, Eigen::Vector2d( .4, .5 ), history );
    EXPECT_GT( history.TrialState().maximum_normal_opening, committed.maximum_normal_opening );
    EXPECT_GT( history.TrialState().maximum_tangential_opening, committed.maximum_tangential_opening );

    history.RollbackStep();
    EXPECT_DOUBLE_EQ( history.TrialState().maximum_normal_opening, committed.maximum_normal_opening );
    EXPECT_DOUBLE_EQ( history.TrialState().maximum_tangential_opening, committed.maximum_tangential_opening );
    EXPECT_DOUBLE_EQ( history.TrialState().normal_unloading_stiffness, committed.normal_unloading_stiffness );
    EXPECT_DOUBLE_EQ( history.TrialState().tangential_unloading_stiffness, committed.tangential_unloading_stiffness );
    EXPECT_DOUBLE_EQ( history.TrialState().normal_opening, committed.normal_opening );
    EXPECT_DOUBLE_EQ( history.TrialState().tangential_opening_1, committed.tangential_opening_1 );
    EXPECT_DOUBLE_EQ( history.CommittedState().maximum_normal_opening, committed.maximum_normal_opening );
    EXPECT_DOUBLE_EQ( history.CommittedState().maximum_tangential_opening, committed.maximum_tangential_opening );
}

TEST( CZMHistory, RevertRestoresPreviousCommittedStep )
{
    const auto law = MakeLaw();
    plugin::CZMHistory history;
    plugin::EvaluateIrreversibleExponentialCZM( law, Eigen::Vector2d( .12, .18 ), history );
    history.CommitStep();
    const auto first_commit = history.CommittedState();

    plugin::EvaluateIrreversibleExponentialCZM( law, Eigen::Vector2d( .28, .34 ), history );
    history.CommitStep();
    EXPECT_GT( history.CommittedState().maximum_normal_opening, first_commit.maximum_normal_opening );

    history.RevertStep();
    ExpectStatesEqual( history.CommittedState(), first_commit );
    ExpectStatesEqual( history.TrialState(), first_commit );
}

TEST( CZMHistory, PureModeLoadingDoesNotHealCoupledStiffness )
{
    const auto law = MakeLaw();

    plugin::CZMHistory shear_history;
    plugin::EvaluateIrreversibleExponentialCZM( law, Eigen::Vector2d( .35, 0. ), shear_history );
    shear_history.CommitStep();
    const auto shear_commit = shear_history.CommittedState();
    ASSERT_GT( shear_commit.normal_unloading_stiffness, 0. );

    const Eigen::Vector2d opening_after_shear( 0., .01 );
    const auto normal_response = plugin::EvaluateIrreversibleExponentialCZM( law, opening_after_shear, shear_history );
    EXPECT_NEAR( normal_response.traction( 1 ), shear_commit.normal_unloading_stiffness * opening_after_shear( 1 ), 1e-13 );
    EXPECT_LE( shear_history.TrialState().normal_unloading_stiffness, shear_commit.normal_unloading_stiffness );

    plugin::CZMHistory opening_history;
    plugin::EvaluateIrreversibleExponentialCZM( law, Eigen::Vector2d( 0., 1. ), opening_history );
    opening_history.CommitStep();
    const auto opening_commit = opening_history.CommittedState();
    ASSERT_GT( opening_commit.tangential_unloading_stiffness, 0. );

    const Eigen::Vector2d shear_after_opening( .01, 0. );
    const auto tangential_response = plugin::EvaluateIrreversibleExponentialCZM( law, shear_after_opening, opening_history );
    EXPECT_NEAR( tangential_response.traction( 0 ), opening_commit.tangential_unloading_stiffness * shear_after_opening( 0 ), 1e-13 );
    EXPECT_LE( opening_history.TrialState().tangential_unloading_stiffness, opening_commit.tangential_unloading_stiffness );
}

TEST( CZMHistory, InvalidInputsReturnZeroWithoutChangingHistory )
{
    const auto law = MakeLaw();
    plugin::CZMHistory history;
    plugin::EvaluateIrreversibleExponentialCZM( law, Eigen::Vector2d( .15, .2 ), history );
    history.CommitStep();
    const auto committed = history.CommittedState();

    plugin::EvaluateIrreversibleExponentialCZM( law, Eigen::Vector2d( .3, .4 ), history );
    ASSERT_GT( history.TrialState().maximum_normal_opening, committed.maximum_normal_opening );

    Eigen::Vector2d invalid_separation( std::numeric_limits<double>::quiet_NaN(), .3 );
    auto evaluation = plugin::EvaluateIrreversibleExponentialCZM( law, invalid_separation, history );
    EXPECT_TRUE( evaluation.traction.isZero() );
    EXPECT_TRUE( evaluation.tangent.isZero() );
    ExpectStatesEqual( history.CommittedState(), committed );
    ExpectStatesEqual( history.TrialState(), committed );
    history.CommitStep();
    ExpectStatesEqual( history.CommittedState(), committed );

    auto invalid_law = law;
    invalid_law.delta_n = 0.;
    evaluation = plugin::EvaluateIrreversibleExponentialCZM( invalid_law, Eigen::Vector2d( .2, .3 ), history );
    EXPECT_TRUE( evaluation.traction.isZero() );
    EXPECT_TRUE( evaluation.tangent.isZero() );
    ExpectStatesEqual( history.CommittedState(), committed );
    ExpectStatesEqual( history.TrialState(), committed );
}

TEST( CZMHistory, TangentialSlipInCompressionDoesNotDisableFutureOpening )
{
    const auto law = MakeLaw();
    plugin::CZMHistory history;
    plugin::EvaluateIrreversibleExponentialCZM( law, Eigen::Vector2d( .3, -.2 ), history );
    history.CommitStep();
    EXPECT_FALSE( history.CommittedState().has_normal_history );
    EXPECT_TRUE( history.CommittedState().has_tangential_history );

    const Eigen::Vector2d opening( 0., .05 );
    const auto envelope = plugin::EvaluateExponentialCZMEnvelope( law, opening );
    const auto evaluation = plugin::EvaluateIrreversibleExponentialCZM( law, opening, history );
    EXPECT_NEAR( evaluation.traction( 1 ), envelope.traction( 1 ), 1e-13 );
    EXPECT_GT( evaluation.traction( 1 ), 0. );
    EXPECT_TRUE( history.TrialState().has_normal_history );
    EXPECT_GT( history.TrialState().normal_unloading_stiffness, 0. );
}

TEST( CZMHistory, SmallCharacteristicLengthsStillActivateHistory )
{
    auto law = MakeLaw();
    law.delta_n = 1e-15;
    law.delta_t = 2e-15;
    law.update_phi();

    plugin::CZMHistory history;
    const Eigen::Vector2d separation( 4e-16, 5e-16 );
    const auto evaluation = plugin::EvaluateIrreversibleExponentialCZM( law, separation, history );
    EXPECT_TRUE( evaluation.traction.allFinite() );
    EXPECT_TRUE( evaluation.tangent.allFinite() );
    EXPECT_TRUE( history.TrialState().has_normal_history );
    EXPECT_TRUE( history.TrialState().has_tangential_history );
    EXPECT_DOUBLE_EQ( history.TrialState().maximum_normal_opening, separation( 1 ) );
    EXPECT_DOUBLE_EQ( history.TrialState().maximum_tangential_opening, std::abs( separation( 0 ) ) );
}

TEST( CZMHistory, IrreversibleResponseRejectsAntiRestoringEnvelopeBranch )
{
    auto law = MakeLaw();
    law.phi_t = 2. * law.phi_n;
    const Eigen::Vector2d separation( 2. * law.delta_t, .2 );
    const auto envelope = plugin::EvaluateExponentialCZMEnvelope( law, separation );
    ASSERT_LT( envelope.traction( 1 ), 0. );

    plugin::CZMHistory history;
    const auto evaluation = plugin::EvaluateIrreversibleExponentialCZM( law, separation, history );
    EXPECT_DOUBLE_EQ( evaluation.traction( 1 ), 0. );
    EXPECT_DOUBLE_EQ( history.TrialState().normal_unloading_stiffness, 0. );
    EXPECT_GE( evaluation.traction( 0 ) * separation( 0 ), 0. );
}

TEST( CZMHistory, DampingUsesAcceptedOpeningAndGuardsZeroLoadIncrement )
{
    const auto law = MakeLaw();
    plugin::CZMHistory history;
    const Eigen::Vector2d accepted( .1, .2 );
    plugin::EvaluateIrreversibleExponentialCZM( law, accepted, history );
    history.CommitStep();

    const Eigen::Vector2d separation( .16, .27 );
    const auto undamped = plugin::EvaluateIrreversibleExponentialCZM( law, separation, history );
    const auto zero_increment = plugin::EvaluateIrreversibleExponentialCZM( law, separation, history, .3, .4, 0. );
    constexpr double tiny_delta_lambda = 1e-16;
    const auto tiny_increment = plugin::EvaluateIrreversibleExponentialCZM( law, separation, history, .3, .4, tiny_delta_lambda );
    ExpectMatricesNear( zero_increment.traction, undamped.traction, 1e-13 );
    ExpectMatricesNear( zero_increment.tangent, undamped.tangent, 1e-13 );
    const double minimum_lambda_scale = std::sqrt( std::numeric_limits<double>::epsilon() );
    const double tiny_normal_stiffness = 2. * .3 / law.delta_n / minimum_lambda_scale;
    const double tiny_tangential_stiffness = 2. * .4 / law.delta_t / minimum_lambda_scale;
    EXPECT_NEAR( tiny_increment.traction( 0 ) - undamped.traction( 0 ),
                 tiny_tangential_stiffness * ( separation( 0 ) - accepted( 0 ) ), 1e-7 );
    EXPECT_NEAR( tiny_increment.traction( 1 ) - undamped.traction( 1 ),
                 tiny_normal_stiffness * ( separation( 1 ) - accepted( 1 ) ), 1e-7 );
    EXPECT_TRUE( zero_increment.traction.allFinite() );
    EXPECT_TRUE( zero_increment.tangent.allFinite() );

    constexpr double delta_lambda = .25;
    constexpr double normal_damping = .3;
    constexpr double tangential_damping = .4;
    const auto damped = plugin::EvaluateIrreversibleExponentialCZM( law, separation, history, normal_damping,
                                                                    tangential_damping, delta_lambda );
    const auto negative_increment = plugin::EvaluateIrreversibleExponentialCZM(
        law, separation, history, normal_damping, tangential_damping, -delta_lambda );
    const double normal_stiffness = 2. * normal_damping / law.delta_n / delta_lambda;
    const double tangential_stiffness = 2. * tangential_damping / law.delta_t / delta_lambda;
    EXPECT_NEAR( damped.traction( 0 ) - undamped.traction( 0 ), tangential_stiffness * ( separation( 0 ) - accepted( 0 ) ), 1e-13 );
    EXPECT_NEAR( damped.traction( 1 ) - undamped.traction( 1 ), normal_stiffness * ( separation( 1 ) - accepted( 1 ) ), 1e-13 );
    EXPECT_NEAR( damped.tangent( 0, 0 ) - undamped.tangent( 0, 0 ), tangential_stiffness, 1e-13 );
    EXPECT_NEAR( damped.tangent( 1, 1 ) - undamped.tangent( 1, 1 ), normal_stiffness, 1e-13 );
    ExpectMatricesNear( negative_increment.traction, damped.traction, 1e-13 );
    ExpectMatricesNear( negative_increment.tangent, damped.tangent, 1e-13 );

    const auto negative_damping = plugin::EvaluateIrreversibleExponentialCZM( law, separation, history, -.3, -.4, delta_lambda );
    const auto nonfinite_damping = plugin::EvaluateIrreversibleExponentialCZM(
        law, separation, history, std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN(), delta_lambda );
    ExpectMatricesNear( negative_damping.traction, undamped.traction, 1e-13 );
    ExpectMatricesNear( negative_damping.tangent, undamped.tangent, 1e-13 );
    ExpectMatricesNear( nonfinite_damping.traction, undamped.traction, 1e-13 );
    ExpectMatricesNear( nonfinite_damping.tangent, undamped.tangent, 1e-13 );
}

TEST( CZMHistory, NormalDampingIsInactiveInCompression )
{
    const auto law = MakeLaw();
    plugin::CZMHistory history;
    plugin::EvaluateIrreversibleExponentialCZM( law, Eigen::Vector2d( .1, .2 ), history );
    history.CommitStep();

    const Eigen::Vector2d compression( .13, -.2 );
    const auto undamped = plugin::EvaluateIrreversibleExponentialCZM( law, compression, history );
    const auto damped = plugin::EvaluateIrreversibleExponentialCZM( law, compression, history, .4, 0., .25 );
    ExpectMatricesNear( damped.traction, undamped.traction, 1e-13 );
    ExpectMatricesNear( damped.tangent, undamped.tangent, 1e-13 );
    EXPECT_DOUBLE_EQ( damped.traction( 1 ), 0. );
}

TEST( CZMHistory, NormalDampingDoesNotCreateClosingOrContactTraction )
{
    const auto law = MakeLaw();
    plugin::CZMHistory history;
    plugin::EvaluateIrreversibleExponentialCZM( law, Eigen::Vector2d( 0., .2 ), history );
    history.CommitStep();

    for ( const double normal_opening : { .1, 0., -1e-12 } )
    {
        const Eigen::Vector2d separation( 0., normal_opening );
        const auto undamped = plugin::EvaluateIrreversibleExponentialCZM( law, separation, history );
        const auto damped = plugin::EvaluateIrreversibleExponentialCZM( law, separation, history, .4, 0., .25 );
        ExpectMatricesNear( damped.traction, undamped.traction, 1e-13 );
        ExpectMatricesNear( damped.tangent, undamped.tangent, 1e-13 );
    }
}

TEST( CZMHistory, ReopeningDampingUsesProjectedCommittedOpening )
{
    const auto law = MakeLaw();
    plugin::CZMHistory history;
    plugin::EvaluateIrreversibleExponentialCZM( law, Eigen::Vector2d( 0., .2 ), history );
    history.CommitStep();
    plugin::EvaluateIrreversibleExponentialCZM( law, Eigen::Vector2d( 0., -.3 ), history );
    history.CommitStep();
    ASSERT_DOUBLE_EQ( history.CommittedState().normal_opening, 0. );

    constexpr double damping = .4;
    constexpr double delta_lambda = .25;
    const Eigen::Vector2d reopening( 0., .1 );
    const auto undamped = plugin::EvaluateIrreversibleExponentialCZM( law, reopening, history );
    const auto damped = plugin::EvaluateIrreversibleExponentialCZM( law, reopening, history, damping, 0., delta_lambda );
    const double damping_stiffness = 2. * damping / law.delta_n / delta_lambda;
    EXPECT_NEAR( damped.traction( 1 ) - undamped.traction( 1 ), damping_stiffness * reopening( 1 ), 1e-13 );
}

TEST( CZMHistory, OnlyTangentialMaximumUsesEnvelopeCapWithoutStiffnessRecovery )
{
    const auto law = MakeLaw();
    plugin::CZMHistory history;
    plugin::EvaluateIrreversibleExponentialCZM( law, Eigen::Vector2d( .2, .25 ), history );
    history.CommitStep();
    const auto committed = history.CommittedState();

    const Eigen::Vector2d separation( .3, .15 );
    const auto envelope = plugin::EvaluateExponentialCZMEnvelope( law, separation );
    const auto evaluation = plugin::EvaluateIrreversibleExponentialCZM( law, separation, history );
    const auto finite_difference = FiniteDifferenceTangent(
        [&law, &history]( const Eigen::VectorXd& value )
        { return plugin::EvaluateIrreversibleExponentialCZM( law, value, history ).traction; }, separation );
    plugin::EvaluateIrreversibleExponentialCZM( law, separation, history );

    EXPECT_DOUBLE_EQ( history.TrialState().maximum_normal_opening, committed.maximum_normal_opening );
    EXPECT_DOUBLE_EQ( history.TrialState().maximum_tangential_opening, .3 );
    EXPECT_LE( history.TrialState().normal_unloading_stiffness, committed.normal_unloading_stiffness );
    EXPECT_LE( history.TrialState().tangential_unloading_stiffness, committed.tangential_unloading_stiffness );
    EXPECT_LE( std::abs( evaluation.traction( 1 ) ), std::abs( envelope.traction( 1 ) ) + 1e-13 );
    EXPECT_LE( std::abs( evaluation.traction( 0 ) ), std::abs( envelope.traction( 0 ) ) + 1e-13 );
    ExpectMatricesNear( evaluation.tangent, finite_difference, 2e-7 );
    EXPECT_GT( std::abs( evaluation.tangent( 0, 1 ) - evaluation.tangent( 1, 0 ) ), 1e-6 );
}

TEST( CZMHistory, OnlyNormalMaximumUsesEnvelopeCapWithoutStiffnessRecovery )
{
    const auto law = MakeLaw();
    plugin::CZMHistory history;
    plugin::EvaluateIrreversibleExponentialCZM( law, Eigen::Vector2d( .2, .25 ), history );
    history.CommitStep();
    const auto committed = history.CommittedState();

    const Eigen::Vector2d separation( .1, .4 );
    const auto envelope = plugin::EvaluateExponentialCZMEnvelope( law, separation );
    const auto evaluation = plugin::EvaluateIrreversibleExponentialCZM( law, separation, history );
    const auto finite_difference = FiniteDifferenceTangent(
        [&law, &history]( const Eigen::VectorXd& value )
        { return plugin::EvaluateIrreversibleExponentialCZM( law, value, history ).traction; }, separation );
    plugin::EvaluateIrreversibleExponentialCZM( law, separation, history );

    EXPECT_DOUBLE_EQ( history.TrialState().maximum_normal_opening, .4 );
    EXPECT_DOUBLE_EQ( history.TrialState().maximum_tangential_opening, committed.maximum_tangential_opening );
    EXPECT_LE( history.TrialState().normal_unloading_stiffness, committed.normal_unloading_stiffness );
    EXPECT_LE( history.TrialState().tangential_unloading_stiffness, committed.tangential_unloading_stiffness );
    EXPECT_LE( std::abs( evaluation.traction( 1 ) ), std::abs( envelope.traction( 1 ) ) + 1e-13 );
    EXPECT_LE( std::abs( evaluation.traction( 0 ) ), std::abs( envelope.traction( 0 ) ) + 1e-13 );
    ExpectMatricesNear( evaluation.tangent, finite_difference, 2e-7 );
}

TEST( CZMHistory, DeepCompressionDoesNotCommitAntiRestoringStiffness )
{
    const auto law = MakeLaw();
    plugin::CZMHistory history;
    plugin::EvaluateIrreversibleExponentialCZM( law, Eigen::Vector2d( .2, .25 ), history );
    history.CommitStep();
    const auto committed = history.CommittedState();

    const Eigen::Vector2d deep_compression( .12, -1.5 * law.delta_n );
    const auto envelope = plugin::EvaluateExponentialCZMEnvelope( law, deep_compression );
    const auto evaluation = plugin::EvaluateIrreversibleExponentialCZM( law, deep_compression, history );
    EXPECT_DOUBLE_EQ( envelope.traction( 1 ), 0. );
    EXPECT_GE( envelope.traction( 0 ) * deep_compression( 0 ), 0. );
    EXPECT_GE( envelope.tangent( 0, 0 ), 0. );
    EXPECT_GE( history.TrialState().normal_unloading_stiffness, 0. );
    EXPECT_GE( history.TrialState().tangential_unloading_stiffness, 0. );
    EXPECT_DOUBLE_EQ( history.TrialState().maximum_normal_opening, committed.maximum_normal_opening );
    EXPECT_DOUBLE_EQ( history.TrialState().normal_unloading_stiffness, committed.normal_unloading_stiffness );
    EXPECT_GE( evaluation.traction( 0 ) * deep_compression( 0 ), 0. );

    history.CommitStep();
    const Eigen::Vector2d reloading( .15, .18 );
    const auto reloaded = plugin::EvaluateIrreversibleExponentialCZM( law, reloading, history );
    EXPECT_GE( reloaded.traction( 0 ) * reloading( 0 ), 0. );
    EXPECT_GE( reloaded.traction( 1 ) * reloading( 1 ), 0. );
    EXPECT_GE( history.TrialState().normal_unloading_stiffness, 0. );
    EXPECT_GE( history.TrialState().tangential_unloading_stiffness, 0. );
}

TEST( CZMHistory, AnalyticEnvelopeTangentMatchesFiniteDifferenceIn2D )
{
    const auto law = MakeLaw();
    const Eigen::Vector2d separation( .13, .22 );
    const auto evaluation = plugin::EvaluateExponentialCZMEnvelope( law, separation );
    const auto finite_difference =
        FiniteDifferenceTangent( [&law]( const Eigen::VectorXd& value )
                                 { return plugin::EvaluateExponentialCZMEnvelope( law, value ).traction; }, separation );

    ExpectMatricesNear( evaluation.tangent, finite_difference, 2e-7 );
    ExpectMatricesNear( evaluation.tangent, evaluation.tangent.transpose(), 1e-14 );
}

TEST( CZMHistory, SecantTangentMatchesFiniteDifferenceIn2D )
{
    const auto law = MakeLaw();
    plugin::CZMHistory history;
    plugin::EvaluateIrreversibleExponentialCZM( law, Eigen::Vector2d( .3, .35 ), history );
    history.CommitStep();
    const Eigen::Vector2d separation( .14, .18 );
    const auto evaluation = plugin::EvaluateIrreversibleExponentialCZM( law, separation, history );
    const auto finite_difference = FiniteDifferenceTangent(
        [&law, &history]( const Eigen::VectorXd& value )
        { return plugin::EvaluateIrreversibleExponentialCZM( law, value, history ).traction; }, separation );

    ExpectMatricesNear( evaluation.tangent, finite_difference, 2e-7 );
    ExpectMatricesNear( evaluation.tangent, evaluation.tangent.transpose(), 1e-14 );
}

TEST( CZMHistory, AnalyticEnvelopeTangentAndTangentialMaximumWorkIn3D )
{
    const auto law = MakeLaw();
    plugin::CZMHistory history;
    const Eigen::Vector3d separation( .12, -.09, .21 );
    const auto evaluation = plugin::EvaluateExponentialCZMEnvelope( law, separation );
    const auto finite_difference =
        FiniteDifferenceTangent( [&law]( const Eigen::VectorXd& value )
                                 { return plugin::EvaluateExponentialCZMEnvelope( law, value ).traction; }, separation );

    ExpectMatricesNear( evaluation.tangent, finite_difference, 2e-7 );
    ExpectMatricesNear( evaluation.tangent, evaluation.tangent.transpose(), 1e-14 );

    plugin::EvaluateIrreversibleExponentialCZM( law, separation, history );
    EXPECT_NEAR( history.TrialState().maximum_tangential_opening, .15, 1e-14 );
    EXPECT_DOUBLE_EQ( history.TrialState().maximum_normal_opening, .21 );
}

TEST( CZMHistory, ThreeDimensionalUnloadingTracksTangentialDirection )
{
    const auto law = MakeLaw();
    plugin::CZMHistory history;
    plugin::EvaluateIrreversibleExponentialCZM( law, Eigen::Vector3d( .2, -.15, .3 ), history );
    history.CommitStep();
    const auto committed = history.CommittedState();

    const Eigen::Vector3d unloading( -.08, .06, .14 );
    const auto evaluation = plugin::EvaluateIrreversibleExponentialCZM( law, unloading, history );
    EXPECT_NEAR( evaluation.traction( 0 ), committed.tangential_unloading_stiffness * unloading( 0 ), 1e-13 );
    EXPECT_NEAR( evaluation.traction( 1 ), committed.tangential_unloading_stiffness * unloading( 1 ), 1e-13 );
    EXPECT_NEAR( evaluation.traction( 2 ), committed.normal_unloading_stiffness * unloading( 2 ), 1e-13 );

    const auto finite_difference = FiniteDifferenceTangent(
        [&law, &history]( const Eigen::VectorXd& value )
        { return plugin::EvaluateIrreversibleExponentialCZM( law, value, history ).traction; }, unloading );
    ExpectMatricesNear( evaluation.tangent, finite_difference, 2e-7 );
}

TEST( CZMHistory, ExtremeSeparationsRemainFiniteAndDerivativeConsistent )
{
    const auto law = MakeLaw();
    const Eigen::Vector2d deep_compression( .13, -400. * law.delta_n );
    const auto compression_evaluation = plugin::EvaluateExponentialCZMEnvelope( law, deep_compression );
    const auto compression_finite_difference =
        FiniteDifferenceTangent( [&law]( const Eigen::VectorXd& value )
                                 { return plugin::EvaluateExponentialCZMEnvelope( law, value ).traction; }, deep_compression );
    EXPECT_TRUE( compression_evaluation.traction.allFinite() );
    EXPECT_TRUE( compression_evaluation.tangent.allFinite() );
    ExpectMatricesNear( compression_evaluation.tangent, compression_finite_difference, 2e-7 );

    const Eigen::Vector2d extreme_tangential( std::sqrt( 350. ) * law.delta_t, .2 );
    const auto tangential_evaluation = plugin::EvaluateExponentialCZMEnvelope( law, extreme_tangential );
    const auto tangential_finite_difference = FiniteDifferenceTangent(
        [&law]( const Eigen::VectorXd& value ) { return plugin::EvaluateExponentialCZMEnvelope( law, value ).traction; },
        extreme_tangential );
    EXPECT_TRUE( tangential_evaluation.traction.allFinite() );
    EXPECT_TRUE( tangential_evaluation.tangent.allFinite() );
    ExpectMatricesNear( tangential_evaluation.tangent, tangential_finite_difference, 2e-7 );
}

TEST( CZMHistory, AnalyticEnvelopeMatchesPureAutodiffFormula )
{
    const auto law = MakeLaw();
    for ( const Eigen::Vector2d separation :
          { Eigen::Vector2d( .13, .22 ), Eigen::Vector2d( -.19, .31 ), Eigen::Vector2d( .11, -.8 ) } )
    {
        const auto analytic = plugin::EvaluateExponentialCZMEnvelope( law, separation );
        const auto automatic = plugin::EvaluateExponentialCZMEnvelopeAutodiff( law, separation );
        ExpectMatricesNear( analytic.traction, automatic.traction, 2e-13 );
        ExpectMatricesNear( analytic.tangent, automatic.tangent, 2e-13 );
    }
}

TEST( CZMIntegrator, RotatingGeneralizedLawReportsDampingUnsupported )
{
    plugin::Memorize memorize( nullptr );
    mfem::ConstantCoefficient sigma_max( 4.2 );
    mfem::ConstantCoefficient tau_max( 3.1 );
    mfem::ConstantCoefficient delta_n( .5 );
    mfem::ConstantCoefficient delta_t( .7 );
    plugin::ExponentialRotADCZMIntegrator integrator( memorize, sigma_max, tau_max, delta_n, delta_t );

    EXPECT_FALSE( integrator.SupportsDamping() );
    integrator.SetDamping( 0., 0. );
}
