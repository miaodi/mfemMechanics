#include "CZM.h"
#include "FEMPlugin.h"
#include "Solvers.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace plugin
{
namespace
{
mfem::real_t OpeningTolerance( const mfem::real_t scale, const mfem::real_t characteristic_length )
{
    return 64. * std::numeric_limits<mfem::real_t>::epsilon() * std::max( std::abs( scale ), std::abs( characteristic_length ) );
}

mfem::real_t DampingStiffness( const mfem::real_t damping, const mfem::real_t length, const mfem::real_t delta_lambda )
{
    if ( damping <= 0. || !std::isfinite( damping ) || !std::isfinite( length ) || !std::isfinite( delta_lambda ) ||
         length == 0. || delta_lambda == 0. )
    {
        return 0.;
    }

    const mfem::real_t lambda_scale =
        std::max( std::abs( delta_lambda ), std::sqrt( std::numeric_limits<mfem::real_t>::epsilon() ) );
    const mfem::real_t stiffness = 2. * damping / std::abs( length ) / lambda_scale;
    return std::isfinite( stiffness ) ? stiffness : 0.;
}

bool ValidExponentialLaw( const ExponentialCZMConst& law )
{
    return std::isfinite( law.delta_n ) && std::isfinite( law.delta_t ) && std::isfinite( law.phi_n ) &&
           std::isfinite( law.phi_t ) && law.delta_n > 0. && law.delta_t > 0. && law.phi_n >= 0. && law.phi_t >= 0.;
}

CZMEvaluation ZeroEvaluation( const int dim )
{
    CZMEvaluation result;
    result.traction = Eigen::VectorXr::Zero( dim );
    result.tangent = Eigen::MatrixXr::Zero( dim, dim );
    return result;
}

mfem::real_t NonnegativeStiffness( const mfem::real_t stiffness )
{
    return std::isfinite( stiffness ) && stiffness > 0. ? stiffness : 0.;
}

enum class ComponentBranch
{
    ENVELOPE,
    SECANT,
    ZERO
};

struct ComponentDecision
{
    ComponentBranch branch;
    mfem::real_t stiffness;
};

ComponentDecision SelectComponentBranch( const bool has_history, const mfem::real_t committed_stiffness, const mfem::real_t envelope_stiffness )
{
    // Each component uses the lower of its accepted secant and the current
    // coupled-envelope secant. Ties stay on the accepted secant, so neither a
    // changing mode mix nor reloading can recover stiffness.
    const mfem::real_t current_envelope_stiffness = NonnegativeStiffness( envelope_stiffness );
    if ( current_envelope_stiffness == 0. )
    {
        return { ComponentBranch::ZERO, 0. };
    }
    if ( !has_history )
    {
        return { ComponentBranch::ENVELOPE, current_envelope_stiffness };
    }

    const mfem::real_t accepted_stiffness = NonnegativeStiffness( committed_stiffness );
    if ( accepted_stiffness <= current_envelope_stiffness )
    {
        return { ComponentBranch::SECANT, accepted_stiffness };
    }
    return { ComponentBranch::ENVELOPE, current_envelope_stiffness };
}

autodiff::dual2nd ExponentialCZMPotential( const ExponentialCZMConst& law, const autodiff::VectorXdual2nd& local_separation )
{
    if ( !ValidExponentialLaw( law ) )
    {
        return 0.;
    }

    const int normal = local_separation.size() - 1;
    autodiff::dual2nd positive_normal_opening = local_separation( normal );
    if ( positive_normal_opening < 0. )
    {
        positive_normal_opening = 0.;
    }

    autodiff::dual2nd tangential_opening_squared = 0.;
    for ( int i = 0; i < normal; i++ )
    {
        tangential_opening_squared += local_separation( i ) * local_separation( i );
    }

    const autodiff::dual2nd normalized_normal = positive_normal_opening / law.delta_n;
    const autodiff::dual2nd mixed_energy =
        law.phi_n - law.phi_t + law.phi_t * autodiff::detail::exp( -tangential_opening_squared / law.delta_t / law.delta_t );
    return law.phi_n - autodiff::detail::exp( -normalized_normal ) * ( 1. + normalized_normal ) * mixed_energy;
}
} // namespace

void CZMHistory::BeginStep() noexcept
{
    mTrial = mCommitted;
}

void CZMHistory::CommitStep() noexcept
{
    mCommitted = mTrial;
}

void CZMHistory::RollbackStep() noexcept
{
    mTrial = mCommitted;
}

CZMEvaluation CZMHistory::EvaluateTrial( const Eigen::VectorXr& local_separation,
                                         const CZMEvaluation& envelope,
                                         const mfem::real_t normal_length,
                                         const mfem::real_t tangential_length,
                                         const mfem::real_t normal_damping,
                                         const mfem::real_t tangential_damping,
                                         const mfem::real_t delta_lambda )
{
    const int dim = local_separation.size();
    MFEM_VERIFY( dim == 2 || dim == 3, "CZM history supports only two- and three-dimensional separations." );
    MFEM_VERIFY( envelope.traction.size() == dim, "CZM envelope traction has an inconsistent size." );

    mTrial = mCommitted;
    const bool has_tangent = envelope.tangent.rows() == dim && envelope.tangent.cols() == dim;
    if ( !local_separation.allFinite() || !envelope.traction.allFinite() || ( has_tangent && !envelope.tangent.allFinite() ) )
    {
        return ZeroEvaluation( dim );
    }

    const int normal = dim - 1;
    const mfem::real_t normal_opening = std::max( local_separation( normal ), mfem::real_t{ 0 } );
    const mfem::real_t tangential_opening = local_separation.head( normal ).stableNorm();
    if ( !std::isfinite( tangential_opening ) )
    {
        return ZeroEvaluation( dim );
    }

    CZMEvaluation result = envelope;
    mTrial.normal_unloading_stiffness = NonnegativeStiffness( mCommitted.normal_unloading_stiffness );
    mTrial.tangential_unloading_stiffness = NonnegativeStiffness( mCommitted.tangential_unloading_stiffness );

    const mfem::real_t normal_tolerance = OpeningTolerance( mCommitted.maximum_normal_opening, normal_length );
    const mfem::real_t tangential_tolerance = OpeningTolerance( mCommitted.maximum_tangential_opening, tangential_length );

    mTrial.normal_opening = normal_opening;
    mTrial.tangential_opening_1 = local_separation( 0 );
    mTrial.tangential_opening_2 = dim == 3 ? local_separation( 1 ) : 0.;

    const bool new_normal_maximum = normal_opening > mCommitted.maximum_normal_opening + normal_tolerance;
    const bool new_tangential_maximum = tangential_opening > mCommitted.maximum_tangential_opening + tangential_tolerance;
    const bool activate_normal_history = mCommitted.has_normal_history || new_normal_maximum ||
                                         ( new_tangential_maximum && local_separation( normal ) >= 0. );
    const bool activate_tangential_history = mCommitted.has_tangential_history || new_tangential_maximum || new_normal_maximum;

    if ( new_normal_maximum )
    {
        mTrial.maximum_normal_opening = normal_opening;
    }

    if ( local_separation( normal ) < 0. )
    {
        // Compression is outside the cohesive opening history. Contact, when
        // needed, is supplied by a separate penalty/contact formulation.
        result.traction( normal ) = 0.;
        if ( has_tangent )
        {
            result.tangent.row( normal ).setZero();
        }
    }
    else if ( activate_normal_history )
    {
        const mfem::real_t envelope_stiffness =
            normal_opening > normal_tolerance
                ? envelope.traction( normal ) / normal_opening
                : ( has_tangent ? envelope.tangent( normal, normal ) : mCommitted.normal_unloading_stiffness );
        const ComponentDecision decision =
            SelectComponentBranch( mCommitted.has_normal_history, mCommitted.normal_unloading_stiffness, envelope_stiffness );
        mTrial.normal_unloading_stiffness = decision.stiffness;
        mTrial.has_normal_history = true;

        if ( decision.branch == ComponentBranch::SECANT )
        {
            result.traction( normal ) = decision.stiffness * local_separation( normal );
            if ( has_tangent )
            {
                result.tangent.row( normal ).setZero();
                result.tangent( normal, normal ) = decision.stiffness;
            }
        }
        else if ( decision.branch == ComponentBranch::ZERO )
        {
            result.traction( normal ) = 0.;
            if ( has_tangent )
            {
                result.tangent.row( normal ).setZero();
            }
        }
    }

    if ( new_tangential_maximum )
    {
        mTrial.maximum_tangential_opening = tangential_opening;
    }

    if ( activate_tangential_history )
    {
        mfem::real_t envelope_stiffness = mCommitted.tangential_unloading_stiffness;
        if ( tangential_opening > tangential_tolerance )
        {
            const Eigen::VectorXr tangential_direction = local_separation.head( normal ) / tangential_opening;
            envelope_stiffness = tangential_direction.dot( envelope.traction.head( normal ) ) / tangential_opening;
        }
        else if ( has_tangent )
        {
            envelope_stiffness = envelope.tangent( 0, 0 );
        }

        const ComponentDecision decision = SelectComponentBranch(
            mCommitted.has_tangential_history, mCommitted.tangential_unloading_stiffness, envelope_stiffness );
        mTrial.tangential_unloading_stiffness = decision.stiffness;
        mTrial.has_tangential_history = true;

        if ( decision.branch == ComponentBranch::SECANT )
        {
            result.traction.head( normal ) = decision.stiffness * local_separation.head( normal );
            if ( has_tangent )
            {
                result.tangent.topRows( normal ).setZero();
                result.tangent.topLeftCorner( normal, normal ).diagonal().setConstant( decision.stiffness );
            }
        }
        else if ( decision.branch == ComponentBranch::ZERO )
        {
            result.traction.head( normal ).setZero();
            if ( has_tangent )
            {
                result.tangent.topRows( normal ).setZero();
            }
        }
    }

    // Normal and tangential rows select their branches independently. If only
    // one component is envelope-governed, the exact branch Jacobian is
    // generally nonsymmetric; mirroring a cross entry would be inconsistent
    // with the returned traction.

    const mfem::real_t normal_opening_increment = normal_opening - mCommitted.normal_opening;
    const mfem::real_t normal_damping_stiffness =
        normal_opening_increment > 0. ? DampingStiffness( normal_damping, normal_length, delta_lambda ) : 0.;
    const mfem::real_t tangential_damping_stiffness = DampingStiffness( tangential_damping, tangential_length, delta_lambda );

    result.traction( normal ) += normal_damping_stiffness * normal_opening_increment;
    result.traction( 0 ) += tangential_damping_stiffness * ( local_separation( 0 ) - mCommitted.tangential_opening_1 );
    if ( dim == 3 )
    {
        result.traction( 1 ) += tangential_damping_stiffness * ( local_separation( 1 ) - mCommitted.tangential_opening_2 );
    }

    if ( has_tangent )
    {
        result.tangent( normal, normal ) += normal_damping_stiffness;
        for ( int i = 0; i < normal; i++ )
        {
            result.tangent( i, i ) += tangential_damping_stiffness;
        }
    }

    if ( !result.traction.allFinite() || ( has_tangent && !result.tangent.allFinite() ) )
    {
        mTrial = mCommitted;
        return ZeroEvaluation( dim );
    }
    return result;
}

CZMEvaluation EvaluateExponentialCZMEnvelope( const ExponentialCZMConst& law, const Eigen::VectorXr& local_separation )
{
    const int dim = local_separation.size();
    MFEM_VERIFY( dim == 2 || dim == 3, "The exponential CZM law supports only two and three dimensions." );

    CZMEvaluation result = ZeroEvaluation( dim );

    if ( !ValidExponentialLaw( law ) || !local_separation.allFinite() )
    {
        return result;
    }

    const int normal = dim - 1;
    const mfem::real_t delta_n = law.delta_n;
    const mfem::real_t delta_t_squared = law.delta_t * law.delta_t;
    const bool positive_normal_branch = local_separation( normal ) >= 0.;
    const mfem::real_t normalized_normal = positive_normal_branch ? local_separation( normal ) / delta_n : 0.;
    const mfem::real_t normalized_tangential_squared = local_separation.head( normal ).squaredNorm() / delta_t_squared;
    if ( !std::isfinite( normalized_normal ) )
    {
        return result;
    }

    const mfem::real_t exponential_normal = std::exp( -normalized_normal );
    if ( exponential_normal == 0. )
    {
        return result;
    }
    const mfem::real_t exponential_tangential =
        std::isfinite( normalized_tangential_squared ) ? std::exp( -normalized_tangential_squared ) : 0.;
    const mfem::real_t mixed_energy = law.phi_n - law.phi_t + law.phi_t * exponential_tangential;
    const mfem::real_t tangential_factor =
        2. * law.phi_t * exponential_normal * exponential_tangential * ( 1. + normalized_normal ) / delta_t_squared;

    if ( tangential_factor != 0. )
    {
        result.traction.head( normal ) = tangential_factor * local_separation.head( normal );
        result.tangent.topLeftCorner( normal, normal ) =
            tangential_factor * Eigen::MatrixXr::Identity( normal, normal ) -
            2. * tangential_factor / delta_t_squared *
                ( local_separation.head( normal ) * local_separation.head( normal ).transpose() );
    }

    if ( positive_normal_branch )
    {
        result.traction( normal ) = normalized_normal * exponential_normal * mixed_energy / delta_n;
        result.tangent( normal, normal ) = exponential_normal * ( 1. - normalized_normal ) * mixed_energy / ( delta_n * delta_n );

        for ( int i = 0; i < normal; i++ )
        {
            const mfem::real_t coupling = -2. * law.phi_t * normalized_normal * exponential_normal *
                                          exponential_tangential * local_separation( i ) / ( delta_n * delta_t_squared );
            result.tangent( i, normal ) = coupling;
            result.tangent( normal, i ) = coupling;
        }
    }

    return result.traction.allFinite() && result.tangent.allFinite() ? result : ZeroEvaluation( dim );
}

CZMEvaluation EvaluateExponentialCZMEnvelopeAutodiff( const ExponentialCZMConst& law, const Eigen::VectorXr& local_separation )
{
    const int dim = local_separation.size();
    MFEM_VERIFY( dim == 2 || dim == 3, "The exponential CZM law supports only two and three dimensions." );

    CZMEvaluation result = ZeroEvaluation( dim );
    if ( !ValidExponentialLaw( law ) || !local_separation.allFinite() )
    {
        return result;
    }

    autodiff::VectorXdual2nd separation = local_separation.cast<autodiff::dual2nd>();
    const auto potential = [&law]( const autodiff::VectorXdual2nd& value )
    { return ExponentialCZMPotential( law, value ); };
    autodiff::dual2nd energy;
    autodiff::VectorXdual gradient;
    result.tangent =
        autodiff::hessian( potential, autodiff::wrt( separation ), autodiff::at( separation ), energy, gradient ).cast<mfem::real_t>();
    for ( int i = 0; i < dim; i++ )
    {
        result.traction( i ) = autodiff::detail::val( gradient( i ) );
    }
    return result.traction.allFinite() && result.tangent.allFinite() ? result : ZeroEvaluation( dim );
}

CZMEvaluation EvaluateIrreversibleExponentialCZM( const ExponentialCZMConst& law,
                                                  const Eigen::VectorXr& local_separation,
                                                  CZMHistory& history,
                                                  const mfem::real_t normal_damping,
                                                  const mfem::real_t tangential_damping,
                                                  const mfem::real_t delta_lambda )
{
    const CZMEvaluation envelope = EvaluateExponentialCZMEnvelope( law, local_separation );
    if ( !ValidExponentialLaw( law ) || !local_separation.allFinite() )
    {
        history.RollbackStep();
        return envelope;
    }
    return history.EvaluateTrial( local_separation, envelope, law.delta_n, law.delta_t, normal_damping,
                                  tangential_damping, delta_lambda );
}

CZMIntegrator::CZMIntegrator( IntegrationPointStorageBase& pointStorage )
    : StepAwareNonlinearFormIntegrator(), mPointStorage{ pointStorage }
{
}

CZMIntegrator::CZMIntegrator( CZMHistoryPointStorage& pointStorage )
    : StepAwareNonlinearFormIntegrator(), mPointStorage{ pointStorage }, mHistoryPointStorage{ &pointStorage }
{
}

void CZMIntegrator::SetDamping( const mfem::real_t normal, const mfem::real_t tangential )
{
    MFEM_VERIFY( std::isfinite( normal ) && std::isfinite( tangential ) && normal >= 0. && tangential >= 0.,
                 "CZM damping coefficients must be finite and nonnegative." );
    MFEM_VERIFY( SupportsDamping() || ( normal == 0. && tangential == 0. ),
                 "This CZM integrator does not support damping for its generalized input." );
    xi_n = normal;
    xi_t = tangential;
}

CZMHistory& CZMIntegrator::GetHistory( const int gauss ) const
{
    MFEM_VERIFY( mHistoryPointStorage != nullptr,
                 "This cohesive integrator requires CZMHistory in its face integration-point storage." );
    return mHistoryPointStorage->GetFacePoint( gauss ).State;
}

CZMEvaluation CZMIntegrator::EvaluateIrreversibleLocalLaw( const ExponentialCZMConst& law,
                                                           const Eigen::VectorXr& local_separation,
                                                           const int gauss ) const
{
    return EvaluateIrreversibleExponentialCZM( law, local_separation, GetHistory( gauss ), xi_n, xi_t,
                                               mStepContext->GetDeltaLambda() );
}

void CZMIntegrator::BeginStep() noexcept
{
    StepAwareNonlinearFormIntegrator::BeginStep();
    if ( mStepDepth > 0 )
    {
        mStepDepth++;
        return;
    }

    VisitHistory( []( CZMHistory& history ) { history.BeginStep(); } );
    mStepDepth = 1;
    mStepRejected = false;
}

void CZMIntegrator::CommitStep() noexcept
{
    MFEM_VERIFY( mStepDepth > 0, "CZM commit requires a matching BeginStep." );
    if ( mStepDepth > 1 )
    {
        mStepDepth--;
        StepAwareNonlinearFormIntegrator::CommitStep();
        return;
    }

    if ( mStepRejected )
    {
        VisitHistory( []( CZMHistory& history ) { history.RollbackStep(); } );
    }
    else
    {
        VisitHistory( []( CZMHistory& history ) { history.CommitStep(); } );
    }
    mStepDepth = 0;
    mStepRejected = false;
    StepAwareNonlinearFormIntegrator::CommitStep();
}

void CZMIntegrator::RollbackStep() noexcept
{
    MFEM_VERIFY( mStepDepth > 0, "CZM rollback requires a matching BeginStep." );
    mStepRejected = true;
    if ( mStepDepth > 1 )
    {
        mStepDepth--;
        StepAwareNonlinearFormIntegrator::RollbackStep();
        return;
    }

    VisitHistory( []( CZMHistory& history ) { history.RollbackStep(); } );
    mStepDepth = 0;
    mStepRejected = false;
    StepAwareNonlinearFormIntegrator::RollbackStep();
}

void CZMIntegrator::AssembleFaceVector( const mfem::FiniteElement& el1,
                                        const mfem::FiniteElement& el2,
                                        mfem::FaceElementTransformations& Tr,
                                        const mfem::Vector& elfun,
                                        mfem::Vector& elvect )
{
    if ( mStepContext == nullptr )
    {
        mfem::mfem_error( "Nonlinear step context is not provided yet.\n" );
    }
    int vdim = Tr.GetSpaceDim();
    int dof1 = el1.GetDof();
    int dof2 = el2.GetDof();
    int dof = dof1 + dof2;
    MFEM_ASSERT( Tr.Elem2No >= 0, "CZMIntegrator is an internal bdr integrator" );
    elvect.SetSize( dof * vdim );
    elvect = 0.0;
    Eigen::Map<Eigen::VectorXr> eigenVec( elvect.GetData(), elvect.Size() );
    Eigen::Map<const Eigen::VectorXr> u( elfun.GetData(), elfun.Size() );

    const mfem::IntegrationRule* ir = IntRule;
    if ( ir == NULL )
    {
        int intorder = 2 * el1.GetOrder();
        ir = &mfem::IntRules.Get( Tr.GetGeometryType(), intorder );
    }

    mPointStorage.InitializeFace( el1, el2, Tr, *ir );

    for ( int i = 0; i < ir->GetNPoints(); i++ )
    {
        // Set the integration point in the face and the neighboring element
        const mfem::IntegrationPoint& ip = ir->IntPoint( i );
        Tr.SetAllIntPoints( &ip );
        EvalCZMLaw( Tr, ip );

        const auto& point = mPointStorage.GetFacePoint( i );
        matrixB( dof1, dof2, point.Shape1, point.Shape2, point.GShapeFace1, point.GShapeFace2, vdim );
        Eigen::VectorXr Delta = mB * u;
        Eigen::VectorXr T;
        Traction( Delta, i, vdim, T );
        eigenVec += mB.transpose() * T * point.Weight;
    }
}

void CZMIntegrator::AssembleFaceGrad( const mfem::FiniteElement& el1,
                                      const mfem::FiniteElement& el2,
                                      mfem::FaceElementTransformations& Tr,
                                      const mfem::Vector& elfun,
                                      mfem::DenseMatrix& elmat )
{
    if ( mStepContext == nullptr )
    {
        mfem::mfem_error( "Nonlinear step context is not provided yet.\n" );
    }
    int vdim = Tr.GetSpaceDim();
    int dof1 = el1.GetDof();
    int dof2 = el2.GetDof();
    int dof = dof1 + dof2;
    MFEM_ASSERT( Tr.Elem2No >= 0, "CZMIntegrator is an internal bdr integrator" );

    elmat.SetSize( dof * vdim );
    elmat = 0.0;
    Eigen::Map<Eigen::MatrixXr> eigenMat( elmat.Data(), dof * vdim, dof * vdim );
    Eigen::Map<const Eigen::VectorXr> u( elfun.GetData(), elfun.Size() );

    const mfem::IntegrationRule* ir = IntRule;
    if ( ir == NULL )
    {
        int intorder = 2 * el1.GetOrder();
        ir = &mfem::IntRules.Get( Tr.GetGeometryType(), intorder );
    }
    mPointStorage.InitializeFace( el1, el2, Tr, *ir );
    for ( int i = 0; i < ir->GetNPoints(); i++ )
    {
        // Set the integration point in the face and the neighboring element
        const mfem::IntegrationPoint& ip = ir->IntPoint( i );
        Tr.SetAllIntPoints( &ip );
        EvalCZMLaw( Tr, ip );

        const auto& point = mPointStorage.GetFacePoint( i );
        matrixB( dof1, dof2, point.Shape1, point.Shape2, point.GShapeFace1, point.GShapeFace2, vdim );
        Eigen::VectorXr Delta = mB * u;

        Eigen::MatrixXr H;
        TractionStiffTangent( Delta, i, vdim, H );
        eigenMat += mB.transpose() * H * mB * point.Weight;
    }
}

void CZMIntegrator::matrixB( const int dof1,
                             const int dof2,
                             const mfem::Vector& shape1,
                             const mfem::Vector& shape2,
                             const mfem::DenseMatrix& gshape1,
                             const mfem::DenseMatrix& gshape2,
                             const int dim )
{
    mB.resize( dim, dim * ( dof1 + dof2 ) );
    mB.setZero();

    for ( int i = 0; i < dof1; i++ )
    {
        for ( int j = 0; j < dim; j++ )
        {
            mB( j, i + j * dof1 ) = shape1( i );
        }
    }
    for ( int i = 0; i < dof2; i++ )
    {
        for ( int j = 0; j < dim; j++ )
        {
            mB( j, i + j * dof2 + dim * dof1 ) = -shape2( i );
        }
    }
}

void LinearCZMIntegrator::Traction( const Eigen::VectorXr& Delta, const int i, const int dim, Eigen::VectorXr& T ) const
{
    if ( dim == 2 )
    {
        T.resize( 2 );
        // Tt
        if ( std::abs( Delta( 0 ) ) <= mDeltaT )
        {
            T( 0 ) = mTauMax * Delta( 0 ) / mDeltaT;
        }
        else if ( mDeltaT < Delta( 0 ) && Delta( 0 ) <= mDeltaTMax )
        {
            T( 0 ) = mTauMax * ( mDeltaTMax - Delta( 0 ) ) / ( mDeltaTMax - mDeltaT );
        }
        else if ( -mDeltaT > Delta( 0 ) && Delta( 0 ) >= -mDeltaTMax )
        {
            T( 0 ) = -mTauMax * ( mDeltaTMax + Delta( 0 ) ) / ( mDeltaTMax - mDeltaT );
        }
        else
        {
            T( 0 ) = 0;
        }
        // Tn
        if ( Delta( 1 ) <= mDeltaN )
        {
            T( 1 ) = mSigmaMax * Delta( 1 ) / mDeltaN;
        }
        else if ( mDeltaN < Delta( 1 ) && Delta( 1 ) <= mDeltaNMax )
        {
            T( 1 ) = mSigmaMax * ( mDeltaNMax - Delta( 1 ) ) / ( mDeltaNMax - mDeltaN );
        }
        else
        {
            T( 1 ) = 0;
        }
    }
}

void LinearCZMIntegrator::TractionStiffTangent( const Eigen::VectorXr& Delta, const int i, const int dim, Eigen::MatrixXr& H ) const
{
    if ( dim == 2 )
    {
        H.resize( 2, 2 );
        H( 1, 0 ) = H( 0, 1 ) = 0.;
        // Tt
        if ( std::abs( Delta( 0 ) ) <= mDeltaT )
        {
            H( 0, 0 ) = mTauMax / mDeltaT;
        }
        else if ( mDeltaT < Delta( 0 ) && Delta( 0 ) <= mDeltaTMax )
        {
            H( 0, 0 ) = -mTauMax / ( mDeltaTMax - mDeltaT );
        }
        else if ( -mDeltaT > Delta( 0 ) && Delta( 0 ) >= -mDeltaTMax )
        {
            H( 0, 0 ) = -mTauMax / ( mDeltaTMax - mDeltaT );
        }
        else
        {
            H( 0, 0 ) = 0;
        }
        // Tn
        if ( Delta( 1 ) <= mDeltaN )
        {
            H( 1, 1 ) = mSigmaMax / mDeltaN;
        }
        else if ( mDeltaN < Delta( 1 ) && Delta( 1 ) <= mDeltaNMax )
        {
            H( 1, 1 ) = -mSigmaMax / ( mDeltaNMax - mDeltaN );
        }
        else
        {
            H( 1, 1 ) = 0;
        }
    }
}

void ExponentialCZMIntegrator::EvalCZMLaw( mfem::ElementTransformation& Tr, const mfem::IntegrationPoint& ip )
{
    mCZMLawConst.sigma_max = mSigmaMax->Eval( Tr, ip );
    mCZMLawConst.tau_max = mTauMax->Eval( Tr, ip );
    mCZMLawConst.delta_n = mDeltaN->Eval( Tr, ip );
    mCZMLawConst.delta_t = mDeltaT->Eval( Tr, ip );
    mCZMLawConst.update_phi();
}

void ExponentialCZMIntegrator::Traction( const Eigen::VectorXr& Delta, const int i, const int dim, Eigen::VectorXr& T ) const
{
    Eigen::MatrixXr DeltaToTN;
    DeltaToTNMat( mPointStorage.GetFaceJacobian( i ), DeltaToTN );
    const Eigen::VectorXr local_separation = DeltaToTN.transpose() * Delta;
    const CZMEvaluation evaluation = EvaluateIrreversibleLocalLaw( mCZMLawConst, local_separation, i );
    T = DeltaToTN * evaluation.traction;
}

void ExponentialCZMIntegrator::TractionStiffTangent( const Eigen::VectorXr& Delta, const int i, const int dim, Eigen::MatrixXr& H ) const
{
    Eigen::MatrixXr DeltaToTN;
    DeltaToTNMat( mPointStorage.GetFaceJacobian( i ), DeltaToTN );
    const Eigen::VectorXr local_separation = DeltaToTN.transpose() * Delta;
    const CZMEvaluation evaluation = EvaluateIrreversibleLocalLaw( mCZMLawConst, local_separation, i );
    H = DeltaToTN * evaluation.tangent * DeltaToTN.transpose();
}

void ADCZMIntegrator::Traction( const Eigen::VectorXr& Delta, const int i, const int dim, Eigen::VectorXr& T ) const
{
    autodiff::VectorXdual2nd delta = Delta.cast<autodiff::dual2nd>();
    autodiff::dual2nd u;
    T = autodiff::gradient( potential, autodiff::wrt( delta ), autodiff::at( delta, i ), u ).cast<mfem::real_t>();
}

void ADCZMIntegrator::TractionStiffTangent( const Eigen::VectorXr& Delta, const int i, const int dim, Eigen::MatrixXr& H ) const
{
    Eigen::VectorXr traction;
    EvaluatePotential( Delta, i, traction, H );
}

void ADCZMIntegrator::EvaluatePotential( const Eigen::VectorXr& Delta, const int i, Eigen::VectorXr& traction, Eigen::MatrixXr& tangent ) const
{
    autodiff::VectorXdual2nd delta = Delta.cast<autodiff::dual2nd>();
    autodiff::dual2nd u;
    autodiff::VectorXdual g;
    tangent = autodiff::hessian( potential, autodiff::wrt( delta ), autodiff::at( delta, i ), u, g ).cast<mfem::real_t>();
    traction.resize( g.size() );
    for ( int j = 0; j < g.size(); j++ )
    {
        traction( j ) = autodiff::detail::val( g( j ) );
    }
}

void ExponentialADCZMIntegrator::EvalCZMLaw( mfem::ElementTransformation& Tr, const mfem::IntegrationPoint& ip )
{
    mCZMLawConst.sigma_max = mSigmaMax->Eval( Tr, ip );
    mCZMLawConst.tau_max = mTauMax->Eval( Tr, ip );
    mCZMLawConst.delta_n = mDeltaN->Eval( Tr, ip );
    mCZMLawConst.delta_t = mDeltaT->Eval( Tr, ip );
    mCZMLawConst.update_phi();
}

void ExponentialADCZMIntegrator::Traction( const Eigen::VectorXr& Delta, const int i, const int dim, Eigen::VectorXr& T ) const
{
    Eigen::MatrixXr DeltaToTN;
    DeltaToTNMat( mPointStorage.GetFaceJacobian( i ), DeltaToTN );
    const Eigen::VectorXr local_separation = DeltaToTN.transpose() * Delta;
    CZMHistory& history = GetHistory( i );
    if ( !ValidExponentialLaw( mCZMLawConst ) || !local_separation.allFinite() )
    {
        history.RollbackStep();
        T = Eigen::VectorXr::Zero( Delta.size() );
        return;
    }

    Eigen::VectorXr envelope_traction;
    Eigen::MatrixXr envelope_tangent;
    EvaluatePotential( Delta, i, envelope_traction, envelope_tangent );

    CZMEvaluation local_envelope;
    local_envelope.traction = DeltaToTN.transpose() * envelope_traction;
    local_envelope.tangent = DeltaToTN.transpose() * envelope_tangent * DeltaToTN;
    const CZMEvaluation evaluation = history.EvaluateTrial( local_separation, local_envelope, mCZMLawConst.delta_n,
                                                            mCZMLawConst.delta_t, xi_n, xi_t, mStepContext->GetDeltaLambda() );
    T = DeltaToTN * evaluation.traction;
}

void ExponentialADCZMIntegrator::TractionStiffTangent( const Eigen::VectorXr& Delta, const int i, const int dim, Eigen::MatrixXr& H ) const
{
    Eigen::MatrixXr DeltaToTN;
    DeltaToTNMat( mPointStorage.GetFaceJacobian( i ), DeltaToTN );
    const Eigen::VectorXr local_separation = DeltaToTN.transpose() * Delta;
    CZMHistory& history = GetHistory( i );
    if ( !ValidExponentialLaw( mCZMLawConst ) || !local_separation.allFinite() )
    {
        history.RollbackStep();
        H = Eigen::MatrixXr::Zero( Delta.size(), Delta.size() );
        return;
    }

    Eigen::VectorXr envelope_traction;
    Eigen::MatrixXr envelope_tangent;
    EvaluatePotential( Delta, i, envelope_traction, envelope_tangent );

    CZMEvaluation local_envelope;
    local_envelope.traction = DeltaToTN.transpose() * envelope_traction;
    local_envelope.tangent = DeltaToTN.transpose() * envelope_tangent * DeltaToTN;
    const CZMEvaluation evaluation = history.EvaluateTrial( local_separation, local_envelope, mCZMLawConst.delta_n,
                                                            mCZMLawConst.delta_t, xi_n, xi_t, mStepContext->GetDeltaLambda() );
    H = DeltaToTN * evaluation.tangent * DeltaToTN.transpose();
}

ExponentialADCZMIntegrator::ExponentialADCZMIntegrator( CZMHistoryPointStorage& pointStorage,
                                                        mfem::Coefficient& sigmaMax,
                                                        mfem::Coefficient& tauMax,
                                                        mfem::Coefficient& deltaN,
                                                        mfem::Coefficient& deltaT )
    : ADCZMIntegrator( pointStorage ), mSigmaMax{ &sigmaMax }, mTauMax{ &tauMax }, mDeltaN{ &deltaN }, mDeltaT{ &deltaT }
{
    InitializePotential();
}

ExponentialADCZMIntegrator::ExponentialADCZMIntegrator( IntegrationPointStorageBase& pointStorage,
                                                        mfem::Coefficient& sigmaMax,
                                                        mfem::Coefficient& tauMax,
                                                        mfem::Coefficient& deltaN,
                                                        mfem::Coefficient& deltaT )
    : ADCZMIntegrator( pointStorage ), mSigmaMax{ &sigmaMax }, mTauMax{ &tauMax }, mDeltaN{ &deltaN }, mDeltaT{ &deltaT }
{
    InitializePotential();
}

void ExponentialADCZMIntegrator::InitializePotential()
{
    // x: diffX, diffY
    potential = [this]( const autodiff::VectorXdual2nd& x, const int i )
    {
        const auto& Jacobian = this->mPointStorage.GetFaceJacobian( i );
        const int dim = Jacobian.Height();
        Eigen::MatrixXr DeltaToTN;
        DeltaToTNMat( Jacobian, DeltaToTN );

        autodiff::VectorXdual2nd local_separation( dim );
        for ( int j = 0; j < dim; j++ )
        {
            local_separation( j ) = DeltaToTN.col( j ).cast<autodiff::dual2nd>().dot( x );
        }
        return ExponentialCZMPotential( mCZMLawConst, local_separation );
    };
}

ExponentialRotADCZMIntegrator::ExponentialRotADCZMIntegrator( IntegrationPointStorageBase& pointStorage,
                                                              mfem::Coefficient& sigmaMax,
                                                              mfem::Coefficient& tauMax,
                                                              mfem::Coefficient& deltaN,
                                                              mfem::Coefficient& deltaT )
    : ExponentialADCZMIntegrator( pointStorage, sigmaMax, tauMax, deltaN, deltaT )
{
    // x: u1x, u1y, u2x, u2y, du1x, du1y, du2x, du2y
    potential = [this]( const autodiff::VectorXdual2nd& x, const int i )
    {
        Eigen::Map<const autodiff::VectorXdual2nd> U1( x.data(), 2 );
        Eigen::Map<const autodiff::VectorXdual2nd> U2( x.data() + 2, 2 );
        Eigen::Map<const autodiff::VectorXdual2nd> dU1( x.data() + 4, 2 );
        Eigen::Map<const autodiff::VectorXdual2nd> dU2( x.data() + 6, 2 );
        const auto& Jacobian = this->mPointStorage.GetFaceJacobian( i );

        autodiff::VectorXdual2nd dA1( 2 );
        dA1 << Jacobian( 0, 0 ), Jacobian( 1, 0 );
        autodiff::VectorXdual2nd diff = U1 - U2;
        autodiff::VectorXdual2nd directionT = dA1 + dA1 + dU1 + dU2;
        directionT.normalize();

        static Eigen::Rotation2Dd rot( EIGEN_PI / 2 );
        autodiff::VectorXdual2nd directionN = rot.toRotationMatrix() * directionT;
        const autodiff::dual2nd DeltaT = directionT.dot( diff );
        const autodiff::dual2nd DeltaN = directionN.dot( diff );
        autodiff::VectorXdual2nd local_separation( 2 );
        local_separation << DeltaT, DeltaN;
        autodiff::dual2nd res = ExponentialCZMPotential( mCZMLawConst, local_separation );

        if ( DeltaN < 0 )
            res += 1e20 * DeltaN * DeltaN;
        return res;
    };
}

void ExponentialRotADCZMIntegrator::Traction( const Eigen::VectorXr& Delta, const int i, const int dim, Eigen::VectorXr& T ) const
{
    ADCZMIntegrator::Traction( Delta, i, dim, T );
}

void ExponentialRotADCZMIntegrator::TractionStiffTangent( const Eigen::VectorXr& Delta, const int i, const int dim, Eigen::MatrixXr& H ) const
{
    ADCZMIntegrator::TractionStiffTangent( Delta, i, dim, H );
}

void ExponentialRotADCZMIntegrator::matrixB( const int dof1,
                                             const int dof2,
                                             const mfem::Vector& shape1,
                                             const mfem::Vector& shape2,
                                             const mfem::DenseMatrix& gshape1,
                                             const mfem::DenseMatrix& gshape2,
                                             const int dim )
{
    if ( dim == 2 )
    {
        mB.resize( 8, 2 * ( dof1 + dof2 ) );
        mB.setZero();

        for ( int i = 0; i < dof1; i++ )
        {
            for ( int j = 0; j < dim; j++ )
            {
                mB( j, i + j * dof1 ) = shape1( i );
            }
        }

        for ( int i = 0; i < dof2; i++ )
        {
            for ( int j = 0; j < dim; j++ )
            {
                mB( 2 + j, i + j * dof2 + dim * dof1 ) = shape2( i );
            }
        }

        for ( int i = 0; i < dof1; i++ )
        {
            for ( int j = 0; j < dim; j++ )
            {
                mB( 4 + j, i + j * dof1 ) = gshape1( i, 0 );
            }
        }

        for ( int i = 0; i < dof2; i++ )
        {
            for ( int j = 0; j < dim; j++ )
            {
                mB( 6 + j, i + j * dof2 + dim * dof1 ) = gshape2( i, 0 );
            }
        }
    }
    else if ( dim == 3 )
    {
        std::cout << "not implemented!\n";
    }
}

void DeltaToTNMat( const mfem::DenseMatrix& Jacobian, Eigen::MatrixXr& DeltaToTN )
{
    int dim = Jacobian.Height();
    DeltaToTN.resize( dim, dim );
    if ( dim == 2 )
    {
        Eigen::Map<const Eigen::Matrix<mfem::real_t, 2, 1>> Jac( Jacobian.Data() );
        static Eigen::Rotation2D<mfem::real_t> rot( EIGEN_PI / 2 );
        DeltaToTN.col( 0 ) = Jac;
        DeltaToTN.col( 0 ).normalize();
        DeltaToTN.col( 1 ) = rot.toRotationMatrix() * DeltaToTN.col( 0 );
    }
    else if ( dim == 3 )
    {
        Eigen::Map<const Eigen::Matrix<mfem::real_t, 3, 2>> Jac( Jacobian.Data() );
        DeltaToTN.col( 0 ) = Jac.col( 0 );
        DeltaToTN.col( 0 ).normalize();
        DeltaToTN.col( 2 ) = Jac.col( 1 ).cross( Jac.col( 0 ) );
        DeltaToTN.col( 2 ).normalize();
        Eigen::Map<const Eigen::Matrix<mfem::real_t, 3, 3>> DeltaToTN33( DeltaToTN.data() );
        DeltaToTN.col( 1 ) = DeltaToTN33.col( 2 ).cross( DeltaToTN33.col( 0 ) );
    }
}
} // namespace plugin
