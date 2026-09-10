#pragma once
#include "mfem.hpp"
#include <CircularBuffer.hpp>
#include <Eigen/Dense>
#include <cstddef>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

namespace plugin
{
/** @brief Exposes nonlinear operators nested inside a composite operator. */
class CompositeNonlinearOperator
{
public:
    virtual ~CompositeNonlinearOperator() = default;

    virtual void GetChildOperators( std::vector<const mfem::Operator*>& children ) const = 0;
};

class NonlinearStepContext
{
public:
    NonlinearStepContext()
    {
    }
    int IterNumber() const
    {
        return it;
    }
    virtual bool Convergence() const = 0;

    template <typename T>
    void SetDataCollectionFunc( T&& func )
    {
        data_collect_func = func;
    }

    template <typename T>
    void SetLUpdateFunc( T&& func )
    {
        L_update_func = func;
    }

    int StepNumber() const
    {
        return step;
    }

    mfem::real_t GetCurLambda() const
    {
        return GetCurrentPseudoTime();
    }

    mfem::real_t GetDeltaLambda() const
    {
        return GetPseudoTimeIncrement();
    }

    /// Current trial pseudo-time. This is a continuation coordinate, not physical time.
    mfem::real_t GetCurrentPseudoTime() const
    {
        return lambda + Delta_lambda;
    }

    mfem::real_t GetPseudoTimeIncrement() const
    {
        return Delta_lambda;
    }

    virtual void SetDelta( const mfem::real_t delta )
    {
        Delta_lambda = delta;
    }

    void SetPrevLambda( const mfem::real_t _lambda ) const
    {
        SetPreviousPseudoTime( _lambda );
    }

    void SetPreviousPseudoTime( const mfem::real_t pseudoTime ) const
    {
        lambda = pseudoTime;
    }

    void RegisterToIntegrators( const mfem::Operator* oper ) const;

    void BeginStep( const mfem::Operator* oper ) const;
    bool CommitStep( const mfem::Operator* oper ) const;
    void RollbackStep( const mfem::Operator* oper ) const;

protected:
    mutable int it = 0; // iter # of each step

    mutable int step = 0; // step #

    // Historical names retained for source compatibility. They represent
    // accepted pseudo-time and the current trial pseudo-time increment.
    mutable mfem::real_t lambda = 0., Delta_lambda = 0;

    mutable std::function<void( int, int, mfem::real_t )> data_collect_func{ nullptr };

    // args: converged, final_iter, lambda, L
    mutable std::function<void( bool, int, mfem::real_t, mfem::real_t& )> L_update_func{ nullptr };
};

class NewtonLineSearch : public mfem::NewtonSolver, public NonlinearStepContext
{
protected:
    mfem::real_t max_eta{ 10. };
    mfem::real_t min_eta{ .1 };
    mfem::real_t eta_coef{ 1.5 };
    int max_line_search_iter{ 10 };
    mfem::real_t tol{ .006 };
    bool line_search{ false };
    mutable mfem::Vector aux_line_search;

public:
    NewtonLineSearch() : NonlinearStepContext()
    {
    }

#ifdef MFEM_USE_MPI
    NewtonLineSearch( MPI_Comm comm_ ) : NewtonSolver( comm_ )
    {
    }
#endif

    void SetLineSearchTol( const mfem::real_t t )
    {
        tol = t;
    }

    void SetMaxEta( const mfem::real_t t )
    {
        max_eta = t;
    }
    void SetLineSearch( const bool ls )
    {
        line_search = ls;
    }

    int GetMaxIter() const
    {
        return this->max_iter;
    }

    virtual bool Convergence() const
    {
        return converged;
    }

    int MyRank() const;

    virtual mfem::real_t ComputeScalingFactor( const mfem::Vector& x, const mfem::Vector& b ) const;
    virtual void SetOperator( const mfem::Operator& op );
    virtual void Mult( const mfem::Vector& b, mfem::Vector& x ) const;
};

class NewtonForPhaseField : public NewtonLineSearch
{
protected:
    mutable mfem::Vector r_u, c_u;
    mutable mfem::Vector r_p, c_p;
    mutable mfem::BlockNonlinearForm* blockOper{ nullptr };
    mfem::Array<int> block_trueOffsets;
    mfem::Solver* phaseSolver{ nullptr };

public:
    NewtonForPhaseField() : NewtonLineSearch()
    {
    }

#ifdef MFEM_USE_MPI
    NewtonForPhaseField( MPI_Comm comm_ ) : NewtonLineSearch( comm_ )
    {
    }
#endif

    /// Use one borrowed linear solver for both blocks (the original behavior).
    void SetSolver( mfem::Solver& solver ) override
    {
        NewtonLineSearch::SetSolver( solver );
        phaseSolver = nullptr;
    }

    /// Borrow separate displacement/phase solvers; both must outlive this solver.
    void SetBlockSolvers( mfem::Solver& displacement, mfem::Solver& phase )
    {
        SetSolver( displacement );
        phaseSolver = &phase;
    }

    virtual void SetOperator( const mfem::Operator& op );
    virtual void Mult( const mfem::Vector& b, mfem::Vector& x ) const;
};

class ALMBase : public mfem::IterativeSolver, public NonlinearStepContext
{
protected:
    static constexpr std::size_t PredictorHistoryCapacity = 2;

    mfem::real_t InnerProduct( const mfem::Vector& a, const mfem::real_t la, const mfem::Vector& b, const mfem::real_t lb ) const;

    void ResizeVectors( const int size ) const;

    void InitializeVariables( const mfem::Vector& u ) const;

    struct Stat
    {
        mfem::real_t L{ 0. };
        mfem::real_t lambda{ 0. };
        mfem::Vector u;
    };

public:
    ALMBase() : NonlinearStepContext()
    {
        converged = true;
    }

#ifdef MFEM_USE_MPI
    ALMBase( MPI_Comm comm_ ) : mfem::IterativeSolver( comm_ )
    {
        converged = true;
    }
#endif

    virtual void SetOperator( const Operator& op );

    /// Set the linear solver for inverting the Jacobian.
    /** This method is equivalent to calling SetPreconditioner(). */
    virtual void SetSolver( Solver& solver )
    {
        prec = &solver;
    }

    /// Solve the nonlinear system with right-hand side @a b.
    /** If `b.Size() != Height()`, then @a b is assumed to be zero. */
    virtual void Mult( const mfem::Vector& b, mfem::Vector& x ) const;

    /** @brief This method can be overloaded in derived classes to perform
        computations that need knowledge of the newest Newton state. */
    virtual void ProcessNewState( const mfem::Vector& x ) const
    {
    }

    virtual void SetDelta( const mfem::real_t l )
    {
        L = l;
        max_delta = l * 1e2;
        min_delta = l * 1e-3;
    }

    void SetPhi( const mfem::real_t p )
    {
        phi = p;
    }

    void SetMaxStep( const int step )
    {
        max_steps = step;
    }

    void SetMaxDelta( const mfem::real_t delta )
    {
        max_delta = delta;
    }

    void SetMinDelta( const mfem::real_t delta )
    {
        min_delta = delta;
    }

    void SetAMRFunc( std::function<bool( const mfem::Vector& )>& f )
    {
        adaptive_mesh_refine_func = &f;
    }

    virtual bool updateStep( const int it, const int step, const mfem::real_t det ) const = 0;

    void SetCheckConvRatio( const bool check )
    {
        check_conv_ratio = check;
    }
    void SetAdaptiveL( const bool adapt )
    {
        adaptive_l = adapt;
    }

    virtual bool Convergence() const
    {
        return converged;
    }

    // predict u_direction_pred and lambda_direction_pred
    void PredictDirection() const;

protected:
    mutable mfem::Vector r, delta_u, u_cur, q, delta_u_bar, delta_u_t, Delta_u;
    mutable mfem::Operator* grad;

    mutable mfem::Vector u_direction_pred;

    mutable mfem::real_t delta_lambda, max_delta{ 1. }, min_delta{ 1. }, L{ 1 }, phi{ 1 }, lambda_direction_pred{ 0. };

    int max_steps{ 100 };

    bool check_conv_ratio{ false };
    bool adaptive_l{ false };
    mutable std::function<bool( const mfem::Vector& )>* adaptive_mesh_refine_func{ nullptr };

    // The current and preceding accepted states define the predictor direction.
    mutable CircularBuffer<Stat, PredictorHistoryCapacity> solution_buffer;
};

class Crisfield : public ALMBase
{
public:
    Crisfield() : ALMBase()
    {
    }

#ifdef MFEM_USE_MPI
    Crisfield( MPI_Comm comm_ ) : ALMBase( comm_ )
    {
    }
#endif

    virtual bool updateStep( const int it, const int step, const mfem::real_t det ) const;
};

class ArcLengthLinearize : public ALMBase
{
public:
    ArcLengthLinearize() : ALMBase()
    {
    }

#ifdef MFEM_USE_MPI
    ArcLengthLinearize( MPI_Comm comm_ ) : ALMBase( comm_ )
    {
    }
#endif

    virtual bool updateStep( const int it, const int step, const mfem::real_t det ) const;
};

template <typename Newton>
class MultiNewtonAdaptive : public Newton
{
public:
    /// Limit total attempts (accepted and rejected) per Mult; zero means unlimited (the default).
    /// Unlimited solves still stop on minimum increments or lack of representable pseudo-time progress.
    /// Legacy int step numbers and zero-based callback attempt indices saturate at INT_MAX.
    void SetMaxStep( const int step )
    {
        MFEM_VERIFY( step >= 0, "The adaptive Newton attempt limit must be nonnegative; zero means unlimited." );
        max_steps = step;
    }

    MultiNewtonAdaptive() : Newton()
    {
    }

#ifdef MFEM_USE_MPI
    MultiNewtonAdaptive( MPI_Comm comm_ ) : Newton( comm_ )
    {
    }
#endif

    /// Solve the nonlinear system with right-hand side @a b.
    /** If `b.Size() != Height()`, then @a b is assumed to be zero. */
    virtual void Mult( const mfem::Vector& b, mfem::Vector& x ) const;
    virtual void SetOperator( const mfem::Operator& op );

    void SetDelta( const mfem::real_t delta ) override
    {
        initial_pseudo_time_increment = delta;
        Newton::SetDelta( delta );
    }

    void SetPseudoTimeInterval( const mfem::real_t initialPseudoTime, const mfem::real_t finalPseudoTime )
    {
        MFEM_VERIFY( mfem::IsFinite( initialPseudoTime ) && mfem::IsFinite( finalPseudoTime ) && finalPseudoTime > initialPseudoTime,
                     "The adaptive Newton pseudo-time interval must be finite and increasing." );
        initial_pseudo_time = initialPseudoTime;
        final_pseudo_time = finalPseudoTime;
    }

    void SetTrialStateFunc( std::function<void( mfem::real_t, mfem::Vector& )> func )
    {
        trial_state_func = std::move( func );
    }

    void SetMaxDelta( const mfem::real_t delta )
    {
        max_delta = delta;
    }

    void SetMinDelta( const mfem::real_t delta )
    {
        min_delta = delta;
    }

protected:
    int max_steps{ 0 };
    mutable mfem::Vector cur;
    mutable mfem::real_t max_delta{ 1. }, min_delta{ 0. };
    mfem::real_t initial_pseudo_time_increment{ 0. };
    mfem::real_t initial_pseudo_time{ 0. }, final_pseudo_time{ 1. };
    std::function<void( mfem::real_t, mfem::Vector& )> trial_state_func;
};
} // namespace plugin
