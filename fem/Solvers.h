#pragma once
#include "Material.h"
#include "mfem.hpp"
#include <Eigen/Dense>
#include <memory>
#include <vector>

namespace plugin
{
class NewtonLineSearch : public mfem::NewtonSolver
{
protected:
    mutable mfem::Vector u_cur;
    double max_eta{ 10. };
    double min_eta{ .1 };
    double eta_coef{ 1.5 };
    int max_line_search_iter{ 10 };
    double tol{ .006 };
    bool line_search{ false };

public:
    NewtonLineSearch()
    {
    }

    void SetLineSearchTol( const double t )
    {
        tol = t;
    }

    void SetMaxEta( const double t )
    {
        max_eta = t;
    }
    void SetLineSearch( const bool ls )
    {
        line_search = ls;
    }

    double GetMaxIter() const
    {
        return this->max_iter;
    }

#ifdef MFEM_USE_MPI
    NewtonLineSearch( MPI_Comm comm_ ) : NewtonSolver( comm_ )
    {
    }
#endif

    virtual double ComputeScalingFactor( const mfem::Vector& x, const mfem::Vector& b ) const;
    virtual void SetOperator( const mfem::Operator& op );
    virtual void Mult( const mfem::Vector& b, mfem::Vector& x ) const;
};

void SetLambdaToIntegrators( const mfem::Operator*, const double l );

class Crisfield : public mfem::IterativeSolver
{
protected:
    double InnerProduct( const mfem::Vector& a, const double la, const mfem::Vector& b, const double lb ) const;

public:
    Crisfield()
    {
        converged = true;
    }

#ifdef MFEM_USE_MPI
    Crisfield( MPI_Comm comm_ ) : mfem::IterativeSolver( comm_ )
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

    void SetDelta( const double l )
    {
        L = l;
        max_delta = l * 1e2;
        min_delta = l * 1e-3;
    }

    void SetPhi( const double p )
    {
        phi = p;
    }

    void SetMaxStep( const int step )
    {
        max_steps = step;
    }

    void SetMaxDelta( const double delta )
    {
        max_delta = delta;
    }

    void SetMinDelta( const double delta )
    {
        min_delta = delta;
    }

    void SetDataCollection( mfem::DataCollection* dc )
    {
        data = dc;
    }

    virtual bool updateStep( const mfem::Vector& delta_u_bar,
                             const mfem::Vector& delta_u_t,
                             const mfem::Vector& Delta_u,
                             const double Delta_lambda,
                             const int it,
                             const int step,
                             mfem::Vector& delta_u,
                             double& delta_lambda ) const;

protected:
    mutable mfem::Vector r, Delta_u, delta_u, u_cur, q, delta_u_bar, delta_u_t, Delta_u_prev;
    mutable mfem::Operator* grad;

    mutable double lambda, Delta_lambda, delta_lambda, Delta_lambda_prev, max_delta{ 1. }, min_delta{ 1. };

    mutable double L{ 1 };
    mutable double phi{ 1 };

    int max_steps{ 100 };
    mutable mfem::DataCollection* data{ nullptr };
};

class ArcLengthLinearize : public Crisfield
{
public:
    ArcLengthLinearize() : Crisfield()
    {
    }

#ifdef MFEM_USE_MPI
    ArcLengthLinearize( MPI_Comm comm_ ) : Crisfield( comm_ )
    {
    }
#endif

    virtual bool updateStep( const mfem::Vector& delta_u_bar,
                             const mfem::Vector& delta_u_t,
                             const mfem::Vector& Delta_u,
                             const double Delta_lambda,
                             const int it,
                             const int step,
                             mfem::Vector& delta_u,
                             double& delta_lambda ) const;
};

class MultiNewtonAdaptive : public NewtonLineSearch
{
public:
    void SetMaxStep( const int step )
    {
        max_steps = step;
    }

    void SetDelta( const double delta )
    {
        delta_lambda = delta;
    }
    MultiNewtonAdaptive() : NewtonLineSearch()
    {
    }

#ifdef MFEM_USE_MPI
    MultiNewtonAdaptive( MPI_Comm comm_ ) : NewtonLineSearch( comm_ )
    {
    }
#endif

    /// Solve the nonlinear system with right-hand side @a b.
    /** If `b.Size() != Height()`, then @a b is assumed to be zero. */
    virtual void Mult( const mfem::Vector& b, mfem::Vector& x ) const;
    virtual void SetOperator( const mfem::Operator& op );

    void SetDataCollection( mfem::DataCollection* dc )
    {
        data = dc;
    }

protected:
    int max_steps{ 100 };
    mutable double delta_lambda{ 1. };
    mutable mfem::IterativeSolver* prec{ nullptr };
    mutable mfem::Vector u_cur;
    const mfem::Operator* oper{ nullptr };
    mutable mfem::DataCollection* data{ nullptr };
};
} // namespace plugin