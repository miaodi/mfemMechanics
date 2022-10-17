#include "Solvers.h"
#include "FEMPlugin.h"
#include "PrettyPrint.h"
#include "util.h"
#include <deque>
#include <slepc.h>
#include <tuple>

namespace plugin
{
void NewtonLineSearch::SetOperator( const mfem::Operator& op )
{
    mfem::NewtonSolver::SetOperator( op );
    u_cur.SetSize( op.Width() );
}

double NewtonLineSearch::ComputeScalingFactor( const mfem::Vector& x, const mfem::Vector& b ) const
{
    if ( !line_search )
        return 1.;

    // initialize
    const bool have_b = ( b.Size() == Height() );
    double sL, sR, s;
    double etaL = 0., etaR = 1., eta = 1., ratio = 1.;
    auto CalcS = [&b, &x, have_b, this]( const double eta )
    {
        add( x, -eta, c, this->u_cur );
        this->oper->Mult( this->u_cur, this->r );
        if ( have_b )
        {
            this->r -= b;
        }
        return this->r * this->c;
    };
    oper->Mult( x, r );
    if ( have_b )
    {
        r -= b;
    }
    sL = CalcS( etaL );
    const double s0 = sL;

    sR = CalcS( etaR );

    // first find the right span
    while ( sL * sR > 0 && etaR < max_eta )
    {
        etaR *= eta_coef;

        sR = CalcS( etaR );
    }

    int it = 0;
    while ( sL * sR < 0 && ratio > tol && it++ < max_line_search_iter && eta > min_eta )
    {
        eta = .5 * ( etaL + etaR );
        s = CalcS( eta );
        ratio = fabs( s / s0 );
        if ( s * sL > 0 )
        {
            sL = s;
            etaL = eta;
        }
        else
        {
            sR = s;
            etaR = eta;
        }
    }
    if ( ratio > tol || sL * sR > 0 )
    {
        eta = 1.;
    }
    std::cout << "eta: " << eta << std::endl;
    return eta;
}

void Crisfield::SetOperator( const mfem::Operator& op )
{
    oper = &op;
    height = op.Height();
    width = op.Width();
    MFEM_ASSERT( height == width, "square Operator is required." );

    r.SetSize( width );
    Delta_u.SetSize( width );
    delta_u.SetSize( width );
    u_cur.SetSize( width );
    q.SetSize( width );
    delta_u_bar.SetSize( width );
    delta_u_t.SetSize( width );
    delta_u_t_p_Delta_u.SetSize( width );
    Delta_u_prev.SetSize( width );
}

void SetLambdaToIntegrators( const mfem::Operator* oper, const double l )
{
    if ( auto nonlinearform = dynamic_cast<mfem::NonlinearForm*>( const_cast<mfem::Operator*>( oper ) ) )
    {
        auto bfnfi = nonlinearform->GetBdrFaceIntegrators();
        for ( int i = 0; i < bfnfi.Size(); i++ )
        {
            if ( auto with_lambda = dynamic_cast<NonlinearFormIntegratorLambda*>( bfnfi[i] ) )
            {
                with_lambda->SetLambda( l );
            }
        }

        auto& dnfi = *nonlinearform->GetDNFI();
        for ( int i = 0; i < dnfi.Size(); i++ )
        {
            if ( auto with_lambda = dynamic_cast<NonlinearFormIntegratorLambda*>( dnfi[i] ) )
            {
                with_lambda->SetLambda( l );
            }
        }
    }
}

double Crisfield::InnerProduct( const mfem::Vector& a, const double la, const mfem::Vector& b, const double lb ) const
{
    return Dot( a, b ) + la * lb * phi * phi;
}

void Crisfield::Mult( const mfem::Vector& b, mfem::Vector& x ) const
{
    MFEM_ASSERT( oper != NULL, "the Operator is not set (use SetOperator)." );
    MFEM_ASSERT( prec != NULL, "the Solver is not set (use SetSolver)." );
    const double goldenRatio = ( 1. + std::sqrt( 5 ) ) / 2;
    mfem::Vector* u;
    if ( auto par_grid_x = dynamic_cast<mfem::ParGridFunction*>( &x ) )
    {
        u = new mfem::Vector( par_grid_x->ParFESpace()->GetTrueVSize() );
        par_grid_x->ParallelProject( *u );
    }
    else
    {
        u = &x;
    }
    // mfem::PetscSolver* petscPrec = nullptr;
    // if ( dynamic_cast<mfem::PetscSolver*>( prec ) )
    // {
    //     petscPrec = dynamic_cast<mfem::PetscSolver*>( prec );
    // }
    int step = 0;
    double norm0{ 0 }, norm{ 0 }, norm_goal{ 0 };
    const bool have_b = ( b.Size() == Height() );
    lambda = 0.;

    int count = 1;

    if ( !iterative_mode )
    {
        *u = 0.0;
    }
    Delta_u_prev = 0.;
    Delta_lambda_prev = 0.;

    ProcessNewState( x );
    // q.Print();
    for ( ; true; )
    {
        if ( lambda >= 1. )
        {
            break;
        }
        if ( step >= max_steps )
        {
            converged = false;
            break;
        }
        // initialize
        if ( converged == false )
        {
            L /= goldenRatio;
        }
        else if ( step )
        {
            L *= std::pow( .7 * max_iter / final_iter, .6 );
            L = std::min( L, max_delta );
        }

        MFEM_VERIFY( L > min_delta, "Required step size is smaller than the minimal bound." );
        delta_u = 0.;
        Delta_u = 0.;
        delta_lambda = 0.;
        Delta_lambda = 0.;
        int it = 0;

        for ( ; true; it++ )
        {
            if ( it >= max_iter )
            {
                converged = false;
                break;
            }
            // initialize newton iteration
            double delta_lambda1, delta_lambda2;

            // TODO: do not understand.
            add( *u, Delta_u, u_cur );

            // compute q
            SetLambdaToIntegrators( oper, 1. + lambda + Delta_lambda );

            oper->Mult( u_cur, q );

            SetLambdaToIntegrators( oper, lambda + Delta_lambda );

            oper->Mult( u_cur, r );
            q -= r;
            q.Neg();
            if ( have_b )
            {
                r -= b;
            }
            r.Neg();
            grad = &oper->GetGradient( u_cur );
            prec->SetOperator( *grad );

            prec->Mult( q, delta_u_t );
            // mfem::OperatorHandle gradHandle( grad, false );
            // PetscBool isSymmetric;
            // MatIsSymmetric( ( mfem::petsc::Mat )( *gradHandle.As<mfem::PetscParMatrix>() ), 0., &isSymmetric );

            prec->Mult( r, delta_u_bar );

            add( Delta_u, delta_u_bar, delta_u_t_p_Delta_u );

            const double a0 = Dot( delta_u_t, delta_u_t ) + phi * phi;
            const double b0 = 2 * Dot( Delta_u, delta_u_t ) + 2 * phi * phi * Delta_lambda;
            const double b1 = 2 * Dot( delta_u_bar, delta_u_t );
            const double c0 = Dot( Delta_u, Delta_u ) + phi * phi * Delta_lambda * Delta_lambda - L * L;
            const double c1 = 2 * Dot( Delta_u, delta_u_bar );
            const double c2 = Dot( delta_u_bar, delta_u_bar );

            const double as = b1 * b1 - 4 * a0 * c2;
            const double bs = 2 * b0 * b1 - 4 * a0 * c1;
            const double cs = b0 * b0 - 4 * a0 * c0;

            double ds = 1.;
            auto func = [as, bs, cs]( const double ds ) { return as * ds * ds + bs * ds + cs; };
            if ( func( ds ) <= 0 )
            {
                util::mfemOut( util::Color::YELLOW, "Complex root detected, adaptive step size (ds) is activated!\n",
                               util::Color::RESET );
                const double beta1 = ( -bs - std::sqrt( bs * bs - 4 * as * cs ) ) / ( 2 * as );
                const double beta2 = ( -bs + std::sqrt( bs * bs - 4 * as * cs ) ) / ( 2 * as );
                util::mfemOut( util::Color::YELLOW, "beta1: ", beta1, ", beta2: ", beta2, '\n', util::Color::RESET );
                ds = std::min( beta1 - .001, ds );
                util::mfemOut( util::Color::YELLOW, "ds=: ", ds, util::Color::RESET );
                if ( ds < .2 )
                {
                    util::mfemOut( util::Color::YELLOW, ", which is smaller than ds_min, restart!\n ", util::Color::RESET );
                    converged = false;
                    break;
                }
                util::mfemOut( "func(ds)= ", func( ds ), '\n' );
            }

            const double a = a0;
            const double b = b0 + b1 * ds;
            const double c = c0 + c1 * ds + c2 * ds * ds;

            delta_lambda1 = ( -1. * b + std::sqrt( b * b - 4 * a * c ) ) / ( 2. * a );
            delta_lambda2 = ( -1. * b - std::sqrt( b * b - 4 * a * c ) ) / ( 2. * a );

            util::mfemOut( "delta_lambda1: ", delta_lambda1, ", delta_lambda2: ", delta_lambda2, "\n", util::Color::RESET );

            if ( it == 0 )
            {
                if ( step == 0 )
                {
                    delta_lambda = delta_lambda1;
                }
                else
                {
                    if ( InnerProduct( delta_u_t, 1, Delta_u_prev, Delta_lambda_prev ) > 0 )
                    {
                        delta_lambda = delta_lambda1;
                    }
                    else
                    {
                        delta_lambda = delta_lambda2;
                    }
                }
            }
            else
            {
                const double t = InnerProduct( Delta_u, Delta_lambda, delta_u_t, 1 );
                if ( t * delta_lambda1 > t * delta_lambda2 )
                {
                    delta_lambda = delta_lambda1;
                }
                else
                {
                    delta_lambda = delta_lambda2;
                }
            }
            // mfem::out << "delta_lambda: " << delta_lambda << '\n';

            add( ds, delta_u_bar, delta_lambda, delta_u_t, delta_u );

            if ( it == 0 )
            {
                norm0 = norm = std::sqrt( InnerProduct( delta_u, delta_lambda, delta_u, delta_lambda ) );
                if ( !mfem::IsFinite( norm0 ) )
                {
                    converged = false;
                    break;
                }
                if ( print_options.first_and_last && !print_options.iterations )
                {
                    mfem::out << "Newton iteration " << std::setw( 2 ) << 0 << " : ||r|| = " << norm << "...\n";
                }
                norm_goal = std::max( rel_tol * norm, abs_tol );
            }
            else
            {
                norm = std::sqrt( InnerProduct( delta_u, delta_lambda, delta_u, delta_lambda ) );
                if ( !mfem::IsFinite( norm ) )
                {
                    converged = false;
                    break;
                }
                if ( print_options.iterations )
                {
                    mfem::out << "Newton iteration " << std::setw( 2 ) << it << " : ||r|| = " << norm;
                    if ( it > 0 )
                    {
                        mfem::out << ", ||r||/||r_0|| = " << norm / norm0;
                    }
                    mfem::out << '\n';
                }
            }
            Monitor( it, norm, r, *u );

            if ( norm <= norm_goal )
            {
                converged = true;
                break;
            }

            // update

            Delta_u += delta_u;
            Delta_lambda += delta_lambda;
        }
        // update
        if ( converged )
        {
            if ( Delta_lambda < 0 )
            {
                util::mfemOut( "alert !!! buckled!!\n" );
            }
            lambda += Delta_lambda;
            *u += Delta_u;
            Delta_lambda_prev = Delta_lambda;
            Delta_u_prev = Delta_u;
            step++;

            final_iter = it;
            final_norm = norm;
            if ( print_options.summary || ( !converged && print_options.warnings ) || print_options.first_and_last )
            {
                mfem::out << "Newton: Number of iterations: " << final_iter << '\n'
                          << "   ||r|| = " << final_norm << '\n';
            }
            if ( !converged && ( print_options.summary || print_options.warnings ) )
            {
                mfem::out << "Newton: No convergence!\n";
            }

            if ( data )
            {
                if ( step % 5 == 0 )
                {
                    if ( auto par_grid_x = dynamic_cast<mfem::ParGridFunction*>( &x ) )
                    {
                        par_grid_x->Distribute( *u );
                    }
                    data->SetCycle( count );
                    data->SetTime( count++ );
                    data->Save();
                }
            }
        }
        util::mfemOut( util::ProgressBar( lambda, converged ), '\n' );
    }
    if ( auto par_grid_x = dynamic_cast<mfem::ParGridFunction*>( &x ) )
    {
        par_grid_x->Distribute( *u );
        delete u;
    }
}

MultiNewtonAdaptive::MultiNewtonAdaptive()
{
}

void MultiNewtonAdaptive::SetOperator( const mfem::Operator& op )
{
    mfem::NewtonSolver::SetOperator( op );
    oper = &op;
    height = op.Height();
    width = op.Width();
    MFEM_ASSERT( height == width, "square Operator is required." );

    u_cur.SetSize( width );
}

void MultiNewtonAdaptive::Mult( const mfem::Vector& b, mfem::Vector& x ) const
{
    double cur_lambda = 0.;
    int step = 0;

    mfem::Vector* u;
    if ( auto par_grid_x = dynamic_cast<mfem::ParGridFunction*>( &x ) )
    {
        u = new mfem::Vector( par_grid_x->ParFESpace()->GetTrueVSize() );
        par_grid_x->ParallelProject( *u );
    }
    else
    {
        u = &x;
    }

    int count = 1;
    u_cur = *u;
    for ( ; true; step++ )
    {
        if ( step == max_steps )
        {
            break;
        }
        if ( cur_lambda >= 1. )
        {
            break;
        }
        if ( step )
        {
            if ( !GetConverged() )
                delta_lambda /= 2;
            else
                delta_lambda *= std::pow( .7 * this->max_iter / GetNumIterations(), .6 );
        }
        delta_lambda = std::min( delta_lambda, 1. - cur_lambda );
        SetLambdaToIntegrators( oper, delta_lambda + cur_lambda );

        mfem::NewtonSolver::Mult( b, *u );
        if ( data )
        {
            if ( auto par_grid_x = dynamic_cast<mfem::ParGridFunction*>( &x ) )
            {
                par_grid_x->Distribute( *u );
            }
            data->SetCycle( count );
            data->SetTime( count++ );
            data->Save();
        }
        if ( GetConverged() )
        {
            cur_lambda += delta_lambda;
            u_cur = *u;
        }
        else
        {
            *u = u_cur;
        }

#ifdef MFEM_USE_MPI
        int rank = 0;
        if ( this->GetComm() != MPI_COMM_NULL )
            MPI_Comm_rank( this->GetComm(), &rank );
        if ( rank == 0 )
        {
            mfem::out << util::ProgressBar( cur_lambda ) << '\n';
        }
#else
        mfem::out << util::ProgressBar( cur_lambda ) << '\n';
#endif
    }
    if ( auto par_grid_x = dynamic_cast<mfem::ParGridFunction*>( &x ) )
    {
        par_grid_x->Distribute( *u );
        delete u;
    }
}
} // namespace plugin