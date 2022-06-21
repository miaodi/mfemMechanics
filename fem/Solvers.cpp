#include "Solvers.h"
#include "FEMPlugin.h"
#include "PrettyPrint.h"

namespace plugin
{
void NewtonLineSearch::SetOperator( const mfem::Operator& op )
{
    mfem::NewtonSolver::SetOperator( op );
    x_cur.SetSize( op.Width() );
}

double NewtonLineSearch::ComputeScalingFactor( const mfem::Vector& x, const mfem::Vector& b ) const
{
    if ( !line_search )
        return 1.;

    // initialize
    const bool have_b = ( b.Size() == Height() );
    double sL, sR, s;
    double etaL = 0., etaR = 1., eta = 1., ratio = 1.;
    auto CalcS = [&b, &x, have_b, this]( const double eta ) {
        add( x, -eta, c, this->x_cur );
        this->oper->Mult( this->x_cur, this->r );
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
    x_cur.SetSize( width );
    q.SetSize( width );
    delta_u_bar.SetSize( width );
    delta_u_t.SetSize( width );
    delta_u_t_p_Delta_x.SetSize( width );
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

    int step = 0;
    double norm0{ 0 }, norm{ 0 }, norm_goal{ 0 };
    const bool have_b = ( b.Size() == Height() );
    lambda = 0.;

    if ( !iterative_mode )
    {
        x = 0.0;
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
            L /= 2;
        }
        else if ( step )
        {
            L *= std::pow( .7 * max_iter / final_iter, .6 );
            L = std::min( L, max_delta );
        }

        MFEM_ASSERT( L > min_delta, "Required step size is smaller than the minimal bound." );
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
            add( x, Delta_u, x_cur );

            // compute q
            SetLambdaToIntegrators( oper, 1. + lambda + Delta_lambda );

            oper->Mult( x_cur, q );

            SetLambdaToIntegrators( oper, lambda + Delta_lambda );

            oper->Mult( x_cur, r );
            q -= r;
            q.Neg();
            if ( have_b )
            {
                r -= b;
            }
            r.Neg();
            grad = &oper->GetGradient( x_cur );
            prec->SetOperator( *grad );

            prec->Mult( q, delta_u_t );
            prec->Mult( r, delta_u_bar );

            add( Delta_u, delta_u_bar, delta_u_t_p_Delta_x );

            const double a1 = Dot( delta_u_t, delta_u_t ) + phi * phi;
            const double a2 = 2 * Dot( delta_u_t_p_Delta_x, delta_u_t ) + 2 * phi * phi * Delta_lambda;
            const double a3 = Dot( delta_u_t_p_Delta_x, delta_u_t_p_Delta_x ) + phi * phi * Delta_lambda * Delta_lambda - L * L;

            MFEM_ASSERT( a2 * a2 - 4 * a1 * a3 >= 0., "complex frequency need cut step size!\n" );

            delta_lambda1 = ( -1. * a2 + std::sqrt( a2 * a2 - 4 * a1 * a3 ) ) / ( 2. * a1 );
            delta_lambda2 = ( -1. * a2 - std::sqrt( a2 * a2 - 4 * a1 * a3 ) ) / ( 2. * a1 );
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

            add( delta_u_bar, delta_lambda, delta_u_t, delta_u );

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
            Monitor( it, norm, r, x );

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
                mfem::out << "alert !!! buckled!!\n";
            }
            lambda += Delta_lambda;
            x += Delta_u;
            Delta_lambda_prev = Delta_lambda;
            Delta_u_prev = Delta_u;
            step++;
        }

#ifdef MFEM_USE_MPI
        int rank = 0;
        if ( this->GetComm() != MPI_COMM_NULL )
            MPI_Comm_rank( this->GetComm(), &rank );
        if ( rank == 0 )
        {
            mfem::out << util::ProgressBar( lambda ) << '\n';
        }
#else
        mfem::out << util::ProgressBar( lambda ) << '\n';

#endif

        final_iter = it;
        final_norm = norm;
        if ( print_options.summary || ( !converged && print_options.warnings ) || print_options.first_and_last )
        {
            mfem::out << "Newton: Number of iterations: " << final_iter << '\n' << "   ||r|| = " << final_norm << '\n';
        }
        if ( !converged && ( print_options.summary || print_options.warnings ) )
        {
            mfem::out << "Newton: No convergence!\n";
        }
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

    x_cur.SetSize( width );
}

void MultiNewtonAdaptive::Mult( const mfem::Vector& b, mfem::Vector& x ) const
{
    double cur_lambda = 0.;
    int step = 0;
    x_cur = x;
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

        mfem::NewtonSolver::Mult( b, x );
        if ( GetConverged() )
        {
            cur_lambda += delta_lambda;
            x_cur = x;
        }
        else
        {
            x = x_cur;
        }

#ifdef MFEM_USE_MPI
        int rank;
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
}
} // namespace plugin