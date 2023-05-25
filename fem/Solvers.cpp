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

void NewtonLineSearch::Mult( const mfem::Vector& b, mfem::Vector& x ) const
{
    using namespace mfem;
    MFEM_ASSERT( oper != NULL, "the Operator is not set (use SetOperator)." );
    MFEM_ASSERT( prec != NULL, "the Solver is not set (use SetSolver)." );

    int it;
    double norm0, norm, norm_goal;
    const bool have_b = ( b.Size() == Height() );

    if ( !iterative_mode )
    {
        x = 0.0;
    }

    ProcessNewState( x );

    oper->Mult( x, r );
    if ( have_b )
    {
        r -= b;
    }

    norm0 = norm = Norm( r );
    if ( print_options.first_and_last && !print_options.iterations )
    {
        mfem::out << "Newton iteration " << std::setw( 2 ) << 0 << " : ||r|| = " << norm << "...\n";
    }
    norm_goal = std::max( rel_tol * norm, abs_tol );

    prec->iterative_mode = false;

    // x_{i+1} = x_i - [DF(x_i)]^{-1} [F(x_i)-b]
    for ( it = 0; true; it++ )
    {
        if ( print_options.iterations )
        {
            mfem::out << "Newton iteration " << std::setw( 2 ) << it << " : ||r|| = " << norm;
            if ( it > 0 )
            {
                mfem::out << ", ||r||/||r_0|| = " << norm / norm0;
            }
            mfem::out << '\n';
        }
        Monitor( it, norm, r, x );

        if ( !mfem::IsFinite( norm ) )
        {
            converged = false;
            break;
        }
        if ( norm <= norm_goal )
        {
            converged = true;
            break;
        }

        if ( it >= max_iter )
        {
            converged = false;
            break;
        }

        grad = &oper->GetGradient( x );
        {
            std::ofstream myfile;
            myfile.open( "mat1.txt" );
            grad->PrintMatlab( myfile );
            myfile.close();
        }

        prec->SetOperator( *grad );

        if ( lin_rtol_type )
        {
            AdaptiveLinRtolPreSolve( x, it, norm );
        }

        prec->Mult( r, c ); // c = [DF(x_i)]^{-1} [F(x_i)-b]

        if ( lin_rtol_type )
        {
            AdaptiveLinRtolPostSolve( c, r, it, norm );
        }

        const double c_scale = ComputeScalingFactor( x, b );
        if ( c_scale == 0.0 )
        {
            converged = false;
            break;
        }
        add( x, -c_scale, c, x );

        ProcessNewState( x );

        oper->Mult( x, r );
        if ( have_b )
        {
            r -= b;
        }
        norm = Norm( r );
    }

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
    return Dot( a, b ) + la * lb * phi;
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
    double norm{ 0 }, norm_goal{ 0 }, normPrev{ 0 }, normPrevPrev{ 0 };
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
            L *= std::pow( .9 * max_iter / final_iter, .6 );
            L = std::min( L, max_delta );
        }

        MFEM_VERIFY( L > min_delta, "Required step size is smaller than the minimal bound." );

        if ( step )
            phi = std::abs( Delta_u_prev.Normlinf() / Delta_lambda_prev );

        util::mfemOut( "L: ", L, ", phi: ", phi, "\n", util::Color::RESET );

        delta_u = 0.;
        Delta_u = 0.;
        delta_lambda = 0.;
        Delta_lambda = 0.;
        int it = 0;

        // mfem::out << std::setprecision( 16 ) << "time: " << lambda << std::endl;
        for ( ; true; it++ )
        {
            if ( it >= max_iter )
            {
                converged = false;
                break;
            }

            // TODO: do not understand.
            add( *u, Delta_u, u_cur );

            // compute q
            SetLambdaToIntegrators( oper, .0001 + lambda + Delta_lambda );
            oper->Mult( u_cur, q );
            SetLambdaToIntegrators( oper, lambda + Delta_lambda );
            oper->Mult( u_cur, r );
            q -= r;
            q.Neg();
            q *= 10000;
            if ( have_b )
            {
                r -= b;
            }
            r.Neg();
            grad = &oper->GetGradient( u_cur );
            // std::ofstream myfile;
            // myfile.open( "mat.txt" );
            // grad->PrintMatlab( myfile );
            // myfile.close();
            prec->SetOperator( *grad );

            prec->Mult( q, delta_u_t );
            // mfem::OperatorHandle gradHandle( grad, false );
            // PetscBool isSymmetric;
            // MatIsSymmetric( ( mfem::petsc::Mat )( *gradHandle.As<mfem::PetscParMatrix>() ), 0., &isSymmetric );
            prec->Mult( r, delta_u_bar );

            if ( !updateStep( delta_u_bar, delta_u_t, it, step ) )
            {
                converged = false;
                break;
            }

            // update
            Delta_u += delta_u;
            Delta_lambda += delta_lambda;

            // convergence check
            normPrevPrev = normPrev;
            normPrev = norm;
            norm = std::sqrt( InnerProduct( delta_u, delta_lambda, delta_u, delta_lambda ) );
            if ( it == 0 )
            {
                norm_goal = std::max( rel_tol * norm, abs_tol );
            }
            if ( !mfem::IsFinite( norm ) )
            {
                converged = false;
                break;
            }
            Monitor( it, norm, r, *u );

            if ( norm <= norm_goal )
            {
                converged = true;
                break;
            }

            // filter out slow convergence case
            if ( it >= std::max( 2, max_iter / 2 ) && util::ConvergenceRate( norm, normPrev, normPrevPrev ) < 1.2 )
            {
                mfem::out << "Convergence rate "
                          << util::ConvergenceRate( norm, normPrev, normPrevPrev ) << " is too small!\n";
                converged = false;
                break;
            }

            // delta_u.Print();
            // if(it==0){

            //     std::ofstream myfile;
            //     myfile.open ("mat.txt");
            //     grad->PrintMatlab(myfile, 16, 16);
            //     myfile.close();
            // }
        }
        // std::exit(0);
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
            L_prev = L;
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
                if ( step % 2 == 0 )
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

bool Crisfield::updateStep( const mfem::Vector& delta_u_bar, const mfem::Vector& delta_u_t, const int it, const int step ) const
{
    const double delta_u_bar_dot_delta_u_t = Dot( delta_u_bar, delta_u_t );
    const double delta_u_bar_dot_delta_u_bar = Dot( delta_u_bar, delta_u_bar );
    const double delta_u_t_dot_delta_u_t = Dot( delta_u_t, delta_u_t );
    const double Delta_u_dot_delta_u_t = Dot( Delta_u, delta_u_t );
    const double Delta_u_dot_delta_u_bar = Dot( Delta_u, delta_u_bar );
    const double Delta_u_dot_Delta_u = Dot( Delta_u, Delta_u );

    double ds = 1.;
    const double as = std::pow( delta_u_bar_dot_delta_u_t, 2 ) - phi * delta_u_bar_dot_delta_u_bar -
                      delta_u_bar_dot_delta_u_bar * delta_u_t_dot_delta_u_t;
    const double bs = 2 * ( Delta_lambda * phi + Delta_u_dot_delta_u_t ) * delta_u_bar_dot_delta_u_t -
                      2 * Delta_u_dot_delta_u_bar * ( phi + delta_u_t_dot_delta_u_t );
    const double cs = ( it == 0 ) ? std::pow( Delta_lambda * phi + Delta_u_dot_delta_u_t, 2 ) +
                                        ( L * L - Delta_lambda * Delta_lambda * phi - Delta_u_dot_Delta_u ) * ( phi + delta_u_t_dot_delta_u_t )
                                  : std::pow( Delta_lambda * phi + Delta_u_dot_delta_u_t, 2 );

    auto func = [as, bs, cs]( const double ds ) { return as * ds * ds + bs * ds + cs; };
    if ( func( ds ) < 0 )
    {
        util::mfemOut( util::Color::YELLOW, "Complex root detected, adaptive step size (ds) is activated!\n", util::Color::RESET );
        const double beta1 = ( -bs + std::sqrt( bs * bs - 4 * as * cs ) ) / ( 2 * as );
        const double beta2 = ( -bs - std::sqrt( bs * bs - 4 * as * cs ) ) / ( 2 * as );
        util::mfemOut( util::Color::YELLOW, "beta1: ", beta1, ", beta2: ", beta2, '\n', util::Color::RESET );
        const double xi = beta2 - beta1;
        ds = std::min( beta2 - xi * .05, ds );
        util::mfemOut( util::Color::YELLOW, "ds=: ", ds, util::Color::RESET );
        if ( ds < .1 )
        {
            util::mfemOut( util::Color::YELLOW, ", which is smaller than ds_min, restart!\n ", util::Color::RESET );
            return false;
        }
        util::mfemOut( "func(ds)= ", func( ds ), '\n' );
    }
    const double a0 = delta_u_t_dot_delta_u_t + phi;
    const double b0 = 2 * Delta_u_dot_delta_u_t + 2 * phi * Delta_lambda;
    const double b1 = 2 * delta_u_bar_dot_delta_u_t;
    const double c0 = Delta_u_dot_Delta_u + phi * Delta_lambda * Delta_lambda - L * L;
    const double c1 = 2 * Delta_u_dot_delta_u_bar;
    const double c2 = delta_u_bar_dot_delta_u_bar;

    const double a = a0;
    const double b = b0 + b1 * ds;
    const double c = c0 + c1 * ds + c2 * ds * ds;
    const double delta_lambda1 = ( -1. * b + std::sqrt( b * b - 4 * a * c ) ) / ( 2. * a );
    const double delta_lambda2 = ( -1. * b - std::sqrt( b * b - 4 * a * c ) ) / ( 2. * a );
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
    return true;
}

bool ArcLengthLinearize::updateStep( const mfem::Vector& delta_u_bar, const mfem::Vector& delta_u_t, const int it, const int step ) const
{
    const double frac = 0.1;
    const double tol = 1e-9;
    // predictor
    if ( it == 0 )
    {
        if ( Delta_u_prev.Norml2() < tol && std::abs( Delta_lambda_prev ) < tol )
        {
            Delta_u = delta_u_t;
            Delta_u *= 1. / delta_u_t.Norml2() * L * frac;
            // Delta_lambda = L * frac / std::sqrt( phi );
            mfem::out << "predict!\n";
        }
        else
        {
            Delta_u = Delta_u_prev;
            Delta_u *= L / L_prev * frac;
            Delta_lambda = Delta_lambda_prev * L / L_prev * frac;
        }
    }
    const double Delta_u_dot_delta_u_t = Dot( Delta_u, delta_u_t );
    const double Delta_u_dot_delta_u_bar = Dot( Delta_u, delta_u_bar );
    const double Delta_u_dot_Delta_u = Dot( Delta_u, Delta_u );

    delta_lambda = ( L * L - Delta_u_dot_Delta_u - phi * Delta_lambda * Delta_lambda - 2 * Delta_u_dot_delta_u_bar ) /
                   ( 2 * Delta_u_dot_delta_u_t + 2 * phi * Delta_lambda );

    add( 1., delta_u_bar, delta_lambda, delta_u_t, delta_u );
    return true;
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

    int count = 1;
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
        // MFEM_VERIFY( delta_lambda > min_delta, "Required step size is smaller than the minimal bound." );

        util::mfemOut( "L: ", delta_lambda, "\n", util::Color::RESET );
        delta_lambda = std::min( delta_lambda, 1. - cur_lambda );
        SetLambdaToIntegrators( oper, delta_lambda + cur_lambda );

        plugin::NewtonLineSearch::Mult( b, *u );
        if ( GetConverged() )
        {
            cur_lambda += delta_lambda;
            u_cur = *u;

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