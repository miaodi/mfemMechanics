#include "Solvers.h"
#include "FEMPlugin.h"
#include "PrettyPrint.h"
#include "util.h"
#include <Eigen/Dense>
#include <deque>
#include <mfem.hpp>
#include <slepc.h>
#include <tuple>

namespace plugin
{

void IterAuxilliary::RegisterToIntegrators( const mfem::Operator* oper ) const
{
    if ( auto nonlinearform = dynamic_cast<mfem::NonlinearForm*>( const_cast<mfem::Operator*>( oper ) ) )
    {
        auto bfnfi = nonlinearform->GetBdrFaceIntegrators();
        for ( int i = 0; i < bfnfi.Size(); i++ )
        {
            if ( auto with_lambda = dynamic_cast<NonlinearFormIntegratorLambda*>( bfnfi[i] ) )
            {
                with_lambda->SetIterAux( this );
            }
        }

        auto& dnfi = *nonlinearform->GetDNFI();
        for ( int i = 0; i < dnfi.Size(); i++ )
        {
            if ( auto with_lambda = dynamic_cast<NonlinearFormIntegratorLambda*>( dnfi[i] ) )
            {
                with_lambda->SetIterAux( this );
            }
        }

        auto& fnfi = nonlinearform->GetInteriorFaceIntegrators();
        for ( int i = 0; i < fnfi.Size(); i++ )
        {
            if ( auto with_lambda = dynamic_cast<NonlinearFormIntegratorLambda*>( fnfi[i] ) )
            {
                with_lambda->SetIterAux( this );
            }
        }
    }

    if ( auto nonlinearform = dynamic_cast<mfem::BlockNonlinearForm*>( const_cast<mfem::Operator*>( oper ) ) )
    {
        auto bfnfi = nonlinearform->GetBdrFaceIntegrators();
        for ( int i = 0; i < bfnfi.Size(); i++ )
        {
            if ( auto with_lambda = dynamic_cast<BlockNonlinearFormIntegratorLambda*>( bfnfi[i] ) )
            {
                with_lambda->SetIterAux( this );
            }
        }

        auto& dnfi = *nonlinearform->GetDNFI();
        for ( int i = 0; i < dnfi.Size(); i++ )
        {
            if ( auto with_lambda = dynamic_cast<BlockNonlinearFormIntegratorLambda*>( dnfi[i] ) )
            {
                with_lambda->SetIterAux( this );
            }
        }
    }
}

void NewtonLineSearch::SetOperator( const mfem::Operator& op )
{
    mfem::NewtonSolver::SetOperator( op );
    RegisterToIntegrators( this->oper );
    aux_line_search.SetSize( width );
}

int NewtonLineSearch::MyRank() const
{
#ifdef MFEM_USE_MPI
    if ( GetComm() == MPI_COMM_NULL )
    {
        return 0;
    }
    else
    {
        return mfem::Mpi::WorldRank();
    }
#else
    return 0;
#endif
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
        add( x, -eta, c, aux_line_search );
        this->oper->Mult( aux_line_search, this->r );
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

    int iter = 0;
    while ( sL * sR < 0 && ratio > tol && iter++ < max_line_search_iter && eta > min_eta )
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
    norm_goal = std::max( rel_tol * norm, abs_tol );

    prec->iterative_mode = false;

    // x_{i+1} = x_i - [DF(x_i)]^{-1} [F(x_i)-b]
    for ( it = 0; true; it++ )
    {
        if ( MyRank() == 0 )
        {
            mfem::out << "Newton iteration " << std::setw( 2 ) << it << " : ||r|| = " << norm;
            if ( it > 0 )
            {
                mfem::out << ", ||r||/||r_0|| = " << norm / norm0;
            }
            mfem::out << '\n';
        }

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
        // {
        //     std::ofstream myfile;
        //     myfile.open( "mat1.txt" );
        //     grad->PrintMatlab( myfile );
        //     myfile.close();
        // }

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

    if ( !converged && MyRank() == 0 )
    {
        mfem::out << "Newton: No convergence!\n";
    }
}

void NewtonForPhaseField::SetOperator( const mfem::Operator& op )
{
    NewtonLineSearch::SetOperator( op );
    blockOper = static_cast<mfem::BlockNonlinearForm*>( const_cast<mfem::Operator*>( &op ) );
    block_trueOffsets = blockOper->GetBlockTrueOffsets();

    r_u = mfem::Vector( r.GetData() + block_trueOffsets[0], block_trueOffsets[1] - block_trueOffsets[0] );
    c_u = mfem::Vector( c.GetData() + block_trueOffsets[0], block_trueOffsets[1] - block_trueOffsets[0] );
    r_p = mfem::Vector( r.GetData() + block_trueOffsets[1], block_trueOffsets[2] - block_trueOffsets[1] );
    c_p = mfem::Vector( c.GetData() + block_trueOffsets[1], block_trueOffsets[2] - block_trueOffsets[1] );
}

void NewtonForPhaseField::Mult( const mfem::Vector& b, mfem::Vector& x ) const
{
    using namespace mfem;
    MFEM_ASSERT( oper != NULL, "the Operator is not set (use SetOperator)." );
    MFEM_ASSERT( prec != NULL, "the Solver is not set (use SetSolver)." );

    double norm0_u, norm_u, norm_goal_u;
    double norm0_p{ 0 }, norm_p{ 0 }, norm_goal_p{ 100 };
    const bool have_b = ( b.Size() == Height() );

    if ( !iterative_mode )
    {
        x = 0.0;
    }

    mfem::Vector& cur_u = static_cast<mfem::BlockVector&>( x ).GetBlock( 0 );
    mfem::Vector& cur_p = static_cast<mfem::BlockVector&>( x ).GetBlock( 1 );

    ProcessNewState( x );

    blockOper->Mult( x, r );
    if ( have_b )
    {
        r -= b;
    }

    norm0_u = norm_u = Norm( r_u );
    norm_goal_u = std::max( rel_tol * norm_u, abs_tol );

    prec->iterative_mode = false;

    // x_{i+1} = x_i - [DF(x_i)]^{-1} [F(x_i)-b]
    for ( it = 0; true; it++ )
    {
        // r_u.Print();
        if ( MyRank() == 0 )
        {
            mfem::out << "Newton iteration " << std::setw( 2 ) << it << " : ||r_u|| = " << norm_u;
            if ( it > 0 )
            {
                mfem::out << " : ||r_p|| = " << norm_p << ", ||r_u||/||r_u_0|| = " << norm_u / norm0_u
                          << ", ||r_p||/||r_p_0|| = " << norm_p / norm0_p << '\n';
            }
        }

        if ( !mfem::IsFinite( norm_u ) || !mfem::IsFinite( norm_p ) )
        {
            converged = false;
            break;
        }
        if ( norm_u <= norm_goal_u && norm_p <= norm_goal_p )
        {
            converged = true;
            break;
        }

        if ( it >= max_iter )
        {
            converged = true;
            break;
        }
        prec->SetOperator( static_cast<mfem::BlockOperator&>( blockOper->GetGradient( x ) ).GetBlock( 0, 0 ) );
        prec->Mult( r_u, c_u ); // c = [DF(x_i)]^{-1} [F(x_i)-b]
        add( cur_u, -1., c_u, cur_u );
        blockOper->Mult( x, r );

        if ( it == 0 )
        {
            norm0_p = norm_p = Norm( r_p );
            norm_goal_p = std::max( rel_tol * norm_p, abs_tol );
            if ( MyRank() == 0 )
                mfem::out << " : ||r_p|| = " << norm_p << "\n";
        }

        prec->SetOperator( static_cast<mfem::BlockOperator&>( blockOper->GetGradient( x ) ).GetBlock( 1, 1 ) );
        prec->Mult( r_p, c_p ); // c = [DF(x_i)]^{-1} [F(x_i)-b]
        add( cur_p, -1., c_p, cur_p );
        blockOper->Mult( x, r );

        norm_u = Norm( r_u );
        norm_p = Norm( r_p );

        if ( have_b )
        {
            r -= b;
        }
    }

    final_iter = it;

    if ( !converged && MyRank() == 0 )
    {
        mfem::out << "Newton: No convergence!\n";
    }
}

double ALMBase::InnerProduct( const mfem::Vector& a, const double la, const mfem::Vector& b, const double lb ) const
{
    return Dot( a, b ) + la * lb * phi;
}

void ALMBase::ResizeVectors( const int size ) const
{
    r.SetSize( size );
    delta_u.SetSize( size );
    u_cur.SetSize( size );
    q.SetSize( size );
    delta_u_bar.SetSize( size );
    delta_u_t.SetSize( size );
    Delta_u.SetSize( size );
    u_direction_pred.SetSize( size );
}

void ALMBase::InitializeVariables( const mfem::Vector& u ) const
{
    ResizeVectors( u.Size() );
}

void ALMBase::SetOperator( const mfem::Operator& op )
{
    oper = &op;
    height = op.Height();
    width = op.Width();
    MFEM_ASSERT( height == width, "square Operator is required." );
    RegisterToIntegrators( this->oper );
}

void ALMBase::PredictDirection() const
{
    // Eigen::Matrix3d mass;
    // mass.setZero();
    // Eigen::Vector3d sol, rhs;

    if ( solution_buffer.size() <= 1 )
    {
        MFEM_ABORT( "Solution buffer is empty, end the simulation." );
    }
    subtract( solution_buffer[0].u, solution_buffer[1].u, u_direction_pred );
    lambda_direction_pred = solution_buffer[0].lambda - solution_buffer[1].lambda;
    u_direction_pred /= solution_buffer[0].L;
    lambda_direction_pred /= solution_buffer[0].L;
}

void ALMBase::Mult( const mfem::Vector& b, mfem::Vector& x ) const
{
    MFEM_ASSERT( oper != NULL, "the Operator is not set (use SetOperator)." );
    MFEM_ASSERT( prec != NULL, "the Solver is not set (use SetSolver)." );

    const double goldenRatio = ( 1. + std::sqrt( 5 ) ) / 2;
    mfem::Vector* u;
    u = &x;

    solution_buffer.unshift();
    solution_buffer[0].L = L;
    solution_buffer[0].lambda = lambda;
    solution_buffer[0].u = *u;
    converged = false;

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

    // if ( !iterative_mode )
    // {
    //     *u = 0.0;
    // }
    InitializeVariables( x );
    u_direction_pred = 0.;
    lambda_direction_pred = 0.;

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

        // update L
        if ( L_update_func )
        {
            L_update_func( converged, final_iter, lambda, L );
        }
        else
        {
            if ( converged == false )
            {
                L /= goldenRatio;
            }
            else if ( step )
            {
                L *= std::min( 1.2, std::pow( .9 * max_iter / final_iter, .6 ) );
                L = std::min( L, max_delta );
            }
        }

        if ( L < min_delta )
        {
            util::mfemOut( util::Color::YELLOW,
                           "Required step size is smaller than the minimal bound, restart with previous solution.\n",
                           util::Color::RESET );
            if ( solution_buffer.size() <= 1 )
            {
                MFEM_ABORT( "Solution buffer is empty, end the simulation." );
            }

            util::mfemOut( "solution_buffer.size(): ", solution_buffer.size(), "\n" );
            *u = solution_buffer[0].u;
            lambda = solution_buffer[0].lambda;
            L = solution_buffer[0].L / goldenRatio;
            solution_buffer.shift();
        }

        if ( step )
            PredictDirection();

        util::mfemOut( "L: ", L, ", phi: ", phi, "\n", util::Color::RESET );

        delta_u = 0.;
        Delta_u = 0.;
        delta_lambda = 0.;
        Delta_lambda = 0.;
        it = 0;

        // mfem::out << std::setprecision( 16 ) << "time: " << lambda << std::endl;
        for ( ; true; it++ )
        {
            if ( it >= max_iter )
            {
                converged = false;
                break;
            }

            add( *u, Delta_u, u_cur );

            // compute r
            oper->Mult( u_cur, r );

            if ( it > 0 )
            {
                // convergence check
                normPrevPrev = normPrev;
                normPrev = norm;
                norm = Norm( delta_u );
                // delta_u.Print();
                if ( it == 1 )
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
                    // filter out slow convergence case
                    if ( check_conv_ratio && it >= std::max( 3, max_iter * 4 / 5 ) &&
                         util::ConvergenceRate( norm, normPrev, normPrevPrev ) < 1.2 )
                    {
                        mfem::out << "Convergence rate " << util::ConvergenceRate( norm, normPrev, normPrevPrev ) << " is too small!\n";
                        converged = false;
                        break;
                    }

                    converged = true;
                    break;
                }
            }

            // compute q
            lambda += 1.;
            oper->Mult( u_cur, q );
            lambda -= 1.;

            q -= r;
            q.Neg();
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
            if ( it == 0 )
                delta_u_bar = 0.;
            else
                prec->Mult( r, delta_u_bar );

            if ( !updateStep( it, step, prec->Det() ) )
            {
                converged = false;
                break;
            }

            // update
            Delta_u += delta_u;
            Delta_lambda += delta_lambda;

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
        if ( GetConverged() )
        {
            if ( Delta_lambda < 0 )
            {
                util::mfemOut( "alert !!! buckled!!\n" );
            }

            if ( adaptive_mesh_refine_func )
            {
                // if ( !( *adaptive_mesh_refine_func )( Delta_u ) )
                // {
                //     mfem::out << "Refine mesh. Redo nonlinear step\n";
                //     ResizeVectors( x.Size() );
                //     u_direction_pred.Update();
                //     continue;
                // }
                // else
                // {
                //     mfem::out << "Stopping criterion satisfied. Stop refining.\n";
                // }
            }

            lambda += Delta_lambda;
            *u += Delta_u;
            step++;
            final_iter = it;
            final_norm = norm;

            if ( adaptive_l )
                phi = std::abs( Norm( Delta_u ) / Delta_lambda );

            if ( data_collect_func )
            {
                ( data_collect_func )( step, count, count );
            }
            solution_buffer.unshift();
            solution_buffer[0].L = L;
            solution_buffer[0].lambda = lambda;
            solution_buffer[0].u = *u;

            count++;
        }
        util::mfemOut( util::ProgressBar( lambda, converged ), '\n' );
    }
}

bool Crisfield::updateStep( const int it, const int step, const double det ) const
{
    const double delta_u_bar_dot_delta_u_t = Dot( delta_u_bar, delta_u_t );
    const double delta_u_bar_dot_delta_u_bar = Dot( delta_u_bar, delta_u_bar );
    const double delta_u_t_dot_delta_u_t = Dot( delta_u_t, delta_u_t );
    const double Delta_u_dot_delta_u_t = Dot( Delta_u, delta_u_t );
    const double Delta_u_dot_delta_u_bar = Dot( Delta_u, delta_u_bar );
    const double Delta_u_dot_Delta_u = Dot( Delta_u, Delta_u );

    // Ritto-Correa et al. 2008
    const double a0 = delta_u_t_dot_delta_u_t + phi;
    const double b0 = 2 * ( Delta_u_dot_delta_u_t + phi * Delta_lambda );
    const double b1 = 2 * delta_u_bar_dot_delta_u_t;
    const double c0 = Delta_u_dot_Delta_u + phi * Delta_lambda * Delta_lambda - L * L;
    const double c1 = 2 * Delta_u_dot_delta_u_bar;
    const double c2 = delta_u_bar_dot_delta_u_bar;

    double ds = 1.;
    double delta_lambda1{ 0. }, delta_lambda2{ 0. };

    const double as = b1 * b1 - 4 * a0 * c2;
    const double bs = 2 * b0 * b1 - 4 * a0 * c1;
    const double cs = b0 * b0 - 4 * a0 * c0;

    auto func = [as, bs, cs]( const double ds ) { return as * ds * ds + bs * ds + cs; };

    if ( func( ds ) < 0 )
    {
        return false;
        util::mfemOut( util::Color::YELLOW, "Complex root detected, adaptive step size (ds) is activated!\n", util::Color::RESET );
        const double det = bs * bs - 4 * as * cs;
        if ( det < 0 )
        {
            util::mfemOut( "bs^2 - 4 * as * cs < 0\n" );
            return false;
        }
        double beta1 = ( -bs + std::sqrt( det ) ) / ( 2 * as );
        double beta2 = ( -bs - std::sqrt( det ) ) / ( 2 * as );
        if ( beta1 > beta2 )
            std::swap( beta1, beta2 );
        util::mfemOut( util::Color::YELLOW, std::setprecision( 16 ), "beta1: ", beta1, ", beta2: ", beta2, '\n', util::Color::RESET );
        const double xi = beta2 - beta1;

        // Zhou 1995
        // ds = std::min( beta2 - xi * .05, ds );
        // util::mfemOut( util::Color::YELLOW, "ds=: ", ds, util::Color::RESET );
        // if ( ds < .1 )
        // {
        //     util::mfemOut( util::Color::YELLOW, ", which is smaller than ds_min, restart!\n ", util::Color::RESET );
        //     return false;
        // }

        // Lam and Morley 1992
        if ( beta2 < 1.0 )
            ds = beta2 - xi;
        else if ( ( beta2 > 1.0 ) && ( -bs / as < 1.0 ) )
            ds = beta2 + xi;
        else if ( ( beta2 < 1.0 ) && ( -bs / as > 1.0 ) )
            ds = beta2 - xi;
        else if ( beta2 > 1.0 )
            ds = beta2 + xi;
        else
        {
            util::mfemOut( "Could not find an appropriate adaptive step size ds ", '\n' );
            return false;
        }
        // util::mfemOut( "func(ds)= ", func( ds ), '\n' );

        const double a = a0;
        const double b = b0 + b1 * ds;

        delta_lambda1 = -1. * b / ( 2. * a );
        delta_lambda2 = -1. * b / ( 2. * a );
    }
    else
    {
        const double a = a0;
        const double b = b0 + b1 * ds;
        const double c = c0 + c1 * ds + c2 * ds * ds;

        delta_lambda1 = ( -1. * b + std::sqrt( b * b - 4 * a * c ) ) / ( 2. * a );
        delta_lambda2 = ( -1. * b - std::sqrt( b * b - 4 * a * c ) ) / ( 2. * a );
    }

    util::mfemOut( "delta_lambda1: ", delta_lambda1, ", delta_lambda2: ", delta_lambda2, ", det", ( det > 0 ), "\n",
                   util::Color::RESET );
    if ( it == 0 )
    {
        // // predictor Ritto-Corrêa and Dinar Camotim
        // if ( step == 0 )
        // {
        //     delta_lambda = delta_lambda1;
        // }
        // else
        // {
        //     if ( InnerProduct( delta_u_t, 1, u_direction_pred, lambda_direction_pred ) > 0 )
        //     {
        //         delta_lambda = delta_lambda1;
        //     }
        //     else
        //     {
        //         delta_lambda = delta_lambda2;
        //     }
        // }
        delta_lambda = det > 0. ? delta_lambda1 : delta_lambda2;
    }
    else
    {
        // corrector Ritto-Corrêa and Dinar Camotim
        const double t = InnerProduct( Delta_u, Delta_lambda, delta_u_t, 1. );
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

bool ArcLengthLinearize::updateStep( const int it, const int step, const double det ) const
{
    const double frac = 1.;
    const double tol = 1e-9;
    if ( it == 0 )
    {
        // predictor
        if ( Norm( u_direction_pred ) < tol && std::abs( lambda_direction_pred ) < tol )
        {
            delta_u = delta_u_t;
            const double L_pred = InnerProduct( delta_u, 1, delta_u, 1 );
            delta_u *= frac * L / L_pred;
            delta_lambda = frac * L / L_pred;
        }
        else
        {
            delta_u = u_direction_pred;
            delta_u *= L * frac;
            delta_lambda = L * frac * lambda_direction_pred;
        }
    }
    else
    {
        // corrector
        const double Delta_u_dot_delta_u_t = Dot( Delta_u, delta_u_t );
        const double Delta_u_dot_delta_u_bar = Dot( Delta_u, delta_u_bar );
        const double Delta_u_dot_Delta_u = Dot( Delta_u, Delta_u );

        delta_lambda = ( L * L - Delta_u_dot_Delta_u - phi * Delta_lambda * Delta_lambda - 2 * Delta_u_dot_delta_u_bar ) /
                       ( 2 * Delta_u_dot_delta_u_t + 2 * phi * Delta_lambda );

        add( 1., delta_u_bar, delta_lambda, delta_u_t, delta_u );
    }
    return true;
}

template <typename Newton>
void MultiNewtonAdaptive<Newton>::SetOperator( const mfem::Operator& op )
{
    Newton::SetOperator( op );
    cur.SetSize( Newton::width );
}

template <typename Newton>
void MultiNewtonAdaptive<Newton>::Mult( const mfem::Vector& b, mfem::Vector& x ) const
{
    Newton::lambda = 0.;

    mfem::Vector* u;
    u = &x;

    cur = *u;
    for ( int count = 0; true; count++ )
    {
        if ( count == max_steps )
        {
            break;
        }
        if ( Newton::lambda >= 1. )
        {
            break;
        }
        if ( count )
        {
            if ( !Newton::GetConverged() )
                Newton::Delta_lambda /= 2;
            else
            {
                Newton::Delta_lambda *= std::min( 1.2, std::pow( this->max_iter * .8 / Newton::GetNumIterations(), .5 ) );
                Newton::Delta_lambda = std::min( Newton::Delta_lambda, max_delta );
            }
        }
        MFEM_VERIFY( Newton::Delta_lambda > min_delta, "Required step size is smaller than the minimal bound." );

        util::mfemOut( "L: ", Newton::Delta_lambda, "\n", util::Color::RESET );
        Newton::Delta_lambda = std::min( Newton::Delta_lambda, 1. - Newton::lambda );

        Newton::Mult( b, *u );
        if ( Newton::GetConverged() )
        {
            Newton::lambda += Newton::Delta_lambda;
            cur = *u;

            if ( Newton::data_collect_func )
            {
                ( Newton::data_collect_func )( count, count, Newton::lambda );
            }
            Newton::step++;
        }
        else
        {
            *u = cur;
        }
        util::mfemOut( util::ProgressBar( Newton::lambda, Newton::GetConverged() ), '\n' );
    }
}

template class MultiNewtonAdaptive<NewtonLineSearch>;

template class MultiNewtonAdaptive<NewtonForPhaseField>;
} // namespace plugin