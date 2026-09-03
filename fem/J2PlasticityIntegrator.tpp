#pragma once

#include "util.h"

#include <algorithm>
#include <vector>

namespace plugin
{
template <typename PointStorage>
const mfem::IntegrationRule& J2PlasticityIntegrator<PointStorage>::IntegrationRule( const mfem::FiniteElement& element ) const
{
    return IntRule ? *IntRule : mfem::IntRules.Get( element.GetGeomType(), 2 * element.GetOrder() + 1 );
}

template <typename PointStorage>
Eigen::Matrix3r J2PlasticityIntegrator<PointStorage>::MechanicalStrain( const Eigen::MatrixXr& displacement,
                                                                        const Eigen::MatrixXr& shapeGradient,
                                                                        const int dimension,
                                                                        mfem::ElementTransformation& transformation,
                                                                        const mfem::IntegrationPoint& integrationPoint )
{
    Eigen::Matrix3r displacementGradient = Eigen::Matrix3r::Zero();
    displacementGradient.topLeftCorner( dimension, dimension ).noalias() = displacement.transpose() * shapeGradient;
    Eigen::Matrix3r mechanicalStrain = .5 * ( displacementGradient + displacementGradient.transpose() );
    if ( !mStressFreeDeformations.Empty() )
    {
        mechanicalStrain -=
            mStressFreeDeformations.EvalSmallStrain( transformation, integrationPoint, mStepContext->GetCurLambda() );
    }
    return mechanicalStrain;
}

template <typename PointStorage>
void J2PlasticityIntegrator<PointStorage>::AssembleElementVector( const mfem::FiniteElement& element,
                                                                  mfem::ElementTransformation& transformation,
                                                                  const mfem::Vector& elementDisplacement,
                                                                  mfem::Vector& elementResidual )
{
    MFEM_VERIFY( mStepContext != nullptr, "J2 plasticity integration requires a nonlinear step context." );
    const int dofs = element.GetDof();
    const int dimension = element.GetDim();
    MFEM_VERIFY( dimension == 2 || dimension == 3, "J2 plasticity supports only two- and three-dimensional elements." );
    MFEM_VERIFY( elementDisplacement.Size() == dofs * dimension, "J2 element displacement has an incompatible size." );

    // MFEM supplies local vector DOFs in component blocks for either global Ordering::Type.
    Eigen::Map<const Eigen::MatrixXr> displacement( elementDisplacement.GetData(), dofs, dimension );
    elementResidual.SetSize( dofs * dimension );
    elementResidual = 0.;
    Eigen::Map<Eigen::VectorXr> residual( elementResidual.GetData(), elementResidual.Size() );

    const mfem::IntegrationRule& rule = IntegrationRule( element );
    mPointStorage.InitializeElement( element, transformation, rule );
    for ( int i = 0; i < rule.GetNPoints(); i++ )
    {
        const mfem::IntegrationPoint& integrationPoint = rule.IntPoint( i );
        transformation.SetIntPoint( &integrationPoint );
        auto& point = mPointStorage.GetElementPoint( i );
        smallDeformMatrixB( dofs, dimension, point.GShape, mB );

        auto& history = point.State.template Get<J2PlasticityMaterial>();
        const auto response =
            mMaterial.Evaluate( MechanicalStrain( displacement, point.GShape, dimension, transformation, integrationPoint ),
                                history.CommittedState(), transformation, integrationPoint );
        history.SetTrialState( response.TrialState );
        const Eigen::Vector6r stress = util::Voigt<mfem::real_t, mfem::real_t>( response.Stress, false );
        residual.noalias() += point.Weight * mB.transpose() * stress;
    }
}

template <typename PointStorage>
void J2PlasticityIntegrator<PointStorage>::AssembleElementGrad( const mfem::FiniteElement& element,
                                                                mfem::ElementTransformation& transformation,
                                                                const mfem::Vector& elementDisplacement,
                                                                mfem::DenseMatrix& elementJacobian )
{
    MFEM_VERIFY( mStepContext != nullptr, "J2 plasticity integration requires a nonlinear step context." );
    const int dofs = element.GetDof();
    const int dimension = element.GetDim();
    MFEM_VERIFY( dimension == 2 || dimension == 3, "J2 plasticity supports only two- and three-dimensional elements." );
    MFEM_VERIFY( elementDisplacement.Size() == dofs * dimension, "J2 element displacement has an incompatible size." );

    // MFEM supplies local vector DOFs in component blocks for either global Ordering::Type.
    Eigen::Map<const Eigen::MatrixXr> displacement( elementDisplacement.GetData(), dofs, dimension );
    elementJacobian.SetSize( dofs * dimension );
    elementJacobian = 0.;
    Eigen::Map<Eigen::MatrixXr> jacobian( elementJacobian.Data(), elementJacobian.Height(), elementJacobian.Width() );

    const mfem::IntegrationRule& rule = IntegrationRule( element );
    mPointStorage.InitializeElement( element, transformation, rule );
    for ( int i = 0; i < rule.GetNPoints(); i++ )
    {
        const mfem::IntegrationPoint& integrationPoint = rule.IntPoint( i );
        transformation.SetIntPoint( &integrationPoint );
        auto& point = mPointStorage.GetElementPoint( i );
        smallDeformMatrixB( dofs, dimension, point.GShape, mB );

        auto& history = point.State.template Get<J2PlasticityMaterial>();
        const auto response =
            mMaterial.Evaluate( MechanicalStrain( displacement, point.GShape, dimension, transformation, integrationPoint ),
                                history.CommittedState(), transformation, integrationPoint );
        history.SetTrialState( response.TrialState );
        jacobian.noalias() += point.Weight * mB.transpose() * response.ConsistentTangent * mB;
    }
}

template <typename PointStorage>
void J2PlasticityIntegrator<PointStorage>::BeginStep() noexcept
{
    StepAwareNonlinearFormIntegrator::BeginStep();
    if ( mStepDepth > 0 )
    {
        mStepDepth++;
        return;
    }

    VisitHistory( []( J2PlasticityHistory& history ) { history.BeginStep(); } );
    mStepDepth = 1;
    mStepRejected = false;
}

template <typename PointStorage>
void J2PlasticityIntegrator<PointStorage>::CommitStep() noexcept
{
    MFEM_VERIFY( mStepDepth > 0, "J2 plasticity commit requires a matching BeginStep." );
    if ( mStepDepth > 1 )
    {
        mStepDepth--;
        StepAwareNonlinearFormIntegrator::CommitStep();
        return;
    }

    if ( mStepRejected )
    {
        VisitHistory( []( J2PlasticityHistory& history ) { history.RollbackStep(); } );
    }
    else
    {
        VisitHistory( []( J2PlasticityHistory& history ) { history.CommitStep(); } );
    }
    mStepDepth = 0;
    mStepRejected = false;
    StepAwareNonlinearFormIntegrator::CommitStep();
}

template <typename PointStorage>
void J2PlasticityIntegrator<PointStorage>::RollbackStep() noexcept
{
    MFEM_VERIFY( mStepDepth > 0, "J2 plasticity rollback requires a matching BeginStep." );
    mStepRejected = true;
    if ( mStepDepth > 1 )
    {
        mStepDepth--;
        StepAwareNonlinearFormIntegrator::RollbackStep();
        return;
    }

    VisitHistory( []( J2PlasticityHistory& history ) { history.RollbackStep(); } );
    mStepDepth = 0;
    mStepRejected = false;
    StepAwareNonlinearFormIntegrator::RollbackStep();
}
} // namespace plugin
