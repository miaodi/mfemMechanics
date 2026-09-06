#pragma once

namespace plugin
{
template <typename PointStorage>
void PhaseFieldIntegrator<PointStorage>::UpdateHistory( PhaseFieldHistory& history, mfem::real_t& positiveEnergy )
{
    positiveEnergy = history.EvaluateTrial( positiveEnergy );
}

template <typename PointStorage>
void PhaseFieldIntegrator<PointStorage>::BeginStep() noexcept
{
    BlockStepAwareNonlinearFormIntegrator::BeginStep();
    if ( mStepDepth > 0 )
    {
        mStepDepth++;
        return;
    }

    VisitHistory( []( PhaseFieldHistory& history ) { history.BeginStep(); } );
    mStepDepth = 1;
    mStepRejected = false;
}

template <typename PointStorage>
void PhaseFieldIntegrator<PointStorage>::CommitStep() noexcept
{
    MFEM_VERIFY( mStepDepth > 0, "Phase-field commit requires a matching BeginStep." );
    if ( mStepDepth > 1 )
    {
        mStepDepth--;
        BlockStepAwareNonlinearFormIntegrator::CommitStep();
        return;
    }

    if ( mStepRejected )
    {
        VisitHistory( []( PhaseFieldHistory& history ) { history.RollbackStep(); } );
    }
    else
    {
        VisitHistory( []( PhaseFieldHistory& history ) { history.CommitStep(); } );
    }
    mStepDepth = 0;
    mStepRejected = false;
    BlockStepAwareNonlinearFormIntegrator::CommitStep();
}

template <typename PointStorage>
void PhaseFieldIntegrator<PointStorage>::RollbackStep() noexcept
{
    MFEM_VERIFY( mStepDepth > 0, "Phase-field rollback requires a matching BeginStep." );
    mStepRejected = true;
    if ( mStepDepth > 1 )
    {
        mStepDepth--;
        BlockStepAwareNonlinearFormIntegrator::RollbackStep();
        return;
    }

    VisitHistory( []( PhaseFieldHistory& history ) { history.RollbackStep(); } );
    mStepDepth = 0;
    mStepRejected = false;
    BlockStepAwareNonlinearFormIntegrator::RollbackStep();
}

template <typename PointStorage>
void PhaseFieldIntegrator<PointStorage>::AssembleElementVector( const mfem::Array<const mfem::FiniteElement*>& el,
                                                                mfem::ElementTransformation& transformation,
                                                                const mfem::Array<const mfem::Vector*>& elementState,
                                                                const mfem::Array<mfem::Vector*>& elementResidual )
{
    MFEM_VERIFY( mStepContext != nullptr, "Phase-field integration requires a nonlinear step context." );

    const int displacementDofs = el[0]->GetDof();
    const int dimension = el[0]->GetDim();
    const int phaseDofs = el[1]->GetDof();

    mDShape.SetSize( phaseDofs, dimension );
    mGShape.SetSize( phaseDofs, dimension );
    shape.SetSize( phaseDofs );

    const mfem::real_t fractureEnergy = mMaterialModel->getGc();
    const mfem::real_t residualStiffness = mMaterialModel->getK();
    const mfem::real_t lengthScale = mMaterialModel->getL0();

    Eigen::Map<const Eigen::MatrixXr> displacement( elementState[0]->GetData(), displacementDofs, dimension );
    Eigen::Map<const Eigen::VectorXr> phaseField( elementState[1]->GetData(), phaseDofs );

    elementResidual[0]->SetSize( displacementDofs * dimension );
    elementResidual[1]->SetSize( phaseDofs );
    *elementResidual[0] = 0.;
    *elementResidual[1] = 0.;

    Eigen::Map<Eigen::VectorXr> displacementResidual( elementResidual[0]->GetData(), displacementDofs * dimension );
    Eigen::Map<Eigen::VectorXr> phaseResidual( elementResidual[1]->GetData(), phaseDofs );

    const mfem::IntegrationRule& rule =
        mIntegrationRule
            ? *mIntegrationRule
            : mfem::IntRules.Get( el[0]->GetGeomType(), 2 * std::max( el[0]->GetOrder(), el[1]->GetOrder() ) + 1 );
    const Eigen::Matrix3r identity = Eigen::Matrix3r::Identity();

    mPointStorage.InitializeElement( *el[0], transformation, rule );
    for ( int i = 0; i < rule.GetNPoints(); i++ )
    {
        const mfem::IntegrationPoint& integrationPoint = rule.IntPoint( i );
        transformation.SetIntPoint( &integrationPoint );
        auto& point = mPointStorage.GetElementPoint( i );
        const Eigen::MatrixXr& shapeGradient = point.GShape;

        mdxdX.setZero();
        mdxdX.block( 0, 0, dimension, dimension ) = displacement.transpose() * shapeGradient;
        mdxdX += identity;

        el[1]->CalcShape( integrationPoint, shape );
        el[1]->CalcDShape( integrationPoint, mDShape );
        Mult( mDShape, transformation.InverseJacobian(), mGShape );
        Eigen::Map<const Eigen::MatrixXr> phaseGradientShape( mGShape.Data(), phaseDofs, dimension );
        Eigen::Map<const Eigen::VectorXr> phaseShape( shape.GetData(), phaseDofs );

        smallDeformMatrixB( displacementDofs, dimension, shapeGradient, mB );

        const mfem::real_t phaseValue = phaseField.dot( phaseShape );
        const Eigen::MatrixXr phaseGradient = phaseField.transpose() * phaseGradientShape;

        mMaterialModel->at( transformation, integrationPoint );
        mMaterialModel->setDeformationGradient( mdxdX );
        mMaterialModel->setPhaseField( phaseValue );

        const mfem::real_t weight = integrationPoint.weight * point.DetdXdXi;
        const auto response = mMaterialModel->EvaluateResponse( false );
        displacementResidual += weight * ( mB.transpose() * response.stress );

        mfem::real_t positiveEnergy = response.positiveEnergy;
        auto& history = point.State.template Get<PhaseFieldElasticMaterial>();
        UpdateHistory( history, positiveEnergy );

        phaseResidual += weight * ( -( 2 * ( 1 - residualStiffness ) * positiveEnergy * ( 1 - phaseValue ) * phaseShape ) +
                                    fractureEnergy * ( lengthScale * phaseGradientShape * phaseGradient.transpose() +
                                                       phaseValue / lengthScale * phaseShape ) );
    }
}

template <typename PointStorage>
void PhaseFieldIntegrator<PointStorage>::AssembleElementGrad( const mfem::Array<const mfem::FiniteElement*>& el,
                                                              mfem::ElementTransformation& transformation,
                                                              const mfem::Array<const mfem::Vector*>& elementState,
                                                              const mfem::Array2D<mfem::DenseMatrix*>& elementJacobian )
{
    MFEM_VERIFY( mStepContext != nullptr, "Phase-field integration requires a nonlinear step context." );

    const int displacementDofs = el[0]->GetDof();
    const int dimension = el[0]->GetDim();
    const int phaseDofs = el[1]->GetDof();

    mDShape.SetSize( phaseDofs, dimension );
    mGShape.SetSize( phaseDofs, dimension );
    shape.SetSize( phaseDofs );

    const mfem::real_t fractureEnergy = mMaterialModel->getGc();
    const mfem::real_t residualStiffness = mMaterialModel->getK();
    const mfem::real_t lengthScale = mMaterialModel->getL0();

    Eigen::Map<const Eigen::MatrixXr> displacement( elementState[0]->GetData(), displacementDofs, dimension );
    Eigen::Map<const Eigen::VectorXr> phaseField( elementState[1]->GetData(), phaseDofs );

    elementJacobian( 0, 0 )->SetSize( displacementDofs * dimension, displacementDofs * dimension );
    elementJacobian( 0, 1 )->SetSize( displacementDofs * dimension, phaseDofs );
    elementJacobian( 1, 0 )->SetSize( phaseDofs, displacementDofs * dimension );
    elementJacobian( 1, 1 )->SetSize( phaseDofs, phaseDofs );
    *elementJacobian( 0, 0 ) = 0.;
    *elementJacobian( 0, 1 ) = 0.;
    *elementJacobian( 1, 0 ) = 0.;
    *elementJacobian( 1, 1 ) = 0.;

    Eigen::Map<Eigen::MatrixXr> displacementJacobian( elementJacobian( 0, 0 )->Data(), displacementDofs * dimension,
                                                      displacementDofs * dimension );
    Eigen::Map<Eigen::MatrixXr> displacementPhaseJacobian( elementJacobian( 0, 1 )->Data(), displacementDofs * dimension, phaseDofs );
    Eigen::Map<Eigen::MatrixXr> phaseDisplacementJacobian( elementJacobian( 1, 0 )->Data(), phaseDofs, displacementDofs * dimension );
    Eigen::Map<Eigen::MatrixXr> phaseJacobian( elementJacobian( 1, 1 )->Data(), phaseDofs, phaseDofs );

    const mfem::IntegrationRule& rule =
        mIntegrationRule
            ? *mIntegrationRule
            : mfem::IntRules.Get( el[0]->GetGeomType(), 2 * std::max( el[0]->GetOrder(), el[1]->GetOrder() ) + 1 );
    const Eigen::Matrix3r identity = Eigen::Matrix3r::Identity();

    mPointStorage.InitializeElement( *el[0], transformation, rule );
    for ( int i = 0; i < rule.GetNPoints(); i++ )
    {
        const mfem::IntegrationPoint& integrationPoint = rule.IntPoint( i );
        transformation.SetIntPoint( &integrationPoint );
        auto& point = mPointStorage.GetElementPoint( i );
        const Eigen::MatrixXr& shapeGradient = point.GShape;

        mdxdX.setZero();
        mdxdX.block( 0, 0, dimension, dimension ) = displacement.transpose() * shapeGradient;
        mdxdX += identity;

        el[1]->CalcShape( integrationPoint, shape );
        el[1]->CalcDShape( integrationPoint, mDShape );
        Mult( mDShape, transformation.InverseJacobian(), mGShape );
        Eigen::Map<const Eigen::MatrixXr> phaseGradientShape( mGShape.Data(), phaseDofs, dimension );
        Eigen::Map<const Eigen::VectorXr> phaseShape( shape.GetData(), phaseDofs );

        smallDeformMatrixB( displacementDofs, dimension, shapeGradient, mB );
        const mfem::real_t phaseValue = phaseField.dot( phaseShape );

        mMaterialModel->at( transformation, integrationPoint );
        mMaterialModel->setDeformationGradient( mdxdX );
        mMaterialModel->setPhaseField( phaseValue );
        const auto response = mMaterialModel->EvaluateResponse( true );

        const mfem::real_t weight = integrationPoint.weight * point.DetdXdXi;
        displacementJacobian += weight * mB.transpose() * *response.tangent * mB;
        displacementPhaseJacobian += weight * ( mB.transpose() * response.phaseStressDerivative ) * phaseShape.transpose();

        mfem::real_t positiveEnergy = response.positiveEnergy;
        auto& history = point.State.template Get<PhaseFieldElasticMaterial>();
        const bool historyIsActive = positiveEnergy > history.CommittedValue();
        UpdateHistory( history, positiveEnergy );

        if ( historyIsActive )
        {
            phaseDisplacementJacobian += weight * phaseShape * response.phaseStressDerivative.transpose() * mB;
        }

        phaseJacobian += weight * ( fractureEnergy * lengthScale * phaseGradientShape * phaseGradientShape.transpose() +
                                    ( fractureEnergy / lengthScale + 2 * ( 1 - residualStiffness ) * positiveEnergy ) *
                                        phaseShape * phaseShape.transpose() );
    }
}
} // namespace plugin
