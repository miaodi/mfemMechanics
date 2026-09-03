#pragma once

#include "util.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace plugin
{
template <typename Material, typename PointStorage>
const mfem::IntegrationRule& SolidMechanicsIntegrator<Material, PointStorage>::IntegrationRule( const mfem::FiniteElement& element ) const
{
    return IntRule ? *IntRule : mfem::IntRules.Get( element.GetGeomType(), 2 * element.GetOrder() + 1 );
}

template <typename Material, typename PointStorage>
template <typename MaterialPoint, typename IntegrationPointData>
auto SolidMechanicsIntegrator<Material, PointStorage>::EvaluateMaterial( const MaterialPoint& materialPoint, IntegrationPointData& point )
{
    if constexpr ( HasPersistentState )
    {
        PrepareHistory( point.State );
        auto& history = point.State.template Get<Material>();
        auto response = mMaterial.Evaluate( materialPoint, history.CommittedState() );
        VerifyResponse( response );
        if ( mStepDepth > 0 )
        {
            history.SetTrialState( response.TrialState );
        }
        return response;
    }
    else
    {
        auto response = mMaterial.Evaluate( materialPoint );
        VerifyResponse( response );
        return response;
    }
}

template <typename Material, typename PointStorage>
template <typename State>
void SolidMechanicsIntegrator<Material, PointStorage>::PrepareHistory( State& state )
{
    if constexpr ( HasPersistentState )
    {
        if ( mStepDepth > 0 )
        {
            state.template PrepareForTransaction<Material>( this, mTransactionGeneration,
                                                            []( auto& history ) { history.BeginStep(); } );
        }
    }
}

template <typename Material, typename PointStorage>
template <typename Response>
void SolidMechanicsIntegrator<Material, PointStorage>::VerifyResponse( const Response& response )
{
    MFEM_VERIFY( response.Stress.allFinite(), "A solid material returned nonfinite stress." );
    MFEM_VERIFY( response.ConsistentTangent.allFinite(), "A solid material returned a nonfinite tangent." );
    const mfem::real_t relativeTolerance = 64. * std::numeric_limits<mfem::real_t>::epsilon();
    for ( int row = 0; row < 3; row++ )
    {
        for ( int column = row + 1; column < 3; column++ )
        {
            const mfem::real_t first = response.Stress( row, column );
            const mfem::real_t second = response.Stress( column, row );
            const mfem::real_t scale = std::max( { mfem::real_t{ 1. }, std::abs( first ), std::abs( second ) } );
            MFEM_VERIFY( std::abs( first / scale - second / scale ) <= relativeTolerance,
                         "A solid material returned nonsymmetric stress." );
        }
    }
}

template <typename Material, typename PointStorage>
Eigen::Matrix3r SolidMechanicsIntegrator<Material, PointStorage>::MechanicalStrain( const Eigen::MatrixXr& displacement,
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
    MFEM_VERIFY( mechanicalStrain.allFinite(), "Small-strain solid mechanics requires finite mechanical strain." );
    return mechanicalStrain;
}

template <typename Material, typename PointStorage>
typename SolidMechanicsIntegrator<Material, PointStorage>::FiniteAssemblyKinematics SolidMechanicsIntegrator<Material, PointStorage>::PrepareFiniteKinematics(
    const Eigen::Matrix3r& deformationGradient,
    const Eigen::MatrixXr& shapeGradient,
    const int dimension,
    mfem::ElementTransformation& transformation,
    const mfem::IntegrationPoint& integrationPoint )
{
    if ( mStressFreeDeformations.Empty() )
    {
        return { std::cref( shapeGradient ), std::cref( deformationGradient ), 1. };
    }

    // Multiplicative decomposition F = F_e F_0, integrated over the stress-free volume.
    mStressFreeDeformationGradient =
        mStressFreeDeformations.EvalDeformationGradient( transformation, integrationPoint, mStepContext->GetCurLambda() );
    if ( dimension < 3 )
    {
        MFEM_VERIFY(
            mStressFreeDeformationGradient.topRightCorner( dimension, 3 - dimension ).isZero() &&
                mStressFreeDeformationGradient.bottomLeftCorner( 3 - dimension, dimension ).isZero(),
            "A reduced-dimensional stress-free deformation cannot couple active and out-of-plane directions." );
    }
    const mfem::real_t activeDeterminant = mStressFreeDeformationGradient.topLeftCorner( dimension, dimension ).determinant();
    const mfem::real_t volumeScale = mStressFreeDeformationGradient.determinant();
    MFEM_VERIFY( std::isfinite( activeDeterminant ) && activeDeterminant > 0. && std::isfinite( volumeScale ) && volumeScale > 0.,
                 "The stress-free deformation must preserve orientation." );

    mInverseStressFreeDeformationGradient = mStressFreeDeformationGradient.inverse();
    mConstitutiveDeformationGradient.noalias() = deformationGradient * mInverseStressFreeDeformationGradient;
    mAssemblyShapeGradient.noalias() = shapeGradient * mInverseStressFreeDeformationGradient.topLeftCorner( dimension, dimension );
    MFEM_VERIFY( mInverseStressFreeDeformationGradient.allFinite() && mConstitutiveDeformationGradient.allFinite() &&
                     mAssemblyShapeGradient.allFinite(),
                 "The finite-strain stress-free decomposition produced nonfinite kinematics." );
    return { std::cref( mAssemblyShapeGradient ), std::cref( mConstitutiveDeformationGradient ), volumeScale };
}

template <typename Material, typename PointStorage>
void SolidMechanicsIntegrator<Material, PointStorage>::AssembleElementVector( const mfem::FiniteElement& element,
                                                                              mfem::ElementTransformation& transformation,
                                                                              const mfem::Vector& elementDisplacement,
                                                                              mfem::Vector& elementResidual )
{
    MFEM_VERIFY( mStepContext != nullptr, "Solid mechanics integration requires a nonlinear step context." );
    const int dofs = element.GetDof();
    const int dimension = element.GetDim();
    MFEM_VERIFY( dimension == 2 || dimension == 3,
                 "Solid mechanics supports only two- and three-dimensional elements." );
    MFEM_VERIFY( elementDisplacement.Size() == dofs * dimension, "Element displacement has an incompatible size." );

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
        const MaterialPointContext context{ transformation, integrationPoint, mStepContext->GetCurLambda() };

        if constexpr ( Material::Kinematics == SolidKinematics::SmallStrain )
        {
            smallDeformMatrixB( dofs, dimension, point.GShape, mB );
            const Eigen::Matrix3r mechanicalStrain =
                MechanicalStrain( displacement, point.GShape, dimension, transformation, integrationPoint );
            const auto response = EvaluateMaterial( SmallStrainMaterialPoint{ mechanicalStrain, context }, point );
            const Eigen::Vector6r stress = util::Voigt<mfem::real_t, mfem::real_t>( response.Stress, false );
            residual.noalias() += point.Weight * mB.transpose() * stress;
        }
        else
        {
            Eigen::Matrix3r deformationGradient = Eigen::Matrix3r::Identity();
            deformationGradient.topLeftCorner( dimension, dimension ).noalias() += displacement.transpose() * point.GShape;
            const mfem::real_t determinant = deformationGradient.determinant();
            MFEM_VERIFY( std::isfinite( determinant ) && determinant > 0.,
                         "Finite-strain solid mechanics requires an orientation-preserving deformation gradient." );
            const auto kinematics =
                PrepareFiniteKinematics( deformationGradient, point.GShape, dimension, transformation, integrationPoint );
            const auto& assemblyGradient = kinematics.ShapeGradient.get();
            const auto& constitutiveDeformationGradient = kinematics.DeformationGradient.get();
            largeDeformMatrixB( dofs, dimension, assemblyGradient, constitutiveDeformationGradient, mB );
            const auto response = EvaluateMaterial( FiniteStrainMaterialPoint{ constitutiveDeformationGradient, context }, point );
            const Eigen::Vector6r stress = util::Voigt<mfem::real_t, mfem::real_t>( response.Stress, false );
            residual.noalias() += point.Weight * kinematics.VolumeScale * mB.transpose() * stress;
        }
    }
}

template <typename Material, typename PointStorage>
void SolidMechanicsIntegrator<Material, PointStorage>::AssembleElementGrad( const mfem::FiniteElement& element,
                                                                            mfem::ElementTransformation& transformation,
                                                                            const mfem::Vector& elementDisplacement,
                                                                            mfem::DenseMatrix& elementJacobian )
{
    MFEM_VERIFY( mStepContext != nullptr, "Solid mechanics integration requires a nonlinear step context." );
    const int dofs = element.GetDof();
    const int dimension = element.GetDim();
    MFEM_VERIFY( dimension == 2 || dimension == 3,
                 "Solid mechanics supports only two- and three-dimensional elements." );
    MFEM_VERIFY( elementDisplacement.Size() == dofs * dimension, "Element displacement has an incompatible size." );

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
        const MaterialPointContext context{ transformation, integrationPoint, mStepContext->GetCurLambda() };

        if constexpr ( Material::Kinematics == SolidKinematics::SmallStrain )
        {
            smallDeformMatrixB( dofs, dimension, point.GShape, mB );
            const Eigen::Matrix3r mechanicalStrain =
                MechanicalStrain( displacement, point.GShape, dimension, transformation, integrationPoint );
            const auto response = EvaluateMaterial( SmallStrainMaterialPoint{ mechanicalStrain, context }, point );
            jacobian.noalias() += point.Weight * mB.transpose() * response.ConsistentTangent * mB;
        }
        else
        {
            Eigen::Matrix3r deformationGradient = Eigen::Matrix3r::Identity();
            deformationGradient.topLeftCorner( dimension, dimension ).noalias() += displacement.transpose() * point.GShape;
            const mfem::real_t determinant = deformationGradient.determinant();
            MFEM_VERIFY( std::isfinite( determinant ) && determinant > 0.,
                         "Finite-strain solid mechanics requires an orientation-preserving deformation gradient." );
            const auto kinematics =
                PrepareFiniteKinematics( deformationGradient, point.GShape, dimension, transformation, integrationPoint );
            const auto& assemblyGradient = kinematics.ShapeGradient.get();
            const auto& constitutiveDeformationGradient = kinematics.DeformationGradient.get();
            largeDeformMatrixB( dofs, dimension, assemblyGradient, constitutiveDeformationGradient, mB );
            const auto response = EvaluateMaterial( FiniteStrainMaterialPoint{ constitutiveDeformationGradient, context }, point );
            const mfem::real_t weight = point.Weight * kinematics.VolumeScale;
            jacobian.noalias() += weight * mB.transpose() * response.ConsistentTangent * mB;

            mGeometricStiffness = ( weight * assemblyGradient * response.Stress.topLeftCorner( dimension, dimension ) *
                                    assemblyGradient.transpose() )
                                      .eval();
            for ( int component = 0; component < dimension; component++ )
            {
                jacobian.block( component * dofs, component * dofs, dofs, dofs ) += mGeometricStiffness;
            }
        }
    }
}

template <typename Material, typename PointStorage>
template <typename Visitor>
void SolidMechanicsIntegrator<Material, PointStorage>::VisitHistory( Visitor&& visitor )
{
    if constexpr ( HasPersistentState )
    {
        mPointStorage.VisitElementStates(
            [this, &visitor]( auto& state )
            {
                PrepareHistory( state );
                visitor( state.template Get<Material>() );
            } );
    }
}

template <typename Material, typename PointStorage>
void SolidMechanicsIntegrator<Material, PointStorage>::FinishHistoryTransaction() noexcept
{
    if constexpr ( HasPersistentState )
    {
        mPointStorage.VisitElementStates( [this]( auto& state )
                                          { state.template FinishTransaction<Material>( this, mTransactionGeneration ); } );
    }
}

template <typename Material, typename PointStorage>
void SolidMechanicsIntegrator<Material, PointStorage>::BeginStep() noexcept
{
    StepAwareNonlinearFormIntegrator::BeginStep();
    if ( mStepDepth > 0 )
    {
        mStepDepth++;
        return;
    }

    MFEM_VERIFY( mTransactionGeneration < std::numeric_limits<std::size_t>::max(),
                 "Solid mechanics transaction generation overflowed." );
    mTransactionGeneration++;
    mStepDepth = 1;
    mStepRejected = false;
    VisitHistory( []( auto& ) {} );
}

template <typename Material, typename PointStorage>
void SolidMechanicsIntegrator<Material, PointStorage>::CommitStep() noexcept
{
    MFEM_VERIFY( mStepDepth > 0, "Solid mechanics commit requires a matching BeginStep." );
    if ( mStepDepth > 1 )
    {
        mStepDepth--;
        StepAwareNonlinearFormIntegrator::CommitStep();
        return;
    }

    if ( mStepRejected )
    {
        VisitHistory( []( auto& history ) { history.RollbackStep(); } );
    }
    else
    {
        VisitHistory( []( auto& history ) { history.CommitStep(); } );
    }
    FinishHistoryTransaction();
    mStepDepth = 0;
    mStepRejected = false;
    StepAwareNonlinearFormIntegrator::CommitStep();
}

template <typename Material, typename PointStorage>
void SolidMechanicsIntegrator<Material, PointStorage>::RollbackStep() noexcept
{
    MFEM_VERIFY( mStepDepth > 0, "Solid mechanics rollback requires a matching BeginStep." );
    mStepRejected = true;
    if ( mStepDepth > 1 )
    {
        mStepDepth--;
        StepAwareNonlinearFormIntegrator::RollbackStep();
        return;
    }

    VisitHistory( []( auto& history ) { history.RollbackStep(); } );
    FinishHistoryTransaction();
    mStepDepth = 0;
    mStepRejected = false;
    StepAwareNonlinearFormIntegrator::RollbackStep();
}
} // namespace plugin
