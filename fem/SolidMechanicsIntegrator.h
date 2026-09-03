#pragma once

#include "FEMPlugin.h"
#include "MaterialPointState.h"
#include "SolidMaterial.h"
#include "Solvers.h"
#include "StressFreeDeformation.h"

#include <Eigen/Dense>
#include <cstddef>
#include <functional>
#include <limits>
#include <mfem.hpp>
#include <tuple>
#include <type_traits>
#include <utility>

namespace plugin
{
template <typename Material>
struct SolidMechanicsTransactionSlot
{
    const void* Owner{ nullptr };
    std::size_t Generation{ 0 };
};

template <typename... Materials>
class SolidMechanicsMaterialPointState : public MaterialPointStateBundle<Materials...>
{
public:
    template <typename Material, typename Prepare>
    void PrepareForTransaction( const void* owner, const std::size_t generation, Prepare&& prepare )
    {
        auto& slot = std::get<SolidMechanicsTransactionSlot<Material>>( mTransactions );
        if ( slot.Owner == owner && slot.Generation == generation )
        {
            return;
        }

        std::forward<Prepare>( prepare )( this->template Get<Material>() );
        slot.Owner = owner;
        slot.Generation = generation;
    }

    template <typename Material>
    void FinishTransaction( const void* owner, const std::size_t generation ) noexcept
    {
        auto& slot = std::get<SolidMechanicsTransactionSlot<Material>>( mTransactions );
        if ( slot.Owner == owner && slot.Generation == generation )
        {
            slot.Owner = nullptr;
            slot.Generation = 0;
        }
    }

private:
    std::tuple<SolidMechanicsTransactionSlot<Materials>...> mTransactions;
};

namespace detail
{
template <typename PointStorage, typename Material, typename = void>
struct StoresMaterialPointState : std::false_type
{
};

template <typename PointStorage, typename Material>
struct StoresMaterialPointState<
    PointStorage,
    Material,
    std::void_t<typename PointStorage::ElementStateType,
                decltype( std::declval<typename PointStorage::ElementStateType&>().template Get<Material>() ),
                decltype( std::declval<typename PointStorage::ElementStateType&>().template PrepareForTransaction<Material>(
                    nullptr, std::size_t{}, std::declval<void ( * )( MaterialPointState<Material>& )>() ) ),
                decltype( std::declval<typename PointStorage::ElementStateType&>().template FinishTransaction<Material>( nullptr, std::size_t{} ) )>>
    : std::is_same<decltype( std::declval<typename PointStorage::ElementStateType&>().template Get<Material>() ), MaterialPointState<Material>&>
{
};
} // namespace detail

template <typename Material>
using SolidMechanicsIntegrationPointState =
    std::conditional_t<std::is_same_v<MaterialPointState<Material>, NoIntegrationPointState>, NoIntegrationPointState, SolidMechanicsMaterialPointState<Material>>;

template <typename Material>
using SolidMechanicsPointStorage = IntegrationPointStorage<SolidMechanicsIntegrationPointState<Material>>;

/// Domain integrator for a statically typed small- or finite-strain material.
///
/// A Material declares `static constexpr SolidKinematics Kinematics` and provides
/// `Evaluate(point)` when stateless, or `Evaluate(point, committedState)` when its
/// MaterialPointTraits state is a committed/trial history. Responses provide
/// Stress and ConsistentTangent; stateful responses additionally provide TrialState.
template <typename Material, typename PointStorage = SolidMechanicsPointStorage<Material>>
class SolidMechanicsIntegrator : public StepAwareNonlinearFormIntegrator
{
    static_assert( std::is_base_of_v<IntegrationPointStorageBase, PointStorage>,
                   "SolidMechanicsIntegrator requires an IntegrationPointStorage specialization." );
    static_assert( Material::Kinematics == SolidKinematics::SmallStrain || Material::Kinematics == SolidKinematics::FiniteStrain,
                   "A solid material must declare supported small- or finite-strain kinematics." );
    static_assert( std::is_same_v<MaterialPointState<Material>, NoIntegrationPointState> ||
                       detail::StoresMaterialPointState<PointStorage, Material>::value,
                   "A stateful solid material requires typed point storage containing that material." );

public:
    SolidMechanicsIntegrator( const Material& material, PointStorage& pointStorage )
        : mMaterial( material ), mPointStorage( pointStorage )
    {
    }
    SolidMechanicsIntegrator( Material&&, PointStorage& ) = delete;
    SolidMechanicsIntegrator( const Material&&, PointStorage& ) = delete;

    SolidMechanicsIntegrator( const SolidMechanicsIntegrator& ) = delete;
    SolidMechanicsIntegrator& operator=( const SolidMechanicsIntegrator& ) = delete;
    SolidMechanicsIntegrator( SolidMechanicsIntegrator&& ) = delete;
    SolidMechanicsIntegrator& operator=( SolidMechanicsIntegrator&& ) = delete;

    void AssembleElementVector( const mfem::FiniteElement& element,
                                mfem::ElementTransformation& transformation,
                                const mfem::Vector& elementDisplacement,
                                mfem::Vector& elementResidual ) override;

    void AssembleElementGrad( const mfem::FiniteElement& element,
                              mfem::ElementTransformation& transformation,
                              const mfem::Vector& elementDisplacement,
                              mfem::DenseMatrix& elementJacobian ) override;

    mfem::real_t GetElementEnergy( const mfem::FiniteElement&, mfem::ElementTransformation&, const mfem::Vector& ) override
    {
        MFEM_ABORT( "Incremental energy is not implemented for SolidMechanicsIntegrator materials." );
        return 0.;
    }

    void AddStressFreeDeformation( StressFreeDeformation& deformation )
    {
        mStressFreeDeformations.Add( deformation );
    }

    void ClearStressFreeDeformations()
    {
        mStressFreeDeformations.Clear();
    }

    void BeginStep() noexcept override;
    void CommitStep() noexcept override;
    void RollbackStep() noexcept override;
    bool CanCommitStep() const noexcept override
    {
        return !mStepRejected;
    }

private:
    static constexpr bool HasPersistentState = !std::is_same_v<MaterialPointState<Material>, NoIntegrationPointState>;

    struct FiniteAssemblyKinematics
    {
        std::reference_wrapper<const Eigen::MatrixXr> ShapeGradient;
        std::reference_wrapper<const Eigen::Matrix3r> DeformationGradient;
        mfem::real_t VolumeScale;
    };

    template <typename Visitor>
    void VisitHistory( Visitor&& visitor );

    void FinishHistoryTransaction() noexcept;

    template <typename MaterialPoint, typename IntegrationPointData>
    auto EvaluateMaterial( const MaterialPoint& materialPoint, IntegrationPointData& point );

    template <typename State>
    void PrepareHistory( State& state );

    template <typename Response>
    static void VerifyResponse( const Response& response );

    Eigen::Matrix3r MechanicalStrain( const Eigen::MatrixXr& displacement,
                                      const Eigen::MatrixXr& shapeGradient,
                                      int dimension,
                                      mfem::ElementTransformation& transformation,
                                      const mfem::IntegrationPoint& integrationPoint );

    FiniteAssemblyKinematics PrepareFiniteKinematics( const Eigen::Matrix3r& deformationGradient,
                                                      const Eigen::MatrixXr& shapeGradient,
                                                      int dimension,
                                                      mfem::ElementTransformation& transformation,
                                                      const mfem::IntegrationPoint& integrationPoint );

    const mfem::IntegrationRule& IntegrationRule( const mfem::FiniteElement& element ) const;

    const Material& mMaterial;
    PointStorage& mPointStorage;
    StressFreeDeformationModel mStressFreeDeformations;
    Eigen::Matrix<mfem::real_t, 6, Eigen::Dynamic> mB;
    Eigen::MatrixXr mAssemblyShapeGradient;
    Eigen::MatrixXr mGeometricStiffness;
    Eigen::Matrix3r mConstitutiveDeformationGradient;
    Eigen::Matrix3r mInverseStressFreeDeformationGradient;
    Eigen::Matrix3r mStressFreeDeformationGradient;
    std::size_t mTransactionGeneration{ 0 };
    int mStepDepth{ 0 };
    bool mStepRejected{ false };
};
} // namespace plugin

#include "SolidMechanicsIntegrator.tpp"
