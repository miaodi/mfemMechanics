#pragma once

#include "FEMPlugin.h"
#include "PhaseFieldMaterial.h"
#include "util.h"
#include <Eigen/Dense>
#include <mfem.hpp>
#include <type_traits>

namespace plugin
{
using PhaseFieldPointStorage = IntegrationPointStorage<PhaseFieldIntegrationPointState>;

template <typename PointStorage>
class PhaseFieldIntegrator : public BlockStepAwareNonlinearFormIntegrator
{
    static_assert( std::is_base_of_v<IntegrationPointStorageBase, PointStorage>,
                   "PhaseFieldIntegrator requires an IntegrationPointStorage specialization." );

public:
    /// Use separate storage for independent PhaseFieldElasticMaterial instances.
    PhaseFieldIntegrator( PhaseFieldElasticMaterial& m, PointStorage& pointStorage )
        : BlockStepAwareNonlinearFormIntegrator(), mMaterialModel( &m ), mPointStorage{ pointStorage }
    {
    }

    /// Perform the local action of the BlockNonlinearFormIntegrator
    void AssembleElementVector( const mfem::Array<const mfem::FiniteElement*>& el,
                                mfem::ElementTransformation& Tr,
                                const mfem::Array<const mfem::Vector*>& elfun,
                                const mfem::Array<mfem::Vector*>& elvec ) override;

    /// Assemble the local gradient matrix
    void AssembleElementGrad( const mfem::Array<const mfem::FiniteElement*>& el,
                              mfem::ElementTransformation& Tr,
                              const mfem::Array<const mfem::Vector*>& elfun,
                              const mfem::Array2D<mfem::DenseMatrix*>& elmats ) override;

    void BeginStep() override;
    void CommitStep() override;
    void RollbackStep() override;
    void RevertStep() override;

protected:
    template <typename Visitor>
    void VisitHistory( Visitor&& visitor )
    {
        mPointStorage.VisitElementStates( [&visitor]( auto& state )
                                          { visitor( state.template Get<PhaseFieldElasticMaterial>() ); } );
    }

    static void UpdateHistory( PhaseFieldHistory& history, mfem::real_t& positiveEnergy );

    PhaseFieldElasticMaterial* mMaterialModel{ nullptr };

    Eigen::Matrix<mfem::real_t, 3, 3> mdxdX;
    Eigen::Matrix<mfem::real_t, 6, Eigen::Dynamic> mB;
    PointStorage& mPointStorage;

    // data for phase field
    mfem::Vector shape;
    mfem::DenseMatrix mDShape, mGShape;
    int mStepDepth{ 0 };
    bool mStepRejected{ false };
};

class BlockNonlinearDirichletPenaltyIntegrator : public BlockStepAwareNonlinearFormIntegrator
{
public:
    BlockNonlinearDirichletPenaltyIntegrator( mfem::VectorCoefficient& QG, mfem::VectorCoefficient& HG )
        : BlockStepAwareNonlinearFormIntegrator(), mIntegrator( QG, HG )
    {
    }

    /// Perform the local action of the BlockNonlinearFormIntegrator
    virtual void AssembleFaceVector( const mfem::Array<const mfem::FiniteElement*>& el1,
                                     const mfem::Array<const mfem::FiniteElement*>& el2,
                                     mfem::FaceElementTransformations& Tr,
                                     const mfem::Array<const mfem::Vector*>& elfun,
                                     const mfem::Array<mfem::Vector*>& elvec );

    /// Assemble the local gradient matrix
    virtual void AssembleFaceGrad( const mfem::Array<const mfem::FiniteElement*>& el1,
                                   const mfem::Array<const mfem::FiniteElement*>& el2,
                                   mfem::FaceElementTransformations& Tr,
                                   const mfem::Array<const mfem::Vector*>& elfun,
                                   const mfem::Array2D<mfem::DenseMatrix*>& elmats );

    virtual void SetStepContext( NonlinearStepContext const* ptr )
    {
        mStepContext = ptr;
        mIntegrator.SetStepContext( ptr );
    }

protected:
    NonlinearDirichletPenaltyIntegrator mIntegrator;
};
} // namespace plugin

#include "PhaseField.tpp"
