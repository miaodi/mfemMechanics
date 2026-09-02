#pragma once

#include "FEMPlugin.h"
#include "PhaseFieldMaterial.h"
#include "util.h"
#include <Eigen/Dense>
#include <memory>
#include <mfem.hpp>
#include <vector>

namespace plugin
{
class PhaseFieldIntegrator : public BlockStepAwareNonlinearFormIntegrator
{
    struct PointData
    {
        double H{ 0. };
        double H_bac{ 0. };
        int success_step{ 0 };
    };

    void UpdateH( const int, double& );

public:
    PhaseFieldIntegrator( PhaseFieldElasticMaterial& m, IntegrationPointStorage& pointStorage )
        : BlockStepAwareNonlinearFormIntegrator(), mMaterialModel( &m ), mPointStorage{ pointStorage }
    {
    }

    /// Perform the local action of the BlockNonlinearFormIntegrator
    virtual void AssembleElementVector( const mfem::Array<const mfem::FiniteElement*>& el,
                                        mfem::ElementTransformation& Tr,
                                        const mfem::Array<const mfem::Vector*>& elfun,
                                        const mfem::Array<mfem::Vector*>& elvec );

    /// Assemble the local gradient matrix
    virtual void AssembleElementGrad( const mfem::Array<const mfem::FiniteElement*>& el,
                                      mfem::ElementTransformation& Tr,
                                      const mfem::Array<const mfem::Vector*>& elfun,
                                      const mfem::Array2D<mfem::DenseMatrix*>& elmats );

    // void setGeomStiff( const bool flg )
    // {
    //     mOnlyGeomStiff = flg;
    // }

    // bool onlyGeomStiff() const
    // {
    //     return mOnlyGeomStiff;
    // }

protected:
    PhaseFieldElasticMaterial* mMaterialModel{ nullptr };

    Eigen::Matrix<mfem::real_t, 3, 3> mdxdX;
    Eigen::Matrix<mfem::real_t, 6, Eigen::Dynamic> mB;
    // Eigen::MatrixXr mGeomStiff;
    IntegrationPointStorage& mPointStorage;
    // bool mOnlyGeomStiff{ false };

    // data for phase field
    mfem::Vector shape;
    mfem::DenseMatrix mDShape, mGShape;
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
