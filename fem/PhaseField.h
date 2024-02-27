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
class PhaseFieldIntegrator : public BlockNonlinearFormIntegratorLambda
{
public:
    PhaseFieldIntegrator( PhaseFieldElasticMaterial& m, Memorize& memo )
        : BlockNonlinearFormIntegratorLambda(), mMaterialModel( &m ), mMemo{memo}
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
    PhaseFieldElasticMaterial* mMaterialModel{nullptr};

    Eigen::Matrix<double, 3, 3> mdxdX;
    Eigen::Matrix<double, 6, Eigen::Dynamic> mB;
    // Eigen::MatrixXd mGeomStiff;
    Memorize& mMemo;
    // bool mOnlyGeomStiff{ false };

    // data for phase field
    mfem::Vector shape;
    mfem::DenseMatrix mDShape, mGShape;
};

class BlockNonlinearDirichletPenaltyIntegrator : public BlockNonlinearFormIntegratorLambda
{
public:
    BlockNonlinearDirichletPenaltyIntegrator( mfem::VectorCoefficient& QG, mfem::VectorCoefficient& HG )
        : BlockNonlinearFormIntegratorLambda(), mIntegrator( QG, HG )
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

    virtual void SetLambda( const double lambda ) const override
    {
        mLambda = lambda;
        mIntegrator.SetLambda( lambda );
    }

protected:
    NonlinearDirichletPenaltyIntegrator mIntegrator;
};
} // namespace plugin