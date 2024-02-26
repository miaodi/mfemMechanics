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

class BlockNonlinearFormIntegratorLambda : public mfem::BlockNonlinearFormIntegrator
{
public:
    BlockNonlinearFormIntegratorLambda( PhaseFieldElasticMaterial& m )
        : mfem::BlockNonlinearFormIntegrator(), mLambda{ 1. }, mMaterialModel{ &m }
    {
    }

    virtual void SetLambda( const double lambda ) const
    {
        mLambda = lambda;
    }

    double GetLambda() const
    {
        return mLambda;
    }

    virtual ~BlockNonlinearFormIntegratorLambda()
    {
    }

    void SetIterAux( IterAuxilliary const* ptr )
    {
        mIterAux = ptr;
    }

protected:
    mutable double mLambda;
    IterAuxilliary const* mIterAux{ nullptr };

    PhaseFieldElasticMaterial* mMaterialModel{ nullptr };
};

class PhaseFieldIntegrator : public BlockNonlinearFormIntegratorLambda
{
public:
    PhaseFieldIntegrator( PhaseFieldElasticMaterial& m, Memorize& memo )
        : BlockNonlinearFormIntegratorLambda( m ), mMemo{ memo }
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
    Eigen::Matrix<double, 3, 3> mdxdX;
    Eigen::Matrix<double, 6, Eigen::Dynamic> mB;
    // Eigen::MatrixXd mGeomStiff;
    Memorize& mMemo;
    // bool mOnlyGeomStiff{ false };

    // data for phase field
    mfem::Vector shape;
    mfem::DenseMatrix mDShape, mGShape;
};
} // namespace plugin