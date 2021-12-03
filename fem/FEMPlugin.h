
#pragma once
#include "Material.h"
#include "mfem.hpp"
#include <memory>
#include <vector>

namespace plugin
{
class ElasticityIntegrator : public mfem::BilinearFormIntegrator
{
public:
    ElasticityIntegrator( ElasticMaterial& m ) : BilinearFormIntegrator()
    {
        mMaterialModel = &m;
    }
    void AssembleElementMatrix( const mfem::FiniteElement& el, mfem::ElementTransformation& Trans, mfem::DenseMatrix& elmat );

    static void resizeRefEleTransVec( const size_t size );

    void matrixB( const int dof, const int dim, const mfem::DenseMatrix& gshape, Eigen::Matrix<double, 6, Eigen::Dynamic>& B ) const;

    void updateDeformationGradient( const int dim,
                                    mfem::ElementTransformation& ref,
                                    mfem::ElementTransformation& cur,
                                    const mfem::IntegrationPoint& ip );

    ~ElasticityIntegrator();

protected:
    mfem::DenseMatrix mDShape, mGShape;

    Eigen::Matrix<double, 3, 3> mdxdX;

    static std::vector<std::unique_ptr<mfem::IsoparametricTransformation>> refEleTransVec;

    ElasticMaterial* mMaterialModel{ nullptr };
};

class NonlinearElasticityIntegrator : public mfem::NonlinearFormIntegrator
{
public:
    NonlinearElasticityIntegrator( ElasticMaterial& m ) : mfem::NonlinearFormIntegrator(), mMaterialModel{ &m }
    {
    }

    /** @brief Computes the integral of W(Jacobian(Trt)) over a target zone
        @param[in] el     Type of FiniteElement.
        @param[in] Ttr    Represents ref->target coordinates transformation.
        @param[in] elfun  Physical coordinates of the zone. */
    virtual double GetElementEnergy( const mfem::FiniteElement& el, mfem::ElementTransformation& Ttr, const mfem::Vector& elfun )
    {
        return 0;
    }

    virtual void AssembleElementVector( const mfem::FiniteElement& el,
                                        mfem::ElementTransformation& Ttr,
                                        const mfem::Vector& elfun,
                                        mfem::Vector& elvect );

    virtual void AssembleElementGrad( const mfem::FiniteElement& el,
                                      mfem::ElementTransformation& Ttr,
                                      const mfem::Vector& elfun,
                                      mfem::DenseMatrix& elmat );

    void matrixB( const int dof, const int dim, const mfem::DenseMatrix& gshape );

protected:
    mfem::DenseMatrix mDShape, mGShape;

    Eigen::Matrix<double, 3, 3> mdxdX;

    Eigen::Matrix<double, 6, Eigen::Dynamic> mB;

    ElasticMaterial* mMaterialModel{ nullptr };
};
} // namespace plugin