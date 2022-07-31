
#pragma once
#include "Material.h"
#include <mfem.hpp>
#include <Eigen/Dense>
#include <memory>
#include <vector>

namespace plugin
{
Eigen::MatrixXd mapper( const int dim, const int dof );

void smallDeformMatrixB( const int, const int, const Eigen::MatrixXd&, Eigen::Matrix<double, 6, Eigen::Dynamic>& );

struct GaussPointStorage
{
    Eigen::MatrixXd GShape;
    double DetdXdXi;
};

class Memorize
{
public:
    Memorize( mfem::Mesh* );

    void InitializeElement( const mfem::FiniteElement&, mfem::ElementTransformation&, const mfem::IntegrationRule& );

    const Eigen::MatrixXd& GetdNdX( const int gauss ) const;

    double GetDetdXdXi( const int gauss ) const;

private:
    std::vector<std::unique_ptr<std::vector<GaussPointStorage>>> mStorage;
    mfem::DenseMatrix mDShape, mGShape;
    int mElementNo{ 0 };
};

class ElasticityIntegrator : public mfem::BilinearFormIntegrator
{
public:
    ElasticityIntegrator( ElasticMaterial& m ) : BilinearFormIntegrator()
    {
        mMaterialModel = &m;
    }
    void AssembleElementMatrix( const mfem::FiniteElement& el, mfem::ElementTransformation& Trans, mfem::DenseMatrix& elmat );

    void matrixB( const int dof, const int dim, const mfem::DenseMatrix& gshape, Eigen::Matrix<double, 6, Eigen::Dynamic>& B ) const;

protected:
    mfem::DenseMatrix mDShape, mGShape;

    ElasticMaterial* mMaterialModel{ nullptr };
};

class NonlinearFormIntegratorLambda : public mfem::NonlinearFormIntegrator
{
public:
    NonlinearFormIntegratorLambda() : mfem::NonlinearFormIntegrator(), mLambda{ 1. }
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

protected:
    mutable double mLambda;
};

class NonlinearElasticityIntegrator : public NonlinearFormIntegratorLambda
{
public:
    NonlinearElasticityIntegrator( ElasticMaterial& m, Memorize& memo )
        : NonlinearFormIntegratorLambda(), mMaterialModel{ &m }, mMemo{ memo }
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

    virtual void SetLambda( const double lambda ) const
    {
        NonlinearFormIntegratorLambda::SetLambda( lambda );
        mMaterialModel->setLambda( lambda );
    }

    void matrixB( const int dof, const int dim, const Eigen::MatrixXd& gshape );

    void setNonlinear( const bool flg )
    {
        mNonlinear = flg;
        mMaterialModel->setLargeDeformation( flg );
    }

    void setGeomStiff( const bool flg )
    {
        mOnlyGeomStiff = flg;
    }

    bool onlyGeomStiff() const
    {
        return mOnlyGeomStiff;
    }

    bool isNonlinear() const
    {
        return mNonlinear;
    }

protected:
    Eigen::Matrix<double, 3, 3> mdxdX;
    Eigen::Matrix<double, 6, Eigen::Dynamic> mB;
    Eigen::MatrixXd mGeomStiff;
    ElasticMaterial* mMaterialModel{ nullptr };
    Memorize& mMemo;

    bool mNonlinear{ true };
    bool mOnlyGeomStiff{ false };
};

class NonlinearVectorBoundaryLFIntegrator : public NonlinearFormIntegratorLambda
{
public:
    NonlinearVectorBoundaryLFIntegrator( mfem::VectorCoefficient& QG ) : NonlinearFormIntegratorLambda(), Q( QG )
    {
    }

    virtual void AssembleFaceVector( const mfem::FiniteElement& el1,
                                     const mfem::FiniteElement& el2,
                                     mfem::FaceElementTransformations& Tr,
                                     const mfem::Vector& elfun,
                                     mfem::Vector& elvect ) override;

    virtual void AssembleFaceGrad( const mfem::FiniteElement& el1,
                                   const mfem::FiniteElement& el2,
                                   mfem::FaceElementTransformations& Tr,
                                   const mfem::Vector& elfun,
                                   mfem::DenseMatrix& elmat ) override;

protected:
    mfem::Vector shape, vec;
    mfem::VectorCoefficient& Q;
};

class NonlinearPressureIntegrator : public NonlinearFormIntegratorLambda
{
public:
    NonlinearPressureIntegrator( mfem::Coefficient& QG ) : NonlinearFormIntegratorLambda(), Q( QG )
    {
    }

    virtual void AssembleFaceVector( const mfem::FiniteElement& el1,
                                     const mfem::FiniteElement& el2,
                                     mfem::FaceElementTransformations& Tr,
                                     const mfem::Vector& elfun,
                                     mfem::Vector& elvect ) override;

    virtual void AssembleFaceGrad( const mfem::FiniteElement& el1,
                                   const mfem::FiniteElement& el2,
                                   mfem::FaceElementTransformations& Tr,
                                   const mfem::Vector& elfun,
                                   mfem::DenseMatrix& elmat ) override;

protected:
    mfem::Vector shape;
    Eigen::MatrixXd mdxdX;
    mfem::DenseMatrix mDShape, mGShape;
    Eigen::MatrixXd mB;
    mfem::Coefficient& Q;
};
} // namespace plugin