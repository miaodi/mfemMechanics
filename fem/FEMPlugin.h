
#pragma once
#include "Material.h"
#include "util.h"
#include <Eigen/Dense>
#include <memory>
#include <mfem.hpp>
#include <vector>

namespace plugin
{
class IterAuxilliary;

Eigen::MatrixXd mapper( const int dim, const int dof );

void smallDeformMatrixB( const int, const int, const Eigen::MatrixXd&, Eigen::Matrix<double, 6, Eigen::Dynamic>& );

void largeDeformMatrixB( const int, const int, const Eigen::MatrixXd&, const Eigen::MatrixXd&, Eigen::Matrix<double, 6, Eigen::Dynamic>& );

struct GaussPointStorage
{
    Eigen::MatrixXd GShape;
    double DetdXdXi;
    util::AnyMap PointData;
};

struct CZMGaussPointStorage
{
    double Weight{ 0. };

    mfem::Vector Shape1, Shape2;
    mfem::DenseMatrix GShapeFace1, GShapeFace2;
    mfem::DenseMatrix Jacobian;
    util::AnyMap PointData;
};

// TODO: should rewrite Memorize class so that Initialize can be registered by integrators.
class Memorize
{
public:
    Memorize( mfem::Mesh* );

    void InitializeElement( const mfem::FiniteElement&, mfem::ElementTransformation&, const mfem::IntegrationRule& );

    void InitializeFace( const mfem::FiniteElement&, const mfem::FiniteElement&, mfem::FaceElementTransformations&, const mfem::IntegrationRule& );

    const Eigen::MatrixXd& GetdNdX( const int gauss ) const;

    const mfem::Vector& GetFace1Shape( const int gauss ) const;

    const mfem::Vector& GetFace2Shape( const int gauss ) const;

    const mfem::DenseMatrix& GetFace1GShape( const int gauss ) const;

    const mfem::DenseMatrix& GetFace2GShape( const int gauss ) const;

    double GetDetdXdXi( const int gauss ) const;

    double GetFaceWeight( const int gauss ) const;

    const mfem::DenseMatrix& GetFaceJacobian( const int gauss ) const;

    void Reset( mfem::Mesh* m )
    {
        mEleStorage = std::vector<std::unique_ptr<std::vector<GaussPointStorage>>>( m->GetNE() );
        mFaceStorage = std::vector<std::unique_ptr<std::vector<CZMGaussPointStorage>>>( m->GetNumFaces() );
    }

    const CZMGaussPointStorage& GetFacePointStorage( const int gauss ) const
    {
        return ( *mFaceStorage[mElementNo] )[gauss];
    }

    CZMGaussPointStorage& GetFacePointStorage( const int gauss )
    {
        return ( *mFaceStorage[mElementNo] )[gauss];
    }

    const util::AnyMap& GetBodyPointData( const int gauss ) const
    {
        return ( *mEleStorage[mElementNo] )[gauss].PointData;
    }

    util::AnyMap& GetBodyPointData( const int gauss )
    {
        return ( *mEleStorage[mElementNo] )[gauss].PointData;
    }

    const util::AnyMap& GetFacePointData( const int gauss ) const
    {
        return ( *mFaceStorage[mElementNo] )[gauss].PointData;
    }

    util::AnyMap& GetFacePointData( const int gauss )
    {
        return ( *mFaceStorage[mElementNo] )[gauss].PointData;
    }

private:
    std::vector<std::unique_ptr<std::vector<GaussPointStorage>>> mEleStorage;
    std::vector<std::unique_ptr<std::vector<CZMGaussPointStorage>>> mFaceStorage;
    mfem::DenseMatrix mDShape1, mDShape2, mGShape1, mGShape2;
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
    NonlinearFormIntegratorLambda() : mfem::NonlinearFormIntegrator()
    {
    }

    virtual ~NonlinearFormIntegratorLambda()
    {
    }

    void SetIterAux( IterAuxilliary const* ptr )
    {
        mIterAux = ptr;
    }

protected:
    IterAuxilliary const* mIterAux{ nullptr };
};

class NonlinearElasticityIntegrator : public NonlinearFormIntegratorLambda
{
public:
    NonlinearElasticityIntegrator( ElasticMaterial& m, Memorize& memo ) : mMaterialModel( &m ), mMemo{ memo }
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

    void setGeomStiff( const bool flg )
    {
        mOnlyGeomStiff = flg;
    }

    bool onlyGeomStiff() const
    {
        return mOnlyGeomStiff;
    }

    void setNonlinear( const bool flg )
    {
        mNonlinear = flg;
        mMaterialModel->setLargeDeformation( flg );
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
    bool mOnlyGeomStiff{ false };
    bool mNonlinear{ true };
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

class NonlinearDirichletPenaltyIntegrator : public NonlinearFormIntegratorLambda
{
public:
    NonlinearDirichletPenaltyIntegrator( mfem::VectorCoefficient& QG, mfem::VectorCoefficient& HG )
        : NonlinearFormIntegratorLambda(), Q( QG ), H( HG )
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

    void matrixB( const int dof, const int dim )
    {
        mB.resize( dim, dim * dof );
        mB.setZero();

        for ( int i = 0; i < dof; i++ )
        {
            for ( int j = 0; j < dim; j++ )
            {
                mB( j, i + j * dof ) = shape( i );
            }
        }
    }

protected:
    mfem::Vector shape, dispEval, penalEval;
    Eigen::MatrixXd mB;
    Eigen::VectorXd mU;
    mfem::VectorCoefficient& Q;
    mfem::VectorCoefficient& H;
};

class NonlinearInternalPenaltyIntegrator : public mfem::NonlinearFormIntegrator
{
public:
    NonlinearInternalPenaltyIntegrator( const double penalty = 1e10 ) : mfem::NonlinearFormIntegrator(), p{ penalty }
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

    void matrixB( const int dof1, const int dof2, const int dim )
    {
        mB.resize( dim, dim * ( dof1 + dof2 ) );
        mB.setZero();

        for ( int i = 0; i < dof1; i++ )
        {
            for ( int j = 0; j < dim; j++ )
            {
                mB( j, i + j * dof1 ) = shape1( i );
            }
        }
        for ( int i = 0; i < dof2; i++ )
        {
            for ( int j = 0; j < dim; j++ )
            {
                mB( j, i + j * dof2 + dim * dof1 ) = -shape2( i );
            }
        }
    }

protected:
    mfem::Vector shape1, shape2;

    Eigen::MatrixXd mB;
    Eigen::VectorXd u;
    double p;
};

class BlockNonlinearFormIntegratorLambda : public mfem::BlockNonlinearFormIntegrator
{
public:
    BlockNonlinearFormIntegratorLambda() : mfem::BlockNonlinearFormIntegrator(), mLambda{ 1. }
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

    virtual void SetIterAux( IterAuxilliary const* ptr )
    {
        mIterAux = ptr;
    }

protected:
    mutable double mLambda;
    IterAuxilliary const* mIterAux{ nullptr };
};
} // namespace plugin
