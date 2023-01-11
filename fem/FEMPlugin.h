
#pragma once
#include "Material.h"
#include <Eigen/Dense>
#include <memory>
#include <mfem.hpp>
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

    virtual ~NonlinearFormIntegratorLambda()
    {
    }

protected:
    mutable double mLambda;
};

class NonlinearFormMaterialIntegratorLambda : public NonlinearFormIntegratorLambda
{
public:
    NonlinearFormMaterialIntegratorLambda( ElasticMaterial& m ) : NonlinearFormIntegratorLambda(), mMaterialModel{ &m }
    {
    }

    virtual void SetLambda( const double lambda ) const
    {
        NonlinearFormIntegratorLambda::SetLambda( lambda );
        mMaterialModel->setLambda( lambda );
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

    virtual ~NonlinearFormMaterialIntegratorLambda()
    {
    }

protected:
    ElasticMaterial* mMaterialModel{ nullptr };
    bool mNonlinear{ true };
};

class NonlinearElasticityIntegrator : public NonlinearFormMaterialIntegratorLambda
{
public:
    NonlinearElasticityIntegrator( ElasticMaterial& m, Memorize& memo )
        : NonlinearFormMaterialIntegratorLambda( m ), mMemo{ memo }
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

    void matrixB( const int dof, const int dim, const Eigen::MatrixXd& gshape );

    void setGeomStiff( const bool flg )
    {
        mOnlyGeomStiff = flg;
    }

    bool onlyGeomStiff() const
    {
        return mOnlyGeomStiff;
    }

protected:
    Eigen::Matrix<double, 3, 3> mdxdX;
    Eigen::Matrix<double, 6, Eigen::Dynamic> mB;
    Eigen::MatrixXd mGeomStiff;
    Memorize& mMemo;
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

class NonlinearCompositeSolidShellIntegrator : public NonlinearFormMaterialIntegratorLambda
{
public:
    NonlinearCompositeSolidShellIntegrator( ElasticMaterial& m ) : NonlinearFormMaterialIntegratorLambda( m )
    {
        mL.resize( 5, 24 );
        mH.resize( 5, 5 );
        mAlpha.resize( 5 );
        mGeomStiff.resize( 24, 24 );
    }

    // virtual void AssembleElementVector( const mfem::FiniteElement& el,
    //                                     mfem::ElementTransformation& Ttr,
    //                                     const mfem::Vector& elfun,
    //                                     mfem::Vector& elvect );

    virtual void AssembleElementGrad( const mfem::FiniteElement& el,
                                      mfem::ElementTransformation& Ttr,
                                      const mfem::Vector& elfun,
                                      mfem::DenseMatrix& elmat );

    void matrixB( const int dof, const int dim, const mfem::IntegrationPoint& ip );

    /** @brief Computes the integral of W(Jacobian(Trt)) over a target zone
        @param[in] el     Type of FiniteElement.
        @param[in] Ttr    Represents ref->target coordinates transformation.
        @param[in] elfun  Physical coordinates of the zone. */
    virtual double GetElementEnergy( const mfem::FiniteElement& el, mfem::ElementTransformation& Ttr, const mfem::Vector& elfun )
    {
        return 0;
    }

protected:
    Eigen::Matrix<double, 3, 3> mg, mGCovariant, mGContravariant, mgA, mgB, mgC, mgD, mgA1, mgA2, mgA3, mgA4;
    Eigen::Matrix<double, 6, 24> mB;
    Eigen::MatrixXd mGeomStiff;
    Eigen::Matrix<double, 8, 3> mDShape, mDShapeA, mDShapeB, mDShapeC, mDShapeD, mDShapeA1, mDShapeA2, mDShapeA3, mDShapeA4;
    Eigen::Matrix6d mStiffModuli, mTransform;
    Eigen::MatrixXd mL, mH;
    Eigen::VectorXd mAlpha;
};

class CZMIntegrator : public mfem::NonlinearFormIntegrator
{
public:
    CZMIntegrator() : mfem::NonlinearFormIntegrator()
    {
    }

    CZMIntegrator( const double sigmaMax,
                   const double tauMax,
                   const double phiN,
                   const double phiT,
                   const double deltaN,
                   const double deltaT,
                   const double deltaNStar = 0 )
        : mfem::NonlinearFormIntegrator(), mSigmaMax{ sigmaMax }, mTauMax{ tauMax }, mPhiN{ phiN }, mPhiT{ phiT }, mDeltaN{ deltaN }, mDeltaT{ deltaT }, mDeltaNStar{ deltaNStar }
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
    double mSigmaMax{ 0. };
    double mTauMax{ 0. };
    double mPhiN{ 0. };
    double mPhiT{ 0. };
    double mDeltaN{ 0. };
    double mDeltaT{ 0. };
    double mDeltaNStar{ 0. };
    mfem::Vector shape1, shape2;

    Eigen::MatrixXd mB;
    Eigen::VectorXd u;
};

class NonlinearDirichletPenaltyIntegrator : public NonlinearFormIntegratorLambda
{
public:
    NonlinearDirichletPenaltyIntegrator( mfem::VectorCoefficient& QG ) : NonlinearFormIntegratorLambda(), Q( QG )
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
    Eigen::MatrixXd mB;
    mfem::VectorCoefficient& Q;
};

} // namespace plugin
