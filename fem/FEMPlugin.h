
#pragma once
#include "IntegrationPointStorage.h"
#include "Material.h"
#include "StressFreeDeformation.h"
#include "util.h"
#include <Eigen/Dense>
#include <array>
#include <functional>
#include <memory>
#include <mfem.hpp>
#include <vector>

namespace plugin
{
class NonlinearStepContext;

Eigen::MatrixXr mapper( const int dim, const int dof );

void smallDeformMatrixB( const int, const int, const Eigen::MatrixXr&, Eigen::Matrix<mfem::real_t, 6, Eigen::Dynamic>& );

void largeDeformMatrixB( const int, const int, const Eigen::MatrixXr&, const Eigen::MatrixXr&, Eigen::Matrix<mfem::real_t, 6, Eigen::Dynamic>& );

class ElasticityIntegrator : public mfem::BilinearFormIntegrator
{
public:
    ElasticityIntegrator( ElasticMaterial& m ) : BilinearFormIntegrator()
    {
        mMaterialModel = &m;
    }
    void AssembleElementMatrix( const mfem::FiniteElement& el, mfem::ElementTransformation& Trans, mfem::DenseMatrix& elmat );

    void matrixB( const int dof, const int dim, const mfem::DenseMatrix& gshape, Eigen::Matrix<mfem::real_t, 6, Eigen::Dynamic>& B ) const;

protected:
    mfem::DenseMatrix mDShape, mGShape;

    ElasticMaterial* mMaterialModel{ nullptr };
};

class StepAwareNonlinearFormIntegrator : public mfem::NonlinearFormIntegrator
{
public:
    StepAwareNonlinearFormIntegrator() : mfem::NonlinearFormIntegrator()
    {
    }

    virtual ~StepAwareNonlinearFormIntegrator()
    {
    }

    void SetStepContext( NonlinearStepContext const* ptr )
    {
        mStepContext = ptr;
    }

    virtual void BeginStep()
    {
        MFEM_VERIFY( mStepContextStackSize < mStepContextStack.size(),
                     "Integrator nesting exceeds the supported depth." );
        mStepContextStack[mStepContextStackSize++] = mStepContext;
    }

    virtual void CommitStep()
    {
        RestoreStepContext();
    }

    virtual void RollbackStep()
    {
        RestoreStepContext();
    }

    virtual void RevertStep()
    {
    }

protected:
    void RestoreStepContext()
    {
        MFEM_VERIFY( mStepContextStackSize > 0, "Integrator step completion requires a matching BeginStep." );
        mStepContextStackSize--;
        if ( mStepContextStackSize > 0 )
        {
            mStepContext = mStepContextStack[mStepContextStackSize - 1];
        }
    }

    NonlinearStepContext const* mStepContext{ nullptr };
    std::array<NonlinearStepContext const*, 32> mStepContextStack{};
    std::size_t mStepContextStackSize{ 0 };
};

class NonlinearElasticityIntegrator : public StepAwareNonlinearFormIntegrator
{
public:
    NonlinearElasticityIntegrator( ElasticMaterial& m, IntegrationPointStorageBase& pointStorage )
        : mMaterialModel( &m ), mPointStorage{ pointStorage }
    {
        mMaterialModel->setLargeDeformation( mNonlinear );
    }

    /** @brief Computes the integral of W(Jacobian(Trt)) over a target zone
        @param[in] el     Type of FiniteElement.
        @param[in] Ttr    Represents ref->target coordinates transformation.
        @param[in] elfun  Physical coordinates of the zone. */
    mfem::real_t GetElementEnergy( const mfem::FiniteElement& el, mfem::ElementTransformation& Ttr, const mfem::Vector& elfun ) override
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

    void AddStressFreeDeformation( StressFreeDeformation& deformation )
    {
        mStressFreeDeformations.Add( deformation );
    }

    void ClearStressFreeDeformations()
    {
        mStressFreeDeformations.Clear();
    }

protected:
    struct AssemblyKinematics
    {
        std::reference_wrapper<const Eigen::MatrixXr> ShapeGradient;
        std::reference_wrapper<const Eigen::Matrix3r> DeformationGradient;
        mfem::real_t VolumeScale;
    };

    AssemblyKinematics PrepareMaterialPoint( const Eigen::Matrix3r& deformationGradient,
                                             const Eigen::MatrixXr& shapeGradient,
                                             int dimension,
                                             mfem::ElementTransformation& transformation,
                                             const mfem::IntegrationPoint& integrationPoint,
                                             mfem::real_t loadFactor );

    Eigen::Matrix<mfem::real_t, 3, 3> mdxdX;
    Eigen::Matrix<mfem::real_t, 6, Eigen::Dynamic> mB;
    Eigen::MatrixXr mGeomStiff;
    Eigen::MatrixXr mElasticShapeGradient;
    Eigen::Matrix3r mElasticDeformationGradient;
    Eigen::Matrix3r mInverseStressFreeDeformationGradient;
    Eigen::Matrix3r mMechanicalStrain;
    Eigen::Matrix3r mStressFreeDeformationGradient;
    ElasticMaterial* mMaterialModel{ nullptr };
    IntegrationPointStorageBase& mPointStorage;
    StressFreeDeformationModel mStressFreeDeformations;
    bool mOnlyGeomStiff{ false };
    bool mNonlinear{ true };
};

class NonlinearVectorBoundaryLFIntegrator : public StepAwareNonlinearFormIntegrator
{
public:
    NonlinearVectorBoundaryLFIntegrator( mfem::VectorCoefficient& QG ) : StepAwareNonlinearFormIntegrator(), Q( QG )
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

class NonlinearPressureIntegrator : public StepAwareNonlinearFormIntegrator
{
public:
    NonlinearPressureIntegrator( mfem::Coefficient& QG ) : StepAwareNonlinearFormIntegrator(), Q( QG )
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
    Eigen::MatrixXr mdxdX;
    mfem::DenseMatrix mDShape, mGShape;
    Eigen::MatrixXr mB;
    mfem::Coefficient& Q;
};

class NonlinearCompositeSolidShellIntegrator : public StepAwareNonlinearFormIntegrator
{
public:
    NonlinearCompositeSolidShellIntegrator( ElasticMaterial& m )
        : StepAwareNonlinearFormIntegrator(), mMaterialModel{ &m }
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
    mfem::real_t GetElementEnergy( const mfem::FiniteElement& el, mfem::ElementTransformation& Ttr, const mfem::Vector& elfun ) override
    {
        return 0;
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
    ElasticMaterial* mMaterialModel{ nullptr };
    Eigen::Matrix<mfem::real_t, 3, 3> mg, mGCovariant, mGContravariant, mgA, mgB, mgC, mgD, mgA1, mgA2, mgA3, mgA4;
    Eigen::Matrix<mfem::real_t, 6, 24> mB;
    Eigen::MatrixXr mGeomStiff;
    Eigen::Matrix<mfem::real_t, 8, 3> mDShape, mDShapeA, mDShapeB, mDShapeC, mDShapeD, mDShapeA1, mDShapeA2, mDShapeA3, mDShapeA4;
    Eigen::Matrix6r mStiffModuli, mTransform;
    Eigen::MatrixXr mL, mH;
    Eigen::VectorXr mAlpha;
    bool mNonlinear{ true };
};

class NonlinearDirichletPenaltyIntegrator : public StepAwareNonlinearFormIntegrator
{
public:
    NonlinearDirichletPenaltyIntegrator( mfem::VectorCoefficient& QG, mfem::VectorCoefficient& HG )
        : StepAwareNonlinearFormIntegrator(), Q( QG ), H( HG )
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
    Eigen::MatrixXr mB;
    Eigen::VectorXr mU;
    mfem::VectorCoefficient& Q;
    mfem::VectorCoefficient& H;
};

class NonlinearInternalPenaltyIntegrator : public mfem::NonlinearFormIntegrator
{
public:
    NonlinearInternalPenaltyIntegrator( const mfem::real_t penalty = 1e10 )
        : mfem::NonlinearFormIntegrator(), p{ penalty }
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

    Eigen::MatrixXr mB;
    Eigen::VectorXr u;
    mfem::real_t p;
};

class BlockStepAwareNonlinearFormIntegrator : public mfem::BlockNonlinearFormIntegrator
{
public:
    BlockStepAwareNonlinearFormIntegrator() : mfem::BlockNonlinearFormIntegrator()
    {
    }

    virtual ~BlockStepAwareNonlinearFormIntegrator()
    {
    }

    virtual void SetStepContext( NonlinearStepContext const* ptr )
    {
        mStepContext = ptr;
    }

    virtual void BeginStep()
    {
        MFEM_VERIFY( mStepContextStackSize < mStepContextStack.size(),
                     "Integrator nesting exceeds the supported depth." );
        mStepContextStack[mStepContextStackSize++] = mStepContext;
    }

    virtual void CommitStep()
    {
        RestoreStepContext();
    }

    virtual void RollbackStep()
    {
        RestoreStepContext();
    }

    virtual void RevertStep()
    {
    }

protected:
    void RestoreStepContext()
    {
        MFEM_VERIFY( mStepContextStackSize > 0, "Integrator step completion requires a matching BeginStep." );
        mStepContextStackSize--;
        if ( mStepContextStackSize > 0 )
        {
            mStepContext = mStepContextStack[mStepContextStackSize - 1];
        }
    }

    NonlinearStepContext const* mStepContext{ nullptr };
    std::array<NonlinearStepContext const*, 32> mStepContextStack{};
    std::size_t mStepContextStackSize{ 0 };
};

class TempDependentNonlinearElasticityIntegrator : public BlockStepAwareNonlinearFormIntegrator, public NonlinearElasticityIntegrator
{
private:
    const mfem::Array2D<mfem::DenseMatrix*>* mElmats;

public:
    TempDependentNonlinearElasticityIntegrator( ElasticMaterial& m, IntegrationPointStorageBase& pointStorage )
        : BlockStepAwareNonlinearFormIntegrator(), NonlinearElasticityIntegrator( m, pointStorage )
    {
    }

    // virtual double GetElementEnergy( const Array<const FiniteElement*>& el, ElementTransformation& Tr, const Array<const Vector*>& elfun );

    // /// Perform the local action of the NonlinearFormIntegrator
    // virtual void AssembleElementVector( const Array<const FiniteElement*>& el,
    //                                     ElementTransformation& Tr,
    //                                     const Array<const Vector*>& elfun,
    //                                     const Array<Vector*>& elvec );

    /// Assemble the local gradient matrix
    virtual void AssembleElementGrad( const mfem::Array<const mfem::FiniteElement*>& el,
                                      mfem::ElementTransformation& Tr,
                                      const mfem::Array<const mfem::Vector*>& elfun,
                                      const mfem::Array2D<mfem::DenseMatrix*>& elmats );

    virtual void SetStepContext( NonlinearStepContext const* ptr )
    {
        BlockStepAwareNonlinearFormIntegrator::SetStepContext( ptr );
        NonlinearElasticityIntegrator::SetStepContext( ptr );
    }
};
} // namespace plugin
