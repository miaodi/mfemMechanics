
#pragma once
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
class IterAuxilliary;

Eigen::MatrixXr mapper( const int dim, const int dof );

void smallDeformMatrixB( const int, const int, const Eigen::MatrixXr&, Eigen::Matrix<mfem::real_t, 6, Eigen::Dynamic>& );

void largeDeformMatrixB( const int, const int, const Eigen::MatrixXr&, const Eigen::MatrixXr&, Eigen::Matrix<mfem::real_t, 6, Eigen::Dynamic>& );

struct GaussPointStorage
{
    Eigen::MatrixXr GShape;
    mfem::real_t DetdXdXi{ 0. };
    util::AnyMap PointData;
};

struct CZMGaussPointStorage
{
    mfem::real_t Weight{ 0. };

    mfem::Vector Shape1, Shape2;
    mfem::DenseMatrix GShapeFace1, GShapeFace2;
    mfem::DenseMatrix Jacobian;
    util::AnyMap PointData;
};

// TODO: should rewrite IntegrationPointStorage class so that Initialize can be registered by integrators.
class IntegrationPointStorage
{
    struct ElementPointSet
    {
        const mfem::FiniteElement* Element{ nullptr };
        const mfem::IntegrationRule* Rule{ nullptr };
        std::vector<GaussPointStorage> Points;
        int Dof{ 0 };
        int Dimension{ 0 };

        bool IsInitialized() const noexcept
        {
            return Element != nullptr;
        }
    };

    struct FacePointSet
    {
        const mfem::FiniteElement* Element1{ nullptr };
        const mfem::FiniteElement* Element2{ nullptr };
        const mfem::IntegrationRule* Rule{ nullptr };
        std::vector<CZMGaussPointStorage> Points;
        int Dof1{ 0 };
        int Dof2{ 0 };
        int Dimension{ 0 };
        int FaceDimension{ 0 };
    };

    struct PointSetBuildWorkspace
    {
        // These matrices must remain owning. Use scoped views for point-storage output.
        mfem::DenseMatrix ReferenceGradient1, ReferenceGradient2;
        mfem::DenseMatrix PhysicalGradient1, PhysicalGradient2;
    };

public:
    IntegrationPointStorage( mfem::Mesh* );

    void InitializeElement( const mfem::FiniteElement&, mfem::ElementTransformation&, const mfem::IntegrationRule& );

    void InitializeFace( const mfem::FiniteElement&, const mfem::FiniteElement&, mfem::FaceElementTransformations&, const mfem::IntegrationRule& );

    const Eigen::MatrixXr& GetdNdX( const int gauss ) const;

    const mfem::Vector& GetFace1Shape( const int gauss ) const;

    const mfem::Vector& GetFace2Shape( const int gauss ) const;

    const mfem::DenseMatrix& GetFace1GShape( const int gauss ) const;

    const mfem::DenseMatrix& GetFace2GShape( const int gauss ) const;

    mfem::real_t GetDetdXdXi( const int gauss ) const;

    mfem::real_t GetFaceWeight( const int gauss ) const;

    const mfem::DenseMatrix& GetFaceJacobian( const int gauss ) const;

    void Reset( mfem::Mesh* m );

    template <typename Visitor>
    void VisitFacePointData( Visitor&& visitor )
    {
        for ( auto& face : mFaceStorage )
        {
            if ( !face )
            {
                continue;
            }

            for ( auto& point : face->Points )
            {
                visitor( point.PointData );
            }
        }
    }

    const CZMGaussPointStorage& GetFacePointStorage( const int gauss ) const;

    CZMGaussPointStorage& GetFacePointStorage( const int gauss );

    const util::AnyMap& GetBodyPointData( const int gauss ) const;

    util::AnyMap& GetBodyPointData( const int gauss );

    const util::AnyMap& GetFacePointData( const int gauss ) const;

    util::AnyMap& GetFacePointData( const int gauss );

private:
    ElementPointSet BuildElementPointSet( const mfem::FiniteElement&, mfem::ElementTransformation&, const mfem::IntegrationRule& );
    FacePointSet BuildFacePointSet( const mfem::FiniteElement&,
                                    const mfem::FiniteElement&,
                                    mfem::FaceElementTransformations&,
                                    const mfem::IntegrationRule& );

    static void VerifyElementPointSet( const ElementPointSet&, const mfem::FiniteElement&, const mfem::IntegrationRule&, int );
    static void VerifyFacePointSet( const FacePointSet&,
                                    const mfem::FiniteElement&,
                                    const mfem::FiniteElement&,
                                    const mfem::FaceElementTransformations&,
                                    const mfem::IntegrationRule&,
                                    int );

    const ElementPointSet& CurrentElementPointSet() const;
    ElementPointSet& CurrentElementPointSet();
    const FacePointSet& CurrentFacePointSet() const;
    FacePointSet& CurrentFacePointSet();

    std::vector<ElementPointSet> mElementStorage;
    std::vector<std::unique_ptr<FacePointSet>> mFaceStorage;
    PointSetBuildWorkspace mBuildWorkspace;
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

    void matrixB( const int dof, const int dim, const mfem::DenseMatrix& gshape, Eigen::Matrix<mfem::real_t, 6, Eigen::Dynamic>& B ) const;

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

    virtual void BeginStep()
    {
        MFEM_VERIFY( mIterAuxStackSize < mIterAuxStack.size(), "Integrator nesting exceeds the supported depth." );
        mIterAuxStack[mIterAuxStackSize++] = mIterAux;
    }

    virtual void CommitStep()
    {
        RestoreIterAux();
    }

    virtual void RollbackStep()
    {
        RestoreIterAux();
    }

    virtual void RevertStep()
    {
    }

protected:
    void RestoreIterAux()
    {
        MFEM_VERIFY( mIterAuxStackSize > 0, "Integrator step completion requires a matching BeginStep." );
        mIterAuxStackSize--;
        if ( mIterAuxStackSize > 0 )
        {
            mIterAux = mIterAuxStack[mIterAuxStackSize - 1];
        }
    }

    IterAuxilliary const* mIterAux{ nullptr };
    std::array<IterAuxilliary const*, 32> mIterAuxStack{};
    std::size_t mIterAuxStackSize{ 0 };
};

class NonlinearElasticityIntegrator : public NonlinearFormIntegratorLambda
{
public:
    NonlinearElasticityIntegrator( ElasticMaterial& m, IntegrationPointStorage& pointStorage )
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
    IntegrationPointStorage& mPointStorage;
    StressFreeDeformationModel mStressFreeDeformations;
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
    Eigen::MatrixXr mdxdX;
    mfem::DenseMatrix mDShape, mGShape;
    Eigen::MatrixXr mB;
    mfem::Coefficient& Q;
};

class NonlinearCompositeSolidShellIntegrator : public NonlinearFormIntegratorLambda
{
public:
    NonlinearCompositeSolidShellIntegrator( ElasticMaterial& m ) : NonlinearFormIntegratorLambda(), mMaterialModel{ &m }
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

class BlockNonlinearFormIntegratorLambda : public mfem::BlockNonlinearFormIntegrator
{
public:
    BlockNonlinearFormIntegratorLambda() : mfem::BlockNonlinearFormIntegrator()
    {
    }

    virtual ~BlockNonlinearFormIntegratorLambda()
    {
    }

    virtual void SetIterAux( IterAuxilliary const* ptr )
    {
        mIterAux = ptr;
    }

    virtual void BeginStep()
    {
        MFEM_VERIFY( mIterAuxStackSize < mIterAuxStack.size(), "Integrator nesting exceeds the supported depth." );
        mIterAuxStack[mIterAuxStackSize++] = mIterAux;
    }

    virtual void CommitStep()
    {
        RestoreIterAux();
    }

    virtual void RollbackStep()
    {
        RestoreIterAux();
    }

    virtual void RevertStep()
    {
    }

protected:
    void RestoreIterAux()
    {
        MFEM_VERIFY( mIterAuxStackSize > 0, "Integrator step completion requires a matching BeginStep." );
        mIterAuxStackSize--;
        if ( mIterAuxStackSize > 0 )
        {
            mIterAux = mIterAuxStack[mIterAuxStackSize - 1];
        }
    }

    IterAuxilliary const* mIterAux{ nullptr };
    std::array<IterAuxilliary const*, 32> mIterAuxStack{};
    std::size_t mIterAuxStackSize{ 0 };
};

class TempDependentNonlinearElasticityIntegrator : public BlockNonlinearFormIntegratorLambda, public NonlinearElasticityIntegrator
{
private:
    const mfem::Array2D<mfem::DenseMatrix*>* mElmats;

public:
    TempDependentNonlinearElasticityIntegrator( ElasticMaterial& m, IntegrationPointStorage& pointStorage )
        : BlockNonlinearFormIntegratorLambda(), NonlinearElasticityIntegrator( m, pointStorage )
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

    virtual void SetIterAux( IterAuxilliary const* ptr )
    {
        BlockNonlinearFormIntegratorLambda::SetIterAux( ptr );
        NonlinearElasticityIntegrator::SetIterAux( ptr );
    }
};
} // namespace plugin
