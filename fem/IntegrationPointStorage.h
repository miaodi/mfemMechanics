#pragma once

#include "MaterialPointState.h"
#include "typeDef.h"
#include <Eigen/Dense>
#include <memory>
#include <mfem.hpp>
#include <type_traits>
#include <vector>

namespace plugin
{
/// Geometry cached at one element integration point.
struct ElementIntegrationPointGeometry
{
    Eigen::MatrixXr GShape;
    mfem::real_t DetdXdXi{ 0. };
    mfem::real_t Weight{ 0. };
};

/// Geometry cached at one interior-face integration point.
struct FaceIntegrationPointGeometry
{
    mfem::real_t Weight{ 0. };
    mfem::Vector Shape1, Shape2;
    mfem::DenseMatrix GShapeFace1, GShapeFace2;
    mfem::DenseMatrix Jacobian;
};

template <typename PointState>
struct ElementIntegrationPointData : ElementIntegrationPointGeometry
{
    /// Model-owned state is colocated with the geometry of its material point.
    PointState State{};
};

template <>
struct ElementIntegrationPointData<NoIntegrationPointState> : ElementIntegrationPointGeometry
{
};

template <typename PointState>
struct FaceIntegrationPointData : FaceIntegrationPointGeometry
{
    /// Interface-law state is colocated with the geometry of its material point.
    PointState State{};
};

template <>
struct FaceIntegrationPointData<NoIntegrationPointState> : FaceIntegrationPointGeometry
{
};

static_assert( sizeof( ElementIntegrationPointData<NoIntegrationPointState> ) == sizeof( ElementIntegrationPointGeometry ) );
static_assert( sizeof( FaceIntegrationPointData<NoIntegrationPointState> ) == sizeof( FaceIntegrationPointGeometry ) );

/// State-independent geometry interface used by integrators that do not own history.
class IntegrationPointStorageBase
{
public:
    virtual ~IntegrationPointStorageBase() = default;

    virtual void InitializeElement( const mfem::FiniteElement&, mfem::ElementTransformation&, const mfem::IntegrationRule& ) = 0;

    virtual void InitializeFace( const mfem::FiniteElement&,
                                 const mfem::FiniteElement&,
                                 mfem::FaceElementTransformations&,
                                 const mfem::IntegrationRule& ) = 0;

    virtual const ElementIntegrationPointGeometry& GetElementPoint( int gauss ) const = 0;
    virtual const FaceIntegrationPointGeometry& GetFacePoint( int gauss ) const = 0;
    virtual void Reset( mfem::Mesh* mesh ) = 0;
    virtual const mfem::Mesh* GetMesh() const noexcept
    {
        return nullptr;
    }

    const Eigen::MatrixXr& GetdNdX( const int gauss ) const
    {
        return GetElementPoint( gauss ).GShape;
    }

    mfem::real_t GetDetdXdXi( const int gauss ) const
    {
        return GetElementPoint( gauss ).DetdXdXi;
    }

    mfem::real_t GetElementWeight( const int gauss ) const
    {
        return GetElementPoint( gauss ).Weight;
    }

    const mfem::Vector& GetFace1Shape( const int gauss ) const
    {
        return GetFacePoint( gauss ).Shape1;
    }

    const mfem::Vector& GetFace2Shape( const int gauss ) const
    {
        return GetFacePoint( gauss ).Shape2;
    }

    const mfem::DenseMatrix& GetFace1GShape( const int gauss ) const
    {
        return GetFacePoint( gauss ).GShapeFace1;
    }

    const mfem::DenseMatrix& GetFace2GShape( const int gauss ) const
    {
        return GetFacePoint( gauss ).GShapeFace2;
    }

    mfem::real_t GetFaceWeight( const int gauss ) const
    {
        return GetFacePoint( gauss ).Weight;
    }

    const mfem::DenseMatrix& GetFaceJacobian( const int gauss ) const
    {
        return GetFacePoint( gauss ).Jacobian;
    }
};

template <typename ElementState = NoIntegrationPointState, typename FaceState = NoIntegrationPointState>
class IntegrationPointStorage final : public IntegrationPointStorageBase
{
public:
    using ElementPoint = ElementIntegrationPointData<ElementState>;
    using FacePoint = FaceIntegrationPointData<FaceState>;
    using ElementStateType = ElementState;
    using FaceStateType = FaceState;

    explicit IntegrationPointStorage( mfem::Mesh* mesh )
    {
        Reset( mesh );
    }

    IntegrationPointStorage( const IntegrationPointStorage& ) = delete;
    IntegrationPointStorage& operator=( const IntegrationPointStorage& ) = delete;
    IntegrationPointStorage( IntegrationPointStorage&& ) = delete;
    IntegrationPointStorage& operator=( IntegrationPointStorage&& ) = delete;

    void InitializeElement( const mfem::FiniteElement& el, mfem::ElementTransformation& transformation, const mfem::IntegrationRule& rule ) override
    {
        mCurrentElement = transformation.ElementNo;
        MFEM_VERIFY(
            mCurrentElement >= 0 && mCurrentElement < static_cast<int>( mElementStorage.size() ),
            "Element quadrature storage does not match the current mesh. Call Reset after changing the mesh." );

        auto& pointSet = mElementStorage[mCurrentElement];
        if ( pointSet.IsInitialized() )
        {
            VerifyElementPointSet( pointSet, el, rule, mCurrentElement );
            return;
        }

        pointSet = BuildElementPointSet( el, transformation, rule );
    }

    void InitializeFace( const mfem::FiniteElement& el1,
                         const mfem::FiniteElement& el2,
                         mfem::FaceElementTransformations& transformation,
                         const mfem::IntegrationRule& rule ) override
    {
        mCurrentFace = transformation.ElementNo;
        MFEM_VERIFY( mCurrentFace >= 0 && mCurrentFace < static_cast<int>( mFaceStorage.size() ),
                     "Face quadrature storage does not match the current mesh. Call Reset after changing the mesh." );

        auto& pointSet = mFaceStorage[mCurrentFace];
        if ( pointSet )
        {
            VerifyFacePointSet( *pointSet, el1, el2, transformation, rule, mCurrentFace );
            return;
        }

        pointSet = std::make_unique<FacePointSet>( BuildFacePointSet( el1, el2, transformation, rule ) );
    }

    const ElementPoint& GetElementPoint( const int gauss ) const override
    {
        const auto& points = CurrentElementPointSet().Points;
        MFEM_VERIFY( gauss >= 0 && gauss < static_cast<int>( points.size() ),
                     "Element quadrature-point index is out of range." );
        return points[gauss];
    }

    ElementPoint& GetElementPoint( const int gauss )
    {
        auto& points = CurrentElementPointSet().Points;
        MFEM_VERIFY( gauss >= 0 && gauss < static_cast<int>( points.size() ),
                     "Element quadrature-point index is out of range." );
        return points[gauss];
    }

    const FacePoint& GetFacePoint( const int gauss ) const override
    {
        const auto& points = CurrentFacePointSet().Points;
        MFEM_VERIFY( gauss >= 0 && gauss < static_cast<int>( points.size() ),
                     "Face quadrature-point index is out of range." );
        return points[gauss];
    }

    FacePoint& GetFacePoint( const int gauss )
    {
        auto& points = CurrentFacePointSet().Points;
        MFEM_VERIFY( gauss >= 0 && gauss < static_cast<int>( points.size() ),
                     "Face quadrature-point index is out of range." );
        return points[gauss];
    }

    template <typename Visitor>
    void VisitElementStates( Visitor&& visitor )
    {
        static_assert( !std::is_same_v<ElementState, NoIntegrationPointState>,
                       "Element state visitation requires a nonempty element state type." );
        for ( auto& element : mElementStorage )
        {
            if ( !element.IsInitialized() )
            {
                continue;
            }

            for ( auto& point : element.Points )
            {
                visitor( point.State );
            }
        }
    }

    template <typename Visitor>
    void VisitElementPoints( Visitor&& visitor )
    {
        for ( int elementNumber = 0; elementNumber < static_cast<int>( mElementStorage.size() ); elementNumber++ )
        {
            auto& element = mElementStorage[elementNumber];
            if ( !element.IsInitialized() )
            {
                continue;
            }

            for ( int pointNumber = 0; pointNumber < static_cast<int>( element.Points.size() ); pointNumber++ )
            {
                visitor( elementNumber, pointNumber, element.Points[pointNumber] );
            }
        }
    }

    template <typename Visitor>
    void VisitElementPoints( Visitor&& visitor ) const
    {
        for ( int elementNumber = 0; elementNumber < static_cast<int>( mElementStorage.size() ); elementNumber++ )
        {
            const auto& element = mElementStorage[elementNumber];
            if ( !element.IsInitialized() )
            {
                continue;
            }

            for ( int pointNumber = 0; pointNumber < static_cast<int>( element.Points.size() ); pointNumber++ )
            {
                visitor( elementNumber, pointNumber, element.Points[pointNumber] );
            }
        }
    }

    template <typename Visitor>
    void VisitFaceStates( Visitor&& visitor )
    {
        static_assert( !std::is_same_v<FaceState, NoIntegrationPointState>,
                       "Face state visitation requires a nonempty face state type." );
        for ( auto& face : mFaceStorage )
        {
            if ( !face )
            {
                continue;
            }

            for ( auto& point : face->Points )
            {
                visitor( point.State );
            }
        }
    }

    void Reset( mfem::Mesh* mesh ) override
    {
        // Geometry and history describe the same material points and are
        // invalidated together when mesh or finite-element data changes.
        mElementStorage.clear();
        mFaceStorage.clear();
        mCurrentElement = -1;
        mCurrentFace = -1;
        mMesh = mesh;

        if ( mesh == nullptr )
        {
            return;
        }

        mElementStorage.resize( mesh->GetNE() );
        mFaceStorage.resize( mesh->GetNumFacesWithGhost() );
    }

    const mfem::Mesh* GetMesh() const noexcept override
    {
        return mMesh;
    }

private:
    struct ElementPointSet
    {
        const mfem::FiniteElement* Element{ nullptr };
        const mfem::IntegrationRule* Rule{ nullptr };
        std::vector<ElementPoint> Points;
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
        std::vector<FacePoint> Points;
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

    ElementPointSet BuildElementPointSet( const mfem::FiniteElement& el,
                                          mfem::ElementTransformation& transformation,
                                          const mfem::IntegrationRule& rule )
    {
        const int dimension = el.GetDim();
        const int numberOfNodes = el.GetDof();
        const int numberOfPoints = rule.GetNPoints();

        ElementPointSet pointSet;
        pointSet.Element = &el;
        pointSet.Rule = &rule;
        pointSet.Dof = numberOfNodes;
        pointSet.Dimension = dimension;
        pointSet.Points.resize( numberOfPoints );

        auto& referenceGradient = mBuildWorkspace.ReferenceGradient1;
        referenceGradient.SetSize( numberOfNodes, dimension );
        for ( int i = 0; i < numberOfPoints; i++ )
        {
            auto& point = pointSet.Points[i];
            point.GShape.resize( numberOfNodes, dimension );
            mfem::DenseMatrix physicalGradientView( point.GShape.data(), numberOfNodes, dimension );
            const mfem::IntegrationPoint& integrationPoint = rule.IntPoint( i );
            transformation.SetIntPoint( &integrationPoint );
            el.CalcDShape( integrationPoint, referenceGradient );
            Mult( referenceGradient, transformation.InverseJacobian(), physicalGradientView );

            point.DetdXdXi = transformation.Weight();
            point.Weight = integrationPoint.weight * point.DetdXdXi;
        }

        return pointSet;
    }

    FacePointSet BuildFacePointSet( const mfem::FiniteElement& el1,
                                    const mfem::FiniteElement& el2,
                                    mfem::FaceElementTransformations& transformation,
                                    const mfem::IntegrationRule& rule )
    {
        const int dimension = el1.GetDim();
        const int numberOfNodes1 = el1.GetDof();
        const int numberOfNodes2 = el2.GetDof();
        const int numberOfPoints = rule.GetNPoints();

        auto& referenceGradient1 = mBuildWorkspace.ReferenceGradient1;
        auto& referenceGradient2 = mBuildWorkspace.ReferenceGradient2;
        auto& physicalGradient1 = mBuildWorkspace.PhysicalGradient1;
        auto& physicalGradient2 = mBuildWorkspace.PhysicalGradient2;
        referenceGradient1.SetSize( numberOfNodes1, dimension );
        referenceGradient2.SetSize( numberOfNodes2, dimension );
        physicalGradient1.SetSize( numberOfNodes1, dimension );
        physicalGradient2.SetSize( numberOfNodes2, dimension );

        FacePointSet pointSet;
        pointSet.Element1 = &el1;
        pointSet.Element2 = &el2;
        pointSet.Rule = &rule;
        pointSet.Dof1 = numberOfNodes1;
        pointSet.Dof2 = numberOfNodes2;
        pointSet.Dimension = dimension;
        pointSet.FaceDimension = transformation.GetDimension();
        pointSet.Points.resize( numberOfPoints );

        for ( int i = 0; i < numberOfPoints; i++ )
        {
            auto& point = pointSet.Points[i];
            point.Shape1.SetSize( numberOfNodes1 );
            point.Shape2.SetSize( numberOfNodes2 );
            const mfem::IntegrationPoint& integrationPoint = rule.IntPoint( i );
            transformation.SetAllIntPoints( &integrationPoint );
            const mfem::IntegrationPoint& elementPoint1 = transformation.GetElement1IntPoint();
            const mfem::IntegrationPoint& elementPoint2 = transformation.GetElement2IntPoint();
            el1.CalcShape( elementPoint1, point.Shape1 );
            el2.CalcShape( elementPoint2, point.Shape2 );

            point.Weight = integrationPoint.weight * transformation.Weight();
            point.Jacobian = transformation.Jacobian();

            el1.CalcDShape( elementPoint1, referenceGradient1 );
            el2.CalcDShape( elementPoint2, referenceGradient2 );
            auto& transformation1 = transformation.GetElement1Transformation();
            auto& transformation2 = transformation.GetElement2Transformation();
            transformation1.SetIntPoint( &elementPoint1 );
            transformation2.SetIntPoint( &elementPoint2 );

            Mult( referenceGradient1, transformation1.InverseJacobian(), physicalGradient1 );
            Mult( referenceGradient2, transformation2.InverseJacobian(), physicalGradient2 );

            point.GShapeFace1.SetSize( numberOfNodes1, transformation.GetDimension() );
            point.GShapeFace2.SetSize( numberOfNodes2, transformation.GetDimension() );
            Mult( physicalGradient1, point.Jacobian, point.GShapeFace1 );
            Mult( physicalGradient2, point.Jacobian, point.GShapeFace2 );
        }

        return pointSet;
    }

    static void VerifyElementPointSet( const ElementPointSet& pointSet,
                                       const mfem::FiniteElement& el,
                                       const mfem::IntegrationRule& rule,
                                       const int elementNumber )
    {
        MFEM_VERIFY( pointSet.Element == &el, "Element " << elementNumber << " was initialized with a different finite element." );
        MFEM_VERIFY( pointSet.Rule == &rule, "Element " << elementNumber << " was initialized with a different integration rule." );
        MFEM_VERIFY( pointSet.Dof == el.GetDof() && pointSet.Dimension == el.GetDim(),
                     "Element " << elementNumber << " quadrature metadata is incompatible with the requested finite element." );
        MFEM_VERIFY( static_cast<int>( pointSet.Points.size() ) == rule.GetNPoints(),
                     "Element " << elementNumber << " quadrature-point count is incompatible with the requested integration rule." );
    }

    static void VerifyFacePointSet( const FacePointSet& pointSet,
                                    const mfem::FiniteElement& el1,
                                    const mfem::FiniteElement& el2,
                                    const mfem::FaceElementTransformations& transformation,
                                    const mfem::IntegrationRule& rule,
                                    const int faceNumber )
    {
        MFEM_VERIFY( pointSet.Element1 == &el1 && pointSet.Element2 == &el2,
                     "Face " << faceNumber << " was initialized with different finite elements." );
        MFEM_VERIFY( pointSet.Rule == &rule, "Face " << faceNumber << " was initialized with a different integration rule." );
        MFEM_VERIFY( pointSet.Dof1 == el1.GetDof() && pointSet.Dof2 == el2.GetDof() &&
                         pointSet.Dimension == el1.GetDim() && pointSet.FaceDimension == transformation.GetDimension(),
                     "Face " << faceNumber << " quadrature metadata is incompatible with the requested finite elements." );
        MFEM_VERIFY( static_cast<int>( pointSet.Points.size() ) == rule.GetNPoints(),
                     "Face " << faceNumber << " quadrature-point count is incompatible with the requested integration rule." );
    }

    const ElementPointSet& CurrentElementPointSet() const
    {
        MFEM_VERIFY( mCurrentElement >= 0 && mCurrentElement < static_cast<int>( mElementStorage.size() ),
                     "No current element quadrature storage is available." );
        const auto& pointSet = mElementStorage[mCurrentElement];
        MFEM_VERIFY( pointSet.IsInitialized(), "Current element quadrature storage has not been initialized." );
        return pointSet;
    }

    ElementPointSet& CurrentElementPointSet()
    {
        MFEM_VERIFY( mCurrentElement >= 0 && mCurrentElement < static_cast<int>( mElementStorage.size() ),
                     "No current element quadrature storage is available." );
        auto& pointSet = mElementStorage[mCurrentElement];
        MFEM_VERIFY( pointSet.IsInitialized(), "Current element quadrature storage has not been initialized." );
        return pointSet;
    }

    const FacePointSet& CurrentFacePointSet() const
    {
        MFEM_VERIFY( mCurrentFace >= 0 && mCurrentFace < static_cast<int>( mFaceStorage.size() ),
                     "No current face quadrature storage is available." );
        MFEM_VERIFY( mFaceStorage[mCurrentFace] != nullptr,
                     "Current face quadrature storage has not been initialized." );
        return *mFaceStorage[mCurrentFace];
    }

    FacePointSet& CurrentFacePointSet()
    {
        MFEM_VERIFY( mCurrentFace >= 0 && mCurrentFace < static_cast<int>( mFaceStorage.size() ),
                     "No current face quadrature storage is available." );
        MFEM_VERIFY( mFaceStorage[mCurrentFace] != nullptr,
                     "Current face quadrature storage has not been initialized." );
        return *mFaceStorage[mCurrentFace];
    }

    std::vector<ElementPointSet> mElementStorage;
    std::vector<std::unique_ptr<FacePointSet>> mFaceStorage;
    PointSetBuildWorkspace mBuildWorkspace;
    mfem::Mesh* mMesh{ nullptr };
    int mCurrentElement{ -1 };
    int mCurrentFace{ -1 };
};

IntegrationPointStorage( mfem::Mesh* ) -> IntegrationPointStorage<>;
} // namespace plugin
