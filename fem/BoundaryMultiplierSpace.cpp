#include "BoundaryMultiplierSpace.h"

#include <utility>

namespace plugin
{
namespace
{
mfem::SubMesh CreateBoundaryMesh( mfem::Mesh& parent, const mfem::Array<int>& attributes )
{
    MFEM_VERIFY( mfem::Device::IsDisabled(), "Boundary multipliers support only MFEM's default CPU backend." );
    MFEM_VERIFY( ( parent.Dimension() == 2 || parent.Dimension() == 3 ) && parent.SpaceDimension() == parent.Dimension(),
                 "Boundary multipliers require a full-dimensional 2D or 3D parent mesh." );
    MFEM_VERIFY( !parent.Nonconforming(), "Boundary multipliers require a conforming parent mesh." );
    MFEM_VERIFY( parent.NURBSext == nullptr, "Boundary multipliers do not support NURBS parent meshes." );
#ifdef MFEM_USE_MPI
    MFEM_VERIFY( dynamic_cast<mfem::ParMesh*>( &parent ) == nullptr,
                 "Boundary multipliers require a serial parent mesh." );
#endif
    if ( const auto* nodes = parent.GetNodes() )
    {
        // MFEM's nested SubMesh transfer routes through the root without
        // composing this child's immediate-parent boundary IDs into root IDs.
        MFEM_VERIFY( dynamic_cast<const mfem::SubMesh*>( &parent ) == nullptr,
                     "Boundary multipliers do not support curved SubMesh parents." );
        const auto& space = *nodes->FESpace();
        const auto* h1 = dynamic_cast<const mfem::H1_FECollection*>( space.FEColl() );
        // SubMesh::SubMesh calls SetCurvature, which creates Gauss-Lobatto
        // geometry, then SubMeshUtils::BuildVdofToVdofMap copies coefficients.
        // That transfer is not a change of basis. Reject incompatible geometry
        // before it can silently change the reference surface/edge metric.
        // Discontinuous geometry also duplicates parent DOFs at submesh
        // vertices, violating that transfer's one-to-one map in MFEM Debug.
        // These restrictions are on GEOMETRY, not the multiplier collection.
        MFEM_VERIFY( !space.IsVariableOrder() && space.GetVDim() == parent.SpaceDimension() && h1 != nullptr &&
                         h1->GetBasisType() == mfem::BasisType::GaussLobatto,
                     "Boundary submesh geometry transfer requires uniform-order Gauss-Lobatto H1 nodes." );
    }
    MFEM_VERIFY( attributes.Size() > 0, "Boundary multipliers require at least one boundary attribute." );
    for ( int i = 0; i < attributes.Size(); ++i )
    {
        MFEM_VERIFY( attributes[i] > 0 && parent.bdr_attributes.Find( attributes[i] ) >= 0,
                     "A boundary multiplier attribute is not present in the parent mesh." );
        for ( int j = 0; j < i; ++j )
        {
            MFEM_VERIFY( attributes[j] != attributes[i], "Boundary multiplier attributes must be unique." );
        }
    }
    for ( int element = 0; element < parent.GetNBE(); ++element )
    {
        if ( attributes.Find( parent.GetBdrAttribute( element ) ) < 0 )
        {
            continue;
        }
        int adjacent = -1;
        int second = -1;
        parent.GetFaceElements( parent.GetBdrElementFaceIndex( element ), &adjacent, &second );
        MFEM_VERIFY( adjacent >= 0 && second < 0, "Boundary multipliers require exterior parent boundary elements." );
    }
    return mfem::SubMesh::CreateFromBoundary( parent, attributes );
}

std::unique_ptr<mfem::FiniteElementCollection> SelectCollection( const mfem::SubMesh& mesh,
                                                                 std::unique_ptr<mfem::FiniteElementCollection> collection )
{
    if ( collection == nullptr )
    {
        collection = std::make_unique<mfem::L2_FECollection>( 0, mesh.Dimension() );
    }
    MFEM_VERIFY( dynamic_cast<const mfem::L2_FECollection*>( collection.get() ) != nullptr &&
                     collection->GetContType() == mfem::FiniteElementCollection::DISCONTINUOUS && collection->GetOrder() >= 0,
                 "Boundary multipliers require a discontinuous L2_FECollection of nonnegative order." );
    MFEM_VERIFY( mesh.GetNE() > 0, "The selected boundary multiplier submesh is empty." );
    // L2_FECollection provides volume elements ONLY in its own dimension;
    // TraceFiniteElementForGeometry must not be used to accept a wrong dimension.
    // Validate before FiniteElementSpace construction can dereference a null FE.
    for ( int element = 0; element < mesh.GetNE(); ++element )
    {
        const auto* finiteElement = collection->FiniteElementForGeometry( mesh.GetElementBaseGeometry( element ) );
        MFEM_VERIFY( finiteElement != nullptr && finiteElement->GetDim() == mesh.Dimension(),
                     "The boundary multiplier collection has an incompatible dimension or element geometry." );
        MFEM_VERIFY( finiteElement->GetRangeType() == mfem::FiniteElement::SCALAR &&
                         finiteElement->GetMapType() == mfem::FiniteElement::VALUE,
                     "Boundary multipliers require scalar VALUE-mapped elements, not integral-mapped elements." );
    }
    return collection;
}
} // namespace

BoundaryMultiplierSpace::BoundaryMultiplierSpace( mfem::Mesh& parent,
                                                  const mfem::Array<int>& attributes,
                                                  std::unique_ptr<mfem::FiniteElementCollection> collection )
    : mMesh( CreateBoundaryMesh( parent, attributes ) ),
      mCollection( SelectCollection( mMesh, std::move( collection ) ) ),
      mSpace( &mMesh, mCollection.get() )
{
    MFEM_VERIFY( mSpace.GetVDim() == 1 && mSpace.GetTrueVSize() == mSpace.GetVSize(),
                 "Boundary multipliers require an unconstrained scalar serial L2 space." );
    VerifyElementMaps();
}

BoundaryMultiplierSpace::~BoundaryMultiplierSpace() = default;

mfem::SubMesh& BoundaryMultiplierSpace::GetMesh() noexcept
{
    return mMesh;
}

const mfem::SubMesh& BoundaryMultiplierSpace::GetMesh() const noexcept
{
    return mMesh;
}

mfem::FiniteElementSpace& BoundaryMultiplierSpace::GetSpace() noexcept
{
    return mSpace;
}

const mfem::FiniteElementSpace& BoundaryMultiplierSpace::GetSpace() const noexcept
{
    return mSpace;
}

void BoundaryMultiplierSpace::VerifyElement( const int element ) const
{
    MFEM_VERIFY( element >= 0 && element < mMesh.GetNE(), "The boundary multiplier element index is out of range." );
}

int BoundaryMultiplierSpace::GetParentBoundaryElement( const int element ) const
{
    VerifyElement( element );
    return mMesh.GetParentElementIDMap()[element];
}

void BoundaryMultiplierSpace::GetElementDofs( const int element, mfem::Array<int>& dofs ) const
{
    VerifyElement( element );
    MFEM_VERIFY( mSpace.GetElementDofs( element, dofs ) == nullptr,
                 "Boundary multipliers do not support element DOF transformations." );
}

void BoundaryMultiplierSpace::CalcShape( const int element, const mfem::IntegrationPoint& parentBoundaryPoint, mfem::Vector& shape ) const
{
    VerifyElement( element );
    const auto& finiteElement = *mSpace.GetFE( element );
    shape.SetSize( finiteElement.GetDof() );
    // VerifyElementMaps establishes identical local vertex order, hence the
    // identity reference map from ORIGINAL parent boundary element to submesh.
    // Applying GetParentFaceOrientations here would incorrectly apply the
    // separate parent boundary-to-mesh-face orientation a second time.
    finiteElement.CalcShape( parentBoundaryPoint, shape );
}

void BoundaryMultiplierSpace::VerifyElementMaps() const
{
    const auto& parent = *mMesh.GetParent();
    MFEM_VERIFY( mMesh.GetParentElementIDMap().Size() == mMesh.GetNE(),
                 "The boundary multiplier parent-element map is inconsistent." );
    mfem::Array<int> parentVertices, submeshVertices, dofs;
    mfem::Array<int> seen( mSpace.GetVSize() );
    seen = 0;
    for ( int element = 0; element < mMesh.GetNE(); ++element )
    {
        const int boundaryElement = GetParentBoundaryElement( element );
        MFEM_VERIFY( boundaryElement >= 0 && boundaryElement < parent.GetNBE(),
                     "A boundary multiplier element has an invalid parent boundary index." );
        parent.GetBdrElementVertices( boundaryElement, parentVertices );
        mMesh.GetElementVertices( element, submeshVertices );
        MFEM_VERIFY( parent.GetBdrElementGeometry( boundaryElement ) == mMesh.GetElementBaseGeometry( element ) &&
                         parentVertices.Size() == submeshVertices.Size(),
                     "Boundary multiplier and parent boundary geometries must agree." );
        // Verified against MFEM mesh/submesh/submesh_utils.cpp:AddElementsToMesh:
        // boundary vertices are copied in local order. SubMesh's final Finalize()
        // defaults to refine=false, fix_orientation=false (mesh/mesh.hpp), so
        // regular element order is retained, including on embedded surfaces.
        // Check that contract, rather than assuming parent mesh-face coordinates
        // or the triangle-specific composed GetParentFaceOrientations map.
        for ( int vertex = 0; vertex < parentVertices.Size(); ++vertex )
        {
            MFEM_VERIFY(
                mMesh.GetParentVertexIDMap()[submeshVertices[vertex]] == parentVertices[vertex],
                "MFEM changed boundary-submesh local vertex order; a reference-coordinate remap is required." );
        }
        GetElementDofs( element, dofs );
        MFEM_VERIFY( dofs.Size() == mSpace.GetFE( element )->GetDof(),
                     "The boundary multiplier DOF map is inconsistent." );
        for ( const int dof : dofs )
        {
            MFEM_VERIFY( dof >= 0 && dof < seen.Size() && seen[dof] == 0,
                         "Boundary multiplier DOFs must be unsigned and local to exactly one element." );
            seen[dof] = 1;
        }
    }
    for ( const int count : seen )
    {
        MFEM_VERIFY( count == 1, "Every boundary multiplier DOF must belong to an element." );
    }
}
} // namespace plugin
