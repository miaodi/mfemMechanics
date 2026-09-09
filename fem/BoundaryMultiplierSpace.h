#pragma once

#include <memory>
#include <mfem.hpp>

namespace plugin
{
/** @brief Owned scalar, value-mapped discontinuous L2 space on a parent boundary.

    The parent is borrowed and must outlive this object. Only serial conforming,
    full-dimensional 2D/3D parent meshes and MFEM's default CPU backend are
    supported. A null collection selects L2_FECollection(0, parent.Dimension()-1).
    A supplied collection is transferred into this object; H1, NURBS, vector,
    integral-mapped, and dimension-incompatible collections are rejected.
    NURBS parent meshes are unsupported. Curved parents must not themselves be
    SubMesh objects and must use uniform-order Gauss-Lobatto H1 geometry nodes.
    MFEM's coefficient-copy geometry transfer is not a basis conversion and
    does not handle the nested or discontinuous geometry cases used here.

    P0 and degree-one P1/Q1 use the same generic basis evaluation as higher
    nonnegative L2 orders, including MFEM's positive basis. Higher-order contact
    stability and quadrature convergence are not established by this abstraction.
    It stores no multiplier solution. Mesh/space accessors permit MFEM output
    objects to borrow them, not topology, geometry, order, or layout changes:
    reconstruct this object and its borrowers after any such change. MFEM's
    mesh/space caches are not documented as reentrant or thread-safe. */
class BoundaryMultiplierSpace
{
public:
    BoundaryMultiplierSpace( mfem::Mesh& parent,
                             const mfem::Array<int>& attributes,
                             std::unique_ptr<mfem::FiniteElementCollection> collection = nullptr );
    ~BoundaryMultiplierSpace();

    BoundaryMultiplierSpace( const BoundaryMultiplierSpace& ) = delete;
    BoundaryMultiplierSpace& operator=( const BoundaryMultiplierSpace& ) = delete;
    BoundaryMultiplierSpace( BoundaryMultiplierSpace&& ) = delete;
    BoundaryMultiplierSpace& operator=( BoundaryMultiplierSpace&& ) = delete;

    [[nodiscard]] mfem::SubMesh& GetMesh() noexcept;
    [[nodiscard]] const mfem::SubMesh& GetMesh() const noexcept;
    [[nodiscard]] mfem::FiniteElementSpace& GetSpace() noexcept;
    [[nodiscard]] const mfem::FiniteElementSpace& GetSpace() const noexcept;
    // mMesh maps multiplier-submesh element indices (not DOFs) to parent
    // boundary-element indices, not to the adjacent volume-element indices.
    [[nodiscard]] int GetParentBoundaryElement( int element ) const;
    void GetElementDofs( int element, mfem::Array<int>& dofs ) const;

    /** Evaluate the local multiplier basis at an ORIGINAL PARENT BOUNDARY-ELEMENT
        reference point, NOT a parent mesh-face point. Resizes shape if needed;
        callers can pre-size it outside quadrature loops. */
    void CalcShape( int element, const mfem::IntegrationPoint& parentBoundaryPoint, mfem::Vector& shape ) const;

private:
    void VerifyElement( int element ) const;
    void VerifyElementMaps() const;

    // Reverse destruction: space first, then its collection, then its mesh.
    // The SubMesh itself retains the borrowed parent address.
    mfem::SubMesh mMesh;
    std::unique_ptr<mfem::FiniteElementCollection> mCollection;
    mfem::FiniteElementSpace mSpace;
};
} // namespace plugin
