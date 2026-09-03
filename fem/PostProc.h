#pragma once
#include "IntegrationPointStorage.h"
#include "J2Plasticity.h"
#include "Material.h"
#include "StressFreeDeformation.h"
#include "mfem.hpp"
#include <Eigen/Dense>
#include <vector>

namespace plugin
{
/**
 * Exposes a planar Cartesian GridFunction as (x, y, 0) for ParaView.
 *
 * Register this object with ParaViewDataCollection::RegisterVCoeffField so
 * filters such as Warp By Vector recognize the result as a vector. The source
 * GridFunction is borrowed and must outlive this coefficient and all saves.
 */
class ParaView2DVectorCoefficient : public mfem::VectorCoefficient
{
public:
    explicit ParaView2DVectorCoefficient( const mfem::GridFunction& field );

    void Eval( mfem::Vector& value, mfem::ElementTransformation& transformation, const mfem::IntegrationPoint& integrationPoint ) override;

private:
    const mfem::GridFunction& mField;
    mfem::Vector mPlanarValue;
};

template <typename PointStorage>
void ProjectCommittedEquivalentPlasticStrain( const PointStorage& pointStorage, mfem::GridFunction& field )
{
    MFEM_VERIFY( field.FESpace() != nullptr && field.FESpace()->GetVDim() == 1,
                 "Equivalent plastic strain output requires a scalar finite-element space." );
    MFEM_VERIFY( field.FESpace()->GetMesh() == pointStorage.GetMesh(),
                 "Equivalent plastic strain output must use the point-storage mesh." );
    const int numberOfElements = field.FESpace()->GetMesh()->GetNE();
    MFEM_VERIFY( field.FESpace()->GetVSize() == numberOfElements,
                 "Equivalent plastic strain output requires a discontinuous piecewise-constant space." );
    std::vector<mfem::real_t> sums( numberOfElements, 0. );
    std::vector<mfem::real_t> weights( numberOfElements, 0. );
    pointStorage.VisitElementPoints(
        [&sums, &weights]( const int elementNumber, const int, const auto& point )
        {
            sums[elementNumber] +=
                point.Weight * point.State.template Get<J2PlasticityMaterial>().CommittedState().EquivalentPlasticStrain;
            weights[elementNumber] += point.Weight;
        } );

    field = 0.;
    mfem::Array<int> elementDofs;
    for ( int elementNumber = 0; elementNumber < numberOfElements; elementNumber++ )
    {
        if ( weights[elementNumber] == 0. )
        {
            continue;
        }
        field.FESpace()->GetElementDofs( elementNumber, elementDofs );
        MFEM_ASSERT( elementDofs.Size() == 1, "Piecewise-constant output must have one DOF per element." );
        field( elementDofs[0] ) = sums[elementNumber] / weights[elementNumber];
    }
}

// A Coefficient for computing the components of the stress.
class StressCoefficient : public mfem::VectorCoefficient
{
protected:
    mfem::GridFunction* u;  // displacement
    mfem::DenseMatrix grad; // auxiliary matrix, used in Eval
    ElasticMaterial* materialModel{ nullptr };
    int dim;
    Eigen::Matrix3r F;
    Eigen::Matrix3r elasticF;
    Eigen::Matrix3r mechanicalStrain;
    Eigen::Matrix3r stressFreeF;
    StressFreeDeformationModel stressFreeDeformations;
    mfem::real_t loadFactor{ 1. };

public:
    StressCoefficient( int dim, ElasticMaterial& mat );

    void SetDisplacement( mfem::GridFunction& u_ )
    {
        u = &u_;
    }

    void AddStressFreeDeformation( StressFreeDeformation& deformation )
    {
        stressFreeDeformations.Add( deformation );
    }

    void ClearStressFreeDeformations()
    {
        stressFreeDeformations.Clear();
    }

    void SetLoadFactor( const mfem::real_t value )
    {
        loadFactor = value;
    }

    mfem::GridFunction* GetDisplacement()
    {
        return u;
    }

    virtual void Eval( mfem::Vector& V, mfem::ElementTransformation& T, const mfem::IntegrationPoint& ip ) override;
};
} // namespace plugin
