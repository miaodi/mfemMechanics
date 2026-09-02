#pragma once
#include "Material.h"
#include "StressFreeDeformation.h"
#include "mfem.hpp"
#include <Eigen/Dense>

namespace plugin
{
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
