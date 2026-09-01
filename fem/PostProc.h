#pragma once
#include "Material.h"
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
    Eigen::Matrix3d F;

public:
    StressCoefficient( int dim, ElasticMaterial& mat );

    void SetDisplacement( mfem::GridFunction& u_ )
    {
        u = &u_;
    }
    
    mfem::GridFunction* GetDisplacement()
    {
        return u;
    }

    virtual void Eval( mfem::Vector& V, mfem::ElementTransformation& T, const mfem::IntegrationPoint& ip ) override;
};
} // namespace plugin