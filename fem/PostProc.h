#pragma once
#include "Material.h"
#include "mfem.hpp"
#include <Eigen/Dense>

namespace plugin
{
// A Coefficient for computing the components of the stress.
class StressCoefficient : public mfem::Coefficient
{
protected:
    mfem::GridFunction* u; // displacement

    mfem::DenseMatrix grad; // auxiliary matrix, used in Eval

    ElasticMaterial* materialModel{ nullptr };
    int si, sj; // component of the stress to evaluate, 0 <= si,sj < dim

    int dim;

    Eigen::Matrix3d F;

public:
    StressCoefficient( int dim, ElasticMaterial& mat );

    void SetDisplacement( mfem::GridFunction& u_ )
    {
        u = &u_;
    }

    void SetComponent( int i, int j )
    {
        si = i;
        sj = j;
    }

    virtual double Eval(  mfem::ElementTransformation& T, const mfem::IntegrationPoint& ip );
};
} // namespace plugin