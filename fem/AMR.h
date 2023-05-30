#pragma once

#include <mfem.hpp>

namespace plugin
{
class CriticalVMStressEstimator : public mfem::ZienkiewiczZhuEstimator
{
    /// Compute the element error estimates.
    void ComputeVMStress();

public:
    CriticalVMStressEstimator( mfem::BilinearFormIntegrator& integ, mfem::GridFunction& sol, mfem::FiniteElementSpace* flux_fes )
        : ZienkiewiczZhuEstimator( integ, sol, flux_fes )
    {
    }

    /// Get a Vector with all element errors.
    virtual const mfem::Vector& GetLocalErrors() override
    {
        if ( MeshIsModified() )
        {
            ComputeVMStress();
        }
        return error_estimates;
    }
};
} // namespace plugin