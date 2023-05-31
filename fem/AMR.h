#pragma once

#include "PostProc.h"
#include <mfem.hpp>

namespace plugin
{
class CriticalVMRefiner : public mfem::MeshOperator
{
protected:
    StressCoefficient& sc;
    mfem::Array<mfem::Refinement> marked_elements;
    double critical_vm{ 0 };
    double critical_h{ 0 };
    long long num_marked_elements{ 0 };

    /** @brief Apply the operator to the mesh.
        @return STOP if a stopping criterion is satisfied or no elements were
        marked for refinement; REFINED + CONTINUE otherwise. */
    virtual int ApplyImpl( mfem::Mesh& mesh );

public:
    /// Construct a ThresholdRefiner using the given ErrorEstimator.
    CriticalVMRefiner( StressCoefficient& s );

    void SetCriticalVM( const double v )
    {
        critical_vm = v;
    }

    void SetCriticalH( const double h )
    {
        critical_h = h;
    }

    /// Get the number of marked elements in the last Apply() call.
    long long GetNumMarkedElements() const
    {
        return num_marked_elements;
    }

    /// Get the threshold used in the last Apply() call.
    double GetCriticalVM() const
    {
        return critical_vm;
    }

    /// Reset the associated estimator.
    virtual void Reset();
};
} // namespace plugin