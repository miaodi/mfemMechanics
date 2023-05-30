
#include "AMR.h"

namespace plugin
{
void CriticalVMStressEstimator::ComputeVMStress()
{
    flux_space->Update( false );
    // In parallel, 'flux' can be a GridFunction, as long as 'flux_space' is a
    // ParFiniteElementSpace and 'solution' is a ParGridFunction.
    GridFunction flux( flux_space );

    if ( !anisotropic )
    {
        aniso_flags.SetSize( 0 );
    }
    total_error = ZZErrorEstimator( integ, solution, flux, error_estimates, anisotropic ? &aniso_flags : NULL,
                                    flux_averaging, with_coeff );

    current_sequence = solution.FESpace()->GetMesh()->GetSequence();
}
}
} // namespace plugin