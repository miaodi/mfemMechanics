#pragma once

#include "Contact.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace contact_example
{
// Diagnostic only: endpoint-inclusive uniform samples, independent of the
// assembly Gauss rule and its boundary-trace interpolation. This is not a
// continuous collision detector or a bound on the gap between samples.
inline mfem::real_t SampleMinimumGap( mfem::Mesh& mesh,
                                      const mfem::GridFunction& displacement,
                                      const plugin::RigidObstacle& obstacle,
                                      const int boundaryAttribute,
                                      const int intervals )
{
    MFEM_VERIFY( mesh.Dimension() == 2 && intervals > 0,
                 "Gap sampling requires a 2D mesh and a positive number of intervals." );
    mfem::real_t minimumGap = std::numeric_limits<mfem::real_t>::infinity();
    mfem::Vector position( mesh.SpaceDimension() ), value;
    plugin::SignedDistanceEvaluation evaluation( mesh.SpaceDimension() );
    for ( int be = 0; be < mesh.GetNBE(); ++be )
    {
        if ( mesh.GetBdrAttribute( be ) != boundaryAttribute )
        {
            continue;
        }
        auto* transformation = mesh.GetBdrFaceTransformations( be );
        MFEM_VERIFY( transformation != nullptr, "Gap sampling requires an adjacent volume element." );
        for ( int sample = 0; sample <= intervals; ++sample )
        {
            mfem::IntegrationPoint point;
            point.Set1w( static_cast<mfem::real_t>( sample ) / intervals, 0. );
            transformation->SetAllIntPoints( &point );
            transformation->Transform( point, position );
            displacement.GetVectorValue( transformation->Elem1No, transformation->GetElement1IntPoint(), value );
            position += value;
            obstacle.Evaluate( position, evaluation );
            MFEM_VERIFY( std::isfinite( evaluation.Gap ), "Gap sampling returned a non-finite gap." );
            minimumGap = std::min( minimumGap, evaluation.Gap );
        }
    }
    MFEM_VERIFY( std::isfinite( minimumGap ), "Gap sampling found no selected boundary." );
    return minimumGap;
}
} // namespace contact_example
