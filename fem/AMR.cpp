
#include "AMR.h"
#include "util.h"
#include <cmath>
namespace plugin
{
CriticalVMRefiner::CriticalVMRefiner( StressCoefficient& s ) : mfem::MeshOperator(), sc( s )
{
}

int CriticalVMRefiner::ApplyImpl( mfem::Mesh& mesh )
{
    num_marked_elements = 0LL;
    marked_elements.SetSize( 0 );
    auto fes = sc.GetDisplacement()->FESpace();
    const mfem::FiniteElement* fe;
    mfem::Vector val( 7 );
    mfem::ElementTransformation* T;
    double max_h = 0;
    for ( int i = 0; i < fes->GetNE(); i++ )
    {
        fe = fes->GetFE( i );
        const auto& nodes = fe->GetNodes();
        max_h = std::max( max_h, util::SmallestCircle( nodes, 2 ) );
        if ( util::SmallestCircle( nodes, 2 ) < critical_h )
            continue;
        T = fes->GetElementTransformation( i );
        auto& int_rule = mfem::IntRules.Get( T->GetGeometryType(), 2 * fes->GetOrder( i ) );
        const auto nip = int_rule.GetNPoints();
        for ( int j = 0; j < nip; j++ )
        {
            auto& fip = int_rule.IntPoint( j );
            sc.Eval( val, *T, fip );
            if ( val( 6 ) > critical_vm )
            {
                marked_elements.Append( mfem::Refinement( i ) );
                break;
            }
        }
    }
    num_marked_elements = mesh.ReduceInt( marked_elements.Size() );
    if ( num_marked_elements == 0LL )
    {
        return STOP;
    }
    std::cout << "max_h: " << max_h << std::endl;
    mesh.GeneralRefinement( marked_elements );
    return CONTINUE + REFINED;
}

void CriticalVMRefiner::Reset()
{
}
} // namespace plugin