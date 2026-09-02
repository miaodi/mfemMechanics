#include "PhaseField.h"
#include "Solvers.h"

namespace plugin
{

void BlockNonlinearDirichletPenaltyIntegrator::AssembleFaceVector( const mfem::Array<const mfem::FiniteElement*>& el1,
                                                                   const mfem::Array<const mfem::FiniteElement*>& el2,
                                                                   mfem::FaceElementTransformations& Tr,
                                                                   const mfem::Array<const mfem::Vector*>& elfun,
                                                                   const mfem::Array<mfem::Vector*>& elvec )
{
    int dof_u = el1[0]->GetDof(), dim = el1[0]->GetDim();
    int dof_p = el1[1]->GetDof();

    elvec[0]->SetSize( dof_u * dim );
    elvec[1]->SetSize( dof_p );

    *elvec[0] = 0.0;
    *elvec[1] = 0.0;

    mIntegrator.AssembleFaceVector( *el1[0], *el1[1], Tr, *elfun[0], *elvec[0] );
}

/// Assemble the local gradient matrix
void BlockNonlinearDirichletPenaltyIntegrator::AssembleFaceGrad( const mfem::Array<const mfem::FiniteElement*>& el1,
                                                                 const mfem::Array<const mfem::FiniteElement*>& el2,
                                                                 mfem::FaceElementTransformations& Tr,
                                                                 const mfem::Array<const mfem::Vector*>& elfun,
                                                                 const mfem::Array2D<mfem::DenseMatrix*>& elmats )
{
    int dof_u = el1[0]->GetDof(), dim = el1[0]->GetDim();
    int dof_p = el1[1]->GetDof();
    // mGeomStiff.resize( dof_u, dof_u );

    Eigen::Map<const Eigen::MatrixXr> u( elfun[0]->GetData(), dof_u, dim );
    Eigen::Map<const Eigen::VectorXr> p( elfun[1]->GetData(), dof_p );

    elmats( 0, 0 )->SetSize( dof_u * dim, dof_u * dim );
    elmats( 0, 1 )->SetSize( dof_u * dim, dof_p );
    elmats( 1, 0 )->SetSize( dof_p, dof_u * dim );
    elmats( 1, 1 )->SetSize( dof_p, dof_p );

    *elmats( 0, 0 ) = 0.0;
    *elmats( 0, 1 ) = 0.0;
    *elmats( 1, 0 ) = 0.0;
    *elmats( 1, 1 ) = 0.0;

    mIntegrator.AssembleFaceGrad( *el1[0], *el1[1], Tr, *elfun[0], *elmats( 0, 0 ) );
}
} // namespace plugin
