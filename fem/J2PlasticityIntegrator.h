#pragma once

#include "FEMPlugin.h"
#include "J2Plasticity.h"
#include "Solvers.h"
#include "StressFreeDeformation.h"

#include <Eigen/Dense>
#include <mfem.hpp>
#include <type_traits>

namespace plugin
{
using J2PlasticityPointStorage = IntegrationPointStorage<J2PlasticityIntegrationPointState>;

template <typename PointStorage>
class J2PlasticityIntegrator : public StepAwareNonlinearFormIntegrator
{
    static_assert( std::is_base_of_v<IntegrationPointStorageBase, PointStorage>,
                   "J2PlasticityIntegrator requires an IntegrationPointStorage specialization." );

public:
    /// Use separate storage for independent J2PlasticityMaterial instances.
    J2PlasticityIntegrator( const J2PlasticityMaterial& material, PointStorage& pointStorage )
        : mMaterial( material ), mPointStorage( pointStorage )
    {
    }

    J2PlasticityIntegrator( const J2PlasticityIntegrator& ) = delete;
    J2PlasticityIntegrator& operator=( const J2PlasticityIntegrator& ) = delete;
    J2PlasticityIntegrator( J2PlasticityIntegrator&& ) = delete;
    J2PlasticityIntegrator& operator=( J2PlasticityIntegrator&& ) = delete;

    void AssembleElementVector( const mfem::FiniteElement& element,
                                mfem::ElementTransformation& transformation,
                                const mfem::Vector& elementDisplacement,
                                mfem::Vector& elementResidual ) override;

    void AssembleElementGrad( const mfem::FiniteElement& element,
                              mfem::ElementTransformation& transformation,
                              const mfem::Vector& elementDisplacement,
                              mfem::DenseMatrix& elementJacobian ) override;

    mfem::real_t GetElementEnergy( const mfem::FiniteElement&, mfem::ElementTransformation&, const mfem::Vector& ) override
    {
        MFEM_ABORT( "Incremental energy is not implemented for J2 plasticity." );
        return 0.;
    }

    void AddStressFreeDeformation( StressFreeDeformation& deformation )
    {
        mStressFreeDeformations.Add( deformation );
    }

    void ClearStressFreeDeformations()
    {
        mStressFreeDeformations.Clear();
    }

    void BeginStep() noexcept override;
    void CommitStep() noexcept override;
    void RollbackStep() noexcept override;
    bool CanCommitStep() const noexcept override
    {
        return !mStepRejected;
    }

private:
    template <typename Visitor>
    void VisitHistory( Visitor&& visitor )
    {
        mPointStorage.VisitElementStates( [&visitor]( auto& state )
                                          { visitor( state.template Get<J2PlasticityMaterial>() ); } );
    }

    Eigen::Matrix3r MechanicalStrain( const Eigen::MatrixXr& displacement,
                                      const Eigen::MatrixXr& shapeGradient,
                                      int dimension,
                                      mfem::ElementTransformation& transformation,
                                      const mfem::IntegrationPoint& integrationPoint );

    const mfem::IntegrationRule& IntegrationRule( const mfem::FiniteElement& element ) const;

    const J2PlasticityMaterial& mMaterial;
    PointStorage& mPointStorage;
    StressFreeDeformationModel mStressFreeDeformations;
    Eigen::Matrix<mfem::real_t, 6, Eigen::Dynamic> mB;
    int mStepDepth{ 0 };
    bool mStepRejected{ false };
};

extern template class J2PlasticityIntegrator<J2PlasticityPointStorage>;
} // namespace plugin

#include "J2PlasticityIntegrator.tpp"
