#pragma once
#include "FEMPlugin.h"
#include "Material.h"

namespace plugin
{

// class NonlinearCompositeSolidShellIntegrator : public NonlinearFormMaterialIntegratorLambda
// {
// public:
//     NonlinearCompositeSolidShellIntegrator( ElasticMaterial& m ) : NonlinearFormMaterialIntegratorLambda( m )
//     {
//         mL.resize( 5, 24 );
//         mH.resize( 5, 5 );
//         mAlpha.resize( 5 );
//         mGeomStiff.resize( 24, 24 );
//     }

//     // virtual void AssembleElementVector( const mfem::FiniteElement& el,
//     //                                     mfem::ElementTransformation& Ttr,
//     //                                     const mfem::Vector& elfun,
//     //                                     mfem::Vector& elvect );

//     virtual void AssembleElementGrad( const mfem::FiniteElement& el,
//                                       mfem::ElementTransformation& Ttr,
//                                       const mfem::Vector& elfun,
//                                       mfem::DenseMatrix& elmat );

//     void matrixB( const int dof, const int dim, const mfem::IntegrationPoint& ip );

//     /** @brief Computes the integral of W(Jacobian(Trt)) over a target zone
//         @param[in] el     Type of FiniteElement.
//         @param[in] Ttr    Represents ref->target coordinates transformation.
//         @param[in] elfun  Physical coordinates of the zone. */
//     virtual double GetElementEnergy( const mfem::FiniteElement& el, mfem::ElementTransformation& Ttr, const mfem::Vector& elfun )
//     {
//         return 0;
//     }

// protected:
//     Eigen::Matrix<double, 3, 3> mg, mGCovariant, mGContravariant, mgA, mgB, mgC, mgD, mgA1, mgA2, mgA3, mgA4;
//     Eigen::Matrix<double, 6, 24> mB;
//     Eigen::MatrixXd mGeomStiff;
//     Eigen::Matrix<double, 8, 3> mDShape, mDShapeA, mDShapeB, mDShapeC, mDShapeD, mDShapeA1, mDShapeA2, mDShapeA3, mDShapeA4;
//     Eigen::Matrix6d mStiffModuli, mTransform;
//     Eigen::MatrixXd mL, mH;
//     Eigen::VectorXd mAlpha;
// };
} // namespace plugin