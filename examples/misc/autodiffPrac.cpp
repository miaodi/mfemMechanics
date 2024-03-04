// C++ includes
#include <iostream>

// autodiff include
#include "Plugin.h"
#include "mfem.hpp"
#include <Eigen/Dense>
#include <SymmetricEigensolver3x3.hpp>
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>
#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>
#include <cmath>
#include <iomanip>
#include <ringbuffer.hpp>
#include <taco.h>
#include <unsupported/Eigen/KroneckerProduct>
using namespace autodiff;

class NonlinearInternalPenaltyIntegrator : public mfem::NonlinearFormIntegrator
{
public:
    NonlinearInternalPenaltyIntegrator( const double penalty = 1e10 ) : mfem::NonlinearFormIntegrator(), p{ penalty }
    {
    }

    virtual void AssembleFaceVector( const mfem::FiniteElement& el1,
                                     const mfem::FiniteElement& el2,
                                     mfem::FaceElementTransformations& Tr,
                                     const mfem::Vector& elfun,
                                     mfem::Vector& elvect ) override
    {
        int vdim = Tr.GetSpaceDim();
        int dof1 = el1.GetDof();
        int dof2 = el2.GetDof();
        int dof = dof1 + dof2;
        MFEM_ASSERT( Tr.Elem2No >= 0, "CZMIntegrator is an internal bdr integrator" );

        shape1.SetSize( dof1 );
        shape2.SetSize( dof2 );
        u.SetSize( vdim );
        elvect.SetSize( dof * vdim );
        elvect = 0.0;

        const mfem::IntegrationRule* ir = IntRule;
        if ( ir == NULL )
        {
            int intorder = 2 * el1.GetOrder();
            ir = &mfem::IntRules.Get( Tr.GetGeometryType(), intorder );
        }

        for ( int i = 0; i < ir->GetNPoints(); i++ )
        {
            const mfem::IntegrationPoint& ip = ir->IntPoint( i );

            // Set the integration point in the face and the neighboring element
            Tr.SetAllIntPoints( &ip );

            // Access the neighboring element's integration point
            const mfem::IntegrationPoint& eip1 = Tr.GetElement1IntPoint();
            const mfem::IntegrationPoint& eip2 = Tr.GetElement2IntPoint();

            el1.CalcShape( eip1, shape1 );
            el2.CalcShape( eip2, shape2 );

            matrixBT( dof1, dof2, vdim );
            mBT.MultTranspose( elfun, u );
            mBT.AddMult( u, elvect, p * ip.weight * Tr.Weight() );
        }
    }

    virtual void AssembleFaceGrad( const mfem::FiniteElement& el1,
                                   const mfem::FiniteElement& el2,
                                   mfem::FaceElementTransformations& Tr,
                                   const mfem::Vector& elfun,
                                   mfem::DenseMatrix& elmat ) override
    {
        int vdim = Tr.GetSpaceDim();
        int dof1 = el1.GetDof();
        int dof2 = el2.GetDof();
        int dof = dof1 + dof2;
        MFEM_ASSERT( Tr.Elem2No >= 0, "CZMIntegrator is an internal bdr integrator" );
        shape1.SetSize( dof1 );
        shape2.SetSize( dof2 );

        elmat.SetSize( dof * vdim );
        elmat = 0.0;

        const mfem::IntegrationRule* ir = IntRule;
        if ( ir == NULL )
        {
            int intorder = 2 * el1.GetOrder();
            ir = &mfem::IntRules.Get( Tr.GetGeometryType(), intorder );
        }

        for ( int i = 0; i < ir->GetNPoints(); i++ )
        {
            const mfem::IntegrationPoint& ip = ir->IntPoint( i );

            // Set the integration point in the face and the neighboring element
            Tr.SetAllIntPoints( &ip );

            // Access the neighboring element's integration point
            const mfem::IntegrationPoint& eip1 = Tr.GetElement1IntPoint();
            const mfem::IntegrationPoint& eip2 = Tr.GetElement2IntPoint();
            el1.CalcShape( eip1, shape1 );
            el2.CalcShape( eip2, shape2 );

            matrixBT( dof1, dof2, vdim );
            AddMult_a_ABt( p * ip.weight * Tr.Weight(), mBT, mBT, elmat );
        }
    }

    void matrixBT( const int dof1, const int dof2, const int dim )
    {
        mBT.SetSize( dim * ( dof1 + dof2 ), dim );
        mBT = 0.;

        for ( int i = 0; i < dof1; i++ )
        {
            for ( int j = 0; j < dim; j++ )
            {
                mBT( i + j * dof1, j ) = shape1( i );
            }
        }
        for ( int i = 0; i < dof2; i++ )
        {
            for ( int j = 0; j < dim; j++ )
            {
                mBT( i + j * dof2 + dim * dof1, j ) = -shape2( i );
            }
        }
    }

protected:
    mfem::Vector shape1, shape2, u;
    mfem::DenseMatrix mBT;
    double p;
};

mfem::SparseMatrix* nonlinear_getgredient( int dim, int num_elements, int order )
{
    using namespace mfem;
    auto MakeCartesianNonaligned = []( const int dim, const int ne )
    {
        Mesh mesh;
        mesh = Mesh::LoadFromFile( "../../data/DCB.msh", 1 );
        if ( dim == 2 )
        {
            mesh = Mesh::MakeCartesian2D( ne, ne, Element::QUADRILATERAL, 1, 1.0, 1.0 );
        }
        else
        {
            mesh = Mesh::MakeCartesian3D( ne, ne, ne, Element::HEXAHEDRON, 1.0, 1.0, 1.0 );
        }

        // Remap vertices so that the mesh is not aligned with axes.
        for ( int i = 0; i < mesh.GetNV(); ++i )
        {
            double* vcrd = mesh.GetVertex( i );
            vcrd[1] += 0.2 * vcrd[0];
            if ( dim == 3 )
            {
                vcrd[2] += 0.3 * vcrd[0];
            }
        }

        return mesh;
    };

    Mesh mesh = MakeCartesianNonaligned( dim, num_elements );

    DG_FECollection fec( order, dim );
    FiniteElementSpace fespace( &mesh, &fec, dim );

    NonlinearForm nlf( &fespace );

    nlf.AddInteriorFaceIntegrator( new NonlinearInternalPenaltyIntegrator() );

    Vector x( fespace.GetTrueVSize() );

    Operator* op = &nlf.GetGradient( x );
    auto spmat = static_cast<SparseMatrix*>( op );
    spmat->SortColumnIndices();
    return spmat;
}

#ifdef MFEM_USE_MPI

mfem::HypreParMatrix* parnonlinear_getgredient( int dim, int num_elements, int order )
{
    using namespace mfem;

    Mpi::Init();
    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    int size;
    MPI_Comm_size( MPI_COMM_WORLD, &size );

    auto MakeCartesianNonaligned = []( const int dim, const int ne )
    {
        Mesh mesh;
        // mesh = Mesh::LoadFromFile( "../../data/DCB.msh", 1 );
        if ( dim == 2 )
        {
            mesh = Mesh::MakeCartesian2D( ne, ne, Element::QUADRILATERAL, 1, 1.0, 1.0 );
        }
        else
        {
            mesh = Mesh::MakeCartesian3D( ne, ne, ne, Element::HEXAHEDRON, 1.0, 1.0, 1.0 );
        }

        // Remap vertices so that the mesh is not aligned with axes.
        for ( int i = 0; i < mesh.GetNV(); ++i )
        {
            double* vcrd = mesh.GetVertex( i );
            vcrd[1] += 0.2 * vcrd[0];
            if ( dim == 3 )
            {
                vcrd[2] += 0.3 * vcrd[0];
            }
        }

        return mesh;
    };

    Mesh mesh = MakeCartesianNonaligned( dim, num_elements );
    ParMesh pmesh( MPI_COMM_WORLD, mesh );

    DG_FECollection fec( order, dim );
    ParFiniteElementSpace fespace( &pmesh, &fec, dim );
    auto dof_truedof = fespace.Dof_TrueDof_Matrix();
    dof_truedof->PrintMatlab( mfem::out );

    Vector Nu( pmesh.attributes.Max() );
    Nu = .0;
    PWConstCoefficient nu_func( Nu );

    Vector E( pmesh.attributes.Max() );
    E = 324E7;
    PWConstCoefficient E_func( E );

    IsotropicElasticMaterial iem( E_func, nu_func );

    plugin::Memorize mm( &pmesh );

    auto intg = new plugin::NonlinearElasticityIntegrator( iem, mm );

    std::cout << fespace.GlobalTrueVSize() << std::endl;

    ParNonlinearForm nlf( &fespace );
    nlf.AddDomainIntegrator( intg );
    //   nlf.AddInteriorFaceIntegrator( new NonlinearInternalPenaltyIntegrator() );
    Vector x( fespace.GetTrueVSize() );
    Operator* op = &nlf.GetGradient( x );
    auto spmat = dynamic_cast<HypreParMatrix*>( op );
    return spmat;
}

#endif

int main()
{
    // const double lambda = 100;
    // const double mu = 50;
    // const double k = 0.;
    // double phi = .2;
    // auto strain_split = []( const auto& strainTensor )
    // {
    //     using T = typename std::decay_t<decltype( strainTensor( 0, 0 ) )>;

    //     Eigen::Matrix<T, 3, 1> eval;
    //     Eigen::Matrix<T, 3, 3> evec;
    //     gte::SymmetricEigensolver3x3<T> eig;

    //     eig( strainTensor( 0, 0 ), strainTensor( 0, 1 ) / 2, strainTensor( 0, 2 ) / 2, strainTensor( 1, 1 ),
    //          strainTensor( 1, 2 ) / 2, strainTensor( 2, 2 ), true, 1, eval, evec );

    //     Eigen::Matrix<T, 3, 3> strainPos;
    //     Eigen::Matrix<T, 3, 3> strainNeg;
    //     strainPos.setZero();
    //     strainNeg.setZero();

    //     for ( int i = 0; i < 3; i++ )
    //     {
    //         if ( eval( i ) < 0 )
    //             strainNeg += eval( i ) * ( evec.col( i ) * evec.col( i ).transpose() );
    //         else
    //             strainPos += eval( i ) * ( evec.col( i ) * evec.col( i ).transpose() );
    //     }
    //     return std::tuple<Eigen::Matrix<T, 3, 3>, Eigen::Matrix<T, 3, 3>>{ strainPos, strainNeg };
    // };

    // auto potential_energy = [&]( const auto& strainVec, const int stat )
    // {
    //     using T = typename std::decay_t<decltype( strainVec( 0 ) )>;

    //     auto curlyBracPos = []( const T& val ) { return val > 0 ? val : static_cast<T>( 0 ); };
    //     auto curlyBracNeg = []( const T& val ) { return val < 0 ? val : static_cast<T>( 0 ); };

    //     const auto strainTensor = util::InverseVoigt( strainVec, true );

    //     auto [strainPos, strainNeg] = strain_split( strainTensor );
    //     const T psiPos = lambda / 2 * autodiff::detail::pow( curlyBracPos( strainTensor.trace() ), 2 ) +
    //                      mu * strainPos.array().square().sum();
    //     if ( stat == 0 )
    //         return psiPos;
    //     const T psiNeg = lambda / 2 * autodiff::detail::pow( curlyBracNeg( strainTensor.trace() ), 2 ) +
    //                      mu * strainNeg.array().square().sum();
    //     if ( stat == 1 )
    //         return psiNeg;
    //     const T res = ( ( 1 - k ) * std::pow( 1 - phi, 2 ) + k ) * psiPos + psiNeg;
    //     return res;
    // };

    // std::random_device rd;
    // std::mt19937 generator( rd() ); // here you could also set a seed
    // std::uniform_real_distribution<double> distribution( -1, 1. );
    // Eigen::Vector6d strainDouble{ distribution( generator ), distribution( generator ), distribution( generator ),
    //                               distribution( generator ), distribution( generator ), distribution( generator ) };
    // Eigen::Vector6dual2nd strain{ strainDouble( 0 ), strainDouble( 1 ), strainDouble( 2 ),
    //                               strainDouble( 3 ), strainDouble( 4 ), strainDouble( 5 ) };
    // // auto psi = potential_energy( strain );
    // // std::cout << "strain: \n" << strain << std::endl;

    // autodiff::dual2nd u;
    // Eigen::VectorXd residual = autodiff::gradient( potential_energy, autodiff::wrt( strain ), autodiff::at( strain, 2
    // ), u ); autodiff::Vector6dual2nd strainIncre{ strainDouble( 0 ) + 1e-8, strainDouble( 1 ), strainDouble( 2 ),
    //                                       strainDouble( 3 ),        strainDouble( 4 ), strainDouble( 5 ) };
    // std::cout << std::setprecision( 16 )
    //           << autodiff::detail::val( potential_energy( strainIncre, 2 ) - potential_energy( strain, 2 ) ) / 1e-8
    //           << std::endl;
    // autodiff::VectorXdual g;
    // // Eigen::MatrixXd stiffness = autodiff::hessian( potential_energy, autodiff::wrt( strain ), autodiff::at( strain
    // ), u, g ); std::cout << std::setprecision( 16 ) << residual.transpose() << std::endl;
    // // std::cout << stiffness.transpose() << std::endl;

    // {
    //     taco::IndexVar I, J, K, L;
    //     // taco::IndexVar m, n, o, p;

    //     taco::Format sd2( { taco::Dense, taco::Dense } );

    //     // Create tensors
    //     taco::Tensor<double> I2( { 3, 3 }, sd2 );
    //     taco::Tensor<double> strainTensor( { 3, 3 }, sd2 );
    //     taco::Tensor<double> strainPosTensor( { 3, 3 }, sd2 );
    //     taco::Tensor<double> strainNegTensor( { 3, 3 }, sd2 );
    //     taco::Tensor<double> stressTensor( { 3, 3 }, sd2 );

    //     // Insert data identity tensor I2
    //     I2.insert( { 0, 0 }, 1. );
    //     I2.insert( { 1, 1 }, 1. );
    //     I2.insert( { 2, 2 }, 1. );

    //     // Pack inserted data as described by the formats
    //     I2.pack();

    //     auto change_storage = []( Eigen::Matrix3d& eigen, taco::Tensor<double>& tensor )
    //     {
    //         auto _array = taco::makeArray<double>( eigen.data(), 9 );
    //         taco::TensorStorage& _storage = tensor.getStorage();
    //         _storage.setValues( _array );
    //         tensor.setStorage( _storage );
    //     };
    //     auto strainTensorEigen = util::InverseVoigt( strainDouble, true );
    //     auto [strainPosTensorEigen, strainNegTensorEigen] = strain_split( strainTensorEigen );

    //     // Eigen::Matrix3d stressTensorEigen;

    //     change_storage( strainTensorEigen, strainTensor );
    //     change_storage( strainPosTensorEigen, strainPosTensor );
    //     change_storage( strainNegTensorEigen, strainNegTensor );
    //     const double pos = strainTensorEigen.trace() > 0 ? strainTensorEigen.trace() : 0;
    //     const double neg = strainTensorEigen.trace() < 0 ? strainTensorEigen.trace() : 0;
    //     // stressTensor( i, j ) = static_cast<double>( ( 1. - k ) * std::pow( 1. - phi, 2 ) ) *
    //     //                        ( lambda * pos * I2( i, j ) + 2 * mu * strainPosTensor( i, j ) );
    //     stressTensor( I, J ) =
    //         ( ( 1 - k ) * std::pow( 1 - phi, 2 ) + k ) * ( lambda * pos * I2( I, J ) + 2 * mu * strainPosTensor( I, J
    //         ) ) + lambda * neg * I2( I, J ) + 2 * mu * strainNegTensor( I, J );
    //     std::cout << std::setprecision( 16 ) << stressTensor( 0, 0 ) << " " << stressTensor( 0, 1 ) << " "
    //               << stressTensor( 0, 2 ) << " " << std::endl;
    //     std::cout << std::setprecision( 16 ) << stressTensor( 1, 0 ) << " " << stressTensor( 1, 1 ) << " "
    //               << stressTensor( 1, 2 ) << " " << std::endl;
    //     std::cout << std::setprecision( 16 ) << stressTensor( 2, 0 ) << " " << stressTensor( 2, 1 ) << " "
    //               << stressTensor( 2, 2 ) << " " << std::endl;

    //     // std::cout << strainPos << std::endl;
    // }

    // {
    //     jnk0le::Ringbuffer<int, 8> ring;
    //     for ( int i = 0; i < 10; i++ )
    //     {
    //         std::cout << i << " " << ring.insert( i ) << " " << *ring.peek() << std::endl;
    //     }
    //     for ( int i = 0; i < 8; i++ )
    //     {
    //         std::cout << i << " " << *ring.peek() << std::endl;
    //         ring.remove();
    //     }
    // }

    {
        auto hypre_mat = parnonlinear_getgredient( 2, 2, 1 );
        std::cout << "hypre_mat->GetNumRows(): " << hypre_mat->GetNumRows() << " OwnsDiag: " << hypre_mat->NNZ() << std::endl;

        // mfem::SparseMatrix diag;
        // hypre_mat->GetDiag(diag);
        // diag.PrintMatlab(mfem::out);
        auto spmat = nonlinear_getgredient( 2, 2, 1 );
    }
    return 0;
}