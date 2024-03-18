


// void NonlinearCompositeSolidShellIntegrator::AssembleElementGrad( const mfem::FiniteElement& el,
//                                                                   mfem::ElementTransformation& Ttr,
//                                                                   const mfem::Vector& elfun,
//                                                                   mfem::DenseMatrix& elmat )
// {
//     double w = 0;
//     int dof = el.GetDof(), dim = el.GetDim();

//     MFEM_ASSERT( dim == 3 && dof == 8, "NonlinearCompositeSolidShellIntegrator only support linearHex elements" );

//     Eigen::Map<const Eigen::MatrixXd> u( elfun.GetData(), dof, dim );
//     elmat.SetSize( dof * dim );
//     elmat = 0.0;

//     Eigen::Map<Eigen::MatrixXd> eigenMat( elmat.Data(), dof * dim, dof * dim );

//     mfem::DenseMatrix mat;
//     mfem::IntegrationPoint ip;

//     // from [-1, 1] to [0, 1]
//     auto convert = []( double& x ) { x = ( x + 1 ) / 2.; };
//     double pt[3];
//     pt[0] = .5, pt[1] = .5, pt[2] = .5;
//     Ttr.SetIntPoint( &ip );
//     mMaterialModel->at( Ttr, ip );
//     mMaterialModel->updateRefModuli();
//     mStiffModuli = mMaterialModel->getRefModuli();
//     const Eigen::Matrix3d orthonormalBasis = Eigen::Matrix3d::Identity();

//     auto preprocessColl = [&]( Eigen::Matrix<double, 3, 3>& g, Eigen::Matrix<double, 8, 3>& DShape )
//     {
//         mat.UseExternalData( mGCovariant.data(), mGCovariant.rows(), mGCovariant.cols() );
//         mat = Ttr.Jacobian();
//         mat.UseExternalData( DShape.data(), DShape.rows(), DShape.cols() );
//         el.CalcDShape( ip, mat );
//         g = mGCovariant + u.transpose() * DShape;
//     };

//     // prepare collocation points
//     {
//         // point A
//         pt[0] = -1, pt[1] = 0, pt[2] = 0;
//         convert( pt[0] ), convert( pt[1] ), convert( pt[2] );
//         ip.Set3( pt[0], pt[1], pt[2] );
//         preprocessColl( mgA, mDShapeA );
//         // point B
//         pt[0] = 0, pt[1] = -1, pt[2] = 0;
//         convert( pt[0] ), convert( pt[1] ), convert( pt[2] );
//         ip.Set3( pt[0], pt[1], pt[2] );
//         preprocessColl( mgB, mDShapeB );
//         // point C
//         pt[0] = 1, pt[1] = 0, pt[2] = 0;
//         convert( pt[0] ), convert( pt[1] ), convert( pt[2] );
//         ip.Set3( pt[0], pt[1], pt[2] );
//         preprocessColl( mgC, mDShapeC );
//         // point D
//         pt[0] = 0, pt[1] = 1, pt[2] = 0;
//         convert( pt[0] ), convert( pt[1] ), convert( pt[2] );
//         ip.Set3( pt[0], pt[1], pt[2] );
//         preprocessColl( mgD, mDShapeD );

//         // point A1
//         pt[0] = -1, pt[1] = -1, pt[2] = 0;
//         convert( pt[0] ), convert( pt[1] ), convert( pt[2] );
//         ip.Set3( pt[0], pt[1], pt[2] );
//         preprocessColl( mgA1, mDShapeA1 );
//         // point A2
//         pt[0] = 1, pt[1] = -1, pt[2] = 0;
//         convert( pt[0] ), convert( pt[1] ), convert( pt[2] );
//         ip.Set3( pt[0], pt[1], pt[2] );
//         preprocessColl( mgA2, mDShapeA2 );
//         // point A3
//         pt[0] = 1, pt[1] = 1, pt[2] = 0;
//         convert( pt[0] ), convert( pt[1] ), convert( pt[2] );
//         ip.Set3( pt[0], pt[1], pt[2] );
//         preprocessColl( mgA3, mDShapeA3 );
//         // point A4
//         pt[0] = -1, pt[1] = 1, pt[2] = 0;
//         convert( pt[0] ), convert( pt[1] ), convert( pt[2] );
//         ip.Set3( pt[0], pt[1], pt[2] );
//         preprocessColl( mgA4, mDShapeA4 );

//         std::cout << mDShapeB << std::endl << std::endl;
//         std::cout << mDShapeA4 << std::endl;

//         mfem::Vector v;
//         v.SetSize( 8 );
//         ip.Set3( 0, 0, 0 );
//         el.CalcShape( ip, v );
//         v.Print();
//         ip.Set3( 1, 0, 0 );
//         el.CalcShape( ip, v );
//         v.Print();
//         ip.Set3( 1, 1, 0 );
//         el.CalcShape( ip, v );
//         v.Print();
//         ip.Set3( 0, 1, 0 );
//         el.CalcShape( ip, v );
//         v.Print();
//         ip.Set3( 0, 0, 1 );
//         el.CalcShape( ip, v );
//         v.Print();
//         ip.Set3( 1, 0, 1 );
//         el.CalcShape( ip, v );
//         v.Print();
//         ip.Set3( 1, 1, 1 );
//         el.CalcShape( ip, v );
//         v.Print();
//         ip.Set3( 0, 1, 1 );
//         el.CalcShape( ip, v );
//         v.Print();
//     }

//     const mfem::IntegrationRule* ir = IntRule;
//     if ( !ir )
//     {
//         ir = &( mfem::IntRules.Get( el.GetGeomType(), 2 * el.GetOrder() + 1 ) ); // <---
//     }

//     for ( int i = 0; i < ir->GetNPoints(); i++ )
//     {
//         const mfem::IntegrationPoint& ip = ir->IntPoint( i );
//         Ttr.SetIntPoint( &ip );

//         preprocessColl( mg, mDShape );
//         matrixB( dof, dim, ip );

//         mMaterialModel->at( Ttr, ip );
//         mMaterialModel->updateRefModuli();

//         w = ip.weight * Ttr.Weight();
//         mGContravariant = mGCovariant.inverse();
//         Eigen::Matrix3d T = orthonormalBasis.transpose() * mGContravariant;
//         mTransform = util::TransformationVoigtForm( T );
//         matrixB( dof, dim, ip );
//         // mGeomStiff =
//         //     ( w * mGShapeEig * mMaterialModel->getPK2StressTensor().block( 0, 0, dim, dim ) * mGShapeEig.transpose() ).eval();
//         eigenMat += w * mB.transpose() * ( mTransform.transpose() * mStiffModuli * mTransform ) * mB;
//         // for ( int j = 0; j < dim; j++ )
//         // {
//         //     eigenMat.block( j * dof, j * dof, dof, dof ) += mGeomStiff;
//         // }
//     }
// }

// void NonlinearCompositeSolidShellIntegrator::matrixB( const int dof, const int dim, const mfem::IntegrationPoint& ip )
// {
//     // from [0, 1] to [-1, 1]
//     auto convert = []( double& x ) { x = 2 * x - 1; };
//     double pt[3];
//     ip.Get( pt, 3 );
//     convert( pt[0] );
//     convert( pt[1] );
//     convert( pt[2] );
//     for ( int i = 0; i < dof; i++ )
//     {
//         mB( 0, i + 0 * dof ) = mDShape( i, 0 ) * mg( 0, 0 );
//         mB( 0, i + 1 * dof ) = mDShape( i, 0 ) * mg( 1, 0 );
//         mB( 0, i + 2 * dof ) = mDShape( i, 0 ) * mg( 2, 0 );

//         mB( 1, i + 0 * dof ) = mDShape( i, 1 ) * mg( 0, 1 );
//         mB( 1, i + 1 * dof ) = mDShape( i, 1 ) * mg( 1, 1 );
//         mB( 1, i + 2 * dof ) = mDShape( i, 1 ) * mg( 2, 1 );

//         mB( 2, i + 0 * dof ) = 1. / 4 *
//                                ( ( 1 - 1 * pt[0] ) * ( 1 - 1 * pt[1] ) * mgA1( 0, 2 ) * mDShapeA1( i, 2 ) +
//                                  ( 1 + 1 * pt[0] ) * ( 1 - 1 * pt[1] ) * mgA2( 0, 2 ) * mDShapeA2( i, 2 ) +
//                                  ( 1 + 1 * pt[0] ) * ( 1 + 1 * pt[1] ) * mgA3( 0, 2 ) * mDShapeA3( i, 2 ) +
//                                  ( 1 - 1 * pt[0] ) * ( 1 + 1 * pt[1] ) * mgA4( 0, 2 ) * mDShapeA4( i, 2 ) );
//         mB( 2, i + 1 * dof ) = 1. / 4 *
//                                ( ( 1 - 1 * pt[0] ) * ( 1 - 1 * pt[1] ) * mgA1( 1, 2 ) * mDShapeA1( i, 2 ) +
//                                  ( 1 + 1 * pt[0] ) * ( 1 - 1 * pt[1] ) * mgA2( 1, 2 ) * mDShapeA2( i, 2 ) +
//                                  ( 1 + 1 * pt[0] ) * ( 1 + 1 * pt[1] ) * mgA3( 1, 2 ) * mDShapeA3( i, 2 ) +
//                                  ( 1 - 1 * pt[0] ) * ( 1 + 1 * pt[1] ) * mgA4( 1, 2 ) * mDShapeA4( i, 2 ) );
//         mB( 2, i + 2 * dof ) = 1. / 4 *
//                                ( ( 1 - 1 * pt[0] ) * ( 1 - 1 * pt[1] ) * mgA1( 2, 2 ) * mDShapeA1( i, 2 ) +
//                                  ( 1 + 1 * pt[0] ) * ( 1 - 1 * pt[1] ) * mgA2( 2, 2 ) * mDShapeA2( i, 2 ) +
//                                  ( 1 + 1 * pt[0] ) * ( 1 + 1 * pt[1] ) * mgA3( 2, 2 ) * mDShapeA3( i, 2 ) +
//                                  ( 1 - 1 * pt[0] ) * ( 1 + 1 * pt[1] ) * mgA4( 2, 2 ) * mDShapeA4( i, 2 ) );

//         mB( 3, i + 0 * dof ) = mDShape( i, 1 ) * mg( 0, 0 ) + mDShape( i, 0 ) * mg( 0, 1 );
//         mB( 3, i + 1 * dof ) = mDShape( i, 1 ) * mg( 1, 0 ) + mDShape( i, 0 ) * mg( 1, 1 );
//         mB( 3, i + 2 * dof ) = mDShape( i, 1 ) * mg( 2, 0 ) + mDShape( i, 0 ) * mg( 2, 1 );

//         mB( 4, i + 0 * dof ) = 1. / 2 *
//                                ( ( 1 - pt[0] ) * ( mgB( 0, 1 ) * mDShapeB( i, 2 ) + mgB( 0, 2 ) * mDShapeB( i, 1 ) ) +
//                                  ( 1 + pt[0] ) * ( mgD( 0, 1 ) * mDShapeD( i, 2 ) + mgD( 0, 2 ) * mDShapeD( i, 1 ) ) );
//         mB( 4, i + 1 * dof ) = 1. / 2 *
//                                ( ( 1 - pt[0] ) * ( mgB( 1, 1 ) * mDShapeB( i, 2 ) + mgB( 1, 2 ) * mDShapeB( i, 1 ) ) +
//                                  ( 1 + pt[0] ) * ( mgD( 1, 1 ) * mDShapeD( i, 2 ) + mgD( 1, 2 ) * mDShapeD( i, 1 ) ) );
//         mB( 4, i + 2 * dof ) = 1. / 2 *
//                                ( ( 1 - pt[0] ) * ( mgB( 2, 1 ) * mDShapeB( i, 2 ) + mgB( 2, 2 ) * mDShapeB( i, 1 ) ) +
//                                  ( 1 + pt[0] ) * ( mgD( 2, 1 ) * mDShapeD( i, 2 ) + mgD( 2, 2 ) * mDShapeD( i, 1 ) ) );

//         mB( 5, i + 0 * dof ) = 1. / 2 *
//                                ( ( 1 - pt[1] ) * ( mgA( 0, 0 ) * mDShapeA( i, 2 ) + mgA( 0, 2 ) * mDShapeA( i, 0 ) ) +
//                                  ( 1 + pt[1] ) * ( mgC( 0, 0 ) * mDShapeC( i, 2 ) + mgC( 0, 2 ) * mDShapeC( i, 0 ) ) );
//         mB( 5, i + 1 * dof ) = 1. / 2 *
//                                ( ( 1 - pt[1] ) * ( mgA( 1, 0 ) * mDShapeA( i, 2 ) + mgA( 1, 2 ) * mDShapeA( i, 0 ) ) +
//                                  ( 1 + pt[1] ) * ( mgC( 1, 0 ) * mDShapeC( i, 2 ) + mgC( 1, 2 ) * mDShapeC( i, 0 ) ) );
//         mB( 5, i + 2 * dof ) = 1. / 2 *
//                                ( ( 1 - pt[1] ) * ( mgA( 2, 0 ) * mDShapeA( i, 2 ) + mgA( 2, 2 ) * mDShapeA( i, 0 ) ) +
//                                  ( 1 + pt[1] ) * ( mgC( 2, 0 ) * mDShapeC( i, 2 ) + mgC( 2, 2 ) * mDShapeC( i, 0 ) ) );
//     }
// }