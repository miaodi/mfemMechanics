#include "FEMPlugin.h"
#include "Material.h"
#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <unsupported/Eigen/KroneckerProduct>

using namespace std;
using namespace Eigen;
int main() {

  mfem::DenseMatrix dm;
  dm.SetSize(5);
  dm = 1.0;
  dm(1, 3) = 2.;
  dm.Print(cout, 10);

  auto ptr = dm.Data();
  Map<Eigen::Matrix<double, 5, 5>> elmat(ptr);
  elmat += MatrixXd::Random(5, 5);
  dm.Print(cout, 10);

  MatrixXd test = MatrixXd::Random(3, 3);
  MatrixXd test2 = Eigen::kroneckerProduct(test, MatrixXd::Identity(3, 3));
  cout<<test2<<endl;
  // mfem::IsoparametricTransformation et1;
  // et1.SetPointMat(dm);
  // mfem::IsoparametricTransformation et2(et1);
  // et2.GetPointMat().Print();
  // mfem::NonlinearElasticityIntegrator et3;
  // et3.resizeRefEleTransVec(10);
  return 0;
}