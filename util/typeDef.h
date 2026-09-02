#pragma once

#include <Eigen/Dense>
#include <autodiff/forward/dual.hpp>
#include <mfem.hpp>

namespace Eigen
{
using MatrixXr = Matrix<mfem::real_t, Dynamic, Dynamic>;
using Matrix2r = Matrix<mfem::real_t, 2, 2>;
using Matrix3r = Matrix<mfem::real_t, 3, 3>;
using Matrix4r = Matrix<mfem::real_t, 4, 4>;
using Matrix5r = Matrix<mfem::real_t, 5, 5>;
using Matrix6r = Matrix<mfem::real_t, 6, 6>;

using VectorXr = Matrix<mfem::real_t, Dynamic, 1>;
using Vector2r = Matrix<mfem::real_t, 2, 1>;
using Vector3r = Matrix<mfem::real_t, 3, 1>;
using Vector4r = Matrix<mfem::real_t, 4, 1>;
using Vector5r = Matrix<mfem::real_t, 5, 1>;
using Vector6r = Matrix<mfem::real_t, 6, 1>;
} // namespace Eigen

namespace autodiff
{
using Vector6dual2nd = Eigen::Vector<dual2nd, 6>;
} // namespace autodiff
