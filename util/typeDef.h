#pragma once

#include <Eigen/Dense>
#include <autodiff/forward/dual.hpp>
namespace Eigen
{
using Matrix4d = Matrix<double, 4, 4>;
using Matrix5d = Matrix<double, 5, 5>;
using Matrix6d = Matrix<double, 6, 6>;

using Vector4d = Matrix<double, 4, 1>;
using Vector5d = Matrix<double, 5, 1>;
using Vector6d = Matrix<double, 6, 1>;
} // namespace Eigen
namespace autodiff
{
using Vector6dual2nd = Eigen::Vector<dual2nd, 6>;
}