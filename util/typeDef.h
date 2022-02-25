#pragma once

#include <Eigen/Dense>
namespace Eigen
{
using Matrix4d = Matrix<double, 4, 4>;
using Matrix5d = Matrix<double, 5, 5>;
using Matrix6d = Matrix<double, 6, 6>;

using Vector4d = Matrix<double, 4, 1>;
using Vector5d = Matrix<double, 5, 1>;
using Vector6d = Matrix<double, 6, 1>;
} // namespace Eigen