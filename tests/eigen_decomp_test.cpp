#include "SymmetricEigensolver3x3.hpp"
#include "util.h"
#include <Eigen/Dense>
#include <gtest/gtest.h>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>

// compare the difference between using dual as constant and using double as constant
TEST( eigen_decomp, decomp1 )
{
    Eigen::Vector6d strain{ 2., 1., 1., 0., 0., 0. };

    Eigen::Vector3d eval;
    Eigen::Matrix3d evec;
    gte::NISymmetricEigensolver3x3<double> eig;

    eig( strain( 0 ), strain( 3 ) / 2, strain( 5 ) / 2, strain( 1 ), strain( 4 ) / 2, strain( 2 ), 1, eval, evec );

    std::cout << eval.transpose() << std::endl;
    std::cout << evec << std::endl;
}