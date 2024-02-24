// C++ includes
#include <iostream>

// autodiff include
#include "Plugin.h"
#include <Eigen/Dense>
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>
#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>
#include <cmath>
#include <iomanip>
#include <unsupported/Eigen/KroneckerProduct>
using namespace autodiff;
int main()
{
    auto potential_energy = []( const autodiff::Vector6dual2nd& strainVec )
    {
        auto strainTensor = util::InverseVoigt( strainVec, true );
        auto eigenValues = strainTensor.eigenvalues();
        // auto eigenVectors = strainTensor.eigenvectors();
        std::cout << strainTensor << std::endl;
        std::cout << eigenValues << std::endl;
    };

    autodiff::Vector6dual2nd strain;
    strain << 1, 2, 3, 4, 5, 6;
    potential_energy( strain );
    return 0;
}