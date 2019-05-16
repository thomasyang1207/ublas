// Copyright 2008 Gunter Winkler <guwi17@gmx.de>
// Distributed under the Boost Software License, Version 1.0. (See
// accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

// switch automatic singular check off
#define BOOST_UBLAS_TYPE_CHECK 0

#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/eigen_solver.hpp>
#include <boost/cstdlib.hpp>

#include "common/testhelper.hpp"

#include <iostream>
#include <sstream>

using namespace boost::numeric::ublas;
using std::string;

static const string matrix_IN[] = {
    // 1
    "[3,3]"
        "((-149.0,-50.0,-154.0),"
        "(  537.0,180.0, 546.0),"
        "(  -27.0, -9.0, -25.0))\0",
    // 2
    "[3,3]"
        "((2.0, 3.0,5.0),"
        "( 2.0,-3.0,7.0),"
        "( 4.0, 1.0,1.0))\0",
    // 3
    "[6,6]"
        "((7.0, 3.0, 4.0,-11.0,-9.0,-2.0),"
        "(-6.0, 4.0,-5.0,  7.0, 1.0,12.0),"
        "(-1.0,-9.0, 2.0,  2.0, 9.0, 1.0),"
        "(-8.0, 0.0,-1.0,  5.0, 0.0, 8.0),"
        "(-4.0, 3.0,-5.0,  7.0, 2.0,10.0),"
        "( 6.0, 1.0, 4.0,-11.0,-7.0,-1.0))\0",
    // 4
    "[4,4]"
        "((0.421761282626275,0.655740699156587, 0.678735154857774,0.655477890177557),"
        "( 0.915735525189067,0.0357116785741896,0.757740130578333,0.171186687811562),"
        "( 0.792207329559554,0.849129305868777, 0.743132468124916,0.706046088019609),"
        "( 0.959492426392903,0.933993247757551, 0.392227019534168,0.0318328463774207))\0",
    // 5
    "[10,10]"
        "((1.0, 2.0, 3.0, 4.0, 3.0, 6.0, 7.0, 8.0, 9.0, 10.0),"
        "( 1.0, 2.0, 3.0, 3.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0),"
        "( 3.0, 2.0, 3.0, 4.2, 5.0, 6.1, 7.0, 8.0, 9.0, 14.0),"
        "( 1.0, 2.1, 3.0, 4.0, 5.0, 6.0, 7.1, 8.0, 9.0, 10.0),"
        "( 1.0, 2.0, 3.0, 4.5, 5.0, 6.0, 7.0, 8.0, 9.0, 30.0),"
        "( 5.0, 6.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0),"
        "( 1.0, 2.0, 3.0,41.0, 5.0, 6.0, 7.0, 1.0, 9.0, 10.0),"
        "( 1.0, 2.0,32.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.3, 10.0),"
        "( 9.0, 2.0, 3.0, 4.0, 0.0, 6.0, 7.0, 8.0, 9.0, 10.0),"
        "(12.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0))\0",
    // 6
    "[10,10]"
        "((0, 2, 3, 4,  3, 6,  7, 8, 9,  10),"
        "( 1, 2, 3, 3,  5, 0,  7, 8, 9,  10),"
        "( 3, 2, 0, 4.2,5, 6.1,0, 8, 9,   0),"
        "( 1, 2, 3, 4,  5, 6,  0, 8, 9,   1),"
        "( 1, 2, 3, 4.5,5, 6,  7, 8, 9,   0),"
        "( 5, 6, 0, 4,  5, 6,  7, 8, 9,   1),"
        "( 0, 2, 3, 0,  5, 6,  7, 1, 9,   4),"
        "( 1, 2, 0, 4,  5, 6,  7, 8, 9.3, 0),"
        "( 0, 2, 3, 4,  0, 6,  7, 8, 9,  10),"
        "( 0, 2, 3, 4,  5, 6,  7, 8, 9,   0))\0",
    // 7
    "[10,10]"
        "((0,     2e-05, 3e-05, 4e-05,   3e-05, 6e-05,   7e-05, 8e-05, 9e-05,   0.0001),"
        "( 1e-05, 2e-05, 3e-05, 3e-05,   5e-05, 0,       7e-05, 8e-05, 9e-05,   0.0001),"
        "( 3e-05, 2e-05, 0,     4.2e-05, 5e-05, 6.1e-05, 0,     8e-05, 9e-05,   0),"
        "( 1e-05, 2e-05, 3e-05, 4e-05,   5e-05, 6e-05,   0,     8e-05, 9e-05,   1e-05),"
        "( 1e-05, 2e-05, 3e-05, 4.5e-05, 5e-05, 6e-05,   7e-05, 8e-05, 9e-05,   0),"
        "( 5e-05, 6e-05, 0,     4e-05,   5e-05, 6e-05,   7e-05, 8e-05, 9e-05,   1e-05),"
        "( 0,     2e-05, 3e-05, 0,       5e-05, 6e-05,   7e-05, 1e-05, 9e-05,   4e-05),"
        "( 1e-05, 2e-05, 0,     4e-05,   5e-05, 6e-05,   7e-05, 8e-05, 9.3e-05, 0),"
        "( 0,     2e-05, 3e-05, 4e-05,   0,     6e-05,   7e-05, 8e-05, 9e-05,   0.0001),"
        "( 0,     2e-05, 3e-05, 4e-05,   5e-05, 6e-05,   7e-05, 8e-05, 9e-05,   0))\0",
    // 8
    "[10,10]"
        "(( 4,  2, 3, 4,   3, 6,   7, 8, 9,  10),"
        "(  0,  2, 3, 3,   5, 0,   7, 8, 9,  10),"
        "(  3,  2, 0, 4.2, 5, 6.1, 0, 8, 9,   0),"
        "(  1,  2, 3, 4,   5, 6,   0, 8, 9,   1),"
        "(  3,  2, 3, 4.5, 5, 6,   0, 8, 9,   0),"
        "(  5,  2, 0, 4,   5, 6,   1, 8, 9,   1),"
        "(  0,  2, 3, 0,   5, 6,   7, 0, 9,   4),"
        "(  1, 13, 0, 4,   5, 6,   2, 8, 9.3, 0),"
        "(100,  2, 3, 0,   0, 6,   7, 8, 9,  10),"
        "(  0,  2, 3, 4,   5, 6,   0, 8, 9,   0))\0",
    // 9
    "[10,10]"
        "((0,     2e+10, 3e+10, 4e+10,   3e+10, 6e+10,   7e+10, 8e+10, 9e+10,   1e+11),"
        "( 1e+10, 2e+10, 3e+10, 3e+10,   5e+10, 0,       7e+10, 8e+10, 9e+10,   1e+11),"
        "( 3e+10, 2e+10, 0,     4.2e+10, 5e+10, 6.1e+10, 0,     8e+10, 9e+10,   0),"
        "( 1e+10, 2e+10, 3e+10, 4e+10,   5e+10, 6e+10,   0,     8e+10, 9e+10,   1e+10),"
        "( 1e+10, 2e+10, 3e+10, 4.5e+10, 5e+10, 6e+10,   7e+10, 8e+10, 9e+10,   0),"
        "( 5e+10, 6e+10, 0,     4e+10,   5e+10, 6e+10,   7e+10, 8e+10, 9e+10,   1e+10),"
        "( 0,     2e+10, 3e+10, 0,       5e+10, 6e+10,   7e+10, 1e+10, 9e+10,   4e+10),"
        "( 1e+10, 2e+10, 0,     4e+10,   5e+10, 6e+10,   7e+10, 8e+10, 9.3e+10, 0),"
        "( 0,     2e+10, 3e+10, 4e+10,   0,     6e+10,   7e+10, 8e+10, 9e+10,   1e+11),"
        "( 0,     2e+10, 3e+10, 4e+10,   5e+10, 6e+10,   7e+10, 8e+10, 9e+10,   0))\0",
    // 10
    "[3,3]"
        "((0.0,0.0,1.0),"
        "( 1.0,0.0,0.0),"
        "( 0.0,1.0,0.0))\0",
};

static const string matrix_EVALR[] = {
    "[3,3]"
        "((1,0.0,0.0),"
        "( 0.0,2,0.0),"
        "( 0.0,0.0,3))\0",
    "[3,3]"
        "((7.547183, 0.0,      0.0),"
        "( 0.0,     -3.773591, 0.0),"
        "( 0.0,      0.0,     -3.773591))\0",
    "[6,6]"
        "((5.0,0.0,0.0,0.0,0.0,0.0),"
        "( 0.0,5.0,0.0,0.0,0.0,0.0),"
        "( 0.0,0.0,1.0,0.0,0.0,0.0),"
        "( 0.0,0.0,0.0,1.0,0.0,0.0),"
        "( 0.0,0.0,0.0,0.0,4.0,0.0),"
        "( 0.0,0.0,0.0,0.0,0.0,3.0))\0",
    "[4,4]"
        "((2.44784, 0.00000, 0.00000, 0.00000),"
        "( 0.00000,-0.56040, 0.00000, 0.00000),"
        "( 0.00000, 0.00000,-0.56040, 0.00000),"
        "( 0.00000, 0.00000, 0.00000,-0.09461))\0",
    "[10,10]"
        "((66.1927, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000),"
        "(0.000000, -10.9351, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000),"
        "(0.000000, 0.000000, -7.02399, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000),"
        "(0.000000, 0.000000, 0.000000, 2.03077,  0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000),"
        "(0.000000, 0.000000, 0.000000, 0.000000, 2.03077,  0.000000, 0.000000, 0.000000, 0.000000, 0.000000),"
        "(0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 3.56282,  0.000000, 0.000000, 0.000000, 0.000000),"
        "(0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,-0.593575, 0.000000, 0.000000, 0.000000),"
        "(0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,-0.593575, 0.000000, 0.000000),"
        "(0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,-0.237288, 0.000000),"
        "(0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.566443))\0",
    "[10,10]"
        "((44.5477, 0,        0,       0,      0,       0,       0,       0,       0,        0),"
        "(  0,      -1.63051, 0,       0,      0,       0,       0,       0,       0,        0),"
        "(  0,      0,       -1.63051, 0,      0,       0,       0,       0,       0,        0),"
        "(  0,      0,        0,      -3.9098, 0,       0,       0,       0,       0,        0),"
        "(  0,      0,        0,       0,     -1.51227, 0,       0,       0,       0,        0),"
        "(  0,      0,        0,       0,      0,      -1.51227, 0,       0,       0,        0),"
        "(  0,      0,        0,       0,      0,       0,       2.59252, 0,       0,        0),"
        "(  0,      0,        0,       0,      0,       0,       0,       2.59252, 0,        0),"
        "(  0,      0,        0,       0,      0,       0,       0,       0,       0.731304, 0),"
        "(  0,      0,        0,       0,      0,       0,       0,       0,       0,        0.731304))\0",
    "[10,10]"
        "((0.000445477, 0, 0, 0, 0, 0, 0, 0, 0, 0),"
        "(0, -1.63051e-05, 0, 0, 0, 0, 0, 0, 0, 0),"
        "(0, 0, -1.63051e-05, 0, 0, 0, 0, 0, 0, 0),"
        "(0, 0, 0, -3.9098e-05, 0, 0, 0, 0, 0, 0),"
        "(0, 0, 0, 0, -1.51227e-05, 0, 0, 0, 0, 0),"
        "(0, 0, 0, 0, 0, -1.51227e-05, 0, 0, 0, 0),"
        "(0, 0, 0, 0, 0, 0, 2.59252e-05, 0, 0, 0),"
        "(0, 0, 0, 0, 0, 0, 0, 2.59252e-05, 0, 0),"
        "(0, 0, 0, 0, 0, 0, 0, 0, 7.31304e-06, 0),"
        "(0, 0, 0, 0, 0, 0, 0, 0, 0, 7.31304e-06))\0",
    "[10,10]"
        "((60.8688, 0, 0, 0, 0, 0, 0, 0, 0, 0),"
        "(0, -9.82873, 0, 0, 0, 0, 0, 0, 0, 0),"
        "(0, 0, -9.82873, 0, 0, 0, 0, 0, 0, 0),"
        "(0, 0, 0, -5.64923, 0, 0, 0, 0, 0, 0),"
        "(0, 0, 0, 0, 4.19257, 0, 0, 0, 0, 0),"
        "(0, 0, 0, 0, 0, 4.19257, 0, 0, 0, 0),"
        "(0, 0, 0, 0, 0, 0, -0.578372, 0, 0, 0),"
        "(0, 0, 0, 0, 0, 0, 0, -0.578372, 0, 0),"
        "(0, 0, 0, 0, 0, 0, 0, 0, 1.10473, 0),"
        "(0, 0, 0, 0, 0, 0, 0, 0, 0, 1.10473))\0",
    "[10,10]"
        "((445477042761.377,        0,            0,            0,            0,            0,           0,           0,          0,          0),"
        "(            0, -16305074095.74424,      0,            0,            0,            0,           0,           0,          0,          0),"
        "(            0,            0, -16305074095.74424,      0,            0,            0,           0,           0,          0,          0),"
        "(            0,            0,            0, -39097951314.96435,      0,            0,           0,           0,          0,          0),"
        "(            0,            0,            0,            0, -15122671703.18412,      0,           0,           0,          0,          0),"
        "(            0,            0,            0,            0,            0, -15122671703.18412,     0,           0,          0,          0),"
        "(            0,            0,            0,            0,            0,            0, 25925160414.7802,      0,          0,          0),"
        "(            0,            0,            0,            0,            0,            0,           0, 25925160414.7802,     0,          0),"
        "(            0,            0,            0,            0,            0,            0,           0,           0, 7313039660.941706,   0),"
        "(            0,            0,            0,            0,            0,            0,           0,           0,          0, 7313039660.941706))\0",
    "[3,3]"
        "((-0.5, 0,   0),"
        "(0,    -0.5, 0),"
        "(0,     0,   1.0))\0"
};

static const string matrix_EVALI[] = {
    "[3,3]"
        "((0.0,0.0,0.0),"
        "( 0.0,0.0,0.0),"
        "( 0.0,0.0,0.0))\0",
    "[3,3]"
        "((0.0,0.0,     0.0),"
        "( 0.0,1.649236,0.0),"
        "( 0.0,0.0,    -1.649236))\0",
    "[6,6]"
        "((6.0, 0.0, 0.0, 0.0,0.0,0.0),"
        "( 0.0,-6.0, 0.0, 0.0,0.0,0.0),"
        "( 0.0, 0.0, 2.0, 0.0,0.0,0.0),"
        "( 0.0, 0.0, 0.0,-2.0,0.0,0.0),"
        "( 0.0, 0.0, 0.0, 0.0,0.0,0.0),"
        "( 0.0, 0.0, 0.0, 0.0,0.0,0.0))\0",
    "[4,4]"
        "((0.00000,0.00000, 0.00000,0.00000),"
        "( 0.00000,0.31770, 0.00000,0.00000),"
        "( 0.00000,0.00000,-0.31770,0.00000),"
        "( 0.00000,0.00000, 0.00000,0.00000))\0",
    "[10,10]"
        "((0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000),"
        "( 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000),"
        "( 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000),"
        "( 0.000000, 0.000000, 0.000000, 4.66122,  0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000),"
        "( 0.000000, 0.000000, 0.000000, 0.000000, -4.66122, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000),"
        "( 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000),"
        "( 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.884104, 0.000000, 0.000000, 0.000000),"
        "( 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,-0.884104, 0.000000, 0.000000),"
        "( 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000),"
        "( 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000))\0",
    "[10,10]"
        "((0, 0,       0,       0, 0,       0,       0,       0,       0,        0),"
        "( 0, 5.34552, 0,       0, 0,       0,       0,       0,       0,        0),"
        "( 0, 0,      -5.34552, 0, 0,       0,       0,       0,       0,        0),"
        "( 0, 0,       0,       0, 0,       0,       0,       0,       0,        0),"
        "( 0, 0,       0,       0, 2.58278, 0,       0,       0,       0,        0),"
        "( 0, 0,       0,       0, 0,      -2.58278, 0,       0,       0,        0),"
        "( 0, 0,       0,       0, 0,       0,       1.58579, 0,       0,        0),"
        "( 0, 0,       0,       0, 0,       0,       0,      -1.58579, 0,        0),"
        "( 0, 0,       0,       0, 0,       0,       0,       0,       0.577581, 0),"
        "( 0, 0,       0,       0, 0,       0,       0,       0,       0,       -0.577581))\0",
    "[10,10]"
        "((0, 0, 0, 0, 0, 0, 0, 0, 0, 0),"
        "(0, 5.34552e-05, 0, 0, 0, 0, 0, 0, 0, 0),"
        "(0, 0, -5.34552e-05, 0, 0, 0, 0, 0, 0, 0),"
        "(0, 0, 0, 0, 0, 0, 0, 0, 0, 0),"
        "(0, 0, 0, 0, 2.58278e-05, 0, 0, 0, 0, 0),"
        "(0, 0, 0, 0, 0, -2.58278e-05, 0, 0, 0, 0),"
        "(0, 0, 0, 0, 0, 0, 1.58579e-05, 0, 0, 0),"
        "(0, 0, 0, 0, 0, 0, 0, -1.58579e-05, 0, 0),"
        "(0, 0, 0, 0, 0, 0, 0, 0, 5.77581e-06, 0),"
        "(0, 0, 0, 0, 0, 0, 0, 0, 0, -5.77581e-06))\0",
    "[10,10]"
        "((0,  0,      0,      0, 0,       0,       0,      0,      0,         0),"
        "( 0, 11.8224, 0,      0, 0,       0,       0,      0,      0,         0),"
        "( 0,  0,    -11.8224, 0, 0,       0,       0,      0,      0,         0),"
        "( 0,  0,      0,      0, 0,       0,       0,      0,      0,         0),"
        "( 0,  0,      0,      0, 3.47865, 0,       0,      0,      0,         0),"
        "( 0,  0,      0,      0, 0,      -3.47865, 0,      0,      0,         0),"
        "( 0,  0,      0,      0, 0,       0,       0.5707, 0,      0,         0),"
        "( 0,  0,      0,      0, 0,       0,       0,     -0.5707, 0,         0),"
        "( 0,  0,      0,      0, 0,       0,       0,      0,      0.0717088, 0),"
        "( 0,  0,      0,      0, 0,       0,       0,      0,      0,        -0.0717088))\0",
    "[10,10]"
        "((0,           0,            0,      0,           0,            0,           0,             0,          0,           0),"
        "( 0, 53455229317.4359,       0,      0,           0,            0,           0,             0,          0,           0),"
        "( 0,           0, -53455229317.4359, 0,           0,            0,           0,             0,          0,           0),"
        "( 0,           0,            0,      0,           0,            0,           0,             0,          0,           0),"
        "( 0,           0,            0,      0, 25827788390.45117,      0,           0,             0,          0,           0),"
        "( 0,           0,            0,      0,           0, -25827788390.45117,     0,             0,          0,           0),"
        "( 0,           0,            0,      0,           0,            0, 15857859115.66493,       0,          0,           0),"
        "( 0,           0,            0,      0,           0,            0,           0,  -15857859115.66493,    0,           0),"
        "( 0,           0,            0,      0,           0,            0,           0,             0, 5775807674.719979,    0),"
        "( 0,           0,            0,      0,           0,            0,           0,             0,          0, -5775807674.719979))\0",
    "[3,3]"
        "((0.8660254037844389, 0,                  0),"
        "( 0,                 -0.8660254037844389, 0),"
        "( 0,                  0,                  0))\0"
};

int main() {

    typedef double TYPE;
    typedef float TYPE2;

    typedef matrix<TYPE> MATRIX;

    MATRIX A;
    MATRIX GT_EVALR;
    MATRIX GT_EVALI;

    int numTestCases = 10;
    typedef typename matrix<TYPE>::size_type size_type;

    for (int k = 0; k < numTestCases; k++){

        std::cout << "Running test case# " << (k + 1) << "\n";

        {
            std::istringstream is(matrix_IN[k]);
            is >> A;
        }
        const size_type n = A.size1();
        MATRIX Zero_Matrix = zero_matrix<TYPE>(n,n);

        {
            std::istringstream is(matrix_EVALR[k]);
            is >> GT_EVALR;
        }

        {
            std::istringstream is(matrix_EVALI[k]);
            is >> GT_EVALI;
        }


        eigen_solver<MATRIX> es(A, EIGVEC);
        matrix<double> evals_r = es.get_eigenvalues_real();
        matrix<double> evals_i = es.get_eigenvalues_imag();
        matrix<double> evecs_r = es.get_eigenvectors_real();
        matrix<double> evecs_i = es.get_eigenvectors_imag();


        assertTrue("Real portion of eigenvalues match:", compare_on_tolerance(evals_r, GT_EVALR));
        assertTrue("Imag portion of eigenvalues match:", compare_on_tolerance(evals_i, GT_EVALI));

        matrix<std::complex<TYPE> > V(n, n);
        matrix<std::complex<TYPE> > D(n, n);
        matrix<std::complex<TYPE> > M(n, n);
        matrix<std::complex<TYPE> > Lambda;

        MATRIX Lambda_Real(n, n);
        MATRIX Lambda_Imag(n, n);

        for (int i = 0; i < n; i++){
            for (int j = 0; j < n; j++){
                V(i, j) = std::complex<TYPE>(evecs_r(i, j), evecs_i(i, j));
                D(i, j) = std::complex<TYPE>(evals_r(i, j), evals_i(i, j));
                M(i, j) = std::complex<TYPE>(A(i, j), 0.0);
            }
        }
        Lambda = prod(M, V) - prod(V, D);
        for (int i = 0; i < n; i++){
            for (int j = 0; j < n; j++){
                Lambda_Real(i, j) = Lambda(i, j).real();
                Lambda_Imag(i, j) = Lambda(i, j).imag();
            }
        }
        assertTrue("Real portion of verifier is zero:", compare_on_tolerance(Lambda_Real, Zero_Matrix));
        assertTrue("Imag portion of verifier is zero:", compare_on_tolerance(Lambda_Imag, Zero_Matrix));

    }

    if (!_fail_counter) {
        std::cout << "\nEigen Solver regression suite passed.\n";
        return boost::exit_success;
    }
    else {
        std::cout << "\nEigen Solver regression suite failed.\n";
        return boost::exit_failure;
    }
}
