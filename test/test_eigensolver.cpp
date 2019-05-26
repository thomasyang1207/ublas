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
#include <iomanip>
#include <sstream>

using namespace boost::numeric::ublas;
using std::string;

static const string matrix_IN[] = {
    // 1
    "[3,3]"
        "((-149,   -50,   -154),"
        "(  537,   180,    546),"
        "(  -27,    -9,    -25))",
    // 2
    "[3,3]"
        "((2,    3,    5),"
        "( 2,   -3,    7),"
        "( 4,    1,    1))",
    // 3
    "[6,6]"
        "((7,    3,    4,  -11,   -9,   -2),"
        "(-6,    4,   -5,    7,    1,   12),"
        "(-1,   -9,    2,    2,    9,    1),"
        "(-8,    0,   -1,    5,    0,    8),"
        "(-4,    3,   -5,    7,    2,   10),"
        "( 6,    1,    4,  -11,   -7,   -1))",
    // 4
    "[4,4]"
        "((0.421761282626275, 0.655740699156587,  0.678735154857774, 0.655477890177557),"
        "( 0.915735525189067, 0.0357116785741896, 0.757740130578333, 0.171186687811562),"
        "( 0.792207329559554, 0.849129305868777,  0.743132468124916, 0.706046088019609),"
        "( 0.959492426392903, 0.933993247757551,  0.392227019534168, 0.0318328463774207))",
    // 5
    "[10,10]"
        "((1,    2,    3,    4,    3,    6,    7,    8,    9,   10),"
        "( 1,    2,    3,    3,    5,    6,    7,    8,    9,   10),"
        "( 3,    2,    3,    4.2,  5,    6.1,  7,    8,    9,   14),"
        "( 1,    2.1,  3,    4,    5,    6,    7.1,  8,    9,   10),"
        "( 1,    2,    3,    4.5,  5,    6,    7,    8,    9,   30),"
        "( 5,    6,    3,    4,    5,    6,    7,    8,    9,   10),"
        "( 1,    2,    3,   41,    5,    6,    7,    1,    9,   10),"
        "( 1,    2,   32,    4,    5,    6,    7,    8,    9.3, 10),"
        "( 9,    2,    3,    4,    0,    6,    7,    8,    9,   10),"
        "(12,    2,    3,    4,    5,    6,    7,    8,    9,   10))",
    // 6
    "[10,10]"
        "((0,    2,    3,    4,    3,    6,    7,    8,    9,   10),"
        "( 1,    2,    3,    3,    5,    0,    7,    8,    9,   10),"
        "( 3,    2,    0,    4.2,  5,    6.1,  0,    8,    9,    0),"
        "( 1,    2,    3,    4,    5,    6,    0,    8,    9,    1),"
        "( 1,    2,    3,    4.5,  5,    6,    7,    8,    9,    0),"
        "( 5,    6,    0,    4,    5,    6,    7,    8,    9,    1),"
        "( 0,    2,    3,    0,    5,    6,    7,    1,    9,    4),"
        "( 1,    2,    0,    4,    5,    6,    7,    8,    9.3,  0),"
        "( 0,    2,    3,    4,    0,    6,    7,    8,    9,   10),"
        "( 0,    2,    3,    4,    5,    6,    7,    8,    9,    0))",
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
        "( 0,     2e-05, 3e-05, 4e-05,   5e-05, 6e-05,   7e-05, 8e-05, 9e-05,   0))",
    // 8
    "[10,10]"
        "(( 4,   2,    3,    4,    3,    6,    7,    8,    9,   10),"
        "(  0,   2,    3,    3,    5,    0,    7,    8,    9,   10),"
        "(  3,   2,    0,    4.2,  5,    6.1,  0,    8,    9,    0),"
        "(  1,   2,    3,    4,    5,    6,    0,    8,    9,    1),"
        "(  3,   2,    3,    4.5,  5,    6,    0,    8,    9,    0),"
        "(  5,   2,    0,    4,    5,    6,    1,    8,    9,    1),"
        "(  0,   2,    3,    0,    5,    6,    7,    0,    9,    4),"
        "(  1,  13,    0,    4,    5,    6,    2,    8,    9.3,  0),"
        "(100,   2,    3,    0,    0,    6,    7,    8,    9,   10),"
        "(  0,   2,    3,    4,    5,    6,    0,    8,    9,    0))",
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
        "( 0,     2e+10, 3e+10, 4e+10,   5e+10, 6e+10,   7e+10, 8e+10, 9e+10,   0))",
    // 10
    "[3,3]"
        "((0, 0, 1),"
        "( 1, 0, 0),"
        "( 0, 1, 0))"
};

static const string matrix_EVALR[] = {
    // 1
    "[3](  1,    2,    3)",
    // 2
    "[3](  7.54718294965635,     -3.77359147482818,     -3.77359147482818)",
    // 3
    "[6](  5,    5,    1,    1,    4,    3)",
    // 4
    "[4](  2.44784336494200,     -5.60395277364049e-01, -5.60395277364049e-01, -9.46145345111045e-02)",
    // 5
    "[10]( 6.61927496926567e+01, -1.09351312839869e+01, -7.02399019336973,      2.03077450277325,      2.03077450277325,      3.56281824787758,     -5.93575284438782e-01, -5.93575284438782e-01, -2.37288291591179e-01,  5.66443391744670e-01)",
    // 6
    "[10]( 4.45477042761377e+01, -1.63050740957442,     -1.63050740957442,     -3.90979513149643,     -1.51226717031842,     -1.51226717031842,      2.59251604147802,      2.59251604147802,      7.31303966094171e-01,  7.31303966094171e-01)",
    // 7
    "[10]( 4.45477042761377e-04, -1.63050740957442e-05, -1.63050740957442e-05, -3.90979513149643e-05, -1.51226717031842e-05, -1.51226717031842e-05,  2.59251604147802e-05,  2.59251604147802e-05,  7.31303966094171e-06,  7.31303966094171e-06)",
    // 8
    "[10]( 6.08688292341761e+01, -9.82873218562485,     -9.82873218562485,     -5.64922864277610,      4.19257224326868,      4.19257224326868,     -5.78371813716572e-01, -5.78371813716572e-01,  1.10473146037276,      1.10473146037276)",
    // 9
    "[10]( 4.45477042761377e+11, -1.63050740957442e+10, -1.63050740957442e+10, -3.90979513149643e+10, -1.51226717031842e+10, -1.51226717031842e+10,  2.59251604147802e+10,  2.59251604147802e+10,  7.31303966094171e+09,  7.31303966094171e+09)",
    // 10
    "[3]( -0.5, -0.5,  1)"
};

static const string matrix_EVALI[] = {
    // 1
    "[3](  0,    0,    0)",
    // 2
    "[3](  0,                     1.64923553705580,     -1.64923553705580)",
    // 3
    "[6](  6,   -6,    2,   -2.0,  0,    0)",
    // 4
    "[4](  0,                     3.17697715588369e-01, -3.17697715588369e-01,  0)",
    // 5
    "[10]( 0,                     0,                     0,                     4.66122081564722,     -4.66122081564722,      0,                     8.84103625275635e-01, -8.84103625275635e-01,  0,                     0)",
    // 6
    "[10]( 0,                     5.34552293174360,     -5.34552293174360,      0,                     2.58277883904511,     -2.58277883904511,      1.58578591156650,     -1.58578591156650,      5.77580767471999e-01, -5.77580767471999e-01)",
    // 7
    "[10]( 0,                     5.34552293174360e-05, -5.34552293174360e-05,  0,                     2.58277883904511e-05, -2.58277883904511e-05,  1.58578591156650e-05, -1.58578591156650e-05,  5.77580767471999e-06, -5.77580767471999e-06)",
    // 8
    "[10]( 0,                     1.18223851603958e+01, -1.18223851603958e+01,  0,                     3.47865363909491,     -3.47865363909491,      5.70699638243428e-01, -5.70699638243428e-01,  7.17087609912465e-02, -7.17087609912465e-02)",
    // 9
    "[10]( 0,                     5.34552293174360e+10, -5.34552293174360e+10,  0,                     2.58277883904511e+10, -2.58277883904511e+10,  1.58578591156650e+10, -1.58578591156650e+10,  5.77580767471999e+09, -5.77580767471999e+09)",
    // 10
    "[3](  0.8660254037844389,   -0.8660254037844389,    0)"
};

int main() {

    typedef double TYPE;

    typedef matrix<TYPE> MATRIX;
    typedef vector<TYPE> VECTOR;

    MATRIX A;
    VECTOR GT_EVALR;
    VECTOR GT_EVALI;

    int numTestCases = 10;
    TYPE tolerance = std::numeric_limits<TYPE>::epsilon() * 256.0;

    typedef typename MATRIX::size_type size_type;
    typedef typename MATRIX::value_type value_type;

    std::cout << std::setw(10) << std::setprecision(5);

    for (int k = 0; k < numTestCases; k++){

        std::cout << "Running test case# " << (k + 1) << std::endl;

        {
            std::istringstream is(matrix_IN[k]);
            is >> A;
        }
        {
            std::istringstream is(matrix_EVALR[k]);
            is >> GT_EVALR;
        }

        {
            std::istringstream is(matrix_EVALI[k]);
            is >> GT_EVALI;
        }

        eigen_solver<MATRIX> es(A, EIGVEC);
        VECTOR evals_r = es.get_eigenvalues_real();
        VECTOR evals_i = es.get_eigenvalues_imag();

        bool assertion;
        double scale = double(norm_1(A));
        assertion = norm_is_small(evals_r - GT_EVALR, scale, tolerance);
        assertTrue("Real portion of eigenvalues match:", assertion);
        if (!assertion)
            std::cout << norm_inf(evals_r - GT_EVALR) / scale << std::endl;

        assertion = norm_is_small(evals_i - GT_EVALI, scale, tolerance);
        assertTrue("Imag portion of eigenvalues match:", assertion);
        if (!assertion)
            std::cout << norm_inf(evals_i - GT_EVALI) / scale << std::endl;

        const size_type n = A.size1();

        const MATRIX evecs_r = es.get_eigenvectors_real();
        const MATRIX evecs_i = es.get_eigenvectors_imag();

        matrix<std::complex<TYPE> > V(n, n);
        matrix<std::complex<TYPE> > D(n, n);
        matrix<std::complex<TYPE> > M(n, n);
        matrix<std::complex<TYPE> > Verifier;

        for (int i = 0; i < n; i++){
            D(i, i) = std::complex<TYPE>(evals_r(i), evals_i(i));
            for (int j = 0; j < n; j++){
                if (i != j)
                    D(i, j) = std::complex<TYPE>(0.0, 0.0);
                V(i, j) = std::complex<TYPE>(evecs_r(i, j), evecs_i(i, j));
                M(i, j) = std::complex<TYPE>(A(i, j), 0.0);
            }
        }
        Verifier = prod(M, V) - prod(V, D);
        scale = double(norm_inf(M));
        assertion = norm_is_small(Verifier, scale, tolerance);
        assertTrue("Verifier is zero:", assertion);
        if (!assertion)
            std::cout << norm_inf(Verifier) / scale << std::endl;

        MATRIX Verifier_Real(n, n);
        MATRIX Verifier_Imag(n, n);
        for (int i = 0; i < n; i++){
            for (int j = 0; j < n; j++){
                Verifier_Real(i, j) = Verifier(i, j).real();
                Verifier_Imag(i, j) = Verifier(i, j).imag();
            }
        }
        assertion = norm_is_small(Verifier_Real, scale, tolerance);
        assertTrue("Real portion of verifier is zero:", assertion);
        if (!assertion)
            std::cout << norm_inf(Verifier_Real) / scale << std::endl;
        assertion = norm_is_small(Verifier_Imag, scale, tolerance);
        assertTrue("Imag portion of verifier is zero:", assertion);
        if (!assertion)
            std::cout << norm_inf(Verifier_Imag) / scale << std::endl;
    }

    if (!_fail_counter) {
        std::cout << std::endl << "Eigen Solver regression suite passed." << std::endl;
        return boost::exit_success;
    }
    else {
        std::cout << std::endl << "Eigen Solver regression suite failed." << std::endl;
        return boost::exit_failure;
    }
}
