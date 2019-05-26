//
// Rajaditya Mukherjee
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/eigen_solver.hpp>
#include <complex>
#include <sstream>

int main () {
    using namespace boost::numeric::ublas;
    using std::string;
    matrix<double> m(3, 3);
    static const string m_string =
        "[3,3](( 2.0,  3.0,  5.0),"
              "( 2.0, -3.0,  7.0),"
              "( 4.0,  1.0,  1.0))";
    std::istringstream is(m_string);
    is >> m;
    eigen_solver<matrix<double> > es(m,EIGVEC);

    vector<double> evals_r = es.get_eigenvalues_real();
    vector<double> evals_i = es.get_eigenvalues_imag();
    matrix<double> evecs_r = es.get_eigenvectors_real();
    matrix<double> evecs_i = es.get_eigenvectors_imag();

    std::cout << "Eigenvalues (Real Part)" << std::endl;
    std::cout << evals_r << std::endl;

    std::cout << "Eigenvalues (Imag Part)" << std::endl;
    std::cout << evals_i << std::endl;

    std::cout << "Eigenvectors (Real Part)" << std::endl;
    std::cout << evecs_r << std::endl;

    std::cout << "Eigenvectors (Imag Part)" << std::endl;
    std::cout << evecs_i << std::endl;

    std::cout << "Verification\n";
    matrix<std::complex<double> > V(3,3);
    matrix<std::complex<double> > D(3,3);
    matrix<std::complex<double> > A(3,3);
    matrix<std::complex<double> > Lambda;
    for (int i = 0; i < 3; i++){
        D(i, i) = std::complex<double>(evals_r(i), evals_i(i));
        for (int j = 0; j < 3; j++){
            V(i, j) = std::complex<double>(evecs_r(i, j), evecs_i(i, j));
            if (i != j)
                D(i, j) = std::complex<double>(0.0, 0.0);
            A(i, j) = std::complex<double>(m(i, j), 0.0);
        }
    }
    Lambda = prod(A, V) - prod(V, D);
    std::cout << Lambda << std::endl;
}

