// Rajaditya Mukherjee
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)


/// \file schur_decomposition.hpp Contains methods for real schur decomposition

#ifndef _BOOST_UBLAS_SCHURDECOMPOSITION_
#define _BOOST_UBLAS_SCHURDECOMPOSITION_

#define EPSA 1.0e-20

#include <boost/numeric/ublas/operation.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/triangular.hpp>
#include <iostream>
#include <limits>


namespace boost {
    namespace numeric {
        namespace ublas {

            /*! @fn schur_decomposition(M &h)
            * @brief Performs Real Schur Decomposition for Matrix \c m (Version without EigenVectors).
            * Replaces the input matrix which is assumed to be in Hessenberg Form by the
            real schur form of the matrix. This uses the Francis Double Shift QR Algorithm to compute the real Schur Form.
            * The diagonals of the schur form are either 1x1 blocks (real eigenvalues) or 2x2 blocks (complex eigenvalues).
            * @tparam[in] m matrix type (like matrix<double>) - input hessenberg form output - schur form
            */
            template<class M>
            void schur_decomposition(M &h) {

                typedef typename M::value_type value_type;
                int n = h.size1();
                int p = n;
                int iter = 0;
                const int max_iter = 100;
                while (p > 2 && iter < max_iter) {
                    int q = p - 1;

                    value_type s = h(q - 1, q - 1) + h(p - 1, p - 1);
                    value_type t = h(q - 1, q - 1) * h(p - 1, p - 1) - h(q - 1, p - 1) * h(p - 1, q - 1);
                    value_type x = h(0, 0) * h(0, 0) +
                        h(0, 1) * h(1, 0) -
                        s * h(0, 0) + t;
                    value_type y = h(1, 0) * (h(0, 0) + h(1, 1) - s);
                    value_type z = h(1, 0) * h(2, 1);

                    //Ask mentor if we can assume that int will support signed type (otherwise we need a different mechanism
                    // On second thought lets make no such assumptions
                    for (int k = 0; k <= p - 3; k++) {
                        vector<value_type> tx(3);
                        tx(0) = x; tx(1) = y; tx(2) = z;
                        vector<value_type> v;
                        value_type beta;
                        householder<vector<value_type> >(tx, v, beta);

                        int r = (std::max)(1, k);

                        matrix<value_type> vvt = outer_prod(v, v);
                        vvt *= beta;
                        int n_vvt = vvt.size1();
                        matrix<value_type> imvvt = identity_matrix<value_type>(n_vvt) -vvt;

                        matrix<value_type> t1 = prod(imvvt, project(h, range(k, k + 3), range(r - 1, n)));
                        project(h, range(k, k + 3), range(r - 1, n)).assign(t1);

                        r = (std::min)(k + 4, p);

                        matrix<value_type> t2 = prod(project(h, range(0, r), range(k, k + 3)), imvvt);
                        project(h, range(0, r), range(k, k + 3)).assign(t2);

                        x = h(k + 1, k);
                        y = h(k + 2, k);

                        if (k < (p - 3)) {
                            z = h(k + 3, k);
                        }

                    }

                    vector<value_type> tx(2);
                    tx(0) = x; tx(1) = y;

                    value_type scale = (std::abs)(tx(0)) + (std::abs)(tx(1));
                    if (scale != value_type(0)) {
                        tx(0) /= scale;
                        tx(1) /= scale;
                    }

                    value_type cg, sg;
                    givens_rotation<value_type>(tx(0), tx(1), cg, sg);

                    //Apply rotation to matrices as needed
                    matrix<value_type> t1 = project(h, range(q - 1, p), range(p - 3, n));
                    int t1_cols = t1.size2();
                    int t1_rows = t1.size1();
                    for (int j = 0; j < t1_cols; j++) {
                        value_type tau1 = t1(0, j);
                        value_type tau2 = t1(t1_rows - 1, j);
                        t1(0, j) = cg*tau1 - sg*tau2;
                        t1(t1_rows - 1, j) = sg*tau1 + cg*tau2;
                    }
                    //std::cout << t1 << "\n";
                    project(h, range(q - 1, p), range(p - 3, n)).assign(t1);

                    matrix<value_type> t2 = project(h, range(0, p), range(p - 2, p));
                    int t2_cols = t2.size2();
                    int t2_rows = t2.size1();
                    for (int j = 0; j < t2_rows; j++) {
                        value_type tau1 = t2(j, 0);
                        value_type tau2 = t2(j, t2_cols - 1);
                        t2(j, 0) = cg*tau1 - sg*tau2;
                        t2(j, t2_cols - 1) = sg*tau1 + cg*tau2;
                    }
                    project(h, range(0, p), range(p - 2, p)).assign(t2);

                    if ((std::abs)(h(p - 1, q - 1)) < (std::numeric_limits<value_type>::epsilon())*((std::abs)(h(p - 1, p - 1)) + (std::abs)(h(q - 1, q - 1)))) {
                        h(p - 1, q - 1) = value_type(0);
                        p = p - 1;
                        q = p - 1;
                        iter = 0;
                    }
                    else if ((std::abs)(h(p - 2, q - 2)) < (std::numeric_limits<value_type>::epsilon())*((std::abs)(h(q - 2, q - 2)) + (std::abs)(h(q - 1, q - 1)))) {
                        h(p - 2, q - 2) = value_type(0);
                        p = p - 2;
                        q = p - 1;
                        iter = 0;
                    }
                    else {
                        iter++;
                        std::cout << iter << std::endl;
                    }
                }
            }

            /// \brief Performs Real Schur Decomposition for Matrix \c m (Version with EigenVectors).
            /// Replaces the input matrix which is assumed to be in Hessenberg Form by the
            /// real schur form of the matrix. This uses the Francis Double Shift QR Algorithm to compute the real Schur Form.
            /// The diagonals of the schur form are either 1x1 blocks (real eigenvalues) or 2x2 blocks (complex eigenvalues).
            /// \param m matrix type (like matrix<double>) - input hessenberg form output - schur form
            /// \param qv matrix type (like matrix<double>) - input accumulated transforms
            template<class M>
            void schur_decomposition(M &h, M &qv) {

                //typedef typename M::size_type int;
                typedef typename M::value_type value_type;

                int n = h.size1();
                int p = n;
                int iter = 0;
                const int max_iter = 100;
                while (p > 2 && iter < max_iter) {
                    int q = p - 1;

                    value_type s = h(q - 1, q - 1) + h(p - 1, p - 1);
                    value_type t = h(q - 1, q - 1) * h(p - 1, p - 1) - h(q - 1, p - 1) * h(p - 1, q - 1);
                    value_type x = h(0, 0) * h(0, 0) +
                        h(0, 1) * h(1, 0) -
                        s * h(0, 0) + t;
                    value_type y = h(1, 0) * (h(0, 0) + h(1, 1) - s);
                    value_type z = h(1, 0) * h(2, 1);

                    //Ask mentor if we can assume that int will support signed type (otherwise we need a different mechanism
                    // On second thought lets make no such assumptions
                    for (int k = 0; k <= p - 3; k++) {
                        vector<value_type> tx(3);
                        tx(0) = x; tx(1) = y; tx(2) = z;

                        //Add some scaling factor here
                        value_type scale = (std::abs)(tx(0)) + (std::abs)(tx(1)) + (std::abs)(tx(2));
                        if (scale != value_type(0)) {
                            tx(0) /= scale;
                            tx(1) /= scale;
                            tx(2) /= scale;
                        }

                        vector<value_type> v;
                        value_type beta;
                        householder<vector<value_type> >(tx, v, beta);

                        int r = (std::max)(1, k);

                        matrix<value_type> vvt = outer_prod(v, v);
                        vvt *= beta;
                        int n_vvt = vvt.size1();
                        matrix<value_type> imvvt = identity_matrix<value_type>(n_vvt) -vvt;

                        matrix<value_type> t1 = prod(imvvt, project(h, range(k, k + 3), range(r - 1, n)));
                        project(h, range(k, k + 3), range(r - 1, n)).assign(t1);

                        r = (std::min)(k + 4, p);

                        matrix<value_type> t2 = prod(project(h, range(0, r), range(k, k + 3)), imvvt);
                        project(h, range(0, r), range(k, k + 3)).assign(t2);

                        matrix<value_type> t3 = prod(project(qv, range(0, n), range(k, k + 3)), imvvt);
                        project(qv, range(0, n), range(k, k + 3)).assign(t3);

                        x = h(k + 1, k);
                        y = h(k + 2, k);

                        if (k < (p - 3)) {
                            z = h(k + 3, k);
                        }

                    }

                    vector<value_type> tx(2);
                    tx(0) = x; tx(1) = y;

                    value_type scale = (std::abs)(tx(0)) + (std::abs)(tx(1)) ;
                    if (scale != value_type(0)) {
                        tx(0) /= scale;
                        tx(1) /= scale;
                    }

                    value_type cg, sg;
                    givens_rotation<value_type>(tx(0), tx(1), cg, sg);

                    //Apply rotation to matrices as needed
                    matrix<value_type> t1 = project(h, range(q - 1, p), range(p - 3, n));
                    int t1_cols = t1.size2();
                    int t1_rows = t1.size1();
                    for (int j = 0; j < t1_cols; j++) {
                        value_type tau1 = t1(0, j);
                        value_type tau2 = t1(t1_rows - 1, j);
                        t1(0, j) = cg*tau1 - sg*tau2;
                        t1(t1_rows - 1, j) = sg*tau1 + cg*tau2;
                    }
                    project(h, range(q - 1, p), range(p - 3, n)).assign(t1);

                    matrix<value_type> t2 = project(h, range(0, p), range(p - 2, p));
                    int t2_cols = t2.size2();
                    int t2_rows = t2.size1();
                    for (int j = 0; j < t2_rows; j++) {
                        value_type tau1 = t2(j, 0);
                        value_type tau2 = t2(j,t2_cols - 1);
                        t2(j, 0) = cg*tau1 - sg*tau2;
                        t2(j, t2_cols - 1) = sg*tau1 + cg*tau2;
                    }
                    project(h, range(0, p), range(p - 2, p)).assign(t2);

                    matrix<value_type> t3 = project(qv, range(0, n), range(p - 2, p));
                    int t3_cols = t3.size2();
                    int t3_rows = t3.size1();
                    for (int j = 0; j < t3_rows; j++) {
                        value_type tau1 = t3(j, 0);
                        value_type tau2 = t3(j, t3_cols - 1);
                        t3(j, 0) = cg*tau1 - sg*tau2;
                        t3(j, t3_cols - 1) = sg*tau1 + cg*tau2;
                    }
                    project(qv, range(0, n), range(p - 2, p)).assign(t3);

                    //Pollution Cleanup
                    /*for (int ci = 0; ci < p; ++ci)
                    {
                        for (int ri = p; ri < n; ri++) {
                            h(ri, ci) = value_type(0);
                        }
                    }
                    for (int ci = p; ci < n - q; ci++) {
                        for (int ri = n - q; ri < n; ri++){
                            h(ri, ci) = value_type(0);
                        }
                    }*/

                    if ((std::abs)(h(p - 1, q - 1)) < (std::numeric_limits<value_type>::epsilon())*((std::abs)(h(p - 1, p - 1)) + (std::abs)(h(q - 1, q - 1)))) {
                        h(p - 1, q - 1) = value_type(0);
                        p = p - 1;
                        q = p - 1;
                        iter = 0;
                    }
                    else if ((std::abs)(h(p - 2, q - 2)) < (std::numeric_limits<value_type>::epsilon())*((std::abs)(h(q - 2, q - 2)) + (std::abs)(h(q - 1, q - 1)))) {
                        h(p - 2, q - 2) = value_type(0);
                        p = p - 2;
                        q = p - 1;
                        iter = 0;
                    }
                    else {
                        iter++;
                    }
                }
            }
#ifdef INCLUDE_UNUSED_SCHUR_CODE
            template<class M>
            void find_small_diag_entry(M &h, typename M::size_type end, typename M::value_type l1_norm, typename M::size_type &small_index)
            {
                //typedef typename M::size_type int;
                typedef typename M::value_type value_type;

                int k = end;
                while (k > 0)
                {
                    value_type s = (std::abs)(h(k - 1, k - 1)) + (std::abs)(h(k, k));
                    if (s == value_type(0))
                        s = l1_norm;
                    if ((std::abs)(h(k, k - 1)) < (std::numeric_limits<value_type>::epsilon() * s))
                        break;
                    --k;
                }
                small_index = k;
            }

            template<class M>
            void row_split(M &h, M &qv, typename M::size_type end, typename M::value_type exceptional_shift_sum)
            {
                //typedef typename M::size_type int;
                typedef typename M::value_type value_type;

                int n = h.size1();

                value_type p = value_type(0.5) * (h(end - 1, end - 1) - h(end, end));
                value_type q = p * p + h(end, end - 1) *h(end - 1, end);
                h(end, end) += exceptional_shift_sum;
                h(end - 1, end - 1) += exceptional_shift_sum;

                //Two real eigenvalues are present so separate them
                if (q >= value_type(0))
                {
                    value_type z = (std::sqrt)((std::abs)(q));
                    vector<value_type> tx(2);
                    if (p >= value_type(0))
                    {
                        tx(0) = p + z;
                    }
                    else
                    {
                        tx(0) = p - z;
                    }
                    tx(1) = h(end, end - 1);
                    value_type cg, sg;
                    givens_rotation<value_type>(tx(0), tx(1), cg, sg);

                    //Apply rotation to matrices as needed
                    matrix<value_type> t1 = project(h, range(end - 1, end + 1), range(end - 1, n));
                    int t1_cols = t1.size2();
                    int t1_rows = t1.size1();
                    for (int j = 0; j < t1_cols; ++j) {
                        value_type tau1 = t1(0, j);
                        value_type tau2 = t1(t1_rows - 1, j);
                        t1(0, j) = cg*tau1 - sg*tau2;
                        t1(t1_rows - 1, j) = sg*tau1 + cg*tau2;
                    }
                    project(h, range(end - 1, end + 1), range(end - 1, n)).assign(t1);

                    matrix<value_type> t2 = project(h, range(0, end + 1), range(end - 1, end + 1));
                    int t2_cols = t2.size2();
                    int t2_rows = t2.size1();
                    for (int j = 0; j < t2_rows; ++j) {
                        value_type tau1 = t2(j, 0);
                        value_type tau2 = t2(j, t2_cols - 1);
                        t2(j, 0) = cg*tau1 - sg*tau2;
                        t2(j, t2_cols - 1) = sg*tau1 + cg*tau2;
                    }
                    project(h, range(0, end + 1), range(end - 1, end + 1)).assign(t2);

                    matrix<value_type> t3 = project(qv, range(0, n), range(end - 1, end + 1));
                    int t3_cols = t3.size2();
                    int t3_rows = t3.size1();
                    for (int j = 0; j < t3_rows; ++j) {
                        value_type tau1 = t3(j, 0);
                        value_type tau2 = t3(j, t3_cols - 1);
                        t3(j, 0) = cg*tau1 - sg*tau2;
                        t3(j, t3_cols - 1) = sg*tau1 + cg*tau2;
                    }
                    project(qv, range(0, n), range(end - 1, end + 1)).assign(t3);
                }

                if (end > 1)
                    h(end - 1, end - 2) = value_type(0);
            }

            template<class M>
            void infer_shifts(M &h, typename M::size_type end, typename M::size_type iter_nos, typename M::value_type exceptional_shift_sum, vector<typename M::value_type> &shift_vector)
            {
                //typedef typename M::size_type int;
                typedef typename M::value_type value_type;

                shift_vector = vector<value_type>(3);

                shift_vector(0) = h(end, end);
                shift_vector(1) = h(end - 1, end - 1);
                shift_vector(2) = h(end, end - 1) * h(end - 1, end);

                //Original Shift
                if (iter_nos == int(10))
                {
                    exceptional_shift_sum += shift_vector(0);
                    for (int i = 0; i <= end; ++i)
                    {
                        h(i, i) -= shift_vector(0);
                    }
                    value_type s = (std::abs)(h(end, end - 1)) + (std::abs)(h(end - 1, end - 2));
                    shift_vector(0) = value_type(0.75) * s;
                    shift_vector(1) = value_type(0.75) * s;
                    shift_vector(2) = value_type(-0.4375) * s * s;
                }

                // Matlabs ad hoc shift (somehow Eigen people got this and its not in public domain)
                // Sometimes I just think Numerical Computing is a big insider job
                if (iter_nos == int(30))
                {
                    value_type s = (shift_vector(1) - shift_vector(0))*value_type(0.5);
                    s = s * s + shift_vector(2);
                    if (s > value_type(0))
                    {
                        s = (std::sqrt)(s);
                        if (shift_vector(1) < shift_vector(0))
                        {
                            s = -s;
                        }
                        s = s + ((shift_vector(1) - shift_vector(0))*value_type(0.5));
                        s = (shift_vector(0) - shift_vector(2)) / s;
                        exceptional_shift_sum += s;
                        for (int i = 0; i <= end; ++i)
                            h(i, i) -= s;
                        shift_vector(0) = shift_vector(1) = shift_vector(2) = value_type(0.964);
                    }
                }

            }

            template<class M>
            void francis_qr_step(M &h, M &qv, typename M::size_type rowStart, typename M::size_type end, vector<typename M::value_type> shifts)
            {
                //typedef typename M::size_type int;
                typedef typename M::value_type value_type;

                int colStart;
                vector<value_type> householderVec(3);

                for (colStart = end - 2; colStart >= rowStart; --colStart)
                {
                    value_type tmm = h(colStart, colStart);
                    value_type r = shifts(0) - tmm;
                    value_type s = shifts(1) - tmm;
                    householderVec(0) = (r * s - shifts(2)) / h(colStart + 1, colStart) + h(colStart, colStart + 1);
                    householderVec(1) = h(colStart + 1, colStart + 1) - tmm - r - s;
                    householderVec(2) = h(colStart + 2, colStart + 1);
                    if (colStart == rowStart) {
                        break;
                    }
                    value_type lhs = h(colStart, colStart - 1) * ((std::abs)(householderVec(1)) + (std::abs)(householderVec(2)));
                    value_type rhs = householderVec(0) * ((std::abs)(h(colStart - 1, colStart - 1)) + (std::abs)(tmm)+(std::abs)(h(colStart + 1, colStart + 1)));
                    if ((std::abs)(lhs) < (std::numeric_limits<value_type>::epsilon()) * rhs)
                    {
                        break;
                    }
                }

                int n = h.size1();
                bool firstTime = true;
                for (int k = colStart; k <= end - 2; ++k)
                {
                    vector<value_type> tx(3);
                    if (firstTime)
                    {
                        tx = householderVec;
                        firstTime = false;
                    }
                    else
                    {
                        tx(0) = h(k, k - 1);
                        tx(1) = h(k + 1, k - 1);
                        tx(2) = h(k + 1, k - 1);
                    }
                    vector<value_type> v;
                    value_type beta;
                    householder<vector<value_type> >(tx, v, beta);

                    if (beta != value_type(0))
                    {
                        if (firstTime && k > rowStart)
                        {
                            h(k, k - 1) = -h(k, k - 1);
                        }
                        else if (!firstTime)
                        {
                            h(k, k - 1) = beta;
                        }

                        matrix<value_type> vvt = outer_prod(v, v);
                        vvt *= beta;
                        int n_vvt = vvt.size1();
                        matrix<value_type> imvvt = identity_matrix<value_type>(n_vvt) - vvt;

                        matrix<value_type> t1 = prod(imvvt, project(h, range(k, k + 3), range(k, n)));
                        project(h, range(k, k + 3), range(k, n)).assign(t1);

                        value_type r = (std::min)(end, k + 3) + 1;
                        matrix<value_type> t2 = prod(project(h, range(0, r), range(k, k + 3)), imvvt);
                        project(h, range(0, r), range(k, k + 3)).assign(t2);

                        matrix<value_type> t3 = prod(project(qv, range(0, n), range(k, k + 3)), imvvt);
                        project(qv, range(0, n), range(k, k + 3)).assign(t3);

                    }
                }

                vector<value_type> tx(2);
                vector<value_type> v;
                tx(0) = h(end - 1, end - 2);
                tx(1) = h(end , end - 2);
                value_type beta;
                householder<vector<value_type> >(tx, v, beta);

                if (beta != value_type(0))
                {
                    h(end - 1, end - 2) = beta;

                    matrix<value_type> vvt = outer_prod(v, v);
                    vvt *= beta;
                    int n_vvt = vvt.size1();
                    matrix<value_type> imvvt = identity_matrix<value_type>(n_vvt) -vvt;

                    matrix<value_type> t1 = prod(imvvt, project(h, range(end - 1, end + 1), range(end - 1, n)));
                    project(h, range(end - 1, end + 1), range(end - 1, n)).assign(t1);

                    matrix<value_type> t2 = prod(project(h, range(0, end + 1), range(end - 1, end + 1)), imvvt);
                    project(h, range(0, end + 1), range(end - 1, end + 1)).assign(t2);

                    matrix<value_type> t3 = prod(project(qv, range(0, n), range(end - 1, end + 1)), imvvt);
                    project(qv, range(0, n), range(end - 1, end + 1)).assign(t3);
                }

                // Round off errors (this creates a lot of issues)
                for (int i = colStart + 2; i <= end; ++i)
                {
                    h(i, i - 2) = value_type(0);
                    if (i > colStart + 2)
                        h(i, i - 3) = value_type(0);
                }

            }
#endif
}}}


#endif
