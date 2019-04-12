/*****************************************************************************/
/* Author: Timothy Costa
******************************************************************************/

#include "tcblas_cuda.hpp"

template <typename T>
void
tcb_compact_gemm_nn_cuda( 
    int layout, 
    int m_in, 
    int n_in, 
    int k, 
    T   alpha, 
    T * a_in, 
    int lda_in, 
    T * b_in, 
    int ldb_in, 
    T   beta, 
    T * c, 
    int ldc 
)
{

    int m, n, k, lda, ldb;
    T * a, * b;

    if (layout == 101) {
        a = b_in;
        b = a_in;
        m = n_in;
        n = m_in;
        lda = ldb_in;
        ldb = lda_in;
    } else {
        a = a_in;
        b = b_in;
        m = m_in;
        n = n_in;
        lda = lda_in;
        ldb = ldb_in;
    }

    assert(beta == 0.0 || beta == 1.0);
    assert(alpha == 1.0 || alpha == -1.0);

    

}

//--- Explicit Instantiations ---//
template void tcb_compact_gemm_nn_cuda<float>(int, int, int, int, 
          float, float *, int, float *, int, float, float *, int);
template void tcb_compact_gemm_nn_cuda<double>(int, int, int, int, 
          double, double *, int, double *, int, double, double *, int);
