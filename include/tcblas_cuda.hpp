/*****************************************************************************/
/* Author: Timothy Costa
******************************************************************************/

#ifndef _TCBLAS_CUDA_H_
#define _TCBLAS_CUDA_H_

//--- GEMM KERNELS ---//
// TRANSA=N, TRANSB=N
template <typename T>
void
tcb_compact_gemm_nn_cuda( int layout, int m, int n, int k, 
    T alpha, T * a, int lda, T * b, int ldb, T beta, T * c, int ldc );

#endif

