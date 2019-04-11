/*****************************************************************************/
/* Author: Timothy Costa
******************************************************************************/

#ifndef _TCBLAS_IA_H_
#define _TCBLAS_IA_H_

#include "xbyak/xbyak.h"
#include "tcblas_util.hpp"

////////////////////////////
//--- LEVEL 3 ROUTINES ---//
////////////////////////////

////////////////
//--- AVX2 ---//
////////////////

//--- GEMM KERNELS ---//
// TRANSA=N, TRANSB=N, F32
class tcb_compact_gemm_nn_avx2_f32: public Xbyak::CodeGenerator
{
    public:
    tcb_compact_gemm_nn_avx2_f32( int layout, int m, int n, int k, float alpha, int lda, int ldb, float beta, int ldc );
};
// TRANSA=N, TRANSB=N, F64
class tcb_compact_gemm_nn_avx2_f64: public Xbyak::CodeGenerator
{
    public:
    tcb_compact_gemm_nn_avx2_f64( int layout, int m, int n, int k, double alpha, int lda, int ldb, double beta, int ldc );
};

//--- TRSM KERNELS ---//
// SIDE=L, UPLO=L, TRANSA=N, DIAG=N, F32
class tcb_compact_trsm_llnn_avx2_f32: public Xbyak::CodeGenerator
{
    public:
    tcb_compact_trsm_llnn_avx2_f32( int layout, int m, int n, float alpha, int lda, int ldb );
};
// SIDE=L, UPLO=L, TRANSA=N, DIAG=N, F64
class tcb_compact_trsm_llnn_avx2_f64: public Xbyak::CodeGenerator
{
    public:
    tcb_compact_trsm_llnn_avx2_f64( int layout, int m, int n, double alpha, int lda, int ldb );
};
// SIDE=L, UPLO=L, TRANSA=N, DIAG=U, F32
class tcb_compact_trsm_llnu_avx2_f32: public Xbyak::CodeGenerator
{
    public:
    tcb_compact_trsm_llnu_avx2_f32( int layout, int m, int n, float alpha, int lda, int ldb );
};
// SIDE=L, UPLO=L, TRANSA=N, DIAG=U, F64
class tcb_compact_trsm_llnu_avx2_f64: public Xbyak::CodeGenerator
{
    public:
    tcb_compact_trsm_llnu_avx2_f64( int layout, int m, int n, double alpha, int lda, int ldb );
};
// SIDE=R, UPLO=U, TRANSA=N, DIAG=N, F32
class tcb_compact_trsm_runn_avx2_f32: public Xbyak::CodeGenerator
{
    public:
    tcb_compact_trsm_runn_avx2_f32( int layout, int m, int n, float alpha, int lda, int ldb );
};
// SIDE=R, UPLO=U, TRANSA=N, DIAG=N, F64
class tcb_compact_trsm_runn_avx2_f64: public Xbyak::CodeGenerator
{
    public:
    tcb_compact_trsm_runn_avx2_f64( int layout, int m, int n, double alpha, int lda, int ldb );
};
// SIDE=R, UPLO=U, TRANSA=N, DIAG=U, F32
class tcb_compact_trsm_runu_avx2_f32: public Xbyak::CodeGenerator
{
    public:
    tcb_compact_trsm_runu_avx2_f32( int layout, int m, int n, float alpha, int lda, int ldb );
};
// SIDE=R, UPLO=U, TRANSA=N, DIAG=U, F64
class tcb_compact_trsm_runu_avx2_f64: public Xbyak::CodeGenerator
{
    public:
    tcb_compact_trsm_runu_avx2_f64( int layout, int m, int n, double alpha, int lda, int ldb );
};

//--- LU KERNELS ---//
class tcb_compact_getrf_avx2_f32: public Xbyak::CodeGenerator
{
    public:
    tcb_compact_getrf_avx2_f32( int layout, int m, int n, int lda);
};
class tcb_compact_getrf_avx2_f64: public Xbyak::CodeGenerator
{
    public:
    tcb_compact_getrf_avx2_f64( int layout, int m, int n, int lda);
};

//////////////////
//--- AVX512 ---//
//////////////////

//--- GEMM KERNELS ---//
// TRANSA=N, TRANSB=N, F32
class tcb_compact_gemm_nn_avx512_f32: public Xbyak::CodeGenerator
{
    public:
    tcb_compact_gemm_nn_avx512_f32( int layout, int m, int n, int k, float alpha, int lda, int ldb, float beta, int ldc );
};
// TRANSA=N, TRANSB=N, F64
class tcb_compact_gemm_nn_avx512_f64: public Xbyak::CodeGenerator
{
    public:
    tcb_compact_gemm_nn_avx512_f64( int layout, int m, int n, int k, double alpha, int lda, int ldb, double beta, int ldc );
};

//--- TRSM KERNELS ---//
// SIDE=L, UPLO=L, TRANSA=N, DIAG=N, F32
class tcb_compact_trsm_llnn_avx512_f32: public Xbyak::CodeGenerator
{
    public:
    tcb_compact_trsm_llnn_avx512_f32( int layout, int m, int n, float alpha, int lda, int ldb );
};
// SIDE=L, UPLO=L, TRANSA=N, DIAG=N, F64
class tcb_compact_trsm_llnn_avx512_f64: public Xbyak::CodeGenerator
{
    public:
    tcb_compact_trsm_llnn_avx512_f64( int layout, int m, int n, double alpha, int lda, int ldb );
};
// SIDE=L, UPLO=L, TRANSA=N, DIAG=U, F32
class tcb_compact_trsm_llnu_avx512_f32: public Xbyak::CodeGenerator
{
    public:
    tcb_compact_trsm_llnu_avx512_f32( int layout, int m, int n, float alpha, int lda, int ldb );
};
// SIDE=L, UPLO=L, TRANSA=N, DIAG=U, F64
class tcb_compact_trsm_llnu_avx512_f64: public Xbyak::CodeGenerator
{
    public:
    tcb_compact_trsm_llnu_avx512_f64( int layout, int m, int n, double alpha, int lda, int ldb );
};
// SIDE=R, UPLO=U, TRANSA=N, DIAG=N, F32
class tcb_compact_trsm_runn_avx512_f32: public Xbyak::CodeGenerator
{
    public:
    tcb_compact_trsm_runn_avx512_f32( int layout, int m, int n, float alpha, int lda, int ldb );
};
// SIDE=R, UPLO=U, TRANSA=N, DIAG=N, F64
class tcb_compact_trsm_runn_avx512_f64: public Xbyak::CodeGenerator
{
    public:
    tcb_compact_trsm_runn_avx512_f64( int layout, int m, int n, double alpha, int lda, int ldb );
};
// SIDE=R, UPLO=U, TRANSA=N, DIAG=U, F32
class tcb_compact_trsm_runu_avx512_f32: public Xbyak::CodeGenerator
{
    public:
    tcb_compact_trsm_runu_avx512_f32( int layout, int m, int n, float alpha, int lda, int ldb );
};
// SIDE=R, UPLO=U, TRANSA=N, DIAG=U, F64
class tcb_compact_trsm_runu_avx512_f64: public Xbyak::CodeGenerator
{
    public:
    tcb_compact_trsm_runu_avx512_f64( int layout, int m, int n, double alpha, int lda, int ldb );
};

//--- LU KERNELS ---//
class tcb_compact_getrf_avx512_f32: public Xbyak::CodeGenerator
{
    public:
    tcb_compact_getrf_avx512_f32( int layout, int m, int n, int lda);
};
class tcb_compact_getrf_avx512_f64: public Xbyak::CodeGenerator
{
    public:
    tcb_compact_getrf_avx512_f64( int layout, int m, int n, int lda);
};

#endif
