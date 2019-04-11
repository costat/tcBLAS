/*****************************************************************************/
/* Author: Timothy Costa
******************************************************************************/

#include "tcblas_ia.hpp"
#include "tcblas_common.hpp"

#define N_UNROLL_AVX2 4
#define M_UNROLL_AVX2 2
#define K_UNROLL_AVX2 1
#define P_UNROLL_AVX2_F32 ELE_IN_REGISTER_AVX2_F32
#define P_UNROLL_AVX2_F64 ELE_IN_REGISTER_AVX2_F64

using namespace Xbyak;

#undef F_NAME

#undef DOUBLE
#define SINGLE
#define F_NAME tcb_compact_gemm_nn_avx2_f32
#include "tcb_compact_gemm_nn_core.cxx"
#undef F_NAME

#undef SINGLE
#define DOUBLE
#define F_NAME tcb_compact_gemm_nn_avx2_f64
#include "tcb_compact_gemm_nn_core.cxx"
#undef F_NAME
