/*****************************************************************************/
/* Author: Timothy Costa
******************************************************************************/

#include "tcblas_ia.hpp"
#include "tcblas_common.hpp"

#define I_UNROLL_AVX512 1
#define JJ_UNROLL_AVX512 4
#define P_UNROLL_AVX512_F32 ELE_IN_REGISTER_AVX512_F32
#define P_UNROLL_AVX512_F64 ELE_IN_REGISTER_AVX512_F64

using namespace Xbyak;

#undef F_NAME

#undef DOUBLE
#define SINGLE
#define F_NAME tcb_compact_getrf_avx512_f32
#include "tcb_compact_getrf_core.cxx"
#undef F_NAME

#undef SINGLE
#define DOUBLE
#define F_NAME tcb_compact_getrf_avx512_f64
#include "tcb_compact_getrf_core.cxx"
#undef F_NAME
