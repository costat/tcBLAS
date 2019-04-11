/*****************************************************************************/
/* Author: Timothy Costa
******************************************************************************/

#include "tcblas_ia.hpp"
#include "tcblas_common.hpp"

#define N_UNROLL_AVX512 4
#define M_UNROLL_AVX512 4
#define P_UNROLL_AVX512_F32 ELE_IN_REGISTER_AVX512_F32
#define P_UNROLL_AVX512_F64 ELE_IN_REGISTER_AVX512_F64

using namespace Xbyak;

#undef F_NAME

#define _TCB_LLN_

#define _TCB_NOUNIT_DIAG_
#undef DOUBLE
#define SINGLE
#define F_NAME tcb_compact_trsm_llnn_avx512_f32
#include "tcb_compact_trsm_lln_run_core.cxx"
#undef F_NAME

#undef SINGLE
#define DOUBLE
#define F_NAME tcb_compact_trsm_llnn_avx512_f64
#include "tcb_compact_trsm_lln_run_core.cxx"
#undef F_NAME

#undef _TCB_NOUNIT_DIAG_
#undef DOUBLE
#define SINGLE
#define F_NAME tcb_compact_trsm_llnu_avx512_f32
#include "tcb_compact_trsm_lln_run_core.cxx"
#undef F_NAME

#undef SINGLE
#define DOUBLE
#define F_NAME tcb_compact_trsm_llnu_avx512_f64
#include "tcb_compact_trsm_lln_run_core.cxx"
#undef F_NAME

#undef _TCB_LLN_
