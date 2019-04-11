/*****************************************************************************/
/* Author: Timothy Costa
******************************************************************************/

#include "tcblas_ia.hpp"
#include "tcblas_common.hpp"

#define N_UNROLL_AVX2 2
#define M_UNROLL_AVX2 2
#define P_UNROLL_AVX2_F32 ELE_IN_REGISTER_AVX2_F32
#define P_UNROLL_AVX2_F64 ELE_IN_REGISTER_AVX2_F64

using namespace Xbyak;

#undef F_NAME

#define _TCB_RUN_

#define _TCB_NOUNIT_DIAG_
#undef DOUBLE
#define SINGLE
#define F_NAME tcb_compact_trsm_runn_avx2_f32
#include "tcb_compact_trsm_lln_run_core.cxx"
#undef F_NAME

#undef SINGLE
#define DOUBLE
#define F_NAME tcb_compact_trsm_runn_avx2_f64
#include "tcb_compact_trsm_lln_run_core.cxx"
#undef F_NAME

#undef _TCB_NOUNIT_DIAG_
#undef DOUBLE
#define SINGLE
#define F_NAME tcb_compact_trsm_runu_avx2_f32
#include "tcb_compact_trsm_lln_run_core.cxx"
#undef F_NAME

#undef SINGLE
#define DOUBLE
#define F_NAME tcb_compact_trsm_runu_avx2_f64
#include "tcb_compact_trsm_lln_run_core.cxx"
#undef F_NAME

#undef _TCB_RUN_
