/*****************************************************************************/
/* Author: Timothy Costa
******************************************************************************/

#ifndef _TCB_BLAS_COMMON_H_
#define _TCB_BLAS_COMMON_H_

#define load_or_zero_ps(reg, addr) do { \
    if (beta) { \
        vmovups(reg, addr); \
    } \
    else { \
        if (_avx512f_) { \
            vpxorq(reg, reg, reg); \
        } else { \
            vxorps(reg, reg, reg); \
        } \
    } \
} while (0)

#define load_or_zero_pd(reg, addr) do { \
    if (beta) { \
        vmovupd(reg, addr); \
    } \
    else { \
        if (_avx512f_) { \
            vpxorq(reg, reg, reg); \
        } else { \
            vxorpd(reg, reg, reg); \
        } \
    } \
} while (0)

#define vfma_or_vfnma_ps(arg1, arg2, arg3) do { \
    if (alpha == -1) { \
        vfnmadd231ps(arg1, arg2, arg3); \
    } \
    else { \
        vfmadd231ps(arg1, arg2, arg3); \
    } \
} while (0)

#define vfma_or_vfnma_pd(arg1, arg2, arg3) do { \
    if (alpha == -1) { \
        vfnmadd231pd(arg1, arg2, arg3); \
    } \
    else { \
        vfmadd231pd(arg1, arg2, arg3); \
    } \
} while (0)

#define trsm_ll_ap(i,j,lda,_is_row_) ((_is_row_) ? ((j)+((i)*(lda))) : ((i)+((j)*(lda))))
#define trsm_ll_bp(i,j,ldb,_is_row_) ((_is_row_) ? ((j)+((i)*(ldb))) : ((i)+((j)*(ldb))))
#define getrf_ap(i,j,lda,_is_row_) ((_is_row_) ? ((j)+((i)*(lda))) : ((i)+((j)*(lda))))

#define   MIN(a,b) ((a) < (b) ? (a) : (b))

#define ELE_IN_REGISTER_AVX2_F32 8
#define ELE_IN_REGISTER_AVX2_F64 4
#define ELE_IN_REGISTER_AVX512_F32 16
#define ELE_IN_REGISTER_AVX512_F64 8

#endif
