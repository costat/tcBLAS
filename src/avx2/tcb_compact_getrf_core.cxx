/*****************************************************************************/
/* Author: Timothy Costa
******************************************************************************/

#if defined(SINGLE)
#define tcb_ftype float
#define VMOVU_PACKED(reg, mat_ptr, mat_offset, load_store) do { \
    if (load_store) vmovups( yword[mat_ptr + mat_offset], reg ); \
    else vmovups( reg, yword[mat_ptr + mat_offset] ); \
} while(0)
#define VSUB_PACKED vsubps
#define VDIV_PACKED vdivps
#define VMUL_PACKED vmulps
#define VBROADCAST vbroadcastss
#define P_UNROLL_AVX2 P_UNROLL_AVX2_F32
#elif defined(DOUBLE)
#define tcb_ftype double 
#define VMOVU_PACKED(reg, mat_ptr, mat_offset, load_store) do { \
    if (load_store) vmovupd( yword[mat_ptr + mat_offset], reg ); \
    else vmovupd( reg, yword[mat_ptr + mat_offset] ); \
} while(0)
#define VSUB_PACKED vsubpd
#define VDIV_PACKED vdivpd
#define VMUL_PACKED vmulpd
#define VBROADCAST vbroadcastsd
#define P_UNROLL_AVX2 P_UNROLL_AVX2_F64
#endif

F_NAME::F_NAME( int layout, int m, int n, int lda )
        : Xbyak::CodeGenerator( 80 * Xbyak::DEFAULT_MAX_CODE_SIZE, nullptr )
{

    auto a_ptr = rdi;
    auto one_ptr = rsi;

    bool _is_row_ = (layout == 101);

    int i, j, j_idx, ii, jj, start, n_rem;

    VBROADCAST(ymm15, ptr [one_ptr]); 

    for (j = 0; j < MIN(m, n); j++) {
        VMOVU_PACKED(ymm0, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(getrf_ap(0+j,0+j,lda,_is_row_))), 0);
        VDIV_PACKED(ymm0, ymm15, ymm0);                                                                                
        for (i = j + 1; i < m; i++) {
            VMOVU_PACKED(ymm1, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(getrf_ap(0+i,0+j,lda,_is_row_))), 0);      
            VMUL_PACKED(ymm1, ymm1, ymm0);
            VMOVU_PACKED(ymm1, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(getrf_ap(0+i,0+j,lda,_is_row_))), 1);
        }
        start = j + 1;
        n_rem = n - start;
        for (j_idx = 0; j_idx < (n_rem/JJ_UNROLL_AVX2)*JJ_UNROLL_AVX2; j_idx+=JJ_UNROLL_AVX2) {
            jj = j_idx + start;
            VMOVU_PACKED(ymm2, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(getrf_ap(0+j,0+jj,lda,_is_row_))), 0);      
            VMOVU_PACKED(ymm3, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(getrf_ap(0+j,1+jj,lda,_is_row_))), 0);      
            VMOVU_PACKED(ymm4, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(getrf_ap(0+j,2+jj,lda,_is_row_))), 0);      
            VMOVU_PACKED(ymm5, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(getrf_ap(0+j,3+jj,lda,_is_row_))), 0);
            for (ii = j+1; ii<m; ii++) {
                VMOVU_PACKED(ymm6, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(getrf_ap(0+ii,0+j,lda,_is_row_))), 0);
                VMOVU_PACKED(ymm7, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(getrf_ap(0+ii,0+jj,lda,_is_row_))), 0);      
                VMUL_PACKED(ymm8, ymm6, ymm2);
                VSUB_PACKED(ymm7, ymm7, ymm8);

                VMOVU_PACKED(ymm9, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(getrf_ap(0+ii,1+jj,lda,_is_row_))), 0);
                VMUL_PACKED(ymm10, ymm6, ymm3);
                VSUB_PACKED(ymm9, ymm9, ymm10);

                VMOVU_PACKED(ymm11, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(getrf_ap(0+ii,2+jj,lda,_is_row_))), 0);
                VMUL_PACKED(ymm12, ymm6, ymm4);
                VSUB_PACKED(ymm11, ymm11, ymm12);
                
                VMOVU_PACKED(ymm13, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(getrf_ap(0+ii,3+jj,lda,_is_row_))), 0);
                VMUL_PACKED(ymm14, ymm6, ymm5);
                VSUB_PACKED(ymm13, ymm13, ymm14);

                VMOVU_PACKED(ymm7, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(getrf_ap(0+ii,0+jj,lda,_is_row_))), 1);      
                VMOVU_PACKED(ymm9, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(getrf_ap(0+ii,1+jj,lda,_is_row_))), 1);      
                VMOVU_PACKED(ymm11, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(getrf_ap(0+ii,2+jj,lda,_is_row_))), 1);      
                VMOVU_PACKED(ymm13, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(getrf_ap(0+ii,3+jj,lda,_is_row_))), 1);      
            }
        }
        if (n_rem & 2 && n_rem & 1) {
            jj = j_idx + start;
            VMOVU_PACKED(ymm2, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(getrf_ap(0+j,0+jj,lda,_is_row_))), 0);      
            VMOVU_PACKED(ymm3, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(getrf_ap(0+j,1+jj,lda,_is_row_))), 0);      
            VMOVU_PACKED(ymm4, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(getrf_ap(0+j,2+jj,lda,_is_row_))), 0);
            for (ii = j+1; ii<m; ii++) {
                VMOVU_PACKED(ymm6, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(getrf_ap(0+ii,0+j,lda,_is_row_))), 0);      
                VMOVU_PACKED(ymm7, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(getrf_ap(0+ii,0+jj,lda,_is_row_))), 0);      
                VMUL_PACKED(ymm8, ymm6, ymm2);
                VSUB_PACKED(ymm7, ymm7, ymm8);

                VMOVU_PACKED(ymm9, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(getrf_ap(0+ii,1+jj,lda,_is_row_))), 0);
                VMUL_PACKED(ymm10, ymm6, ymm3);
                VSUB_PACKED(ymm9, ymm9, ymm10);

                VMOVU_PACKED(ymm11, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(getrf_ap(0+ii,2+jj,lda,_is_row_))), 0);
                VMUL_PACKED(ymm12, ymm6, ymm4);
                VSUB_PACKED(ymm11, ymm11, ymm12);
                
                VMOVU_PACKED(ymm7, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(getrf_ap(0+ii,0+jj,lda,_is_row_))), 1);      
                VMOVU_PACKED(ymm9, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(getrf_ap(0+ii,1+jj,lda,_is_row_))), 1);      
                VMOVU_PACKED(ymm11, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(getrf_ap(0+ii,2+jj,lda,_is_row_))), 1);
            }
        }
        else if (n_rem & 2) {
            jj = j_idx + start;
            VMOVU_PACKED(ymm2, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(getrf_ap(0+j,0+jj,lda,_is_row_))), 0);      
            VMOVU_PACKED(ymm3, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(getrf_ap(0+j,1+jj,lda,_is_row_))), 0);      
            for (ii = j+1; ii<m; ii++) {
                VMOVU_PACKED(ymm6, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(getrf_ap(0+ii,0+j,lda,_is_row_))), 0);      
                VMOVU_PACKED(ymm7, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(getrf_ap(0+ii,0+jj,lda,_is_row_))), 0);
                VMUL_PACKED(ymm8, ymm6, ymm2);
                VSUB_PACKED(ymm7, ymm7, ymm8);

                VMOVU_PACKED(ymm9, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(getrf_ap(0+ii,1+jj,lda,_is_row_))), 0);      
                VMUL_PACKED(ymm10, ymm6, ymm3);
                VSUB_PACKED(ymm9, ymm9, ymm10);

                VMOVU_PACKED(ymm7, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(getrf_ap(0+ii,0+jj,lda,_is_row_))), 1);      
                VMOVU_PACKED(ymm9, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(getrf_ap(0+ii,1+jj,lda,_is_row_))), 1);      
            }
        }
        else if (n_rem & 1) {
            jj = j_idx + start;
            VMOVU_PACKED(ymm2, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(getrf_ap(0+j,0+jj,lda,_is_row_))), 0);      
            for (ii = j+1; ii<m; ii++) {
                VMOVU_PACKED(ymm6, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(getrf_ap(0+ii,0+j,lda,_is_row_))), 0);
                VMOVU_PACKED(ymm7, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(getrf_ap(0+ii,0+jj,lda,_is_row_))), 0);      
                VMUL_PACKED(ymm8, ymm6, ymm2);
                VSUB_PACKED(ymm7, ymm7, ymm8);

                VMOVU_PACKED(ymm7, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(getrf_ap(0+ii,0+jj,lda,_is_row_))), 1);      
            }
        }
    }

    ret();
}

#undef tcb_ftype 
#undef VMOVU_PACKED 
#undef VSUB_PACKED 
#undef VDIV_PACKED
#undef VMUL_PACKED 
#undef VBROADCAST 
#undef P_UNROLL_AVX2
