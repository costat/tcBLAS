/*****************************************************************************/
/* Author: Timothy Costa
******************************************************************************/

#if defined(SINGLE)
#define tcb_ftype float
#define VMOVU_PACKED(reg, mat_ptr, mat_offset, load_store) do { \
    if (load_store) vmovups( zword[mat_ptr + mat_offset], reg ); \
    else vmovups( reg, zword[mat_ptr + mat_offset] ); \
} while(0)
#define VSUB_PACKED vsubps
#define VDIV_PACKED vdivps
#define VMUL_PACKED vmulps
#define VBROADCAST vbroadcastss
#define P_UNROLL_AVX512 P_UNROLL_AVX512_F32
#elif defined(DOUBLE)
#define tcb_ftype double 
#define VMOVU_PACKED(reg, mat_ptr, mat_offset, load_store) do { \
    if (load_store) vmovupd( zword[mat_ptr + mat_offset], reg ); \
    else vmovupd( reg, zword[mat_ptr + mat_offset] ); \
} while(0)
#define VSUB_PACKED vsubpd
#define VDIV_PACKED vdivpd
#define VMUL_PACKED vmulpd
#define VBROADCAST vbroadcastsd
#define P_UNROLL_AVX512 P_UNROLL_AVX512_F64
#endif

F_NAME::F_NAME( int layout, int m, int n, int lda )
        : Xbyak::CodeGenerator( 80 * Xbyak::DEFAULT_MAX_CODE_SIZE, nullptr )
{

    auto a_ptr = rdi;
    auto one_ptr = rsi;

    bool _is_row_ = (layout == 101);

    int i, j, j_idx, ii, jj, start, n_rem;

    VBROADCAST(zmm15, ptr [one_ptr]); 

    for (j = 0; j < MIN(m, n); j++) {
        VMOVU_PACKED(zmm0, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(getrf_ap(0+j,0+j,lda,_is_row_))), 0);
        VDIV_PACKED(zmm0, zmm15, zmm0);                                                                                
        for (i = j + 1; i < m; i++) {
            VMOVU_PACKED(zmm1, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(getrf_ap(0+i,0+j,lda,_is_row_))), 0);      
            VMUL_PACKED(zmm1, zmm1, zmm0);
            VMOVU_PACKED(zmm1, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(getrf_ap(0+i,0+j,lda,_is_row_))), 1);
        }
        start = j + 1;
        n_rem = n - start;
        for (j_idx = 0; j_idx < (n_rem/JJ_UNROLL_AVX512)*JJ_UNROLL_AVX512; j_idx+=JJ_UNROLL_AVX512) {
            jj = j_idx + start;
            VMOVU_PACKED(zmm2, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(getrf_ap(0+j,0+jj,lda,_is_row_))), 0);      
            VMOVU_PACKED(zmm3, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(getrf_ap(0+j,1+jj,lda,_is_row_))), 0);      
            VMOVU_PACKED(zmm4, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(getrf_ap(0+j,2+jj,lda,_is_row_))), 0);      
            VMOVU_PACKED(zmm5, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(getrf_ap(0+j,3+jj,lda,_is_row_))), 0);
            for (ii = j+1; ii<m; ii++) {
                VMOVU_PACKED(zmm6, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(getrf_ap(0+ii,0+j,lda,_is_row_))), 0);
                VMOVU_PACKED(zmm7, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(getrf_ap(0+ii,0+jj,lda,_is_row_))), 0);      
                VMUL_PACKED(zmm8, zmm6, zmm2);
                VSUB_PACKED(zmm7, zmm7, zmm8);

                VMOVU_PACKED(zmm9, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(getrf_ap(0+ii,1+jj,lda,_is_row_))), 0);
                VMUL_PACKED(zmm10, zmm6, zmm3);
                VSUB_PACKED(zmm9, zmm9, zmm10);

                VMOVU_PACKED(zmm11, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(getrf_ap(0+ii,2+jj,lda,_is_row_))), 0);
                VMUL_PACKED(zmm12, zmm6, zmm4);
                VSUB_PACKED(zmm11, zmm11, zmm12);
                
                VMOVU_PACKED(zmm13, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(getrf_ap(0+ii,3+jj,lda,_is_row_))), 0);
                VMUL_PACKED(zmm14, zmm6, zmm5);
                VSUB_PACKED(zmm13, zmm13, zmm14);

                VMOVU_PACKED(zmm7, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(getrf_ap(0+ii,0+jj,lda,_is_row_))), 1);      
                VMOVU_PACKED(zmm9, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(getrf_ap(0+ii,1+jj,lda,_is_row_))), 1);      
                VMOVU_PACKED(zmm11, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(getrf_ap(0+ii,2+jj,lda,_is_row_))), 1);      
                VMOVU_PACKED(zmm13, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(getrf_ap(0+ii,3+jj,lda,_is_row_))), 1);      
            }
        }
        if (n_rem & 2 && n_rem & 1) {
            jj = j_idx + start;
            VMOVU_PACKED(zmm2, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(getrf_ap(0+j,0+jj,lda,_is_row_))), 0);      
            VMOVU_PACKED(zmm3, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(getrf_ap(0+j,1+jj,lda,_is_row_))), 0);      
            VMOVU_PACKED(zmm4, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(getrf_ap(0+j,2+jj,lda,_is_row_))), 0);
            for (ii = j+1; ii<m; ii++) {
                VMOVU_PACKED(zmm6, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(getrf_ap(0+ii,0+j,lda,_is_row_))), 0);      
                VMOVU_PACKED(zmm7, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(getrf_ap(0+ii,0+jj,lda,_is_row_))), 0);      
                VMUL_PACKED(zmm8, zmm6, zmm2);
                VSUB_PACKED(zmm7, zmm7, zmm8);

                VMOVU_PACKED(zmm9, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(getrf_ap(0+ii,1+jj,lda,_is_row_))), 0);
                VMUL_PACKED(zmm10, zmm6, zmm3);
                VSUB_PACKED(zmm9, zmm9, zmm10);

                VMOVU_PACKED(zmm11, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(getrf_ap(0+ii,2+jj,lda,_is_row_))), 0);
                VMUL_PACKED(zmm12, zmm6, zmm4);
                VSUB_PACKED(zmm11, zmm11, zmm12);
                
                VMOVU_PACKED(zmm7, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(getrf_ap(0+ii,0+jj,lda,_is_row_))), 1);      
                VMOVU_PACKED(zmm9, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(getrf_ap(0+ii,1+jj,lda,_is_row_))), 1);      
                VMOVU_PACKED(zmm11, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(getrf_ap(0+ii,2+jj,lda,_is_row_))), 1);
            }
        }
        else if (n_rem & 2) {
            jj = j_idx + start;
            VMOVU_PACKED(zmm2, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(getrf_ap(0+j,0+jj,lda,_is_row_))), 0);      
            VMOVU_PACKED(zmm3, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(getrf_ap(0+j,1+jj,lda,_is_row_))), 0);      
            for (ii = j+1; ii<m; ii++) {
                VMOVU_PACKED(zmm6, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(getrf_ap(0+ii,0+j,lda,_is_row_))), 0);      
                VMOVU_PACKED(zmm7, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(getrf_ap(0+ii,0+jj,lda,_is_row_))), 0);
                VMUL_PACKED(zmm8, zmm6, zmm2);
                VSUB_PACKED(zmm7, zmm7, zmm8);

                VMOVU_PACKED(zmm9, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(getrf_ap(0+ii,1+jj,lda,_is_row_))), 0);      
                VMUL_PACKED(zmm10, zmm6, zmm3);
                VSUB_PACKED(zmm9, zmm9, zmm10);

                VMOVU_PACKED(zmm7, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(getrf_ap(0+ii,0+jj,lda,_is_row_))), 1);      
                VMOVU_PACKED(zmm9, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(getrf_ap(0+ii,1+jj,lda,_is_row_))), 1);      
            }
        }
        else if (n_rem & 1) {
            jj = j_idx + start;
            VMOVU_PACKED(zmm2, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(getrf_ap(0+j,0+jj,lda,_is_row_))), 0);      
            for (ii = j+1; ii<m; ii++) {
                VMOVU_PACKED(zmm6, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(getrf_ap(0+ii,0+j,lda,_is_row_))), 0);
                VMOVU_PACKED(zmm7, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(getrf_ap(0+ii,0+jj,lda,_is_row_))), 0);      
                VMUL_PACKED(zmm8, zmm6, zmm2);
                VSUB_PACKED(zmm7, zmm7, zmm8);

                VMOVU_PACKED(zmm7, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(getrf_ap(0+ii,0+jj,lda,_is_row_))), 1);      
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
#undef P_UNROLL_AVX512
