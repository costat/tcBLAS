/*****************************************************************************/
/* Author: Timothy Costa
******************************************************************************/

#if defined(SINGLE)
#define tcb_ftype float
#define SET_ZERO_PACKED vxorps
#define VMOVU_PACKED(reg, mat_ptr, mat_offset, load_store) do { \
    if (load_store) vmovups( yword[mat_ptr + mat_offset], reg ); \
    else vmovups( reg, yword[mat_ptr + mat_offset] ); \
} while(0)
#define VFMADD231_PACKED vfmadd231ps
#define VSUB_PACKED vsubps
#define VDIV_PACKED vdivps
#define VMUL_PACKED vmulps
#define VBROADCAST vbroadcastss
#define P_UNROLL_AVX2 P_UNROLL_AVX2_F32
#define SET_ZERO_SINGLE xorps
#elif defined(DOUBLE)
#define tcb_ftype double 
#define SET_ZERO_PACKED vxorpd
#define VMOVU_PACKED(reg, mat_ptr, mat_offset, load_store) do { \
    if (load_store) vmovupd( yword[mat_ptr + mat_offset], reg ); \
    else vmovupd( reg, yword[mat_ptr + mat_offset] ); \
} while(0)
#define VFMADD231_PACKED vfmadd231pd
#define VSUB_PACKED vsubpd
#define VDIV_PACKED vdivpd
#define VMUL_PACKED vmulpd
#define VBROADCAST vbroadcastsd
#define P_UNROLL_AVX2 P_UNROLL_AVX2_F64
#endif

F_NAME::F_NAME( int layout, int m, int n, tcb_ftype alpha, int lda, int ldb )
        : Xbyak::CodeGenerator( 80 * Xbyak::DEFAULT_MAX_CODE_SIZE, nullptr )
{

    assert(alpha == 1.0);

    auto a_ptr = rdi;
    auto b_ptr = rsi;
    auto one_ptr = rdx;

    int n_in, m_in;
    bool _is_row_;
    int i, j, ii;

    // deal with LL/RU
#if defined(_TCB_LLN_)
    if (layout == 101)
    {
        m_in = m;
        n_in = n;
        _is_row_ = 1;
    }
    else
    {
        m_in = m;
        n_in = n;
        _is_row_ = 0;
    }
#elif defined(_TCB_RUN_)
    if (layout == 101)
    {
        m_in = n;
        n_in = m;
        _is_row_ = 0;
    }
    else
    {
        m_in = n;
        n_in = m;
        _is_row_ = 1;
    }
#endif

#ifdef _TCB_NOUNIT_DIAG_
    // set all elements of ymm15 to 1
    VBROADCAST(ymm15, ptr [one_ptr]); 
#endif

    // zero accumulation registers
    SET_ZERO_PACKED(ymm0, ymm0, ymm0);                          // T11
    if (m_in > 1) SET_ZERO_PACKED(ymm1, ymm1, ymm1);            // T21
    if (n_in > 1) {
        SET_ZERO_PACKED(ymm2, ymm2, ymm2);                      // T12
        if (m_in > 1) SET_ZERO_PACKED(ymm3, ymm3, ymm3);        // T22
    }

    for (j=0; j<(n_in/N_UNROLL_AVX2)*N_UNROLL_AVX2; j+=N_UNROLL_AVX2) {
        for (i=0; i<(m_in/M_UNROLL_AVX2)*M_UNROLL_AVX2; i+=M_UNROLL_AVX2) {
            // gemm update
            for (ii=0; ii<i; ii++) {
                VMOVU_PACKED(ymm5, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(trsm_ll_ap(0+i,0+ii,lda,_is_row_))), 0);    // A1
                VMOVU_PACKED(ymm6, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(trsm_ll_ap(1+i,0+ii,lda,_is_row_))), 0);    // A2
                    
                VMOVU_PACKED(ymm4, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(trsm_ll_bp(0+ii,0+j,ldb,_is_row_))), 0);
                VFMADD231_PACKED(ymm0, ymm4, ymm5);
                VFMADD231_PACKED(ymm1, ymm4, ymm6);

                VMOVU_PACKED(ymm4, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(trsm_ll_bp(0+ii,1+j,ldb,_is_row_))), 0);
                VFMADD231_PACKED(ymm2, ymm4, ymm5);
                VFMADD231_PACKED(ymm3, ymm4, ymm6);
            }
            // update the 4x4 B matrix
            // 1st
            VMOVU_PACKED(ymm8, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(trsm_ll_bp(0+i,0+j,ldb,_is_row_))), 0);        // B1
            VMOVU_PACKED(ymm9, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(trsm_ll_bp(0+i,1+j,ldb,_is_row_))), 0);        // B2

#ifdef _TCB_NOUNIT_DIAG_
            VMOVU_PACKED(ymm4, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(trsm_ll_ap(0+i,0+i,lda,_is_row_))), 0);        // A1
            VDIV_PACKED(ymm4, ymm15, ymm4);                                                                                 // A1 /= ONE
#endif

            VSUB_PACKED(ymm0, ymm8, ymm0);                                                                                  // T11 = B1-T11

#ifdef _TCB_NOUNIT_DIAG_
            VMUL_PACKED(ymm0, ymm0, ymm4);                                                                                  // T11 *= ONE/A1
#endif

            VMOVU_PACKED(ymm0, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(trsm_ll_bp(0+i,0+j,ldb,_is_row_))), 1);        // Store T11 -> B1

            VSUB_PACKED(ymm2, ymm9, ymm2);                                                                                  // T12 = B2-T12

#ifdef _TCB_NOUNIT_DIAG_
            VMUL_PACKED(ymm2, ymm2, ymm4);                                                                                  // T12 *= ONE/A1
#endif

            VMOVU_PACKED(ymm2, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(trsm_ll_bp(0+i,1+j,ldb,_is_row_))), 1);        // Store T12 -> B2

            // 2nd
#ifdef _TCB_NOUNIT_DIAG_
            VMOVU_PACKED(ymm4, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(trsm_ll_ap(1+i,1+i,lda,_is_row_))), 0);        // A1
            VDIV_PACKED(ymm4, ymm15, ymm4);                                                                                 // A1 /= ONE
#endif

            VMOVU_PACKED(ymm5, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(trsm_ll_ap(1+i,0+i,lda,_is_row_))), 0);        // A2

            VMOVU_PACKED(ymm8, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(trsm_ll_bp(1+i,0+j,ldb,_is_row_))), 0);        // B1
            VMOVU_PACKED(ymm9, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(trsm_ll_bp(1+i,1+j,ldb,_is_row_))), 0);        // B2

            VFMADD231_PACKED(ymm1, ymm5, ymm0);                                                                             // T21 += A2*T11
            VSUB_PACKED(ymm1, ymm8, ymm1);                                                                                  // T21 = B1 - T21

#ifdef _TCB_NOUNIT_DIAG_
            VMUL_PACKED(ymm1, ymm1, ymm4);                                                                                  // T21 *= ONE/A1
#endif
            VMOVU_PACKED(ymm1, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(trsm_ll_bp(1+i,0+j,ldb,_is_row_))), 1);        // Store T21 -> B1

            SET_ZERO_PACKED(ymm0, ymm0, ymm0);                                                                              // ZERO T11
            SET_ZERO_PACKED(ymm1, ymm1, ymm1);                                                                              // ZERO T21

            VFMADD231_PACKED(ymm3, ymm5, ymm2);                                                                             // T22 += A2*T12
            VSUB_PACKED(ymm3, ymm9, ymm3);                                                                                  // T22 = B2 - T22
#ifdef _TCB_NOUNIT_DIAG_
            VMUL_PACKED(ymm3, ymm3, ymm4);                                                                                  // T22 *= ONE/A1
#endif
            VMOVU_PACKED(ymm3, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(trsm_ll_bp(1+i,1+j,ldb,_is_row_))), 1);        // Store T22 -> B2

            SET_ZERO_PACKED(ymm2, ymm2, ymm2);                                                                              // ZERO T12
            SET_ZERO_PACKED(ymm3, ymm3, ymm3);                                                                              // ZERO T22
        }
        if (m_in & 1) {
           // gemm update
            for (ii=0; ii<i; ii++) {
                VMOVU_PACKED(ymm5, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(trsm_ll_ap(0+i,0+ii,lda,_is_row_))), 0);   // A1
                    
                VMOVU_PACKED(ymm4, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(trsm_ll_bp(0+ii,0+j,ldb,_is_row_))), 0);
                VFMADD231_PACKED(ymm0, ymm4, ymm5);

                VMOVU_PACKED(ymm4, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(trsm_ll_bp(0+ii,1+j,ldb,_is_row_))), 0);
                VFMADD231_PACKED(ymm2, ymm4, ymm5);
            }
            // update the 4x4 B matrix
            // 1st
            VMOVU_PACKED(ymm8, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(trsm_ll_bp(0+i,0+j,ldb,_is_row_))), 0);        // B1
            VMOVU_PACKED(ymm9, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(trsm_ll_bp(0+i,1+j,ldb,_is_row_))), 0);        // B2
#ifdef _TCB_NOUNIT_DIAG_
            VMOVU_PACKED(ymm4, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(trsm_ll_ap(0+i,0+i,lda,_is_row_))), 0);        // A1
            VDIV_PACKED(ymm4, ymm15, ymm4);                                                                                 // A1 /= ONE
#endif

            VSUB_PACKED(ymm0, ymm8, ymm0);                                                                                  // T11 = B1-T11

#ifdef _TCB_NOUNIT_DIAG_
            VMUL_PACKED(ymm0, ymm0, ymm4);                                                                                  // T11 *= ONE/A1
#endif

            VMOVU_PACKED(ymm0, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(trsm_ll_bp(0+i,0+j,ldb,_is_row_))), 1);        // Store T11 -> B1

            SET_ZERO_PACKED(ymm0, ymm0, ymm0);                                                                              // ZERO T11

            VSUB_PACKED(ymm2, ymm9, ymm2);                                                                                  // T12 = B2-T12

#ifdef _TCB_NOUNIT_DIAG_
            VMUL_PACKED(ymm2, ymm2, ymm4);                                                                                  // T12 *= ONE/A1
#endif

            VMOVU_PACKED(ymm2, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(trsm_ll_bp(0+i,1+j,ldb,_is_row_))), 1);        // Store T12 -> B2

            SET_ZERO_PACKED(ymm2, ymm2, ymm2);                                                                              // ZERO T12
        }
    }
    if (n_in & 1) {
        for (i=0; i<(m_in/M_UNROLL_AVX2)*M_UNROLL_AVX2; i+=M_UNROLL_AVX2) {
            // gemm update
            for (ii=0; ii<i; ii++) {
                VMOVU_PACKED(ymm5, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(trsm_ll_ap(0+i,0+ii,lda,_is_row_))), 0);    // A1
                VMOVU_PACKED(ymm6, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(trsm_ll_ap(1+i,0+ii,lda,_is_row_))), 0);    // A2
                    
                VMOVU_PACKED(ymm4, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(trsm_ll_bp(0+ii,0+j,ldb,_is_row_))), 0);
                VFMADD231_PACKED(ymm0, ymm4, ymm5);
                VFMADD231_PACKED(ymm1, ymm4, ymm6);

            }
            // update the 4x4 B matrix
            // 1st
            VMOVU_PACKED(ymm8, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(trsm_ll_bp(0+i,0+j,ldb,_is_row_))), 0);        // B1

#ifdef _TCB_NOUNIT_DIAG_
            VMOVU_PACKED(ymm4, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(trsm_ll_ap(0+i,0+i,lda,_is_row_))), 0);        // A1
            VDIV_PACKED(ymm4, ymm15, ymm4);                                                                                 // A1 /= ONE
#endif

            VSUB_PACKED(ymm0, ymm8, ymm0);                                                                                  // T11 = B1-T11
#ifdef _TCB_NOUNIT_DIAG_
            VMUL_PACKED(ymm0, ymm0, ymm4);                                                                                  // T11 *= ONE/A1
#endif
            VMOVU_PACKED(ymm0, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(trsm_ll_bp(0+i,0+j,ldb,_is_row_))), 1);        // Store T11 -> B1

            // 2nd
#ifdef _TCB_NOUNIT_DIAG_
            VMOVU_PACKED(ymm4, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(trsm_ll_ap(1+i,1+i,lda,_is_row_))), 0);        // A1
            VDIV_PACKED(ymm4, ymm15, ymm4);                                                                                 // A1 /= ONE
#endif

            VMOVU_PACKED(ymm5, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(trsm_ll_ap(1+i,0+i,lda,_is_row_))), 0);        // A2

            VMOVU_PACKED(ymm8, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(trsm_ll_bp(1+i,0+j,ldb,_is_row_))), 0);        // B1

            VFMADD231_PACKED(ymm1, ymm5, ymm0);                                                                             // T21 += A2*T11
            VSUB_PACKED(ymm1, ymm8, ymm1);                                                                                  // T21 = B1 - T21
#ifdef _TCB_NOUNIT_DIAG_
            VMUL_PACKED(ymm1, ymm1, ymm4);                                                                                  // T21 *= ONE/A1
#endif
            VMOVU_PACKED(ymm1, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(trsm_ll_bp(1+i,0+j,ldb,_is_row_))), 1);        // Store T21 -> B1

            SET_ZERO_PACKED(ymm0, ymm0, ymm0);                                                                              // ZERO T11
            SET_ZERO_PACKED(ymm1, ymm1, ymm1);                                                                              // ZERO T21

        }
        if (m_in & 1) {
           // gemm update
            for (ii=0; ii<i; ii++) {
                VMOVU_PACKED(ymm5, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(trsm_ll_ap(0+i,0+ii,lda,_is_row_))), 0);   // A1
                    
                VMOVU_PACKED(ymm4, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(trsm_ll_bp(0+ii,0+j,ldb,_is_row_))), 0);
                VFMADD231_PACKED(ymm0, ymm4, ymm5);

            }
            // update the 4x4 B matrix
            // 1st
            VMOVU_PACKED(ymm8, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(trsm_ll_bp(0+i,0+j,ldb,_is_row_))), 0);        // B1

#ifdef _TCB_NOUNIT_DIAG_
            VMOVU_PACKED(ymm4, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(trsm_ll_ap(0+i,0+i,lda,_is_row_))), 0);        // A1
            VDIV_PACKED(ymm4, ymm15, ymm4);                                                                                 // A1 /= ONE
#endif

            VSUB_PACKED(ymm0, ymm8, ymm0);                                                                                  // T11 = B1-T11
#ifdef _TCB_NOUNIT_DIAG_
            VMUL_PACKED(ymm0, ymm0, ymm4);                                                                                  // T11 *= ONE/A1
#endif

            VMOVU_PACKED(ymm0, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(trsm_ll_bp(0+i,0+j,ldb,_is_row_))), 1);        // Store T11 -> B1

        }

    }

    ret();
}

#undef tcb_ftype
#undef SET_ZERO_PACKED 
#undef VMOVU_PACKED 
#undef VFMADD231_PACKED 
#undef VSUB_PACKED 
#undef VDIV_PACKED
#undef VMUL_PACKED
#undef VBROADCAST
#undef P_UNROLL_AVX2
