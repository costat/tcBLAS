/*****************************************************************************/
/* Author: Timothy Costa
******************************************************************************/

#if defined(SINGLE)
#define tcb_ftype float
#define SET_ZERO_PACKED vpxorq 
#define VMOVU_PACKED(reg, mat_ptr, mat_offset, load_store) do { \
    if (load_store) vmovups( zword[mat_ptr + mat_offset], reg ); \
    else vmovups( reg, zword[mat_ptr + mat_offset] ); \
} while(0)
#define VFMADD231_PACKED vfmadd231ps
#define VSUB_PACKED vsubps
#define VDIV_PACKED vdivps
#define VMUL_PACKED vmulps
#define VBROADCAST vbroadcastss
#define P_UNROLL_AVX512 P_UNROLL_AVX512_F32
#elif defined(DOUBLE)
#define tcb_ftype double 
#define SET_ZERO_PACKED vpxorq
#define VMOVU_PACKED(reg, mat_ptr, mat_offset, load_store) do { \
    if (load_store) vmovupd( zword[mat_ptr + mat_offset], reg ); \
    else vmovupd( reg, zword[mat_ptr + mat_offset] ); \
} while(0)
#define VFMADD231_PACKED vfmadd231pd
#define VSUB_PACKED vsubpd
#define VDIV_PACKED vdivpd
#define VMUL_PACKED vmulpd
#define VBROADCAST vbroadcastsd
#define P_UNROLL_AVX512 P_UNROLL_AVX512_F64
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

 #if defined(_TCB_LLN_)
    if (layout == 101) {
        m_in = m; 
        n_in = n; 
        _is_row_ = 1;
    }
    else {
        m_in = m;
        n_in = n;
        _is_row_ = 0;
    }
#elif defined(_TCB_RUN_)
    if (layout == 101) {
        m_in = n; 
        n_in = m; 
        _is_row_ = 0;
    }
    else {
        m_in = n;
        n_in = m;
        _is_row_ = 1;
    }
#endif

    int n_rem = n_in;
    int m_rem = m_in;

#ifdef _TCB_NOUNIT_DIAG_
    // set all elements of ymm15 to 1
    VBROADCAST(zmm31, ptr [one_ptr]); 
#endif

    // zero accumulation registers
    SET_ZERO_PACKED(zmm0, zmm0, zmm0);                          // T11
    if (m_in > 1) SET_ZERO_PACKED(zmm1, zmm1, zmm1);            // T21
    if (m_in > 2) SET_ZERO_PACKED(zmm2, zmm2, zmm2);            // T31
    if (m_in > 3) SET_ZERO_PACKED(zmm3, zmm3, zmm3);            // T41
    if (n_in > 1) {
        SET_ZERO_PACKED(zmm4, zmm4, zmm4);                      // T12
        if (m_in > 1) SET_ZERO_PACKED(zmm5, zmm5, zmm5);        // T22
        if (m_in > 2) SET_ZERO_PACKED(zmm6, zmm6, zmm6);        // T32
        if (m_in > 3) SET_ZERO_PACKED(zmm7, zmm7, zmm7);        // T42
    }
    if (n_in > 2) {
        SET_ZERO_PACKED(zmm8, zmm8, zmm8);                      // T13
        if (m_in > 1) SET_ZERO_PACKED(zmm9, zmm9, zmm9);        // T23
        if (m_in > 2) SET_ZERO_PACKED(zmm10, zmm10, zmm10);     // T33
        if (m_in > 3) SET_ZERO_PACKED(zmm11, zmm11, zmm11);     // T43
    }
    if (n_in > 3) {
        SET_ZERO_PACKED(zmm12, zmm12, zmm12);                   // T14
        if (m_in > 1) SET_ZERO_PACKED(zmm13, zmm13, zmm13);     // T24
        if (m_in > 2) SET_ZERO_PACKED(zmm14, zmm14, zmm14);     // T34
        if (m_in > 3) SET_ZERO_PACKED(zmm15, zmm15, zmm15);     // T44
    }

    for (j=0; j<(n_in/N_UNROLL_AVX512)*N_UNROLL_AVX512; j+=N_UNROLL_AVX512) {
        n_rem -= N_UNROLL_AVX512;
        for (i=0; i<(m_in/M_UNROLL_AVX512)*M_UNROLL_AVX512; i+=M_UNROLL_AVX512) {
            m_rem -= M_UNROLL_AVX512;
            // gemm update
            for (ii=0; ii<i; ii++) {
                VMOVU_PACKED(zmm17, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_ap(0+i,0+ii,lda,_is_row_))), 0);    // A1
                VMOVU_PACKED(zmm18, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_ap(1+i,0+ii,lda,_is_row_))), 0);    // A2
                VMOVU_PACKED(zmm19, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_ap(2+i,0+ii,lda,_is_row_))), 0);    // A3
                VMOVU_PACKED(zmm20, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_ap(3+i,0+ii,lda,_is_row_))), 0);    // A4
                    
                VMOVU_PACKED(zmm16, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(0+ii,0+j,ldb,_is_row_))), 0);
                VFMADD231_PACKED(zmm0, zmm16, zmm17);
                VFMADD231_PACKED(zmm1, zmm16, zmm18);
                VFMADD231_PACKED(zmm2, zmm16, zmm19);
                VFMADD231_PACKED(zmm3, zmm16, zmm20);

                VMOVU_PACKED(zmm16, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(0+ii,1+j,ldb,_is_row_))), 0);
                VFMADD231_PACKED(zmm4, zmm16, zmm17);
                VFMADD231_PACKED(zmm5, zmm16, zmm18);
                VFMADD231_PACKED(zmm6, zmm16, zmm19);
                VFMADD231_PACKED(zmm7, zmm16, zmm20);

                VMOVU_PACKED(zmm16, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(0+ii,2+j,ldb,_is_row_))), 0);
                VFMADD231_PACKED(zmm8, zmm16, zmm17);
                VFMADD231_PACKED(zmm9, zmm16, zmm18);
                VFMADD231_PACKED(zmm10, zmm16, zmm19);
                VFMADD231_PACKED(zmm11, zmm16, zmm20);

                VMOVU_PACKED(zmm16, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(0+ii,3+j,ldb,_is_row_))), 0);
                VFMADD231_PACKED(zmm12, zmm16, zmm17);
                VFMADD231_PACKED(zmm13, zmm16, zmm18);
                VFMADD231_PACKED(zmm14, zmm16, zmm19);
                VFMADD231_PACKED(zmm15, zmm16, zmm20);
            }
            // update the 4x4 B matrix
            // 1st
            VMOVU_PACKED(zmm20, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(0+i,0+j,ldb,_is_row_))), 0);          // B1
            VMOVU_PACKED(zmm21, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(0+i,1+j,ldb,_is_row_))), 0);          // B2
            VMOVU_PACKED(zmm22, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(0+i,2+j,ldb,_is_row_))), 0);          // B3
            VMOVU_PACKED(zmm23, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(0+i,3+j,ldb,_is_row_))), 0);          // B4

#ifdef _TCB_NOUNIT_DIAG_
            VMOVU_PACKED(zmm16, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_ap(0+i,0+i,lda,_is_row_))), 0);          // A1
            VDIV_PACKED(zmm16, zmm31, zmm16);                                                                               // A1 /= ONE
#endif

            VSUB_PACKED(zmm0, zmm20, zmm0);                                                                                 // T11 = B1-T11

#ifdef _TCB_NOUNIT_DIAG_
            VMUL_PACKED(zmm0, zmm0, zmm16);                                                                                 // T11 *= ONE/A1
#endif

            VMOVU_PACKED(zmm0, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(0+i,0+j,ldb,_is_row_))), 1);           // Store T11 -> B1

            VSUB_PACKED(zmm4, zmm21, zmm4);                                                                                 // T12 = B2-T12
#ifdef _TCB_NOUNIT_DIAG_
            VMUL_PACKED(zmm4, zmm4, zmm16);                                                                                 // T12 *= ONE/A1
#endif

            VMOVU_PACKED(zmm4, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(0+i,1+j,ldb,_is_row_))), 1);           // Store T12 -> B2

            VSUB_PACKED(zmm8, zmm22, zmm8);                                                                                 // T13 = B3-T13
#ifdef _TCB_NOUNIT_DIAG_
            VMUL_PACKED(zmm8, zmm8, zmm16);                                                                                 // T13 *= ONE/A1
#endif

            VMOVU_PACKED(zmm8, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(0+i,2+j,ldb,_is_row_))), 1);           // Store T13 -> B3

            VSUB_PACKED(zmm12, zmm23, zmm12);                                                                               // T14 = B4-T14
#ifdef _TCB_NOUNIT_DIAG_
            VMUL_PACKED(zmm12, zmm12, zmm16);                                                                               // T14 *= ONE/A1
#endif

            VMOVU_PACKED(zmm12, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(0+i,3+j,ldb,_is_row_))), 1);          // Store T14 -> B4

            // 2nd
#ifdef _TCB_NOUNIT_DIAG_
            VMOVU_PACKED(zmm16, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_ap(1+i,1+i,lda,_is_row_))), 0);          // A1
            VDIV_PACKED(zmm16, zmm31, zmm16);                                                                               // A1 /= ONE
#endif

            VMOVU_PACKED(zmm17, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_ap(1+i,0+i,lda,_is_row_))), 0);          // A2

            VMOVU_PACKED(zmm20, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(1+i,0+j,ldb,_is_row_))), 0);          // B1
            VMOVU_PACKED(zmm21, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(1+i,1+j,ldb,_is_row_))), 0);          // B2
            VMOVU_PACKED(zmm22, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(1+i,2+j,ldb,_is_row_))), 0);          // B3
            VMOVU_PACKED(zmm23, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(1+i,3+j,ldb,_is_row_))), 0);          // B4

            VFMADD231_PACKED(zmm1, zmm17, zmm0);                                                                            // T21 += A2*T11
            VSUB_PACKED(zmm1, zmm20, zmm1);                                                                                 // T21 = B1 - T21
#ifdef _TCB_NOUNIT_DIAG_
            VMUL_PACKED(zmm1, zmm1, zmm16);                                                                                 // T21 *= ONE/A1
#endif
            VMOVU_PACKED(zmm1, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(1+i,0+j,ldb,_is_row_))), 1);      // Store T21 -> B1

            VFMADD231_PACKED(zmm5, zmm17, zmm4);                                                                            // T22 += A2*T12
            VSUB_PACKED(zmm5, zmm21, zmm5);                                                                                 // T22 = B2 - T22
#ifdef _TCB_NOUNIT_DIAG_
            VMUL_PACKED(zmm5, zmm5, zmm16);                                                                                 // T22 *= ONE/A1
#endif
            VMOVU_PACKED(zmm5, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(1+i,1+j,ldb,_is_row_))), 1);      // Store T22 -> B2

            VFMADD231_PACKED(zmm9, zmm17, zmm8);                                                                            // T23 += A2*T13
            VSUB_PACKED(zmm9, zmm22, zmm9);                                                                                 // T23 = B3 - T23
#ifdef _TCB_NOUNIT_DIAG_
            VMUL_PACKED(zmm9, zmm9, zmm16);                                                                                 // T23 *= ONE/A1
#endif
            VMOVU_PACKED(zmm9, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(1+i,2+j,ldb,_is_row_))), 1);      // Store T23 -> B3

            VFMADD231_PACKED(zmm13, zmm17, zmm12);                                                                          // T24 += A2*T14
            VSUB_PACKED(zmm13, zmm23, zmm13);                                                                               // T24 = B3 - T24
#ifdef _TCB_NOUNIT_DIAG_
            VMUL_PACKED(zmm13, zmm13, zmm16);                                                                               // T24 *= ONE/A1
#endif
            VMOVU_PACKED(zmm13, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(1+i,3+j,ldb,_is_row_))), 1);     // Store T24 -> B4

            // 3rd
#ifdef _TCB_NOUNIT_DIAG_
            VMOVU_PACKED(zmm16, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_ap(2+i,2+i,lda,_is_row_))), 0);     // A1
            VDIV_PACKED(zmm16, zmm31, zmm16);                                                                               // A1 /= ONE
#endif

            VMOVU_PACKED(zmm17, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_ap(2+i,0+i,lda,_is_row_))), 0);     // A2
            VMOVU_PACKED(zmm18, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_ap(2+i,1+i,lda,_is_row_))), 0);     // A3

            VMOVU_PACKED(zmm20, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(2+i,0+j,ldb,_is_row_))), 0);     // B1
            VMOVU_PACKED(zmm21, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(2+i,1+j,ldb,_is_row_))), 0);     // B2
            VMOVU_PACKED(zmm22, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(2+i,2+j,ldb,_is_row_))), 0);     // B3
            VMOVU_PACKED(zmm23, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(2+i,3+j,ldb,_is_row_))), 0);     // B4

            VFMADD231_PACKED(zmm2, zmm17, zmm0);                                                                            // T31 += A2 * T11
            VFMADD231_PACKED(zmm2, zmm18, zmm1);                                                                            // T31 += A3 * T21
            VSUB_PACKED(zmm2, zmm20, zmm2);                                                                                 // T31 = B1 - T31
#ifdef _TCB_NOUNIT_DIAG_
            VMUL_PACKED(zmm2, zmm2, zmm16);                                                                                 // T31 *= ONE/A1
#endif
            VMOVU_PACKED(zmm2, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(2+i,0+j,ldb,_is_row_))), 1);      // Store T31 -> B1

            VFMADD231_PACKED(zmm6, zmm17, zmm4);                                                                            // T32 += A2 * T12
            VFMADD231_PACKED(zmm6, zmm18, zmm5);                                                                            // T32 += A3 * T22
            VSUB_PACKED(zmm6, zmm21, zmm6);                                                                                 // T32 = B2 - T32
#ifdef _TCB_NOUNIT_DIAG_
            VMUL_PACKED(zmm6, zmm6, zmm16);                                                                                 // T32 *= ONE/A1
#endif
            VMOVU_PACKED(zmm6, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(2+i,1+j,ldb,_is_row_))), 1);      // Store T32 -> B2

            VFMADD231_PACKED(zmm10, zmm17, zmm8);                                                                           // T33 += A2 * T13
            VFMADD231_PACKED(zmm10, zmm18, zmm9);                                                                           // T33 += A3 * T23
            VSUB_PACKED(zmm10, zmm22, zmm10);                                                                               // T33 = B3 - T33
#ifdef _TCB_NOUNIT_DIAG_
            VMUL_PACKED(zmm10, zmm10, zmm16);                                                                               // T33 *= ONE/A1
#endif
            VMOVU_PACKED(zmm10, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(2+i,2+j,ldb,_is_row_))), 1);     // Store T33 -> B3

            VFMADD231_PACKED(zmm14, zmm17, zmm12);                                                                          // T34 += A2 * T14
            VFMADD231_PACKED(zmm14, zmm18, zmm13);                                                                          // T34 += A3 * T24
            VSUB_PACKED(zmm14, zmm23, zmm14);                                                                               // T34 = B4 - T34
#ifdef _TCB_NOUNIT_DIAG_
            VMUL_PACKED(zmm14, zmm14, zmm16);                                                                               // T34 *= ONE/A1
#endif
            VMOVU_PACKED(zmm14, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(2+i,3+j,ldb,_is_row_))), 1);     // Store T34 -> B4

            // 4th
#ifdef _TCB_NOUNIT_DIAG_
            VMOVU_PACKED(zmm16, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_ap(3+i,3+i,lda,_is_row_))), 0);     // A1
            VDIV_PACKED(zmm16, zmm31, zmm16);                                                                               // A1 /= ONE
#endif

            VMOVU_PACKED(zmm17, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_ap(3+i,0+i,lda,_is_row_))), 0);     // A2
            VMOVU_PACKED(zmm18, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_ap(3+i,1+i,lda,_is_row_))), 0);     // A3
            VMOVU_PACKED(zmm19, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_ap(3+i,2+i,lda,_is_row_))), 0);     // A4
            
            VMOVU_PACKED(zmm20, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(3+i,0+j,ldb,_is_row_))), 0);     // B1
            VMOVU_PACKED(zmm21, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(3+i,1+j,ldb,_is_row_))), 0);     // B2
            VMOVU_PACKED(zmm22, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(3+i,2+j,ldb,_is_row_))), 0);     // B3
            VMOVU_PACKED(zmm23, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(3+i,3+j,ldb,_is_row_))), 0);     // B4

            VFMADD231_PACKED(zmm3, zmm17, zmm0);                                                                            // T41 += A2 * T11
            VFMADD231_PACKED(zmm3, zmm18, zmm1);                                                                            // T41 += A3 * T21
            VFMADD231_PACKED(zmm3, zmm19, zmm2);                                                                            // T41 += A4 * T31
            VSUB_PACKED(zmm3, zmm20, zmm3);                                                                                 // T41 = B1 - T41
#ifdef _TCB_NOUNIT_DIAG_
            VMUL_PACKED(zmm3, zmm3, zmm16);                                                                                 // T41 *= ONE/A1
#endif
            VMOVU_PACKED(zmm3, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(3+i,0+j,ldb,_is_row_))), 1);      // Store T41 -> B1

            SET_ZERO_PACKED(zmm0, zmm0, zmm0);                                                      // ZERO T11
            SET_ZERO_PACKED(zmm1, zmm1, zmm1);                                                      // ZERO T21
            SET_ZERO_PACKED(zmm2, zmm2, zmm2);                                                      // ZERO T31
            SET_ZERO_PACKED(zmm3, zmm3, zmm3);                                                      // ZERO T41

            VFMADD231_PACKED(zmm7, zmm17, zmm4);                                                                            // T42 += A2 * T12
            VFMADD231_PACKED(zmm7, zmm18, zmm5);                                                                            // T42 += A3 * T22
            VFMADD231_PACKED(zmm7, zmm19, zmm6);                                                                            // T42 += A4 * T32
            VSUB_PACKED(zmm7, zmm21, zmm7);                                                                                 // T42 = B2 - T42
#ifdef _TCB_NOUNIT_DIAG_
            VMUL_PACKED(zmm7, zmm7, zmm16);                                                                                 // T42 *= ONE/A1
#endif
            VMOVU_PACKED(zmm7, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(3+i,1+j,ldb,_is_row_))), 1);      // Store T42 -> B2

            SET_ZERO_PACKED(zmm4, zmm4, zmm4);                                                      // ZERO T12
            SET_ZERO_PACKED(zmm5, zmm5, zmm5);                                                      // ZERO T22
            SET_ZERO_PACKED(zmm6, zmm6, zmm6);                                                      // ZERO T32
            SET_ZERO_PACKED(zmm7, zmm7, zmm7);                                                      // ZERO T42

            VFMADD231_PACKED(zmm11, zmm17, zmm8);                                                                           // T43 += A2 * T13
            VFMADD231_PACKED(zmm11, zmm18, zmm9);                                                                           // T43 += A3 * T23
            VFMADD231_PACKED(zmm11, zmm19, zmm10);                                                                          // T43 += A4 * T33
            VSUB_PACKED(zmm11, zmm22, zmm11);                                                                               // T43 = B3 - T43
#ifdef _TCB_NOUNIT_DIAG_
            VMUL_PACKED(zmm11, zmm11, zmm16);                                                                               // T43 *= ONE/A1
#endif
            VMOVU_PACKED(zmm11, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(3+i,2+j,ldb,_is_row_))), 1);     // STORE T43 -> B3

            SET_ZERO_PACKED(zmm8, zmm8, zmm8);                                                      // ZERO T13
            SET_ZERO_PACKED(zmm9, zmm9, zmm9);                                                      // ZERO T23
            SET_ZERO_PACKED(zmm10, zmm10, zmm10);                                                   // ZERO T33
            SET_ZERO_PACKED(zmm11, zmm11, zmm11);                                                   // ZERO T43

            VFMADD231_PACKED(zmm15, zmm17, zmm12);                                                                          // T44 += A2 * T14
            VFMADD231_PACKED(zmm15, zmm18, zmm13);                                                                          // T44 += A3 * T24
            VFMADD231_PACKED(zmm15, zmm19, zmm14);                                                                          // T44 += A4 * T34
            VSUB_PACKED(zmm15, zmm23, zmm15);                                                                               // T44 = B4 - T44
#ifdef _TCB_NOUNIT_DIAG_
            VMUL_PACKED(zmm15, zmm15, zmm16);                                                                               // T44 *= ONE/A1
#endif
            VMOVU_PACKED(zmm15, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(3+i,3+j,ldb,_is_row_))), 1);     // STORE T44 -> B4

            SET_ZERO_PACKED(zmm12, zmm12, zmm12);                                                   // ZERO T14
            SET_ZERO_PACKED(zmm13, zmm13, zmm13);                                                   // ZERO T24
            SET_ZERO_PACKED(zmm14, zmm14, zmm14);                                                   // ZERO T34
            SET_ZERO_PACKED(zmm15, zmm15, zmm15);                                                   // ZERO T44
        } // m loop
        // m tails
        if (m_in & 2 && m_in & 1) {
            // gemm update
            for (ii=0; ii<i; ii++) {
                VMOVU_PACKED(zmm17, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_ap(0+i,0+ii,lda,_is_row_))), 0);    // A1
                VMOVU_PACKED(zmm18, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_ap(1+i,0+ii,lda,_is_row_))), 0);    // A2
                VMOVU_PACKED(zmm19, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_ap(2+i,0+ii,lda,_is_row_))), 0);    // A3
                    
                VMOVU_PACKED(zmm16, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(0+ii,0+j,ldb,_is_row_))), 0);
                VFMADD231_PACKED(zmm0, zmm16, zmm17);
                VFMADD231_PACKED(zmm1, zmm16, zmm18);
                VFMADD231_PACKED(zmm2, zmm16, zmm19);

                VMOVU_PACKED(zmm16, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(0+ii,1+j,ldb,_is_row_))), 0);
                VFMADD231_PACKED(zmm4, zmm16, zmm17);
                VFMADD231_PACKED(zmm5, zmm16, zmm18);
                VFMADD231_PACKED(zmm6, zmm16, zmm19);

                VMOVU_PACKED(zmm16, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(0+ii,2+j,ldb,_is_row_))), 0);
                VFMADD231_PACKED(zmm8, zmm16, zmm17);
                VFMADD231_PACKED(zmm9, zmm16, zmm18);
                VFMADD231_PACKED(zmm10, zmm16, zmm19);

                VMOVU_PACKED(zmm16, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(0+ii,3+j,ldb,_is_row_))), 0);
                VFMADD231_PACKED(zmm12, zmm16, zmm17);
                VFMADD231_PACKED(zmm13, zmm16, zmm18);
                VFMADD231_PACKED(zmm14, zmm16, zmm19);
            }
            // update the 4x4 B matrix
            // 1st
            VMOVU_PACKED(zmm20, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(0+i,0+j,ldb,_is_row_))), 0);     // B1
            VMOVU_PACKED(zmm21, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(0+i,1+j,ldb,_is_row_))), 0);     // B2
            VMOVU_PACKED(zmm22, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(0+i,2+j,ldb,_is_row_))), 0);     // B3
            VMOVU_PACKED(zmm23, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(0+i,3+j,ldb,_is_row_))), 0);     // B4

#ifdef _TCB_NOUNIT_DIAG_
            VMOVU_PACKED(zmm16, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_ap(0+i,0+i,lda,_is_row_))), 0);     // A1
            VDIV_PACKED(zmm16, zmm31, zmm16);                                                                               // A1 /= ONE
#endif

            VSUB_PACKED(zmm0, zmm20, zmm0);                                                                                 // T11 = B1-T11
#ifdef _TCB_NOUNIT_DIAG_
            VMUL_PACKED(zmm0, zmm0, zmm16);                                                                                 // T11 *= ONE/A1
#endif
            VMOVU_PACKED(zmm0, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(0+i,0+j,ldb,_is_row_))), 1);       // Store T11 -> B1

            VSUB_PACKED(zmm4, zmm21, zmm4);                                                                                 // T12 = B2-T12
#ifdef _TCB_NOUNIT_DIAG_
            VMUL_PACKED(zmm4, zmm4, zmm16);                                                                                 // T12 *= ONE/A1
#endif
            VMOVU_PACKED(zmm4, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(0+i,1+j,ldb,_is_row_))), 1);       // Store T12 -> B2

            VSUB_PACKED(zmm8, zmm22, zmm8);                                                                                 // T13 = B3-T13
#ifdef _TCB_NOUNIT_DIAG_
            VMUL_PACKED(zmm8, zmm8, zmm16);                                                                                 // T13 *= ONE/A1
#endif
            VMOVU_PACKED(zmm8, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(0+i,2+j,ldb,_is_row_))), 1);       // Store T13 -> B3

            VSUB_PACKED(zmm12, zmm23, zmm12);                                                                               // T14 = B4-T14
#ifdef _TCB_NOUNIT_DIAG_
            VMUL_PACKED(zmm12, zmm12, zmm16);                                                                               // T14 += ONE/A1
#endif
            VMOVU_PACKED(zmm12, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(0+i,3+j,ldb,_is_row_))), 1);      // Store T14 -> B4

            // 2nd
#ifdef _TCB_NOUNIT_DIAG_
            VMOVU_PACKED(zmm16, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_ap(1+i,1+i,lda,_is_row_))), 0);     // A1
            VDIV_PACKED(zmm16, zmm31, zmm16);                                                                               // A1 /= ONE
#endif
            VMOVU_PACKED(zmm17, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_ap(1+i,0+i,lda,_is_row_))), 0);     // A2

            VMOVU_PACKED(zmm20, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(1+i,0+j,ldb,_is_row_))), 0);     // B1
            VMOVU_PACKED(zmm21, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(1+i,1+j,ldb,_is_row_))), 0);     // B2
            VMOVU_PACKED(zmm22, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(1+i,2+j,ldb,_is_row_))), 0);     // B3
            VMOVU_PACKED(zmm23, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(1+i,3+j,ldb,_is_row_))), 0);     // B4

            VFMADD231_PACKED(zmm1, zmm17, zmm0);                                                                            // T21 += A2*T11
            VSUB_PACKED(zmm1, zmm20, zmm1);                                                                                 // T21 = B1 - T21
#ifdef _TCB_NOUNIT_DIAG_
            VMUL_PACKED(zmm1, zmm1, zmm16);                                                                                 // T21 *= ONE/A1
#endif
            VMOVU_PACKED(zmm1, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(1+i,0+j,ldb,_is_row_))), 1);      // Store T21 -> B1

            VFMADD231_PACKED(zmm5, zmm17, zmm4);                                                                            // T22 += A2*T12
            VSUB_PACKED(zmm5, zmm21, zmm5);                                                                                 // T22 = B2 - T22
#ifdef _TCB_NOUNIT_DIAG_
            VMUL_PACKED(zmm5, zmm5, zmm16);                                                                                 // T22 *= ONE/A1
#endif
            VMOVU_PACKED(zmm5, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(1+i,1+j,ldb,_is_row_))), 1);      // Store T22 -> B2

            VFMADD231_PACKED(zmm9, zmm17, zmm8);                                                                            // T23 += A2*T13
            VSUB_PACKED(zmm9, zmm22, zmm9);                                                                                 // T23 = B3 - T23
#ifdef _TCB_NOUNIT_DIAG_
            VMUL_PACKED(zmm9, zmm9, zmm16);                                                                                 // T23 *= ONE/A1
#endif
            VMOVU_PACKED(zmm9, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(1+i,2+j,ldb,_is_row_))), 1);      // Store T23 -> B3

            VFMADD231_PACKED(zmm13, zmm17, zmm12);                                                                          // T24 += A2*T14
            VSUB_PACKED(zmm13, zmm23, zmm13);                                                                               // T24 = B3 - T24
#ifdef _TCB_NOUNIT_DIAG_
            VMUL_PACKED(zmm13, zmm13, zmm16);                                                                               // T24 *= ONE/A1
#endif
            VMOVU_PACKED(zmm13, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(1+i,3+j,ldb,_is_row_))), 1);     // Store T24 -> B4

            // 3rd
#ifdef _TCB_NOUNIT_DIAG_
            VMOVU_PACKED(zmm16, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_ap(2+i,2+i,lda,_is_row_))), 0);     // A1
            VDIV_PACKED(zmm16, zmm31, zmm16);                                                                               // A1 /= ONE
#endif
            VMOVU_PACKED(zmm17, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_ap(2+i,0+i,lda,_is_row_))), 0);     // A2
            VMOVU_PACKED(zmm18, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_ap(2+i,1+i,lda,_is_row_))), 0);     // A3

            VMOVU_PACKED(zmm20, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(2+i,0+j,ldb,_is_row_))), 0);     // B1
            VMOVU_PACKED(zmm21, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(2+i,1+j,ldb,_is_row_))), 0);     // B2
            VMOVU_PACKED(zmm22, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(2+i,2+j,ldb,_is_row_))), 0);     // B3
            VMOVU_PACKED(zmm23, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(2+i,3+j,ldb,_is_row_))), 0);     // B4

            VFMADD231_PACKED(zmm2, zmm17, zmm0);                                                                            // T31 += A2 * T11
            VFMADD231_PACKED(zmm2, zmm18, zmm1);                                                                            // T31 += A3 * T21
            VSUB_PACKED(zmm2, zmm20, zmm2);                                                                                 // T31 = B1 - T31
#ifdef _TCB_NOUNIT_DIAG_
            VMUL_PACKED(zmm2, zmm2, zmm16);                                                                                 // T31 *= ONE/A1
#endif
            VMOVU_PACKED(zmm2, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(2+i,0+j,ldb,_is_row_))), 1);      // Store T31 -> B1

            if (n_rem > 0) SET_ZERO_PACKED(zmm0, zmm0, zmm0);                                                                   // ZERO T11
            if (n_rem > 0) SET_ZERO_PACKED(zmm1, zmm1, zmm1);                                                                   // ZERO T21
            if (n_rem > 0) SET_ZERO_PACKED(zmm2, zmm2, zmm2);                                                                   // ZERO T31

            VFMADD231_PACKED(zmm6, zmm17, zmm4);                                                                            // T32 += A2 * T12
            VFMADD231_PACKED(zmm6, zmm18, zmm5);                                                                            // T32 += A3 * T22
            VSUB_PACKED(zmm6, zmm21, zmm6);                                                                                 // T32 = B2 - T32
#ifdef _TCB_NOUNIT_DIAG_
            VMUL_PACKED(zmm6, zmm6, zmm16);                                                                                 // T32 *= ONE/A1
#endif
            VMOVU_PACKED(zmm6, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(2+i,1+j,ldb,_is_row_))), 1);      // Store T32 -> B2

            if (n_rem > 0) SET_ZERO_PACKED(zmm4, zmm4, zmm4);                                                                   // ZERO T12
            if (n_rem > 0) SET_ZERO_PACKED(zmm5, zmm5, zmm5);                                                                   // ZERO T22
            if (n_rem > 0) SET_ZERO_PACKED(zmm6, zmm6, zmm6);                                                                   // ZERO T32

            VFMADD231_PACKED(zmm10, zmm17, zmm8);                                                                           // T33 += A2 * T13
            VFMADD231_PACKED(zmm10, zmm18, zmm9);                                                                           // T33 += A3 * T23
            VSUB_PACKED(zmm10, zmm22, zmm10);                                                                               // T33 = B3 - T33
#ifdef _TCB_NOUNIT_DIAG_
            VMUL_PACKED(zmm10, zmm10, zmm16);                                                                               // T33 *= ONE/A1
#endif
            VMOVU_PACKED(zmm10, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(2+i,2+j,ldb,_is_row_))), 1);     // Store T33 -> B3

            if (n_rem > 0) SET_ZERO_PACKED(zmm8, zmm8, zmm8);                                                                   // ZERO T13
            if (n_rem > 0) SET_ZERO_PACKED(zmm9, zmm9, zmm9);                                                                   // ZERO T23
            if (n_rem > 0) SET_ZERO_PACKED(zmm10, zmm10, zmm10);                                                                // ZERO T33

            VFMADD231_PACKED(zmm14, zmm17, zmm12);                                                                          // T34 += A2 * T14
            VFMADD231_PACKED(zmm14, zmm18, zmm13);                                                                          // T34 += A3 * T24
            VSUB_PACKED(zmm14, zmm23, zmm14);                                                                               // T34 = B4 - T34
#ifdef _TCB_NOUNIT_DIAG_
            VMUL_PACKED(zmm14, zmm14, zmm16);                                                                               // T34 *= ONE/A1
 #endif           
            VMOVU_PACKED(zmm14, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(2+i,3+j,ldb,_is_row_))), 1);     // Store T34 -> B4
            
            if (n_rem > 0) SET_ZERO_PACKED(zmm12, zmm12, zmm12);                                                                // ZERO T14
            if (n_rem > 0) SET_ZERO_PACKED(zmm13, zmm13, zmm13);                                                                // ZERO T24
            if (n_rem > 0) SET_ZERO_PACKED(zmm14, zmm14, zmm14);                                                                // ZERO T34
        }
        else if (m_in & 2) {
            // gemm update
            for (ii=0; ii<i; ii++) {
                VMOVU_PACKED(zmm17, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_ap(0+i,0+ii,lda,_is_row_))), 0);    // A1
                VMOVU_PACKED(zmm18, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_ap(1+i,0+ii,lda,_is_row_))), 0);    // A2
                    
                VMOVU_PACKED(zmm16, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(0+ii,0+j,ldb,_is_row_))), 0);
                VFMADD231_PACKED(zmm0, zmm16, zmm17);
                VFMADD231_PACKED(zmm1, zmm16, zmm18);

                VMOVU_PACKED(zmm16, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(0+ii,1+j,ldb,_is_row_))), 0);
                VFMADD231_PACKED(zmm4, zmm16, zmm17);
                VFMADD231_PACKED(zmm5, zmm16, zmm18);

                VMOVU_PACKED(zmm16, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(0+ii,2+j,ldb,_is_row_))), 0);
                VFMADD231_PACKED(zmm8, zmm16, zmm17);
                VFMADD231_PACKED(zmm9, zmm16, zmm18);

                VMOVU_PACKED(zmm16, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(0+ii,3+j,ldb,_is_row_))), 0);
                VFMADD231_PACKED(zmm12, zmm16, zmm17);
                VFMADD231_PACKED(zmm13, zmm16, zmm18);
            }
            // update the 4x4 B matrix
            // 1st
            VMOVU_PACKED(zmm20, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(0+i,0+j,ldb,_is_row_))), 0);     // B1
            VMOVU_PACKED(zmm21, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(0+i,1+j,ldb,_is_row_))), 0);     // B2
            VMOVU_PACKED(zmm22, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(0+i,2+j,ldb,_is_row_))), 0);     // B3
            VMOVU_PACKED(zmm23, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(0+i,3+j,ldb,_is_row_))), 0);     // B4
#ifdef _TCB_NOUNIT_DIAG_
            VMOVU_PACKED(zmm16, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_ap(0+i,0+i,lda,_is_row_))), 0);     // A1
            VDIV_PACKED(zmm16, zmm31, zmm16);                                                                               // A1 /= ONE
#endif
            VSUB_PACKED(zmm0, zmm20, zmm0);                                                                                 // T11 = B1-T11
#ifdef _TCB_NOUNIT_DIAG_
            VMUL_PACKED(zmm0, zmm0, zmm16);                                                                                 // T11 *= ONE/A1
#endif
            VMOVU_PACKED(zmm0, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(0+i,0+j,ldb,_is_row_))), 1);       // Store T11 -> B1

            VSUB_PACKED(zmm4, zmm21, zmm4);                                                                                 // T12 = B2-T12
#ifdef _TCB_NOUNIT_DIAG_
            VMUL_PACKED(zmm4, zmm4, zmm16);                                                                                 // T12 *= ONE/A1
#endif
            VMOVU_PACKED(zmm4, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(0+i,1+j,ldb,_is_row_))), 1);       // Store T12 -> B2

            VSUB_PACKED(zmm8, zmm22, zmm8);                                                                                 // T13 = B3-T13
#ifdef _TCB_NOUNIT_DIAG_
            VMUL_PACKED(zmm8, zmm8, zmm16);                                                                                 // T13 *= ONE/A1
#endif
            VMOVU_PACKED(zmm8, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(0+i,2+j,ldb,_is_row_))), 1);       // Store T13 -> B3

            VSUB_PACKED(zmm12, zmm23, zmm12);                                                                               // T14 = B4-T14
#ifdef _TCB_NOUNIT_DIAG_
            VMUL_PACKED(zmm12, zmm12, zmm16);                                                                               // T14 += ONE/A1
#endif
            VMOVU_PACKED(zmm12, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(0+i,3+j,ldb,_is_row_))), 1);      // Store T14 -> B4

            // 2nd
#ifdef _TCB_NOUNIT_DIAG_
            VMOVU_PACKED(zmm16, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_ap(1+i,1+i,lda,_is_row_))), 0);     // A1
            VDIV_PACKED(zmm16, zmm31, zmm16);                                                                               // A1 /= ONE
#endif
            VMOVU_PACKED(zmm17, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_ap(1+i,0+i,lda,_is_row_))), 0);     // A2

            VMOVU_PACKED(zmm20, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(1+i,0+j,ldb,_is_row_))), 0);     // B1
            VMOVU_PACKED(zmm21, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(1+i,1+j,ldb,_is_row_))), 0);     // B2
            VMOVU_PACKED(zmm22, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(1+i,2+j,ldb,_is_row_))), 0);     // B3
            VMOVU_PACKED(zmm23, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(1+i,3+j,ldb,_is_row_))), 0);     // B4

            VFMADD231_PACKED(zmm1, zmm17, zmm0);                                                                            // T21 += A2*T11
            VSUB_PACKED(zmm1, zmm20, zmm1);                                                                                 // T21 = B1 - T21
#ifdef _TCB_NOUNIT_DIAG_
            VMUL_PACKED(zmm1, zmm1, zmm16);                                                                                 // T21 *= ONE/A1
#endif
            VMOVU_PACKED(zmm1, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(1+i,0+j,ldb,_is_row_))), 1);      // Store T21 -> B1

            if (n_rem > 0) SET_ZERO_PACKED(zmm0, zmm0, zmm0);                                                                   // ZERO T11
            if (n_rem > 0) SET_ZERO_PACKED(zmm1, zmm1, zmm1);                                                                   // ZERO T21

            VFMADD231_PACKED(zmm5, zmm17, zmm4);                                                                            // T22 += A2*T12
            VSUB_PACKED(zmm5, zmm21, zmm5);                                                                                 // T22 = B2 - T22
#ifdef _TCB_NOUNIT_DIAG_
            VMUL_PACKED(zmm5, zmm5, zmm16);                                                                                 // T22 *= ONE/A1
#endif
            VMOVU_PACKED(zmm5, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(1+i,1+j,ldb,_is_row_))), 1);      // Store T22 -> B2

            if (n_rem > 0) SET_ZERO_PACKED(zmm4, zmm4, zmm4);                                                                   // ZERO T12
            if (n_rem > 0) SET_ZERO_PACKED(zmm5, zmm5, zmm5);                                                                   // ZERO T22
 
            VFMADD231_PACKED(zmm9, zmm17, zmm8);                                                                            // T23 += A2*T13
            VSUB_PACKED(zmm9, zmm22, zmm9);                                                                                 // T23 = B3 - T23
#ifdef _TCB_NOUNIT_DIAG_
            VMUL_PACKED(zmm9, zmm9, zmm16);                                                                                 // T23 *= ONE/A1
#endif
            VMOVU_PACKED(zmm9, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(1+i,2+j,ldb,_is_row_))), 1);      // Store T23 -> B3

            if (n_rem > 0) SET_ZERO_PACKED(zmm8, zmm8, zmm8);                                                                   // ZERO T13
            if (n_rem > 0) SET_ZERO_PACKED(zmm9, zmm9, zmm9);                                                                   // ZERO T23
 
            VFMADD231_PACKED(zmm13, zmm17, zmm12);                                                                          // T24 += A2*T14
            VSUB_PACKED(zmm13, zmm23, zmm13);                                                                               // T24 = B3 - T24
#ifdef _TCB_NOUNIT_DIAG_
            VMUL_PACKED(zmm13, zmm13, zmm16);                                                                               // T24 *= ONE/A1
#endif
            VMOVU_PACKED(zmm13, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(1+i,3+j,ldb,_is_row_))), 1);     // Store T24 -> B4

            if (n_rem > 0) SET_ZERO_PACKED(zmm12, zmm12, zmm12);                                                                // ZERO T14
            if (n_rem > 0) SET_ZERO_PACKED(zmm13, zmm13, zmm13);                                                                // ZERO T24
        }
        else if (m_in & 1) {
           // gemm update
            for (ii=0; ii<i; ii++) {
                VMOVU_PACKED(zmm17, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_ap(0+i,0+ii,lda,_is_row_))), 0);    // A1
                    
                VMOVU_PACKED(zmm16, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(0+ii,0+j,ldb,_is_row_))), 0);
                VFMADD231_PACKED(zmm0, zmm16, zmm17);

                VMOVU_PACKED(zmm16, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(0+ii,1+j,ldb,_is_row_))), 0);
                VFMADD231_PACKED(zmm4, zmm16, zmm17);

                VMOVU_PACKED(zmm16, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(0+ii,2+j,ldb,_is_row_))), 0);
                VFMADD231_PACKED(zmm8, zmm16, zmm17);

                VMOVU_PACKED(zmm16, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(0+ii,3+j,ldb,_is_row_))), 0);
                VFMADD231_PACKED(zmm12, zmm16, zmm17);
            }
            // update the 4x4 B matrix
            // 1st
            VMOVU_PACKED(zmm20, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(0+i,0+j,ldb,_is_row_))), 0);     // B1
            VMOVU_PACKED(zmm21, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(0+i,1+j,ldb,_is_row_))), 0);     // B2
            VMOVU_PACKED(zmm22, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(0+i,2+j,ldb,_is_row_))), 0);     // B3
            VMOVU_PACKED(zmm23, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(0+i,3+j,ldb,_is_row_))), 0);     // B4
#ifdef _TCB_NOUNIT_DIAG_
            VMOVU_PACKED(zmm16, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_ap(0+i,0+i,lda,_is_row_))), 0);     // A1
            VDIV_PACKED(zmm16, zmm31, zmm16);                                                                               // A1 /= ONE
#endif

            VSUB_PACKED(zmm0, zmm20, zmm0);                                                                                 // T11 = B1-T11
#ifdef _TCB_NOUNIT_DIAG_
            VMUL_PACKED(zmm0, zmm0, zmm16);                                                                                 // T11 *= ONE/A1
#endif
            VMOVU_PACKED(zmm0, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(0+i,0+j,ldb,_is_row_))), 1);      // Store T11 -> B1

            if (n_rem > 0) SET_ZERO_PACKED(zmm0, zmm0, zmm0);                                                                   // ZERO T11

            VSUB_PACKED(zmm4, zmm21, zmm4);                                                                                 // T12 = B2-T12
#ifdef _TCB_NOUNIT_DIAG_
            VMUL_PACKED(zmm4, zmm4, zmm16);                                                                                 // T12 *= ONE/A1
#endif
            VMOVU_PACKED(zmm4, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(0+i,1+j,ldb,_is_row_))), 1);      // Store T12 -> B2

            if (n_rem > 0) SET_ZERO_PACKED(zmm4, zmm4, zmm4);                                                                   // ZERO T12

            VSUB_PACKED(zmm8, zmm22, zmm8);                                                                                 // T13 = B3-T13
#ifdef _TCB_NOUNIT_DIAG_
            VMUL_PACKED(zmm8, zmm8, zmm16);                                                                                 // T13 *= ONE/A1
#endif
            VMOVU_PACKED(zmm8, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(0+i,2+j,ldb,_is_row_))), 1);      // Store T13 -> B3

            if (n_rem > 0) SET_ZERO_PACKED(zmm8, zmm8, zmm8);                                                                   // ZERO T13

            VSUB_PACKED(zmm12, zmm23, zmm12);                                                                               // T14 = B4-T14
#ifdef _TCB_NOUNIT_DIAG_
            VMUL_PACKED(zmm12, zmm12, zmm16);                                                                               // T14 += ONE/A1
#endif
            VMOVU_PACKED(zmm12, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(0+i,3+j,ldb,_is_row_))), 1);     // Store T14 -> B4
 
            if (n_rem > 0) SET_ZERO_PACKED(zmm12, zmm12, zmm12);                                                                // ZERO T14
        }
    } // n loop
    // n tails
    if (n_in & 2 && n_in & 1) {
        m_rem = m_in;
        for (i=0; i<(m_in/M_UNROLL_AVX512)*M_UNROLL_AVX512; i+=M_UNROLL_AVX512) {
            m_rem -= M_UNROLL_AVX512;
            // gemm update
            for (ii=0; ii<i; ii++) {
                VMOVU_PACKED(zmm17, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_ap(0+i,0+ii,lda,_is_row_))), 0);    // A1
                VMOVU_PACKED(zmm18, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_ap(1+i,0+ii,lda,_is_row_))), 0);    // A2
                VMOVU_PACKED(zmm19, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_ap(2+i,0+ii,lda,_is_row_))), 0);    // A3
                VMOVU_PACKED(zmm20, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_ap(3+i,0+ii,lda,_is_row_))), 0);    // A4
                    
                VMOVU_PACKED(zmm16, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(0+ii,0+j,ldb,_is_row_))), 0);
                VFMADD231_PACKED(zmm0, zmm16, zmm17);
                VFMADD231_PACKED(zmm1, zmm16, zmm18);
                VFMADD231_PACKED(zmm2, zmm16, zmm19);
                VFMADD231_PACKED(zmm3, zmm16, zmm20);

                VMOVU_PACKED(zmm16, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(0+ii,1+j,ldb,_is_row_))), 0);
                VFMADD231_PACKED(zmm4, zmm16, zmm17);
                VFMADD231_PACKED(zmm5, zmm16, zmm18);
                VFMADD231_PACKED(zmm6, zmm16, zmm19);
                VFMADD231_PACKED(zmm7, zmm16, zmm20);

                VMOVU_PACKED(zmm16, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(0+ii,2+j,ldb,_is_row_))), 0);
                VFMADD231_PACKED(zmm8, zmm16, zmm17);
                VFMADD231_PACKED(zmm9, zmm16, zmm18);
                VFMADD231_PACKED(zmm10, zmm16, zmm19);
                VFMADD231_PACKED(zmm11, zmm16, zmm20);

            }
            // update the 4x4 B matrix
            // 1st
            VMOVU_PACKED(zmm20, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(0+i,0+j,ldb,_is_row_))), 0);     // B1
            VMOVU_PACKED(zmm21, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(0+i,1+j,ldb,_is_row_))), 0);     // B2
            VMOVU_PACKED(zmm22, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(0+i,2+j,ldb,_is_row_))), 0);     // B3
#ifdef _TCB_NOUNIT_DIAG_
            VMOVU_PACKED(zmm16, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_ap(0+i,0+i,lda,_is_row_))), 0);     // A1
            VDIV_PACKED(zmm16, zmm31, zmm16);                                                                               // A1 /= ONE
#endif
            VSUB_PACKED(zmm0, zmm20, zmm0);                                                                                 // T11 = B1-T11
#ifdef _TCB_NOUNIT_DIAG_
            VMUL_PACKED(zmm0, zmm0, zmm16);                                                                                 // T11 *= ONE/A1
#endif
            VMOVU_PACKED(zmm0, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(0+i,0+j,ldb,_is_row_))), 1);       // Store T11 -> B1

            VSUB_PACKED(zmm4, zmm21, zmm4);                                                                                 // T12 = B2-T12
#ifdef _TCB_NOUNIT_DIAG_
            VMUL_PACKED(zmm4, zmm4, zmm16);                                                                                 // T12 *= ONE/A1
#endif
            VMOVU_PACKED(zmm4, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(0+i,1+j,ldb,_is_row_))), 1);       // Store T12 -> B2

            VSUB_PACKED(zmm8, zmm22, zmm8);                                                                                 // T13 = B3-T13
#ifdef _TCB_NOUNIT_DIAG_
            VMUL_PACKED(zmm8, zmm8, zmm16);                                                                                 // T13 *= ONE/A1
#endif
            VMOVU_PACKED(zmm8, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(0+i,2+j,ldb,_is_row_))), 1);       // Store T13 -> B3

            // 2nd
#ifdef _TCB_NOUNIT_DIAG_
            VMOVU_PACKED(zmm16, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_ap(1+i,1+i,lda,_is_row_))), 0);     // A1
            VDIV_PACKED(zmm16, zmm31, zmm16);                                                                               // A1 /= ONE
#endif
            VMOVU_PACKED(zmm17, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_ap(1+i,0+i,lda,_is_row_))), 0);     // A2

            VMOVU_PACKED(zmm20, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(1+i,0+j,ldb,_is_row_))), 0);     // B1
            VMOVU_PACKED(zmm21, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(1+i,1+j,ldb,_is_row_))), 0);     // B2
            VMOVU_PACKED(zmm22, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(1+i,2+j,ldb,_is_row_))), 0);     // B3

            VFMADD231_PACKED(zmm1, zmm17, zmm0);                                                                            // T21 += A2*T11
            VSUB_PACKED(zmm1, zmm20, zmm1);                                                                                 // T21 = B1 - T21
#ifdef _TCB_NOUNIT_DIAG_
            VMUL_PACKED(zmm1, zmm1, zmm16);                                                                                 // T21 *= ONE/A1
#endif
            VMOVU_PACKED(zmm1, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(1+i,0+j,ldb,_is_row_))), 1);      // Store T21 -> B1

            VFMADD231_PACKED(zmm5, zmm17, zmm4);                                                                            // T22 += A2*T12
            VSUB_PACKED(zmm5, zmm21, zmm5);                                                                                 // T22 = B2 - T22
#ifdef _TCB_NOUNIT_DIAG_
            VMUL_PACKED(zmm5, zmm5, zmm16);                                                                                 // T22 *= ONE/A1
#endif
            VMOVU_PACKED(zmm5, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(1+i,1+j,ldb,_is_row_))), 1);      // Store T22 -> B2

            VFMADD231_PACKED(zmm9, zmm17, zmm8);                                                                            // T23 += A2*T13
            VSUB_PACKED(zmm9, zmm22, zmm9);                                                                                 // T23 = B3 - T23
#ifdef _TCB_NOUNIT_DIAG_
            VMUL_PACKED(zmm9, zmm9, zmm16);                                                                                 // T23 *= ONE/A1
#endif
            VMOVU_PACKED(zmm9, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(1+i,2+j,ldb,_is_row_))), 1);      // Store T23 -> B3

            // 3rd
#ifdef _TCB_NOUNIT_DIAG_
            VMOVU_PACKED(zmm16, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_ap(2+i,2+i,lda,_is_row_))), 0);     // A1
            VDIV_PACKED(zmm16, zmm31, zmm16);                                                                               // A1 /= ONE
#endif
            VMOVU_PACKED(zmm17, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_ap(2+i,0+i,lda,_is_row_))), 0);     // A2
            VMOVU_PACKED(zmm18, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_ap(2+i,1+i,lda,_is_row_))), 0);     // A3

            VMOVU_PACKED(zmm20, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(2+i,0+j,ldb,_is_row_))), 0);     // B1
            VMOVU_PACKED(zmm21, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(2+i,1+j,ldb,_is_row_))), 0);     // B2
            VMOVU_PACKED(zmm22, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(2+i,2+j,ldb,_is_row_))), 0);     // B3

            VFMADD231_PACKED(zmm2, zmm17, zmm0);                                                                            // T31 += A2 * T11
            VFMADD231_PACKED(zmm2, zmm18, zmm1);                                                                            // T31 += A3 * T21
            VSUB_PACKED(zmm2, zmm20, zmm2);                                                                                 // T31 = B1 - T31
#ifdef _TCB_NOUNIT_DIAG_
            VMUL_PACKED(zmm2, zmm2, zmm16);                                                                                 // T31 *= ONE/A1
#endif
            VMOVU_PACKED(zmm2, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(2+i,0+j,ldb,_is_row_))), 1);      // Store T31 -> B1

            VFMADD231_PACKED(zmm6, zmm17, zmm4);                                                                            // T32 += A2 * T12
            VFMADD231_PACKED(zmm6, zmm18, zmm5);                                                                            // T32 += A3 * T22
            VSUB_PACKED(zmm6, zmm21, zmm6);                                                                                 // T32 = B2 - T32
#ifdef _TCB_NOUNIT_DIAG_
            VMUL_PACKED(zmm6, zmm6, zmm16);                                                                                 // T32 *= ONE/A1
#endif
            VMOVU_PACKED(zmm6, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(2+i,1+j,ldb,_is_row_))), 1);      // Store T32 -> B2

            VFMADD231_PACKED(zmm10, zmm17, zmm8);                                                                           // T33 += A2 * T13
            VFMADD231_PACKED(zmm10, zmm18, zmm9);                                                                           // T33 += A3 * T23
            VSUB_PACKED(zmm10, zmm22, zmm10);                                                                               // T33 = B3 - T33
#ifdef _TCB_NOUNIT_DIAG_
            VMUL_PACKED(zmm10, zmm10, zmm16);                                                                               // T33 *= ONE/A1
#endif
            VMOVU_PACKED(zmm10, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(2+i,2+j,ldb,_is_row_))), 1);     // Store T33 -> B3

            // 4th
#ifdef _TCB_NOUNIT_DIAG_
            VMOVU_PACKED(zmm16, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_ap(3+i,3+i,lda,_is_row_))), 0);     // A1
            VDIV_PACKED(zmm16, zmm31, zmm16);                                                                               // A1 /= ONE
#endif
            VMOVU_PACKED(zmm17, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_ap(3+i,0+i,lda,_is_row_))), 0);     // A2
            VMOVU_PACKED(zmm18, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_ap(3+i,1+i,lda,_is_row_))), 0);     // A3
            VMOVU_PACKED(zmm19, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_ap(3+i,2+i,lda,_is_row_))), 0);     // A4
            
            VMOVU_PACKED(zmm20, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(3+i,0+j,ldb,_is_row_))), 0);     // B1
            VMOVU_PACKED(zmm21, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(3+i,1+j,ldb,_is_row_))), 0);     // B2
            VMOVU_PACKED(zmm22, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(3+i,2+j,ldb,_is_row_))), 0);     // B3

            VFMADD231_PACKED(zmm3, zmm17, zmm0);                                                                            // T41 += A2 * T11
            VFMADD231_PACKED(zmm3, zmm18, zmm1);                                                                            // T41 += A3 * T21
            VFMADD231_PACKED(zmm3, zmm19, zmm2);                                                                            // T41 += A4 * T31
            VSUB_PACKED(zmm3, zmm20, zmm3);                                                                                 // T41 = B1 - T41
#ifdef _TCB_NOUNIT_DIAG_
            VMUL_PACKED(zmm3, zmm3, zmm16);                                                                                 // T41 *= ONE/A1
#endif
            VMOVU_PACKED(zmm3, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(3+i,0+j,ldb,_is_row_))), 1);      // Store T41 -> B1

            SET_ZERO_PACKED(zmm0, zmm0, zmm0);                                                      // ZERO T11
            SET_ZERO_PACKED(zmm1, zmm1, zmm1);                                                      // ZERO T21
            SET_ZERO_PACKED(zmm2, zmm2, zmm2);                                                      // ZERO T31
            SET_ZERO_PACKED(zmm3, zmm3, zmm3);                                                      // ZERO T41

            VFMADD231_PACKED(zmm7, zmm17, zmm4);                                                                            // T42 += A2 * T12
            VFMADD231_PACKED(zmm7, zmm18, zmm5);                                                                            // T42 += A3 * T22
            VFMADD231_PACKED(zmm7, zmm19, zmm6);                                                                            // T42 += A4 * T32
            VSUB_PACKED(zmm7, zmm21, zmm7);                                                                                 // T42 = B2 - T42
#ifdef _TCB_NOUNIT_DIAG_
            VMUL_PACKED(zmm7, zmm7, zmm16);                                                                                 // T42 *= ONE/A1
#endif
            VMOVU_PACKED(zmm7, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(3+i,1+j,ldb,_is_row_))), 1);      // Store T42 -> B2

            SET_ZERO_PACKED(zmm4, zmm4, zmm4);                                                      // ZERO T12
            SET_ZERO_PACKED(zmm5, zmm5, zmm5);                                                      // ZERO T22
            SET_ZERO_PACKED(zmm6, zmm6, zmm6);                                                      // ZERO T32
            SET_ZERO_PACKED(zmm7, zmm7, zmm7);                                                      // ZERO T42

            VFMADD231_PACKED(zmm11, zmm17, zmm8);                                                                           // T43 += A2 * T13
            VFMADD231_PACKED(zmm11, zmm18, zmm9);                                                                           // T43 += A3 * T23
            VFMADD231_PACKED(zmm11, zmm19, zmm10);                                                                          // T43 += A4 * T33
            VSUB_PACKED(zmm11, zmm22, zmm11);                                                                               // T43 = B3 - T43
#ifdef _TCB_NOUNIT_DIAG_
            VMUL_PACKED(zmm11, zmm11, zmm16);                                                                               // T43 *= ONE/A1
#endif
            VMOVU_PACKED(zmm11, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(3+i,2+j,ldb,_is_row_))), 1);     // STORE T43 -> B3

            SET_ZERO_PACKED(zmm8, zmm8, zmm8);                                                      // ZERO T13
            SET_ZERO_PACKED(zmm9, zmm9, zmm9);                                                      // ZERO T23
            SET_ZERO_PACKED(zmm10, zmm10, zmm10);                                                   // ZERO T33
            SET_ZERO_PACKED(zmm11, zmm11, zmm11);                                                   // ZERO T43
        } // m loop
        // m tails
        if (m_in & 2 && m_in & 1) {
            // gemm update
            for (ii=0; ii<i; ii++) {
                VMOVU_PACKED(zmm17, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_ap(0+i,0+ii,lda,_is_row_))), 0);    // A1
                VMOVU_PACKED(zmm18, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_ap(1+i,0+ii,lda,_is_row_))), 0);    // A2
                VMOVU_PACKED(zmm19, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_ap(2+i,0+ii,lda,_is_row_))), 0);    // A3
                    
                VMOVU_PACKED(zmm16, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(0+ii,0+j,ldb,_is_row_))), 0);
                VFMADD231_PACKED(zmm0, zmm16, zmm17);
                VFMADD231_PACKED(zmm1, zmm16, zmm18);
                VFMADD231_PACKED(zmm2, zmm16, zmm19);

                VMOVU_PACKED(zmm16, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(0+ii,1+j,ldb,_is_row_))), 0);
                VFMADD231_PACKED(zmm4, zmm16, zmm17);
                VFMADD231_PACKED(zmm5, zmm16, zmm18);
                VFMADD231_PACKED(zmm6, zmm16, zmm19);

                VMOVU_PACKED(zmm16, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(0+ii,2+j,ldb,_is_row_))), 0);
                VFMADD231_PACKED(zmm8, zmm16, zmm17);
                VFMADD231_PACKED(zmm9, zmm16, zmm18);
                VFMADD231_PACKED(zmm10, zmm16, zmm19);
            }
            // update the 4x4 B matrix
            // 1st
            VMOVU_PACKED(zmm20, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(0+i,0+j,ldb,_is_row_))), 0);     // B1
            VMOVU_PACKED(zmm21, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(0+i,1+j,ldb,_is_row_))), 0);     // B2
            VMOVU_PACKED(zmm22, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(0+i,2+j,ldb,_is_row_))), 0);     // B3
#ifdef _TCB_NOUNIT_DIAG_
            VMOVU_PACKED(zmm16, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_ap(0+i,0+i,lda,_is_row_))), 0);     // A1
            VDIV_PACKED(zmm16, zmm31, zmm16);                                                                               // A1 /= ONE
#endif
            VSUB_PACKED(zmm0, zmm20, zmm0);                                                                                 // T11 = B1-T11
#ifdef _TCB_NOUNIT_DIAG_
            VMUL_PACKED(zmm0, zmm0, zmm16);                                                                                 // T11 *= ONE/A1
#endif
            VMOVU_PACKED(zmm0, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(0+i,0+j,ldb,_is_row_))), 1);       // Store T11 -> B1

            VSUB_PACKED(zmm4, zmm21, zmm4);                                                                                 // T12 = B2-T12
#ifdef _TCB_NOUNIT_DIAG_
            VMUL_PACKED(zmm4, zmm4, zmm16);                                                                                 // T12 *= ONE/A1
#endif
            VMOVU_PACKED(zmm4, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(0+i,1+j,ldb,_is_row_))), 1);       // Store T12 -> B2

            VSUB_PACKED(zmm8, zmm22, zmm8);                                                                                 // T13 = B3-T13
#ifdef _TCB_NOUNIT_DIAG_
            VMUL_PACKED(zmm8, zmm8, zmm16);                                                                                 // T13 *= ONE/A1
#endif
            VMOVU_PACKED(zmm8, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(0+i,2+j,ldb,_is_row_))), 1);       // Store T13 -> B3

            // 2nd
#ifdef _TCB_NOUNIT_DIAG_
            VMOVU_PACKED(zmm16, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_ap(1+i,1+i,lda,_is_row_))), 0);     // A1
            VDIV_PACKED(zmm16, zmm31, zmm16);                                                                               // A1 /= ONE
#endif
            VMOVU_PACKED(zmm17, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_ap(1+i,0+i,lda,_is_row_))), 0);     // A2

            VMOVU_PACKED(zmm20, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(1+i,0+j,ldb,_is_row_))), 0);     // B1
            VMOVU_PACKED(zmm21, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(1+i,1+j,ldb,_is_row_))), 0);     // B2
            VMOVU_PACKED(zmm22, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(1+i,2+j,ldb,_is_row_))), 0);     // B3

            VFMADD231_PACKED(zmm1, zmm17, zmm0);                                                                            // T21 += A2*T11
            VSUB_PACKED(zmm1, zmm20, zmm1);                                                                                 // T21 = B1 - T21
#ifdef _TCB_NOUNIT_DIAG_
            VMUL_PACKED(zmm1, zmm1, zmm16);                                                                                 // T21 *= ONE/A1
#endif
            VMOVU_PACKED(zmm1, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(1+i,0+j,ldb,_is_row_))), 1);      // Store T21 -> B1

            VFMADD231_PACKED(zmm5, zmm17, zmm4);                                                                            // T22 += A2*T12
            VSUB_PACKED(zmm5, zmm21, zmm5);                                                                                 // T22 = B2 - T22
#ifdef _TCB_NOUNIT_DIAG_
            VMUL_PACKED(zmm5, zmm5, zmm16);                                                                                 // T22 *= ONE/A1
#endif
            VMOVU_PACKED(zmm5, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(1+i,1+j,ldb,_is_row_))), 1);      // Store T22 -> B2

            VFMADD231_PACKED(zmm9, zmm17, zmm8);                                                                            // T23 += A2*T13
            VSUB_PACKED(zmm9, zmm22, zmm9);                                                                                 // T23 = B3 - T23
#ifdef _TCB_NOUNIT_DIAG_
            VMUL_PACKED(zmm9, zmm9, zmm16);                                                                                 // T23 *= ONE/A1
#endif
            VMOVU_PACKED(zmm9, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(1+i,2+j,ldb,_is_row_))), 1);      // Store T23 -> B3

            // 3rd
#ifdef _TCB_NOUNIT_DIAG_
            VMOVU_PACKED(zmm16, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_ap(2+i,2+i,lda,_is_row_))), 0);     // A1
            VDIV_PACKED(zmm16, zmm31, zmm16);                                                                               // A1 /= ONE
#endif

            VMOVU_PACKED(zmm17, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_ap(2+i,0+i,lda,_is_row_))), 0);     // A2
            VMOVU_PACKED(zmm18, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_ap(2+i,1+i,lda,_is_row_))), 0);     // A3

            VMOVU_PACKED(zmm20, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(2+i,0+j,ldb,_is_row_))), 0);     // B1
            VMOVU_PACKED(zmm21, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(2+i,1+j,ldb,_is_row_))), 0);     // B2
            VMOVU_PACKED(zmm22, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(2+i,2+j,ldb,_is_row_))), 0);     // B3

            VFMADD231_PACKED(zmm2, zmm17, zmm0);                                                                            // T31 += A2 * T11
            VFMADD231_PACKED(zmm2, zmm18, zmm1);                                                                            // T31 += A3 * T21
            VSUB_PACKED(zmm2, zmm20, zmm2);                                                                                 // T31 = B1 - T31
#ifdef _TCB_NOUNIT_DIAG_
            VMUL_PACKED(zmm2, zmm2, zmm16);                                                                                 // T31 *= ONE/A1
#endif
            VMOVU_PACKED(zmm2, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(2+i,0+j,ldb,_is_row_))), 1);      // Store T31 -> B1

            if (n_rem > 0) SET_ZERO_PACKED(zmm0, zmm0, zmm0);                                                                   // ZERO T11
            if (n_rem > 0) SET_ZERO_PACKED(zmm1, zmm1, zmm1);                                                                   // ZERO T21
            if (n_rem > 0) SET_ZERO_PACKED(zmm2, zmm2, zmm2);                                                                   // ZERO T31

            VFMADD231_PACKED(zmm6, zmm17, zmm4);                                                                            // T32 += A2 * T12
            VFMADD231_PACKED(zmm6, zmm18, zmm5);                                                                            // T32 += A3 * T22
            VSUB_PACKED(zmm6, zmm21, zmm6);                                                                                 // T32 = B2 - T32
#ifdef _TCB_NOUNIT_DIAG_
            VMUL_PACKED(zmm6, zmm6, zmm16);                                                                                 // T32 *= ONE/A1
#endif
            VMOVU_PACKED(zmm6, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(2+i,1+j,ldb,_is_row_))), 1);      // Store T32 -> B2

            if (n_rem > 0) SET_ZERO_PACKED(zmm4, zmm4, zmm4);                                                                   // ZERO T12
            if (n_rem > 0) SET_ZERO_PACKED(zmm5, zmm5, zmm5);                                                                   // ZERO T22
            if (n_rem > 0) SET_ZERO_PACKED(zmm6, zmm6, zmm6);                                                                   // ZERO T32

            VFMADD231_PACKED(zmm10, zmm17, zmm8);                                                                           // T33 += A2 * T13
            VFMADD231_PACKED(zmm10, zmm18, zmm9);                                                                           // T33 += A3 * T23
            VSUB_PACKED(zmm10, zmm22, zmm10);                                                                               // T33 = B3 - T33
#ifdef _TCB_NOUNIT_DIAG_
            VMUL_PACKED(zmm10, zmm10, zmm16);                                                                               // T33 *= ONE/A1
#endif
            VMOVU_PACKED(zmm10, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(2+i,2+j,ldb,_is_row_))), 1);     // Store T33 -> B3

            if (n_rem > 0) SET_ZERO_PACKED(zmm8, zmm8, zmm8);                                                                   // ZERO T13
            if (n_rem > 0) SET_ZERO_PACKED(zmm9, zmm9, zmm9);                                                                   // ZERO T23
            if (n_rem > 0) SET_ZERO_PACKED(zmm10, zmm10, zmm10);                                                                // ZERO T33
        }
        else if (m_in & 2) {
            // gemm update
            for (ii=0; ii<i; ii++) {
                VMOVU_PACKED(zmm17, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_ap(0+i,0+ii,lda,_is_row_))), 0);    // A1
                VMOVU_PACKED(zmm18, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_ap(1+i,0+ii,lda,_is_row_))), 0);    // A2
                    
                VMOVU_PACKED(zmm16, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(0+ii,0+j,ldb,_is_row_))), 0);
                VFMADD231_PACKED(zmm0, zmm16, zmm17);
                VFMADD231_PACKED(zmm1, zmm16, zmm18);

                VMOVU_PACKED(zmm16, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(0+ii,1+j,ldb,_is_row_))), 0);
                VFMADD231_PACKED(zmm4, zmm16, zmm17);
                VFMADD231_PACKED(zmm5, zmm16, zmm18);

                VMOVU_PACKED(zmm16, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(0+ii,2+j,ldb,_is_row_))), 0);
                VFMADD231_PACKED(zmm8, zmm16, zmm17);
                VFMADD231_PACKED(zmm9, zmm16, zmm18);

            }
            // update the 4x4 B matrix
            // 1st
            VMOVU_PACKED(zmm20, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(0+i,0+j,ldb,_is_row_))), 0);     // B1
            VMOVU_PACKED(zmm21, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(0+i,1+j,ldb,_is_row_))), 0);     // B2
            VMOVU_PACKED(zmm22, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(0+i,2+j,ldb,_is_row_))), 0);     // B3
#ifdef _TCB_NOUNIT_DIAG_
            VMOVU_PACKED(zmm16, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_ap(0+i,0+i,lda,_is_row_))), 0);     // A1
            VDIV_PACKED(zmm16, zmm31, zmm16);                                                                               // A1 /= ONE
#endif
            VSUB_PACKED(zmm0, zmm20, zmm0);                                                                                 // T11 = B1-T11
#ifdef _TCB_NOUNIT_DIAG_
            VMUL_PACKED(zmm0, zmm0, zmm16);                                                                                 // T11 *= ONE/A1
#endif
            VMOVU_PACKED(zmm0, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(0+i,0+j,ldb,_is_row_))), 1);       // Store T11 -> B1

            VSUB_PACKED(zmm4, zmm21, zmm4);                                                                                 // T12 = B2-T12
#ifdef _TCB_NOUNIT_DIAG_
            VMUL_PACKED(zmm4, zmm4, zmm16);                                                                                 // T12 *= ONE/A1
#endif
            VMOVU_PACKED(zmm4, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(0+i,1+j,ldb,_is_row_))), 1);       // Store T12 -> B2

            VSUB_PACKED(zmm8, zmm22, zmm8);                                                                                 // T13 = B3-T13
#ifdef _TCB_NOUNIT_DIAG_
            VMUL_PACKED(zmm8, zmm8, zmm16);                                                                                 // T13 *= ONE/A1
#endif
            VMOVU_PACKED(zmm8, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(0+i,2+j,ldb,_is_row_))), 1);       // Store T13 -> B3

            // 2nd
#ifdef _TCB_NOUNIT_DIAG_
            VMOVU_PACKED(zmm16, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_ap(1+i,1+i,lda,_is_row_))), 0);     // A1
            VDIV_PACKED(zmm16, zmm31, zmm16);                                                                               // A1 /= ONE
#endif
            VMOVU_PACKED(zmm17, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_ap(1+i,0+i,lda,_is_row_))), 0);     // A2

            VMOVU_PACKED(zmm20, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(1+i,0+j,ldb,_is_row_))), 0);     // B1
            VMOVU_PACKED(zmm21, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(1+i,1+j,ldb,_is_row_))), 0);     // B2
            VMOVU_PACKED(zmm22, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(1+i,2+j,ldb,_is_row_))), 0);     // B3

            VFMADD231_PACKED(zmm1, zmm17, zmm0);                                                                            // T21 += A2*T11
            VSUB_PACKED(zmm1, zmm20, zmm1);                                                                                 // T21 = B1 - T21
#ifdef _TCB_NOUNIT_DIAG_
            VMUL_PACKED(zmm1, zmm1, zmm16);                                                                                 // T21 *= ONE/A1
#endif
            VMOVU_PACKED(zmm1, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(1+i,0+j,ldb,_is_row_))), 1);      // Store T21 -> B1

            if (n_rem > 0) SET_ZERO_PACKED(zmm0, zmm0, zmm0);                                                                   // ZERO T11
            if (n_rem > 0) SET_ZERO_PACKED(zmm1, zmm1, zmm1);                                                                   // ZERO T21

            VFMADD231_PACKED(zmm5, zmm17, zmm4);                                                                            // T22 += A2*T12
            VSUB_PACKED(zmm5, zmm21, zmm5);                                                                                 // T22 = B2 - T22
#ifdef _TCB_NOUNIT_DIAG_
            VMUL_PACKED(zmm5, zmm5, zmm16);                                                                                 // T22 *= ONE/A1
#endif
            VMOVU_PACKED(zmm5, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(1+i,1+j,ldb,_is_row_))), 1);      // Store T22 -> B2

            if (n_rem > 0) SET_ZERO_PACKED(zmm4, zmm4, zmm4);                                                                   // ZERO T12
            if (n_rem > 0) SET_ZERO_PACKED(zmm5, zmm5, zmm5);                                                                   // ZERO T22
 
            VFMADD231_PACKED(zmm9, zmm17, zmm8);                                                                            // T23 += A2*T13
            VSUB_PACKED(zmm9, zmm22, zmm9);                                                                                 // T23 = B3 - T23
#ifdef _TCB_NOUNIT_DIAG_
            VMUL_PACKED(zmm9, zmm9, zmm16);                                                                                 // T23 *= ONE/A1
#endif
            VMOVU_PACKED(zmm9, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(1+i,2+j,ldb,_is_row_))), 1);      // Store T23 -> B3

            if (n_rem > 0) SET_ZERO_PACKED(zmm8, zmm8, zmm8);                                                                   // ZERO T13
            if (n_rem > 0) SET_ZERO_PACKED(zmm9, zmm9, zmm9);                                                                   // ZERO T23
        }
        else if (m_in & 1) {
           // gemm update
            for (ii=0; ii<i; ii++) {
                VMOVU_PACKED(zmm17, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_ap(0+i,0+ii,lda,_is_row_))), 0);    // A1
                    
                VMOVU_PACKED(zmm16, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(0+ii,0+j,ldb,_is_row_))), 0);
                VFMADD231_PACKED(zmm0, zmm16, zmm17);

                VMOVU_PACKED(zmm16, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(0+ii,1+j,ldb,_is_row_))), 0);
                VFMADD231_PACKED(zmm4, zmm16, zmm17);

                VMOVU_PACKED(zmm16, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(0+ii,2+j,ldb,_is_row_))), 0);
                VFMADD231_PACKED(zmm8, zmm16, zmm17);
            }
            // update the 4x4 B matrix
            // 1st
            VMOVU_PACKED(zmm20, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(0+i,0+j,ldb,_is_row_))), 0);     // B1
            VMOVU_PACKED(zmm21, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(0+i,1+j,ldb,_is_row_))), 0);     // B2
            VMOVU_PACKED(zmm22, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(0+i,2+j,ldb,_is_row_))), 0);     // B3
#ifdef _TCB_NOUNIT_DIAG_
            VMOVU_PACKED(zmm16, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_ap(0+i,0+i,lda,_is_row_))), 0);     // A1
            VDIV_PACKED(zmm16, zmm31, zmm16);                                                                               // A1 /= ONE
#endif
            VSUB_PACKED(zmm0, zmm20, zmm0);                                                                                 // T11 = B1-T11
#ifdef _TCB_NOUNIT_DIAG_
            VMUL_PACKED(zmm0, zmm0, zmm16);                                                                                 // T11 *= ONE/A1
#endif
            VMOVU_PACKED(zmm0, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(0+i,0+j,ldb,_is_row_))), 1);      // Store T11 -> B1

            if (n_rem > 0) SET_ZERO_PACKED(zmm0, zmm0, zmm0);                                                                   // ZERO T11

            VSUB_PACKED(zmm4, zmm21, zmm4);                                                                                 // T12 = B2-T12
#ifdef _TCB_NOUNIT_DIAG_
            VMUL_PACKED(zmm4, zmm4, zmm16);                                                                                 // T12 *= ONE/A1
#endif
            VMOVU_PACKED(zmm4, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(0+i,1+j,ldb,_is_row_))), 1);      // Store T12 -> B2

            if (n_rem > 0) SET_ZERO_PACKED(zmm4, zmm4, zmm4);                                                                   // ZERO T12

            VSUB_PACKED(zmm8, zmm22, zmm8);                                                                                 // T13 = B3-T13
#ifdef _TCB_NOUNIT_DIAG_
            VMUL_PACKED(zmm8, zmm8, zmm16);                                                                                 // T13 *= ONE/A1
#endif
            VMOVU_PACKED(zmm8, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(0+i,2+j,ldb,_is_row_))), 1);      // Store T13 -> B3

            if (n_rem > 0) SET_ZERO_PACKED(zmm8, zmm8, zmm8);                                                                   // ZERO T13
        }
    }
    else if (n_in & 2) {
        m_rem = m_in;
        for (i=0; i<(m_in/M_UNROLL_AVX512)*M_UNROLL_AVX512; i+=M_UNROLL_AVX512) {
            m_rem -= M_UNROLL_AVX512;
            // gemm update
            for (ii=0; ii<i; ii++) {
                VMOVU_PACKED(zmm17, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_ap(0+i,0+ii,lda,_is_row_))), 0);    // A1
                VMOVU_PACKED(zmm18, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_ap(1+i,0+ii,lda,_is_row_))), 0);    // A2
                VMOVU_PACKED(zmm19, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_ap(2+i,0+ii,lda,_is_row_))), 0);    // A3
                VMOVU_PACKED(zmm20, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_ap(3+i,0+ii,lda,_is_row_))), 0);    // A4
                    
                VMOVU_PACKED(zmm16, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(0+ii,0+j,ldb,_is_row_))), 0);
                VFMADD231_PACKED(zmm0, zmm16, zmm17);
                VFMADD231_PACKED(zmm1, zmm16, zmm18);
                VFMADD231_PACKED(zmm2, zmm16, zmm19);
                VFMADD231_PACKED(zmm3, zmm16, zmm20);

                VMOVU_PACKED(zmm16, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(0+ii,1+j,ldb,_is_row_))), 0);
                VFMADD231_PACKED(zmm4, zmm16, zmm17);
                VFMADD231_PACKED(zmm5, zmm16, zmm18);
                VFMADD231_PACKED(zmm6, zmm16, zmm19);
                VFMADD231_PACKED(zmm7, zmm16, zmm20);
            }
            // update the 4x4 B matrix
            // 1st
            VMOVU_PACKED(zmm20, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(0+i,0+j,ldb,_is_row_))), 0);     // B1
            VMOVU_PACKED(zmm21, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(0+i,1+j,ldb,_is_row_))), 0);     // B2
#ifdef _TCB_NOUNIT_DIAG_
            VMOVU_PACKED(zmm16, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_ap(0+i,0+i,lda,_is_row_))), 0);     // A1
            VDIV_PACKED(zmm16, zmm31, zmm16);                                                                               // A1 /= ONE
#endif
            VSUB_PACKED(zmm0, zmm20, zmm0);                                                                                 // T11 = B1-T11
#ifdef _TCB_NOUNIT_DIAG_
            VMUL_PACKED(zmm0, zmm0, zmm16);                                                                                 // T11 *= ONE/A1
#endif
            VMOVU_PACKED(zmm0, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(0+i,0+j,ldb,_is_row_))), 1);       // Store T11 -> B1

            VSUB_PACKED(zmm4, zmm21, zmm4);                                                                                 // T12 = B2-T12
#ifdef _TCB_NOUNIT_DIAG_
            VMUL_PACKED(zmm4, zmm4, zmm16);                                                                                 // T12 *= ONE/A1
#endif
            VMOVU_PACKED(zmm4, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(0+i,1+j,ldb,_is_row_))), 1);       // Store T12 -> B2

            // 2nd
#ifdef _TCB_NOUNIT_DIAG_
            VMOVU_PACKED(zmm16, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_ap(1+i,1+i,lda,_is_row_))), 0);     // A1
            VDIV_PACKED(zmm16, zmm31, zmm16);                                                                               // A1 /= ONE
#endif
            VMOVU_PACKED(zmm17, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_ap(1+i,0+i,lda,_is_row_))), 0);     // A2

            VMOVU_PACKED(zmm20, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(1+i,0+j,ldb,_is_row_))), 0);     // B1
            VMOVU_PACKED(zmm21, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(1+i,1+j,ldb,_is_row_))), 0);     // B2

            VFMADD231_PACKED(zmm1, zmm17, zmm0);                                                                            // T21 += A2*T11
            VSUB_PACKED(zmm1, zmm20, zmm1);                                                                                 // T21 = B1 - T21
#ifdef _TCB_NOUNIT_DIAG_
            VMUL_PACKED(zmm1, zmm1, zmm16);                                                                                 // T21 *= ONE/A1
#endif
            VMOVU_PACKED(zmm1, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(1+i,0+j,ldb,_is_row_))), 1);      // Store T21 -> B1

            VFMADD231_PACKED(zmm5, zmm17, zmm4);                                                                            // T22 += A2*T12
            VSUB_PACKED(zmm5, zmm21, zmm5);                                                                                 // T22 = B2 - T22
#ifdef _TCB_NOUNIT_DIAG_
            VMUL_PACKED(zmm5, zmm5, zmm16);                                                                                 // T22 *= ONE/A1
#endif
            VMOVU_PACKED(zmm5, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(1+i,1+j,ldb,_is_row_))), 1);      // Store T22 -> B2

            // 3rd
#ifdef _TCB_NOUNIT_DIAG_
            VMOVU_PACKED(zmm16, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_ap(2+i,2+i,lda,_is_row_))), 0);     // A1
            VDIV_PACKED(zmm16, zmm31, zmm16);                                                                               // A1 /= ONE
#endif
            VMOVU_PACKED(zmm17, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_ap(2+i,0+i,lda,_is_row_))), 0);     // A2
            VMOVU_PACKED(zmm18, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_ap(2+i,1+i,lda,_is_row_))), 0);     // A3

            VMOVU_PACKED(zmm20, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(2+i,0+j,ldb,_is_row_))), 0);     // B1
            VMOVU_PACKED(zmm21, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(2+i,1+j,ldb,_is_row_))), 0);     // B2

            VFMADD231_PACKED(zmm2, zmm17, zmm0);                                                                            // T31 += A2 * T11
            VFMADD231_PACKED(zmm2, zmm18, zmm1);                                                                            // T31 += A3 * T21
            VSUB_PACKED(zmm2, zmm20, zmm2);                                                                                 // T31 = B1 - T31
#ifdef _TCB_NOUNIT_DIAG_
            VMUL_PACKED(zmm2, zmm2, zmm16);                                                                                 // T31 *= ONE/A1
#endif
            VMOVU_PACKED(zmm2, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(2+i,0+j,ldb,_is_row_))), 1);      // Store T31 -> B1

            VFMADD231_PACKED(zmm6, zmm17, zmm4);                                                                            // T32 += A2 * T12
            VFMADD231_PACKED(zmm6, zmm18, zmm5);                                                                            // T32 += A3 * T22
            VSUB_PACKED(zmm6, zmm21, zmm6);                                                                                 // T32 = B2 - T32
#ifdef _TCB_NOUNIT_DIAG_
            VMUL_PACKED(zmm6, zmm6, zmm16);                                                                                 // T32 *= ONE/A1
#endif
            VMOVU_PACKED(zmm6, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(2+i,1+j,ldb,_is_row_))), 1);      // Store T32 -> B2

            // 4th
#ifdef _TCB_NOUNIT_DIAG_
            VMOVU_PACKED(zmm16, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_ap(3+i,3+i,lda,_is_row_))), 0);     // A1
            VDIV_PACKED(zmm16, zmm31, zmm16);                                                                               // A1 /= ONE
#endif
            VMOVU_PACKED(zmm17, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_ap(3+i,0+i,lda,_is_row_))), 0);     // A2
            VMOVU_PACKED(zmm18, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_ap(3+i,1+i,lda,_is_row_))), 0);     // A3
            VMOVU_PACKED(zmm19, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_ap(3+i,2+i,lda,_is_row_))), 0);     // A4
            
            VMOVU_PACKED(zmm20, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(3+i,0+j,ldb,_is_row_))), 0);     // B1
            VMOVU_PACKED(zmm21, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(3+i,1+j,ldb,_is_row_))), 0);     // B2

            VFMADD231_PACKED(zmm3, zmm17, zmm0);                                                                            // T41 += A2 * T11
            VFMADD231_PACKED(zmm3, zmm18, zmm1);                                                                            // T41 += A3 * T21
            VFMADD231_PACKED(zmm3, zmm19, zmm2);                                                                            // T41 += A4 * T31
            VSUB_PACKED(zmm3, zmm20, zmm3);                                                                                 // T41 = B1 - T41
#ifdef _TCB_NOUNIT_DIAG_
            VMUL_PACKED(zmm3, zmm3, zmm16);                                                                                 // T41 *= ONE/A1
#endif
            VMOVU_PACKED(zmm3, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(3+i,0+j,ldb,_is_row_))), 1);      // Store T41 -> B1

            SET_ZERO_PACKED(zmm0, zmm0, zmm0);                                                      // ZERO T11
            SET_ZERO_PACKED(zmm1, zmm1, zmm1);                                                      // ZERO T21
            SET_ZERO_PACKED(zmm2, zmm2, zmm2);                                                      // ZERO T31
            SET_ZERO_PACKED(zmm3, zmm3, zmm3);                                                      // ZERO T41

            VFMADD231_PACKED(zmm7, zmm17, zmm4);                                                                            // T42 += A2 * T12
            VFMADD231_PACKED(zmm7, zmm18, zmm5);                                                                            // T42 += A3 * T22
            VFMADD231_PACKED(zmm7, zmm19, zmm6);                                                                            // T42 += A4 * T32
            VSUB_PACKED(zmm7, zmm21, zmm7);                                                                                 // T42 = B2 - T42
#ifdef _TCB_NOUNIT_DIAG_
            VMUL_PACKED(zmm7, zmm7, zmm16);                                                                                 // T42 *= ONE/A1
#endif
            VMOVU_PACKED(zmm7, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(3+i,1+j,ldb,_is_row_))), 1);      // Store T42 -> B2

            SET_ZERO_PACKED(zmm4, zmm4, zmm4);                                                      // ZERO T12
            SET_ZERO_PACKED(zmm5, zmm5, zmm5);                                                      // ZERO T22
            SET_ZERO_PACKED(zmm6, zmm6, zmm6);                                                      // ZERO T32
            SET_ZERO_PACKED(zmm7, zmm7, zmm7);                                                      // ZERO T42
        } // m loop
        // m tails
        if (m_in & 2 && m_in & 1) {
            // gemm update
            for (ii=0; ii<i; ii++) {
                VMOVU_PACKED(zmm17, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_ap(0+i,0+ii,lda,_is_row_))), 0);    // A1
                VMOVU_PACKED(zmm18, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_ap(1+i,0+ii,lda,_is_row_))), 0);    // A2
                VMOVU_PACKED(zmm19, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_ap(2+i,0+ii,lda,_is_row_))), 0);    // A3
                    
                VMOVU_PACKED(zmm16, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(0+ii,0+j,ldb,_is_row_))), 0);
                VFMADD231_PACKED(zmm0, zmm16, zmm17);
                VFMADD231_PACKED(zmm1, zmm16, zmm18);
                VFMADD231_PACKED(zmm2, zmm16, zmm19);

                VMOVU_PACKED(zmm16, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(0+ii,1+j,ldb,_is_row_))), 0);
                VFMADD231_PACKED(zmm4, zmm16, zmm17);
                VFMADD231_PACKED(zmm5, zmm16, zmm18);
                VFMADD231_PACKED(zmm6, zmm16, zmm19);
            }
            // update the 4x4 B matrix
            // 1st
            VMOVU_PACKED(zmm20, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(0+i,0+j,ldb,_is_row_))), 0);     // B1
            VMOVU_PACKED(zmm21, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(0+i,1+j,ldb,_is_row_))), 0);     // B2
#ifdef _TCB_NOUNIT_DIAG_
            VMOVU_PACKED(zmm16, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_ap(0+i,0+i,lda,_is_row_))), 0);     // A1
#endif
            VDIV_PACKED(zmm16, zmm31, zmm16);                                                                               // A1 /= ONE

            VSUB_PACKED(zmm0, zmm20, zmm0);                                                                                 // T11 = B1-T11
#ifdef _TCB_NOUNIT_DIAG_
            VMUL_PACKED(zmm0, zmm0, zmm16);                                                                                 // T11 *= ONE/A1
#endif
            VMOVU_PACKED(zmm0, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(0+i,0+j,ldb,_is_row_))), 1);       // Store T11 -> B1

            VSUB_PACKED(zmm4, zmm21, zmm4);                                                                                 // T12 = B2-T12
#ifdef _TCB_NOUNIT_DIAG_
            VMUL_PACKED(zmm4, zmm4, zmm16);                                                                                 // T12 *= ONE/A1
#endif
            VMOVU_PACKED(zmm4, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(0+i,1+j,ldb,_is_row_))), 1);       // Store T12 -> B2

            // 2nd
#ifdef _TCB_NOUNIT_DIAG_
            VMOVU_PACKED(zmm16, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_ap(1+i,1+i,lda,_is_row_))), 0);     // A1
            VDIV_PACKED(zmm16, zmm31, zmm16);                                                                               // A1 /= ONE
#endif
            VMOVU_PACKED(zmm17, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_ap(1+i,0+i,lda,_is_row_))), 0);     // A2

            VMOVU_PACKED(zmm20, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(1+i,0+j,ldb,_is_row_))), 0);     // B1
            VMOVU_PACKED(zmm21, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(1+i,1+j,ldb,_is_row_))), 0);     // B2

            VFMADD231_PACKED(zmm1, zmm17, zmm0);                                                                            // T21 += A2*T11
            VSUB_PACKED(zmm1, zmm20, zmm1);                                                                                 // T21 = B1 - T21
#ifdef _TCB_NOUNIT_DIAG_
            VMUL_PACKED(zmm1, zmm1, zmm16);                                                                                 // T21 *= ONE/A1
#endif
            VMOVU_PACKED(zmm1, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(1+i,0+j,ldb,_is_row_))), 1);      // Store T21 -> B1

            VFMADD231_PACKED(zmm5, zmm17, zmm4);                                                                            // T22 += A2*T12
            VSUB_PACKED(zmm5, zmm21, zmm5);                                                                                 // T22 = B2 - T22
#ifdef _TCB_NOUNIT_DIAG_
            VMUL_PACKED(zmm5, zmm5, zmm16);                                                                                 // T22 *= ONE/A1
#endif
            VMOVU_PACKED(zmm5, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(1+i,1+j,ldb,_is_row_))), 1);      // Store T22 -> B2

            // 3rd
#ifdef _TCB_NOUNIT_DIAG_
            VMOVU_PACKED(zmm16, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_ap(2+i,2+i,lda,_is_row_))), 0);     // A1
            VDIV_PACKED(zmm16, zmm31, zmm16);                                                                               // A1 /= ONE
#endif
            VMOVU_PACKED(zmm17, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_ap(2+i,0+i,lda,_is_row_))), 0);     // A2
            VMOVU_PACKED(zmm18, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_ap(2+i,1+i,lda,_is_row_))), 0);     // A3

            VMOVU_PACKED(zmm20, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(2+i,0+j,ldb,_is_row_))), 0);     // B1
            VMOVU_PACKED(zmm21, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(2+i,1+j,ldb,_is_row_))), 0);     // B2

            VFMADD231_PACKED(zmm2, zmm17, zmm0);                                                                            // T31 += A2 * T11
            VFMADD231_PACKED(zmm2, zmm18, zmm1);                                                                            // T31 += A3 * T21
            VSUB_PACKED(zmm2, zmm20, zmm2);                                                                                 // T31 = B1 - T31
#ifdef _TCB_NOUNIT_DIAG_
            VMUL_PACKED(zmm2, zmm2, zmm16);                                                                                 // T31 *= ONE/A1
#endif
            VMOVU_PACKED(zmm2, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(2+i,0+j,ldb,_is_row_))), 1);      // Store T31 -> B1

            if (n_rem > 0) SET_ZERO_PACKED(zmm0, zmm0, zmm0);                                                                   // ZERO T11
            if (n_rem > 0) SET_ZERO_PACKED(zmm1, zmm1, zmm1);                                                                   // ZERO T21
            if (n_rem > 0) SET_ZERO_PACKED(zmm2, zmm2, zmm2);                                                                   // ZERO T31

            VFMADD231_PACKED(zmm6, zmm17, zmm4);                                                                            // T32 += A2 * T12
            VFMADD231_PACKED(zmm6, zmm18, zmm5);                                                                            // T32 += A3 * T22
            VSUB_PACKED(zmm6, zmm21, zmm6);                                                                                 // T32 = B2 - T32
#ifdef _TCB_NOUNIT_DIAG_
            VMUL_PACKED(zmm6, zmm6, zmm16);                                                                                 // T32 *= ONE/A1
#endif
            VMOVU_PACKED(zmm6, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(2+i,1+j,ldb,_is_row_))), 1);      // Store T32 -> B2

            if (n_rem > 0) SET_ZERO_PACKED(zmm4, zmm4, zmm4);                                                                   // ZERO T12
            if (n_rem > 0) SET_ZERO_PACKED(zmm5, zmm5, zmm5);                                                                   // ZERO T22
            if (n_rem > 0) SET_ZERO_PACKED(zmm6, zmm6, zmm6);                                                                   // ZERO T32
        }
        else if (m_in & 2) {
            // gemm update
            for (ii=0; ii<i; ii++) {
                VMOVU_PACKED(zmm17, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_ap(0+i,0+ii,lda,_is_row_))), 0);    // A1
                VMOVU_PACKED(zmm18, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_ap(1+i,0+ii,lda,_is_row_))), 0);    // A2
                    
                VMOVU_PACKED(zmm16, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(0+ii,0+j,ldb,_is_row_))), 0);
                VFMADD231_PACKED(zmm0, zmm16, zmm17);
                VFMADD231_PACKED(zmm1, zmm16, zmm18);

                VMOVU_PACKED(zmm16, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(0+ii,1+j,ldb,_is_row_))), 0);
                VFMADD231_PACKED(zmm4, zmm16, zmm17);
                VFMADD231_PACKED(zmm5, zmm16, zmm18);
            }
            // update the 4x4 B matrix
            // 1st
            VMOVU_PACKED(zmm20, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(0+i,0+j,ldb,_is_row_))), 0);     // B1
            VMOVU_PACKED(zmm21, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(0+i,1+j,ldb,_is_row_))), 0);     // B2
#ifdef _TCB_NOUNIT_DIAG_
            VMOVU_PACKED(zmm16, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_ap(0+i,0+i,lda,_is_row_))), 0);     // A1
            VDIV_PACKED(zmm16, zmm31, zmm16);                                                                               // A1 /= ONE
#endif
            VSUB_PACKED(zmm0, zmm20, zmm0);                                                                                 // T11 = B1-T11
#ifdef _TCB_NOUNIT_DIAG_
            VMUL_PACKED(zmm0, zmm0, zmm16);                                                                                 // T11 *= ONE/A1
#endif
            VMOVU_PACKED(zmm0, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(0+i,0+j,ldb,_is_row_))), 1);       // Store T11 -> B1

            VSUB_PACKED(zmm4, zmm21, zmm4);                                                                                 // T12 = B2-T12
#ifdef _TCB_NOUNIT_DIAG_
            VMUL_PACKED(zmm4, zmm4, zmm16);                                                                                 // T12 *= ONE/A1
#endif
            VMOVU_PACKED(zmm4, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(0+i,1+j,ldb,_is_row_))), 1);       // Store T12 -> B2

            // 2nd
#ifdef _TCB_NOUNIT_DIAG_
            VMOVU_PACKED(zmm16, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_ap(1+i,1+i,lda,_is_row_))), 0);     // A1
            VDIV_PACKED(zmm16, zmm31, zmm16);                                                                               // A1 /= ONE
#endif
            VMOVU_PACKED(zmm17, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_ap(1+i,0+i,lda,_is_row_))), 0);     // A2

            VMOVU_PACKED(zmm20, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(1+i,0+j,ldb,_is_row_))), 0);     // B1
            VMOVU_PACKED(zmm21, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(1+i,1+j,ldb,_is_row_))), 0);     // B2

            VFMADD231_PACKED(zmm1, zmm17, zmm0);                                                                            // T21 += A2*T11
            VSUB_PACKED(zmm1, zmm20, zmm1);                                                                                 // T21 = B1 - T21
#ifdef _TCB_NOUNIT_DIAG_
            VMUL_PACKED(zmm1, zmm1, zmm16);                                                                                 // T21 *= ONE/A1
#endif
            VMOVU_PACKED(zmm1, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(1+i,0+j,ldb,_is_row_))), 1);      // Store T21 -> B1

            if (n_rem > 0) SET_ZERO_PACKED(zmm0, zmm0, zmm0);                                                                   // ZERO T11
            if (n_rem > 0) SET_ZERO_PACKED(zmm1, zmm1, zmm1);                                                                   // ZERO T21

            VFMADD231_PACKED(zmm5, zmm17, zmm4);                                                                            // T22 += A2*T12
            VSUB_PACKED(zmm5, zmm21, zmm5);                                                                                 // T22 = B2 - T22
#ifdef _TCB_NOUNIT_DIAG_
            VMUL_PACKED(zmm5, zmm5, zmm16);                                                                                 // T22 *= ONE/A1
#endif
            VMOVU_PACKED(zmm5, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(1+i,1+j,ldb,_is_row_))), 1);      // Store T22 -> B2

            if (n_rem > 0) SET_ZERO_PACKED(zmm4, zmm4, zmm4);                                                                   // ZERO T12
            if (n_rem > 0) SET_ZERO_PACKED(zmm5, zmm5, zmm5);                                                                   // ZERO T22
        }
        else if (m_in & 1) {
           // gemm update
            for (ii=0; ii<i; ii++) {
                VMOVU_PACKED(zmm17, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_ap(0+i,0+ii,lda,_is_row_))), 0);    // A1
                    
                VMOVU_PACKED(zmm16, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(0+ii,0+j,ldb,_is_row_))), 0);
                VFMADD231_PACKED(zmm0, zmm16, zmm17);

                VMOVU_PACKED(zmm16, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(0+ii,1+j,ldb,_is_row_))), 0);
                VFMADD231_PACKED(zmm4, zmm16, zmm17);
            }
            // update the 4x4 B matrix
            // 1st
            VMOVU_PACKED(zmm20, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(0+i,0+j,ldb,_is_row_))), 0);     // B1
            VMOVU_PACKED(zmm21, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(0+i,1+j,ldb,_is_row_))), 0);     // B2
#ifdef _TCB_NOUNIT_DIAG_
            VMOVU_PACKED(zmm16, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_ap(0+i,0+i,lda,_is_row_))), 0);     // A1
            VDIV_PACKED(zmm16, zmm31, zmm16);                                                                               // A1 /= ONE
#endif
            VSUB_PACKED(zmm0, zmm20, zmm0);                                                                                 // T11 = B1-T11
#ifdef _TCB_NOUNIT_DIAG_
            VMUL_PACKED(zmm0, zmm0, zmm16);                                                                                 // T11 *= ONE/A1
#endif
            VMOVU_PACKED(zmm0, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(0+i,0+j,ldb,_is_row_))), 1);      // Store T11 -> B1

            if (n_rem > 0) SET_ZERO_PACKED(zmm0, zmm0, zmm0);                                                                   // ZERO T11

            VSUB_PACKED(zmm4, zmm21, zmm4);                                                                                 // T12 = B2-T12
#ifdef _TCB_NOUNIT_DIAG_
            VMUL_PACKED(zmm4, zmm4, zmm16);                                                                                 // T12 *= ONE/A1
#endif
            VMOVU_PACKED(zmm4, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(0+i,1+j,ldb,_is_row_))), 1);      // Store T12 -> B2

            if (n_rem > 0) SET_ZERO_PACKED(zmm4, zmm4, zmm4);                                                                   // ZERO T12
        }
    }
    else if (n_in & 1) {
        m_rem = m_in;
        for (i=0; i<(m_in/M_UNROLL_AVX512)*M_UNROLL_AVX512; i+=M_UNROLL_AVX512) {
            m_rem -= M_UNROLL_AVX512;
            // gemm update
            for (ii=0; ii<i; ii++) {
                VMOVU_PACKED(zmm17, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_ap(0+i,0+ii,lda,_is_row_))), 0);    // A1
                VMOVU_PACKED(zmm18, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_ap(1+i,0+ii,lda,_is_row_))), 0);    // A2
                VMOVU_PACKED(zmm19, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_ap(2+i,0+ii,lda,_is_row_))), 0);    // A3
                VMOVU_PACKED(zmm20, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_ap(3+i,0+ii,lda,_is_row_))), 0);    // A4
                    
                VMOVU_PACKED(zmm16, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(0+ii,0+j,ldb,_is_row_))), 0);
                VFMADD231_PACKED(zmm0, zmm16, zmm17);
                VFMADD231_PACKED(zmm1, zmm16, zmm18);
                VFMADD231_PACKED(zmm2, zmm16, zmm19);
                VFMADD231_PACKED(zmm3, zmm16, zmm20);
            }
            // update the 4x4 B matrix
            // 1st
            VMOVU_PACKED(zmm20, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(0+i,0+j,ldb,_is_row_))), 0);     // B1
#ifdef _TCB_NOUNIT_DIAG_
            VMOVU_PACKED(zmm16, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_ap(0+i,0+i,lda,_is_row_))), 0);     // A1
            VDIV_PACKED(zmm16, zmm31, zmm16);                                                                               // A1 /= ONE
#endif
            VSUB_PACKED(zmm0, zmm20, zmm0);                                                                                 // T11 = B1-T11
#ifdef _TCB_NOUNIT_DIAG_
            VMUL_PACKED(zmm0, zmm0, zmm16);                                                                                 // T11 *= ONE/A1
#endif
            VMOVU_PACKED(zmm0, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(0+i,0+j,ldb,_is_row_))), 1);       // Store T11 -> B1

            // 2nd
#ifdef _TCB_NOUNIT_DIAG_
            VMOVU_PACKED(zmm16, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_ap(1+i,1+i,lda,_is_row_))), 0);     // A1
            VDIV_PACKED(zmm16, zmm31, zmm16);                                                                               // A1 /= ONE
#endif
            VMOVU_PACKED(zmm17, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_ap(1+i,0+i,lda,_is_row_))), 0);     // A2

            VMOVU_PACKED(zmm20, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(1+i,0+j,ldb,_is_row_))), 0);     // B1

            VFMADD231_PACKED(zmm1, zmm17, zmm0);                                                                            // T21 += A2*T11
            VSUB_PACKED(zmm1, zmm20, zmm1);                                                                                 // T21 = B1 - T21
#ifdef _TCB_NOUNIT_DIAG_
            VMUL_PACKED(zmm1, zmm1, zmm16);                                                                                 // T21 *= ONE/A1
#endif
            VMOVU_PACKED(zmm1, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(1+i,0+j,ldb,_is_row_))), 1);      // Store T21 -> B1

            // 3rd
#ifdef _TCB_NOUNIT_DIAG_
            VMOVU_PACKED(zmm16, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_ap(2+i,2+i,lda,_is_row_))), 0);     // A1
            VDIV_PACKED(zmm16, zmm31, zmm16);                                                                               // A1 /= ONE
#endif
            VMOVU_PACKED(zmm17, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_ap(2+i,0+i,lda,_is_row_))), 0);     // A2
            VMOVU_PACKED(zmm18, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_ap(2+i,1+i,lda,_is_row_))), 0);     // A3

            VMOVU_PACKED(zmm20, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(2+i,0+j,ldb,_is_row_))), 0);     // B1

            VFMADD231_PACKED(zmm2, zmm17, zmm0);                                                                            // T31 += A2 * T11
            VFMADD231_PACKED(zmm2, zmm18, zmm1);                                                                            // T31 += A3 * T21
            VSUB_PACKED(zmm2, zmm20, zmm2);                                                                                 // T31 = B1 - T31
#ifdef _TCB_NOUNIT_DIAG_
            VMUL_PACKED(zmm2, zmm2, zmm16);                                                                                 // T31 *= ONE/A1
#endif
            VMOVU_PACKED(zmm2, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(2+i,0+j,ldb,_is_row_))), 1);      // Store T31 -> B1

            // 4th
#ifdef _TCB_NOUNIT_DIAG_
            VMOVU_PACKED(zmm16, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_ap(3+i,3+i,lda,_is_row_))), 0);     // A1
            VDIV_PACKED(zmm16, zmm31, zmm16);                                                                               // A1 /= ONE
#endif
            VMOVU_PACKED(zmm17, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_ap(3+i,0+i,lda,_is_row_))), 0);     // A2
            VMOVU_PACKED(zmm18, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_ap(3+i,1+i,lda,_is_row_))), 0);     // A3
            VMOVU_PACKED(zmm19, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_ap(3+i,2+i,lda,_is_row_))), 0);     // A4
            
            VMOVU_PACKED(zmm20, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(3+i,0+j,ldb,_is_row_))), 0);     // B1

            VFMADD231_PACKED(zmm3, zmm17, zmm0);                                                                            // T41 += A2 * T11
            VFMADD231_PACKED(zmm3, zmm18, zmm1);                                                                            // T41 += A3 * T21
            VFMADD231_PACKED(zmm3, zmm19, zmm2);                                                                            // T41 += A4 * T31
            VSUB_PACKED(zmm3, zmm20, zmm3);                                                                                 // T41 = B1 - T41
#ifdef _TCB_NOUNIT_DIAG_
            VMUL_PACKED(zmm3, zmm3, zmm16);                                                                                 // T41 *= ONE/A1
#endif
            VMOVU_PACKED(zmm3, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(3+i,0+j,ldb,_is_row_))), 1);      // Store T41 -> B1

            SET_ZERO_PACKED(zmm0, zmm0, zmm0);                                                      // ZERO T11
            SET_ZERO_PACKED(zmm1, zmm1, zmm1);                                                      // ZERO T21
            SET_ZERO_PACKED(zmm2, zmm2, zmm2);                                                      // ZERO T31
            SET_ZERO_PACKED(zmm3, zmm3, zmm3);                                                      // ZERO T41

        } // m loop
        // m tails
        if (m_in & 2 && m_in & 1) {
            // gemm update
            for (ii=0; ii<i; ii++) {
                VMOVU_PACKED(zmm17, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_ap(0+i,0+ii,lda,_is_row_))), 0);    // A1
                VMOVU_PACKED(zmm18, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_ap(1+i,0+ii,lda,_is_row_))), 0);    // A2
                VMOVU_PACKED(zmm19, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_ap(2+i,0+ii,lda,_is_row_))), 0);    // A3
                    
                VMOVU_PACKED(zmm16, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(0+ii,0+j,ldb,_is_row_))), 0);
                VFMADD231_PACKED(zmm0, zmm16, zmm17);
                VFMADD231_PACKED(zmm1, zmm16, zmm18);
                VFMADD231_PACKED(zmm2, zmm16, zmm19);
            }
            // update the 4x4 B matrix
            // 1st
            VMOVU_PACKED(zmm20, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(0+i,0+j,ldb,_is_row_))), 0);     // B1
#ifdef _TCB_NOUNIT_DIAG_
            VMOVU_PACKED(zmm16, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_ap(0+i,0+i,lda,_is_row_))), 0);     // A1
            VDIV_PACKED(zmm16, zmm31, zmm16);                                                                               // A1 /= ONE
#endif
            VSUB_PACKED(zmm0, zmm20, zmm0);                                                                                 // T11 = B1-T11
#ifdef _TCB_NOUNIT_DIAG_
            VMUL_PACKED(zmm0, zmm0, zmm16);                                                                                 // T11 *= ONE/A1
#endif
            VMOVU_PACKED(zmm0, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(0+i,0+j,ldb,_is_row_))), 1);       // Store T11 -> B1

            // 2nd
#ifdef _TCB_NOUNIT_DIAG_
            VMOVU_PACKED(zmm16, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_ap(1+i,1+i,lda,_is_row_))), 0);     // A1
            VDIV_PACKED(zmm16, zmm31, zmm16);                                                                               // A1 /= ONE
#endif
            VMOVU_PACKED(zmm17, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_ap(1+i,0+i,lda,_is_row_))), 0);     // A2

            VMOVU_PACKED(zmm20, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(1+i,0+j,ldb,_is_row_))), 0);     // B1

            VFMADD231_PACKED(zmm1, zmm17, zmm0);                                                                            // T21 += A2*T11
            VSUB_PACKED(zmm1, zmm20, zmm1);                                                                                 // T21 = B1 - T21
#ifdef _TCB_NOUNIT_DIAG_
            VMUL_PACKED(zmm1, zmm1, zmm16);                                                                                 // T21 *= ONE/A1
#endif
            VMOVU_PACKED(zmm1, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(1+i,0+j,ldb,_is_row_))), 1);      // Store T21 -> B1

            // 3rd
#ifdef _TCB_NOUNIT_DIAG_
            VMOVU_PACKED(zmm16, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_ap(2+i,2+i,lda,_is_row_))), 0);     // A1
            VDIV_PACKED(zmm16, zmm31, zmm16);                                                                               // A1 /= ONE
#endif
            VMOVU_PACKED(zmm17, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_ap(2+i,0+i,lda,_is_row_))), 0);     // A2
            VMOVU_PACKED(zmm18, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_ap(2+i,1+i,lda,_is_row_))), 0);     // A3

            VMOVU_PACKED(zmm20, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(2+i,0+j,ldb,_is_row_))), 0);     // B1

            VFMADD231_PACKED(zmm2, zmm17, zmm0);                                                                            // T31 += A2 * T11
            VFMADD231_PACKED(zmm2, zmm18, zmm1);                                                                            // T31 += A3 * T21
            VSUB_PACKED(zmm2, zmm20, zmm2);                                                                                 // T31 = B1 - T31
#ifdef _TCB_NOUNIT_DIAG_
            VMUL_PACKED(zmm2, zmm2, zmm16);                                                                                 // T31 *= ONE/A1
#endif
            VMOVU_PACKED(zmm2, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(2+i,0+j,ldb,_is_row_))), 1);      // Store T31 -> B1

            if (n_rem > 0) SET_ZERO_PACKED(zmm0, zmm0, zmm0);                                                                   // ZERO T11
            if (n_rem > 0) SET_ZERO_PACKED(zmm1, zmm1, zmm1);                                                                   // ZERO T21
            if (n_rem > 0) SET_ZERO_PACKED(zmm2, zmm2, zmm2);                                                                   // ZERO T31

        }
        else if (m_in & 2) {
            // gemm update
            for (ii=0; ii<i; ii++) {
                VMOVU_PACKED(zmm17, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_ap(0+i,0+ii,lda,_is_row_))), 0);    // A1
                VMOVU_PACKED(zmm18, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_ap(1+i,0+ii,lda,_is_row_))), 0);    // A2
                    
                VMOVU_PACKED(zmm16, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(0+ii,0+j,ldb,_is_row_))), 0);
                VFMADD231_PACKED(zmm0, zmm16, zmm17);
                VFMADD231_PACKED(zmm1, zmm16, zmm18);

            }
            // update the 4x4 B matrix
            // 1st
            VMOVU_PACKED(zmm20, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(0+i,0+j,ldb,_is_row_))), 0);     // B1
#ifdef _TCB_NOUNIT_DIAG_
            VMOVU_PACKED(zmm16, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_ap(0+i,0+i,lda,_is_row_))), 0);     // A1
            VDIV_PACKED(zmm16, zmm31, zmm16);                                                                               // A1 /= ONE
#endif
            VSUB_PACKED(zmm0, zmm20, zmm0);                                                                                 // T11 = B1-T11
#ifdef _TCB_NOUNIT_DIAG_
            VMUL_PACKED(zmm0, zmm0, zmm16);                                                                                 // T11 *= ONE/A1
#endif
            VMOVU_PACKED(zmm0, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(0+i,0+j,ldb,_is_row_))), 1);       // Store T11 -> B1

            // 2nd
#ifdef _TCB_NOUNIT_DIAG_
            VMOVU_PACKED(zmm16, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_ap(1+i,1+i,lda,_is_row_))), 0);     // A1
            VDIV_PACKED(zmm16, zmm31, zmm16);                                                                               // A1 /= ONE
#endif
            VMOVU_PACKED(zmm17, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_ap(1+i,0+i,lda,_is_row_))), 0);     // A2

            VMOVU_PACKED(zmm20, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(1+i,0+j,ldb,_is_row_))), 0);     // B1

            VFMADD231_PACKED(zmm1, zmm17, zmm0);                                                                            // T21 += A2*T11
            VSUB_PACKED(zmm1, zmm20, zmm1);                                                                                 // T21 = B1 - T21
#ifdef _TCB_NOUNIT_DIAG_
            VMUL_PACKED(zmm1, zmm1, zmm16);                                                                                 // T21 *= ONE/A1
#endif
            VMOVU_PACKED(zmm1, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(1+i,0+j,ldb,_is_row_))), 1);      // Store T21 -> B1

            if (n_rem > 0) SET_ZERO_PACKED(zmm0, zmm0, zmm0);                                                                   // ZERO T11
            if (n_rem > 0) SET_ZERO_PACKED(zmm1, zmm1, zmm1);                                                                   // ZERO T21
        }
        else if (m_in & 1) {
           // gemm update
            for (ii=0; ii<i; ii++) {
                VMOVU_PACKED(zmm17, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_ap(0+i,0+ii,lda,_is_row_))), 0);    // A1
                    
                VMOVU_PACKED(zmm16, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(0+ii,0+j,ldb,_is_row_))), 0);
                VFMADD231_PACKED(zmm0, zmm16, zmm17);

            }
            // update the 4x4 B matrix
            // 1st
            VMOVU_PACKED(zmm20, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(0+i,0+j,ldb,_is_row_))), 0);     // B1
#ifdef _TCB_NOUNIT_DIAG_
            VMOVU_PACKED(zmm16, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_ap(0+i,0+i,lda,_is_row_))), 0);     // A1
            VDIV_PACKED(zmm16, zmm31, zmm16);                                                                               // A1 /= ONE
#endif
            VSUB_PACKED(zmm0, zmm20, zmm0);                                                                                 // T11 = B1-T11
#ifdef _TCB_NOUNIT_DIAG_
            VMUL_PACKED(zmm0, zmm0, zmm16);                                                                                 // T11 *= ONE/A1
#endif
            VMOVU_PACKED(zmm0, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(trsm_ll_bp(0+i,0+j,ldb,_is_row_))), 1);      // Store T11 -> B1

            if (n_rem > 0) SET_ZERO_PACKED(zmm0, zmm0, zmm0);                                                                   // ZERO T11

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
#undef P_UNROLL_AVX512
