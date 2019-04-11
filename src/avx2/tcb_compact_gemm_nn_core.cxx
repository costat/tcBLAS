/*****************************************************************************/
/* Author: Timothy Costa
******************************************************************************/

#if defined(SINGLE)
#define tcb_ftype float
#define VMOVU_PACKED(reg, mat_ptr, mat_offset, load_store) do { \
    if (load_store) vmovups( yword[mat_ptr + mat_offset], reg ); \
    else vmovups( reg, yword[mat_ptr + mat_offset] ); \
} while(0)
#define LOAD_OR_ZERO_PACKED(reg, mat_ptr, mat_offset) do { \
    if (beta) { \
        VMOVU_PACKED(reg, mat_ptr, mat_offset, 0); \
    } \
    else { \
        vxorps(reg, reg, reg); \
    } \
} while (0)
#define VFMA_OR_VFNMA_PACKED vfma_or_vfnma_ps
#define P_UNROLL_AVX2 P_UNROLL_AVX2_F32
#elif defined(DOUBLE)
#define tcb_ftype double 
#define VMOVU_PACKED(reg, mat_ptr, mat_offset, load_store) do { \
    if (load_store) vmovupd( yword[mat_ptr + mat_offset], reg ); \
    else vmovupd( reg, yword[mat_ptr + mat_offset] ); \
} while(0)
#define LOAD_OR_ZERO_PACKED(reg, mat_ptr, mat_offset) do { \
    if (beta) { \
        VMOVU_PACKED(reg, mat_ptr, mat_offset, 0); \
    } \
    else { \
        vxorpd(reg, reg, reg); \
    } \
} while (0)
#define VFMA_OR_VFNMA_PACKED vfma_or_vfnma_pd
#define P_UNROLL_AVX2 P_UNROLL_AVX2_F64
#endif

F_NAME::F_NAME( int layout, int m, int n, int k, tcb_ftype alpha, int lda, int ldb, tcb_ftype beta, int ldc )
        : Xbyak::CodeGenerator( 80 * Xbyak::DEFAULT_MAX_CODE_SIZE, nullptr )
{

    auto a_ptr = rdi;
    auto b_ptr = rsi;
    auto c_ptr = rdx;
    int m_in, n_in, lda_in, ldb_in;

    if (layout == 101) {
        b_ptr = rdi;
        a_ptr = rsi;
        m_in = n;
        n_in = m;
        lda_in = ldb;
        ldb_in = lda;
    }
    else {
        m_in = m;
        n_in = n;
        lda_in = lda;
        ldb_in = ldb;
    }
 
    bool _avx512f_ = 0;

    int i, j;

    assert(beta == 0.0 || beta == 1.0);
    assert(alpha == 1.0 || alpha == -1.0);

    for (j = 0; j < (n_in/N_UNROLL_AVX2)*N_UNROLL_AVX2; j += N_UNROLL_AVX2) {
        for (i = 0; i < (m_in/M_UNROLL_AVX2)*M_UNROLL_AVX2; i+= M_UNROLL_AVX2) {
            // load C
            LOAD_OR_ZERO_PACKED(ymm0, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(0+i+ldc*(0+j))));
            LOAD_OR_ZERO_PACKED(ymm1, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(0+i+ldc*(1+j))));
            LOAD_OR_ZERO_PACKED(ymm2, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(0+i+ldc*(2+j))));
            LOAD_OR_ZERO_PACKED(ymm3, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(0+i+ldc*(3+j))));
            LOAD_OR_ZERO_PACKED(ymm4, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(1+i+ldc*(0+j))));
            LOAD_OR_ZERO_PACKED(ymm5, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(1+i+ldc*(1+j))));
            LOAD_OR_ZERO_PACKED(ymm6, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(1+i+ldc*(2+j))));
            LOAD_OR_ZERO_PACKED(ymm7, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(1+i+ldc*(3+j))));
            for (int kk = 0; kk < (k/K_UNROLL_AVX2)*K_UNROLL_AVX2; kk += K_UNROLL_AVX2) {
                // load A1
                VMOVU_PACKED(ymm8, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(0+i+lda_in*(0+kk))), 0);
                // load B1
                VMOVU_PACKED(ymm9, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(0+kk+ldb_in*(0+j))), 0);
                // load B2
                VMOVU_PACKED(ymm10, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(0+kk+ldb_in*(1+j))), 0);
                VFMA_OR_VFNMA_PACKED(ymm0, ymm8, ymm9); // C1 += A1 * B1
                VFMA_OR_VFNMA_PACKED(ymm1, ymm8, ymm10); // C2 += A1 * B2

                // load A2
                VMOVU_PACKED(ymm11, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(1+i+lda_in*(0+kk))), 0);
                VFMA_OR_VFNMA_PACKED(ymm4, ymm11, ymm9); // C3 += A2 * B1
                VFMA_OR_VFNMA_PACKED(ymm5, ymm11, ymm10); // C4 += A2 * B2

                // load B3
                VMOVU_PACKED(ymm12, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(0+kk+ldb_in*(2+j))), 0);
                // load B4
                VMOVU_PACKED(ymm13, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(0+kk+ldb_in*(3+j))), 0);
                VFMA_OR_VFNMA_PACKED(ymm2, ymm8, ymm12);
                VFMA_OR_VFNMA_PACKED(ymm3, ymm8, ymm13);

                VFMA_OR_VFNMA_PACKED(ymm6, ymm11, ymm12);
                VFMA_OR_VFNMA_PACKED(ymm7, ymm11, ymm13);
            }
            // store C
            VMOVU_PACKED(ymm0, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(0+i+ldc*(0+j))), 1);
            VMOVU_PACKED(ymm1, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(0+i+ldc*(1+j))), 1);
            VMOVU_PACKED(ymm2, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(0+i+ldc*(2+j))), 1);
            VMOVU_PACKED(ymm3, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(0+i+ldc*(3+j))), 1);
            VMOVU_PACKED(ymm4, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(1+i+ldc*(0+j))), 1);
            VMOVU_PACKED(ymm5, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(1+i+ldc*(1+j))), 1);
            VMOVU_PACKED(ymm6, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(1+i+ldc*(2+j))), 1);
            VMOVU_PACKED(ymm7, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(1+i+ldc*(3+j))), 1);
        }
        // m tail
        if (m_in & 1) {
            // load C
            LOAD_OR_ZERO_PACKED(ymm0, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(0+i+ldc*(0+j))));
            LOAD_OR_ZERO_PACKED(ymm1, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(0+i+ldc*(1+j))));
            LOAD_OR_ZERO_PACKED(ymm2, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(0+i+ldc*(2+j))));
            LOAD_OR_ZERO_PACKED(ymm3, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(0+i+ldc*(3+j))));
            for (int kk = 0; kk < (k/K_UNROLL_AVX2)*K_UNROLL_AVX2; kk += K_UNROLL_AVX2) {
                // load A1
                VMOVU_PACKED(ymm8, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(0+i+lda_in*(0+kk))), 0);
                // load B1
                VMOVU_PACKED(ymm9, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(0+kk+ldb_in*(0+j))), 0);
                // load B2
                VMOVU_PACKED(ymm10, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(0+kk+ldb_in*(1+j))), 0);
                VFMA_OR_VFNMA_PACKED(ymm0, ymm8, ymm9); // C1 += A1 * B1
                VFMA_OR_VFNMA_PACKED(ymm1, ymm8, ymm10); // C2 += A1 * B2

                // load B3
                VMOVU_PACKED(ymm12, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(0+kk+ldb_in*(2+j))), 0);
                // load B4
                VMOVU_PACKED(ymm13, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(0+kk+ldb_in*(3+j))), 0);
                VFMA_OR_VFNMA_PACKED(ymm2, ymm8, ymm12);
                VFMA_OR_VFNMA_PACKED(ymm3, ymm8, ymm13);
            }
            // store C
            VMOVU_PACKED(ymm0, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(0+i+ldc*(0+j))), 1);
            VMOVU_PACKED(ymm1, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(0+i+ldc*(1+j))), 1);
            VMOVU_PACKED(ymm2, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(0+i+ldc*(2+j))), 1);
            VMOVU_PACKED(ymm3, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(0+i+ldc*(3+j))), 1);
        }
    } // j/n loop
    // j tail: j remainder of 1
    if (n_in & 2 && n_in & 1) {
       for (i = 0; i < (m_in/M_UNROLL_AVX2)*M_UNROLL_AVX2; i+= M_UNROLL_AVX2) {
            // load C
            LOAD_OR_ZERO_PACKED(ymm0, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(0+i+ldc*(0+j))));
            LOAD_OR_ZERO_PACKED(ymm1, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(0+i+ldc*(1+j))));
            LOAD_OR_ZERO_PACKED(ymm2, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(0+i+ldc*(2+j))));
            LOAD_OR_ZERO_PACKED(ymm4, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(1+i+ldc*(0+j))));
            LOAD_OR_ZERO_PACKED(ymm5, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(1+i+ldc*(1+j))));
            LOAD_OR_ZERO_PACKED(ymm6, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(1+i+ldc*(2+j))));
            for (int kk = 0; kk < (k/K_UNROLL_AVX2)*K_UNROLL_AVX2; kk += K_UNROLL_AVX2) {
                // load A1
                VMOVU_PACKED(ymm8, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(0+i+lda_in*(0+kk))), 0);
                // load B1
                VMOVU_PACKED(ymm9, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(0+kk+ldb_in*(0+j))), 0);
                // load B2
                VMOVU_PACKED(ymm10, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(0+kk+ldb_in*(1+j))), 0);
                VFMA_OR_VFNMA_PACKED(ymm0, ymm8, ymm9); // C1 += A1 * B1
                VFMA_OR_VFNMA_PACKED(ymm1, ymm8, ymm10); // C2 += A1 * B2

                // load A2
                VMOVU_PACKED(ymm11, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(1+i+lda_in*(0+kk))), 0);
                VFMA_OR_VFNMA_PACKED(ymm4, ymm11, ymm9); // C3 += A2 * B1
                VFMA_OR_VFNMA_PACKED(ymm5, ymm11, ymm10); // C4 += A2 * B2

                // load B3
                VMOVU_PACKED(ymm12, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(0+kk+ldb_in*(2+j))), 0);
                // load B4
                VFMA_OR_VFNMA_PACKED(ymm2, ymm8, ym12);

                VFMA_OR_VFNMA_PACKED(ymm6, ymm11, ymm12);
            }
            // store C
            VMOVU_PACKED(ymm0, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(0+i+ldc*(0+j))), 1);
            VMOVU_PACKED(ymm1, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(0+i+ldc*(1+j))), 1);
            VMOVU_PACKED(ymm2, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(0+i+ldc*(2+j))), 1);
            VMOVU_PACKED(ymm4, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(1+i+ldc*(0+j))), 1);
            VMOVU_PACKED(ymm5, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(1+i+ldc*(1+j))), 1);
            VMOVU_PACKED(ymm6, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(1+i+ldc*(2+j))), 1);
        }
        // m tail
        if (m_in & 1) {
            // load C
            LOAD_OR_ZERO_PACKED(ymm0, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(0+i+ldc*(0+j))));
            LOAD_OR_ZERO_PACKED(ymm1, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(0+i+ldc*(1+j))));
            LOAD_OR_ZERO_PACKED(ymm2, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(0+i+ldc*(2+j))));
            for (int kk = 0; kk < (k/K_UNROLL_AVX2)*K_UNROLL_AVX2; kk += K_UNROLL_AVX2) {
                // load A1
                VMOVU_PACKED(ymm8, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(0+i+lda_in*(0+kk))), 0);
                // load B1
                VMOVU_PACKED(ymm9, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(0+kk+ldb_in*(0+j))), 0);
                // load B2
                VMOVU_PACKED(ymm10, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(0+kk+ldb_in*(1+j))), 0);
                VFMA_OR_VFNMA_PACKED(ymm0, ymm8, ymm9); // C1 += A1 * B1
                VFMA_OR_VFNMA_PACKED(ymm1, ymm8, ymm10); // C2 += A1 * B2

                // load B3
                VMOVU_PACKED(ymm12, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(0+kk+ldb_in*(2+j))), 0);
                // load B4
                VFMA_OR_VFNMA_PACKED(ymm2, ymm8, ymm12);
            }
            // store C
            VMOVU_PACKED(ymm0, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(0+i+ldc*(0+j))), 1);
            VMOVU_PACKED(ymm1, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(0+i+ldc*(1+j))), 1);
            VMOVU_PACKED(ymm2, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(0+i+ldc*(2+j))), 1);
        }
    }
    else if (n_in & 2) {
        for (i = 0; i < (m_in/M_UNROLL_AVX2)*M_UNROLL_AVX2; i+= M_UNROLL_AVX2) {
            // load C
            LOAD_OR_ZERO_PACKED(ymm0, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(0+i+ldc*(0+j))));
            LOAD_OR_ZERO_PACKED(ymm1, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(0+i+ldc*(1+j))));
            LOAD_OR_ZERO_PACKED(ymm4, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(1+i+ldc*(0+j))));
            LOAD_OR_ZERO_PACKED(ymm5, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(1+i+ldc*(1+j))));
            for (int kk = 0; kk < (k/K_UNROLL_AVX2)*K_UNROLL_AVX2; kk += K_UNROLL_AVX2) {
                // load A1
                VMOVU_PACKED(ymm8, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(0+i+lda_in*(0+kk))), 0);
                // load B1
                VMOVU_PACKED(ymm9, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(0+kk+ldb_in*(0+j))), 0);
                // load B2
                VMOVU_PACKED(ymm10, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(0+kk+ldb_in*(1+j))), 0);
                VFMA_OR_VFNMA_PACKED(ymm0, ymm8, ymm9); // C1 += A1 * B1
                VFMA_OR_VFNMA_PACKED(ymm1, ymm8, ymm10); // C2 += A1 * B2

                // load A2
                VMOVU_PACKED(ymm11, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(1+i+lda_in*(0+kk))), 0);
                VFMA_OR_VFNMA_PACKED(ymm4, ymm11, ymm9); // C3 += A2 * B1
                VFMA_OR_VFNMA_PACKED(ymm5, ymm11, ymm10); // C4 += A2 * B2
            }
            // store C
            VMOVU_PACKED(ymm0, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(0+i+ldc*(0+j))), 1);
            VMOVU_PACKED(ymm1, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(0+i+ldc*(1+j))), 1);
            VMOVU_PACKED(ymm4, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(1+i+ldc*(0+j))), 1);
            VMOVU_PACKED(ymm5, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(1+i+ldc*(1+j))), 1);
        }
        // m tail
        if (m_in & 1) {
            // load C
            LOAD_OR_ZERO_PACKED(ymm0, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(0+i+ldc*(0+j))));
            LOAD_OR_ZERO_PACKED(ymm1, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(0+i+ldc*(1+j))));
            for (int kk = 0; kk < (k/K_UNROLL_AVX2)*K_UNROLL_AVX2; kk += K_UNROLL_AVX2) {
                // load A1
                VMOVU_PACKED(ymm8, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(0+i+lda_in*(0+kk))), 0);
                // load B1
                VMOVU_PACKED(ymm9, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(0+kk+ldb_in*(0+j))), 0);
                // load B2
                VMOVU_PACKED(ymm10, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(0+kk+ldb_in*(1+j))), 0);
                VFMA_OR_VFNMA_PACKED(ymm0, ymm8, ymm9); // C1 += A1 * B1
                VFMA_OR_VFNMA_PACKED(ymm1, ymm8, ymm10); // C2 += A1 * B2
            }
            // store C
            VMOVU_PACKED(ymm0, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(0+i+ldc*(0+j))), 1);
            VMOVU_PACKED(ymm1, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(0+i+ldc*(1+j))), 1);
        }
    }
    else if (n_in & 1) {
        for (i = 0; i < (m_in/M_UNROLL_AVX2)*M_UNROLL_AVX2; i+= M_UNROLL_AVX2) {
            // load C
            LOAD_OR_ZERO_PACKED(ymm0, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(0+i+ldc*(0+j))));
            LOAD_OR_ZERO_PACKED(ymm4, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(1+i+ldc*(0+j))));
            for (int kk = 0; kk < (k/K_UNROLL_AVX2)*K_UNROLL_AVX2; kk += K_UNROLL_AVX2) {
                // load A1
                VMOVU_PACKED(ymm8, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(0+i+lda_in*(0+kk))), 0);
                // load B1
                VMOVU_PACKED(ymm9, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(0+kk+ldb_in*(0+j))), 0);
                // load B2
                VFMA_OR_VFNMA_PACKED(ymm0, ymm8, ymm9); // C1 += A1 * B1

                // load A2
                VMOVU_PACKED(ymm11, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(1+i+lda_in*(0+kk))), 0);
                VFMA_OR_VFNMA_PACKED(ymm4, ymm11, ymm9); // C3 += A2 * B1
            }
            // store C
            VMOVU_PACKED(ymm0, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(0+i+ldc*(0+j))), 1);
            VMOVU_PACKED(ymm4, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(1+i+ldc*(0+j))), 1);
        }
        // m tail
        if (m_in & 1) {
            // load C
            LOAD_OR_ZERO_PACKED(ymm0, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(0+i+ldc*(0+j))));
            for (int kk = 0; kk < (k/K_UNROLL_AVX2)*K_UNROLL_AVX2; kk += K_UNROLL_AVX2) {
                // load A1
                VMOVU_PACKED(ymm8, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(0+i+lda_in*(0+kk))), 0);
                // load B1
                VMOVU_PACKED(ymm9, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(0+kk+ldb_in*(0+j))), 0);
                // lod B);
                VFMA_OR_VFNMA_PACKED(ymm0, ymm8, ymm9); // C1 += A1 * B1
            }
            // store C
            VMOVU_PACKED(ymm0, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX2*(0+i+ldc*(0+j))), 1);
        }
    }
    ret();
}

#undef tcb_ftype
#undef LOAD_OR_ZERO_PACKED
#undef VMOVU_PACKED
#undef VFMA_OR_VFNMA_PACKED
#undef P_UNROLL_AVX2
