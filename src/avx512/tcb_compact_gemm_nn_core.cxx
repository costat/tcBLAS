/*****************************************************************************/
/* Author: Timothy Costa
******************************************************************************/

#if defined(SINGLE)
#define tcb_ftype float
#define VMOVU_PACKED(reg, mat_ptr, mat_offset, load_store) do { \
    if (load_store) vmovups( zword[mat_ptr + mat_offset], reg ); \
    else vmovups( reg, zword[mat_ptr + mat_offset] ); \
} while(0)
#define VFMA_OR_VFNMA_PACKED vfma_or_vfnma_ps
#define P_UNROLL_AVX512 P_UNROLL_AVX512_F32
#elif defined(DOUBLE)
#define tcb_ftype double 
#define VMOVU_PACKED(reg, mat_ptr, mat_offset, load_store) do { \
    if (load_store) vmovupd( zword[mat_ptr + mat_offset], reg ); \
    else vmovupd( reg, zword[mat_ptr + mat_offset] ); \
} while(0)
#define VFMA_OR_VFNMA_PACKED vfma_or_vfnma_pd
#define P_UNROLL_AVX512 P_UNROLL_AVX512_F64
#endif

#define LOAD_OR_ZERO_PACKED(reg, mat_ptr, mat_offset) do { \
    if (beta) { \
        VMOVU_PACKED(reg, mat_ptr, mat_offset, 0); \
    } \
    else { \
        vpxorq(reg, reg, reg); \
    } \
} while (0)

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

    bool _avx512f_ = 1;

    int i, j, n_rem, m_rem;

    for (j = 0; j < (n_in/N_UNROLL_AVX512)*N_UNROLL_AVX512; j += N_UNROLL_AVX512) {
        for (i = 0; i < (m_in/M_UNROLL_AVX512)*M_UNROLL_AVX512; i+= M_UNROLL_AVX512) {
            // load C
            LOAD_OR_ZERO_PACKED(zmm0, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+ldc*(0+j))));
            LOAD_OR_ZERO_PACKED(zmm1, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(1+i+ldc*(0+j))));
            LOAD_OR_ZERO_PACKED(zmm2, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(2+i+ldc*(0+j))));
            LOAD_OR_ZERO_PACKED(zmm3, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(3+i+ldc*(0+j))));

            LOAD_OR_ZERO_PACKED(zmm4, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+ldc*(1+j))));
            LOAD_OR_ZERO_PACKED(zmm5, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(1+i+ldc*(1+j))));
            LOAD_OR_ZERO_PACKED(zmm6, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(2+i+ldc*(1+j))));
            LOAD_OR_ZERO_PACKED(zmm7, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(3+i+ldc*(1+j))));

            LOAD_OR_ZERO_PACKED(zmm8, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+ldc*(2+j))));
            LOAD_OR_ZERO_PACKED(zmm9, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(1+i+ldc*(2+j))));
            LOAD_OR_ZERO_PACKED(zmm10, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(2+i+ldc*(2+j))));
            LOAD_OR_ZERO_PACKED(zmm11, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(3+i+ldc*(2+j))));

            LOAD_OR_ZERO_PACKED(zmm12, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+ldc*(3+j))));
            LOAD_OR_ZERO_PACKED(zmm13, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(1+i+ldc*(3+j))));
            LOAD_OR_ZERO_PACKED(zmm14, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(2+i+ldc*(3+j))));
            LOAD_OR_ZERO_PACKED(zmm15, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(3+i+ldc*(3+j))));

            LOAD_OR_ZERO_PACKED(zmm16, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+ldc*(4+j))));
            LOAD_OR_ZERO_PACKED(zmm17, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(1+i+ldc*(4+j))));
            LOAD_OR_ZERO_PACKED(zmm18, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(2+i+ldc*(4+j))));
            LOAD_OR_ZERO_PACKED(zmm19, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(3+i+ldc*(4+j))));

            for (int kk = 0; kk < (k/K_UNROLL_AVX512)*K_UNROLL_AVX512; kk += K_UNROLL_AVX512) {
                VMOVU_PACKED(zmm20, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+kk+ldb_in*(0+j))), 0);
                VMOVU_PACKED(zmm21, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+lda_in*(0+kk))), 0);
                VFMA_OR_VFNMA_PACKED(zmm0, zmm20, zmm21); 
                VMOVU_PACKED(zmm22, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(1+i+lda_in*(0+kk))), 0);
                VFMA_OR_VFNMA_PACKED(zmm1, zmm20, zmm22);
                VMOVU_PACKED(zmm23, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(2+i+lda_in*(0+kk))), 0);
                VFMA_OR_VFNMA_PACKED(zmm2, zmm20, zmm23);
                VMOVU_PACKED(zmm24, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(3+i+lda_in*(0+kk))), 0);
                VFMA_OR_VFNMA_PACKED(zmm3, zmm20, zmm24);

                VMOVU_PACKED(zmm25, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+kk+ldb_in*(1+j))), 0);
                VFMA_OR_VFNMA_PACKED(zmm4, zmm25, zmm21); 
                VFMA_OR_VFNMA_PACKED(zmm5, zmm25, zmm22);
                VFMA_OR_VFNMA_PACKED(zmm6, zmm25, zmm23);
                VFMA_OR_VFNMA_PACKED(zmm7, zmm25, zmm24);

                VMOVU_PACKED(zmm26, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+kk+ldb_in*(2+j))), 0);
                VFMA_OR_VFNMA_PACKED(zmm8, zmm26, zmm21); 
                VFMA_OR_VFNMA_PACKED(zmm9, zmm26, zmm22);
                VFMA_OR_VFNMA_PACKED(zmm10, zmm26, zmm23);
                VFMA_OR_VFNMA_PACKED(zmm11, zmm26, zmm24);

                VMOVU_PACKED(zmm27, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+kk+ldb_in*(3+j))), 0);
                VFMA_OR_VFNMA_PACKED(zmm12, zmm27, zmm21); 
                VFMA_OR_VFNMA_PACKED(zmm13, zmm27, zmm22);
                VFMA_OR_VFNMA_PACKED(zmm14, zmm27, zmm23);
                VFMA_OR_VFNMA_PACKED(zmm15, zmm27, zmm24);

                VMOVU_PACKED(zmm28, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+kk+ldb_in*(4+j))), 0);
                VFMA_OR_VFNMA_PACKED(zmm16, zmm28, zmm21); 
                VFMA_OR_VFNMA_PACKED(zmm17, zmm28, zmm22);
                VFMA_OR_VFNMA_PACKED(zmm18, zmm28, zmm23);
                VFMA_OR_VFNMA_PACKED(zmm19, zmm28, zmm24);
            }
            // store C
            VMOVU_PACKED(zmm0, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+ldc*(0+j))), 1);
            VMOVU_PACKED(zmm1, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(1+i+ldc*(0+j))), 1);
            VMOVU_PACKED(zmm2, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(2+i+ldc*(0+j))), 1);
            VMOVU_PACKED(zmm3, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(3+i+ldc*(0+j))), 1);

            VMOVU_PACKED(zmm4, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+ldc*(1+j))), 1);
            VMOVU_PACKED(zmm5, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(1+i+ldc*(1+j))), 1);
            VMOVU_PACKED(zmm6, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(2+i+ldc*(1+j))), 1);
            VMOVU_PACKED(zmm7, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(3+i+ldc*(1+j))), 1);

            VMOVU_PACKED(zmm8, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+ldc*(2+j))), 1);
            VMOVU_PACKED(zmm9, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(1+i+ldc*(2+j))), 1);
            VMOVU_PACKED(zmm10, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(2+i+ldc*(2+j))), 1);
            VMOVU_PACKED(zmm11, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(3+i+ldc*(2+j))), 1);

            VMOVU_PACKED(zmm12, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+ldc*(3+j))), 1);
            VMOVU_PACKED(zmm13, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(1+i+ldc*(3+j))), 1);
            VMOVU_PACKED(zmm14, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(2+i+ldc*(3+j))), 1);
            VMOVU_PACKED(zmm15, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(3+i+ldc*(3+j))), 1);

            VMOVU_PACKED(zmm16, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+ldc*(4+j))), 1);
            VMOVU_PACKED(zmm17, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(1+i+ldc*(4+j))), 1);
            VMOVU_PACKED(zmm18, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(2+i+ldc*(4+j))), 1);
            VMOVU_PACKED(zmm19, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(3+i+ldc*(4+j))), 1);
        } // i/m loop
        m_rem = m_in-i;
        // m tail
        if (m_rem & 2 && m_rem & 1) {
            // load C
            LOAD_OR_ZERO_PACKED(zmm0, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+ldc*(0+j))));
            LOAD_OR_ZERO_PACKED(zmm1, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(1+i+ldc*(0+j))));
            LOAD_OR_ZERO_PACKED(zmm2, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(2+i+ldc*(0+j))));

            LOAD_OR_ZERO_PACKED(zmm4, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+ldc*(1+j))));
            LOAD_OR_ZERO_PACKED(zmm5, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(1+i+ldc*(1+j))));
            LOAD_OR_ZERO_PACKED(zmm6, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(2+i+ldc*(1+j))));

            LOAD_OR_ZERO_PACKED(zmm8, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+ldc*(2+j))));
            LOAD_OR_ZERO_PACKED(zmm9, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(1+i+ldc*(2+j))));
            LOAD_OR_ZERO_PACKED(zmm10, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(2+i+ldc*(2+j))));

            LOAD_OR_ZERO_PACKED(zmm12, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+ldc*(3+j))));
            LOAD_OR_ZERO_PACKED(zmm13, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(1+i+ldc*(3+j))));
            LOAD_OR_ZERO_PACKED(zmm14, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(2+i+ldc*(3+j))));

            LOAD_OR_ZERO_PACKED(zmm16, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+ldc*(4+j))));
            LOAD_OR_ZERO_PACKED(zmm17, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(1+i+ldc*(4+j))));
            LOAD_OR_ZERO_PACKED(zmm18, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(2+i+ldc*(4+j))));


            for (int kk = 0; kk < (k/K_UNROLL_AVX512)*K_UNROLL_AVX512; kk += K_UNROLL_AVX512) {
                VMOVU_PACKED(zmm20, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+kk+ldb_in*(0+j))), 0);
                VMOVU_PACKED(zmm21, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+lda_in*(0+kk))), 0);
                VFMA_OR_VFNMA_PACKED(zmm0, zmm20, zmm21); 
                VMOVU_PACKED(zmm22, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(1+i+lda_in*(0+kk))), 0);
                VFMA_OR_VFNMA_PACKED(zmm1, zmm20, zmm22);
                VMOVU_PACKED(zmm23, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(2+i+lda_in*(0+kk))), 0);
                VFMA_OR_VFNMA_PACKED(zmm2, zmm20, zmm23);

                VMOVU_PACKED(zmm25, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+kk+ldb_in*(1+j))), 0);
                VFMA_OR_VFNMA_PACKED(zmm4, zmm25, zmm21); 
                VFMA_OR_VFNMA_PACKED(zmm5, zmm25, zmm22);
                VFMA_OR_VFNMA_PACKED(zmm6, zmm25, zmm23);

                VMOVU_PACKED(zmm26, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+kk+ldb_in*(2+j))), 0);
                VFMA_OR_VFNMA_PACKED(zmm8, zmm26, zmm21); 
                VFMA_OR_VFNMA_PACKED(zmm9, zmm26, zmm22);
                VFMA_OR_VFNMA_PACKED(zmm10, zmm26, zmm23);

                VMOVU_PACKED(zmm27, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+kk+ldb_in*(3+j))), 0);
                VFMA_OR_VFNMA_PACKED(zmm12, zmm27, zmm21); 
                VFMA_OR_VFNMA_PACKED(zmm13, zmm27, zmm22);
                VFMA_OR_VFNMA_PACKED(zmm14, zmm27, zmm23);

                VMOVU_PACKED(zmm28, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+kk+ldb_in*(4+j))), 0);
                VFMA_OR_VFNMA_PACKED(zmm16, zmm28, zmm21); 
                VFMA_OR_VFNMA_PACKED(zmm17, zmm28, zmm22);
                VFMA_OR_VFNMA_PACKED(zmm18, zmm28, zmm23);
            }
            // store C
            VMOVU_PACKED(zmm0, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+ldc*(0+j))), 1);
            VMOVU_PACKED(zmm1, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(1+i+ldc*(0+j))), 1);
            VMOVU_PACKED(zmm2, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(2+i+ldc*(0+j))), 1);

            VMOVU_PACKED(zmm4, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+ldc*(1+j))), 1);
            VMOVU_PACKED(zmm5, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(1+i+ldc*(1+j))), 1);
            VMOVU_PACKED(zmm6, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(2+i+ldc*(1+j))), 1);

            VMOVU_PACKED(zmm8, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+ldc*(2+j))), 1);
            VMOVU_PACKED(zmm9, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(1+i+ldc*(2+j))), 1);
            VMOVU_PACKED(zmm10, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(2+i+ldc*(2+j))), 1);

            VMOVU_PACKED(zmm12, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+ldc*(3+j))), 1);
            VMOVU_PACKED(zmm13, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(1+i+ldc*(3+j))), 1);
            VMOVU_PACKED(zmm14, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(2+i+ldc*(3+j))), 1);

            VMOVU_PACKED(zmm16, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+ldc*(4+j))), 1);
            VMOVU_PACKED(zmm17, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(1+i+ldc*(4+j))), 1);
            VMOVU_PACKED(zmm18, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(2+i+ldc*(4+j))), 1);
        }
        else if (m_rem & 2) {
            // load C
            LOAD_OR_ZERO_PACKED(zmm0, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+ldc*(0+j))));
            LOAD_OR_ZERO_PACKED(zmm1, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(1+i+ldc*(0+j))));

            LOAD_OR_ZERO_PACKED(zmm4, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+ldc*(1+j))));
            LOAD_OR_ZERO_PACKED(zmm5, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(1+i+ldc*(1+j))));

            LOAD_OR_ZERO_PACKED(zmm8, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+ldc*(2+j))));
            LOAD_OR_ZERO_PACKED(zmm9, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(1+i+ldc*(2+j))));

            LOAD_OR_ZERO_PACKED(zmm12, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+ldc*(3+j))));
            LOAD_OR_ZERO_PACKED(zmm13, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(1+i+ldc*(3+j))));

            LOAD_OR_ZERO_PACKED(zmm16, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+ldc*(4+j))));
            LOAD_OR_ZERO_PACKED(zmm17, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(1+i+ldc*(4+j))));

            for (int kk = 0; kk < (k/K_UNROLL_AVX512)*K_UNROLL_AVX512; kk += K_UNROLL_AVX512) {
                VMOVU_PACKED(zmm20, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+kk+ldb_in*(0+j))), 0);
                VMOVU_PACKED(zmm21, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+lda_in*(0+kk))), 0);
                VFMA_OR_VFNMA_PACKED(zmm0, zmm20, zmm21); 
                VMOVU_PACKED(zmm22, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(1+i+lda_in*(0+kk))), 0);
                VFMA_OR_VFNMA_PACKED(zmm1, zmm20, zmm22);

                VMOVU_PACKED(zmm25, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+kk+ldb_in*(1+j))), 0);
                VFMA_OR_VFNMA_PACKED(zmm4, zmm25, zmm21); 
                VFMA_OR_VFNMA_PACKED(zmm5, zmm25, zmm22);

                VMOVU_PACKED(zmm26, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+kk+ldb_in*(2+j))), 0);
                VFMA_OR_VFNMA_PACKED(zmm8, zmm26, zmm21); 
                VFMA_OR_VFNMA_PACKED(zmm9, zmm26, zmm22);

                VMOVU_PACKED(zmm27, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+kk+ldb_in*(3+j))), 0);
                VFMA_OR_VFNMA_PACKED(zmm12, zmm27, zmm21); 
                VFMA_OR_VFNMA_PACKED(zmm13, zmm27, zmm22);

                VMOVU_PACKED(zmm28, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+kk+ldb_in*(4+j))), 0);
                VFMA_OR_VFNMA_PACKED(zmm16, zmm28, zmm21); 
                VFMA_OR_VFNMA_PACKED(zmm17, zmm28, zmm22);

            }
            // store C
            VMOVU_PACKED(zmm0, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+ldc*(0+j))), 1);
            VMOVU_PACKED(zmm1, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(1+i+ldc*(0+j))), 1);

            VMOVU_PACKED(zmm4, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+ldc*(1+j))), 1);
            VMOVU_PACKED(zmm5, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(1+i+ldc*(1+j))), 1);

            VMOVU_PACKED(zmm8, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+ldc*(2+j))), 1);
            VMOVU_PACKED(zmm9, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(1+i+ldc*(2+j))), 1);

            VMOVU_PACKED(zmm12, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+ldc*(3+j))), 1);
            VMOVU_PACKED(zmm13, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(1+i+ldc*(3+j))), 1);

            VMOVU_PACKED(zmm16, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+ldc*(4+j))), 1);
            VMOVU_PACKED(zmm17, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(1+i+ldc*(4+j))), 1);
        }
        else if (m_rem & 1) {
            // load C
            LOAD_OR_ZERO_PACKED(zmm0, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+ldc*(0+j))));

            LOAD_OR_ZERO_PACKED(zmm4, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+ldc*(1+j))));

            LOAD_OR_ZERO_PACKED(zmm8, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+ldc*(2+j))));

            LOAD_OR_ZERO_PACKED(zmm12, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+ldc*(3+j))));

            LOAD_OR_ZERO_PACKED(zmm16, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+ldc*(4+j))));
            for (int kk = 0; kk < (k/K_UNROLL_AVX512)*K_UNROLL_AVX512; kk += K_UNROLL_AVX512) {
                VMOVU_PACKED(zmm20, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+kk+ldb_in*(0+j))), 0);
                VMOVU_PACKED(zmm21, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+lda_in*(0+kk))), 0);
                VFMA_OR_VFNMA_PACKED(zmm0, zmm20, zmm21); 

                VMOVU_PACKED(zmm25, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+kk+ldb_in*(1+j))), 0);
                VFMA_OR_VFNMA_PACKED(zmm4, zmm25, zmm21); 

                VMOVU_PACKED(zmm26, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+kk+ldb_in*(2+j))), 0);
                VFMA_OR_VFNMA_PACKED(zmm8, zmm26, zmm21); 

                VMOVU_PACKED(zmm27, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+kk+ldb_in*(3+j))), 0);
                VFMA_OR_VFNMA_PACKED(zmm12, zmm27, zmm21); 

                VMOVU_PACKED(zmm28, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+kk+ldb_in*(4+j))), 0);
                VFMA_OR_VFNMA_PACKED(zmm16, zmm28, zmm21); 
            }
            // store C
            VMOVU_PACKED(zmm0, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+ldc*(0+j))), 1);

            VMOVU_PACKED(zmm4, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+ldc*(1+j))), 1);

            VMOVU_PACKED(zmm8, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+ldc*(2+j))), 1);

            VMOVU_PACKED(zmm12, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+ldc*(3+j))), 1);

            VMOVU_PACKED(zmm16, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+ldc*(4+j))), 1);
                
        }

    }
    n_rem = n_in-j;
    if (n_rem & 4) {
        for (i = 0; i < (m_in/M_UNROLL_AVX512)*M_UNROLL_AVX512; i+= M_UNROLL_AVX512) {
            // load C
            LOAD_OR_ZERO_PACKED(zmm0, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+ldc*(0+j))));
            LOAD_OR_ZERO_PACKED(zmm1, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(1+i+ldc*(0+j))));
            LOAD_OR_ZERO_PACKED(zmm2, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(2+i+ldc*(0+j))));
            LOAD_OR_ZERO_PACKED(zmm3, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(3+i+ldc*(0+j))));

            LOAD_OR_ZERO_PACKED(zmm4, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+ldc*(1+j))));
            LOAD_OR_ZERO_PACKED(zmm5, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(1+i+ldc*(1+j))));
            LOAD_OR_ZERO_PACKED(zmm6, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(2+i+ldc*(1+j))));
            LOAD_OR_ZERO_PACKED(zmm7, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(3+i+ldc*(1+j))));

            LOAD_OR_ZERO_PACKED(zmm8, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+ldc*(2+j))));
            LOAD_OR_ZERO_PACKED(zmm9, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(1+i+ldc*(2+j))));
            LOAD_OR_ZERO_PACKED(zmm10, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(2+i+ldc*(2+j))));
            LOAD_OR_ZERO_PACKED(zmm11, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(3+i+ldc*(2+j))));

            LOAD_OR_ZERO_PACKED(zmm12, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+ldc*(3+j))));
            LOAD_OR_ZERO_PACKED(zmm13, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(1+i+ldc*(3+j))));
            LOAD_OR_ZERO_PACKED(zmm14, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(2+i+ldc*(3+j))));
            LOAD_OR_ZERO_PACKED(zmm15, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(3+i+ldc*(3+j))));
            for (int kk = 0; kk < (k/K_UNROLL_AVX512)*K_UNROLL_AVX512; kk += K_UNROLL_AVX512) {
                VMOVU_PACKED(zmm20, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+kk+ldb_in*(0+j))), 0);
                VMOVU_PACKED(zmm21, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+lda_in*(0+kk))), 0);
                VFMA_OR_VFNMA_PACKED(zmm0, zmm20, zmm21); 
                VMOVU_PACKED(zmm22, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(1+i+lda_in*(0+kk))), 0);
                VFMA_OR_VFNMA_PACKED(zmm1, zmm20, zmm22);
                VMOVU_PACKED(zmm23, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(2+i+lda_in*(0+kk))), 0);
                VFMA_OR_VFNMA_PACKED(zmm2, zmm20, zmm23);
                VMOVU_PACKED(zmm24, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(3+i+lda_in*(0+kk))), 0);
                VFMA_OR_VFNMA_PACKED(zmm3, zmm20, zmm24);

                VMOVU_PACKED(zmm25, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+kk+ldb_in*(1+j))), 0);
                VFMA_OR_VFNMA_PACKED(zmm4, zmm25, zmm21); 
                VFMA_OR_VFNMA_PACKED(zmm5, zmm25, zmm22);
                VFMA_OR_VFNMA_PACKED(zmm6, zmm25, zmm23);
                VFMA_OR_VFNMA_PACKED(zmm7, zmm25, zmm24);

                VMOVU_PACKED(zmm26, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+kk+ldb_in*(2+j))), 0);
                VFMA_OR_VFNMA_PACKED(zmm8, zmm26, zmm21); 
                VFMA_OR_VFNMA_PACKED(zmm9, zmm26, zmm22);
                VFMA_OR_VFNMA_PACKED(zmm10, zmm26, zmm23);
                VFMA_OR_VFNMA_PACKED(zmm11, zmm26, zmm24);

                VMOVU_PACKED(zmm27, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+kk+ldb_in*(3+j))), 0);
                VFMA_OR_VFNMA_PACKED(zmm12, zmm27, zmm21); 
                VFMA_OR_VFNMA_PACKED(zmm13, zmm27, zmm22);
                VFMA_OR_VFNMA_PACKED(zmm14, zmm27, zmm23);
                VFMA_OR_VFNMA_PACKED(zmm15, zmm27, zmm24);
            }
            // store C
            VMOVU_PACKED(zmm0, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+ldc*(0+j))), 1);
            VMOVU_PACKED(zmm1, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(1+i+ldc*(0+j))), 1);
            VMOVU_PACKED(zmm2, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(2+i+ldc*(0+j))), 1);
            VMOVU_PACKED(zmm3, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(3+i+ldc*(0+j))), 1);

            VMOVU_PACKED(zmm4, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+ldc*(1+j))), 1);
            VMOVU_PACKED(zmm5, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(1+i+ldc*(1+j))), 1);
            VMOVU_PACKED(zmm6, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(2+i+ldc*(1+j))), 1);
            VMOVU_PACKED(zmm7, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(3+i+ldc*(1+j))), 1);

            VMOVU_PACKED(zmm8, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+ldc*(2+j))), 1);
            VMOVU_PACKED(zmm9, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(1+i+ldc*(2+j))), 1);
            VMOVU_PACKED(zmm10, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(2+i+ldc*(2+j))), 1);
            VMOVU_PACKED(zmm11, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(3+i+ldc*(2+j))), 1);

            VMOVU_PACKED(zmm12, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+ldc*(3+j))), 1);
            VMOVU_PACKED(zmm13, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(1+i+ldc*(3+j))), 1);
            VMOVU_PACKED(zmm14, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(2+i+ldc*(3+j))), 1);
            VMOVU_PACKED(zmm15, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(3+i+ldc*(3+j))), 1);
        } // i/m loop
        m_rem = m_in-i;
        // m tail
        if (m_rem & 2 && m_rem & 1) {
            // load C
            LOAD_OR_ZERO_PACKED(zmm0, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+ldc*(0+j))));
            LOAD_OR_ZERO_PACKED(zmm1, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(1+i+ldc*(0+j))));
            LOAD_OR_ZERO_PACKED(zmm2, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(2+i+ldc*(0+j))));

            LOAD_OR_ZERO_PACKED(zmm4, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+ldc*(1+j))));
            LOAD_OR_ZERO_PACKED(zmm5, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(1+i+ldc*(1+j))));
            LOAD_OR_ZERO_PACKED(zmm6, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(2+i+ldc*(1+j))));

            LOAD_OR_ZERO_PACKED(zmm8, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+ldc*(2+j))));
            LOAD_OR_ZERO_PACKED(zmm9, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(1+i+ldc*(2+j))));
            LOAD_OR_ZERO_PACKED(zmm10, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(2+i+ldc*(2+j))));

            LOAD_OR_ZERO_PACKED(zmm12, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+ldc*(3+j))));
            LOAD_OR_ZERO_PACKED(zmm13, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(1+i+ldc*(3+j))));
            LOAD_OR_ZERO_PACKED(zmm14, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(2+i+ldc*(3+j))));
            for (int kk = 0; kk < (k/K_UNROLL_AVX512)*K_UNROLL_AVX512; kk += K_UNROLL_AVX512) {
                VMOVU_PACKED(zmm20, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+kk+ldb_in*(0+j))), 0);
                VMOVU_PACKED(zmm21, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+lda_in*(0+kk))), 0);
                VFMA_OR_VFNMA_PACKED(zmm0, zmm20, zmm21); 
                VMOVU_PACKED(zmm22, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(1+i+lda_in*(0+kk))), 0);
                VFMA_OR_VFNMA_PACKED(zmm1, zmm20, zmm22);
                VMOVU_PACKED(zmm23, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(2+i+lda_in*(0+kk))), 0);
                VFMA_OR_VFNMA_PACKED(zmm2, zmm20, zmm23);

                VMOVU_PACKED(zmm25, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+kk+ldb_in*(1+j))), 0);
                VFMA_OR_VFNMA_PACKED(zmm4, zmm25, zmm21); 
                VFMA_OR_VFNMA_PACKED(zmm5, zmm25, zmm22);
                VFMA_OR_VFNMA_PACKED(zmm6, zmm25, zmm23);

                VMOVU_PACKED(zmm26, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+kk+ldb_in*(2+j))), 0);
                VFMA_OR_VFNMA_PACKED(zmm8, zmm26, zmm21); 
                VFMA_OR_VFNMA_PACKED(zmm9, zmm26, zmm22);
                VFMA_OR_VFNMA_PACKED(zmm10, zmm26, zmm23);

                VMOVU_PACKED(zmm27, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+kk+ldb_in*(3+j))), 0);
                VFMA_OR_VFNMA_PACKED(zmm12, zmm27, zmm21); 
                VFMA_OR_VFNMA_PACKED(zmm13, zmm27, zmm22);
                VFMA_OR_VFNMA_PACKED(zmm14, zmm27, zmm23);

            }
            // store C
            VMOVU_PACKED(zmm0, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+ldc*(0+j))), 1);
            VMOVU_PACKED(zmm1, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(1+i+ldc*(0+j))), 1);
            VMOVU_PACKED(zmm2, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(2+i+ldc*(0+j))), 1);

            VMOVU_PACKED(zmm4, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+ldc*(1+j))), 1);
            VMOVU_PACKED(zmm5, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(1+i+ldc*(1+j))), 1);
            VMOVU_PACKED(zmm6, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(2+i+ldc*(1+j))), 1);

            VMOVU_PACKED(zmm8, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+ldc*(2+j))), 1);
            VMOVU_PACKED(zmm9, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(1+i+ldc*(2+j))), 1);
            VMOVU_PACKED(zmm10, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(2+i+ldc*(2+j))), 1);

            VMOVU_PACKED(zmm12, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+ldc*(3+j))), 1);
            VMOVU_PACKED(zmm13, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(1+i+ldc*(3+j))), 1);
            VMOVU_PACKED(zmm14, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(2+i+ldc*(3+j))), 1);
        }
        else if (m_rem & 2) {
            // load C
            LOAD_OR_ZERO_PACKED(zmm0, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+ldc*(0+j))));
            LOAD_OR_ZERO_PACKED(zmm1, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(1+i+ldc*(0+j))));

            LOAD_OR_ZERO_PACKED(zmm4, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+ldc*(1+j))));
            LOAD_OR_ZERO_PACKED(zmm5, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(1+i+ldc*(1+j))));

            LOAD_OR_ZERO_PACKED(zmm8, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+ldc*(2+j))));
            LOAD_OR_ZERO_PACKED(zmm9, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(1+i+ldc*(2+j))));

            LOAD_OR_ZERO_PACKED(zmm12, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+ldc*(3+j))));
            LOAD_OR_ZERO_PACKED(zmm13, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(1+i+ldc*(3+j))));
            for (int kk = 0; kk < (k/K_UNROLL_AVX512)*K_UNROLL_AVX512; kk += K_UNROLL_AVX512) {
                VMOVU_PACKED(zmm20, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+kk+ldb_in*(0+j))), 0);
                VMOVU_PACKED(zmm21, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+lda_in*(0+kk))), 0);
                VFMA_OR_VFNMA_PACKED(zmm0, zmm20, zmm21); 
                VMOVU_PACKED(zmm22, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(1+i+lda_in*(0+kk))), 0);
                VFMA_OR_VFNMA_PACKED(zmm1, zmm20, zmm22);

                VMOVU_PACKED(zmm25, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+kk+ldb_in*(1+j))), 0);
                VFMA_OR_VFNMA_PACKED(zmm4, zmm25, zmm21); 
                VFMA_OR_VFNMA_PACKED(zmm5, zmm25, zmm22);

                VMOVU_PACKED(zmm26, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+kk+ldb_in*(2+j))), 0);
                VFMA_OR_VFNMA_PACKED(zmm8, zmm26, zmm21); 
                VFMA_OR_VFNMA_PACKED(zmm9, zmm26, zmm22);

                VMOVU_PACKED(zmm27, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+kk+ldb_in*(3+j))), 0);
                VFMA_OR_VFNMA_PACKED(zmm12, zmm27, zmm21); 
                VFMA_OR_VFNMA_PACKED(zmm13, zmm27, zmm22);

            }
            // store C
            VMOVU_PACKED(zmm0, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+ldc*(0+j))), 1);
            VMOVU_PACKED(zmm1, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(1+i+ldc*(0+j))), 1);

            VMOVU_PACKED(zmm4, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+ldc*(1+j))), 1);
            VMOVU_PACKED(zmm5, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(1+i+ldc*(1+j))), 1);

            VMOVU_PACKED(zmm8, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+ldc*(2+j))), 1);
            VMOVU_PACKED(zmm9, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(1+i+ldc*(2+j))), 1);

            VMOVU_PACKED(zmm12, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+ldc*(3+j))), 1);
            VMOVU_PACKED(zmm13, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(1+i+ldc*(3+j))), 1);
        }
        else if (m_rem & 1) {
            // load C
            LOAD_OR_ZERO_PACKED(zmm0, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+ldc*(0+j))));

            LOAD_OR_ZERO_PACKED(zmm4, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+ldc*(1+j))));

            LOAD_OR_ZERO_PACKED(zmm8, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+ldc*(2+j))));

            LOAD_OR_ZERO_PACKED(zmm12, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+ldc*(3+j))));
            for (int kk = 0; kk < (k/K_UNROLL_AVX512)*K_UNROLL_AVX512; kk += K_UNROLL_AVX512) {
                VMOVU_PACKED(zmm20, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+kk+ldb_in*(0+j))), 0);
                VMOVU_PACKED(zmm21, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+lda_in*(0+kk))), 0);
                VFMA_OR_VFNMA_PACKED(zmm0, zmm20, zmm21); 

                VMOVU_PACKED(zmm25, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+kk+ldb_in*(1+j))), 0);
                VFMA_OR_VFNMA_PACKED(zmm4, zmm25, zmm21); 

                VMOVU_PACKED(zmm26, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+kk+ldb_in*(2+j))), 0);
                VFMA_OR_VFNMA_PACKED(zmm8, zmm26, zmm21); 

                VMOVU_PACKED(zmm27, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+kk+ldb_in*(3+j))), 0);
                VFMA_OR_VFNMA_PACKED(zmm12, zmm27, zmm21); 

            }
            // store C
            VMOVU_PACKED(zmm0, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+ldc*(0+j))), 1);

            VMOVU_PACKED(zmm4, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+ldc*(1+j))), 1);

            VMOVU_PACKED(zmm8, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+ldc*(2+j))), 1);

            VMOVU_PACKED(zmm12, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+ldc*(3+j))), 1);
        }
    } // j/n loop
    else if (n_rem & 2 && n_rem & 1) {
        for (i = 0; i < (m_in/M_UNROLL_AVX512)*M_UNROLL_AVX512; i+= M_UNROLL_AVX512) {
            // load C
            LOAD_OR_ZERO_PACKED(zmm0, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+ldc*(0+j))));
            LOAD_OR_ZERO_PACKED(zmm1, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(1+i+ldc*(0+j))));
            LOAD_OR_ZERO_PACKED(zmm2, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(2+i+ldc*(0+j))));
            LOAD_OR_ZERO_PACKED(zmm3, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(3+i+ldc*(0+j))));

            LOAD_OR_ZERO_PACKED(zmm4, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+ldc*(1+j))));
            LOAD_OR_ZERO_PACKED(zmm5, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(1+i+ldc*(1+j))));
            LOAD_OR_ZERO_PACKED(zmm6, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(2+i+ldc*(1+j))));
            LOAD_OR_ZERO_PACKED(zmm7, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(3+i+ldc*(1+j))));

            LOAD_OR_ZERO_PACKED(zmm8, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+ldc*(2+j))));
            LOAD_OR_ZERO_PACKED(zmm9, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(1+i+ldc*(2+j))));
            LOAD_OR_ZERO_PACKED(zmm10, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(2+i+ldc*(2+j))));
            LOAD_OR_ZERO_PACKED(zmm11, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(3+i+ldc*(2+j))));

            for (int kk = 0; kk < (k/K_UNROLL_AVX512)*K_UNROLL_AVX512; kk += K_UNROLL_AVX512) {
                VMOVU_PACKED(zmm20, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+kk+ldb_in*(0+j))), 0);
                VMOVU_PACKED(zmm21, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+lda_in*(0+kk))), 0);
                VFMA_OR_VFNMA_PACKED(zmm0, zmm20, zmm21); 
                VMOVU_PACKED(zmm22, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(1+i+lda_in*(0+kk))), 0);
                VFMA_OR_VFNMA_PACKED(zmm1, zmm20, zmm22);
                VMOVU_PACKED(zmm23, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(2+i+lda_in*(0+kk))), 0);
                VFMA_OR_VFNMA_PACKED(zmm2, zmm20, zmm23);
                VMOVU_PACKED(zmm24, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(3+i+lda_in*(0+kk))), 0);
                VFMA_OR_VFNMA_PACKED(zmm3, zmm20, zmm24);

                VMOVU_PACKED(zmm25, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+kk+ldb_in*(1+j))), 0);
                VFMA_OR_VFNMA_PACKED(zmm4, zmm25, zmm21); 
                VFMA_OR_VFNMA_PACKED(zmm5, zmm25, zmm22);
                VFMA_OR_VFNMA_PACKED(zmm6, zmm25, zmm23);
                VFMA_OR_VFNMA_PACKED(zmm7, zmm25, zmm24);

                VMOVU_PACKED(zmm26, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+kk+ldb_in*(2+j))), 0);
                VFMA_OR_VFNMA_PACKED(zmm8, zmm26, zmm21); 
                VFMA_OR_VFNMA_PACKED(zmm9, zmm26, zmm22);
                VFMA_OR_VFNMA_PACKED(zmm10, zmm26, zmm23);
                VFMA_OR_VFNMA_PACKED(zmm11, zmm26, zmm24);

            }
            // store C
            VMOVU_PACKED(zmm0, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+ldc*(0+j))), 1);
            VMOVU_PACKED(zmm1, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(1+i+ldc*(0+j))), 1);
            VMOVU_PACKED(zmm2, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(2+i+ldc*(0+j))), 1);
            VMOVU_PACKED(zmm3, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(3+i+ldc*(0+j))), 1);

            VMOVU_PACKED(zmm4, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+ldc*(1+j))), 1);
            VMOVU_PACKED(zmm5, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(1+i+ldc*(1+j))), 1);
            VMOVU_PACKED(zmm6, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(2+i+ldc*(1+j))), 1);
            VMOVU_PACKED(zmm7, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(3+i+ldc*(1+j))), 1);

            VMOVU_PACKED(zmm8, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+ldc*(2+j))), 1);
            VMOVU_PACKED(zmm9, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(1+i+ldc*(2+j))), 1);
            VMOVU_PACKED(zmm10, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(2+i+ldc*(2+j))), 1);
            VMOVU_PACKED(zmm11, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(3+i+ldc*(2+j))), 1);

        } // i/m loop
        // m tail
        m_rem = m_in-i; 
        if (m_rem & 2 && m_rem & 1) {
            // load C
            LOAD_OR_ZERO_PACKED(zmm0, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+ldc*(0+j))));
            LOAD_OR_ZERO_PACKED(zmm1, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(1+i+ldc*(0+j))));
            LOAD_OR_ZERO_PACKED(zmm2, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(2+i+ldc*(0+j))));

            LOAD_OR_ZERO_PACKED(zmm4, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+ldc*(1+j))));
            LOAD_OR_ZERO_PACKED(zmm5, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(1+i+ldc*(1+j))));
            LOAD_OR_ZERO_PACKED(zmm6, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(2+i+ldc*(1+j))));

            LOAD_OR_ZERO_PACKED(zmm8, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+ldc*(2+j))));
            LOAD_OR_ZERO_PACKED(zmm9, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(1+i+ldc*(2+j))));
            LOAD_OR_ZERO_PACKED(zmm10, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(2+i+ldc*(2+j))));

            for (int kk = 0; kk < (k/K_UNROLL_AVX512)*K_UNROLL_AVX512; kk += K_UNROLL_AVX512) {
                VMOVU_PACKED(zmm20, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+kk+ldb_in*(0+j))), 0);
                VMOVU_PACKED(zmm21, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+lda_in*(0+kk))), 0);
                VFMA_OR_VFNMA_PACKED(zmm0, zmm20, zmm21); 
                VMOVU_PACKED(zmm22, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(1+i+lda_in*(0+kk))), 0);
                VFMA_OR_VFNMA_PACKED(zmm1, zmm20, zmm22);
                VMOVU_PACKED(zmm23, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(2+i+lda_in*(0+kk))), 0);
                VFMA_OR_VFNMA_PACKED(zmm2, zmm20, zmm23);

                VMOVU_PACKED(zmm25, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+kk+ldb_in*(1+j))), 0);
                VFMA_OR_VFNMA_PACKED(zmm4, zmm25, zmm21); 
                VFMA_OR_VFNMA_PACKED(zmm5, zmm25, zmm22);
                VFMA_OR_VFNMA_PACKED(zmm6, zmm25, zmm23);

                VMOVU_PACKED(zmm26, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+kk+ldb_in*(2+j))), 0);
                VFMA_OR_VFNMA_PACKED(zmm8, zmm26, zmm21); 
                VFMA_OR_VFNMA_PACKED(zmm9, zmm26, zmm22);
                VFMA_OR_VFNMA_PACKED(zmm10, zmm26, zmm23);

            }
            // store C
            VMOVU_PACKED(zmm0, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+ldc*(0+j))), 1);
            VMOVU_PACKED(zmm1, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(1+i+ldc*(0+j))), 1);
            VMOVU_PACKED(zmm2, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(2+i+ldc*(0+j))), 1);

            VMOVU_PACKED(zmm4, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+ldc*(1+j))), 1);
            VMOVU_PACKED(zmm5, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(1+i+ldc*(1+j))), 1);
            VMOVU_PACKED(zmm6, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(2+i+ldc*(1+j))), 1);

            VMOVU_PACKED(zmm8, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+ldc*(2+j))), 1);
            VMOVU_PACKED(zmm9, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(1+i+ldc*(2+j))), 1);
            VMOVU_PACKED(zmm10, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(2+i+ldc*(2+j))), 1);

        }
        else if (m_rem & 2) {
            // load C
            LOAD_OR_ZERO_PACKED(zmm0, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+ldc*(0+j))));
            LOAD_OR_ZERO_PACKED(zmm1, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(1+i+ldc*(0+j))));

            LOAD_OR_ZERO_PACKED(zmm4, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+ldc*(1+j))));
            LOAD_OR_ZERO_PACKED(zmm5, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(1+i+ldc*(1+j))));

            LOAD_OR_ZERO_PACKED(zmm8, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+ldc*(2+j))));
            LOAD_OR_ZERO_PACKED(zmm9, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(1+i+ldc*(2+j))));

            for (int kk = 0; kk < (k/K_UNROLL_AVX512)*K_UNROLL_AVX512; kk += K_UNROLL_AVX512) {
                VMOVU_PACKED(zmm20, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+kk+ldb_in*(0+j))), 0);
                VMOVU_PACKED(zmm21, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+lda_in*(0+kk))), 0);
                VFMA_OR_VFNMA_PACKED(zmm0, zmm20, zmm21); 
                VMOVU_PACKED(zmm22, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(1+i+lda_in*(0+kk))), 0);
                VFMA_OR_VFNMA_PACKED(zmm1, zmm20, zmm22);

                VMOVU_PACKED(zmm25, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+kk+ldb_in*(1+j))), 0);
                VFMA_OR_VFNMA_PACKED(zmm4, zmm25, zmm21); 
                VFMA_OR_VFNMA_PACKED(zmm5, zmm25, zmm22);

                VMOVU_PACKED(zmm26, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+kk+ldb_in*(2+j))), 0);
                VFMA_OR_VFNMA_PACKED(zmm8, zmm26, zmm21); 
                VFMA_OR_VFNMA_PACKED(zmm9, zmm26, zmm22);

            }
            // store C
            VMOVU_PACKED(zmm0, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+ldc*(0+j))), 1);
            VMOVU_PACKED(zmm1, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(1+i+ldc*(0+j))), 1);

            VMOVU_PACKED(zmm4, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+ldc*(1+j))), 1);
            VMOVU_PACKED(zmm5, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(1+i+ldc*(1+j))), 1);

            VMOVU_PACKED(zmm8, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+ldc*(2+j))), 1);
            VMOVU_PACKED(zmm9, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(1+i+ldc*(2+j))), 1);

        }
        else if (m_rem & 1) {
            // load C
            LOAD_OR_ZERO_PACKED(zmm0, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+ldc*(0+j))));

            LOAD_OR_ZERO_PACKED(zmm4, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+ldc*(1+j))));

            LOAD_OR_ZERO_PACKED(zmm8, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+ldc*(2+j))));

            for (int kk = 0; kk < (k/K_UNROLL_AVX512)*K_UNROLL_AVX512; kk += K_UNROLL_AVX512) {
                VMOVU_PACKED(zmm20, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+kk+ldb_in*(0+j))), 0);
                VMOVU_PACKED(zmm21, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+lda_in*(0+kk))), 0);
                VFMA_OR_VFNMA_PACKED(zmm0, zmm20, zmm21); 

                VMOVU_PACKED(zmm25, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+kk+ldb_in*(1+j))), 0);
                VFMA_OR_VFNMA_PACKED(zmm4, zmm25, zmm21); 

                VMOVU_PACKED(zmm26, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+kk+ldb_in*(2+j))), 0);
                VFMA_OR_VFNMA_PACKED(zmm8, zmm26, zmm21); 

            }
            // store C
            VMOVU_PACKED(zmm0, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+ldc*(0+j))), 1);

            VMOVU_PACKED(zmm4, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+ldc*(1+j))), 1);

            VMOVU_PACKED(zmm8, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+ldc*(2+j))), 1);

        }
    }
    else if (n_rem & 2) {
        for (i = 0; i < (m_in/M_UNROLL_AVX512)*M_UNROLL_AVX512; i+= M_UNROLL_AVX512) {
            // load C
            LOAD_OR_ZERO_PACKED(zmm0, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+ldc*(0+j))));
            LOAD_OR_ZERO_PACKED(zmm1, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(1+i+ldc*(0+j))));
            LOAD_OR_ZERO_PACKED(zmm2, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(2+i+ldc*(0+j))));
            LOAD_OR_ZERO_PACKED(zmm3, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(3+i+ldc*(0+j))));

            LOAD_OR_ZERO_PACKED(zmm4, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+ldc*(1+j))));
            LOAD_OR_ZERO_PACKED(zmm5, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(1+i+ldc*(1+j))));
            LOAD_OR_ZERO_PACKED(zmm6, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(2+i+ldc*(1+j))));
            LOAD_OR_ZERO_PACKED(zmm7, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(3+i+ldc*(1+j))));

            for (int kk = 0; kk < (k/K_UNROLL_AVX512)*K_UNROLL_AVX512; kk += K_UNROLL_AVX512) {
                VMOVU_PACKED(zmm20, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+kk+ldb_in*(0+j))), 0);
                VMOVU_PACKED(zmm21, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+lda_in*(0+kk))), 0);
                VFMA_OR_VFNMA_PACKED(zmm0, zmm20, zmm21); 
                VMOVU_PACKED(zmm22, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(1+i+lda_in*(0+kk))), 0);
                VFMA_OR_VFNMA_PACKED(zmm1, zmm20, zmm22);
                VMOVU_PACKED(zmm23, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(2+i+lda_in*(0+kk))), 0);
                VFMA_OR_VFNMA_PACKED(zmm2, zmm20, zmm23);
                VMOVU_PACKED(zmm24, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(3+i+lda_in*(0+kk))), 0);
                VFMA_OR_VFNMA_PACKED(zmm3, zmm20, zmm24);

                VMOVU_PACKED(zmm25, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+kk+ldb_in*(1+j))), 0);
                VFMA_OR_VFNMA_PACKED(zmm4, zmm25, zmm21); 
                VFMA_OR_VFNMA_PACKED(zmm5, zmm25, zmm22);
                VFMA_OR_VFNMA_PACKED(zmm6, zmm25, zmm23);
                VFMA_OR_VFNMA_PACKED(zmm7, zmm25, zmm24);

            }
            // store C
            VMOVU_PACKED(zmm0, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+ldc*(0+j))), 1);
            VMOVU_PACKED(zmm1, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(1+i+ldc*(0+j))), 1);
            VMOVU_PACKED(zmm2, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(2+i+ldc*(0+j))), 1);
            VMOVU_PACKED(zmm3, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(3+i+ldc*(0+j))), 1);

            VMOVU_PACKED(zmm4, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+ldc*(1+j))), 1);
            VMOVU_PACKED(zmm5, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(1+i+ldc*(1+j))), 1);
            VMOVU_PACKED(zmm6, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(2+i+ldc*(1+j))), 1);
            VMOVU_PACKED(zmm7, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(3+i+ldc*(1+j))), 1);

        }
        m_rem = m_in-i;
        if (m_rem & 2 && m_rem & 1) {
            // load C
            LOAD_OR_ZERO_PACKED(zmm0, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+ldc*(0+j))));
            LOAD_OR_ZERO_PACKED(zmm1, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(1+i+ldc*(0+j))));
            LOAD_OR_ZERO_PACKED(zmm2, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(2+i+ldc*(0+j))));

            LOAD_OR_ZERO_PACKED(zmm4, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+ldc*(1+j))));
            LOAD_OR_ZERO_PACKED(zmm5, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(1+i+ldc*(1+j))));
            LOAD_OR_ZERO_PACKED(zmm6, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(2+i+ldc*(1+j))));

            for (int kk = 0; kk < (k/K_UNROLL_AVX512)*K_UNROLL_AVX512; kk += K_UNROLL_AVX512) {
                VMOVU_PACKED(zmm20, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+kk+ldb_in*(0+j))), 0);
                VMOVU_PACKED(zmm21, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+lda_in*(0+kk))), 0);
                VFMA_OR_VFNMA_PACKED(zmm0, zmm20, zmm21); 
                VMOVU_PACKED(zmm22, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(1+i+lda_in*(0+kk))), 0);
                VFMA_OR_VFNMA_PACKED(zmm1, zmm20, zmm22);
                VMOVU_PACKED(zmm23, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(2+i+lda_in*(0+kk))), 0);
                VFMA_OR_VFNMA_PACKED(zmm2, zmm20, zmm23);

                VMOVU_PACKED(zmm25, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+kk+ldb_in*(1+j))), 0);
                VFMA_OR_VFNMA_PACKED(zmm4, zmm25, zmm21); 
                VFMA_OR_VFNMA_PACKED(zmm5, zmm25, zmm22);
                VFMA_OR_VFNMA_PACKED(zmm6, zmm25, zmm23);

            }
            // store C
            VMOVU_PACKED(zmm0, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+ldc*(0+j))), 1);
            VMOVU_PACKED(zmm1, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(1+i+ldc*(0+j))), 1);
            VMOVU_PACKED(zmm2, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(2+i+ldc*(0+j))), 1);

            VMOVU_PACKED(zmm4, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+ldc*(1+j))), 1);
            VMOVU_PACKED(zmm5, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(1+i+ldc*(1+j))), 1);
            VMOVU_PACKED(zmm6, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(2+i+ldc*(1+j))), 1);

        }
        else if (m_rem & 2) {
            // load C
            LOAD_OR_ZERO_PACKED(zmm0, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+ldc*(0+j))));
            LOAD_OR_ZERO_PACKED(zmm1, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(1+i+ldc*(0+j))));

            LOAD_OR_ZERO_PACKED(zmm4, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+ldc*(1+j))));
            LOAD_OR_ZERO_PACKED(zmm5, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(1+i+ldc*(1+j))));

            for (int kk = 0; kk < (k/K_UNROLL_AVX512)*K_UNROLL_AVX512; kk += K_UNROLL_AVX512) {
                VMOVU_PACKED(zmm20, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+kk+ldb_in*(0+j))), 0);
                VMOVU_PACKED(zmm21, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+lda_in*(0+kk))), 0);
                VFMA_OR_VFNMA_PACKED(zmm0, zmm20, zmm21); 
                VMOVU_PACKED(zmm22, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(1+i+lda_in*(0+kk))), 0);
                VFMA_OR_VFNMA_PACKED(zmm1, zmm20, zmm22);

                VMOVU_PACKED(zmm25, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+kk+ldb_in*(1+j))), 0);
                VFMA_OR_VFNMA_PACKED(zmm4, zmm25, zmm21); 
                VFMA_OR_VFNMA_PACKED(zmm5, zmm25, zmm22);

            }
            // store C
            VMOVU_PACKED(zmm0, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+ldc*(0+j))), 1);
            VMOVU_PACKED(zmm1, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(1+i+ldc*(0+j))), 1);

            VMOVU_PACKED(zmm4, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+ldc*(1+j))), 1);
            VMOVU_PACKED(zmm5, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(1+i+ldc*(1+j))), 1);

        }
        else if (m_rem & 1) {
            // load C
            LOAD_OR_ZERO_PACKED(zmm0, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+ldc*(0+j))));

            LOAD_OR_ZERO_PACKED(zmm4, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+ldc*(1+j))));

            for (int kk = 0; kk < (k/K_UNROLL_AVX512)*K_UNROLL_AVX512; kk += K_UNROLL_AVX512) {
                VMOVU_PACKED(zmm20, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+kk+ldb_in*(0+j))), 0);
                VMOVU_PACKED(zmm21, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+lda_in*(0+kk))), 0);
                VFMA_OR_VFNMA_PACKED(zmm0, zmm20, zmm21); 

                VMOVU_PACKED(zmm25, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+kk+ldb_in*(1+j))), 0);
                VFMA_OR_VFNMA_PACKED(zmm4, zmm25, zmm21); 

            }
            // store C
            VMOVU_PACKED(zmm0, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+ldc*(0+j))), 1);

            VMOVU_PACKED(zmm4, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+ldc*(1+j))), 1);

        }
    }
    else if (n_rem & 1) {
        for (i = 0; i < (m_in/M_UNROLL_AVX512)*M_UNROLL_AVX512; i+= M_UNROLL_AVX512) {
            // load C
            LOAD_OR_ZERO_PACKED(zmm0, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+ldc*(0+j))));
            LOAD_OR_ZERO_PACKED(zmm1, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(1+i+ldc*(0+j))));
            LOAD_OR_ZERO_PACKED(zmm2, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(2+i+ldc*(0+j))));
            LOAD_OR_ZERO_PACKED(zmm3, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(3+i+ldc*(0+j))));

            for (int kk = 0; kk < (k/K_UNROLL_AVX512)*K_UNROLL_AVX512; kk += K_UNROLL_AVX512) {
                VMOVU_PACKED(zmm20, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+kk+ldb_in*(0+j))), 0);
                VMOVU_PACKED(zmm21, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+lda_in*(0+kk))), 0);
                VFMA_OR_VFNMA_PACKED(zmm0, zmm20, zmm21); 

                VMOVU_PACKED(zmm22, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(1+i+lda_in*(0+kk))), 0);
                VFMA_OR_VFNMA_PACKED(zmm1, zmm20, zmm22);

                VMOVU_PACKED(zmm23, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(2+i+lda_in*(0+kk))), 0);
                VFMA_OR_VFNMA_PACKED(zmm2, zmm20, zmm23);

                VMOVU_PACKED(zmm24, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(3+i+lda_in*(0+kk))), 0);
                VFMA_OR_VFNMA_PACKED(zmm3, zmm20, zmm24);

            }
            // store C
            VMOVU_PACKED(zmm0, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+ldc*(0+j))), 1);
            VMOVU_PACKED(zmm1, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(1+i+ldc*(0+j))), 1);
            VMOVU_PACKED(zmm2, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(2+i+ldc*(0+j))), 1);
            VMOVU_PACKED(zmm3, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(3+i+ldc*(0+j))), 1);

        }
        m_rem = m_in-i;
        if (m_rem & 2 && m_rem & 1) {
            // load C
            LOAD_OR_ZERO_PACKED(zmm0, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+ldc*(0+j))));
            LOAD_OR_ZERO_PACKED(zmm1, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(1+i+ldc*(0+j))));
            LOAD_OR_ZERO_PACKED(zmm2, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(2+i+ldc*(0+j))));

            for (int kk = 0; kk < (k/K_UNROLL_AVX512)*K_UNROLL_AVX512; kk += K_UNROLL_AVX512) {
                VMOVU_PACKED(zmm20, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+kk+ldb_in*(0+j))), 0);
                VMOVU_PACKED(zmm21, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+lda_in*(0+kk))), 0);
                VFMA_OR_VFNMA_PACKED(zmm0, zmm20, zmm21); 
                VMOVU_PACKED(zmm22, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(1+i+lda_in*(0+kk))), 0);
                VFMA_OR_VFNMA_PACKED(zmm1, zmm20, zmm22);
                VMOVU_PACKED(zmm23, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(2+i+lda_in*(0+kk))), 0);
                VFMA_OR_VFNMA_PACKED(zmm2, zmm20, zmm23);

            }
            // store C
            VMOVU_PACKED(zmm0, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+ldc*(0+j))), 1);
            VMOVU_PACKED(zmm1, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(1+i+ldc*(0+j))), 1);
            VMOVU_PACKED(zmm2, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(2+i+ldc*(0+j))), 1);

        }
        else if (m_rem & 2) {
            // load C
            LOAD_OR_ZERO_PACKED(zmm0, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+ldc*(0+j))));
            LOAD_OR_ZERO_PACKED(zmm1, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(1+i+ldc*(0+j))));

            for (int kk = 0; kk < (k/K_UNROLL_AVX512)*K_UNROLL_AVX512; kk += K_UNROLL_AVX512) {
                VMOVU_PACKED(zmm20, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+kk+ldb_in*(0+j))), 0);
                VMOVU_PACKED(zmm21, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+lda_in*(0+kk))), 0);
                VFMA_OR_VFNMA_PACKED(zmm0, zmm20, zmm21); 
                VMOVU_PACKED(zmm22, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(1+i+lda_in*(0+kk))), 0);
                VFMA_OR_VFNMA_PACKED(zmm1, zmm20, zmm22);

            }
            // store C
            VMOVU_PACKED(zmm0, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+ldc*(0+j))), 1);
            VMOVU_PACKED(zmm1, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(1+i+ldc*(0+j))), 1);

        }
        else if (m_rem & 1) {
            // load C
            LOAD_OR_ZERO_PACKED(zmm0, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+ldc*(0+j))));

            for (int kk = 0; kk < (k/K_UNROLL_AVX512)*K_UNROLL_AVX512; kk += K_UNROLL_AVX512) {
                VMOVU_PACKED(zmm20, b_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+kk+ldb_in*(0+j))), 0);
                VMOVU_PACKED(zmm21, a_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+lda_in*(0+kk))), 0);
                VFMA_OR_VFNMA_PACKED(zmm0, zmm20, zmm21); 

            }
            // store C
            VMOVU_PACKED(zmm0, c_ptr, sizeof(tcb_ftype)*(P_UNROLL_AVX512*(0+i+ldc*(0+j))), 1);

        }
    }

    ret();
}

#undef tcb_ftype
#undef LOAD_OR_ZERO_PACKED
#undef VMOVU_PACKED
#undef VFMA_OR_VFNMA_PACKED
#undef P_UNROLL_AVX512
