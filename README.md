# README #

tcBLAS is a code where I experiment with BLAS performance. Current features:
 - JIT compiled compact GEMM, TRSM and GETRF for AVX2 and AVX512 enabled Intel architectures.

### Build ###

An example build sequence from the tcBLAS root directory, where <TCB_INSTALL_DIR> is the install directory:

- $ mkdir build && cd build
- $ cmake .. -DCMAKE_INSTALL_PREFIX=<TCB_INSTALL_DIR>
- $ make install -j <NPROCS>

### Notes ###
- An oddity 
    - Currently the Compact TRSM kernels must be generated for 3 input arguments (pointer to a, pointer to b, and pointer to location of a single float with value 1.0). This will be fixed later, it was a quick patch to generate a vector of ones to minimize divisions.
    - Same for LU (no pivot)

### Contact ###
Timothy B. Costa
costa.timothy@gmail.com
