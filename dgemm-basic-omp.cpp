#include <iostream>
#include <omp.h>
#include "likwid-stuff.h"

const char* dgemm_desc = "Basic implementation, OpenMP-enabled, three-loop dgemm.";

/*
 * This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are n-by-n matrices stored in row-major format.
 * On exit, A and B maintain their input values.
 */
void square_dgemm(int n, double* A, double* B, double* C) 
{
    #pragma omp parallel
    {
        LIKWID_MARKER_START(MY_MARKER_REGION_NAME);

        #pragma omp for collapse(2) schedule(static)
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                for (int k = 0; k < n; ++k) {
                    C[i * n + j] +=
                        A[i * n + k] * B[k * n + j];
                }
            }
        }

        LIKWID_MARKER_STOP(MY_MARKER_REGION_NAME);
    }
}
