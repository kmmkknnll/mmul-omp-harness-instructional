#include <iostream>
#include <string.h>
#include <stdlib.h>
#include <omp.h>
#include "likwid-stuff.h"

const char* dgemm_desc = "Blocked dgemm, OpenMP-enabled";


/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are n-by-n matrices stored in row-major format.
 * On exit, A and B maintain their input values. */

void square_dgemm_blocked(int n, int block_size, double* A, double* B, double* C)
{
    int num_blocks = n / block_size;

    #pragma omp parallel
    {

        std::vector<double> A_small(block_size * block_size);
        std::vector<double> B_small(block_size * block_size);
        std::vector<double> C_small(block_size * block_size);

        LIKWID_MARKER_START(MY_MARKER_REGION_NAME);

        #pragma omp for collapse(2) schedule(static)
        for (int block_row = 0; block_row < num_blocks; block_row++) {
            for (int block_col = 0; block_col < num_blocks; block_col++) {
                int row_start = block_row * block_size;
                int col_start = block_col * block_size;

                // 把这块 C 从大矩阵读到本地，之后在本地累加，不反复读写大矩阵
                for (int i = 0; i < block_size; i++)
                    for (int j = 0; j < block_size; j++)
                        C_small[i*block_size+j] = C[(row_start+i)*n + (col_start+j)];

                for (int block_k = 0; block_k < num_blocks; block_k++) {
                    int k_start = block_k * block_size;

                    // 拷贝 A 子块: A_small[i][k] = A[row_start+i][k_start+k]
                    for (int i = 0; i < block_size; i++)
                        for (int k = 0; k < block_size; k++)
                            A_small[i*block_size+k] = A[(row_start+i)*n + (k_start+k)];

                    // 拷贝 B 子块，同时转置: B_small[j][k] = B[k_start+k][col_start+j]
                    for (int k = 0; k < block_size; k++)
                        for (int j = 0; j < block_size; j++)
                            B_small[j*block_size+k] = B[(k_start+k)*n + (col_start+j)];

                    // 乘法：A_small 和 B_small 内层都按 k 连续访问
                    for (int i = 0; i < block_size; i++) {
                        for (int j = 0; j < block_size; j++) {
                            double sum = C_small[i*block_size+j];
                            for (int k = 0; k < block_size; k++)
                                sum += A_small[i*block_size+k] * B_small[j*block_size+k];
                            C_small[i*block_size+j] = sum;
                        }
                    }


                // 这块 C 在所有 block_k 累加完后，一次性写回大矩阵
                for (int i = 0; i < block_size; i++)
                    for (int j = 0; j < block_size; j++)
                        C[(row_start+i)*n + (col_start+j)] = C_small[i*block_size+j];
               }
        }
    LIKWID_MARKER_STOP(MY_MARKER_REGION_NAME);
    }
}