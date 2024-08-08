// 使用寄存器缓存共享内存
#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>

template<const int BM,
        const int BN,
        const int BK,
        const int TM,
        const int TN>
__global__ void mysgemm_v6(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    // blockIdx.x 和 blockIdx.y 用于获取当前block在整个网格中的索引，这允许我们确定在大矩阵中处理的位置。
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // 计算每个block中行和列的线程数量，以及每个线程负责的元素数量。
    int block_row_thread = BN / TN;
    int block_col_thread = BM / TM;
    int thread_num = block_row_thread * block_col_thread; // 一个线程负责计算block中TM*TN个元素

    int tx = (threadIdx.x % block_row_thread) * TN;
    int ty = (threadIdx.x / block_row_thread) * TM;

    // 声明共享内存数组 As 和 Bs，用于缓存从全局内存中加载的矩阵 A 和 B 的数据。
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    // 移动到当前block
    A = &A[by * BM * K];
    B = &B[bx * BN];
    C = &C[by * BM * N + bx * BN];

    /*
    线程块的元素加载：计算每个线程在共享内存中应加载的 A 和 B 的位置，以及遍历时的步长（stride）。

    当前线程负责搬运全局内存中第a_tile_row行，第a_tile_col列元素至共享内存第a_tile_row行，第a_tile_col列
    a_tile_stride表示block中线程可搬运a_tile_stride行至共享内存；

    若BM=64,BK=8,thread_num=512,则a_tile_stride=64,a_tile_stride=BM，表示每个线程搬运一轮即可完成所需元素的搬运;
    若BM=128,BK=8,thread_num=512,则a_tile_stride=64,表示每个线程搬运两轮即可完成所需元素的搬运;
    */
    int a_tile_row = threadIdx.x / BK;
    int a_tile_col = threadIdx.x % BK;
    int a_tile_stride = thread_num / BK;

    int b_tile_row = threadIdx.x / BN;
    int b_tile_col = threadIdx.x % BN;
    int b_tile_stride = thread_num / BN;

    float tmp[TM][TN] = {0.}; // 每个线程负责TM*TN个元素，则需要申请TM*TN个寄存器保存累加值，额外的一个寄存器用于缓存；
    float a_frag[TM] = {0.};
    float b_frag[TN] = {0.};

    #pragma unroll
    // 在每次迭代中，我们处理 BK 列的块。
    for (int k = 0; k < K; k += BK) {
        #pragma unroll
        // 将全局内存中的矩阵 A 的一部分加载到共享内存 As 中。
        // BM 为总行数，i 每次增加 a_tile_stride，这意味着我们可以选择性地加载多行数据。
        for (int i = 0; i < BM; i += a_tile_stride) {
            // (a_tile_row + i) 是当前处理的行号，BK 是该块的宽度。此处，多个线程负责不同的行（以 a_tile_stride 为步长），
            As[(a_tile_row + i) * BK + a_tile_col] = A[(a_tile_row + i) * K + a_tile_col];
        }
        #pragma unroll
        for (int i = 0; i < BK; i += b_tile_stride) {
            Bs[(b_tile_row + i) * BN + b_tile_col] = B[(b_tile_row + i) * N + b_tile_col];
        }
        __syncthreads();

        // 移动到下一个块
        A += BK;     // 对于矩阵 A，移动 BK 列。向右移动
        B += BK * N; // 对于矩阵 B，移动 BK * N 行，因为 B 是按列存储的。向下移动

        // 进行计算
        #pragma unroll
        // 外层循环按列迭代，处理当前块中的每一列。每次迭代将对 A 矩阵的一列和 B 矩阵的一行进行处理
        for (int i = 0; i < BK; i++) {
            #pragma unroll
            for (int j = 0; j < TM; j++) {
                // 将 A 矩阵 As 中的当前列的多个行数据加载到 a_frag 中。
                a_frag[j] = As[(ty + j) * BK + i];
            }
            #pragma unroll
            for (int l = 0; l < TN; l++) {
                // 将 B 矩阵 Bs 中的当前行的多个列数据加载到 b_frag 中。
                b_frag[l] = Bs[tx + l + i * BN];
            }
            #pragma unroll
            for (int j = 0; j < TM; j++) {
                #pragma unroll
                for (int l = 0; l < TN; l++)
                // 外层循环依次遍历 a_frag 的每一行（每一个 j），内层循环遍历 b_frag 的每一列（每一个 l）。
                // 对于每一对 (j, l)，计算 a_frag[j] * b_frag[l] 的乘积，并累加到 tmp[j][l] 中。
                    tmp[j][l] += a_frag[j] * b_frag[l];
            }
        }
        __syncthreads();
    }

    // 写回
    #pragma unroll
    for (int j = 0; j < TM; j++) {
        for (int l = 0; l < TN; l++)
            C[(ty + j) * N + tx + l] = alpha * tmp[j][l] + beta * C[(ty + j) * N + tx + l];
    }
}