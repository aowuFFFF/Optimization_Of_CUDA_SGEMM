// 朴素
#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>

/*

Matrix sizes:
MxK * KxN = MxN

*/

__global__ __launch_bounds__(1024) void
mysgemm_v1(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {

    int gx = blockIdx.x * blockDim.x + threadIdx.x; // 全局x
    int gy = blockIdx.y * blockDim.y + threadIdx.y; // 全局y

    float tmp = 0.;
    for (int i = 0; i < K; i++) {
        tmp += A[gx * K + i] * B[i * N + gy]; // 两次全局内存访问和一次FMA（累加乘）
    }
    C[gx * N + gy] = alpha * tmp + beta * C[gx * N + gy];
}
