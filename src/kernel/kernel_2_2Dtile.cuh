// 全局内存合并
#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>

template<const int tile_width>
__global__ void mysgemm_v1_2Dtile(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {

  int start_row = (blockDim.y*blockIdx.y + threadIdx.y)*tile_width;
  int start_col = (blockDim.x*blockIdx.x + threadIdx.x)*tile_width;
  int end_row = start_row + tile_width;
  int end_col = start_col + tile_width;

  for (int row = start_row; row < end_row; row++) {
    for(int col = start_col; col < end_col; col++) {
      float sum = 0;
      for (int i = 0; i < K; i++) {
        sum += A[row * K + i]*B[i * K + col];
      }
      C[row*K+col] = sum;
    }
  }
}
