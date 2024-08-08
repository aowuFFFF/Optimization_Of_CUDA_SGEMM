#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#define BLOCKSIZE 256

/*******************/
/* iDivUp FUNCTION */
/*******************/
int iDivUp(int a, int b){ 
    return ((a % b) != 0) ? (a / b + 1) : (a / b); 
}

/********************/
/* CUDA ERROR CHECK */
/********************/
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

/********************/
/* ADD_FLOAT KERNEL */
/********************/

// 在这段代码中，虽然每个线程处理了四个float值，但它们是作为单独的标量浮点数来处理的，并没有使用float4这种数据结构。
__global__ void add_float(float *d_a, float *d_b, float *d_c, unsigned int N) {

    const int tid = 4 * threadIdx.x + blockIdx.x * (4 * blockDim.x);

    if (tid < N) {

        float a1 = d_a[tid];
        float b1 = d_b[tid];

        float a2 = d_a[tid+1];
        float b2 = d_b[tid+1];

        float a3 = d_a[tid+2];
        float b3 = d_b[tid+2];

        float a4 = d_a[tid+3];
        float b4 = d_b[tid+3];

        float c1 = a1 + b1;
        float c2 = a2 + b2;
        float c3 = a3 + b3;
        float c4 = a4 + b4;

        d_c[tid] = c1;
        d_c[tid+1] = c2;
        d_c[tid+2] = c3;
        d_c[tid+3] = c4;

        //if ((tid < 1800) && (tid > 1790)) {
            //printf("%i %i %i %f %f %f\n", tid, threadIdx.x, blockIdx.x, a1, b1, c1);
            //printf("%i %i %i %f %f %f\n", tid+1, threadIdx.x, blockIdx.x, a2, b2, c2);
            //printf("%i %i %i %f %f %f\n", tid+2, threadIdx.x, blockIdx.x, a3, b3, c3);
            //printf("%i %i %i %f %f %f\n", tid+3, threadIdx.x, blockIdx.x, a4, b4, c4);
        //}

    }

}

/*********************/
/* ADD_FLOAT2 KERNEL */
/*********************/
__global__ void add_float2(float2 *d_a, float2 *d_b, float2 *d_c, unsigned int N) {

    const int tid = 2 * threadIdx.x + blockIdx.x * (2 * blockDim.x);

    if (tid < N) {

        float2 a1 = d_a[tid];
        float2 b1 = d_b[tid];

        float2 a2 = d_a[tid+1];
        float2 b2 = d_b[tid+1];

        float2 c1;
        c1.x = a1.x + b1.x;
        c1.y = a1.y + b1.y;

        float2 c2;
        c2.x = a2.x + b2.x;
        c2.y = a2.y + b2.y;

        d_c[tid] = c1;
        d_c[tid+1] = c2;

    }

}

/*********************/
/* ADD_FLOAT4 KERNEL */
/*********************/
__global__ void add_float4(float4 *d_a, float4 *d_b, float4 *d_c, unsigned int N) {

    const int tid = 1 * threadIdx.x + blockIdx.x * (1 * blockDim.x);

    if (tid < N/4) {

        float4 a1 = d_a[tid];
        float4 b1 = d_b[tid];

        float4 c1;
        c1.x = a1.x + b1.x;
        c1.y = a1.y + b1.y;
        c1.z = a1.z + b1.z;
        c1.w = a1.w + b1.w;

        d_c[tid] = c1;

    }

}

struct __device_builtin__ __builtin_align__(16) float_test
{
    float x, y, z, w;
} typedef float_test;

__global__ void add_float_test(float_test *d_a, float_test *d_b, float_test *d_c, unsigned int N) {

    const int tid = 1 * threadIdx.x + blockIdx.x * (1 * blockDim.x);

    if (tid < N/4) {

        float_test a1 = d_a[tid];
        float_test b1 = d_b[tid];

        float_test c1;
        c1.x = a1.x + b1.x;
        c1.y = a1.y + b1.y;
        c1.z = a1.z + b1.z;
        c1.w = a1.w + b1.w;

        d_c[tid] = c1;

    }

}

/********/
/* MAIN */
/********/
int main() {

    const int N = 4*10000000; //定义一个常量N，表示向量的大小，这里设置为4亿。

    const float a = 3.f;
    const float b = 5.f;

    // --- float

    thrust::device_vector<float> d_A(N, a); //Thrust库中的一个容器类，用于在GPU上分配和管理内存。
    thrust::device_vector<float> d_B(N, b); //分别创建三个device_vector，初始化d_A和d_B为值a和b，d_C为空。
    thrust::device_vector<float> d_C(N);

    float time;
    cudaEvent_t start, stop;  // 声明两个CUDA事件，用于记录GPU操作的开始和结束时间。
    cudaEventCreate(&start);  // 创建CUDA事件。
    cudaEventCreate(&stop);   
    cudaEventRecord(start, 0); // 记录操作开始的时间。
    add_float<<<iDivUp(N/4, BLOCKSIZE), BLOCKSIZE>>>(thrust::raw_pointer_cast(d_A.data()), thrust::raw_pointer_cast(d_B.data()), thrust::raw_pointer_cast(d_C.data()), N);
    cudaEventRecord(stop, 0);  // 记录操作结束的时间。
    cudaEventSynchronize(stop); // 等待事件完成，确保时间测量的准确性。
    cudaEventElapsedTime(&time, start, stop); // 计算从开始到结束的时间差。
    printf("float  Elapsed time:  %3.3f ms \n", time);  // 打印执行时间。
    gpuErrchk(cudaPeekAtLastError());    // 检查CUDA错误。
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    thrust::host_vector<float> h_float = d_C; // 将GPU上的device_vector复制到CPU上的host_vector。
    for (int i=0; i<N; i++) {
        if (h_float[i] != (a+b)) {
            printf("Error for add_float at %i: result is %f\n",i, h_float[i]);
            return -1;
        }
    }

    // --- float2

    thrust::device_vector<float> d_A2(N, a);
    thrust::device_vector<float> d_B2(N, b);
    thrust::device_vector<float> d_C2(N);

    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    add_float2<<<iDivUp(N/4, BLOCKSIZE), BLOCKSIZE>>>((float2*)thrust::raw_pointer_cast(d_A2.data()), (float2*)thrust::raw_pointer_cast(d_B2.data()), (float2*)thrust::raw_pointer_cast(d_C2.data()), N);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("float2 Elapsed time:  %3.3f ms \n", time); gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    thrust::host_vector<float> h_float2 = d_C2;
    for (int i=0; i<N; i++) {
        if (h_float2[i] != (a+b)) {
            printf("Error for add_float2 at %i: result is %f\n",i, h_float2[i]);
            return -1;
        }
    }

    // --- float4

    thrust::device_vector<float> d_A4(N, a);
    thrust::device_vector<float> d_B4(N, b);
    thrust::device_vector<float> d_C4(N);

    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    add_float4<<<iDivUp(N/4, BLOCKSIZE), BLOCKSIZE>>>((float4*)thrust::raw_pointer_cast(d_A4.data()), (float4*)thrust::raw_pointer_cast(d_B4.data()), (float4*)thrust::raw_pointer_cast(d_C4.data()), N);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("float4 Elapsed time:  %3.3f ms \n", time); gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    thrust::host_vector<float> h_float4 = d_C4;
    for (int i=0; i<N; i++) {
        if (h_float4[i] != (a+b)) {
            printf("Error for add_float4 at %i: result is %f\n",i, h_float4[i]);
            return -1;
        }
    }

    // ---test

    // thrust::device_vector<float> d_A4(N, a);
    // thrust::device_vector<float> d_B4(N, b);
    // thrust::device_vector<float> d_C4(N);

    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    add_float_test<<<iDivUp(N/4, BLOCKSIZE), BLOCKSIZE>>>((float_test*)thrust::raw_pointer_cast(d_A4.data()), (float_test*)thrust::raw_pointer_cast(d_B4.data()), (float_test*)thrust::raw_pointer_cast(d_C4.data()), N);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("float4 Elapsed time:  %3.3f ms \n", time); gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    // thrust::host_vector<float> h_float4 = d_C4;
    for (int i=0; i<N; i++) {
        if (h_float4[i] != (a+b)) {
            printf("Error for add_float4 at %i: result is %f\n",i, h_float4[i]);
            return -1;
        }
    }

    return 0;
}