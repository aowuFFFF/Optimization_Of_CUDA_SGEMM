
# 概述

面向NVIDIA GPU，使用CUDA编程逐步优化矩阵乘法

运算性能：
| Kernel | 描述 | GFLOPs/s | Performance relative to cuBLAS |
| --- | --- | --- | --- |
| 0: cuBLAS | 官方库函数 | 14220.4 | 100% |
| 1: Naive | 朴素实现 | 226.1 | 1.5% |
| 2: GMEM Coalescing | 全局内存合并 | 2270.7 | 16.0% |
| 3: SMEM Caching | 共享内存缓存 | 4247.5 | 29.9% |
| 4: 1D Blocktiling | 一维Thread Tile并行优化 | 8415.0 | 59.1% |
| 5: 2D Blocktiling | 二维Thread Tile并行优化 | 11629.7 | 81.9% |
| 6: Register cache SMEM | 使用寄存器缓存共享内存 | 11617.8 | 81.7% |
| 7: Vectorized Mem Access | FLOAT4向量访存 | 12638.8 | 88.2% |
| 8: Autotuning | 自动调整 | 13519.7 | 93.9% |
| 9:Double buffering | 双缓存 | 13696.8 | 96.3% |
| 10: Warptiling | warp分块 | 12142.5 | 85.4% |

> NVIDIA V100，矩阵尺寸4096
> 
# 配置

- NVIDIA CUDA version: `CUDA 11.8`；

# 目录

```
NVIDIA_SGEMM_PRACTICE                                   # 根目录
    ├── images                                          # 图片结果
    │   ├── kernel_x_vs_x.png
    │   └── kernel_culas_vs_x.png
    ├── test                                            # 测试结果
    │     ├── test_kernel_0.txt 
    │     ├── test_kernel_1.txt 
    │     └── test_kernel_x.txt 
    ├── scripts                                         # Autotuning 性能测试脚本
    │   ├── kernel_8_autotuner.sh
    │   ├── kernel_8_autotuner_x.sh
    │   └── kernel_8_plot_x.py
    └── src                                             # 源文件
    │    ├── kernel
    │    │  ├── kernel_1.cuh                            # 声明和定义
    │    │  ├── kernel_2.cuh
    │    │  └── kernel_x.cuh
    │    ├── kernel.cuh
    │    ├── utils.cuh                                  # 辅助函数
    │    └── utils.cu
    ├── plot_all.py                                     # 根据test结果绘制所有kernel性能对比图
    ├── plot.py                                         # 根据test结果绘制两个kernel性能对比图
    ├── run.sh                                          # 运行编译后可执行文件
    ├── sgemm.cu                                        # 主程序
    └── CMakeLists.txt                                  # 编译相关
```


# 运行

1. 编译
`cd build && cmake .. && make`
2. 运行run.sh，统计各个核函数计算效率，结果保存在test目录；
3. 计算效率折线绘图: 保存至images文件夹

> `python plot.py 0 1`表示绘制CUBLAS和kernel_1计算效率对比图；
> `python plot_all.py`表示绘制所有kernel计算效率对比图；

4. kernel_8 Autotuning脚本运行：

> `./kernel_9_autotuner.sh`表示输出所有合理的参数组合及其性能结果，具体使用方法见文件注释；

> `./kernel_9_autotuner_x.sh`表示输出所有合理的参数组合及 指定的矩阵大小 其性能结果，具体使用方法见文件注释；

> `./kernel_8_autotuner_plot.py`绘制指定的矩阵大小Autotuning结果图，并输出最大性能的参数组合，具体使用方法见文件注释；


# 参考链接

1、https://siboehm.com/articles/22/CUDA-MMM

2、https://github.com/siboehm/SGEMM_CUDA

3、https://github.com/wangzyon/NVIDIA_SGEMM_PRACTICE
