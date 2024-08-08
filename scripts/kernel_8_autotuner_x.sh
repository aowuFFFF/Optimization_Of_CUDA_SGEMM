# --------------------------------------------------------------------
# 概述：
# 1. 本脚本用于生成 BM、BN、BK、TM 和 TN 的不同组合，并在指定的矩阵大小下评估其性能表现。
#    只需修改 `search_size` 变量，即可获得特定矩阵大小下的性能数据。
# 2. 运行时间将作为结果的一部分附加到输出文件的末尾。
#
# 运行方式：
#   ./kernel_8_autotuner_x.sh
#
# 输出文件：
#   结果将保存至 ./kernel_8_autotune_x.txt 中。
#
# 可视化结果：
#   在得到输出文件后，可以通过运行 kernel_8_plot_x.py 文件进行结果可视化，
#   并在终端中输出性能最佳的组合序列。
# --------------------------------------------------------------------


#!/usr/bin/env bash

set -u

# Define the range of values for each parameter
BK_VALUES=(8 16 32 64)
TM_VALUES=(4 8 16 32)
TN_VALUES=(4 8 16 32)
BM_VALUES=(64 128 256)
BN_VALUES=(64 128 256)
NUM_THREADS_VALUES=(256)

cd "$(dirname "$0")"
cd "../build"

RUNNER="../src/utils.cu"
KERNEL="../src/kernel/kernel_8.cuh"
OUTPUT="../scripts/kernel_8_autotune_x.txt"
search_size="4096"

# Clear the output file
echo "" > $OUTPUT

# Set GPU to use
export DEVICE="0"

TOTAL_CONFIGS="$(( ${#NUM_THREADS_VALUES[@]} * ${#BK_VALUES[@]} * ${#TM_VALUES[@]} * ${#TN_VALUES[@]} * ${#BM_VALUES[@]} * ${#BN_VALUES[@]} ))"
CONFIG_NUM=0

# 记录开始时间
start_time=$(date "+%Y-%m-%d %H:%M:%S")
echo "Script started at: $start_time" | tee -a $OUTPUT

# Loop through all combinations of parameters
for bk in ${BK_VALUES[@]}; do
  for tm in ${TM_VALUES[@]}; do
    for tn in ${TN_VALUES[@]}; do
      for bm in ${BM_VALUES[@]}; do
        for bn in ${BN_VALUES[@]}; do
          for nt in ${NUM_THREADS_VALUES[@]}; do
            echo ""
            CONFIG_NUM=$(( $CONFIG_NUM + 1 ))

            # skip configurations that don't fullfil preconditions
            config="BK=$bk TM=$tm TN=$tn BM=$bm BN=$bn NT=$nt"
            if [[ $(( ($nt * 4) % bk )) -ne 0 ]]; then
              echo "VECTORIZE: Skipping $config because (NUM_THREADS * 4) % BK = $(( ($nt * 4) % bk )) != 0))"
              continue
            fi
            if [[ $(( ($nt * 4) % bn )) -ne 0 ]]; then
              echo "VECTORIZE: Skipping $config because (NUM_THREADS * 4) % BN = $(( ($nt * 4) % bn )) != 0))"
              continue
            fi
            if [[ $(( $bn % (16 * $tn ) )) -ne 0 ]]; then
              echo "QUANTIZATION: Skipping $config because BN % (16 * TN) = $(( $bn % (16 * $tn ) )) != 0))"
              continue
            fi
            if [[ $(( $bm % (16 * $tm ) )) -ne 0 ]]; then
              echo "QUANTIZATION: Skipping $config because BM % (16 * TM) = $(( $bm % (16 * $tm ) )) != 0))"
              continue
            fi
            if [[ $(( ($bm * $bk) % ( 4 * $nt ) )) -ne 0 ]]; then
              echo "VECTORIZE: Skipping $config because (BM * BK) % (4 * NUM_THREADS) = $(( ($bm * $bk) % ( 4 * 256 ) )) != 0))"
              continue
            fi
            if [[ $(( ($bn * $bk) % ( 4 * $nt ) )) -ne 0 ]]; then
              echo "VECTORIZE: Skipping $config because (BN * BK) % (4 * NUM_THREADS) = $(( ($bn * $bk) % ( 4 * 256 ) )) != 0))"
              continue
            fi

            # Update the parameters in the source code
            sed -i "s/const uint K9_BK = .*/const uint K9_BK = $bk;/" $RUNNER
            sed -i "s/const uint K9_TM = .*/const uint K9_TM = $tm;/" $RUNNER
            sed -i "s/const uint K9_TN = .*/const uint K9_TN = $tn;/" $RUNNER
            sed -i "s/const uint K9_BM = .*/const uint K9_BM = $bm;/" $RUNNER
            sed -i "s/const uint K9_BN = .*/const uint K9_BN = $bn;/" $RUNNER
            sed -i "s/const int K9_NUM_THREADS = .*/const int K9_NUM_THREADS = $nt;/" $KERNEL
            
            # Rebuild the program
            make 

            # 单个kernel运行
            kernel_num=8
            # file_name="test_kernel_${kernel_num}.txt"
            # echo "8" |& tee -a $OUTPUT
            /home/ubuntu/yujie/NVIDIA_SGEMM_PRACTICE-master/sgemm 8 | grep "size: ($search_size)."  | tr '\n' ' ' |& tee -a $OUTPUT

            echo "($CONFIG_NUM/$TOTAL_CONFIGS): BK=$bk TM=$tm TN=$tn BM=$bm BN=$bn NUM_THREADS=$nt" |& tee -a $OUTPUT
            # Run the benchmark and get the result
            # Kill the program after 4 seconds if it doesn't finish
            # timeout -v 4 /home/ubuntu/yujie/NVIDIA_SGEMM_PRACTICE-master/sgemm 8 | tee -a $OUTPUT
          done
        done
      done
    done
  done
done

# 记录结束时间
end_time=$(date "+%Y-%m-%d %H:%M:%S")
echo "Script ended at: $end_time" | tee -a $OUTPUT

# 计算并输出执行时间
echo "Execution time: $(($(date -d "$end_time" +%s) - $(date -d "$start_time" +%s))) seconds" | tee -a $OUTPUT