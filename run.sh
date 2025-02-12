#!/bin/bash

#全部kernel运行

rm ./test/test_kernel*
echo -n "test_kernel:"
for((i=0;i<=10;i++))
do
    echo -n "${i}..."
    file_name="./test/test_kernel_${i}.txt"
	./sgemm ${i} >> ${file_name}
done

# 单个kernel运行
# kernel_num=8
# file_name="test_kernel_${kernel_num}.txt"
# ./sgemm ${kernel_num} | tee ./test/${file_name}