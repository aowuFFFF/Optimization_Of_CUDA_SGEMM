# --------------------------------------------------------------------
# 概述：
# 1. 本 Python 脚本用于生成性能折线图，展示 kernel_8_autotune_x.txt 文件的性能表现。
# 2. 最优的性能组合将在终端中输出。
#
# 运行方式：
#   python kernel_8_autotune_x.txt
#
# 输出文件：
#   结果将保存为 kernel_8_combinations_visualization.png。
#
# --------------------------------------------------------------------

import os
import re
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import argparse


file_path = '/home/ubuntu/yujie/NVIDIA_SGEMM_PRACTICE-master/scripts/kernel_8_autotune_x.txt'  # 替换为你的文件路径

# 正则表达式模式
performance_pattern = r"performance: \((.*?)\) GFLOPS\."
x_axis_pattern = r"size: \((\d+)\). \((\d+)/576\):"

# 初始化存储数据的列表
performances = []
x_axis_values = []  # 使用分子作为x轴数据


# 我想匹配下面的443：size: (4096). (443/576):
# 读取文件并提取数据
with open(file_path, 'r') as file:
    for line in file:
        performance_match = re.search(performance_pattern, line)
        x_axis_match = re.search(x_axis_pattern, line)
        
        if performance_match:
            performance = float(performance_match.group(1))
            performances.append(performance)
        
        if x_axis_match:
            x_value = int(x_axis_match.group(2))  # 提取分子
            x_axis_values.append(x_value)


# 使用len()函数获取列表的大小
list_size1 = len(performances)
list_size2 = len(x_axis_values)

# 打印列表的大小
# print("The size of the performances is:", list_size1)
# print("The size of the x_axis_values is:", list_size2)

# 找到最大性能值及其索引
max_performance = max(performances)
max_performance_index = performances.index(max_performance)
max_x_value = x_axis_values[max_performance_index]



# 在图表上突出显示最大性能点
plt.scatter(max_x_value, max_performance, color='red', s=200, marker='*')  # 使用红色星号标记

# 添加文本标签显示最大性能值
plt.text(max_x_value, max_performance, f'Max: {max_performance:.2f} GFLOPS', fontsize=14, ha='center', va='bottom', color='white')

# 使用annotate添加箭头指向最大性能点
plt.annotate('', xy=(max_x_value, max_performance), xytext=(max_x_value, max_performance - 20),
             arrowprops=dict(facecolor='black', arrowstyle='->', connectionstyle='arc3'))

# 在终端打印最大性能值及其对应的参数组合的值
# print(f"Maximum Performance: {max_performance} GFLOPS at X-Axis Value {max_x_value}")
keyword = str(max_performance)
with open(file_path, 'r', encoding='utf-8') as file:
    for line_number, line in enumerate(file, start=1):
        if keyword in line:
            print(f"Maximum Performance : {line.strip()}")


# 绘制折线图
plt.figure(figsize=(30, 15))
plt.plot(x_axis_values, performances, marker='o')  # 使用分子作为x轴数据
plt.title('Performance vs X-Axis Value')
plt.xlabel('X-Axis Value (Molecular Part)')
plt.ylabel('Performance (GFLOPS)')
plt.grid(True)
# plt.show()


# 保存图表到文件
save_path = './kernel_8_combinations_visualization.png'  # 替换为您想要保存图表的路径
plt.savefig(save_path)