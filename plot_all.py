# 绘制所有kernel性能对比
# 指令：python plot_all.py

import os
import re
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

def parse_file(file):
    with open(file, 'r') as f:
        lines = [line.strip() for line in f.readlines()]

    data = []
    pattern = r"Average elasped time: \((.*?)\) second, performance: \((.*?)\) GFLOPS. size: \((.*?)\)."
    for line in lines:
        r = re.match(pattern, line)
        if r:
            gflops = float(r.group(2))
            data.append(gflops)
    return data

def plot(datas, save_dir):
    x = [(i + 1) * 256 for i in range(len(datas[0]))]
    fig = plt.figure(figsize=(18, 10))

    # 定义颜色和标记样式
    colors = ['k', 'b', 'g', 'r', 'c', 'm', 'y', 'orange', 'purple', 'pink', 'brown']
    markers = ['s', '^', 'o', 'D', 'x', '+', '*', 'P', '<', '>', 'H']

    # 绘制每个数据集
    for i, (data, color, marker) in enumerate(zip(datas, colors, markers)):
        label = f"kernel_{i}"  # 使用i作为标签
        plt.plot(x, data, c=color, linewidth=2, label=label)
        plt.scatter(x, data, marker=marker, s=60, c='', edgecolors=color, linewidth=2)

    plt.legend()
    plt.tick_params(labelsize=10)
    plt.xlabel("Matrix size (M=N=K)", fontsize=12, fontweight='bold')
    plt.ylabel("Performance (GFLOPS)", fontsize=12, fontweight='bold')
    plt.title("Comparison of kernels", fontsize=16, fontweight='bold')

    x_major_locator = MultipleLocator(256)
    plt.gca().xaxis.set_major_locator(x_major_locator)

    # 使用固定地址保存图像
    plt.savefig(os.path.join(save_dir, 'all_kernels_comparison.png'))
    plt.show()  # 显示图像

def main():
    root = os.path.dirname(os.path.abspath(__file__))

    # 固定保存地址
    save_dir = os.path.join(root, 'images/')  # 可以根据需要修改路径

    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)

    datas = []
    for i in range(11):  # 修改为读取11个数据文件
        data = parse_file(os.path.join(root, f'test/test_kernel_{i}.txt'))
        datas.append(data)

    plot(datas, save_dir)

if __name__ == "__main__":
    main()

