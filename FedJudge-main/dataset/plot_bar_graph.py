# import matplotlib.pyplot as plt
# import numpy as np

# # 数据：任务的名字、Base、CAT 和 CAPO 的得分
# tasks = ['HARMLESS', 'MMLU', 'ARC-E', 'ARC-C', 'MT-BENCH', 'GCG', 'AUTODAN', 'PAIR']
# base_scores = [100, 38.9, 71.4, 41.5, 57.6, 70, 15, 0]
# cat_scores = [100, 38.3, 60.5, 39.8, 46.4, 17.5, 12.5, 0]
# capo_scores = [100, 37.5, 68.8, 37.1, 45.8, 12.5, 0, 0]

# # 创建条形图
# x = np.arange(len(tasks))  # 任务位置
# width = 0.25  # 每个条形的宽度

# fig, ax = plt.subplots(figsize=(10, 6))

# # 设置颜色
# base_color =  '#003366' #'#1f3a6e'  # 深蓝色
# # cat_color = '#98c9e8'   # 浅蓝色
# cat_color = '#99CC99' # '#85c1b1'    # 浅绿色
# capo_color = '#669966' #'#7b9b51'  # 绿色

# # 绘制 Base、CAT 和 CAPO 的条形
# bars1 = ax.bar(x - width, base_scores, width, label='Base', color=base_color)
# bars2 = ax.bar(x, cat_scores, width, label='CAT', color=cat_color)
# bars3 = ax.bar(x + width, capo_scores, width, label='CAPO', color=capo_color)
# # 添加标签、标题和自定义x轴
# ax.set_xlabel('Tasks')
# ax.set_ylabel('Score / ASR')
# ax.set_title('Comparison of Base, CAT, and CAPO Scores across Tasks')
# ax.set_xticks(x)
# ax.set_xticklabels(tasks, rotation=45, ha="right")

# # 添加数字标签
# for bars in [bars1, bars2, bars3]:
#     for bar in bars:
#         yval = bar.get_height()
#         ax.text(bar.get_x() + bar.get_width() / 2, yval + 1, f'{yval:.1f}', ha='center', va='bottom', rotation=45)

# # 添加字符串标签（Utility↑和ASR↓）
# ax.text(0.5, 110, 'Utility↑', fontsize=12, ha='center')
# ax.text(5.5, 110, 'ASR↓', fontsize=12, ha='center')

# # 添加图例
# ax.legend()

# # 显示图表
# plt.tight_layout()
# plt.show()



# save_path = '/home/chen/pyh/FedJudge-main/dataset/pic1'
# import os
# os.makedirs(save_path, exist_ok=True)  # 如果目录不存在，则创建
# file_path = os.path.join(save_path, 'chart.png')  # 拼接文件路径
# plt.savefig(file_path, bbox_inches='tight', dpi=300)  # 保存图表，设置分辨率为 300 DPI

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# 读取数据
df = pd.read_csv('/home/chen/pyh/FedJudge-main/dataset/processed_file.csv')

# 重命名 run_adv 列为 μ
df = df.rename(columns={'run_adv': 'μ'})

# 提取需要对比的数据对
pairs = []
for dataset in df['dataset'].unique():
    for feature in df[df['dataset'] == dataset]['sensitive_feature'].unique():
        pair = df[(df['dataset'] == dataset) & (df['sensitive_feature'] == feature)]
        if len(pair) == 2:  # 确保有 μ=0 和 μ≠0 的数据
            pairs.append(pair)

# 设置颜色
color_μ0 = '#003366'  # μ=0 的颜色
color_μ1 = '#99CC99'  # μ≠0 的颜色

# 创建柱状图
for pair in pairs:
    dataset = pair['dataset'].values[0]
    feature = pair['sensitive_feature'].values[0]
    
    # 提取数据
    μ0_data = pair[pair['μ'] == 0].iloc[0]
    μ1_data = pair[pair['μ'] != 0].iloc[0]
    
    # 创建柱状图
    x = np.arange(3)  # test_accuracy, M_deo, M_dpd
    width = 0.35  # 柱子的宽度
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 绘制 μ=0 的柱子
    bars1 = ax.bar(x - width/2, [μ0_data['test_accuracy'], μ0_data['M_deo'], μ0_data['M_dpd']], 
                   width, label=f'μ=0 (FedAvg)', color=color_μ0)
    
    # 绘制 μ≠0 的柱子
    bars2 = ax.bar(x + width/2, [μ1_data['test_accuracy'], μ1_data['M_deo'], μ1_data['M_dpd']], 
                   width, label=f'μ={μ1_data["μ"]} (FedJustice)', color=color_μ1)
    
    # 添加标签、标题和自定义x轴
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Value')
    ax.set_title(f'{dataset} - {feature}')
    ax.set_xticks(x)
    ax.set_xticklabels(['test_accuracy', 'M_deo', 'M_dpd'])
    
    # 在横轴下方标注 μ 的值
    for i in range(3):
        ax.text(i - width/2, -0.1, f'μ=0', ha='center', va='top')
        ax.text(i + width/2, -0.1, f'μ={μ1_data["μ"]}', ha='center', va='top')
    
    # 在柱子顶部标注具体数值
    # 在柱子顶部标注具体数值（斜着写）
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 0.02, 
                    f'{height:.3f}', ha='center', va='bottom', rotation=45)  # 添加 rotation=45 使文字斜着写
    
    # 添加图例
    ax.legend()
    
    # 显示图表
    plt.tight_layout()
    plt.show()
    
    # 保存图表
    save_path = '/home/chen/pyh/FedJudge-main/dataset/pic1'
    os.makedirs(save_path, exist_ok=True)  # 如果目录不存在，则创建
    file_path = os.path.join(save_path, f'{dataset}_{feature}_chart.png')  # 拼接文件路径
    plt.savefig(file_path, bbox_inches='tight', dpi=300)  # 保存图表，设置分辨率为 300 DPI