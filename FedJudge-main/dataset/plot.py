# import pandas as pd

# import matplotlib.pyplot as plt
# import numpy as np
# import os

# # 读取 CSV 文件
# df = pd.read_csv('/home/chen/pyh/FedJudge-main/dataset/processed_file.csv', encoding='utf-8')

# # 重命名 run_adv 列为 μ
# df = df.rename(columns={'run_adv': 'μ'})

# # 清理数据：去掉 dataset 和 sensitive_feature 列中的空格和不可见字符
# # df['sensitive_feature'] = df['sensitive_feature'].str.strip().str.replace(r'\s+', '', regex=True)
# # df['dataset'] = df['dataset'].str.strip().str.replace(r'\s+', '', regex=True)

# # 设置保存图表的路径
# save_path = '/home/chen/pyh/FedJudge-main/dataset/pic'
# os.makedirs(save_path, exist_ok=True)  # 创建目录（如果不存在）
# # 定义归一化函数，并在归一化后除以 2
# def transform(data):
#     """
#     对输入的数据进行 Min-Max 归一化，缩放到 [0, 1] 范围，然后除以 2。
    
#     参数:
#     data (np.array 或 pd.Series): 需要处理的数据。
    
#     返回:
#     np.array: 归一化并除以 2 后的数据。
#     """
#     min_val = np.min(data)
#     max_val = np.max(data)
#     # 避免除以零的情况
#     if max_val == min_val:
#         return np.zeros_like(data)
#     normalized_data = (data - min_val) / (max_val - min_val)
#     return normalized_data / 2  # 归一化后除以 2
# # # 定义 log 变换
# # def transform_dpd(x):
# #     return -np.log2(x) / 10

# # def transform_deo(x):
# #     return -np.log2(x) / 10

# # 动态获取所有数据集
# datasets = {
#     dataset: {"data_size": 4000, "test_size": 0.4}  # 假设 data_size 和 test_size 固定
#     for dataset in df['dataset'].unique()
# }

# # 动态获取所有敏感特征
# sensitive_features = df['sensitive_feature'].unique().tolist()
# print("Adjusted datasets:", datasets)
# print("Adjusted sensitive features:", sensitive_features)

# # 遍历每个数据集
# for dataset_name, size_info in datasets.items():
#     # 过滤当前数据集的数据
#     dataset_data = df[df['dataset'] == dataset_name]
    
#     # 遍历每个敏感特征
#     for feature in sensitive_features:
#         # 过滤当前敏感特征的数据
#         feature_data = dataset_data[dataset_data['sensitive_feature'].str.lower() == feature.lower()].sort_values(by='μ')
        
#         # 如果数据为空，跳过绘图
#         if feature_data.empty:
#             print(f"No data found for dataset {dataset_name} and sensitive feature {feature}. Skipping plot.")
#             continue
        
#         # 分离 μ = -1 的数据，并将其改为 -0.1
#         special_point = feature_data[feature_data['μ'] == -0.1].copy()
#         # special_point['μ'] = -0.1
#         feature_data = feature_data[feature_data['μ'] != -0.1]
        
#         # 创建画布
#         plt.figure(figsize=(10, 6))
        
#         # 画 test_accuracy 折线
#         plt.plot(feature_data['μ'], feature_data['test_accuracy'], marker='o', label='Accuracy', linestyle='-')
        
#         # 画 M_dpd 折线（log 变换）
#         plt.plot(feature_data['μ'], transform(feature_data['M_dpd']), marker='s', label='Normalized(M_dpd) / 2', linestyle='-')
        
#         # 画 M_deo 折线（log 变换）
#         plt.plot(feature_data['μ'], transform(feature_data['M_deo']), marker='^', label='Normalized(M_deo) / 2', linestyle='-')
        
#         # 特殊点：μ = -0.1 用红色 x 标记
#         if not special_point.empty:
#             plt.scatter(special_point['μ'], special_point['test_accuracy'],
#                         color='blue', marker='x', s=100, label='μ = -0.1')
#             plt.scatter(special_point['μ'] + 0.005, transform(special_point['M_dpd']),  # 向右偏移
#                         color='orange', marker='x', s=100, label='μ = -0.1 (M_dpd)')
#             plt.scatter(special_point['μ'] - 0.005, transform(special_point['M_deo']),  # 向左偏移
#                         color='green', marker='x', s=100, label='μ = -0.1 (M_deo)')
        
#         # 调整横轴刻度
#         plt.xticks(np.append(feature_data['μ'], special_point['μ']))
        
#         # 添加标题和标签
#         plt.title(f'Dataset: {dataset_name}, Data Size: {size_info["data_size"]}, Test Size: {size_info["test_size"]}\nSensitive Feature: {feature}')
#         plt.xlabel('μ')  # 横轴标签改为 μ
#         plt.ylabel('Metrics')
#         # 调整图例位置，避免遮挡
#         plt.legend(loc='upper left', bbox_to_anchor=(1, 1))  # 将图例放在图表右侧
#         plt.grid(True)
#         # 调整布局，确保图例不会遮挡内容
#         plt.tight_layout()        
#         # 生成保存的文件名
#         # if dataset_name == "german_credit" and feature.lower() == "marital-status":
#         #     file_name = f'{dataset_name}_DataSize4000_TestRatio{size_info["test_size"]}_{feature}.png'
#         # else:
#         file_name = f'{dataset_name}_DataSize{size_info["data_size"]}_TestRatio{size_info["test_size"]}_{feature}.png'
#         plt.savefig(os.path.join(save_path, file_name))
#         plt.close()

# print("绘图完成，图表已保存至:", save_path)

import pandas as pd
import matplotlib.pyplot as plt
import os

# 读取 CSV 文件
df = pd.read_csv('/home/chen/pyh/FedJudge-main/dataset/processed_file.csv', encoding='utf-8')

# 设置保存图表的路径
save_path = '/home/chen/pyh/FedJudge-main/dataset/pic_tradeoff'  # 您的自定义路径
os.makedirs(save_path, exist_ok=True)  # 创建目录（如果不存在）

# 动态获取所有数据集和敏感特征
datasets = df['dataset'].unique()
sensitive_features = df['sensitive_feature'].unique()

# 遍历每个数据集和敏感特征
for dataset in datasets:
    for feature in sensitive_features:
        # 过滤当前数据集和敏感特征的数据
        data = df[(df['dataset'] == dataset) & (df['sensitive_feature'] == feature)]
        
        # 如果数据为空，跳过绘图
        if data.empty:
            print(f"No data found for dataset {dataset} and sensitive feature {feature}. Skipping plot.")
            continue
        
        # 按照 M_deo 排序
        data = data.sort_values(by='M_deo')
        
        # 创建画布
        plt.figure(figsize=(10, 6))
        
        # 绘制折线图
        plt.plot(data['M_deo'], data['test_accuracy'], marker='o', label='Accuracy vs M_deo', linestyle='-')
        
        # 添加标题和标签
        plt.title(f'Tradeoff: Accuracy vs M_deo\nDataset: {dataset}, Sensitive Feature: {feature}')
        plt.xlabel('M_deo')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        # 调整布局，确保图例不会遮挡内容
        plt.tight_layout()
        
        # 生成保存的文件名
        file_name = f'{dataset}_{feature}_tradeoff_line.png'
        plt.savefig(os.path.join(save_path, file_name))
        plt.close()

print("绘图完成，图表已保存至:", save_path)