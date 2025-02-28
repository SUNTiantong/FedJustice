import pandas as pd

# 读取 CSV 文件
# df = pd.read_csv('/home/chen/pyh/FedJudge-main/dataset/mu1_result.csv')
df = pd.read_csv('/home/chen/pyh/FedJudge-main/dataset/processed_file.csv')
# ！!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# 保留所需的列
df = df[['dataset', 'sensitive_feature','run_adv', 'test_accuracy', 'M_deo']]
# 1. 将列名 'run_adv' 改为 'Method'
df = df.rename(columns={'run_adv': 'Method'})

# 2. 替换列中的值
df['Method'] = df['Method'].replace({
    -2.0: 'FedProx',
    -3.0: 'FedSGD'
})
# 定义一个函数来格式化数字
def format_number(x):
    if x < 1e-4:
        return f"{x:.1e}"  # 保留一位有效数字，使用科学计数法
    else:
        return f"{x:.4g}"  # 保留四位有效数字

# 应用格式化函数到 M_deo 和 M_dpd 列
df['M_deo'] = df['M_deo'].apply(lambda x: format_number(x))
# df['M_dpd'] = df['M_dpd'].apply(lambda x: format_number(x))

# 保存结果到新的 CSV 文件
df.to_csv('/home/chen/pyh/FedJudge-main/dataset/processed_file.csv', index=False)

# import pandas as pd

# # 读取CSV文件
# file_path = '/home/chen/pyh/FedJudge-main/dataset/mu1_result.csv'
# df = pd.read_csv(file_path)

# # 按dataset和sensitive_feature排序
# df_sorted = df.sort_values(by=['dataset', 'sensitive_feature'])

# # 将排序后的数据覆盖写回原文件
# df_sorted.to_csv(file_path, index=False)

# print(f"数据已整理并覆盖保存到原文件 {file_path}")