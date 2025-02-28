import pandas as pd

# 输入和输出文件路径
input_file = "/home/chen/pyh/FedJudge-main/dataset/adult.csv"  # 请确保 adult.csv 文件在当前目录下，或者填写完整路径
output_file = "/home/chen/pyh/FedJudge-main/dataset/test_adult.csv"

try:
    # 读取 CSV 文件
    df = pd.read_csv(input_file)

    # 检查是否有足够的行数
    if len(df) < 100:
        print(f"警告：文件 {input_file} 中的行数少于 100 行，将提取所有行。")

    # 提取倒数 100 行数据
    last_100_rows = df.tail(2000)

    # 保存为新的 CSV 文件，指定编码方式为 utf-8
    last_100_rows.to_csv(output_file, index=False, encoding="utf-8")

    print(f"新文件已成功保存为：{output_file}，编码方式为 utf-8")

except FileNotFoundError:
    print(f"错误：无法找到文件 {input_file}，请检查文件路径是否正确。")

except pd.errors.EmptyDataError:
    print(f"错误：文件 {input_file} 为空，请检查文件内容。")

except Exception as e:
    print(f"发生错误：{e}")

