import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
import json
# import numpy as np
class DatasetPartition:
    """This class is used to divide the dataset for training between inference section and training one.
    The Cross-validation method used is 4 Stratification based on Holdout/ShuffleSplit"""

    def __init__(self, df: pd.DataFrame, split_test=True, split_percentage=0.9):
        """
        Constructor is build to split dataset and then reorder the dataset
        :param df: dataset to be split into validation set and training set
        :param split_test: if True StrafifiedShuffle is applied otherwise split x from y
        :param split_percentage: declared in case of necessity, not used
        """
        self.df = df
        self.split_test = split_test

        if self.split_test:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=1 - split_percentage, random_state=0)
            training_index, testing_index = list(sss.split(self.df.iloc[:, :self.df.shape[1] - 1].values,
                                                           self.df.iloc[:, self.df.shape[1] - 1:self.df.shape[1]].values))[
                0]
#training_index 900个，testing_index 100个
            self.x_training = self.df.iloc[training_index, :self.df.shape[1] ].values
            # self.y_training = self.df.iloc[training_index, self.df.shape[1] - 1:self.df.shape[1]].values
            self.x_testing = self.df.iloc[testing_index, :self.df.shape[1] ].values
            # self.y_testing = self.df.iloc[testing_index, self.df.shape[1] - 1:self.df.shape[1]].values

        else:
            self.x_training = self.df.iloc[:, :self.df.shape[1] ].values
            # self.y_training = self.df.iloc[:, self.df.shape[1] - 1:self.df.shape[1]].values

        # MANUAL Holdout
        # tr_size = int(df.shape[0] * split_percentage)
        # self.df_training, self.df_testing = np.split(df, [tr_size], axis=0)
        # self.x_training = self.df_training.iloc[:, :len(self.df_training.keys()) - 1].values
        # self.y_training = self.df_training.iloc[:, len(self.df_training.keys()) - 1:len(self.df_training.keys())].values
        # self.x_testing = self.df_testing.iloc[:, :len(self.df_testing.keys()) - 1].values
        # self.y_testing = self.df_testing.iloc[:, len(self.df_testing.keys()) - 1:len(self.df_testing.keys())].values

    def split(self):
        """
        If split constructor parameter is True then testing is also returned otherwise it's None
        :return: x_training, y_training, x_testing, y_testing
        """
        # return self.x_training, self.y_training, \
        #        self.x_testing if self.split_test else None, self.y_testing if self.split_test else None
        return self.x_training, \
               self.x_testing if self.split_test else None



# 定义一个 dotdict 类，方便访问嵌套字典中的元素
class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    def __getattr__(self, name):
        value = self.get(name)
        if isinstance(value, dict):
            return dotdict(value)
        return value

def fetch_adult_data(df):
    """
    Simulates fetch_ucirepo for a locally stored CSV file (adult.csv).
    
    Parameters:
        file_path (str): Path to the local CSV file.
        
    Returns:
        result (dotdict): object containing dataset metadata, dataframes, and variable info.
    """
    
    df.columns = df.columns.str.replace('_', '-')
    # 将 'gender' 列重命名为 'sex'
    df.rename(columns={'gender': 'sex'}, inplace=True)

    # 假设一些默认的变量信息（根据实际数据调整）
    variables = [
        {'name': 'age', 'role': 'Feature'},
        {'name': 'workclass', 'role': 'Feature'},
        {'name': 'fnlwgt', 'role': 'Feature'},
        {'name': 'education', 'role': 'Feature'},
        {'name': 'educational-num', 'role': 'Feature'},
        {'name': 'marital-status', 'role': 'Feature'},
        {'name': 'occupation', 'role': 'Feature'},
        {'name': 'relationship', 'role': 'Feature'},
        {'name': 'race', 'role': 'Feature'},
        {'name': 'sex', 'role': 'Feature'},  # 更新为 'sex'
        {'name': 'capital-gain', 'role': 'Feature'},
        {'name': 'capital-loss', 'role': 'Feature'},
        {'name': 'hours-per-week', 'role': 'Feature'},
        {'name': 'native-country', 'role': 'Feature'},
        {'name': 'income', 'role': 'Target'}
    ]
    # 根据 role 分类变量
    variables_by_role = {
        'ID': [],
        'Feature': [],
        'Target': [],
        'Other': []
    }

    for variable in variables:
        if variable['role'] not in variables_by_role:
            raise ValueError('Role must be one of "ID", "Feature", "Target", or "Other"')
        variables_by_role[variable['role']].append(variable['name'])

    # 提取数据集中的 ID、特征和目标
    ids_df = df[variables_by_role['ID']] if len(variables_by_role['ID']) > 0 else None
    features_df = df[variables_by_role['Feature']] if len(variables_by_role['Feature']) > 0 else None
    targets_df = df[variables_by_role['Target']] if len(variables_by_role['Target']) > 0 else None

    # 将所有数据放入一个字典中
    data = {
        'ids': ids_df,
        'features': features_df,
        'targets': targets_df,
        'original': df,
        'headers': df.columns,
    }

    # 将变量信息转换为 DataFrame，便于查看
    variables_df = pd.DataFrame.from_records(variables)

    # 构建元数据字典
    metadata = {
        'uci_id': 'adult',  # 设置一个假设的 ID
        'name': 'Adult Dataset',
        'data_url': None,  # 本地没有 URL
        'variables': variables,
        'additional_info': None,
        'intro_paper': None
    }

    # 生成最终的结果字典
    result = {
        'data': dotdict(data),
        'metadata': dotdict(metadata),
        'variables': variables_df
    }

    # 返回结果对象，使用 dotdict 方便通过点访问数据
    return dotdict(result)




def fetch_german_data(df):
    """
    读取本地 CSV 文件并包装成字典，方便访问
    """


    # 假设目标列为 "Target"，根据需要对其他列做处理
    # df.columns = df.columns.str.replace(' ', '_')  # 替换空格为下划线，方便后续访问

    variables = [
        {'name': 'Checking account', 'role': 'Feature'},
        {'name': 'Duration', 'role': 'Feature'},
        {'name': 'Credit history', 'role': 'Feature'},
        {'name': 'Purpose', 'role': 'Feature'},
        {'name': 'Credit amount', 'role': 'Feature'},
        {'name': 'Savings account', 'role': 'Feature'},
        {'name': 'Employment since', 'role': 'Feature'},
        {'name': 'Installment rate', 'role': 'Feature'},
        {'name': 'sex', 'role': 'Feature'},
        {'name': 'marital-status', 'role': 'Feature'},
        {'name': 'Debtors & guarantors', 'role': 'Feature'},
        {'name': 'Residence since', 'role': 'Feature'},
        {'name': 'Property', 'role': 'Feature'},
        {'name': 'Age', 'role': 'Feature'},
        {'name': 'Other installments', 'role': 'Feature'},
        {'name': 'Housing', 'role': 'Feature'},
        {'name': 'Existing credits', 'role': 'Feature'},
        {'name': 'Job', 'role': 'Feature'},
        {'name': 'Kept people', 'role': 'Feature'},
        {'name': 'Phone', 'role': 'Feature'},
        {'name': 'Foreign-worker', 'role': 'Feature'},
        {'name': 'Target', 'role': 'Target'}
    ]

    # 分类变量
    variables_by_role = {
        'ID': [],
        'Feature': [],
        'Target': [],
        'Other': []
    }

    for variable in variables:
        if variable['role'] not in variables_by_role:
            raise ValueError('Role must be one of "ID", "Feature", "Target", or "Other"')
        variables_by_role[variable['role']].append(variable['name'])

    # 提取数据集中的 ID、特征和目标
    ids_df = df[variables_by_role['ID']] if len(variables_by_role['ID']) > 0 else None
    features_df = df[variables_by_role['Feature']] if len(variables_by_role['Feature']) > 0 else None
    targets_df = df[variables_by_role['Target']] if len(variables_by_role['Target']) > 0 else None

    # 将所有数据放入一个字典中
    data = {
        'ids': ids_df,
        'features': features_df,
        'targets': targets_df,
        'original': df,
        'headers': df.columns,
    }

    # 将变量信息转换为 DataFrame，便于查看
    variables_df = pd.DataFrame.from_records(variables)

    # 构建元数据字典
    metadata = {
        'uci_id': 'example',  # 假设一个 UCI 数据集 ID
        'name': 'Custom Dataset',
        'data_url': None,  # 本地文件路径
        'variables': variables,
        'additional_info': None,
        'intro_paper': None
    }

    # 生成最终的结果字典
    result = {
        'data': dotdict(data),
        'metadata': dotdict(metadata),
        'variables': variables_df
    }

    # 返回结果对象，使用 dotdict 方便通过点访问数据
    return dotdict(result)


def fetch_bank_marketing_data(df):
    """
    Simulates fetch_ucirepo for a locally stored CSV file (bank_marketing.csv).
    
    Parameters:
        df (DataFrame): The dataframe containing the Bank Marketing dataset.
        
    Returns:
        result (dotdict): object containing dataset metadata, dataframes, and variable info.
    """

    path_to_save = './dataset/refined_bank_marketing.csv'
    import os
    # 检查文件夹是否存在，不存在则创建
    if not os.path.exists(path_to_save):
        # 去掉引号，如果是读取文件时已经做了此步骤，这里可以跳过
        # df = pd.read_csv('bank_marketing.csv', quotechar='"')

        # 修改列名：将 'marital' 改为 'marital-status'，'y' 改为 'Target'
        df = df.rename(columns={'marital': 'marital-status', 'y': 'Target'})

        # 将 'Target' 列中的 'no' 替换为 0，'yes' 替换为 1
        df['Target'] = df['Target'].map({'no': 0, 'yes': 1}).astype(int)

        # 清理列名，替换可能存在的下划线（或其他字符）
        df.columns = df.columns.str.replace('_', '-')
        # 保存到本地
        df.to_csv(path_to_save, index=False)

    # 读取本地 CSV 文件
    df = pd.read_csv(path_to_save)
    
    # 假设一些默认的变量信息（根据实际数据调整）
    variables = [
        {'name': 'age', 'role': 'Feature'},
        {'name': 'job', 'role': 'Feature'},
        {'name': 'marital-status', 'role': 'Feature'},
        {'name': 'education', 'role': 'Feature'},
        {'name': 'default', 'role': 'Feature'},
        {'name': 'balance', 'role': 'Feature'},
        {'name': 'housing', 'role': 'Feature'},
        {'name': 'loan', 'role': 'Feature'},
        {'name': 'contact', 'role': 'Feature'},
        {'name': 'day', 'role': 'Feature'},
        {'name': 'month', 'role': 'Feature'},
        {'name': 'duration', 'role': 'Feature'},
        {'name': 'campaign', 'role': 'Feature'},
        {'name': 'pdays', 'role': 'Feature'},
        {'name': 'previous', 'role': 'Feature'},
        {'name': 'poutcome', 'role': 'Feature'},
        {'name': 'Target', 'role': 'Target'}  # 目标变量是 'y'，重命名为 'Target'
    ]
    
    # 根据 role 分类变量
    variables_by_role = {
        'ID': [],
        'Feature': [],
        'Target': [],
        'Other': []
    }

    for variable in variables:
        if variable['role'] not in variables_by_role:
            raise ValueError('Role must be one of "ID", "Feature", "Target", or "Other"')
        variables_by_role[variable['role']].append(variable['name'])

    # 提取数据集中的 ID、特征和目标
    ids_df = df[variables_by_role['ID']] if len(variables_by_role['ID']) > 0 else None
    features_df = df[variables_by_role['Feature']] if len(variables_by_role['Feature']) > 0 else None
    targets_df = df[variables_by_role['Target']] if len(variables_by_role['Target']) > 0 else None

    # 将所有数据放入一个字典中
    data = {
        'ids': ids_df,
        'features': features_df,
        'targets': targets_df,
        'original': df,
        'headers': df.columns,
    }

    # 将变量信息转换为 DataFrame，便于查看
    variables_df = pd.DataFrame.from_records(variables)

    # 构建元数据字典
    metadata = {
        'uci_id': 'bank_marketing',  # 设置一个假设的 ID
        'name': 'Bank Marketing Dataset',
        'data_url': None,  # 本地没有 URL
        'variables': variables,
        'additional_info': None,
        'intro_paper': None
    }

    # 生成最终的结果字典
    result = {
        'data': dotdict(data),
        'metadata': dotdict(metadata),
        'variables': variables_df
    }

    # 返回结果对象，使用 dotdict 方便通过点访问数据
    return dotdict(result)


def fetch_compas_data(df):
    """
    Simulates fetch_ucirepo for a locally stored CSV file (compas.csv).
    
    Parameters:
        df (DataFrame): The dataframe containing the Bank Marketing dataset.
        
    Returns:
        result (dotdict): object containing dataset metadata, dataframes, and variable info.
    """

    path_to_save = '/home/chen/pyh/FedJudge-main/dataset/refined_compas.csv'
    # absolute_path = os.path.abspath(path_to_save)
    import os
    # 检查文件夹是否存在，不存在则创建
    if not os.path.exists(path_to_save):
        # 修改列名:'two_year_recid' 改为 'Target'
        df = df.rename(columns={ 'two_year_recid': 'Target'})

        # 保存到本地
        df.to_csv(path_to_save, index=False)

    # 读取本地 CSV 文件
    df = pd.read_csv(path_to_save)
#53列
    # 假设一些默认的变量信息（根据实际数据调整）
    variables = [
        {'name': 'id', 'role': 'Feature'},
        {'name': 'name', 'role': 'Feature'},
        {'name': 'first', 'role': 'Feature'},
        {'name': 'last', 'role': 'Feature'},
        {'name': 'compas_screening_date', 'role': 'Feature'},
        {'name': 'sex', 'role': 'Feature'},
        {'name': 'dob', 'role': 'Feature'},
        {'name': 'age', 'role': 'Feature'},
        {'name': 'age_cat', 'role': 'Feature'},
        {'name': 'race', 'role': 'Feature'},
        {'name': 'juv_fel_count', 'role': 'Feature'},
        {'name': 'decile_score', 'role': 'Feature'},
        {'name': 'juv_misd_count', 'role': 'Feature'},
        {'name': 'juv_other_count', 'role': 'Feature'},
        {'name': 'priors_count', 'role': 'Feature'},
        {'name': 'days_b_screening_arrest', 'role': 'Feature'},
        {'name': 'c_jail_in', 'role': 'Feature'},
        {'name': 'c_jail_out', 'role': 'Feature'},
        {'name': 'c_case_number', 'role': 'Feature'},
        {'name': 'c_offense_date', 'role': 'Feature'},
        {'name': 'c_arrest_date', 'role': 'Feature'},
        {'name': 'c_days_from_compas', 'role': 'Feature'},
        {'name': 'c_charge_degree', 'role': 'Feature'},
        {'name': 'c_charge_desc', 'role': 'Feature'},
        {'name': 'is_recid', 'role': 'Feature'},
        {'name': 'r_case_number', 'role': 'Feature'},
        {'name': 'r_charge_degree', 'role': 'Feature'},
        {'name': 'r_days_from_arrest', 'role': 'Feature'},
        {'name': 'r_offense_date', 'role': 'Feature'},
        {'name': 'r_charge_desc', 'role': 'Feature'},
        {'name': 'r_jail_in', 'role': 'Feature'},
        {'name': 'r_jail_out', 'role': 'Feature'},
        {'name': 'violent_recid', 'role': 'Feature'},
        {'name': 'is_violent_recid', 'role': 'Feature'},
        {'name': 'vr_case_number', 'role': 'Feature'},
        {'name': 'vr_charge_degree', 'role': 'Feature'},
        {'name': 'vr_offense_date', 'role': 'Feature'},
        {'name': 'vr_charge_desc', 'role': 'Feature'},
        {'name': 'type_of_assessment', 'role': 'Feature'},
        {'name': 'decile_score', 'role': 'Feature'},
        {'name': 'score_text', 'role': 'Feature'},
        {'name': 'screening_date', 'role': 'Feature'},
        {'name': 'v_type_of_assessment', 'role': 'Feature'},
        {'name': 'v_decile_score', 'role': 'Feature'},
        {'name': 'v_score_text', 'role': 'Feature'},
        {'name': 'v_screening_date', 'role': 'Feature'},
        {'name': 'in_custody', 'role': 'Feature'},
        {'name': 'out_custody', 'role': 'Feature'},
        {'name': 'priors_count', 'role': 'Feature'},
        {'name': 'start', 'role': 'Feature'},
        {'name': 'end', 'role': 'Feature'},
        {'name': 'event', 'role': 'Feature'},
        {'name': 'Target', 'role': 'Target'}  # 目标变量
    ]


    
    # 根据 role 分类变量
    variables_by_role = {
        'ID': [],
        'Feature': [],
        'Target': [],
        'Other': []
    }

    for variable in variables:
        if variable['role'] not in variables_by_role:
            raise ValueError('Role must be one of "ID", "Feature", "Target", or "Other"')
        variables_by_role[variable['role']].append(variable['name'])

    # 提取数据集中的 ID、特征和目标
    ids_df = df[variables_by_role['ID']] if len(variables_by_role['ID']) > 0 else None
    features_df = df[variables_by_role['Feature']] if len(variables_by_role['Feature']) > 0 else None
    targets_df = df[variables_by_role['Target']] if len(variables_by_role['Target']) > 0 else None

    # 将所有数据放入一个字典中
    data = {
        'ids': ids_df,
        'features': features_df,
        'targets': targets_df,
        'original': df,
        'headers': df.columns,
    }

    # 将变量信息转换为 DataFrame，便于查看
    variables_df = pd.DataFrame.from_records(variables)

    # 构建元数据字典
    metadata = {
        'uci_id': 'bank_marketing',  # 设置一个假设的 ID
        'name': 'Bank Marketing Dataset',
        'data_url': None,  # 本地没有 URL
        'variables': variables,
        'additional_info': None,
        'intro_paper': None
    }

    # 生成最终的结果字典
    result = {
        'data': dotdict(data),
        'metadata': dotdict(metadata),
        'variables': variables_df
    }

    # 返回结果对象，使用 dotdict 方便通过点访问数据
    return dotdict(result)

def fetch_dict_from_df(data,file_path=None):
    # 读取本地的 CSV 文件

    if data=='adult':
        num_samples=4000
        file_path="/home/chen/pyh/FedJudge-main/dataset/adult.csv" #if file_path is None else file_path
        df = pd.read_csv(file_path)
        dict =fetch_adult_data(df)

    elif data=='german_credit':
        num_samples=1000
        file_path="/home/chen/pyh/FedJudge-main/dataset/german_credit.csv"
        df = pd.read_csv(file_path)
        dict =fetch_german_data(df)

    elif data == 'bank_marketing':
        num_samples = 1000
        file_path = "/home/chen/pyh/FedJudge-main/dataset/bank_marketing.csv" if file_path is None else file_path
        df = pd.read_csv(file_path, delimiter=';')  # 注意 CSV 文件使用分号（;）分隔
        dict = fetch_bank_marketing_data(df)
    elif data == 'compas':
        num_samples = 1000  # 你可以根据实际情况调整样本数
        file_path = "/home/chen/pyh/FedJudge-main/dataset/compas-scores-two-years.csv" if file_path is None else file_path
        df = pd.read_csv(file_path)
        dict = fetch_compas_data(df)
    else:
        raise ValueError(f"Unsupported dataset: {data}. Please choose 'adult' or 'german_credit'.")
    return dict

# a=fetch_dict_from_df("german_credit")
# print(a)