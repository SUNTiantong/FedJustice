import json
from loguru import logger
from torch.utils.data import Dataset, DataLoader
from ucimlrepo import fetch_ucirepo
import numpy as np
import pandas as pd
from torchvision import datasets, transforms
from component.local_preprocess import fetch_dict_from_df


def random_except(low,high,except_num):
    while True:
        i = np.random.randint(low,high)
        if i != except_num:
            return i

# fetch dataset

class AdultDataset(Dataset):
    def __init__(self,few_shot,dataset_type,number_of_samples=None,fairness=None,fairprompt=False,feature_to_process='sex'):
        
        # adult = pd.read_csv(dataset_path)
        adult = fetch_dict_from_df(data='adult')
        X = adult.data.features
        y = (adult.data.targets == ">50K").astype(int)

        if number_of_samples is not None:
            self.number_of_samples = number_of_samples
        else:
            self.number_of_samples = len(X)
            number_of_samples = len(X)

        self.fairprompt  = False

        if fairness is not None:
            #either one of fairness or fairprompt should be taken as true.
            X["income"] = y
            x_0 = int(number_of_samples/2*(1+fairness)/2)
            x_1 = int(max(number_of_samples/2 - x_0,0))
            x_2 = int(number_of_samples/2*(1-fairness)/2)
            x_3 = int(max(number_of_samples/2 - x_2,0))
            if feature_to_process == "sex":
                X = pd.concat(
                    [X[(X["income"] ==1) & (X["sex"] == "Male")].sample(n=x_0,replace=True),
                    X[(X["income"] ==1) & (X["sex"] == "Female")].sample(n=x_1,replace=True),
                    X[(X["income"] ==0) & (X["sex"] == "Male")].sample(n=x_2,replace=True),
                    X[(X["income"] ==0 )& (X["sex"] == "Female")].sample(n=x_3,replace=True)])
            elif feature_to_process == "race":
                X = pd.concat(
                    [X[(X["income"] == 1) & (X["race"] == "White")].sample(n=x_0, replace=True),
                     X[(X["income"] == 1) & (X["race"] != "White")].sample(n=x_1, replace=True),
                     X[(X["income"] == 0) & (X["race"] == "White")].sample(n=x_2, replace=True),
                     X[(X["income"] == 0) & (X["race"] != "White")].sample(n=x_3, replace=True)]
                )
            elif feature_to_process == "marital-status":
                # 将marital-status的特征拆分为Never-married和其他
                X = pd.concat(
                    [X[(X["income"] == 1) & (X["marital-status"] == "Never-married")].sample(n=x_0, replace=True),
                    X[(X["income"] == 1) & (X["marital-status"] != "Never-married")].sample(n=x_1, replace=True),
                    X[(X["income"] == 0) & (X["marital-status"] == "Never-married")].sample(n=x_2, replace=True),
                    X[(X["income"] == 0) & (X["marital-status"] != "Never-married")].sample(n=x_3, replace=True)]
                )
            # elif feature_to_process == "marital-status":
            #     # 将marital-status的特征拆分为Never-married和其他
            #     X = pd.concat(
            #         [X[(X["income"] == 1) & (X["marital-status"] == "Married-civ-spouse")].sample(n=x_0, replace=True),
            #         X[(X["income"] == 1) & (X["marital-status"] == "Never-married")].sample(n=x_1, replace=True),
            #         X[(X["income"] == 0) & (X["marital-status"] == "Married-civ-spouse")].sample(n=x_2, replace=True),
            #         X[(X["income"] == 0) & (X["marital-status"] == "Never-married")].sample(n=x_3, replace=True)]
            #     )
            else:
                raise ValueError("Invalid feature_to_process. Use 'sex' or 'race'.")
            #shuffle
            X = X.sample(frac=1)
            X = X.reset_index(drop=True)
            self.y = X["income"]
            X.drop(columns=["income"],inplace=True)
            self.X =  X

        else:
            X["income"] = y
            X = X.sample(frac=1)
            X = X.reset_index(drop=True)
            self.y = X["income"]
            X.drop(columns=["income"],inplace=True)
            self.X =  X
            self.fairprompt = fairprompt

        # 设置需要处理的特征
        self.features = X.columns.tolist()
        #要与下面的prompts设置对应
        # self.features = ["age","workclass","education","educational-num","marital-status","occupation",'relationship',"race","sex","capital-gain","capital-loss","hours-per-week","native-country"]
        self.feature_to_remove = False

        self.few_shot = few_shot
        self.fairness = fairness
        self.feature_to_process = feature_to_process
        self.dataset_type = dataset_type

    def __len__(self):
        return self.number_of_samples


    def __getitem__(self, index):
        prompt = "Predict whether the annual income of the person is greater than $50k\n"
        new_prompts = []
        #shuffle new_prompts
        np.random.shuffle(new_prompts)
        prompt +="".join(new_prompts)
        idx = random_except(0,len(self.X),index)
        prompts = []#!!!!!!!!!!!

        features_list=self.X.loc[idx][ self.features].tolist()
        addition_prompt="Text: A person in 1996 has the following attributes: age {0}, workclass {1}, fnlwgt {2}, education {3}, educational-num {4}, marital-status {5}, occupation {6}, relationship {7}, race {8}, gender {9}, capital-gain {10}, capital-loss {11}, hours-per-week {12}, native-country {13}.\nPrediction is ?".format(*features_list)

        if self.dataset_type == 'lora':
            # 如果指定了要删除的特征!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            if self.feature_to_remove:
                self.X[self.feature_to_process] = ""
            prompts.append(prompt + addition_prompt)
            return prompts,self.X.loc[idx][self.feature_to_process], self.y[idx]
        
        elif self.dataset_type == 'adv':    

            # 统计每个特征值对应的样本量
            feature_counts = self.X[self.feature_to_process].value_counts()
            # 排序，确保样本数最多的特征值在最前面
            sorted_feature_values = feature_counts.index.tolist()
            # 获取特征的总数
            total_feature_count = len(sorted_feature_values)

            feature_index = self.features.index(self.feature_to_process)

            # 循环并生成 prompts
            for _ in range(total_feature_count - 1):  #[0,6-1)  5个
                # 处理最常见的特征值
                feature_value=sorted_feature_values[0]
                features_list[feature_index] = feature_value
                addition_prompt="Text: A person in 1996 has the following attributes: age {0}, workclass {1}, fnlwgt {2}, education {3}, educational-num {4}, marital-status {5}, occupation {6}, relationship {7}, race {8}, gender {9}, capital-gain {10}, capital-loss {11}, hours-per-week {12}, native-country {13}.\nPrediction is ?".format(*features_list)
                prompts.append(prompt + addition_prompt)
                
            for j in range(1, total_feature_count):   #[1,6) 5个

                feature_value = sorted_feature_values[j]
                # 获取self.features中的特征名称的索引
                # 替换指定位置的值
                features_list[feature_index] = feature_value
                addition_prompt="Text: A person in 1996 has the following attributes: age {0}, workclass {1}, fnlwgt {2}, education {3}, educational-num {4}, marital-status {5}, occupation {6}, relationship {7}, race {8}, gender {9}, capital-gain {10}, capital-loss {11}, hours-per-week {12}, native-country {13}.\nPrediction is ?".format(*features_list)
                prompts.append(prompt + addition_prompt)
            
            return prompts,self.X.loc[idx][self.feature_to_process], self.y[idx]
        else:
            raise ValueError("Invalid dataset_type. Use 'lora' or 'adv'.")        


#############
#############
#############
#Sampling:::::::::::::::::::::
def Adult_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:48842
    :param num_users:100
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    # num_items=100
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):#0到num-1的list遍历
        idx_set = set(np.random.choice(all_idxs, num_items, replace=False))
        #获得数据；组装成dict。
        dict_users[i]=[dataset[y] for y in  idx_set]
        # 从 all_idxs 中随机选择 num_items 个样本索引，分配给第 i 个客户端。replace=False 确保每个样本只被选一次。
        
        all_idxs = list(set(all_idxs) - idx_set)

    return dict_users



def Adult_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 200, 224  # 将数据划分为 200 个 shard，每个 shard 包含 224 个数据
    idx_shard = [i for i in range(num_shards)]  # 创建一个包含 shard 索引的列表
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}  # 初始化每个用户的数据字典，值为空数组
    idxs = np.arange(num_shards * num_imgs)  # 创建一个索引数组，表示数据集中所有图片的索引
    labels = dataset.labels.numpy()  # 获取数据集的标签，并转换为 NumPy 数组格式

    # 对标签进行排序
    idxs_labels = np.vstack((idxs, labels))  # 将索引和标签垂直堆叠，便于按标签排序
    #Eg.idxs_labels= [[0, 1, 2, ..., 599]
    #                 ,[1, 0, 1, ..., 2]]. 
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]  # 根据标签排序索引


    idxs = idxs_labels[0, :]  # 提取排序后的索引

    # 分配数据
    for i in range(num_users):  # 遍历每个用户 100人;
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))  # 随机选择 2 个 shard
        idx_shard = list(set(idx_shard) - rand_set)  # 从可用 shard 列表中移除已分配的 shard
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
            # 将选中的 shard 数据索引分配给该用户，这里是循环，所以塞了2*300张图片，其中每个shard内部都有单增的label序列；
    return dict_users  # 返回分配给每个用户的索引字典，value是600个数据索引

#GermanCreditDataset
class GermanCreditDataset(Dataset):
    def __init__(self, few_shot,dataset_type, number_of_samples=None, fairness=None, fairprompt=False, feature_to_process='sex'):
        # 加载数据
        german_credit = fetch_dict_from_df('german_credit')
        X = german_credit.data.features
        y = german_credit.data.targets['Target']

        if number_of_samples is not None:
            self.number_of_samples = number_of_samples
        else:
            self.number_of_samples = len(X)
            number_of_samples = len(X)

        self.fairprompt = fairprompt

        if fairness is not None:
            # 进行公平性调整
            X["Target"] = y
            x_0 = int(number_of_samples/2*(1+fairness)/2)
            x_1 = int(max(number_of_samples/2 - x_0,0))
            x_2 = int(number_of_samples/2*(1-fairness)/2)
            x_3 = int(max(number_of_samples/2 - x_2,0))
            if feature_to_process == "sex":
                X = pd.concat(
                    [X[(X["Target"] ==1) & (X["sex"] == "Male")].sample(n=x_0,replace=True),
                    X[(X["Target"] ==1) & (X["sex"] == "Female")].sample(n=x_1,replace=True),
                    X[(X["Target"] ==0) & (X["sex"] == "Male")].sample(n=x_2,replace=True),
                    X[(X["Target"] ==0 )& (X["sex"] == "Female")].sample(n=x_3,replace=True)])

            elif feature_to_process == "marital-status":
                # 将marital-status的特征拆分为Never-married和其他
                X = pd.concat(
                    [X[(X["Target"] == 1) & (X["marital-status"] == "single")].sample(n=x_0, replace=True),
                    X[(X["Target"] == 1) & (X["marital-status"] != "single")].sample(n=x_1, replace=True),
                    X[(X["Target"] == 0) & (X["marital-status"] == "single")].sample(n=x_2, replace=True),
                    X[(X["Target"] == 0) & (X["marital-status"] != "single")].sample(n=x_3, replace=True)]
                )
            else:
                raise ValueError("Invalid feature_to_process. Use 'sex' or 'race'.")
            X = X.sample(frac=1).reset_index(drop=True)
            self.y = X["Target"]
            X.drop(columns=["Target"], inplace=True)
            self.X = X
        else:
            X["Target"] = y
            X = X.sample(frac=1).reset_index(drop=True)
            self.y = X["Target"]
            X.drop(columns=["Target"], inplace=True)
            self.X = X
            self.fairprompt = fairprompt

        # 设置需要处理的特征
        self.features = X.columns.tolist()
        self.feature_to_remove = False

        self.few_shot = few_shot
        self.fairness = fairness
        self.feature_to_process = feature_to_process
        self.dataset_type = dataset_type

    def __len__(self):
        return self.number_of_samples



    def __getitem__(self, index):
            prompt = "Predict whether the credit is good(1) or bad(0).\n"
            new_prompts = []
            #shuffle new_prompts
            np.random.shuffle(new_prompts)
            prompt +="".join(new_prompts)
            idx = random_except(0,len(self.X),index)
            prompts = []#!!!!!!!!!!!

            features_list=self.X.loc[idx][ self.features].tolist()
            addition_prompt="Text: A person has the following attributes: Checking account {0}, Duration {1}, Credit history {2}, Purpose {3}, Credit amount {4}, Savings account {5}, Employment since {6}, Installment rate {7}, sex {8}, marital-status {9}, Debtors & guarantors {10}, Residence since {11}, Property {12}, Age {13}, Other installments {14}, Housing {15}, Existing credits {16}, Job {17}, Kept people {18}, Phone {19}, Foreign-worker {20}\nPrediction is ?".format(
                *features_list
                )

            if self.dataset_type == 'lora':
                # 如果指定了要删除的特征!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                if self.feature_to_remove:
                    self.X[self.feature_to_process] = ""
                prompts.append(prompt + addition_prompt)
                return prompts,self.X.loc[idx][self.feature_to_process], self.y[idx]
            
            elif self.dataset_type == 'adv':    

                # 统计每个特征值对应的样本量
                feature_counts = self.X[self.feature_to_process].value_counts()
                # 排序，确保样本数最多的特征值在最前面
                sorted_feature_values = feature_counts.index.tolist()
                # 获取特征的总数
                total_feature_count = len(sorted_feature_values)

                feature_index = self.features.index(self.feature_to_process)

                # 循环并生成 prompts
                for _ in range(total_feature_count - 1):  #[0,6-1)  5个
                    # 处理最常见的特征值
                    feature_value=sorted_feature_values[0]
                    features_list[feature_index] = feature_value
                    addition_prompt="Text: A person has the following attributes: Checking account {0}, Duration {1}, Credit history {2}, Purpose {3}, Credit amount {4}, Savings account {5}, Employment since {6}, Installment rate {7}, sex {8}, marital-status {9}, Debtors & guarantors {10}, Residence since {11}, Property {12}, Age {13}, Other installments {14}, Housing {15}, Existing credits {16}, Job {17}, Kept people {18}, Phone {19}, Foreign-worker {20}\nPrediction is ?".format(
                        *features_list
                        )

                    prompts.append(prompt + addition_prompt)
                    
                for j in range(1, total_feature_count):   #[1,6) 5个

                    feature_value = sorted_feature_values[j]
                    # 获取self.features中的特征名称的索引
                    # 替换指定位置的值
                    features_list[feature_index] = feature_value
                    addition_prompt="Text: A person has the following attributes: Checking account {0}, Duration {1}, Credit history {2}, Purpose {3}, Credit amount {4}, Savings account {5}, Employment since {6}, Installment rate {7}, sex {8}, marital-status {9}, Debtors & guarantors {10}, Residence since {11}, Property {12}, Age {13}, Other installments {14}, Housing {15}, Existing credits {16}, Job {17}, Kept people {18}, Phone {19}, Foreign-worker {20}\nPrediction is ?".format(
                        *features_list
                        )

                    prompts.append(prompt + addition_prompt)
                
                return prompts,self.X.loc[idx][self.feature_to_process], self.y[idx]
            else:
                raise ValueError("Invalid dataset_type. Use 'lora' or 'adv'.")        


class BankMarketingDataset(Dataset):
    def __init__(self, few_shot,dataset_type, number_of_samples=None, fairness=None, fairprompt=False, feature_to_process='sex'):
        # 加载数据
        german_credit = fetch_dict_from_df('bank_marketing')
        X = german_credit.data.features
        y = german_credit.data.targets['Target']

        if number_of_samples is not None:
            self.number_of_samples = number_of_samples
        else:
            self.number_of_samples = len(X)
            number_of_samples = len(X)

        self.fairprompt = fairprompt

        if fairness is not None:
            # 进行公平性调整
            X["Target"] = y
            x_0 = int(number_of_samples/2*(1+fairness)/2)
            x_1 = int(max(number_of_samples/2 - x_0,0))
            x_2 = int(number_of_samples/2*(1-fairness)/2)
            x_3 = int(max(number_of_samples/2 - x_2,0))
            if feature_to_process == "sex":
                X = pd.concat(
                    [X[(X["Target"] ==1) & (X["sex"] == "Male")].sample(n=x_0,replace=True),
                    X[(X["Target"] ==1) & (X["sex"] == "Female")].sample(n=x_1,replace=True),
                    X[(X["Target"] ==0) & (X["sex"] == "Male")].sample(n=x_2,replace=True),
                    X[(X["Target"] ==0 )& (X["sex"] == "Female")].sample(n=x_3,replace=True)])
            elif feature_to_process == "race":
                X = pd.concat(
                    [X[(X["Target"] == 1) & (X["race"] == "White")].sample(n=x_0, replace=True),
                     X[(X["Target"] == 1) & (X["race"] != "White")].sample(n=x_1, replace=True),
                     X[(X["Target"] == 0) & (X["race"] == "White")].sample(n=x_2, replace=True),
                     X[(X["Target"] == 0) & (X["race"] != "White")].sample(n=x_3, replace=True)]
                )
            elif feature_to_process == "marital-status":
                # 将marital-status的特征拆分为Never-married和其他
                X = pd.concat(
                    [X[(X["Target"] == 1) & (X["marital-status"] == "single")].sample(n=x_0, replace=True),
                    X[(X["Target"] == 1) & (X["marital-status"] != "single")].sample(n=x_1, replace=True),
                    X[(X["Target"] == 0) & (X["marital-status"] == "single")].sample(n=x_2, replace=True),
                    X[(X["Target"] == 0) & (X["marital-status"] != "single")].sample(n=x_3, replace=True)]
                )
            elif feature_to_process == "Foreign-worker":
                X = pd.concat(
                    [X[(X["Target"] ==1) & (X["Foreign-worker"] == "yes")].sample(n=x_0,replace=True),
                    X[(X["Target"] ==1) & (X["Foreign-worker"] == "no")].sample(n=x_1,replace=True),
                    X[(X["Target"] ==0) & (X["Foreign-worker"] == "yes")].sample(n=x_2,replace=True),
                    X[(X["Target"] ==0 )& (X["Foreign-worker"] == "no")].sample(n=x_3,replace=True)])
            elif feature_to_process == "education":
                X = pd.concat(
                    [X[(X["Target"] ==1) & (X["education"] == "secondary")].sample(n=x_0,replace=True),
                    X[(X["Target"] ==1) & (X["education"] != "secondary")].sample(n=x_1,replace=True),
                    X[(X["Target"] ==0) & (X["education"] == "secondary")].sample(n=x_2,replace=True),
                    X[(X["Target"] ==0 )& (X["education"] != "secondary")].sample(n=x_3,replace=True)])
            else:
                raise ValueError("Invalid feature_to_process. Use 'sex' or 'race'.")
            X = X.sample(frac=1).reset_index(drop=True)
            self.y = X["Target"]
            X.drop(columns=["Target"], inplace=True)
            self.X = X
        else:
            X["Target"] = y
            X = X.sample(frac=1).reset_index(drop=True)
            self.y = X["Target"]
            X.drop(columns=["Target"], inplace=True)
            self.X = X
            self.fairprompt = fairprompt

        # 设置需要处理的特征
        self.features = X.columns.tolist()
        self.feature_to_remove = False

        self.few_shot = few_shot
        self.fairness = fairness
        self.feature_to_process = feature_to_process
        self.dataset_type = dataset_type

    def __len__(self):
        return self.number_of_samples

    def __getitem__(self, index):
        prompt = "Predict whether a person will subscribe to a term deposit (yes=1, no=0).\n"        
        new_prompts = []
        #shuffle new_prompts
        np.random.shuffle(new_prompts)
        prompt +="".join(new_prompts)
        idx = random_except(0,len(self.X),index)
        prompts = []#!!!!!!!!!!!

        features_list=self.X.loc[idx][ self.features].tolist()
        addition_prompt="Text: A person has the following attributes: age {0}, job {1}, marital {2}, education {3}, balance {4}, housing {5}, loan {6}, contact {7}, day {8}, month {9}, duration {10}, campaign {11}, pdays {12}, previous {13}, poutcome {14} \nPrediction is ?".format(
            *features_list)

        if self.dataset_type == 'lora':
            # 如果指定了要删除的特征!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            if self.feature_to_remove:
                self.X[self.feature_to_process] = ""
            prompts.append(prompt + addition_prompt)
            return prompts,self.X.loc[idx][self.feature_to_process], self.y[idx]
        
        elif self.dataset_type == 'adv':    

            # 统计每个特征值对应的样本量
            feature_counts = self.X[self.feature_to_process].value_counts()
            # 排序，确保样本数最多的特征值在最前面
            sorted_feature_values = feature_counts.index.tolist()
            # 获取特征的总数
            total_feature_count = len(sorted_feature_values)

            feature_index = self.features.index(self.feature_to_process)

            # 循环并生成 prompts
            for _ in range(total_feature_count - 1):  #[0,6-1)  5个
                # 处理最常见的特征值
                feature_value=sorted_feature_values[0]
                features_list[feature_index] = feature_value
                addition_prompt="Text: A person has the following attributes: age {0}, job {1}, marital {2}, education {3}, balance {4}, housing {5}, loan {6}, contact {7}, day {8}, month {9}, duration {10}, campaign {11}, pdays {12}, previous {13}, poutcome {14} \nPrediction is ?".format(
                    *features_list)
                prompts.append(prompt + addition_prompt)
                
            for j in range(1, total_feature_count):   #[1,6) 5个

                feature_value = sorted_feature_values[j]
                # 获取self.features中的特征名称的索引
                # 替换指定位置的值
                features_list[feature_index] = feature_value
                addition_prompt="Text: A person has the following attributes: age {0}, job {1}, marital {2}, education {3}, balance {4}, housing {5}, loan {6}, contact {7}, day {8}, month {9}, duration {10}, campaign {11}, pdays {12}, previous {13}, poutcome {14} \nPrediction is ?".format(
                    *features_list)
                prompts.append(prompt + addition_prompt)
            
            return prompts,self.X.loc[idx][self.feature_to_process], self.y[idx]
        else:
            raise ValueError("Invalid dataset_type. Use 'lora' or 'adv'.")        

class CompasDataset(Dataset):
    def __init__(self,few_shot,dataset_type,number_of_samples=None,fairness=None,fairprompt=False,feature_to_process='sex'):
        
        # adult = pd.read_csv(dataset_path)
        compas = fetch_dict_from_df(data='compas')
        X = compas.data.features
        y = compas.data.targets['Target']

        if number_of_samples is not None:
            self.number_of_samples = number_of_samples
        else:
            self.number_of_samples = len(X)
            number_of_samples = len(X)

        self.fairprompt  = False

        if fairness is not None:
            #either one of fairness or fairprompt should be taken as true.
            X["Target"] = y
            x_0 = int(number_of_samples/2*(1+fairness)/2)
            x_1 = int(max(number_of_samples/2 - x_0,0))
            x_2 = int(number_of_samples/2*(1-fairness)/2)
            x_3 = int(max(number_of_samples/2 - x_2,0))
            if feature_to_process == "sex":
                X = pd.concat(
                    [X[(X["Target"] ==1) & (X["sex"] == "Male")].sample(n=x_0,replace=True),
                    X[(X["Target"] ==1) & (X["sex"] == "Female")].sample(n=x_1,replace=True),
                    X[(X["Target"] ==0) & (X["sex"] == "Male")].sample(n=x_2,replace=True),
                    X[(X["Target"] ==0 )& (X["sex"] == "Female")].sample(n=x_3,replace=True)])
            elif feature_to_process == "race":
                X = pd.concat(
                    [X[(X["Target"] == 1) & (X["race"] == "African-American")].sample(n=x_0, replace=True),
                     X[(X["Target"] == 1) & (X["race"] != "African-American")].sample(n=x_1, replace=True),
                     X[(X["Target"] == 0) & (X["race"] == "African-American")].sample(n=x_2, replace=True),
                     X[(X["Target"] == 0) & (X["race"] != "African-American")].sample(n=x_3, replace=True)]
                )
            elif feature_to_process == "marital-status":#not available
                # 将marital-status的特征拆分为Never-married和其他
                X = pd.concat(
                    [X[(X["Target"] == 1) & (X["marital-status"] == "single")].sample(n=x_0, replace=True),
                    X[(X["Target"] == 1) & (X["marital-status"] != "single")].sample(n=x_1, replace=True),
                    X[(X["Target"] == 0) & (X["marital-status"] == "single")].sample(n=x_2, replace=True),
                    X[(X["Target"] == 0) & (X["marital-status"] != "single")].sample(n=x_3, replace=True)]
                )
            else:
                raise ValueError("Invalid feature_to_process. Use 'sex' or 'race'.")
            X = X.sample(frac=1).reset_index(drop=True)
            self.y = X["Target"]
            X.drop(columns=["Target"], inplace=True)
            self.X = X
        else:
            X["Target"] = y
            X = X.sample(frac=1).reset_index(drop=True)
            self.y = X["Target"]
            X.drop(columns=["Target"], inplace=True)
            self.X = X
            self.fairprompt = fairprompt

        # 设置需要处理的特征
        self.features = X.columns.tolist()
        self.feature_to_remove = False

        self.few_shot = few_shot
        self.fairness = fairness
        self.feature_to_process = feature_to_process
        self.dataset_type = dataset_type

    def __len__(self):
        return self.number_of_samples

    def __getitem__(self, index):
        prompt = "Predict whether the person will reoffend within 2 years(yes is 1,no is 0).\n"     
        new_prompts = []
        #shuffle new_prompts
        np.random.shuffle(new_prompts)
        prompt +="".join(new_prompts)
        idx = random_except(0,len(self.X),index)
        prompts = []#!!!!!!!!!!!

        features_list=self.X.loc[idx][ self.features].tolist()
        addition_prompt = "Text: A person has the following attributes: sex {0}, age {1}, race {2}, juv_fel_count {3}, decile_score {4}, juv_misd_count {5}, juv_other_count {6}, priors_count {7}, days_b_screening_arrest {8}, c_jail_in {9}, c_jail_out {10}, c_case_number {11}, c_offense_date {12}, c_arrest_date {13}, c_days_from_compas {14}, c_charge_degree {15}, c_charge_desc {16}, is_recid {17}, r_case_number {18}, r_charge_degree {19}, r_days_from_arrest {20}, r_offense_date {21}, r_charge_desc {22}, r_jail_in {23}, r_jail_out {24}, violent_recid {25}, is_violent_recid {26}, vr_case_number {27}, vr_charge_degree {28}, vr_offense_date {29}, vr_charge_desc {30}, type_of_assessment {31}, decile_score {32}, score_text {33}, screening_date {34}, v_type_of_assessment {35}, v_decile_score {36}, v_score_text {37}, v_screening_date {38}, in_custody {39}, out_custody {40}, priors_count {41}, start {42}, end {43}, event {44}. Prediction is ?".format(
            *features_list)


        if self.dataset_type == 'lora':
            # 如果指定了要删除的特征!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            if self.feature_to_remove:
                self.X[self.feature_to_process] = ""
            prompts.append(prompt + addition_prompt)
            return prompts,self.X.loc[idx][self.feature_to_process], self.y[idx]
        
        elif self.dataset_type == 'adv':    

            # 统计每个特征值对应的样本量
            feature_counts = self.X[self.feature_to_process].value_counts()
            # 排序，确保样本数最多的特征值在最前面
            sorted_feature_values = feature_counts.index.tolist()
            # 获取特征的总数
            total_feature_count = len(sorted_feature_values)

            feature_index = self.features.index(self.feature_to_process)

            # 循环并生成 prompts
            for _ in range(total_feature_count - 1):  #[0,6-1)  5个
                # 处理最常见的特征值
                feature_value=sorted_feature_values[0]
                features_list[feature_index] = feature_value
                addition_prompt = "Text: A person has the following attributes: sex {0}, age {1}, race {2}, juv_fel_count {3}, decile_score {4}, juv_misd_count {5}, juv_other_count {6}, priors_count {7}, days_b_screening_arrest {8}, c_jail_in {9}, c_jail_out {10}, c_case_number {11}, c_offense_date {12}, c_arrest_date {13}, c_days_from_compas {14}, c_charge_degree {15}, c_charge_desc {16}, is_recid {17}, r_case_number {18}, r_charge_degree {19}, r_days_from_arrest {20}, r_offense_date {21}, r_charge_desc {22}, r_jail_in {23}, r_jail_out {24}, violent_recid {25}, is_violent_recid {26}, vr_case_number {27}, vr_charge_degree {28}, vr_offense_date {29}, vr_charge_desc {30}, type_of_assessment {31}, decile_score {32}, score_text {33}, screening_date {34}, v_type_of_assessment {35}, v_decile_score {36}, v_score_text {37}, v_screening_date {38}, in_custody {39}, out_custody {40}, priors_count {41}, start {42}, end {43}, event {44}. Prediction is ?".format(
                    *features_list)
                prompts.append(prompt + addition_prompt)
                
            for j in range(1, total_feature_count):   #[1,6) 5个

                feature_value = sorted_feature_values[j]
                # 获取self.features中的特征名称的索引
                # 替换指定位置的值
                features_list[feature_index] = feature_value
                addition_prompt = "Text: A person has the following attributes: sex {0}, age {1}, race {2}, juv_fel_count {3}, decile_score {4}, juv_misd_count {5}, juv_other_count {6}, priors_count {7}, days_b_screening_arrest {8}, c_jail_in {9}, c_jail_out {10}, c_case_number {11}, c_offense_date {12}, c_arrest_date {13}, c_days_from_compas {14}, c_charge_degree {15}, c_charge_desc {16}, is_recid {17}, r_case_number {18}, r_charge_degree {19}, r_days_from_arrest {20}, r_offense_date {21}, r_charge_desc {22}, r_jail_in {23}, r_jail_out {24}, violent_recid {25}, is_violent_recid {26}, vr_case_number {27}, vr_charge_degree {28}, vr_offense_date {29}, vr_charge_desc {30}, type_of_assessment {31}, decile_score {32}, score_text {33}, screening_date {34}, v_type_of_assessment {35}, v_decile_score {36}, v_score_text {37}, v_screening_date {38}, in_custody {39}, out_custody {40}, priors_count {41}, start {42}, end {43}, event {44}. Prediction is ?".format(
                    *features_list)
                prompts.append(prompt + addition_prompt)
            
            return prompts,self.X.loc[idx][self.feature_to_process], self.y[idx]
        else:
            raise ValueError("Invalid dataset_type. Use 'lora' or 'adv'.")        
