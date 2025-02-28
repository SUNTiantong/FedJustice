import pickle
from component.dataset import Adult_iid,Adult_noniid,AdultDataset,GermanCreditDataset,BankMarketingDataset,CompasDataset
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
# 加载训练集
print("Load Dataset")
few_shot = 0
fairness = 0.5
fairprompt = False
class CustomSubset(Subset):
    def __init__(self, dataset, indices, feature_to_remove=True):
        super().__init__(dataset, indices)  # 调用父类的构造函数
        self.dataset = dataset               # 保存数据集对象
        self.dataset.feature_to_remove = feature_to_remove  # 修改数据集中的特征去除标志


def getdata(args,data='adult',few_shot=few_shot,number_of_samples=-1, client_num=3, fairness=fairness, fairprompt=fairprompt, feature_to_process='sex', feature_to_remove=False):
    test_size=args.test_size
    if data=='adult': 
        # 处理gender特征（例如'sex'、'marital-status'、'race'等）可能包括删除指定特征
        raw_dataset = AdultDataset(few_shot=few_shot,dataset_type='lora', number_of_samples=number_of_samples, fairness=fairness, 
                                            fairprompt=fairprompt,  feature_to_process=feature_to_process)

        # 处理dataset，不考虑run_adv的值
        raw_dataset_for_adv = AdultDataset(few_shot=few_shot,dataset_type='adv', number_of_samples=number_of_samples, fairness=fairness, 
                            fairprompt=fairprompt, feature_to_process=feature_to_process)
        # test_size = 0.5
    elif data=='german_credit': 
        raw_dataset = GermanCreditDataset(few_shot=few_shot, dataset_type='lora', number_of_samples=number_of_samples, fairness=fairness,
                                fairprompt=fairprompt, feature_to_process=feature_to_process)
        # 处理dataset，
        raw_dataset_for_adv = GermanCreditDataset(few_shot=few_shot, dataset_type='adv' , number_of_samples=number_of_samples, fairness=fairness, 
                            fairprompt=fairprompt, feature_to_process=feature_to_process)
        # test_size = 0.1

    elif data == 'bank_marketing':
        # 处理Bank-Marketing数据集
        raw_dataset = BankMarketingDataset(few_shot=few_shot, dataset_type='lora', number_of_samples=number_of_samples, fairness=fairness,
                                           fairprompt=fairprompt, feature_to_process=feature_to_process)

        # 处理dataset，不考虑run_adv的值
        raw_dataset_for_adv = BankMarketingDataset(few_shot=few_shot, dataset_type='adv', number_of_samples=number_of_samples, fairness=fairness, 
                                                  fairprompt=fairprompt, feature_to_process=feature_to_process)
        # test_size = 0.1
    elif data == 'compas':
        # 处理Compas数据集
        raw_dataset = CompasDataset(few_shot=few_shot, dataset_type='lora', number_of_samples=number_of_samples, fairness=fairness,
                                     fairprompt=fairprompt, feature_to_process=feature_to_process)

        # 处理dataset，不考虑run_adv的值
        raw_dataset_for_adv = CompasDataset(few_shot=few_shot, dataset_type='adv', number_of_samples=number_of_samples, fairness=fairness, 
                                            fairprompt=fairprompt, feature_to_process=feature_to_process)
        # test_size = 0.1
    else:
        raise ValueError(f"Unsupported dataset: {data}")
    
    test_size=args.test_size
    # 生成对应的字典
    train_indices, test_indices = train_test_split(
        range(len(raw_dataset.X)), 
        test_size=test_size, 
        stratify=raw_dataset.y,  # 使用y标签进行分层抽样# 在划分数据集时，使用分层抽样来保证训练集和测试集中的类别比例相同
        random_state=42
    )
    # # 使用 Subset 来从数据集中提取训练集和测试集
    # train_dataset = CustomSubset(raw_dataset, train_indices)
  
    # 生成训练集字典，使用带有特征移除的训练集
    if feature_to_remove:
        train_dataset = CustomSubset(raw_dataset, train_indices, feature_to_remove=True)  # 训练集去除特征  # 设置训练集去除特征
    else:
        train_dataset = CustomSubset(raw_dataset, train_indices, feature_to_remove=False)
    
    test_dataset = CustomSubset(raw_dataset, test_indices, feature_to_remove=False)
    # 生成对应的训练集字典
    dict_users_train_dataset = Adult_iid(train_dataset, client_num)
    dict_users_train_dataset_adv = Adult_iid(raw_dataset_for_adv, client_num)

    return dict_users_train_dataset, dict_users_train_dataset_adv, test_dataset



# ad=getdata(data='adult',few_shot=few_shot,number_of_samples=1000, client_num=3, fairness=fairness, fairprompt=fairprompt, feature_to_process='sex',feature_to_remove=True)
