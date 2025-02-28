# from datasets import load_dataset

# dataset = load_dataset("iamgroot42/mimir", "pile_cc", split="ngram_7_0.2")
# print(dataset)
import pickle
from component.dataset import SFTDataset,AdultDataset,AdultDatasetGender,AdultDatasetTest,Adult_iid,Adult_noniid
import os
# 加载训练集
print("Load Dataset")
few_shot = 0
fairness = 0.5
fairprompt = False
number_of_samples=192
# folder_path = '/home/chen/pyh/FedJudge-main/used_dataset'
# file_path2 = os.path.join(folder_path, 'nogenderdataset.pkl')
# with open(file_path2, 'wb') as f:
#     dataset = AdultDataset(few_shot=few_shot, number_of_samples=number_of_samples, fairness=fairness, fairprompt=fairprompt)
#     sample = dataset[0]
#     pickle.dump(dataset, f)

dataset = AdultDataset(few_shot=few_shot, number_of_samples=number_of_samples, fairness=fairness, fairprompt=fairprompt)
sample = dataset[0]

# AdultDatasetGender_nogender