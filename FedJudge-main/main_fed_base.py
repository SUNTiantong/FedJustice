#TODO adv需要写在主函数，从而每个lora都配置一个，不妨另开一个文件来import adv
from transformers import AutoTokenizer, BitsAndBytesConfig
import deepspeed
from peft import (
    # prepare_model_for_int8_training,
    LoraConfig,
    TaskType,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from torchvision import datasets, transforms
from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
from utils.options import args_parser


from transformers import (
    set_seed,
    HfArgumentParser,
    TrainingArguments,
    AutoModelForCausalLM,
    AutoConfig
)
import torch.nn.functional as F
import argparse
from loguru import logger
import os
from os.path import join
import torch.nn as nn
import torch
import bitsandbytes as bnb
from collections import defaultdict
import copy
# from component.collator import SFTDataCollator
# from component.dataset import SFTDataset,AdultDataset,AdultDatasetGender,Adult_iid,Adult_noniid
from component.argument import QLoRAArguments
# from component.trainer import LoRATrainer,ModifiedTrainer
from component.loss import TargetLMLoss
import warnings
warnings.filterwarnings("ignore")
from component.xinxiede import LocalUpdate,test
from component.withadv import CategoricalEmb
import Load_Dataset
import pickle

# device = torch.device("cuda:0") 
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed





def verify_model_dtype(model):
    """
    查看模型种各种类型的参数的情况
    """
    dtype2param_num = defaultdict(int)  # 每种数据类型的参数量
    dtype2param_name = defaultdict(list)  # 每种数据类型的参数名称
    dtype2trainable_param_num = defaultdict(int)  # 每种数据类型参与训练的参数量
    dtype2trainable_param_name = defaultdict(list)  # 每种数据类型参与训练的参数名称
    for name, p in model.named_parameters():
        dtype = p.dtype
        dtype2param_num[dtype] += p.numel()
        dtype2param_name[dtype].append(name)
        if p.requires_grad:
            dtype2trainable_param_num[dtype] += p.numel()
            dtype2trainable_param_name[dtype].append(name)
    # 统计全部参数中，各种类型参数分布
    total = 0
    print('verify all params of the model')
    for k, v in dtype2param_num.items():
        total += v
    for k, v in dtype2param_num.items():
        print(k, v, v / total)
    for k, v in dtype2trainable_param_name.items():
        print(k, v)

    print()
    # 统计可训练参数中，各种类型参数分布
    print('verify trainable params the model')
    total_trainable = 0
    for k, v in dtype2trainable_param_num.items():
        total_trainable += v
    for k, v in dtype2trainable_param_num.items():
        print(k, v, v / total_trainable)
    for k, v in dtype2trainable_param_num.items():
        print(k, v)


def setup_everything():

    # 创建 ArgumentParser 对象
    parser0 = HfArgumentParser(QLoRAArguments)
    args = parser0.parse_args()
    parser = argparse.ArgumentParser()

    #下面和args无关h_
    training_args=args
    # train_args_file = args.train_args_file
    # parser = HfArgumentParser((QLoRAArguments, TrainingArguments))
    # h_args, training_args = parser.parse_json_file(json_file=train_args_file)
    # print(training_args)
# getdata_3000 3 10 3 50 
    # 创建输出目录
    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)
    # logger.add(join(training_args.output_dir, 'train.log'))
    # logger.info("train_args:{}".format(training_args))
    # 设置随机种子
    # set_seed(training_args.seed)
    return args, training_args


def local_update(wi,client_idx,model,adv_model,args,training_args,gender_dataset,dataset,tokenizer,server_model):
    training_args.num_train_epochs = 1    
    print("local update: idx={}".format(client_idx))
    # trainer.train()  # 检查点

    # Initialize adv_models for each client

    local = LocalUpdate(args=args, tokenizer=tokenizer,gender_dataset=gender_dataset,dataset=dataset)
    local.train(lora_model=model,adv_model = adv_model,client_idx=client_idx,round=wi,server_model=server_model) 

    model.save_pretrained(training_args.output_dir)
    return get_peft_model_state_dict(model)
"""
1. 这里只实例化了LocalUpdate这个类， 没用到train这个函数， 而且也没有传入lora_model和adv_model。实例化的时候会执行init函数，要想执行train，需要local.train()， train最后返回了三个参数，你这里也需要有三个参数接收。然后保存
2. 训练过程中可以输出他的训练损失。
3. 在完成聚合之后，对这个模型测试公平性，输出公平性指标。
4. 如果每个客户端数据太少，可能效果并不明显。因为模型很大，数据很小的话。容易过拟合。之后可以逐步加一下。
"""


def parallel_local_update(client_models, dict_users_train_dataset, dict_users_train_dataset_adv, args, training_args,tokenizer):
    """
    并行化客户端的训练
    """
    with ThreadPoolExecutor(max_workers=len(client_models)) as executor:
        futures = []
        for idx, model in enumerate(client_models):
            futures.append(executor.submit(local_update_with_memory_management, idx, model, args, training_args, dict_users_train_dataset, dict_users_train_dataset_adv, tokenizer))
        
        w_locals = []
        client_weights = []
        for future in as_completed(futures):
            local_w, deo = future.result()
            w_locals.append(local_w)
            client_weights.append(deo)
    
    return w_locals, client_weights

def local_update_with_memory_management(idx, model, args, training_args, dict_users_train_dataset, dict_users_train_dataset_adv, tokenizer):
    """
    带有内存管理的本地更新函数
    逐个加载并删除 LoRA 模型，以避免显存不足
    """
    # 加载并设置模型到当前设备（GPU）
    model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # 执行模型的训练更新（假设 `local_update` 是训练的主要函数）
    # 注意，`local_update` 需要修改成适配本地内存管理的代码
    local_w, deo = local_update(idx, model, args, training_args, dict_users_train_dataset, dict_users_train_dataset_adv, tokenizer)

    # 删除模型并清理显存
    del model  # 删除模型，释放内存
    torch.cuda.empty_cache()  # 清空缓存显存

    return local_w, deo


def communication( server_model, w_locals, client_models,client_weights):
    client_num = len(w_locals)    
    with torch.no_grad():
        w_avg = copy.deepcopy(w_locals[0])
        for k in w_avg.keys():
            w_avg[k] = client_weights[0]*w_locals[0][k]

        for k in w_avg.keys():
            for i in range(1, client_num):  # i: 参与训练的 clients_num
                w_avg[k] += client_weights[i]*w_locals[i][k] # 各部分权重加和   

        ## 完成参数聚合
        set_peft_model_state_dict(server_model, w_avg)
        ## 分发更新后的参数
        w_globals = get_peft_model_state_dict(server_model)
        for client_idx in range(client_num):
            set_peft_model_state_dict(client_models[client_idx], w_globals)

    return server_model, client_models


def training(args, training_args):
    """
    初始化各个组件
    """
    logger.info('Initializing components...')
    # 下面的设置至关重要，否则无法多卡训练
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    training_args.ddp_find_unused_parameters = False
    device_map = "auto"
    # if we are in a distributed setting, we need to set the device map and max memory per device
    if os.environ.get('LOCAL_RANK') is not None:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        device_map = {'': local_rank}

    # 加载模型
    print("*** model loading ***")
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B",
                                              cache_dir="./research/hal-gaudisac/fairness/gpt_neo_cache",
                                              bos_token="[BOS]",
                                              eos_token="[EOS]",
                                              unk_token="[UNK]",
                                              pad_token="[PAD]",
                                              mask_token="[MASK]",
                                              sep_token="[SEP]",  # 添加sep_token
                                              cls_token="[CLS]")  # 添加cls_token
                                              
    client_tokenizer=tokenizer

    # 部分tokenizer没有pad_token_id
    if client_tokenizer.pad_token_id is None:
        client_tokenizer.pad_token_id = client_tokenizer.unk_token_id
    # 如果两者相同，模型训练时不会计算eos_token_id的loss
    if client_tokenizer.pad_token_id == client_tokenizer.eos_token_id:
        raise Exception('pad_token_id should not be equal to eos_token_id')    
    cache_dir = os.path.expanduser("~/pyh/FariCon/research/hal-gaudisac/fairness/gpt_neo_cache")
     # 加载模型
    server_model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B",
                                                 cache_dir=cache_dir, 
                                                 pad_token_id=tokenizer.eos_token_id,
                                                 use_cache=False,
                                                 load_in_8bit=False, 
                                                trust_remote_code=True,
                                                device_map="auto"
                                                 )
    
    client_model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B",
                                                 cache_dir=cache_dir, 
                                                 pad_token_id=tokenizer.eos_token_id,
                                                 use_cache=False,
                                                 load_in_8bit=False, 
                                                trust_remote_code=True,
                                                device_map="auto"
                                                 )
    print("server model loading...")
    # print(server_model)
    print("client model loading...")
    # print(client_model)
    print("*** model finish ***")
    server_model.gradient_checkpointing_enable()
    client_model.gradient_checkpointing_enable()
    # note: use gradient checkpointing to save memory at the expense of slower backward pass.
    server_model.enable_input_require_grads()
    client_model.enable_input_require_grads()


    # LoRA配置
    config = LoraConfig(
        task_type="CAUSAL_LM",
        r=4,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.01,
    )
    # LoRA配置
    config1 = LoraConfig(
        task_type="CAUSAL_LM",
        r=4,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.01,
    )
    config2 = LoraConfig(
        task_type="CAUSAL_LM",
        r=4,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.01,
    )
    # LoRA配置
    config3 = LoraConfig(
        task_type="CAUSAL_LM",
        r=4,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.01,
    )
    # # LoRA配置
    config4 = LoraConfig(
        task_type="CAUSAL_LM",
        r=4,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.01,
    )

    # 设置中心服务器
    server_model = get_peft_model(server_model, config1)
    # server_model.print_trainable_parameters()
    server_model.config.torch_dtype = torch.float32




    client_num = args.client_num
    # client_num = 3  # 默认是3个客户端，法考（考察对法学知识的掌握），法院数据，法律咨询数据
#################公平性更改
    # client_weights = process_fairness(M_deo)
    client_weights = [1 / client_num] * client_num  #  权重可提前计算，按照数据量

    # client_models = [copy.deepcopy(server_model) for idx in range(client_num)]
    # client_models = [get_peft_model(client_model, config2),get_peft_model(client_model, config3),get_peft_model(client_model, config4)]
    # 假设你有多个配置（config2, config3, ..., configN）
    configs = [config2, config3, config4]  # 示例配置列表
    client_models = [get_peft_model(client_model, configs[i]) for i in range(client_num)]
    adv_models = [CategoricalEmb(client_model.get_input_embeddings()) for client_model in client_models]


    # 加载训练集
    print("Load Dataset")
    few_shot = 0
    fairness = 0.5
    fairprompt = False
#数据集 
# 调用示例
    # dict_users_train_dataset, dict_users_train_dataset_adv = Load_Dataset.load_data(args)
    data=args.data
    feature_to_process=args.feature_to_process
    data_size=args.data_size
    if args.run_adv==-0.1:
        feature_to_remove=True
    elif args.run_adv!=-0.1:
        feature_to_remove=False
    
    else:
        raise Exception('args.run_adv ({args.run_adv}) is not well to set the value of feature_to_remove.')   
    dict_users_train_dataset, dict_users_train_dataset_adv, test_dataset=Load_Dataset.getdata(args,data=data,few_shot=few_shot,number_of_samples=data_size, client_num=3, 
                                                       fairness=fairness, fairprompt=fairprompt, 
                                                       feature_to_process=feature_to_process, feature_to_remove=feature_to_remove
                                                       )
    print("Load Dataset finish")
#创建路径并储存数据文件；
    fed_all = training_args.output_dir
    print("CUDA_VISIBLE_DEVICES:", os.getenv("CUDA_VISIBLE_DEVICES"))
    print("CUDA Device Count:", torch.cuda.device_count())
    print("*** starting training ***")
    # client_models = [get_peft_model(client_model, config2)]
    for iter in range(args.rounds):
        w_locals = []
        # print("============ Train round {} ============".format(wi + a_iter * wk_iters))
        for client_idx, model in enumerate(client_models):

            Result = local_update(iter,client_idx,model,adv_models[client_idx],args,training_args,dict_users_train_dataset[client_idx],dict_users_train_dataset_adv[client_idx],client_tokenizer,server_model)
            w_local =Result
            w_locals.append(copy.deepcopy(w_local))#有client_num个数据

            del w_local
            torch.cuda.empty_cache()

        with torch.no_grad():
            server_model, client_models = communication( server_model, w_locals, client_models,client_weights)
            server_model.save_pretrained(fed_all)
            del w_locals
            torch.cuda.empty_cache()
            w_locals=[]
    server_model.save_pretrained(fed_all)
    print("server_model已保存")
    #测试test
    test_accuracy,test_avg_loss,M_deo,M_dpd= test(server_model,test_dataset,args.batch_size,client_tokenizer)
    # print(test_accuracy,'\n',test_avg_loss,'\n',M_deo,'\n',M_dpd)
    file_withadv='/home/chen/pyh/FedJudge-main/dataset/test_results.csv'
    file_withoutadv="/home/chen/pyh/FedJudge-main/dataset/test_results_withoutadv.csv"
    file=file_withadv
    # if args.run_adv==1:
    #     file=file_withadv
    # elif args.run_adv==0:
    #     file=file_withoutadv
    # else:
    #     print("args.run_adv赋值错误为",args.run_adv)
    #     raise ValueError 
    
    # 将数据追加到 CSV 文件
    with open(file, mode='a', newline='') as f:
        writer = csv.writer(f)
        # 如果文件是空的，首先写入表头
        if f.tell() == 0:  # 文件为空时
            writer.writerow(["dataset","sensitive_feature","dataset_size","test_size","run_adv","rounds","local_epochs","client_num","batch_size",'test_accuracy',  'M_deo', 'M_dpd'])  # 写入表头
        # 添加数据
        writer.writerow([args.data,args.feature_to_process,args.data_size,args.test_size,args.run_adv,args.rounds,args.local_epochs,args.client_num,args.batch_size,test_accuracy, M_deo, M_dpd])  # 写入数据
    print("测试结果已保存")
    return server_model


def main():
    # 进行一些配置和检查
    args, training_args = setup_everything()
    # 加载各种组件
    trainer = training(args, training_args)



if __name__ == "__main__":
    main()
