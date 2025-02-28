from dataclasses import dataclass, field
from typing import Optional
import torch

@dataclass
class CustomizedArguments:
    """
    一些自定义参数
    """
    max_seq_length: int = field(metadata={"help": "输入最大长度"})
    train_file: str = field( metadata={"help": "训练集"})
    model_name_or_path: str = field(default = "baichuan-inc/baichuan-7B", metadata={"help": "预训练权重路径"})
    eval_file: Optional[str] = field(default="", metadata={"help": "the file of training data"})


@dataclass
class QLoRAArguments:
    """
    一些自定义参数
    """

    # max_seq_length: int = field(metadata={"help": "输入最大长度"})
    # train_file: str = field(metadata={"help": "训练集"})
    # train_file_fed1: str = field(metadata={"help": "训练集"})
    # train_file_fed2: str = field(metadata={"help": "训练集"})
    # train_file_fed3: str = field(metadata={"help": "训练集"})
    # output_dir_fed1: str = field(metadata={"help": "联邦模型保存位置"})
    # output_dir_fed2: str = field(metadata={"help": "联邦模型保存位置"})
    # output_dir_fed3: str = field(metadata={"help": "联邦模型保存位置"})
    # fed_epochs: int = field(metadata={"help": "联邦模型运行epoch"})
    output_dir: str = field(default="fed-all", metadata={"help": "联邦模型保存位置"})
###############
    data: str = field(default="adult", metadata={"help": "联邦模型运行轮次"})
    feature_to_process:str = field(default="sex", metadata={"help": "使用的敏感特征"})
    data_size: int = field(default=3000, metadata={"help": "数据量大小"})
    test_size: float = field(default=0.1, metadata={"help": "测试集大小"})
    run_adv: float  = field(default =0, metadata={"help": "是否运行adv_model"})
    rounds: int = field(default=4, metadata={"help": "联邦模型运行轮次"})
    local_epochs: int = field(default=5, metadata={"help": "联邦模型运行epoch"})
    client_num: int = field(default =1, metadata={"help": "输入最大长度"})
    batch_size: int = field(default =50, metadata={"help": "batch_size"})

################
# getdata_3000 3 10 3 50 

    device: str = field(default='cuda:0' if torch.cuda.is_available() else 'cpu', metadata={"help": "设备，例如 'cuda:0' 或 'cpu'"})
    model_name_or_path: str = field(default = "baichuan-inc/baichuan-7B", metadata={"help": "预训练权重路径"})

    server_model_name_or_path: str = field(default="baichuan-inc/baichuan-7B",metadata={"help": "服务器预训练权重路径"})
    client_model_name_or_path: str = field(default="baichuan-inc/baichuan-7B",metadata={"help": "客户端预训练权重路径"})
    task_type: str = field(default="", metadata={"help": "预训练任务：[sft, pretrain]"})
    eval_file: Optional[str] = field(default="", metadata={"help": "the file of training data"})
    lora_rank: Optional[int] = field(default=64, metadata={"help": "lora rank"})
    lora_alpha: Optional[int] = field(default=16, metadata={"help": "lora alpha"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "lora dropout"})
    lr: Optional[float] = field(default=1e-5, metadata={"help": "learning rate"})
    yita: Optional[float] = field(default=1, metadata={"help": "Tradeoff parameter"})
    # device: Optional[str] = field(default="cuda:0", metadata={"help": "Device to run on (e.g., 'cuda:0', 'cuda:1', 'cpu')"})