export HF_ENDPOINT="https://hf-mirror.com" 
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 #export TMPDIR="/data/tmp"
export CUDA_VISIBLE_DEVICES=1 
# python main_fed_base.py   
# 直接定义所有参数组合--data "getdata_3000_nogender"
python main_fed_base.py  --data_size 192 --run_adv 1  --feature_to_process "marital-status" --rounds 1 --local_epochs 1 --client_num 3 --batch_size 50
# --train_args_file train_args/lora/baichuan-7b-fed-lora-base.json
# --server_model_name_or_path "EleutherAI/gpt-neo-125m" --client_model_name_or_path "EleutherAI/gpt-neo-125m"