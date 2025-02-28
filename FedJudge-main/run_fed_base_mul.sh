
#!/bin/bash
export HF_ENDPOINT=https://hf-mirror.com 
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 #export TMPDIR="/data/tmp"
export CUDA_VISIBLE_DEVICES=0


data_values=(  "adult" )
# feature_to_process_values=( "sex" "marital-status" "race" )
feature_to_process_values=( "sex"  )
# data_values=(  "german_credit" )
# feature_to_process_values=( "sex" "marital-status"  )

# data_values=(  "bank_marketing" )
# feature_to_process_values=(  "marital-status" "education"  )

# data_values=(  "compas" )
# feature_to_process_values=( "sex" "race" )

# "adult" "german_credit"  "bank_marketing" "compas"

#"sex" "marital-status" "race""education" 
data_size_values=(1000)
test_size_values=(0.4)

run_adv_values=(-2)
# ( $(seq 0.2 0.04 0.6)   )
rounds_values=(5 )
local_epochs_values=( 1  )
client_num_values=(  3)
# batch_size_values=(50)
batch_size_values=(50)

# data_values=(  "compas" )
# # "adult" "german_credit"  "bank_marketing" "compas"
# feature_to_process_values=(  "sex" )
# #"sex" "Foreign-worker" "marital-status" "race"
# data_size_values=(200)
# test_size_values=(0.4)
# run_adv_values=($(seq 0.2 0.04 0.6))
# # (-0.1 $(seq 0.2 0.2 0.8))
# #(-0.1 $(seq 0 0.2 1))

# rounds_values=(5 )
# local_epochs_values=( 1  )
# client_num_values=(  3)
# batch_size_values=(10)

# data_values=(  "compas" )
# # "adult" "german_credit"  "bank_marketing" "compas"
# feature_to_process_values=(  "sex" )
# #"sex" "Foreign-worker" "marital-status" "race"
# data_size_values=(200)
# test_size_values=(0.4)
# run_adv_values=(0.92)


# rounds_values=(5 )
# local_epochs_values=( 1  )
# client_num_values=(  3)
# batch_size_values=(10)

# 遍历每个参数的所有可能值
for data in "${data_values[@]}"; do
  for feature_to_process in "${feature_to_process_values[@]}"; do
    for data_size in "${data_size_values[@]}"; do
      for test_size in "${test_size_values[@]}"; do  # 新增的 test_size 循环
        for run_adv in "${run_adv_values[@]}"; do    
          for rounds in "${rounds_values[@]}"; do
            for local_epochs in "${local_epochs_values[@]}"; do
              for client_num in "${client_num_values[@]}"; do
                for batch_size in "${batch_size_values[@]}"; do
                  # 打印当前的参数组合
                  echo "Running with data=$data, feature_to_process=$feature_to_process, data_size=$data_size, test_size=$test_size, run_adv=$run_adv, rounds=$rounds, local_epochs=$local_epochs, client_num=$client_num, batch_size=$batch_size"

                  # 运行 Python 脚本，并传递参数
                  python main_fed_base.py  --data $data --feature_to_process $feature_to_process --data_size $data_size --test_size $test_size --run_adv $run_adv  --rounds $rounds --local_epochs $local_epochs --client_num $client_num --batch_size $batch_size
                done
              done
            done
          done
        done
      done  # 结束 test_size 循环
    done
  done
done

# #!/bin/bash
# export HF_ENDPOINT=https://hf-mirror.com
# export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
# export TMPDIR="/data/tmp"

# # 定义参数
# data_values=("adult")
# feature_to_process_values=("sex" "marital-status" "race")
# data_size_values=(1000)
# run_adv_values=(1 )
# rounds_values=(5)
# local_epochs_values=(1)
# client_num_values=(3)
# batch_size_values=(50)

# # 检测空闲 GPU
# get_idle_gpu() {
#     nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | \
#     awk -F ', ' '$2 < 15000 {print $1}' | \
#     head -n 1
# }

# # 生成所有参数组合
# generate_commands() {
#     for data in "${data_values[@]}"; do
#         for feature_to_process in "${feature_to_process_values[@]}"; do
#             for data_size in "${data_size_values[@]}"; do
#                 for run_adv in "${run_adv_values[@]}"; do
#                     for rounds in "${rounds_values[@]}"; do
#                         for local_epochs in "${local_epochs_values[@]}"; do
#                             for client_num in "${client_num_values[@]}"; do
#                                 for batch_size in "${batch_size_values[@]}"; do
#                                     echo "python main_fed_base.py --data $data --feature_to_process $feature_to_process --data_size $data_size --run_adv $run_adv --rounds $rounds --local_epochs $local_epochs --client_num $client_num --batch_size $batch_size"
#                                 done
#                             done
#                         done
#                     done
#                 done
#             done
#         done
#     done
# }

# # 逐个运行任务
# generate_commands | while read -r command; do
#     while true; do
#         gpu=$(get_idle_gpu)
#         if [ -n "$gpu" ]; then
#             echo "Running on GPU $gpu: $command"
#             CUDA_VISIBLE_DEVICES=$gpu $command
#             break
#         else
#             echo "No idle GPU found. Waiting..."
#             sleep 10
#         fi
#     done
# done

