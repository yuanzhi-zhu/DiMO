#!/bin/bash

####################################################################################################
### general settings
####################################################################################################

export PYTHONPATH=$PYTHONPATH:./
# export NCCL_IB_GID_INDEX=3
# export OMP_NUM_THREADS=8
# export NCCL_IB_DISABLE=0
# export NCCL_SOCKET_IFNAME=eth0

export TRANSFORMERS_OFFLINE=1 # to avoid downloading
export HYDRA_FULL_ERROR=1 # to have the full traceback
export WANDB_CACHE_DIR=$SCRATCH/wandb_cache
export TMPDIR=$JOBSCRATCH
export HF_HUB_OFFLINE=1
export WANDB_MODE=offline
export IS_CLUSTER=True

echo LIST_IDX=$SLURM_ARRAY_TASK_ID

ln -s $WORK/inception-2015-12-05.pt /tmp/

num_nodes=$ARNOLD_NUM
host_ip=$METIS_WORKER_0_HOST

num_nodes=1
# host_ip=$METIS_WORKER_0_HOST
host_ip=127.0.0.1
host_port=2336
DID=0,1,2,3,4,5,6,7
# DID=0,1,2,3
# DID=4,5,6,7
# DID=0,1
DID=0
# calculate the number of GPUs
IFS=',' read -r -a array <<< "$DID"
num_gpu_per_node=${#array[@]}
# host_port=$(cut -d ',' -f 2 <<< "$METIS_WORKER_0_PORT") # which port to use, num 1/2/3/4...
# num_gpu_per_node=$ARNOLD_WORKER_GPU
num_proc=$((num_gpu_per_node * num_nodes))
node_rank=0 # echo -n "****** provide node rank: <<< "; read node_rank
if [ $# -eq 0 ]
  then
    echo "No argument provided. Using the default node rank: $node_rank"
  else
    node_rank=$1
    echo "Using node rank: $node_rank"
fi

DISTRIBUTED_ARGS="host_ip: $host_ip, host_port: $host_port, #proc_per_node: $num_gpu_per_node, node_rank/#nodes: $node_rank/$num_nodes <#<#<"
echo '****** DISTRIBUTED_ARGS: '$DISTRIBUTED_ARGS
echo "****** starting training "; echo " "; echo " ";


####################################################################################################
### training settings
####################################################################################################

# CUDA_VISIBLE_DEVICES=0 accelerate launch \
CUDA_VISIBLE_DEVICES=$DID accelerate launch \
        --main_process_ip $host_ip \
        --main_process_port $host_port \
        --num_machines $num_nodes \
        --machine_rank $node_rank \
        --num_processes $num_proc \
        --num_cpu_threads_per_process 8 \
    ./train_DIMO_MaskGit.py \
    --config ./configs/DIMO_MaskGit.yaml \
    --train_batch_size 64 \
    --gradient_accumulation_steps 1 \
    # --is_debug true \
    # --resume_from_checkpoint_path outputs/MaskGit_discrete_benchmark/DiMO-GPUS2-bs64-grad_accu2-glr1e-05-flr1e-05-Jeffreys_-0.2-fix_rrandom_0.1_0.9-emb_pert_fix_0.1-tcfg2.0-fcfg1.0-fcfgt1.0-r_mode_arccos-r_mode_f_arccos-reduce_sum-seed3407-bf16/2025-03-11T10-29/meta_checkpoints \
    # --max_train_steps 20001 \
    