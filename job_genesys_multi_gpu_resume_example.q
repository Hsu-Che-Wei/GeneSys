#!/bin/bash
#SBATCH --job-name=job_genesys_multi_gpu_duke
#SBATCH --output=job_output_%j.log
#SBATCH --error=job_error_%j.log
#SBATCH --time=240:00:00
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=110G
#SBATCH --partition=pbenfeylab-gpu
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=che-wei.hsu@duke.edu

# ------------------------------
# Rendezvous & rank environment
# ------------------------------
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=$((29500 + ($SLURM_JOB_ID % 1000)))
export NNODES=$SLURM_JOB_NUM_NODES
export NODE_RANK=$SLURM_NODEID
export RDZV_ID=$SLURM_JOB_ID
export TORCH_RUN_RDZV_TIMEOUT=1200

# ------------------------------
# NCCL / debug / networking
# Adjust IFNAME to your cluster
# ------------------------------
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# If you have Infiniband, use ib0; if not, use your ethernet (e.g., eth0)
# export NCCL_SOCKET_IFNAME=ib0
export NCCL_SOCKET_IFNAME=eth0
# If you do NOT have IB or it's flaky, uncomment the next line:
export NCCL_IB_DISABLE=1

# Prevent unexpected CPU thread contention
export OMP_NUM_THREADS=4

# Deterministic runs (optional). If you pass --deterministic to the script,
# uncomment the next line to keep CuBLAS deterministic too:
# export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ------------------------------
# Python env & working dir
# ------------------------------
cd /hpc/group/pbenfeylab/CheWei/CW_data/genesys
source activate genesys

# ------------------------------
# Launch (1 GPU per node)
# Increase rendezvous timeout for HPC networks
# ------------------------------

## --batch_size = 512 / number of GPU deployed
#srun torchrun \
#  --nnodes=${NNODES} \
#  --nproc_per_node=1 \
#  --node_rank=${NODE_RANK} \
#  --rdzv_id=${RDZV_ID} \
#  --rdzv_backend=c10d \
#  --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
#  ./genesys_multi_gpu/genesys_cli_multi_gpu.py \
#    --train --raw_counts \
#    --anndata ./Root_Atlas_RNA_downsampled_100000_cells.h5ad \
#    --bprint ./lineage.txt \
#    --epochs 100 --batch_size 128 \
#    --path ./root_100k_ckpt \
#    --amp_off \
#    --dist --num_workers 1 \
#    --sync_bn --seed 42 \
#    --verbose

## Resume training
srun torchrun \
  --nnodes=${NNODES} \
  --nproc_per_node=1 \
  --node_rank=${NODE_RANK} \
  --rdzv_id=${RDZV_ID} \
  --rdzv_backend=c10d \
  --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
  ./genesys_multi_gpu/genesys_cli_multi_gpu.py \
    --train --raw_counts \
    --anndata ./Root_Atlas_RNA_downsampled_100000_cells.h5ad \
    --bprint ./lineage.txt \
    --epochs 100 --batch_size 128 \
    --path ./root_100k_ckpt \
    --amp_off \
    --dist --pin_memory --persistent_workers --num_workers 1 \
    --sync_bn --seed 42 \
    --verbose \
    --resume_from ./root_100k_ckpt/genesys_training_cycle4_best.pth \ ## Checkpoint to resume from
    --training_logs ../../Bash_scripts/job_output_37873017.log ## training log that contains the training losses for every epoch before interruption   



