#!/bin/bash
#SBATCH --partition=p_nlp
#SBATCH --job-name=holo-vllm
#SBATCH --output=/mnt/nlpgpu-io1/data/qianranm/research/GUI/CUA-Gym/slurm/logs/vllm.%j.out
#SBATCH --error=/mnt/nlpgpu-io1/data/qianranm/research/GUI/CUA-Gym/slurm/logs/vllm.%j.err
#SBATCH --gres=gpu:8
#SBATCH --constraint=48GBgpu
#SBATCH --exclusive
#SBATCH --mem=200G
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00

mkdir -p /mnt/nlpgpu-io1/data/qianranm/research/GUI/CUA-Gym/slurm/logs

echo "=== Starting vLLM on node: $(hostname) ==="
echo "=== GPUs: $CUDA_VISIBLE_DEVICES ==="

export HF_HOME=/nlp/data/huggingface_cache
export CC=gcc-13
export CXX=g++-13

source ~/miniconda3/etc/profile.d/conda.sh
conda activate gui

# pip install vllm -q

python -m vllm.entrypoints.openai.api_server \
  --model Hcompany/Holo-3.1-35B-A3B \
  --served-model-name holo-3.1 \
  --tensor-parallel-size 8 \
  --max-model-len 16384 \
  --dtype bfloat16 \
  --port 8000 \
  --host 0.0.0.0
