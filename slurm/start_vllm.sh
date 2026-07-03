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

# python -m vllm.entrypoints.openai.api_server \
#   --model Qwen/Qwen3.6-35B-A3B \
#   --served-model-name qwen-3.6 \
#   --tensor-parallel-size 8 \
#   --max-model-len 262144 \
#   --reasoning-parser qwen3 \
#   --dtype bfloat16 \
#   --port 8000 \
#   --host 0.0.0.0

# Serving parameters follow H Company's official vLLM guidance
# (https://hub.hcompany.ai/holo-desktop-cli/how-to/run-a-local-model-server):
#   --max-model-len 65537      official context length (their hosted API serves 64k)
#   --reasoning-parser qwen3   required so thinking runs before the structured-
#                              output grammar kicks in, and `content` comes back
#                              as clean JSON without the <think> block
#   --limit-mm-per-prompt      official image cap (agent loop keeps last 3)
# (The official command also has --enable-auto-tool-choice/--tool-call-parser,
# but those only matter for native function calling; we use structured outputs.)
python -m vllm.entrypoints.openai.api_server \
  --model Hcompany/Holo-3.1-35B-A3B \
  --served-model-name holo-3.1 \
  --tensor-parallel-size 8 \
  --max-model-len 65537 \
  --reasoning-parser qwen3 \
  --limit-mm-per-prompt '{"image": 5, "video": 0}' \
  --chat-template-content-format openai \
  --dtype bfloat16 \
  --port 8000 \
  --host 0.0.0.0
