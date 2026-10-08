# =============================================================================
# 法律 RAG 语料投毒 · 环境搭建（Linux / CUDA）
# 用法： bash setup.sh
# =============================================================================
set -euo pipefail

ENV_NAME="${ENV_NAME:-legalrag}"
PY_VER="${PY_VER:-3.10}"

echo "==> [1/6] 创建 conda 环境 ${ENV_NAME} (python ${PY_VER})"
if ! command -v conda &>/dev/null; then
  echo "未找到 conda，请先安装 miniconda: https://docs.conda.io/en/latest/miniconda.html"
  exit 1
fi
# 初始化 conda 的 shell 钩子（非交互 bash 下需要）
eval "$(conda shell.bash hook)"
conda create -y -n "${ENV_NAME}" python="${PY_VER}"
conda activate "${ENV_NAME}"

echo "==> [2/6] 安装 PyTorch（按你的 CUDA 版本调整；这里默认 CUDA 12.1）"
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo "==> [3/6] 安装检索与向量库组件"
pip install faiss-gpu sentence-transformers rank-bm25
# 若 faiss-gpu 装不上，退回 CPU 版： pip install faiss-cpu

echo "==> [4/6] 安装重排器"
pip install FlagEmbedding

echo "==> [5/6] 安装推理与训练组件"
pip install vllm transformers datasets accelerate peft trl bitsandbytes
pip install flash-attn --no-build-isolation || echo "flash-attn 安装失败可忽略，只是会慢一些"

echo "==> [6/6] 安装工具类"
pip install numpy pandas pyyaml tqdm rich tabulate openai

echo
echo "==> 完成。请执行： conda activate ${ENV_NAME}"
echo "==> 验证：       python -c 'import torch, faiss, vllm, trl; print(torch.cuda.is_available())'"
