"""极简 .env 加载（纯标准库，不需要 python-dotenv）。

规则：
  * 读取「仓库根目录/.env」（没有则尝试当前目录的 .env）；
  * 已存在的环境变量不会被覆盖；
  * 支持简单格式：KEY=VALUE、# 注释、可选的 export 前缀与引号。

典型 .env 内容（复制 .env.example 后填写）：
    LLM_API_KEY=sk-xxx
    LLM_MODEL=deepseek-v4-flash          # 可选：不写就用命令行默认值
    LLM_BASE_URL=https://api.deepseek.com
    HF_ENDPOINT=https://hf-mirror.com    # 国内首次下载本地 bge 模型用
"""
from __future__ import annotations

import os
from pathlib import Path


def _candidates() -> list[Path]:
    root = Path(__file__).resolve().parent.parent   # 仓库根目录
    return [root / ".env", Path.cwd() / ".env"]


def load_dotenv() -> dict[str, str]:
    """加载 .env（幂等）。返回本次实际写入的环境变量（不含已存在的）。"""
    loaded: dict[str, str] = {}
    for p in _candidates():
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):]
            if "=" not in line:
                continue
            k, _, v = line.partition("=")
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if k and k not in os.environ:
                os.environ[k] = v
                loaded[k] = v
        break   # 只用第一个存在的 .env
    return loaded
