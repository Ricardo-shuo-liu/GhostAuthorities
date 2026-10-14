"""快速自检：在你本机验证「API 大模型 + ChromaDB」链路是否就绪。

用法（在仓库根目录执行）：
    pip install chromadb            # 先装好
    python scripts/check_chroma.py  # 用 toy 数据 + mock 向量跑通 Chroma 后端

它做两件事：
  1) 用 --index-backend chroma 跑一次基线（验证 Chroma 读写与检索）；
  2) 再跑一次 A1 注入 + unknown_dates=accept（验证元数据过滤逻辑）。
真实 API 是否可用，用 `--generator api` 加你的 key 再跑一遍即可。
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNNER = ROOT / "run_e0.py"


def run(tag: str, extra: list[str]) -> None:
    outdir = ROOT / "results" / f"check_chroma_{tag}"
    cmd = [sys.executable, str(RUNNER),
           "--corpus", "data/toy/corpus.jsonl",
           "--questions", "data/toy/questions.jsonl",
           "--background", "data/toy/background.jsonl",
           "--embed-backend", "mock", "--generator", "mock",
           "--index-backend", "chroma",
           "--outdir", str(outdir), *extra]
    print("\n$", " ".join(cmd[1:]), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    if not (ROOT / "data" / "toy" / "corpus.jsonl").exists():
        print("[提示] 先运行 python scripts/make_toy_data.py")
        subprocess.run([sys.executable, "scripts/make_toy_data.py"], cwd=ROOT, check=True)
    run("base", ["--attack", "none"])
    run("a1_accept", ["--attack", "A1", "--as-of", "fact",
                      "--unknown-dates", "accept"])
    print("\nOK：ChromaDB 后端可用。接下来把 --generator 换成 api、"
          "--embed-backend 换成 local（或 api）跑真实实验。")


if __name__ == "__main__":
    main()
