"""实验脚本共用工具：跑条件、读结果、构造语料变体。

设计：每个实验一个文件夹（experiments/eX_xxx/run.py），脚本通过
`run_condition()` 调用 run_e0.py 跑单个条件，再读取 answers.jsonl 汇总。

默认技术栈（按要求）：大模型走 API（OpenAI 兼容）、RAG 用 ChromaDB；
离线自检时加 `--mock`（mock 生成器 + mock 向量 + memory 索引）。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from .env import load_dotenv
from .victim import Embedder

load_dotenv()   # 让 .env 里的 key/模型名对实验脚本与子进程生效

ROOT = Path(__file__).resolve().parent.parent
RUNNER = ROOT / "run_e0.py"
DATA = ROOT / "data"


# ---------------------------------------------------------------- CLI 公共参数

def add_pipeline_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--generator", default="api", choices=["api", "mock"],
                    help="大模型后端：默认 API（OpenAI 兼容）")
    ap.add_argument("--llm-model", default=os.environ.get("LLM_MODEL", "deepseek-chat"))
    ap.add_argument("--llm-base-url",
                    default=os.environ.get("LLM_BASE_URL", "https://api.deepseek.com"))
    ap.add_argument("--embed-backend", default="local",
                    choices=["local", "api", "mock"])
    ap.add_argument("--embed-model",
                    default=os.environ.get("EMBED_MODEL", "BAAI/bge-small-zh-v1.5"))
    ap.add_argument("--embed-base-url", default=None,
                    help="embed-backend=api 时的 OpenAI 兼容地址")
    ap.add_argument("--llm-api-key-env", default="LLM_API_KEY")
    ap.add_argument("--llm-max-tokens",
                    type=int, default=int(os.environ.get("LLM_MAX_TOKENS", "2048")))
    ap.add_argument("--embed-api-key-env", default="EMBED_API_KEY")
    ap.add_argument("--embed-device", default=os.environ.get("EMBED_DEVICE", "auto"),
                    help="本地向量模型设备：auto / cpu / cuda:0")
    ap.add_argument("--embed-batch", type=int, default=32,
                    help="本地向量编码批大小（GPU 可调大到 128）")
    ap.add_argument("--chunker", default="fixed512",
                    choices=["fixed512", "fixed512o", "article"])
    ap.add_argument("--llm-temperature", type=float, default=0.0)
    ap.add_argument("--sleep", type=float, default=0.0,
                    help="每次 LLM 调用后的固定间隔秒数（限速用）")
    ap.add_argument("--index-backend", default="chroma",
                    choices=["chroma", "memory"],
                    help="RAG 向量库：默认 chromadb")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--cite-filter", default="on", choices=["on", "off"],
                    help="查询带『第X条』时只在该条文文档内检索（默认 on）")
    ap.add_argument("--prompt-mode", default="default", choices=["default", "strict"],
                    help="strict=禁止使用记忆、只依据检索条文（用于量化记忆效应）")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 题（冒烟用）")
    ap.add_argument("--outroot", type=Path, default=None)
    ap.add_argument("--mock", action="store_true",
                    help="离线自检：mock 生成器 + mock 向量 + memory 索引")


def pipeline_args(a: argparse.Namespace) -> list[str]:
    if a.mock:
        return ["--generator", "mock", "--embed-backend", "mock",
                "--index-backend", "memory"]
    args = ["--generator", a.generator,
            "--llm-model", a.llm_model, "--llm-base-url", a.llm_base_url,
            "--llm-api-key-env", a.llm_api_key_env,
            "--llm-max-tokens", str(a.llm_max_tokens),
            "--embed-backend", a.embed_backend, "--embed-model", a.embed_model,
            "--embed-api-key-env", a.embed_api_key_env,
            "--embed-device", a.embed_device,
            "--embed-batch", str(a.embed_batch),
            "--chunker", a.chunker,
            "--llm-temperature", str(a.llm_temperature),
            "--sleep", str(a.sleep),
            "--index-backend", a.index_backend,
            "--cite-filter", a.cite_filter]
    if a.prompt_mode != "default":
        args += ["--prompt-mode", a.prompt_mode]
    if a.embed_base_url:
        args += ["--embed-base-url", a.embed_base_url]
    return args


def limit_args(a: argparse.Namespace) -> list[str]:
    return ["--limit", str(a.limit)] if getattr(a, "limit", 0) else []


# ---------------------------------------------------------------- 跑条件 / 读结果

def run_condition(exp: str, cond_id: str, args_list: list[str],
                  outroot: Path | None = None) -> Path:
    outdir = (outroot or (ROOT / "results" / exp)) / cond_id
    cmd = [sys.executable, str(RUNNER), *args_list,
           "--outdir", str(outdir), "--resume"]
    print("\n$", " ".join(map(str, cmd[1:])), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)
    return outdir


def load_records(outdir: Path) -> list[dict]:
    rows = []
    with open(Path(outdir) / "answers.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_summary(outdir: Path) -> dict:
    return json.loads((Path(outdir) / "summary.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------- 语料变体

def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_current_only_corpus(versions_path: Path, out_path: Path) -> Path:
    """语料 A：每个条文只保留最新版本（用于 E1 的对照条件）。"""
    rows = _read_jsonl(versions_path)
    latest: dict[str, dict] = {}
    for r in rows:
        aid = r["article_id"]
        if aid not in latest or r["effective_from"] > latest[aid]["effective_from"]:
            latest[aid] = r
    _write_jsonl(list(latest.values()), out_path)
    print(f"[语料] current-only: {len(rows)} -> {len(latest)} 个版本 -> {out_path}")
    return out_path


def build_dedup_corpus(versions_path: Path, out_path: Path, embedder: Embedder,
                       threshold: float = 0.95, keep: str = "first") -> Path:
    """模拟 D2 去重：同一条文内，与已保留版本余弦相似度 ≥ 阈值的版本被删除。

    keep="first" 保留最早版本（最朴素的去重实现），"last" 保留最新版本。
    """
    rows = _read_jsonl(versions_path)
    by_aid: dict[str, list[dict]] = {}
    for r in rows:
        by_aid.setdefault(r["article_id"], []).append(r)
    kept: list[dict] = []
    for aid, vs in by_aid.items():
        vs.sort(key=lambda x: x["effective_from"])
        if keep == "last":
            vs = list(reversed(vs))
        embs = embedder.encode([v["text"] for v in vs])
        keep_flags = []
        for i, e in enumerate(embs):
            dup = any(sum(a * b for a, b in zip(e, embs[j])) >= threshold
                      for j in range(i) if keep_flags[j])
            keep_flags.append(not dup)
        kept += [v for v, k in zip(vs, keep_flags) if k]
    _write_jsonl(kept, out_path)
    print(f"[语料] dedup(阈值 {threshold}, keep={keep}): "
          f"{len(rows)} -> {len(kept)} 个版本 -> {out_path}")
    return out_path


def make_embedder(a: argparse.Namespace) -> Embedder:
    backend = "mock" if a.mock else a.embed_backend
    return Embedder(backend=backend, model=a.embed_model)
