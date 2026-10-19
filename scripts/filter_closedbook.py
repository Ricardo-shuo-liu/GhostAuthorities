"""闭卷筛选：多模型（默认 7 个）无检索提问；任何模型答对 → 剔除该题。

结果与过程全部落盘：
  clean_queries.jsonl   通过筛选的干净题库（单独文件，old/new 合并）
  questions_old.jsonl   干净题库按 bank 拆分（old）
  questions_new.jsonl   干净题库按 bank 拆分（new）
  rejected.jsonl        被剔除的题 + 判定证据
  traces.jsonl          每一次模型调用的完整留痕：
                        qid/model/sample/prompt/content/reasoning_content/
                        finish_reason/usage/correct/gold/ts
  cache_llm.jsonl       请求缓存（含 reasoning，重跑不重复计费）
  report.json           统计（模型命中数、old/new 与条文分布）

判定规则：只要 7 个模型中有**任何一个**在任意一次采样中答对
（最后一行 `第X条｜Y年` 与 gold 完全一致），该题即剔除。

用法（仓库根目录）：
    # 先小规模校准（200 题）
    python scripts/filter_closedbook.py --limit 200
    # 全量
    python scripts/filter_closedbook.py
    # 可选：所有模型都问一遍（不提前停止），留痕更完整但调用量×模型数
    python scripts/filter_closedbook.py --full-eval

模型列表：configs/filter_models.txt（一行一个，# 注释）；
环境变量：SILICONFLOW_API_KEY / SILICONFLOW_BASE_URL。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from ragpoison.env import load_dotenv  # noqa: E402
from ragpoison.common import cn2int  # noqa: E402

DATA = ROOT / "data"

_NUM = r"[0-9零一两二三四五六七八九十百千]+"
LINE_RE = re.compile(
    rf"第\s*({_NUM}(?:-\d+)?)\s*(?:条)?(?:\s*之\s*({_NUM}))?"
    rf"\s*[|｜]\s*(\d+)\s*年")
ART_RE = re.compile(
    rf"第\s*({_NUM}(?:-\d+)?)\s*(?:条)?(?:\s*之\s*({_NUM}))?")
YEAR_RE = re.compile(rf"({_NUM})\s*年")


def normalize_article_id(main: str, sub: str | None = None) -> str | None:
    """把 253、253-1、第二百五十三条之一 统一成 253 或 253-1。"""
    if "-" in main:
        left, right = main.split("-", 1)
        a, b = cn2int(left), cn2int(right)
        return f"{a}-{b}" if a is not None and b is not None else main
    a = cn2int(main)
    if a is None:
        return None
    if sub:
        b = cn2int(sub)
        if b is None:
            return None
        return f"{a}-{b}"
    return str(a)


def parse_answer(text: str) -> tuple[str | None, int | None]:
    for line in reversed([l for l in (text or "").splitlines() if l.strip()][-4:]):
        m = LINE_RE.search(line)
        if m:
            return normalize_article_id(m.group(1), m.group(2)), int(m.group(3))
    aid = None
    years = None
    m = ART_RE.search(text or "")
    if m:
        aid = normalize_article_id(m.group(1), m.group(2))
    ys = [cn2int(y) for y in YEAR_RE.findall(text or "")]
    ys = [y for y in ys if y is not None and 0 < y <= 100]
    if ys:
        years = max(ys)
    return aid, years


def http_post(url: str, payload: dict, api_key: str, timeout: float,
              retries: int = 3) -> dict:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    last = "unknown"
    for attempt in range(retries + 1):
        req = urllib.request.Request(
            url, data=body, method="POST",
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {api_key}"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = e.read().decode("utf-8", "ignore")[:300]
            except Exception:
                pass
            last = f"HTTP {e.code}: {detail}"
            if e.code in (408, 409, 429, 500, 502, 503, 504) and attempt < retries:
                time.sleep(2 * (2 ** attempt))
                continue
            raise RuntimeError(last)
        except Exception as e:
            last = repr(e)
            if attempt < retries:
                time.sleep(2 * (2 ** attempt))
                continue
            raise RuntimeError(last)
    raise RuntimeError(last)


class Cache:
    """缓存完整响应（含 reasoning_content），重跑不重复计费。"""

    def __init__(self, path: Path):
        self.path = path
        self.data: dict[str, dict] = {}
        self.lock = threading.Lock()
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    r = json.loads(line)
                    self.data[r["key"]] = r["response"]
                except Exception:
                    continue

    def get(self, key: str):
        with self.lock:
            return self.data.get(key)

    def set(self, key: str, response: dict):
        with self.lock:
            self.data[key] = response
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"key": key, "response": response},
                                   ensure_ascii=False) + "\n")


def ask(base_url: str, model: str, api_key: str, prompt: str, cache: Cache,
        temperature: float, max_tokens: int, timeout: float) -> dict:
    key = hashlib.sha1(f"{model}\x1f{temperature}\x1f{prompt}".encode()).hexdigest()
    hit = cache.get(key)
    if hit is not None:
        return hit
    resp = http_post(base_url.rstrip("/") + "/chat/completions", {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature, "max_tokens": max_tokens,
    }, api_key, timeout)
    choice = (resp.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    out = {
        "content": msg.get("content") or "",
        "reasoning_content": msg.get("reasoning_content") or "",
        "finish_reason": choice.get("finish_reason"),
        "usage": resp.get("usage"),
    }
    cache.set(key, out)
    return out


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path,
                    default=DATA / "queries" / "candidates_10000.jsonl")
    ap.add_argument("--out-dir", type=Path, default=DATA / "queries" / "filtered")
    ap.add_argument("--models-file", type=Path,
                    default=ROOT / "configs" / "filter_models.txt")
    ap.add_argument("--models", default=None, help="逗号分隔，覆盖模型列表文件")
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--limit-per-cell", type=int, default=0,
                    help="按 article_id × bank 分层，每格最多取 N 题（优先于 --limit）")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--full-eval", action="store_true",
                    help="命中后仍继续问完所有模型/采样（留痕最完整，调用量最大）")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    if a.models:
        models = [m.strip() for m in a.models.split(",") if m.strip()]
    else:
        models = [l.strip() for l in a.models_file.read_text(encoding="utf-8").splitlines()
                  if l.strip() and not l.strip().startswith("#")]
    if not models:
        raise SystemExit("模型列表为空：编辑 configs/filter_models.txt 或用 --models 指定")
    if not a.input.exists():
        raise SystemExit(f"输入文件不存在：{a.input}")

    questions = []
    with open(a.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    if a.limit_per_cell:
        by_cell: dict[tuple[str, str], list[dict]] = {}
        for q in questions:
            by_cell.setdefault((q.get("article_id"), q.get("bank")), []).append(q)
        questions = [
            q for cell in sorted(by_cell)
            for q in by_cell[cell][:a.limit_per_cell]
        ]
    elif a.limit:
        questions = questions[:a.limit]

    max_calls = len(questions) * len(models) * a.samples
    print(f"题目 {len(questions)} × 模型 {len(models)} × 采样 {a.samples} "
          f"→ 最多 {max_calls} 次调用"
          + ("（--full-eval：全部调用都会发生）" if a.full_eval else "（命中即停）"))
    if a.dry_run:
        return

    key = os.environ.get("SILICONFLOW_API_KEY", "")
    if not key:
        raise SystemExit("环境变量 SILICONFLOW_API_KEY 为空")
    base_url = os.environ.get("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")
    a.out_dir.mkdir(parents=True, exist_ok=True)
    cache = Cache(a.out_dir / "cache_llm.jsonl")
    f_clean = a.out_dir / "clean_queries.jsonl"
    f_old = a.out_dir / "questions_old.jsonl"
    f_new = a.out_dir / "questions_new.jsonl"
    f_rej = a.out_dir / "rejected.jsonl"
    f_trace = a.out_dir / "traces.jsonl"
    lock = threading.Lock()

    done: dict[str, str] = {}
    for p, tag in ((f_clean, "pass"), (f_rej, "reject")):
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                try:
                    done[json.loads(line)["qid"]] = tag
                except Exception:
                    continue
    todo = [q for q in questions if q["qid"] not in done]
    print(f"待筛 {len(todo)}（已完成 {len(done)}）")

    stats = {
        "input": str(a.input), "models": models, "samples": a.samples,
        "pass": 0, "reject": 0, "errors": 0,
        "model_attempts": Counter(), "model_hits": Counter(),
        "pass_by_cell": Counter(), "reject_by_cell": Counter(),
    }

    def write_jsonl(path: Path, rec: dict):
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def judge(q: dict) -> tuple[str, dict]:
        gold = q["gold_nuggets"]
        prompt = ("请只凭你已有的知识回答下面的问题，不要假设有任何检索结果或法条文本；"
                  "如果你无法确定，请如实说明。\n\n" + q["query"])
        hit = None
        attempts = []
        for model in models:
            for si in range(a.samples):
                with lock:
                    stats["model_attempts"][model] += 1
                try:
                    resp = ask(base_url, model, key, prompt, cache,
                               a.temperature + 0.1 * si, a.max_tokens, a.timeout)
                    err = None
                except Exception as e:
                    resp = {"content": "", "reasoning_content": "",
                            "finish_reason": "error", "usage": None}
                    err = str(e)[:300]
                    with lock:
                        stats["errors"] += 1
                aid, years = parse_answer(resp["content"])
                ok = (aid == str(gold["article_id"]) and years == int(gold["years"]))
                trace = {
                    "ts": datetime.now().isoformat(timespec="seconds"),
                    "qid": q["qid"], "article_id": q.get("article_id"),
                    "bank": q.get("bank"), "model": model, "sample": si + 1,
                    "prompt": prompt,
                    "content": resp["content"],
                    "reasoning_content": resp["reasoning_content"],
                    "finish_reason": resp["finish_reason"],
                    "usage": resp["usage"],
                    "correct": ok, "gold": gold, "error": err,
                }
                with lock:
                    write_jsonl(f_trace, trace)
                attempts.append({"model": model, "sample": si + 1, "correct": ok})
                if ok:
                    if hit is None:
                        hit = trace
                        with lock:
                            stats["model_hits"][model] += 1
                    if not a.full_eval:
                        break
            if hit is not None and not a.full_eval:
                break
        cell = f"{q.get('article_id')}/{q.get('bank')}"
        if hit is not None:
            rec = {"qid": q["qid"], "article_id": q.get("article_id"),
                   "bank": q.get("bank"), "gold": gold,
                   "hit_model": hit["model"], "hit_sample": hit["sample"],
                   "hit_answer": hit["content"][:400],
                   "hit_reasoning": hit["reasoning_content"][:2000],
                   "attempts": attempts}
            with lock:
                stats["reject"] += 1
                stats["reject_by_cell"][cell] += 1
                write_jsonl(f_rej, rec)
            return "reject", rec
        with lock:
            stats["pass"] += 1
            stats["pass_by_cell"][cell] += 1
            write_jsonl(f_clean, q)
            write_jsonl(f_old if q.get("bank") == "old" else f_new, q)
        return "pass", q

    with ThreadPoolExecutor(max_workers=max(1, a.workers)) as ex:
        futs = {ex.submit(judge, q): q for q in todo}
        for i, f in enumerate(as_completed(futs), 1):
            f.result()
            if i % 20 == 0 or i == len(todo):
                print(f"  进度 {i}/{len(todo)} | 通过 {stats['pass']} "
                      f"剔除 {stats['reject']} 错误 {stats['errors']}")

    report = {
        "input": stats["input"], "models": models, "samples": a.samples,
        "n_input": len(questions),
        "n_pass": stats["pass"], "n_reject": stats["reject"],
        "n_errors": stats["errors"],
        "model_attempts": dict(stats["model_attempts"]),
        "model_hits": dict(stats["model_hits"]),
        "pass_by_cell": dict(sorted(stats["pass_by_cell"].items())),
        "reject_by_cell": dict(sorted(stats["reject_by_cell"].items())),
        "outputs": {
            "clean_queries": str(f_clean), "questions_old": str(f_old),
            "questions_new": str(f_new), "rejected": str(f_rej),
            "traces": str(f_trace),
        },
    }
    (a.out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n完成：通过 {stats['pass']} / 剔除 {stats['reject']} / 错误 {stats['errors']}")
    print(f"干净题库（单独文件）-> {f_clean}")
    print(f"筛选留痕 -> {f_trace}")
    print(f"报告 -> {a.out_dir / 'report.json'}")


if __name__ == "__main__":
    main()
