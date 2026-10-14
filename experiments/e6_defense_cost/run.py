#!/usr/bin/env python3
"""E6 — 干净语料上，防御本身的代价有多大？（对应设计书 H3 补充）

语料 B（干净、无攻击）× {无防御 D0, D2 去重, D5 版本感知}，同一批题。
判定：D5 的 Clean-Δ ≈ 0；D2 的 Clean-Δ 明显为负（误删合法多版本）。
同时输出检索耗时（retrieval_ms）作为延迟代价的参考。

用法：
    python experiments/e6_defense_cost/run.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ragpoison import experiments as exp  # noqa: E402
from ragpoison import stats  # noqa: E402

EXP = "e6_defense_cost"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    exp.add_pipeline_args(ap)
    ap.add_argument("--corpus", type=Path, default=exp.DATA / "corpus.jsonl")
    ap.add_argument("--questions", type=Path, default=exp.DATA / "questions.jsonl")
    ap.add_argument("--background", type=Path, default=exp.DATA / "background.jsonl")
    ap.add_argument("--dedup-threshold", type=float, default=0.95)
    a = ap.parse_args()

    dedup = exp.build_dedup_corpus(
        a.corpus, exp.DATA / f"corpus_dedup_{a.dedup_threshold}.jsonl",
        exp.make_embedder(a), threshold=a.dedup_threshold)
    common = [*exp.pipeline_args(a), "--questions", str(a.questions),
              "--background", str(a.background), "--attack", "none",
              "--k", str(a.k), *exp.limit_args(a)]

    conds = {
        "D0_none": [*common, "--corpus", str(a.corpus)],
        "D2_dedup": [*common, "--corpus", str(dedup)],
        "D5_version": [*common, "--corpus", str(a.corpus),
                       "--as-of", "fact", "--unknown-dates", "reject"],
    }
    recs, res = {}, {}
    for cid, args in conds.items():
        d = exp.run_condition(EXP, cid, args, a.outroot)
        recs[cid] = exp.load_records(d)
        res[cid] = {"acc": stats.accuracy(recs[cid]),
                    "retrieval_ms": sum(r.get("retrieval_ms", 0) for r in recs[cid])
                    / max(len(recs[cid]), 1)}

    print(f"\n== E6 干净语料上的防御代价 ==")
    print(f"{'条件':<10} {'准确率':>8} {'Clean-Δ':>9} {'检索耗时(ms)':>12}")
    for cid in conds:
        delta = res[cid]["acc"] - res["D0_none"]["acc"]
        print(f"{cid:<10} {res[cid]['acc']:>8.3f} {delta:>+9.3f} "
              f"{res[cid]['retrieval_ms']:>12.1f}")
    d5_delta = res["D5_version"]["acc"] - res["D0_none"]["acc"]
    d2_delta = res["D2_dedup"]["acc"] - res["D0_none"]["acc"]
    ok = d5_delta >= -0.05 and d2_delta <= -0.10
    print(f"\n判定：D5 Clean-Δ={d5_delta:+.3f}（应≈0），"
          f"D2 Clean-Δ={d2_delta:+.3f}（应明显为负）→ "
          f"{'成立（要选对防御）' if ok else '未达判据'}")

    outroot = a.outroot or (exp.ROOT / "results" / EXP)
    outroot.mkdir(parents=True, exist_ok=True)
    (outroot / "report.json").write_text(json.dumps({
        "experiment": EXP, "results": res,
        "D5_clean_delta": d5_delta, "D2_clean_delta": d2_delta, "pass": ok,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"报告 -> {outroot / 'report.json'}")


if __name__ == "__main__":
    main()
