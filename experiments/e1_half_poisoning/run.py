#!/usr/bin/env python3
"""E1 — 多版本共存本身会不会让系统答错？（对应设计书 H1）

设计：同一批题，只改一件事——语料里有没有旧版本。
  语料 A（对照）：每个条文只留现行版本；
  语料 B（实验）：现行版 + 旧版本（= data/corpus.jsonl 默认语料，即"半投毒"状态）。
预期（H1 成立）：G2（应答题新版）在 B 上比 A 掉 ≥20 个百分点，且按条文聚类显著。

用法（在仓库根目录执行）：
    python experiments/e1_half_poisoning/run.py                # 真实 API + ChromaDB
    python experiments/e1_half_poisoning/run.py --mock         # 离线自检
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ragpoison import experiments as exp  # noqa: E402
from ragpoison import stats  # noqa: E402

EXP = "e1_half_poisoning"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    exp.add_pipeline_args(ap)
    ap.add_argument("--corpus", type=Path, default=exp.DATA / "corpus.jsonl")
    ap.add_argument("--questions", type=Path, default=exp.DATA / "questions.jsonl")
    ap.add_argument("--background", type=Path, default=exp.DATA / "background.jsonl")
    a = ap.parse_args()

    corpus_a = exp.build_current_only_corpus(
        a.corpus, exp.DATA / "corpus_current_only.jsonl")

    common = [*exp.pipeline_args(a), "--questions", str(a.questions),
              "--background", str(a.background), "--k", str(a.k),
              *exp.limit_args(a)]
    dirs = {}
    dirs["A_current_only"] = exp.run_condition(
        EXP, "A_current_only", [*common, "--corpus", str(corpus_a), "--attack", "none"], a.outroot)
    dirs["B_multi_version"] = exp.run_condition(
        EXP, "B_multi_version", [*common, "--corpus", str(a.corpus), "--attack", "none"], a.outroot)

    rec_a = exp.load_records(dirs["A_current_only"])
    rec_b = exp.load_records(dirs["B_multi_version"])
    acc_a, acc_b = stats.group_accuracy(rec_a), stats.group_accuracy(rec_b)

    print(f"\n== E1 结果（题组准确率）==")
    print(f"{'题组':<6} {'语料A(仅现行)':>12} {'语料B(含旧版)':>12} {'变化':>10}")
    for g in ("G1", "G2", "G3"):
        d = acc_b.get(g, 0.0) - acc_a.get(g, 0.0)
        print(f"{g:<6} {acc_a.get(g, 0):>12.3f} {acc_b.get(g, 0):>12.3f} {d:>+10.3f}")

    boot = stats.paired_cluster_bootstrap(
        rec_a, rec_b, where=lambda r: r.get("group") == "G2")
    print(f"\nG2 配对差值（B-A）= {boot['delta']:+.3f} "
          f"95%CI [{boot['lo']:+.3f}, {boot['hi']:+.3f}] p≈{boot['p']:.4f} "
          f"(按条文聚类, n={boot['n']})")
    ok = boot["delta"] <= -0.20 and boot["p"] < 0.05
    print(f"判定 H1：{'成立（G2 下降 ≥20pp 且显著）' if ok else '未达判据，检查检索器/题目'}") 

    outroot = a.outroot or (exp.ROOT / "results" / EXP)
    outroot.mkdir(parents=True, exist_ok=True)
    (outroot / "report.json").write_text(json.dumps({
        "experiment": EXP, "group_accuracy_A": acc_a, "group_accuracy_B": acc_b,
        "G2_paired_delta": boot, "H1_pass": ok,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n报告 -> {outroot / 'report.json'}")


if __name__ == "__main__":
    main()
