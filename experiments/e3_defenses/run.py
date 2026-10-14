#!/usr/bin/env python3
"""E3 — 业界推荐的防线拦得住吗？（对应设计书 H3）

固定 E2 的攻击（A3×5），逐个加防御，测「挡攻击」与「引用校验盖章」两件事。

已实现：
  D1 引用校验  ：在攻击条件下统计引用通过率（预期 ≈100% → 放大器）
  D2 去重      ：对语料做相似度去重后重跑（预期挡攻击但误删合法多版本；
                 误删代价由 E6 在干净语料上量化）
未实现（需后续补充，脚本会明确报错而不是假装跑过）：
  D3 共识投票、D4 密度/困惑度检测

用法：
    python experiments/e3_defenses/run.py
    python experiments/e3_defenses/run.py --mock
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ragpoison import experiments as exp  # noqa: E402
from ragpoison import stats  # noqa: E402

EXP = "e3_defenses"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    exp.add_pipeline_args(ap)
    ap.add_argument("--corpus", type=Path, default=exp.DATA / "corpus.jsonl")
    ap.add_argument("--questions", type=Path, default=exp.DATA / "questions.jsonl")
    ap.add_argument("--background", type=Path, default=exp.DATA / "background.jsonl")
    ap.add_argument("--n-inject", type=int, default=5)
    ap.add_argument("--dedup-threshold", type=float, default=0.95)
    ap.add_argument("--include-d3", action="store_true", help="D3 尚未实现")
    ap.add_argument("--include-d4", action="store_true", help="D4 尚未实现")
    a = ap.parse_args()
    if a.include_d3 or a.include_d4:
        raise SystemExit("D3（共识投票）/ D4（异常检测）尚未实现，请先跑 D1/D2。")

    common = [*exp.pipeline_args(a), "--questions", str(a.questions),
              "--background", str(a.background), "--k", str(a.k),
              *exp.limit_args(a), "--attack", "A3", "--n-inject", str(a.n_inject)]
    recs, accs = {}, {}
    for cid, corpus in (("none", a.corpus),
                        ("D2_dedup", exp.build_dedup_corpus(
                            a.corpus, exp.DATA / f"corpus_dedup_{a.dedup_threshold}.jsonl",
                            exp.make_embedder(a), threshold=a.dedup_threshold))):
        d = exp.run_condition(EXP, cid, [*common, "--corpus", str(corpus)], a.outroot)
        recs[cid] = exp.load_records(d)
        accs[cid] = stats.group_accuracy(recs[cid])

    print(f"\n== E3 结果（攻击 A3×{a.n_inject} 下）==")
    print(f"{'防御':<10} {'G2 准确率':>10} {'引用通过率':>10}")
    for cid, label in (("none", "无"), ("D2_dedup", "D2 去重")):
        rs = recs[cid]
        cite = sum(r.get("citation_ok") for r in rs) / max(len(rs), 1)
        print(f"{label:<10} {accs[cid].get('G2', 0):>10.3f} {cite:>10.3f}")

    cite_rate = sum(r.get("citation_ok") for r in recs["none"]) / max(len(recs["none"]), 1)
    d1_ok = accs["none"].get("G2", 0.0) <= 0.20 and cite_rate >= 0.95
    recovery = accs["D2_dedup"].get("G2", 0.0) - accs["none"].get("G2", 0.0)
    print(f"\nD1 判定：攻击下 G2={accs['none'].get('G2', 0):.3f}，"
          f"引用通过率={cite_rate:.3f} → "
          f"{'放大器成立（不降且盖章）' if d1_ok else '未达判据'}")
    print(f"D2 判定：G2 恢复 {recovery:+.3f}（挡攻击效果；其干净语料代价见 E6）")

    outroot = a.outroot or (exp.ROOT / "results" / EXP)
    outroot.mkdir(parents=True, exist_ok=True)
    (outroot / "report.json").write_text(json.dumps({
        "experiment": EXP, "acc": accs, "citation_pass_rate": cite_rate,
        "D1_amplifier_pass": d1_ok, "D2_G2_recovery": recovery,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"报告 -> {outroot / 'report.json'}")


if __name__ == "__main__":
    main()
