#!/usr/bin/env python3
"""E4 — 缺陷真的是攻击的原因吗？（对应设计书 H5 / DEQ）

2×2 因子：缺陷（多版本无过滤 vs 版本感知过滤）× 攻击（无 vs A3×5）。
  A = 缺陷未修·无攻击   B = 缺陷未修·有攻击
  C = 缺陷已修·无攻击   D = 缺陷已修·有攻击
DEQ = (err_B - err_A) - (err_D - err_C)；DEQ>0 且显著 → 缺陷是攻击的使能条件。

用法：
    python experiments/e4_deq/run.py
    python experiments/e4_deq/run.py --mock
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ragpoison import experiments as exp  # noqa: E402
from ragpoison import stats  # noqa: E402

EXP = "e4_deq"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    exp.add_pipeline_args(ap)
    ap.add_argument("--corpus", type=Path, default=exp.DATA / "corpus.jsonl")
    ap.add_argument("--questions", type=Path, default=exp.DATA / "questions.jsonl")
    ap.add_argument("--background", type=Path, default=exp.DATA / "background.jsonl")
    ap.add_argument("--n-inject", type=int, default=5)
    a = ap.parse_args()

    common = [*exp.pipeline_args(a), "--corpus", str(a.corpus),
              "--questions", str(a.questions), "--background", str(a.background),
              "--k", str(a.k), *exp.limit_args(a)]
    fixed = ["--as-of", "fact", "--unknown-dates", "reject"]

    cells = {
        "A": [*common, "--attack", "none"],
        "B": [*common, "--attack", "A3", "--n-inject", str(a.n_inject)],
        "C": [*common, *fixed, "--attack", "none"],
        "D": [*common, *fixed, "--attack", "A3", "--n-inject", str(a.n_inject)],
    }
    recs = {c: exp.load_records(exp.run_condition(EXP, c, args, a.outroot))
            for c, args in cells.items()}
    err = {c: 1.0 - stats.accuracy(r) for c, r in recs.items()}

    deq = (err["B"] - err["A"]) - (err["D"] - err["C"])
    boot = stats.deq_bootstrap(recs)
    print(f"\n== E4 / DEQ ==")
    print(f"A 缺陷未修·无攻击  错误率 {err['A']:.3f}")
    print(f"B 缺陷未修·有攻击  错误率 {err['B']:.3f}")
    print(f"C 缺陷已修·无攻击  错误率 {err['C']:.3f}")
    print(f"D 缺陷已修·有攻击  错误率 {err['D']:.3f}")
    print(f"DEQ = (B-A)-(D-C) = {deq:+.3f}，"
          f"95%CI [{boot['lo']:+.3f}, {boot['hi']:+.3f}]，p≈{boot['p']:.4f}（按条文聚类）")
    ok = deq > 0 and boot["p"] < 0.05
    print(f"判定 H5：{'成立（缺陷是攻击的使能条件）' if ok else '未达判据（DEQ≈0 走现象报告路线）'}")

    outroot = a.outroot or (exp.ROOT / "results" / EXP)
    outroot.mkdir(parents=True, exist_ok=True)
    (outroot / "report.json").write_text(json.dumps({
        "experiment": EXP, "error_rates": err, "deq": deq,
        "deq_bootstrap": boot, "H5_pass": ok,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"报告 -> {outroot / 'report.json'}")


if __name__ == "__main__":
    main()
