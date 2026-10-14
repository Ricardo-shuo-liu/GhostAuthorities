#!/usr/bin/env python3
"""E7 — 换一部法律，结论还在吗？（泛化实验，脚手架）

设计：把 E1（半投毒）与 E2（零伪造攻击）的流程搬到第二部法律上
（建议《行政处罚法》2021 修订——它有与刑法第 12 条同款的"从旧兼从轻"规则）。

数据准备（当前仓库尚未包含第二部法律的数据）：
  1) 收集该法的历年合订本，参照 scripts/build_from_consolidated.py 的
     SNAPSHOTS 结构做一份转换脚本（或手工整理），产出
     data/derived/<法名>_article_versions.jsonl（字段与刑法版本表一致）；
  2) 用 ragpoison.gold 生成题目：
       python -m ragpoison.gold --versions <该法版本表> --law <法名> \
           --only <条文列表> --corpus data/<法名>_corpus.jsonl \
           --questions data/<法名>_questions.jsonl --facts-json <档位表述>.json
  3) 运行本脚本（提供上述文件路径）：
       python experiments/e7_generalization/run.py \
           --corpus2 data/<法名>_corpus.jsonl \
           --questions2 data/<法名>_questions.jsonl \
           --background2 data/<法名>_background.jsonl   # 可选

判定：与刑法方向一致（半投毒使 G2 下降；攻击进一步下降）即支持泛化主张。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ragpoison import experiments as exp  # noqa: E402
from ragpoison import stats  # noqa: E402

EXP = "e7_generalization"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    exp.add_pipeline_args(ap)
    ap.add_argument("--corpus2", type=Path, default=None)
    ap.add_argument("--questions2", type=Path, default=None)
    ap.add_argument("--background2", type=Path, default=None)
    ap.add_argument("--n-inject", type=int, default=5)
    a = ap.parse_args()

    if not (a.corpus2 and a.questions2):
        raise SystemExit(
            "E7 需要第二部法律的数据（见本文件开头的准备步骤）。\n"
            "当前仅提供刑法数据；准备好 --corpus2/--questions2 后再运行。")
    if not a.corpus2.exists() or not a.questions2.exists():
        raise SystemExit(f"找不到文件：{a.corpus2} / {a.questions2}")

    corpus_a = exp.build_current_only_corpus(
        a.corpus2, a.corpus2.with_name(a.corpus2.stem + "_current_only.jsonl"))
    common = [*exp.pipeline_args(a), "--questions", str(a.questions2),
              "--k", str(a.k), *exp.limit_args(a)]
    if a.background2 and a.background2.exists():
        common += ["--background", str(a.background2)]
    conds = {
        "A_current_only": [*common, "--corpus", str(corpus_a), "--attack", "none"],
        "B_multi_version": [*common, "--corpus", str(a.corpus2), "--attack", "none"],
        "B_attack_k5": [*common, "--corpus", str(a.corpus2),
                        "--attack", "A3", "--n-inject", str(a.n_inject)],
    }
    accs = {}
    for cid, args in conds.items():
        d = exp.run_condition(EXP, cid, args, a.outroot)
        accs[cid] = stats.group_accuracy(exp.load_records(d))

    print(f"\n== E7 泛化结果 ==")
    print(f"{'条件':<16} {'G1':>7} {'G2':>7} {'G3':>7}")
    for cid in conds:
        c = accs[cid]
        print(f"{cid:<16} {c.get('G1', 0):>7.3f} {c.get('G2', 0):>7.3f} {c.get('G3', 0):>7.3f}")
    half = accs["B_multi_version"].get("G2", 0) - accs["A_current_only"].get("G2", 0)
    atk = accs["B_attack_k5"].get("G2", 0) - accs["B_multi_version"].get("G2", 0)
    ok = half <= -0.20 and atk <= -0.30
    print(f"\n半投毒 G2 变化 {half:+.3f}；攻击再变化 {atk:+.3f}；"
          f"方向{'一致（支持泛化）' if ok else '待核（样本/数据量可能过小）'}")

    outroot = a.outroot or (exp.ROOT / "results" / EXP)
    outroot.mkdir(parents=True, exist_ok=True)
    (outroot / "report.json").write_text(json.dumps({
        "experiment": EXP, "acc": accs, "half_poison_delta": half,
        "attack_delta": atk, "consistent": ok,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"报告 -> {outroot / 'report.json'}")


if __name__ == "__main__":
    main()
