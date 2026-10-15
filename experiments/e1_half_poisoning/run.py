#!/usr/bin/env python3
"""E1 — 半投毒：两个题库 × 两个语料（对应设计书 H1）

题库（由 generate_questions.py 生成，按《刑法》第 12 条划分）：
  old —— 需要「过期数据」（2024 现行之前的旧版文本）的题；
  new —— 需要「实效数据」（现行版本文本）的题（含新行为新审判、持续/连续犯跨法）。
语料：
  A 仅现行（= E1 阶段 A，只有 2024 版）；B 现行 + 旧版（半投毒状态）。

跑法（4 个条件）：
  A×old、A×new、B×old、B×new。
判据（设计书 H1）：
  1) A 上 old 题库准确率应≈0（语料里没有旧版文本，想对也没材料）
     —— 注意 LLM 可能凭参数记忆答对，这可作为发现记录；
  2) A→B 时 new 题库准确率下降 ≥20pp 且按条文聚类显著 → 「半投毒」成立。

用法（仓库根目录执行）：
    python experiments/e1_half_poisoning/run.py            # API + ChromaDB
    python experiments/e1_half_poisoning/run.py --mock     # 离线自检
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


def load_bank(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    exp.add_pipeline_args(ap)
    ap.add_argument("--corpus", type=Path, default=exp.DATA / "corpus.jsonl")
    ap.add_argument("--background", type=Path, default=exp.DATA / "background.jsonl")
    ap.add_argument("--banks-dir", type=Path, default=exp.DATA / "e1_questions")
    ap.add_argument("--stage", default="AB", choices=["A", "B", "AB"],
                    help="A=只用现行（2024）语料；B=含旧版语料；AB=全跑（默认）")
    a = ap.parse_args()

    banks = {
        "old": a.banks_dir / "questions_old.jsonl",
        "new": a.banks_dir / "questions_new.jsonl",
    }
    for name, p in banks.items():
        if not p.exists():
            raise SystemExit(
                f"缺少 {name} 题库 {p}；先运行：\n"
                f"  python experiments/e1_half_poisoning/generate_questions.py")
    n_old, n_new = (len(load_bank(p)) for p in banks.values())
    print(f"题库：old {n_old} 道 / new {n_new} 道")

    corpus_a = exp.build_current_only_corpus(
        a.corpus, exp.DATA / "corpus_current_only.jsonl")
    base = [*exp.pipeline_args(a), "--background", str(a.background),
            "--k", str(a.k), *exp.limit_args(a), "--attack", "none"]

    stages = []
    if "A" in a.stage:
        stages.append(("A_current", corpus_a))
    if "B" in a.stage:
        stages.append(("B_full", a.corpus))

    dirs, recs = {}, {}
    for cname, corpus in stages:
        for bname, qpath in banks.items():
            cid = f"{cname}_{bname}bank"
            d = exp.run_condition(
                EXP, cid, [*base, "--corpus", str(corpus),
                           "--questions", str(qpath)], a.outroot)
            dirs[cid] = d
            recs[cid] = exp.load_records(d)

    acc = {cid: stats.accuracy(rs) for cid, rs in recs.items()}
    n_correct = {cid: sum(1 for r in rs if r.get("correct")) for cid, rs in recs.items()}
    n_total = {cid: len(rs) for cid, rs in recs.items()}
    n_failed = {cid: sum(1 for r in rs if r.get("gen_failed")) for cid, rs in recs.items()}

    print(f"\n== E1 结果（准确率，stage={a.stage}）==")
    print(f"{'语料':<14}{'old 题库':>16}{'new 题库':>16}")
    for cname, label in (("A_current", "A 仅现行(2024)"), ("B_full", "B 含旧版")):
        if f"{cname}_oldbank" not in acc and f"{cname}_newbank" not in acc:
            continue
        cells = []
        for bname in ("old", "new"):
            cid = f"{cname}_{bname}bank"
            cells.append(f"{acc[cid]:.3f} ({n_correct[cid]}/{n_total[cid]})"
                         if cid in acc else "—")
        print(f"{label:<14}{cells[0]:>16}{cells[1]:>16}")
    if any(n_failed.values()):
        print("生成失败题数（已按答错计入，可重跑补齐）：",
              {k: v for k, v in n_failed.items() if v})

    report = {"experiment": EXP, "stage": a.stage,
              "n_old_bank": n_old, "n_new_bank": n_new, "accuracy": acc,
              "n_correct": n_correct, "n_total": n_total}
    report["n_gen_failed"] = n_failed
    if a.stage == "AB":
        boot_new = stats.paired_cluster_bootstrap(recs["A_current_newbank"],
                                                  recs["B_full_newbank"])
        boot_old = stats.paired_cluster_bootstrap(recs["A_current_oldbank"],
                                                  recs["B_full_oldbank"])
        print(f"\nnew 题库 A→B 变化：{boot_new['delta']:+.3f} "
              f"95%CI [{boot_new['lo']:+.3f}, {boot_new['hi']:+.3f}] p≈{boot_new['p']:.4f}")
        print(f"old 题库 A→B 变化：{boot_old['delta']:+.3f} "
              f"95%CI [{boot_old['lo']:+.3f}, {boot_old['hi']:+.3f}] p≈{boot_old['p']:.4f}")
        h1 = boot_new["delta"] <= -0.20 and boot_new["p"] < 0.05
        print(f"判定 H1（半投毒）：{'成立（new 题库在含旧版语料上显著下降 ≥20pp）' if h1 else '未达判据'}")
        print(f"参考：old 题库在 A（仅现行）上准确率 = {acc['A_current_oldbank']:.3f}"
              f"（设计预期≈0；偏高说明模型可能凭记忆作答，作为发现记录）")
        report.update({"old_bank_A_to_B": boot_old, "new_bank_A_to_B": boot_new,
                       "H1_pass": h1})
    else:
        print(f"\n（stage={a.stage}：只跑了部分条件，H1 判定需要 --stage AB）")

    outroot = a.outroot or (exp.ROOT / "results" / EXP)
    outroot.mkdir(parents=True, exist_ok=True)
    (outroot / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n报告 -> {outroot / 'report.json'}")


if __name__ == "__main__":
    main()
