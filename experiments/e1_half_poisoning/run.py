#!/usr/bin/env python3
"""E1 — 半投毒：题库 × 语料（对应设计书 H1）

题库（generate_questions.py 生成；优先使用通过 admission_test.py 的 filtered/）：
  old      需要「过期数据」（旧版文本）的题
  new      需要「实效数据」（现行文本）的题
  spanning 持续/连续犯跨法题 —— 独立分组，**不进入主判据**
语料：
  A 仅现行（只有 2024 版）；B 现行 + 旧版（半投毒状态）。

主判据（设计书 H1）：new 题库在 B 上比 A 显著下降 ≥20pp → 半投毒成立。
说明：模型参数记忆是"常量"，不改变 A/B 差值；old 题库在 A 上偏高时，
按"记忆覆盖率"如实报告（论文口径见 data/e1_questions/README.md）。

用法（仓库根目录执行）：
    python experiments/e1_half_poisoning/generate_questions.py     # 生成题库
    python experiments/e1_half_poisoning/admission_test.py ...     # 入场过滤
    python experiments/e1_half_poisoning/run.py --stage A          # 只跑 2024 语料
    python experiments/e1_half_poisoning/run.py                    # A+B 全跑
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
    ap.add_argument("--banks", default="old,new,spanning",
                    help="要跑的题库（逗号分隔）：old,new,spanning；"
                         "只看 2024 语料下的 old/new 就用 --banks old,new")
    a = ap.parse_args()

    filtered = a.banks_dir / "filtered"
    bank_dir = filtered if (filtered / "questions_old.jsonl").exists() else a.banks_dir
    if bank_dir is a.banks_dir:
        print("[提示] 未找到 filtered/ 题库（未经过滤）；建议先跑 admission_test.py。"
              "本次使用原始题库。")
    wanted = [b.strip() for b in a.banks.split(",") if b.strip()]
    banks: dict[str, Path] = {}
    for name in wanted:
        p = bank_dir / f"questions_{name}.jsonl"
        if p.exists():
            banks[name] = p
        elif name in ("old", "new"):
            raise SystemExit(f"缺少题库 {p}；先运行 generate_questions.py")
    counts = {name: len(load_bank(p)) for name, p in banks.items()}
    print("题库：", counts)

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
    n_failed = {cid: sum(1 for r in rs if r.get("gen_failed")) for cid, rs in recs.items()}

    print(f"\n== E1 结果（准确率，stage={a.stage}）==")
    print(f"{'语料':<16}{'old':>14}{'new':>14}{'spanning*':>14}")
    for cname, label in (("A_current", "A 仅现行(2024)"), ("B_full", "B 含旧版")):
        if f"{cname}_oldbank" not in acc:
            continue
        cells = []
        for bname in ("old", "new", "spanning"):
            cid = f"{cname}_{bname}bank"
            cells.append(f"{acc[cid]:.3f} ({n_correct[cid]}/{len(recs[cid])})"
                         if cid in acc else "—")
        print(f"{label:<16}{cells[0]:>14}{cells[1]:>14}{cells[2]:>14}")
    if any(n_failed.values()):
        print("生成失败题数（已按答错计，续跑会自动补做）：",
              {k: v for k, v in n_failed.items() if v})
    print("* spanning = 持续/连续犯跨法题，独立分组，不参与 H1 判定")

    report = {"experiment": EXP, "stage": a.stage, "bank_dir": str(bank_dir),
              "n_banks": counts, "accuracy": acc,
              "n_correct": n_correct, "n_gen_failed": n_failed,
              "note": "spanning excluded from H1 verdict"}
    if a.stage == "AB" and "old" in banks and "new" in banks:
        print()
        for bname, label in (("new", "new 题库（主判据）"), ("old", "old 题库")):
            boot = stats.paired_cluster_bootstrap(
                recs[f"A_current_{bname}bank"], recs[f"B_full_{bname}bank"])
            print(f"{label} A→B 变化：{boot['delta']:+.3f} "
                  f"95%CI [{boot['lo']:+.3f}, {boot['hi']:+.3f}] p≈{boot['p']:.4f}")
            report[f"{bname}_bank_A_to_B"] = boot
        h1 = (report["new_bank_A_to_B"]["delta"] <= -0.20
              and report["new_bank_A_to_B"]["p"] < 0.05)
        report["H1_pass"] = h1
        print(f"判定 H1（半投毒）："
              f"{'成立（new 题库在含旧版语料上显著下降 ≥20pp）' if h1 else '未达判据'}")
        if acc.get("A_current_oldbank", 0) > 0.2:
            print(f"提示：old 题库在 A（无旧版文本）上准确率 "
                  f"{acc['A_current_oldbank']:.3f} → 模型记忆贡献显著，"
                  f"请在论文中按『记忆覆盖率』单独报告（详见题库 README）。")
    else:
        if a.stage == "AB":
            print("\n（未跑 old+new 两库；H1 判定需要 --stage AB --banks old,new）")
        else:
            print(f"\n（stage={a.stage}：只跑了部分条件，H1 判定需要 --stage AB）")

    outroot = a.outroot or (exp.ROOT / "results" / EXP)
    outroot.mkdir(parents=True, exist_ok=True)
    (outroot / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n报告 -> {outroot / 'report.json'}")


if __name__ == "__main__":
    main()
