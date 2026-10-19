"""统计 query 分布（条号 × old/new、时间锚点、年限），检查是否存在条文偏好。

用法：
    python scripts/query_stats.py
    python scripts/query_stats.py --input data/queries/filtered/clean_queries.jsonl

输出：屏幕表格 + data/queries/distribution_report.json（默认）。
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path,
                    default=DATA / "queries" / "candidates_10000.jsonl")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    rows = [json.loads(l) for l in a.input.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not rows:
        raise SystemExit("输入为空")

    cells = Counter((r["article_id"], r["bank"]) for r in rows)
    aids = sorted({r["article_id"] for r in rows})
    banks = ["old", "new"]
    total = len(rows)
    ideal = total / (len(aids) * len(banks))
    months = Counter(re.search(r"((?:19|20)\d{2}年\d{1,2}月)", r["query"]).group(1)
                     if re.search(r"((?:19|20)\d{2}年\d{1,2}月)", r["query"]) else "?"
                     for r in rows)
    forbidden_re = re.compile(
        r"第[0-9一二三四五六七八九十百千零两]+条|《[^》]*》|"
        r"\d{4}[-/.]\d{1,2}([-/.]\d{1,2})?|刑法|修正案|司法解释")
    bad_regex = sum(1 for r in rows if forbidden_re.search(r["query"]))
    bad_anchor = sum(1 for r in rows
                     if len(re.findall(r"(?:19|20)\d{2}年\d{1,2}月", r["query"])) != 1)
    dup = len(rows) - len({r["query"] for r in rows})
    lens = [len(r["query"]) for r in rows]

    print(f"{'条号':<6}{'old':>8}{'new':>8}{'合计':>8}{'偏差':>10}")
    report_cells = {}
    for aid in aids:
        o, n = cells[(aid, "old")], cells[(aid, "new")]
        report_cells[aid] = {"old": o, "new": n, "total": o + n}
        print(f"{aid:<6}{o:>8}{n:>8}{o + n:>8}{o + n - 2 * ideal:>+10.1f}")
    print(f"\n总数 {total}；每格理想 {ideal:.1f}；"
          f"最小格 {min(cells.values())} / 最大格 {max(cells.values())} "
          f"(比 {max(cells.values()) / max(1, min(cells.values())):.2f}×)")
    print(f"正则违规 {bad_regex} / 锚点异常 {bad_anchor} / 完全重复 {dup}")
    print(f"长度 均值 {sum(lens)/len(lens):.1f}（{min(lens)}–{max(lens)}）")
    print("时间锚点分布（前 12）:", months.most_common(12))

    out = a.out or a.input.with_name("distribution_report.json")
    report = {
        "input": str(a.input), "total": total,
        "articles": len(aids),
        "banks": dict(Counter(r["bank"] for r in rows)),
        "cells": report_cells, "ideal_per_cell": round(ideal, 2),
        "min_cell": min(cells.values()), "max_cell": max(cells.values()),
        "max_min_ratio": round(max(cells.values()) / max(1, min(cells.values())), 3),
        "regex_violations": bad_regex, "anchor_violations": bad_anchor,
        "exact_duplicates": dup,
        "length": {"mean": round(sum(lens) / len(lens), 1),
                    "min": min(lens), "max": max(lens)},
        "month_anchors": dict(months.most_common()),
    }
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"报告 -> {out}")


if __name__ == "__main__":
    main()
