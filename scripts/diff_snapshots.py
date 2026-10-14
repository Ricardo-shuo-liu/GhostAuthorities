"""对比任意两份合订本的差异（逐条：A 版文本 vs B 版文本）。

这就是"单纯 diff"的按条版本：选任意两份合订本，输出每一处差异。

用法（在仓库根目录执行）：
    python scripts/diff_snapshots.py --a 2020年 --b 2023年
    python scripts/diff_snapshots.py --a 2017年 --b 2023年
    python scripts/diff_snapshots.py --a 2021-03-01 --b 2024-03-01 --out /tmp/x.csv

--a/--b 支持"合订本年份键"（如 2020年）或"生效日期"（如 2021-03-01）。
默认输出 data/derived/diffs/compare_{A}_vs_{B}.csv，列：
    article_id / 变更 / {A}文本 / {B}文本
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build_from_consolidated import SNAPSHOTS, art_key, parse_snapshot  # noqa: E402

ROOT = HERE.parent
DATA = ROOT / "data"


def resolve(token: str):
    for key, eff, label in SNAPSHOTS:
        if token in (key, eff, key.replace("年", "")):
            path = DATA / f"中华人民共和国刑法 ({key}).txt"
            if not path.exists():
                raise SystemExit(f"缺少源文件：{path}")
            return key, eff, label, parse_snapshot(path)
    raise SystemExit(f"无法识别 '{token}'（可用：1949…2023 的年份键，或 1997-10-01 这类生效日期）")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, help="较旧/起点：年份键或生效日期")
    ap.add_argument("--b", required=True, help="较新/终点：年份键或生效日期")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--list", type=int, default=40, help="屏幕列出最多多少条条号")
    a = ap.parse_args()

    ka, ea, la, sa = resolve(a.a)
    kb, eb, lb, sb = resolve(a.b)
    rows = []
    for aid in sorted(set(sa) | set(sb), key=art_key):
        old, new = sa.get(aid), sb.get(aid)
        if old == new:
            continue
        if old is None:
            kind = "新增"
        elif new is None:
            kind = "废止（删去）"
        else:
            kind = "修改"
        rows.append({"article_id": aid, "变更": kind,
                     f"{ka}文本": old or "", f"{kb}文本": new or ""})

    out = a.out or (DATA / "derived" / "diffs" / f"compare_{ka}_vs_{kb}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["article_id", "变更",
                                          f"{ka}文本", f"{kb}文本"])
        w.writeheader()
        w.writerows(rows)

    n_mod = sum(1 for r in rows if r["变更"] == "修改")
    n_add = sum(1 for r in rows if r["变更"] == "新增")
    n_del = sum(1 for r in rows if r["变更"].startswith("废止"))
    print(f"{ka}（{ea}） vs {kb}（{eb}）：共 {len(rows)} 条有差异"
          f"（修改 {n_mod} / 新增 {n_add} / 废止 {n_del}）")
    ids = [r["article_id"] for r in rows]
    print("条号：" + "、".join(ids[:a.list]) + ("…" if len(ids) > a.list else ""))
    print(f"明细 -> {out}")


if __name__ == "__main__":
    main()
