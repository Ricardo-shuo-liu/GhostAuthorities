"""人工核对小工具：查看某一条文的版本沿革与变化点。

数据来源：data/derived/article_versions.jsonl（清洗后的版本表）。

用法（在仓库根目录执行）：
    python scripts/show_article.py --article 176          # 沿革表 + 变化点
    python scripts/show_article.py --article 176 --diff   # 打印每次修改前后全文
    python scripts/show_article.py --article 176 --text   # 打印现行（最新基线）全文
    python scripts/show_article.py --article 273 --folder 2006-06-29

说明：raw/刑法 是差异视图（每个文件夹只放该日生效的新版本文件）；
本工具帮你把差异拼成完整链条，并把每条版本对应的 raw 文件路径列出来。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent


def load_versions(path: Path, aid: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d.get("article_id") == aid:
                rows.append(d)
    rows.sort(key=lambda d: d["effective_from"])
    return rows


def find_raw_file(raw_dir: Path, v: dict) -> Path | None:
    """定位版本文件：历史旧版在生效日期文件夹；现行版本在最新文件夹（{条号}.txt）。"""
    aid = v["article_id"]
    p = raw_dir / v["effective_from"] / f"{aid}_{v['effective_from']}.txt"
    if p.exists():
        return p
    for d in sorted((x for x in raw_dir.iterdir() if x.is_dir()), reverse=True):
        q = d / f"{aid}.txt"
        if q.exists():
            return q
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="查看某条文的版本沿革（人工核对）")
    ap.add_argument("--article", required=True, help="条号，如 176 或 162-1")
    ap.add_argument("--versions", type=Path,
                    default=PROJECT / "data" / "derived" / "article_versions.jsonl")
    ap.add_argument("--raw", type=Path,
                    default=PROJECT / "data" / "raw" / "刑法")
    ap.add_argument("--diff", action="store_true", help="打印每次修改的前后全文")
    ap.add_argument("--text", action="store_true", help="只打印现行（最新基线）全文")
    ap.add_argument("--folder", default=None,
                    help="查看指定生效日期文件夹里该条的版本（如 2006-06-29）")
    a = ap.parse_args()

    if not a.versions.exists():
        raise SystemExit(f"缺少版本表 {a.versions}，先跑 scripts/build_from_consolidated.py")
    vs = load_versions(a.versions, a.article)
    if not vs:
        raise SystemExit(f"版本表里没有第 {a.article} 条")

    if a.folder:
        hit = [v for v in vs if v["effective_from"] == a.folder]
        if not hit:
            print(f"第{a.article}条在 {a.folder} 没有新版本文件"
                  f"（该次修正未改动此条；上一次修改见沿革表）")
            return
        p = find_raw_file(a.raw, hit[0])
        print(hit[0]["text"])
        print(f"\n[来源] {p}")
        return

    if a.text:
        print(f"# 现行文本（自 {vs[-1]['effective_from']} 起，最新基线）\n\n{vs[-1]['text']}")
        return

    # 沿革表
    print(f"第{a.article}条 沿革：共 {len(vs)} 个版本，"
          f"最新基线 = {vs[-1]['effective_from']}")
    if vs[-1]["effective_to"]:
        print(f"注意：该条自 {vs[-1]['effective_to']} 之后不再有新版（已被废止/删去）")
    print()
    print(f"{'生效日期':<12} {'版本':<8} 说明")
    print("-" * 72)
    prev = None
    for i, v in enumerate(vs, 1):
        if prev is None:
            note = f"初始版本（{v['amended_by']}）"
        else:
            delta = len(v["text"]) - len(prev["text"])
            note = f"修改（{v['amended_by']}，vs 上一版 {delta:+d} 字）"
        if v["effective_to"] is None:
            note += "　[现行]"
        rel = find_raw_file(a.raw, v)
        marker = str(rel.relative_to(PROJECT)) if rel else "(raw 文件缺失)"
        print(f"{v['effective_from']:<12} v{i:<7} {note}")
        print(f"{'':<12} {'':<8} -> {marker}")
        prev = v

    if a.diff:
        print(f"\n{'=' * 72}")
        for i in range(1, len(vs)):
            old, new = vs[i - 1], vs[i]
            print(f"\n旧版（{old['effective_from']}，{old['amended_by']}）：\n{old['text']}")
            print(f"\n新版（{new['effective_from']}，{new['amended_by']}）：\n{new['text']}")
            print("-" * 72)


if __name__ == "__main__":
    main()
