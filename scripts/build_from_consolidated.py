"""把「历年合订本」转换为本项目语料格式。

输入：仓库根目录 data/ 下的 15 份《中华人民共和国刑法》合订本
（每一份 = 某次修正/决定之后的完整现行文本；1997 至 2023）。
文件与生效日的对应见 SNAPSHOTS（已逐一对内容校验过：1998 决定、
2009 决定与修正案一至十二，共 15 个快照）。

输出（默认落到仓库根 data/ 下）：
  raw/刑法/{最新日期}/{条号}.txt
      现行全量：每条一个文件，即当前仍有效的整部《刑法》（对照基线）
  raw/刑法/{生效日期}/{条号}_{生效日期}.txt
      历史旧版：该日生效、后来被修正案替换掉的版本文件
  raw/刑法/{生效日期}/{条号}_异常_源文本.txt
      被清理的源数据瑕疵原文（人工复核用）
  raw/刑法/README.md                        目录说明（结构、对照方法、勘误）
  derived/article_versions.jsonl            全量版本表（含来源快照与修正标签）
  derived/changes.csv                       逐快照原始 diff（源数据审计用）
  derived/anomalies.csv                     被清理的源数据瑕疵（修正笔误/回退）
  derived/targets_candidates.csv            刑档变化过的条文（实验选题池）
  background.jsonl                          最新合订本全量条文（检索背景层，不出题）

源数据瑕疵清理规则（可用 --no-collapse 关闭）：
  对某条文，若某段文本（一个或多个连续快照）被前后**完全相同**的另一段
  文本夹住（A -> B -> A），且 B 与 A 高度相似或是更早出现过的旧文本，
  则判定 B 为源数据笔误/错贴（实测命中 5 处：第 88、190、273、419、425 条
  的 2006 年快照），丢弃该段并合并前后段。丢弃记录写入 anomalies.csv，
  raw 中保留原貌，建议人工对照官方文本复核。

用法（在仓库根目录执行）：
    python scripts/build_from_consolidated.py                  # 读 data/，写到 data/
    python scripts/build_from_consolidated.py --only 176,192   # raw 只写这些条文
    python scripts/build_from_consolidated.py --dry-run        # 只分析、不写文件
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # 使 ragpoison 可导入
from ragpoison.common import cn2int, penalty_max_months  # noqa: E402

HERE = Path(__file__).resolve().parent          # scripts/
PROJECT = HERE.parent                           # legal-rag-poison/

# (文件年份键, 生效日期, 修正来源标签)
SNAPSHOTS = [
    ("1997年", "1997-10-01", "1997年刑法（修订施行）"),
    ("1998年", "1998-12-29", "《关于惩治骗购外汇、逃汇和非法买卖外汇犯罪的决定》"),
    ("1999年", "1999-12-25", "刑法修正案（一）"),
    ("2001年8月", "2001-08-31", "刑法修正案（二）"),
    ("2001年12月", "2001-12-29", "刑法修正案（三）"),
    ("2002年", "2002-12-28", "刑法修正案（四）"),
    ("2005年", "2005-02-28", "刑法修正案（五）"),
    ("2006年", "2006-06-29", "刑法修正案（六）"),
    ("2009年2月", "2009-02-28", "刑法修正案（七）"),
    ("2009年8月", "2009-08-27", "《关于修改部分法律的决定》"),
    ("2011年", "2011-05-01", "刑法修正案（八）"),
    ("2015年", "2015-11-01", "刑法修正案（九）"),
    ("2017年", "2017-11-04", "刑法修正案（十）"),
    ("2020年", "2021-03-01", "刑法修正案（十一）"),
    ("2023年", "2024-03-01", "刑法修正案（十二）"),
]

ART_RE = re.compile(
    r"^第([一二三四五六七八九十百零两]+)条(?:之([一二三四五六七八九十]+))?[　 \t]*",
    re.M)
HEAD_RE = re.compile(r"^第[一二三四五六七八九十百零两]+[编章节][　 \t]*.*$", re.M)
REPEAL_RE = re.compile(r"^[（(]\s*(?:删去|已删除|废止)\s*[）)]$")


def art_key(aid: str) -> tuple[int, int]:
    main, _, sub = aid.partition("-")
    return int(main), int(sub or 0)


def parse_snapshot(path: Path) -> dict[str, str | None]:
    """把一个合订本解析成 {条号: 正文}；被删去的条文值为 None。"""
    raw = path.read_text(encoding="utf-8")
    ms = list(ART_RE.finditer(raw))
    out: dict[str, str | None] = {}
    for i, m in enumerate(ms):
        num = cn2int(m.group(1))
        if num is None:
            continue
        aid = str(num)
        if m.group(2):
            sub = cn2int(m.group(2))
            if sub is not None:
                aid += f"-{sub}"
        body = raw[m.end(): ms[i + 1].start() if i + 1 < len(ms) else len(raw)]
        body = HEAD_RE.sub("", body)      # 去掉夹进正文的章/节标题
        body = re.sub(r"\s+", "", body)   # 中文条文无空白语义，压成单行
        out[aid] = None if REPEAL_RE.match(body) else body
    return out


def fmt_penalty(months: int | None) -> str:
    if months is None:
        return "?"
    if months >= 10000:
        return "死刑"
    if months >= 8000:
        return "无期"
    return f"{months // 12}年"


def _lev(a: str, b: str, cap: int = 9) -> int:
    """编辑距离（带提前退出），用于判断笔误级别的差异。"""
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
        if min(prev) > cap:
            return cap + 1
    return prev[-1]


def collapse_sandwiched_runs(runs: list[dict]) -> tuple[list[dict], list[dict]]:
    """清理「被相同文本夹住的异常段」：A -> B -> A，其中 B 是笔误/错贴。

    runs: [{"text", "start", "end"}]（按快照序、相邻同文已合并）。
    返回 (清理后的 runs, 被丢弃的记录)。
    """
    runs = [dict(r) for r in runs]
    dropped: list[dict] = []
    changed = True
    while changed:
        changed = False
        for j in range(1, len(runs) - 1):
            a, b, c = runs[j - 1], runs[j], runs[j + 1]
            if b["text"] is None or a["text"] is None or c["text"] is None:
                continue
            if a["text"] != c["text"]:
                continue
            # B 是更早出现过的旧文本（如第190条 2006 错贴 1997 文本）
            earlier = {r["text"] for r in runs[:j] if r["text"] is not None}
            is_old_text = b["text"] in earlier
            # 或 B 与 A 只差笔误级别（如「有斯徒刑」「拘投」）
            is_typo = _lev(b["text"], a["text"]) <= 8
            if not (is_old_text or is_typo):
                continue
            dropped.append(b)
            runs = runs[:j - 1] + [{"text": a["text"],
                                    "start": a["start"], "end": c["end"]}] + runs[j + 2:]
            changed = True
            break
    return runs, dropped


def write_snapshot_diffs(snaps: list[tuple[str, str, str, dict]],
                         diffs_dir: Path) -> list[tuple[str, int]]:
    """把每次修正相对上一份合订本的改动（相邻对比）写成逐条明细 CSV。

    每行：条号 / 变更类型（修改|新增|废止）/ 旧文本 / 新文本。
    """
    diffs_dir.mkdir(parents=True, exist_ok=True)
    written: list[tuple[str, int]] = []
    for i in range(1, len(snaps)):
        key, eff, label, snap = snaps[i]
        prev = snaps[i - 1][3]
        rows = []
        for aid in sorted(set(snap) | set(prev), key=art_key):
            old, new = prev.get(aid), snap.get(aid)
            if old == new:
                continue
            if old is None:
                kind = "新增"
            elif new is None:
                kind = "废止（删去）"
            else:
                kind = "修改"
            rows.append({"article_id": aid, "变更": kind,
                         "旧文本": old or "", "新文本": new or ""})
        if rows:
            path = diffs_dir / f"{eff}_{label}.csv"
            with open(path, "w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["article_id", "变更", "旧文本", "新文本"])
                w.writeheader()
                w.writerows(rows)
            written.append((path.name, len(rows)))
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description="合订本 -> 项目语料格式")
    ap.add_argument("--src", type=Path, default=PROJECT / "data",
                    help="合订本所在目录（默认仓库根 data/）")
    ap.add_argument("--out", type=Path, default=PROJECT / "data",
                    help="输出目录（默认 legal-rag-poison/data）")
    ap.add_argument("--only", default=None,
                    help="只写这些条文的 raw 文件，逗号分隔，如 176,191,192")
    ap.add_argument("--no-collapse", action="store_true",
                    help="不清理源数据瑕疵（默认清理 A->B->A 异常段）")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    only = {x.strip() for x in a.only.split(",")} if a.only else None

    # ---------- 1. 解析 ----------
    snaps: list[tuple[str, str, str, dict]] = []
    for key, eff, label in SNAPSHOTS:
        path = a.src / f"中华人民共和国刑法 ({key}).txt"
        if not path.exists():
            raise SystemExit(f"缺少快照文件：{path}")
        snaps.append((key, eff, label, parse_snapshot(path)))

    n_base = len(snaps[0][3])
    if n_base < 452:
        print(f"[告警] 1997 合订本解析出 {n_base} 条（应为 452 条）："
              f"数据源缺第452条（施行日期/附件），与实验无关，可忽略。\n")

    # ---------- 2. 逐快照原始 diff（源数据审计） ----------
    changes_rows = []
    for i, (key, eff, label, snap) in enumerate(snaps):
        if i == 0:
            changes_rows.append({
                "snapshot": key, "effective_date": eff, "label": label,
                "n_added": len(snap), "n_changed": 0, "n_repealed": 0,
                "added_articles": "", "changed_articles": "", "repealed_articles": "",
            })
            continue
        prev = snaps[i - 1][3]
        added = sorted(set(snap) - set(prev), key=art_key)
        gone = sorted(set(prev) - set(snap), key=art_key)
        changed = sorted(
            [x for x in snap if x in prev and snap[x] != prev[x]
             and snap[x] is not None and prev[x] is not None], key=art_key)
        repealed = sorted([x for x in snap if x in prev and snap[x] is None
                           and prev[x] is not None], key=art_key)
        changes_rows.append({
            "snapshot": key, "effective_date": eff, "label": label,
            "n_added": len(added), "n_changed": len(changed),
            "n_repealed": len(repealed) + len(gone),
            "added_articles": ";".join(added),
            "changed_articles": ";".join(changed),
            "repealed_articles": ";".join(repealed + gone),
        })
        print(f"{key:10s} {eff}  {label:36s} 新增 {len(added):2d} "
              f"修改 {len(changed):2d} 废止 {len(repealed) + len(gone):2d}")
    print("（以上为原始 diff，含源数据笔误；下面的版本表已做清理）")

    # ---------- 3. 每条的版本段（runs） ----------
    all_aids = sorted({aid for _, _, _, s in snaps for aid in s}, key=art_key)
    anomalies: list[dict] = []
    versions: list[dict] = []
    repealed_articles: set[str] = set()
    for aid in all_aids:
        first = next(i for i, (_, _, _, s) in enumerate(snaps) if aid in s)
        runs: list[dict] = []
        for i in range(first, len(snaps)):
            text = snaps[i][3].get(aid)          # 缺席 = 废止后的状态
            if runs and runs[-1]["text"] == text:
                runs[-1]["end"] = i
            else:
                runs.append({"text": text, "start": i, "end": i})
        if not a.no_collapse:
            runs, dropped = collapse_sandwiched_runs(runs)
            for b in dropped:
                anomalies.append({
                    "article_id": aid,
                    "snapshot": snaps[b["start"]][0],
                    "effective_date": snaps[b["start"]][1],
                    "label": snaps[b["start"]][2],
                    "reason": "文本被前后完全相同的另一段夹住：疑似源数据笔误或错贴旧文本",
                    "text_head": (b["text"] or "")[:80],
                    "text": b["text"] or "",
                })
        if runs and runs[-1]["text"] is None:
            repealed_articles.add(aid)
        for j, run in enumerate(runs):
            if run["text"] is None:
                continue
            eff_from = snaps[run["start"]][1]
            eff_to = None
            if j + 1 < len(runs):
                nxt = snaps[runs[j + 1]["start"]][1]
                eff_to = (datetime.strptime(nxt, "%Y-%m-%d").date()
                          - timedelta(days=1)).isoformat()
            label = snaps[run["start"]][2]
            versions.append({
                "article_id": aid, "law": "刑法",
                "version_id": f"{aid}@{eff_from}",
                "text": run["text"],
                "effective_from": eff_from, "effective_to": eff_to,
                "amended_by": label,
                "penalty_max_months": penalty_max_months(run["text"]),
                "source": f"刑法合订本({snaps[run['start']][0]})",
            })

    versions.sort(key=lambda v: (*art_key(v["article_id"]), v["effective_from"]))
    print(f"\n共 {len(versions)} 个版本，覆盖 {len(all_aids)} 条；"
          f"已废止（默认排除）：{sorted(repealed_articles, key=art_key) or '无'}")
    if anomalies:
        print(f"[清理] 丢弃 {len(anomalies)} 个源数据瑕疵段"
              f"（详见 derived/anomalies.csv，建议对照官方文本复核）：")
        for r in anomalies:
            print(f"  第{r['article_id']}条 {r['snapshot']}: {r['text_head'][:36]}")

    # ---------- 4. 候选题池 ----------
    by_aid: dict[str, list[dict]] = {}
    for v in versions:
        by_aid.setdefault(v["article_id"], []).append(v)
    candidates = []
    for aid, vs in by_aid.items():
        vs = sorted(vs, key=lambda x: x["effective_from"])
        penalty_events, labels = [], []
        for a_, b_ in zip(vs, vs[1:]):
            labels.append(f"{b_['effective_from']}:{b_['amended_by']}")
            if a_["penalty_max_months"] is not None \
                    and b_["penalty_max_months"] is not None \
                    and a_["penalty_max_months"] != b_["penalty_max_months"]:
                penalty_events.append(
                    f"{b_['effective_from']}:{fmt_penalty(a_['penalty_max_months'])}"
                    f"->{fmt_penalty(b_['penalty_max_months'])}")
        candidates.append({
            "article_id": aid,
            "n_versions": len(vs),
            "penalty_changed": "Y" if penalty_events else "N",
            "penalty_events": ";".join(penalty_events),
            "amendment_labels": ";".join(labels),
            "latest_penalty": fmt_penalty(vs[-1]["penalty_max_months"]),
            "repealed": "Y" if aid in repealed_articles else "N",
        })
    candidates.sort(key=lambda c: (c["penalty_changed"] != "Y",
                                   -c["n_versions"], art_key(c["article_id"])))
    n_pen = sum(1 for c in candidates if c["penalty_changed"] == "Y")
    print(f"\n刑档变化过的条文（实验选题池）：{n_pen} 条")
    for c in [c for c in candidates if c["penalty_changed"] == "Y"][:12]:
        print(f"  {c['article_id']:>6}  {c['n_versions']}版  {c['penalty_events']}")
    print()

    if a.dry_run:
        print("--dry-run：未写任何文件。")
        return

    # ---------- 5. 写文件 ----------
    raw_dir = a.out / "raw" / "刑法"
    der_dir = a.out / "derived"
    raw_dir.mkdir(parents=True, exist_ok=True)
    der_dir.mkdir(parents=True, exist_ok=True)

    # 清理上次生成的文件（只删本脚本命名规则内的 txt，不动其它文件）
    stale_res = [
        re.compile(r"^\d+(?:-\d+)?\.txt$"),                     # 新结构：条号.txt
        re.compile(r"^\d+(?:-\d+)?_\d{4}-\d{2}-\d{2}\.txt$"),   # 旧结构：版本文件
        re.compile(r"^\d+(?:-\d+)?_异常_源文本\.txt$"),          # 旧结构：异常留档
    ]

    def _aid_of(name: str) -> str | None:
        m = re.match(r"^(\d+(?:-\d+)?)[._]", name)
        return m.group(1) if m else None

    n_stale = 0
    for f in list(raw_dir.rglob("*.txt")):
        if not any(r.match(f.name) for r in stale_res):
            continue
        aid = _aid_of(f.name)
        if only and aid and aid not in only:
            continue
        f.unlink()
        n_stale += 1
    if n_stale:
        print(f"[清理] 删除上次生成的 {n_stale} 个文件（准备按新目录结构重写）")
    legacy = raw_dir / "_来源对照.md"
    if legacy.exists():
        legacy.unlink()                       # 旧版说明文件，已并入 README.md
    for d in sorted((p for p in raw_dir.iterdir() if p.is_dir()), reverse=True):
        try:
            if not any(d.iterdir()):
                d.rmdir()
        except OSError:
            pass

    # 最新文件夹 = 现行全量（每条一个 {条号}.txt）
    # 其他文件夹 = 该日生效、后来被替换掉的旧版本（{条号}_{生效日期}.txt）
    # 对照方式：历史旧版文件 vs 最新文件夹里同条的现行文本
    latest_date = snaps[-1][1]
    files_written, folders = 0, set()
    for v in versions:
        aid = v["article_id"]
        if only and aid not in only:
            continue
        is_current = v["effective_to"] is None and aid not in repealed_articles
        if is_current:
            folder = raw_dir / latest_date
            fname = f"{aid}.txt"
        else:
            folder = raw_dir / v["effective_from"]
            fname = f"{aid}_{v['effective_from']}.txt"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / fname).write_text(v["text"] + "\n", encoding="utf-8")
        folders.add(folder.name)
        files_written += 1

    # 被清理的源数据瑕疵：原文留档到对应日期文件夹，供人工复核
    for r_ in anomalies:
        if only and r_["article_id"] not in only:
            continue
        folder = raw_dir / r_["effective_date"]
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"{r_['article_id']}_异常_源文本.txt").write_text(
            r_["text"] + "\n", encoding="utf-8")
        folders.add(r_["effective_date"])
        files_written += 1
    folders = len(folders)

    # raw 目录 README：结构、对照方法、勘误
    lines = [
        "# raw/刑法：现行全量 + 历史旧版（人工核对层）",
        "",
        "## 这个目录是什么",
        "",
        f"* **{latest_date}/（最新文件夹）= 现行全量**：每条一个 `{{条号}}.txt`，",
        "  即当前仍然有效的整部《刑法》——这是对照的基线。",
        "* **其他文件夹 = 历史旧版**：只放**从该生效日期起适用、后来被修正案",
        "  替换掉**的版本文件（`{条号}_{生效日期}.txt`）；某条在某个文件夹里",
        "  缺席 = 它没有在那个日期产生新版本。",
        "",
        "这样每个版本只存一份（总计约 700 个文件），且**现行文本不再散落各处，",
        "全部集中在最新文件夹**。程序用的完整版本链见",
        "`../../derived/article_versions.jsonl`。",
        "",
        "## 怎么对照（重要）",
        "",
        "1. **看单条沿革**（推荐）：`python scripts/show_article.py --article 176`",
        "   按版本表列出该条的变化点（只看这些年份即可核对），",
        "   加 `--diff` 打印每次修改的前后全文；文件路径也会列出来。",
        "2. **旧版 vs 现行**：拿历史文件夹里的旧版本文件，与最新文件夹里同条",
        "   的现行文件对比，就是该条从那一版到今天的净变化；",
        "   逐步变化用 show_article --diff。",
        "3. **看全量修改清单**：`../../derived/changes.csv` 列出了 15 组相邻",
        "   快照的 diff（哪次修正改了哪些条）。",
        "4. 关于“2023 和 2022 还是 2023 和 2021”：本目录只有**修法节点**",
        "   快照，没有“每年一份”；最近两次修正是修正案十一（2021-03-01 生效）",
        "   与修正案十二（2024-03-01 生效）。",
        "",
        "## 勘误说明",
        "",
        "* 被自动清理的 5 处源数据瑕疵（第 88、190、273、419、425 条在 2006 年",
        "  文本中的笔误/错贴，如「有斯徒刑」「拘投」）以",
        "  `{条号}_异常_源文本.txt` 留档在对应日期文件夹；清理记录见",
        "  `../../derived/anomalies.csv`，建议对照官方文本复核；",
        "* 第 199 条自 2015-11-01 起被修正案九删去：因此它没有现行文件，",
        "  只有历史版本（1997、2011）。",
        "",
        "## 文件夹 ↔ 修正来源",
        "",
        "| 生效日期（文件夹） | 合订本年份键 | 修正来源 |",
        "|---|---|---|",
    ]
    for key, eff, label, _ in snaps:
        lines.append(f"| {eff} | {key} | {label} |")
    (raw_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    with open(der_dir / "article_versions.jsonl", "w", encoding="utf-8") as f:
        for v in versions:
            f.write(json.dumps(v, ensure_ascii=False) + "\n")

    with open(der_dir / "changes.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(changes_rows[0].keys()))
        w.writeheader()
        w.writerows(changes_rows)

    # 每次修正的逐条改动明细（旧文本 vs 新文本），供人工核对
    diffs_dir = der_dir / "diffs"
    written_diffs = write_snapshot_diffs(snaps, diffs_dir)
    diffs_readme = [
        "# 每次修正的改动明细（相邻对比）",
        "",
        "每个 CSV = 某份合订本相对上一份的差异，即**那次修正改了什么**：",
        "列：`article_id` / `变更`（修改|新增|废止）/ `旧文本` / `新文本`。",
        "",
        "例：`2024-03-01_刑法修正案（十二）.csv` = 修正案十二改动的条文，",
        "旧文本与新文本并排，可直接核对。",
        "",
        "文件清单：",
    ]
    for name, n in written_diffs:
        diffs_readme.append(f"* `{name}` —— {n} 条改动")
    (diffs_dir / "README.md").write_text("\n".join(diffs_readme) + "\n", encoding="utf-8")

    with open(der_dir / "anomalies.csv", "w", encoding="utf-8", newline="") as f:
        fields = ["article_id", "snapshot", "effective_date", "label",
                  "reason", "text_head"]
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(anomalies)

    with open(der_dir / "targets_candidates.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(candidates[0].keys()))
        w.writeheader()
        w.writerows(candidates)

    # 背景层：最新合订本的全部现行条文（只进索引、不出题）
    latest: dict[str, dict] = {}
    for v in versions:
        latest[v["article_id"]] = v            # 已按时间排序，最后写入=最新
    bg_rows = []
    for aid, v in latest.items():
        if aid in repealed_articles:
            continue
        bg_rows.append({
            "article_id": aid, "law": "刑法",
            "version_id": v["version_id"], "text": v["text"],
            "effective_from": v["effective_from"], "effective_to": None,
            "amended_by": v["amended_by"],
            "penalty_max_months": v["penalty_max_months"],
            "source": v["source"], "meta": {"background": True},
        })
    bg_path = a.out / "background.jsonl"
    with open(bg_path, "w", encoding="utf-8") as f:
        for r in bg_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("已写出：")
    print(f"  {raw_dir}/                {folders} 个生效日期文件夹，"
          f"{files_written} 个条文文件"
          + ("（--only 过滤）" if only else ""))
    print(f"  {raw_dir / 'README.md'}              目录说明（怎么对比/基线）")
    print(f"  {der_dir / 'article_versions.jsonl'}   全量版本表（{len(versions)} 版本）")
    print(f"  {der_dir / 'changes.csv'}              逐快照原始 diff")
    print(f"  {diffs_dir}/               每次修正的逐条改动明细"
          f"（{len(written_diffs)} 个文件）")
    print(f"  {der_dir / 'anomalies.csv'}            源数据瑕疵（{len(anomalies)} 段）")
    print(f"  {der_dir / 'targets_candidates.csv'}   选题池（{n_pen} 条刑档变化）")
    print(f"  {bg_path}   背景层 {len(bg_rows)} 条现行条文")
    print("\n下一步（示例，换成你的 target 与 facts）：")
    print("  python -m ragpoison.gold --versions data/derived/article_versions.jsonl \\")
    print("      --only 176,191,192 \\")
    print("      --corpus data/corpus.jsonl --questions data/questions.jsonl \\")
    print("      --facts-json data/facts.json")


if __name__ == "__main__":
    main()
