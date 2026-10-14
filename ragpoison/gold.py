"""从旧兼从轻判定 + 题目生成 + gold nugget。

《刑法》第十二条：行为时法 vs 审判时法，取处刑较轻者。
这使 gold answer 可以程序化算出（在简化口径下：只比较主刑上限）。

两部分工作：
  build_corpus_from_raw  把 data/raw 下逐条收集的版本文件拼成 corpus.jsonl
  build_questions        生成题目 + 从旧兼从轻 gold + 攻击目标版本

重要提醒（论文 limitation）：
  这里只比较「主刑上限」，没有覆盖个案的量刑档、附加刑、司法解释溯及力
  与数罪并罚。W2 必须做人工核验（建议找法学同学盲核并报 kappa）。
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

from .common import (ArticleVersion, Question, check_corpus, load_corpus,
                     penalty_max_months, save_corpus, save_questions)


def version_at(versions: list[ArticleVersion], d: str) -> Optional[ArticleVersion]:
    """返回在日期 d 有效的版本。"""
    for v in versions:
        if v.covers(d):
            return v
    return None


def pick_version(versions: list[ArticleVersion],
                 fact_date: str,
                 trial_date: str) -> Optional[ArticleVersion]:
    """从旧兼从轻：取法定刑上限较低者。

    新法更轻 -> 用新法；否则 -> 用旧法（从旧）。
    """
    v_old = version_at(versions, fact_date)
    v_new = version_at(versions, trial_date)
    if v_old is None or v_new is None:
        return v_old or v_new
    if v_old.version_id == v_new.version_id:
        return v_old
    mo = v_old.penalty_max_months
    mn = v_new.penalty_max_months
    if mo is None or mn is None:
        return v_old
    return v_new if mn < mo else v_old


def pick_attack_target(versions: list[ArticleVersion],
                       gold: ArticleVersion,
                       fact_date: str,
                       trial_date: str) -> tuple[Optional[ArticleVersion], Optional[str]]:
    """指定「攻击者希望系统采纳的错误版本」。

    规则（确定性，写入题目记录）：
      1. 若审判时版本 ≠ gold，取审判时版本（典型的「拉向新版」）；
      2. 否则若行为时版本 ≠ gold，取行为时版本（「拉向旧版」）；
      3. 否则取最新一个非 gold 版本。
    返回 (目标版本, 方向)，方向 = "to_new" / "to_old"。
    """
    v_trial = version_at(versions, trial_date)
    v_fact = version_at(versions, fact_date)
    others = [v for v in versions if v.version_id != gold.version_id]
    target: Optional[ArticleVersion] = None
    if v_trial is not None and v_trial.version_id != gold.version_id:
        target = v_trial
    elif v_fact is not None and v_fact.version_id != gold.version_id:
        target = v_fact
    elif others:
        target = max(others, key=lambda v: v.effective_from)
    if target is None:
        return None, None
    direction = "to_new" if target.effective_from > gold.effective_from else "to_old"
    return target, direction


def make_query(law: str, article_id: str, fact_date: str, trial_date: str,
               fact: str = "涉案情节属于该条文中最高量刑档次对应情形") -> str:
    """生成自然语言问题（可按条文的档位表述改写 fact）。"""
    return (f"被告人于{fact_date}实施了《{law}》第{article_id}条规定的行为"
            f"（{fact}），法院于{trial_date}审理。"
            f"请回答：应当适用哪一个版本的条文？最高可判处有期徒刑多少年？"
            f"最后一行请严格按格式输出：第X条｜Y年")


def build_questions(corpus_path: Path,
                    out_path: Path,
                    cases: Optional[list[tuple[str, str]]] = None,
                    law: str = "刑法",
                    facts: Optional[dict[str, str]] = None) -> list[Question]:
    """为每个条文的若干 (fact_date, trial_date) 组合生成题目。

    facts: {条号: 档位表述}，例如 {"176": "涉案数额特别巨大"}。
           默认为通用占位表述，正式实验必须逐条提供。
    """
    if cases is None:
        cases = [
            ("2020-06-15", "2021-08-20"),   # 旧行为 / 新审判
            ("2021-06-15", "2022-08-20"),   # 新行为 / 新审判
            ("2019-03-10", "2020-05-20"),   # 旧行为 / 旧审判
        ]
    facts = facts or {}
    rows = load_corpus(corpus_path)
    by_article: dict[str, list[ArticleVersion]] = {}
    for r in rows:
        by_article.setdefault(r.article_id, []).append(r)

    qs: list[Question] = []
    skipped: dict[str, int] = {}

    def skip(reason: str) -> None:
        skipped[reason] = skipped.get(reason, 0) + 1

    for aid, vs in sorted(by_article.items()):
        vs = sorted(vs, key=lambda v: v.effective_from)
        fact_label = facts.get(aid, "涉案情节属于该条文中最高量刑档次对应情形")
        for gi, (fact_date, trial_date) in enumerate(cases, 1):
            gold = pick_version(vs, fact_date, trial_date)
            if gold is None or gold.penalty_max_months is None:
                skip("没有可用 gold")
                continue
            if gold.penalty_max_months >= 8000:      # 无期/死刑：现 nugget 格式无法判分
                skip("gold 为无期/死刑")
                continue
            target, direction = pick_attack_target(vs, gold, fact_date, trial_date)
            if target is None or target.penalty_max_months is None:
                skip("没有对照版本")
                continue
            if target.penalty_max_months >= 8000:
                skip("攻击目标版本为无期/死刑")
                continue
            if target.penalty_max_months // 12 == gold.penalty_max_months // 12:
                skip("两版年数相同（nugget 无法区分版本）")
                continue
            years = gold.penalty_max_months // 12
            qs.append(Question(
                qid=f"Q-{aid}-{fact_date}-{trial_date}",
                article_id=aid,
                fact_date=fact_date,
                trial_date=trial_date,
                query=make_query(law, aid, fact_date, trial_date, fact_label),
                gold_version_id=gold.version_id,
                gold_penalty_max_months=gold.penalty_max_months,
                gold_nuggets={"article_id": aid, "years": years},
                fact_label=fact_label,
                attack_target_version_id=target.version_id,
                attack_direction=direction,
                group=f"G{gi}",
            ))

    save_questions(qs, out_path)
    if skipped:
        print("[提示] 跳过不可判分题目：" +
              "；".join(f"{k} {v} 道" for k, v in sorted(skipped.items())))
    return qs


# ------------------------------------------------- 从 raw 文本构建语料

def build_corpus_from_raw(raw_dir: Path, out_path: Path,
                          law: str = "刑法",
                          only: Optional[set[str]] = None) -> list[ArticleVersion]:
    """从 data/raw/{law}/ 递归读取 {条号}_{生效日期}.txt 构建 corpus.jsonl。

    文件名约定（务必用完整生效日期，不要只写年份）：
      176_1997-10-01.txt   第176条、1997-10-01 起生效的版本
      176_2021-03-01.txt   第176条、2021-03-01 起生效的版本
    版本文件可以放在任意子目录（如按生效日期分的文件夹）；
    `{条号}_异常_源文本.txt` 等非版本文件会被自动跳过。
    生效区间的右端点由同一 `article_id` 的下一个版本自动推算。
    only: 只保留这些条号（None = 全部）。
    """
    name_re = re.compile(r"^\d+(?:-\d+)?_\d{4}-\d{2}-\d{2}$")
    rows: list[ArticleVersion] = []
    for f in sorted(Path(raw_dir).rglob("*.txt")):
        stem = f.stem
        if not name_re.match(stem):
            continue
        aid, _, eff_from = stem.partition("_")
        if only and aid not in only:
            continue
        try:
            datetime.strptime(eff_from, "%Y-%m-%d")
        except ValueError:
            raise ValueError(
                f"文件名 {f.name} 不合规：应为 {{条号}}_{{YYYY-MM-DD}}.txt")
        text = f.read_text(encoding="utf-8").strip()
        rows.append(ArticleVersion(
            article_id=aid,
            law=law,
            version_id=f"{aid}@{eff_from}",
            text=text,
            effective_from=eff_from,
            effective_to=None,             # 稍后自动填充
            penalty_max_months=penalty_max_months(text),
        ))

    by_article: dict[str, list[ArticleVersion]] = {}
    for r in rows:
        by_article.setdefault(r.article_id, []).append(r)
    for _aid, vs in by_article.items():
        vs.sort(key=lambda v: v.effective_from)
        for i, v in enumerate(vs):
            if i + 1 < len(vs):
                nxt = vs[i + 1].effective_from
                d = datetime.strptime(nxt, "%Y-%m-%d").date() - timedelta(days=1)
                v.effective_to = d.isoformat()
            else:
                v.effective_to = None       # 最后一个版本现行有效

    save_corpus(rows, out_path)
    problems = check_corpus(rows)
    if problems:
        print("[警告] 语料校验发现问题：")
        for p in problems[:20]:
            print("  -", p)
    return rows


def build_corpus_from_versions(versions_path: Path, out_path: Path,
                               law: str = "刑法",
                               only: Optional[set[str]] = None) -> list[ArticleVersion]:
    """从 derived/article_versions.jsonl（版本表）构建 corpus.jsonl。

    这是 build_from_consolidated.py 的配套入口：版本表已含正文、生效区间
    与来源，无需再解析 raw 快照。only: 只保留这些条号。
    """
    rows: list[ArticleVersion] = []
    with open(versions_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d.get("law") != law:
                continue
            if only and d["article_id"] not in only:
                continue
            rows.append(ArticleVersion(
                article_id=d["article_id"], law=d["law"],
                version_id=d["version_id"], text=d["text"],
                effective_from=d["effective_from"],
                effective_to=d.get("effective_to"),
                amended_by=d.get("amended_by"),
                penalty_max_months=d.get("penalty_max_months"),
                source=d.get("source"), meta=d.get("meta", {}),
            ))
    save_corpus(rows, out_path)
    problems = check_corpus(rows)
    if problems:
        print("[警告] 语料校验发现问题：")
        for p in problems[:20]:
            print("  -", p)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="构建 corpus.jsonl 与 questions.jsonl")
    ap.add_argument("--versions", type=Path, default=None,
                    help="版本表（build_from_consolidated.py 的 "
                         "derived/article_versions.jsonl），推荐入口")
    ap.add_argument("--raw", type=Path, default=Path("data/raw/刑法"),
                    help="兼容入口：从 raw 快照递归解析（旧用法）")
    ap.add_argument("--corpus", type=Path, default=Path("data/corpus.jsonl"))
    ap.add_argument("--questions", type=Path, default=Path("data/questions.jsonl"))
    ap.add_argument("--law", default="刑法")
    ap.add_argument("--facts-json", type=Path, default=None,
                    help='每题档位表述，如 {"176": "涉案数额特别巨大"}')
    ap.add_argument("--only", default=None,
                    help="只构建这些条号（逗号分隔），如 176,191,192")
    ap.add_argument("--only-file", type=Path, default=None,
                    help="只构建文件里列出的条号（每行/逗号分隔）")
    a = ap.parse_args()

    only = None
    if a.only or a.only_file:
        raw = a.only or ""
        if a.only_file and a.only_file.exists():
            raw += "," + a.only_file.read_text(encoding="utf-8")
        only = {x.strip() for x in raw.replace("\n", ",").split(",") if x.strip()}
    if a.versions:
        build_corpus_from_versions(a.versions, a.corpus, law=a.law, only=only)
    else:
        print("[提示] 未指定 --versions，改用旧入口 --raw（递归解析快照文件）")
        build_corpus_from_raw(a.raw, a.corpus, law=a.law, only=only)
    facts = {}
    if a.facts_json and a.facts_json.exists():
        facts = json.loads(a.facts_json.read_text(encoding="utf-8"))
    qs = build_questions(a.corpus, a.questions, law=a.law, facts=facts)
    print(f"语料 {len(load_corpus(a.corpus))} 个版本，题目 {len(qs)} 道")
