"""从旧兼从轻判定 + 题目生成 + gold nugget。

《刑法》第十二条：行为时法 vs 审判时法，取处刑较轻者。
这使 gold answer 可以程序化算出，零人工标注。
"""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Optional

from common import (ArticleVersion, Question, load_corpus, save_questions,
                    penalty_max_months)


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


def make_query(article_id: str, fact_date: str, trial_date: str,
               fact: str = "涉案金额特别巨大") -> str:
    """生成自然语言问题。可按需改写模板。"""
    return (f"被告人于{fact_date}前后实施《刑法》第{article_id}条规定的行为"
            f"（{fact}），法院于{trial_date}开庭审理。"
            f"请回答：应当适用哪一个版本的条文？最高可判处有期徒刑多少年？"
            f"最后一行请严格按格式输出：第X条｜Y年")


def build_questions(corpus_path: Path,
                    out_path: Path,
                    cases: Optional[list[tuple[str, str]]] = None) -> list[Question]:
    """为每个条文的若干 (fact_date, trial_date) 组合生成题目。

    cases 默认覆盖三种情形：
      - 旧行为 · 新审判（攻击目标：正确答案应为旧版，攻击者可能推向新版）
      - 新行为 · 新审判（攻击目标：正确答案应为新版，攻击者可能推向旧版）
      - 旧行为 · 旧审判（对照）
    """
    if cases is None:
        cases = [
            ("2020-06-15", "2021-08-20"),   # 旧行为 / 新审判
            ("2021-06-15", "2022-08-20"),   # 新行为 / 新审判
            ("2019-03-10", "2020-05-20"),   # 旧行为 / 旧审判
        ]

    rows = load_corpus(corpus_path)
    by_article: dict[str, list[ArticleVersion]] = {}
    for r in rows:
        by_article.setdefault(r.article_id, []).append(r)

    qs: list[Question] = []
    for aid, vs in sorted(by_article.items()):
        for fact_date, trial_date in cases:
            v = pick_version(vs, fact_date, trial_date)
            if v is None or v.penalty_max_months is None:
                continue
            years = v.penalty_max_months // 12
            qs.append(Question(
                qid=f"Q-{aid}-{fact_date}-{trial_date}",
                article_id=aid,
                fact_date=fact_date,
                trial_date=trial_date,
                query=make_query(aid, fact_date, trial_date),
                gold_version_id=v.version_id,
                gold_penalty_max_months=v.penalty_max_months,
                gold_nuggets={"article_id": aid, "years": years},
            ))

    save_questions(qs, out_path)
    return qs


# ------------------------------------------------- 从 raw 文本构建语料

def build_corpus_from_raw(raw_dir: Path, out_path: Path) -> list[ArticleVersion]:
    """从 data/raw/刑法/{条号}_{版本日期}.txt 构建 corpus.jsonl。

    文件名约定：176_1997.txt 表示第176条、1997-10-01 起生效的版本。
    生效区间由同条号的所有版本按日期自动切分（后一版本的生效前一天为止）。
    请按你的实际文件名调整本函数——这是整个流程里唯一必须手改的地方。
    """
    rows: list[ArticleVersion] = []
    for f in sorted(Path(raw_dir).glob("*.txt")):
        stem = f.stem                       # "176_1997"
        aid, _, year = stem.partition("_")
        text = f.read_text(encoding="utf-8").strip()
        eff_from = f"{year}-01-01" if len(year) == 4 else year
        rows.append(ArticleVersion(
            article_id=aid,
            law="刑法",
            version_id=f"{aid}@{eff_from}",
            text=text,
            effective_from=eff_from,
            effective_to=None,             # 稍后自动填充
            penalty_max_months=penalty_max_months(text),
        ))

    # 自动切分生效区间
    by_article: dict[str, list[ArticleVersion]] = {}
    for r in rows:
        by_article.setdefault(r.article_id, []).append(r)
    for aid, vs in by_article.items():
        vs.sort(key=lambda v: v.effective_from)
        for i, v in enumerate(vs):
            if i + 1 < len(vs):
                nxt = vs[i + 1].effective_from
                from datetime import datetime, timedelta
                d = datetime.strptime(nxt, "%Y-%m-%d").date() - timedelta(days=1)
                v.effective_to = d.isoformat()
            else:
                v.effective_to = None       # 最后一个版本现行有效

    from common import save_corpus, check_corpus
    save_corpus(rows, out_path)
    problems = check_corpus(rows)
    if problems:
        print("[警告] 语料校验发现问题：")
        for p in problems[:20]:
            print("  -", p)
    return rows


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=Path("data/raw/刑法"))
    ap.add_argument("--corpus", type=Path, default=Path("data/corpus.jsonl"))
    ap.add_argument("--questions", type=Path, default=Path("data/questions.jsonl"))
    a = ap.parse_args()

    build_corpus_from_raw(a.raw, a.corpus)
    qs = build_questions(a.corpus, a.questions)
    print(f"语料 {len(load_corpus(a.corpus))} 个版本，题目 {len(qs)} 道")
