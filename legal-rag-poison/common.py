"""公共数据结构与工具：中文数字、刑期解析、语料/题目的读写。"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Iterable, Optional

# ---------------------------------------------------------------- 中文数字

_CN = {'零': 0, '一': 1, '两': 2, '二': 2, '三': 3, '四': 4,
       '五': 5, '六': 6, '七': 7, '八': 8, '九': 9}


def cn2int(s: str) -> Optional[int]:
    """'十五' -> 15, '二十' -> 20, '二十五' -> 25, '10' -> 10"""
    s = s.strip()
    if s.isdigit():
        return int(s)
    if s in _CN:
        return _CN[s]
    if s == '十':
        return 10
    if s.startswith('十') and len(s) == 2:
        return 10 + _CN.get(s[1], 0)
    if s.endswith('十') and len(s) == 2:
        return _CN.get(s[0], 0) * 10
    if '十' in s:
        a, _, b = s.partition('十')
        return _CN.get(a, 0) * 10 + _CN.get(b, 0)
    return None


_YEAR_RE = re.compile(r'([0-9零一两二三四五六七八九十]{1,3})\s*年')


def parse_years(text: str) -> list[int]:
    """抽出文本里所有 'X年' 的 X（年数）。无法解析的跳过。"""
    out = []
    for m in _YEAR_RE.finditer(text):
        v = cn2int(m.group(1))
        if v is not None and 0 < v <= 100:
            out.append(v)
    return out


_SEG_RE = re.compile(
    r'([0-9零一两二三四五六七八九十]{1,3})年(以下|以上)?有期徒刑')

#: 《刑法》第45条：单个罪名有期徒刑上限 15 年。
#: 所以「处十年以上有期徒刑」的上限是 15 年，不是 10 年。
STATUTE_MAX_YEARS = 15


def penalty_max_months(text: str) -> Optional[int]:
    """从条文正文抽主刑上限（月）。死刑/无期返回很大的数。

    关键区分：
      「X年以下有期徒刑」 -> 上限 X
      「X年以上有期徒刑」 -> 上限 15（刑法第45条）
    注意：「三年以上十年以下有期徒刑」只按「十年以下」计，上限 10。

    这是粗抽取，W2 必须抽样人工核验（计划里是 50 例）。
    """
    if '死刑' in text:
        return 10_000
    if '无期徒刑' in text:
        return 8_000
    caps: list[int] = []
    for num, mark in _SEG_RE.findall(text):
        v = cn2int(num)
        if v is None:
            continue
        caps.append(STATUTE_MAX_YEARS if mark == '以上' else v)
    return max(caps) * 12 if caps else None


# ---------------------------------------------------------------- 数据结构

@dataclass
class ArticleVersion:
    """一条法条的一个版本。"""
    article_id: str              # 条号，稳定标识，如 "176"
    law: str                     # "刑法"
    version_id: str              # 唯一，如 "176@1997-10-01"
    text: str                    # 条文正文原文（verbatim）
    effective_from: str          # "1997-10-01"
    effective_to: Optional[str]  # "2021-02-28"，None 表示现行有效
    amended_by: Optional[str] = None
    penalty_max_months: Optional[int] = None
    meta: dict = field(default_factory=dict)   # 攻击时可能篡改的字段放这里

    def covers(self, date: str) -> bool:
        """该版本在给定日期是否有效（闭区间）。"""
        if date < self.effective_from:
            return False
        if self.effective_to is not None and date > self.effective_to:
            return False
        return True


@dataclass
class Question:
    qid: str
    article_id: str
    fact_date: str               # 行为发生日
    trial_date: str              # 审判日
    query: str                   # 给系统的自然语言问题
    gold_version_id: str         # 由从旧兼从轻算出
    gold_penalty_max_months: int
    gold_nuggets: dict           # {"article_id": "176", "years": 15}

    def to_dict(self):
        return asdict(self)


# ---------------------------------------------------------------- IO

def _read_jsonl(p: Path) -> Iterable[dict]:
    with open(p, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_corpus(p: Path) -> list[ArticleVersion]:
    return [ArticleVersion(**d) for d in _read_jsonl(Path(p))]


def save_corpus(rows: Iterable[ArticleVersion], p: Path) -> None:
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    with open(p, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + '\n')


def load_questions(p: Path) -> list[Question]:
    return [Question(**d) for d in _read_jsonl(Path(p))]


def save_questions(rows: Iterable[Question], p: Path) -> None:
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    with open(p, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + '\n')


# ---------------------------------------------------------------- 校验

def check_corpus(rows: list[ArticleVersion]) -> list[str]:
    """返回问题列表。空列表表示通过。"""
    problems = []
    by_article: dict[str, list[ArticleVersion]] = {}
    for r in rows:
        by_article.setdefault(r.article_id, []).append(r)

    for aid, vs in by_article.items():
        vs = sorted(vs, key=lambda v: v.effective_from)
        for v in vs:
            if v.penalty_max_months is None:
                problems.append(f"{v.version_id}: penalty_max_months 为空")
            if v.effective_to and v.effective_to < v.effective_from:
                problems.append(f"{v.version_id}: effective_to < effective_from")
        # 检查版本区间是否首尾相接且不重叠
        for a, b in zip(vs, vs[1:]):
            if a.effective_to is None:
                problems.append(f"{a.version_id}: 非最后版本却 effective_to=None")
                continue
            nxt = _next_day(a.effective_to)
            if b.effective_from != nxt:
                problems.append(
                    f"{aid}: 版本区间不连续 {a.effective_to} -> {b.effective_from}")
    return problems


def _next_day(d: str) -> str:
    from datetime import date, timedelta
    y, m, dd = map(int, d.split('-'))
    return (date(y, m, dd) + timedelta(days=1)).isoformat()
