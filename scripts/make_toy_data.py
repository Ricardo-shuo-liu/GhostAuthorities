"""生成离线自检用的玩具语料（虚构《示例法》，不是真实法律）。

目的只有一个：让你在没网、没 API key 的情况下，把
「语料 -> 题目 -> 检索 -> 注入 -> 逐题作答 -> 落盘」整条流水线跑通。
真实实验请用官方文本构建 data/corpus.jsonl（见 README）。

用法（在 legal-rag-poison/ 目录下执行）：
    python scripts/make_toy_data.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ragpoison.common import ArticleVersion, penalty_max_months, save_corpus  # noqa: E402
from ragpoison.gold import build_questions  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "data" / "toy"

# 虚构条文：(条号, 生效日, 失效日, 正文)
ROWS = [
    ("7", "1997-10-01", "2021-02-28",
     "示例法第七条：实施示例行为的，处三年以下有期徒刑；"
     "情节严重的，处三年以上十年以下有期徒刑。"),
    ("7", "2021-03-01", None,
     "示例法第七条：实施示例行为的，处三年以下有期徒刑；"
     "情节严重的，处三年以上十年以下有期徒刑；"
     "情节特别严重的，处十年以上有期徒刑。"),
    ("8", "1997-10-01", "2021-02-28",
     "示例法第八条：实施示例行为的，处五年以下有期徒刑。"),
    ("8", "2021-03-01", None,
     "示例法第八条：实施示例行为的，处三年以下有期徒刑。"),
]

FACTS = {
    "7": "涉案情节特别严重",
    "8": "涉案情节严重",
}

BACKGROUND_N = 200   # 虚构背景条文数量，模拟"大语料里只有少数焦点条文"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    corpus = []
    for aid, eff_from, eff_to, text in ROWS:
        corpus.append(ArticleVersion(
            article_id=aid,
            law="示例法",
            version_id=f"{aid}@{eff_from}",
            text=text,
            effective_from=eff_from,
            effective_to=eff_to,
            penalty_max_months=penalty_max_months(text),
        ))
    save_corpus(corpus, OUT / "corpus.jsonl")

    # 背景语料：只进索引做干扰项，不出题（对应全量现行刑法背景层）
    background = []
    for i in range(BACKGROUND_N):
        aid = str(100 + i)
        text = (f"第{aid}号示例规范：涉及事项{i}的登记与备案要求，"
                f"依相关规定办理。")
        background.append(ArticleVersion(
            article_id=aid,
            law="示例法",
            version_id=f"{aid}@1997-10-01",
            text=text,
            effective_from="1997-10-01",
            effective_to=None,
            penalty_max_months=penalty_max_months(text),
        ))
    save_corpus(background, OUT / "background.jsonl")

    qs = build_questions(OUT / "corpus.jsonl", OUT / "questions.jsonl",
                         law="示例法", facts=FACTS)
    print(f"已生成 {OUT/'corpus.jsonl'}（{len(corpus)} 个版本）、"
          f"{OUT/'background.jsonl'}（{len(background)} 条背景）与 "
          f"{OUT/'questions.jsonl'}（{len(qs)} 道题）")


if __name__ == "__main__":
    main()
