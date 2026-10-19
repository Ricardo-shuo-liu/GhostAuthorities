"""Build the versioned corpora used by the RAG experiments.

Outputs
-------
data/corpus.jsonl
    All versions of the articles in ``query_targets.json`` (the focus test
    articles).  This is the versioned corpus used for E1 old/new comparison.
data/corpus_current_only.jsonl
    The current version of every article.  This is the current-only control
    corpus.
data/corpus_all_versions.jsonl
    Every cleaned article version (504 articles / 684 versions).  Useful for
    realistic retrieval-noise experiments and for checking that no cleaned
    source data were lost.

The script is deterministic and does not call any API.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def article_key(aid: str) -> tuple[int, int]:
    main, _, sub = aid.partition("-")
    return int(main), int(sub or 0)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    targets = json.loads((DATA / "query_targets.json").read_text(encoding="utf-8"))
    focus_ids = set(targets)
    rows = []
    with open(DATA / "derived" / "article_versions.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    rows.sort(key=lambda r: (article_key(r["article_id"]), r["effective_from"]))

    focus = [r for r in rows if r["article_id"] in focus_ids]
    current = [r for r in rows if r.get("effective_to") is None]
    missing = sorted(focus_ids - {r["article_id"] for r in focus}, key=article_key)
    if missing:
        raise SystemExit(f"query_targets 中的条文在版本表中不存在: {missing}")

    write_jsonl(DATA / "corpus.jsonl", focus)
    write_jsonl(DATA / "corpus_current_only.jsonl", current)
    write_jsonl(DATA / "corpus_all_versions.jsonl", rows)

    manifest = {
        "focus_articles": len(focus_ids),
        "focus_versions": len(focus),
        "current_articles": len({r["article_id"] for r in current}),
        "current_versions": len(current),
        "all_articles": len({r["article_id"] for r in rows}),
        "all_versions": len(rows),
        "focus_versions_by_bank": {
            "old": sum(1 for r in focus if r.get("effective_to") is not None),
            "new": sum(1 for r in focus if r.get("effective_to") is None),
        },
        "focus_per_article": dict(sorted(
            Counter(r["article_id"] for r in focus).items(),
            key=lambda kv: article_key(kv[0]))),
    }
    (DATA / "derived" / "corpus_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print("written: data/corpus.jsonl, data/corpus_current_only.jsonl, "
          "data/corpus_all_versions.jsonl")


if __name__ == "__main__":
    main()
