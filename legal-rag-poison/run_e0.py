"""E0 + E1 主流程：建索引 -> 无攻击基线 -> 注入零伪造载荷 -> 再测。

用法：
    # 只跑基线（E0）
    python run_e0.py --corpus data/corpus.jsonl --questions data/questions.jsonl

    # 加上 A3 副本放大（E1，最简单的零伪造攻击）
    python run_e0.py --attack A3 --n-inject 5

    # 开启版本感知修复（D5）
    python run_e0.py --attack A3 --as-of
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import load_corpus, load_questions
from victim import Index, Generator, LegalRAG
from attack import (a1_bare_injection, a2_metadata_forge, a3_duplicate,
                    a4_secondary_citation, b1_random, b2_forged)
from evaluate import extract_prediction, accuracy, tvr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=Path("data/corpus.jsonl"))
    ap.add_argument("--questions", type=Path, default=Path("data/questions.jsonl"))
    ap.add_argument("--chunker", default="fixed512",
                    choices=["fixed512", "fixed512o", "article"])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--attack", default=None,
                    choices=[None, "A1", "A2", "A3", "A4", "B1", "B2"])
    ap.add_argument("--n-inject", type=int, default=5)
    ap.add_argument("--as-of", action="store_true",
                    help="启用版本感知过滤（D5 修复）")
    ap.add_argument("--backend", default="api", choices=["api", "local"])
    ap.add_argument("--model", default="deepseek-chat")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 题（冒烟用）")
    ap.add_argument("--out", type=Path, default=Path("results/e0.json"))
    a = ap.parse_args()

    corpus = load_corpus(a.corpus)
    questions = load_questions(a.questions)
    if a.limit:
        questions = questions[:a.limit]
    print(f"语料 {len(corpus)} 版本 / 题目 {len(questions)} 道")

    # ---------- 建索引
    idx = Index()
    idx.build(corpus, chunker=a.chunker)

    # ---------- 注入
    if a.attack:
        by_vid = {v.version_id: v for v in corpus}
        # 攻击目标：每题 gold 之外的另一个版本（即攻击者希望系统采用的版本）
        payloads = []
        for q in questions:
            alts = [v for v in corpus
                    if v.article_id == q.article_id
                    and v.version_id != q.gold_version_id]
            if not alts:
                continue
            tgt = alts[0]
            if a.attack == "A1":
                payloads += a1_bare_injection(tgt)
            elif a.attack == "A2":
                payloads += a2_metadata_forge(tgt)
            elif a.attack == "A3":
                payloads += a3_duplicate(tgt, k=a.n_inject)
            elif a.attack == "A4":
                payloads += a4_secondary_citation(tgt)   # 需自备真实释义文本
            elif a.attack == "B1":
                payloads += b1_random(corpus, n=a.n_inject)
            elif a.attack == "B2":
                payloads += b2_forged(tgt.article_id,
                                      (tgt.penalty_max_months or 120) // 12)
        # 注入进索引
        if payloads:
            new_vecs = idx.model.encode([d.text for d in payloads],
                                        normalize_embeddings=True)
            import numpy as np
            idx.vecs = np.vstack([idx.vecs, new_vecs])
            idx.docs += payloads
        print(f"注入 {len(payloads)} 个文档（{a.attack}）")

    # ---------- 跑
    gen = Generator(backend=a.backend, model=a.model)
    rag = LegalRAG(idx, gen)

    results = []
    for q in questions:
        as_of = q.fact_date if a.as_of else None
        out = rag.answer(q.query, k=a.k, as_of=as_of)
        pred = extract_prediction(out["answer"])
        alts = [v for v in corpus
                if v.article_id == q.article_id
                and v.version_id != q.gold_version_id]
        attacker_target = None
        if a.attack in ("A1", "A2", "A3", "A4", "B2") and alts:
            t = alts[0]
            attacker_target = {"article_id": t.article_id,
                               "years": (t.penalty_max_months or 0) // 12}
        results.append({
            "qid": q.qid,
            "pred": pred,
            "gold_nuggets": q.gold_nuggets,
            "gold_version_id": q.gold_version_id,
            "target_version_id": (alts[0].version_id
                                  if a.attack and alts else None),
            "attacker_target": attacker_target,
            "retrieved_versions": out["retrieved_versions"],
            "answer": out["answer"][:400],
        })

    acc = accuracy(results)
    tv = tvr(results)
    summary = {
        "chunker": a.chunker, "k": a.k, "attack": a.attack,
        "n_inject": a.n_inject, "as_of": a.as_of,
        "n_questions": len(questions),
        "accuracy": round(acc, 4),
        "tvr": round(tv, 4),
        "error_rate": round(1 - acc, 4),
    }

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"summary": summary, "results": results},
                                ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n===== 结果 =====")
    for k, v in summary.items():
        print(f"{k:>14}: {v}")
    print(f"\n详情 -> {a.out}")


if __name__ == "__main__":
    main()
