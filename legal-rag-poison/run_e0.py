"""E0/E1 主流程：建索引 -> 基线或注入零伪造载荷 -> 逐题跑 -> 本地落盘。

设计要点（相对旧版代码的修复）：
  * 注入按「唯一目标版本」聚合一次，不再每题循环重复注入（旧版 A1 会被
    每个问题各注入一次，注入预算口径直接错掉）；
  * 每条记录区分「命中真本」与「命中注入副本」（二者 version_id 相同，
    靠 injected 标记区分）；
  * 结果逐题 append 到 answers.jsonl，支持 --resume 断点续跑；
  * LLM 与向量都有本地缓存，重跑不重复花钱；
  * 生成与向量后端可切 api / local / mock（mock 完全离线）。

用法示例：
    # 离线自检（不联网、不花钱）
    python run_e0.py --corpus data/toy/corpus.jsonl \\
        --questions data/toy/questions.jsonl \\
        --embed-backend mock --generator mock \\
        --attack none --outdir results/smoke_base

    # 真实 API（DeepSeek 示例）
    export LLM_API_KEY=sk-xxx
    python run_e0.py --corpus data/corpus.jsonl --questions data/questions.jsonl \\
        --embed-backend local --embed-model BAAI/bge-small-zh-v1.5 \\
        --generator api --llm-model deepseek-chat \\
        --llm-base-url https://api.deepseek.com \\
        --attack A3 --n-inject 5 --outdir results/e1_a3
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

from attack import (a1_bare_injection, a2_metadata_forge, a3_duplicate,
                    a4_secondary_citation, b1_random, b2_forged)
from common import load_corpus, load_questions
from evaluate import accuracy, asr, extract_prediction
from victim import (ApiGenerator, Doc, Embedder, Index, JsonlCache, LegalRAG,
                    MockGenerator)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="E0/E1 实验主流程")
    ap.add_argument("--corpus", type=Path, default=Path("data/corpus.jsonl"))
    ap.add_argument("--questions", type=Path, default=Path("data/questions.jsonl"))
    ap.add_argument("--background", type=Path, default=None,
                    help="背景语料（只进索引做干扰项，不出题不参与 gold），"
                         "如全量现行刑法 background.jsonl")
    ap.add_argument("--outdir", type=Path, required=True,
                    help="本次实验的输出目录（结果、缓存都放这里）")

    ap.add_argument("--chunker", default="fixed512",
                    choices=["fixed512", "fixed512o", "article"])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--context-metadata", default="none",
                    choices=["none", "version_id"],
                    help="检索上下文里是否暴露版本号/日期（实验变量，默认不暴露）")

    ap.add_argument("--attack", default="none",
                    choices=["none", "A1", "A2", "A3", "A4", "B1", "B2"])
    ap.add_argument("--n-inject", type=int, default=5,
                    help="A3 副本数 / B1 随机文本数")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--commentary-file", type=Path, default=None,
                    help='A4 用：{"version_id": "真实释义原文"} 的 JSON 文件')

    ap.add_argument("--as-of", default="none", choices=["none", "fact", "trial"],
                    help="版本感知过滤 D5 用哪个日期：不启用 / 行为日 / 审判日")
    ap.add_argument("--unknown-dates", default="reject",
                    choices=["reject", "accept"],
                    help="D5 对无日期元数据文档（A1 载荷）的处置策略")

    ap.add_argument("--embed-backend", default="local",
                    choices=["local", "api", "mock"])
    ap.add_argument("--embed-model", default="BAAI/bge-small-zh-v1.5")
    ap.add_argument("--embed-base-url", default=None)
    ap.add_argument("--embed-api-key-env", default="EMBED_API_KEY")
    ap.add_argument("--embed-batch", type=int, default=32)

    ap.add_argument("--generator", default="api", choices=["api", "mock"])
    ap.add_argument("--llm-model", default="deepseek-chat")
    ap.add_argument("--llm-base-url", default="https://api.deepseek.com")
    ap.add_argument("--llm-api-key-env", default="LLM_API_KEY")
    ap.add_argument("--llm-temperature", type=float, default=0.0)
    ap.add_argument("--llm-max-tokens", type=int, default=512)
    ap.add_argument("--llm-timeout", type=float, default=120.0)
    ap.add_argument("--llm-retries", type=int, default=3)
    ap.add_argument("--sleep", type=float, default=0.0,
                    help="每次 LLM 调用后的固定间隔秒数（限速用）")

    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 题（冒烟用）")
    ap.add_argument("--resume", action="store_true",
                    help="跳过 answers.jsonl 里已完成的 qid，断点续跑")
    return ap.parse_args()


def load_done_qids(answers_path: Path) -> set[str]:
    done: set[str] = set()
    if answers_path.exists():
        with open(answers_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["qid"])
                except Exception:
                    continue
    return done


def build_payloads(args: argparse.Namespace, corpus, questions) -> list[Doc]:
    """按唯一目标版本聚合载荷，一次注入（修复旧版每题重复注入的 bug）。"""
    if args.attack == "none":
        return []
    if args.attack == "B1":
        return b1_random(corpus, n=args.n_inject, seed=args.seed)

    vmap = {v.version_id: v for v in corpus}
    target_vids: list[str] = []
    for q in questions:
        vid = q.attack_target_version_id
        if vid and vid in vmap and vid not in target_vids:
            target_vids.append(vid)

    commentary = {}
    if args.attack == "A4":
        if args.commentary_file and args.commentary_file.exists():
            commentary = json.loads(
                args.commentary_file.read_text(encoding="utf-8"))
        else:
            print("[警告] A4 需要 --commentary-file（真实释义文本 JSON），"
                  "本次不注入任何载荷")
            return []

    docs: list[Doc] = []
    for vid in target_vids:
        v = vmap[vid]
        if args.attack == "A1":
            docs += a1_bare_injection(v)
        elif args.attack == "A2":
            docs += a2_metadata_forge(v)
        elif args.attack == "A3":
            docs += a3_duplicate(v, k=args.n_inject)
        elif args.attack == "A4":
            docs += a4_secondary_citation(v, commentary.get(vid))
        elif args.attack == "B2":
            docs += b2_forged(v.article_id, (v.penalty_max_months or 120) // 12)
    return docs


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.outdir / "cache"
    answers_path = args.outdir / "answers.jsonl"

    corpus = load_corpus(args.corpus)
    questions = load_questions(args.questions)
    if args.limit:
        questions = questions[:args.limit]
    if not questions:
        raise SystemExit("题目为空，先检查 questions.jsonl")
    vmap = {v.version_id: v for v in corpus}
    print(f"语料 {len(corpus)} 版本 / 题目 {len(questions)} 道 / "
          f"攻击 {args.attack} / outdir {args.outdir}")

    # ---------- 建索引 ----------
    embedder = Embedder(
        backend=args.embed_backend, model=args.embed_model,
        base_url=args.embed_base_url,
        api_key=os.environ.get(args.embed_api_key_env, ""),
        batch_size=args.embed_batch, cache_dir=cache_dir)
    index = Index(embedder)
    index.build(corpus, chunker=args.chunker)
    if args.background:
        background = load_corpus(args.background)
        index.add_versions(background, chunker=args.chunker, background=True)
        print(f"背景语料 {len(background)} 个版本（只做检索干扰，不出题）")

    # ---------- 注入 ----------
    payloads = build_payloads(args, corpus, questions)
    index.add_docs(payloads)
    if payloads:
        print(f"注入载荷 {len(payloads)} 个文档（{args.attack}）")

    # ---------- 生成器 ----------
    if args.generator == "mock":
        gen = MockGenerator()
    else:
        gen = ApiGenerator(
            model=args.llm_model, base_url=args.llm_base_url,
            api_key=os.environ.get(args.llm_api_key_env, ""),
            temperature=args.llm_temperature, max_tokens=args.llm_max_tokens,
            timeout=args.llm_timeout, retries=args.llm_retries,
            cache=JsonlCache(cache_dir / "llm.jsonl"), sleep=args.sleep)
    rag = LegalRAG(index, gen)

    # ---------- 落盘配置 ----------
    config = {k: (str(v) if isinstance(v, Path) else v)
              for k, v in vars(args).items()}
    (args.outdir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------- 逐题跑（追加写、可续跑） ----------
    done = load_done_qids(answers_path) if args.resume else set()
    if done:
        print(f"续跑：已完成 {len(done)} 题，跳过")
    with open(answers_path, "a", encoding="utf-8") as fout:
        for i, q in enumerate(questions):
            if q.qid in done:
                continue
            as_of = None
            if args.as_of == "fact":
                as_of = q.fact_date
            elif args.as_of == "trial":
                as_of = q.trial_date
            out = rag.answer(q.query, k=args.k, as_of=as_of,
                             unknown_dates=args.unknown_dates,
                             context_metadata=args.context_metadata)
            pred = extract_prediction(out["answer"])

            target_v = vmap.get(q.attack_target_version_id or "")
            attacker_target = None
            if target_v is not None:
                attacker_target = {
                    "article_id": target_v.article_id,
                    "years": (target_v.penalty_max_months or 0) // 12,
                }
            retrieved_versions = out["retrieved_versions"]
            record = {
                "qid": q.qid,
                "cond": {
                    "attack": args.attack, "n_inject": args.n_inject,
                    "k": args.k, "chunker": args.chunker,
                    "as_of": args.as_of, "unknown_dates": args.unknown_dates,
                    "context_metadata": args.context_metadata,
                    "generator": args.generator, "llm_model": args.llm_model,
                    "embed_backend": args.embed_backend,
                    "embed_model": args.embed_model,
                },
                "article_id": q.article_id,
                "fact_date": q.fact_date,
                "trial_date": q.trial_date,
                "query": q.query,
                "gold_version_id": q.gold_version_id,
                "gold_nuggets": q.gold_nuggets,
                "target_version_id": q.attack_target_version_id,
                "attack_direction": q.attack_direction,
                "attacker_target": attacker_target,
                "pred": [pred[0], pred[1]],
                "correct": accuracy([{"pred": pred, "gold_nuggets": q.gold_nuggets}]),
                "answer": out["answer"],
                "retrieved": out["retrieved"],
                "retrieved_versions": retrieved_versions,
                "gold_in_topk": q.gold_version_id in retrieved_versions,
                "target_any_in_topk": bool(q.attack_target_version_id
                                           and q.attack_target_version_id in retrieved_versions),
                "target_injected_in_topk": any(
                    d["injected"] and d["version_id"] == q.attack_target_version_id
                    for d in out["retrieved"]),
                "ts": datetime.now().isoformat(timespec="seconds"),
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()
            if (i + 1) % 20 == 0 or i + 1 == len(questions):
                print(f"  进度 {i+1}/{len(questions)}")

    # ---------- 汇总 ----------
    records = []
    with open(answers_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    n = len(records)
    acc = accuracy([{"pred": r["pred"], "gold_nuggets": r["gold_nuggets"]}
                    for r in records])
    summary = {
        "attack": args.attack, "n_questions": n,
        "n_payloads": len(payloads),
        "accuracy": round(acc, 4), "error_rate": round(1 - acc, 4),
        "gold_in_topk_rate": round(
            sum(r["gold_in_topk"] for r in records) / n, 4),
        "target_any_in_topk_rate": round(
            sum(r["target_any_in_topk"] for r in records) / n, 4),
        "target_injected_in_topk_rate": round(
            sum(r["target_injected_in_topk"] for r in records) / n, 4)
        if args.attack != "none" else None,
        "asr": round(asr(records), 4) if args.attack != "none" else None,
    }
    # 按方向拆分「预测命中攻击者目标」的比例（方案预告的 to_old/to_new 不对称性）
    for direction in ("to_old", "to_new"):
        sub = [r for r in records if r.get("attack_direction") == direction]
        if sub:
            summary[f"target_hit_{direction}"] = round(asr(sub), 4)
            summary[f"n_{direction}"] = len(sub)
    (args.outdir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n===== 结果 =====")
    for k, v in summary.items():
        print(f"{k:>26}: {v}")
    print(f"\n逐题数据 -> {answers_path}")
    print(f"汇总     -> {args.outdir / 'summary.json'}")


if __name__ == "__main__":
    main()
