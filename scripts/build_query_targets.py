"""为全部合格条文生成 query 目标配置（key_facts / tier_hint / avoid）。

合格条件（与 generate_queries.py 一致）：
  * 至少 2 个版本（确实被修正过）；
  * 所有版本主刑上限为有限年（剔除含无期/死刑的条文）；
  * 后法不减轻（刑档单调不减）→ 单锚点「案发时法」与「从旧兼从轻」一致；
  * 现行版本与最早版本的刑档不同（有判别性）。

输出：data/query_targets.json（保留已有 6 条人工配置，其余由 LLM 生成）
      data/query_targets_report.json（生成/保留清单）

用法（仓库根目录，需 .env 中的 LLM_* 配置）：
    python scripts/build_query_targets.py --dry-run
    python scripts/build_query_targets.py
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from ragpoison.env import load_dotenv                      # noqa: E402
from generate_queries import call_llm, http_post            # noqa: E402

DATA = ROOT / "data"


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--versions", type=Path, default=DATA / "derived" / "article_versions.jsonl")
    ap.add_argument("--candidates", type=Path, default=DATA / "derived" / "targets_candidates.csv")
    ap.add_argument("--out", type=Path, default=DATA / "query_targets.json")
    ap.add_argument("--base-url",
                    default=os.environ.get("QUERY_GEN_BASE_URL")
                    or os.environ.get("LLM_BASE_URL", "https://api.deepseek.com"))
    ap.add_argument("--model",
                    default=os.environ.get("QUERY_GEN_MODEL")
                    or os.environ.get("LLM_MODEL", "deepseek-chat"))
    ap.add_argument("--api-key-env", default="LLM_API_KEY")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    vers: dict[str, list[dict]] = {}
    for line in a.versions.read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            vers.setdefault(d["article_id"], []).append(d)
    for vs in vers.values():
        vs.sort(key=lambda v: v["effective_from"])

    candidate_rows = {
        row["article_id"]: row
        for row in csv.DictReader(open(a.candidates, "r", encoding="utf-8"))
    }
    pen = [aid for aid, row in candidate_rows.items()
           if row["penalty_changed"] == "Y"]
    eligible, reasons = [], {}
    for aid in pen:
        vs = vers.get(aid, [])
        pms = [v["penalty_max_months"] for v in vs]
        if len(vs) < 2 or any(p is None for p in pms):
            reasons[aid] = "版本不足/刑档缺失"
            continue
        if any(p >= 8000 for p in pms):
            reasons[aid] = "含无期/死刑"
            continue
        if any(b < a_ for a_, b in zip(pms, pms[1:])):
            reasons[aid] = "后法减轻（单锚点不适用）"
            continue
        if pms[-1] == pms[0]:
            reasons[aid] = "无刑档差异"
            continue
        eligible.append(aid)

    # 对全部条文做一次可复核的资格审计。主实验的判分口径是
    # “最高有期徒刑年数”，因此无期/死刑不会被伪装成一个普通数字。
    audit_rows = []
    reason_counts: Counter[str] = Counter()
    for aid in sorted(vers, key=lambda x: (int(x.split("-")[0]),
                                           int((x.split("-")[1:] + ["0"])[0]))):
        vs = vers[aid]
        pms = [v.get("penalty_max_months") for v in vs]
        row = candidate_rows.get(aid, {})
        if len(vs) < 2:
            reason = "只有1个版本，无法构造版本判别题"
        elif row.get("penalty_changed") != "Y":
            reason = "有修订，但主刑上限没有变化"
        elif any(p is None for p in pms):
            reason = "刑档解析缺失"
        elif any(p >= 8000 for p in pms):
            reason = "含无期徒刑/死刑，当前数值判分不适用"
        elif any(b < a_ for a_, b in zip(pms, pms[1:])):
            reason = "后法减轻；单个行为时点无法得到唯一的从旧兼从轻答案"
        elif pms[-1] == pms[0]:
            reason = "首末版本主刑上限相同"
        else:
            reason = "合格：有限刑档且后法不减轻"
        reason_counts[reason] += 1
        audit_rows.append({
            "article_id": aid,
            "law": vs[0].get("law", "刑法"),
            "n_versions": len(vs),
            "penalty_changed": row.get("penalty_changed", "N"),
            "penalty_events": row.get("penalty_events", ""),
            "eligible": "Y" if aid in eligible else "N",
            "reason": reason,
            "version_penalty_years": ";".join(
                f"{v.get('effective_from')}:"
                + ("?" if p is None
                   else "无期" if p >= 8000 and p < 10000
                   else "死刑" if p >= 10000
                   else f"{p//12}")
                for v, p in zip(vs, pms)
            ),
        })
    audit_path = DATA / "derived" / "query_eligibility.csv"
    with open(audit_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(audit_rows[0]))
        writer.writeheader()
        writer.writerows(audit_rows)

    existing = {}
    if a.out.exists():
        existing = json.loads(a.out.read_text(encoding="utf-8"))
    keep = {k: v for k, v in existing.items() if k in eligible}
    todo = [aid for aid in eligible if aid not in keep]
    print(f"合格条文 {len(eligible)} 条；已有配置 {len(keep)} 条；待生成 {len(todo)} 条")
    if a.dry_run:
        print("合格清单:", ", ".join(eligible))
        print("剔除原因:", reasons)
        return

    key = os.environ.get(a.api_key_env, "")
    if not key:
        raise SystemExit(f"环境变量 {a.api_key_env} 为空")

    generated = {}
    for i, aid in enumerate(todo, 1):
        vs = vers[aid]
        texts = "\n\n".join(f"版本（{v['effective_from']} 起）：{v['text']}" for v in vs)
        prompt = (
            "你在为中文法律检索实验设计案情要素。下面是同一条文的各个历史版本"
            "（不要直接引用版本日期或条号）。请输出一个 JSON 对象，字段：\n"
            "{"
            "\"key_facts\": \"3–6 个具体案情要素（主体、行为、对象、数额/情节等），"
            "确保该行为落入最重的法定刑档；不同版本下也要对应各自的最重档\","
            "\"tier_hint\": \"最高刑档对应的情节表述（如‘数额特别巨大’）\","
            "\"avoid\": \"2–3 个容易混淆的相邻罪名及区分要点\""
            "}\n"
            "要求：key_facts 用自然语言、不得包含条号/法名/日期；只输出 JSON。\n\n"
            + texts
        )
        try:
            raw = call_llm(a.base_url, a.model, key, prompt, a.max_tokens, 120)
            m = re.search(r"\{.*\}", raw, re.S)
            obj = json.loads(m.group(0)) if m else {}
            if not all(k in obj and str(obj[k]).strip() for k in ("key_facts", "tier_hint")):
                raise ValueError("字段缺失")
            generated[aid] = {
                "tier_hint": str(obj["tier_hint"]).strip(),
                "key_facts": str(obj["key_facts"]).strip(),
                "avoid": str(obj.get("avoid", "无")).strip(),
            }
            print(f"  [{i}/{len(todo)}] 第{aid}条 OK")
        except Exception as e:
            print(f"  [{i}/{len(todo)}] 第{aid}条 失败：{e}")

    merged = {**keep, **generated}
    a.out.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    report = {
        "n_articles": len(vers),
        "eligible": eligible,
        "curated": sorted(keep),
        "generated": sorted(generated),
        "failed": sorted(set(todo) - set(generated)),
        "excluded": reasons,
        "eligibility_counts": dict(reason_counts),
        "audit_csv": str(audit_path),
    }
    a.out.with_name("query_targets_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已写入 {a.out}（共 {len(merged)} 条，其中新生成 {len(generated)} 条）")


if __name__ == "__main__":
    main()
