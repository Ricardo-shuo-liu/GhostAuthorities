"""生成 10000 条初始 query（old/new 各 5000，自然案情，非模板）。

设计要点：
  * 案情由 LLM 撰写（人物/公司/金额/情节各不相同），脚本只负责约束与校验；
  * 八条标准在生成时自检：
      1) 无条号/法名/ISO 日期；2) 恰好一个时间锚点（年月，落在版本有效期内）；
      3) 罪状要素取自条文（提示词给定）；4) gold 可算且后法不减轻；
      5) 答案格式固定为最后一行「第X条｜Y年」；6) 闭卷过滤交给 filter_closedbook.py；
      7) 区分性细节由 query_targets.json 提供；8) 跨版本边界题不在本文件（单独处理）。
  * 输出 JSONL：qid / article_id / bank / query / fact_date / gold_version_id /
    gold_years / gold_nuggets / note

用法（仓库根目录；需一个 OpenAI 兼容的 LLM API）：
    # 只生成排期（不调 API，检查 old/new 是否 5000/5000）
    python scripts/generate_queries.py --plan-only

    # 正式生成（默认 10000 条，old/new 各半；先小规模试跑）
    python scripts/generate_queries.py --num 200 --out data/queries/candidates_200.jsonl
    python scripts/generate_queries.py --num 10000

环境变量（或命令行覆盖）：
    QUERY_GEN_BASE_URL   默认 https://api.siliconflow.cn/v1
    QUERY_GEN_MODEL      默认 deepseek-ai/DeepSeek-V3（按你的账号可用模型改）
    QUERY_GEN_API_KEY_ENV 默认 SILICONFLOW_API_KEY
"""
from __future__ import annotations

import argparse
import calendar
import hashlib
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from ragpoison.common import ArticleVersion, load_corpus  # noqa: E402
from ragpoison.env import load_dotenv                     # noqa: E402

DATA = ROOT / "data"

TAIL_VARIANTS = [
    ("依照案发时有效的法律规定，该行为最高可判处有期徒刑多少年？"
     "最后一行请严格按格式输出：第X条｜Y年"),
    ("如果按照行为发生时施行的法律处理，这种行为最高会被判处多少年有期徒刑？"
     "最后一行请严格按格式输出：第X条｜Y年"),
    ("假定法院依据案发时适用的法律裁判，责任人最高可能被判处多少年有期徒刑？"
     "最后一行请严格按格式输出：第X条｜Y年"),
    ("以行为发生时的法律规定为准，这类行为的有期徒刑上限是多少年？"
     "最后一行请严格按格式输出：第X条｜Y年"),
    ("若按照案件发生当时有效的法律判断，最高刑期可达到多少年？"
     "最后一行请严格按格式输出：第X条｜Y年"),
    ("根据该行为发生当时适用的法律，法院最高可以判处多少年有期徒刑？"
     "最后一行请严格按格式输出：第X条｜Y年"),
]

FORBIDDEN_PATTERNS = [
    ("条号", re.compile(r"第[0-9一二三四五六七八九十百千零两]+条")),
    ("法条名", re.compile(r"《[^》]*》")),
    ("ISO 日期", re.compile(r"\d{4}[-/.]\d{1,2}([-/.]\d{1,2})?")),
    ("法名", re.compile(r"刑法|修正案|司法解释")),
    ("元话语", re.compile(r"从旧兼从轻|案发时有效|适用哪一版|版本")),
]


def sanitize_hint(text: str) -> str:
    """把目标配置中的答案坐标从生成提示词里移除。

    key_facts 只应该告诉模型“发生了什么”，不能把条号、法名或版本日期
    当作提示信号带进 query 生成阶段。
    """
    s = str(text or "")
    s = re.sub(r"《[^》]*》", "相关法律", s)
    s = re.sub(r"第\s*[0-9一二三四五六七八九十百千零两]+"
               r"(?:\s*条)?(?:\s*之[一二三四五六七八九十]+)?",
               "相关条款", s)
    s = re.sub(r"\b(?:19|20)\d{2}[-/.]\d{1,2}(?:[-/.]\d{1,2})?(?![0-9])",
               "", s)
    s = re.sub(r"\b(?:19|20)\d{2}\s*年", "", s)
    s = s.replace("刑法修正案", "法律修改")
    s = s.replace("刑法", "法律").replace("修正案", "法律修改")
    s = s.replace("司法解释", "相关解释")
    return re.sub(r"\s+", " ", s).strip()


def http_post(url: str, payload: dict, api_key: str, timeout: float,
              retries: int = 3) -> dict:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    last = "unknown"
    for attempt in range(retries + 1):
        req = urllib.request.Request(
            url, data=body, method="POST",
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {api_key}"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = e.read().decode("utf-8", "ignore")[:300]
            except Exception:
                pass
            last = f"HTTP {e.code}: {detail}"
            if e.code in (408, 409, 429, 500, 502, 503, 504) and attempt < retries:
                time.sleep(2 * (2 ** attempt))
                continue
            raise RuntimeError(last)
        except Exception as e:
            last = repr(e)
            if attempt < retries:
                time.sleep(2 * (2 ** attempt))
                continue
            raise RuntimeError(last)
    raise RuntimeError(last)


def last_day(y: int, m: int) -> date:
    return date(y, m, calendar.monthrange(y, m)[1])


def month_inside(ym: tuple[int, int], start: date, end: Optional[date]) -> bool:
    first = date(ym[0], ym[1], 1)
    if first < start:
        return False
    if end is not None and last_day(*ym) > end:
        return False
    return True


def era_months(start_iso: str, end_iso: Optional[str]) -> list[tuple[int, int]]:
    s = date.fromisoformat(start_iso)
    e = date.fromisoformat(end_iso) if end_iso else None
    if e is None:
        cands = [date(2021, 6, 1), date(2022, 3, 1), date(2023, 8, 1),
                 date(2024, 9, 1), date(2025, 6, 1), date(2026, 4, 1)]
    else:
        span = max((e - s).days, 1)
        cands = [s + timedelta(days=max(20, int(span * f)))
                 for f in (0.10, 0.35, 0.60, 0.85)]
    out = []
    for c in cands:
        ym = (c.year, c.month)
        if ym not in out and month_inside(ym, s, e):
            out.append(ym)
    return out


def versions_at(vs: list[ArticleVersion], d: date) -> Optional[ArticleVersion]:
    iso = d.isoformat()
    for v in vs:
        if v.covers(iso):
            return v
    return None


def target_gold_years(target: dict, version: ArticleVersion) -> Optional[int]:
    """Return the tier-specific gold years when configured, else max years."""
    mapping = target.get("gold_years_by_version") or {}
    if version.version_id in mapping:
        return int(mapping[version.version_id])
    if version.penalty_max_months is None:
        return None
    return version.penalty_max_months // 12


@dataclass
class Seed:
    article_id: str
    bank: str
    ym: tuple[int, int]
    gold_version_id: str
    gold_years: int


def build_seeds(total: int, versions: dict[str, list[ArticleVersion]],
                targets: dict) -> tuple[list[Seed], dict[tuple[str, str], int]]:
    """按「条文 × old/new」12 个格子严格配额；返回 (seeds, quotas)。"""
    banks = {"old": [], "new": []}
    pool = {aid: vs for aid, vs in versions.items() if aid in targets}
    aids = sorted(pool)
    # 为每个 bank 生成候选 (article, ym, gold)
    for aid in aids:
        vs = pool[aid]
        if len(vs) < 2:
            continue
        target = targets[aid]
        ys = [target_gold_years(target, v) for v in vs]
        if any(y is None for y in ys):
            continue
        cur_y = ys[-1]
        if all(y == cur_y for y in ys):
            continue
        eras = []
        for i, v in enumerate(vs):
            end = ((date.fromisoformat(vs[i + 1].effective_from) - timedelta(days=1)).isoformat()
                   if i + 1 < len(vs) else None)
            eras.append((v, end))
        current = vs[-1]
        # 对普通“最高刑期”目标，保留后法不减轻约束；tier 目标由
        # gold_years_by_version 显式指定，不把“最高刑”当作答案。
        if "gold_years_by_version" not in target:
            pms = [v.penalty_max_months or 0 for v in vs]
            if any(b < a for a, b in zip(pms, pms[1:])):
                continue
        for i, (v, end) in enumerate(eras):
            for ym in era_months(v.effective_from, end):
                g = versions_at(vs, date(ym[0], ym[1], 15))
                if g is None:
                    continue
                gy = target_gold_years(target, g)
                if gy is None:
                    continue
                bank = "old" if g.version_id != current.version_id else "new"
                if bank == "old" and gy == cur_y:
                    continue
                if bank == "new" and not any(y != cur_y for y in ys[:-1]):
                    continue
                banks[bank].append(Seed(aid, bank, ym, g.version_id, gy))
    cells = sorted((aid, bank) for aid in pool for bank in ("old", "new")
                   if banks[bank] and any(s.article_id == aid for s in banks[bank]))
    if not cells:
        raise SystemExit("没有可用格子（检查条文/版本表）")
    base, rem = divmod(total, len(cells))
    quotas = {c: base + (1 if i < rem else 0) for i, c in enumerate(cells)}
    rng = random.Random(20261019)
    seeds: list[Seed] = []
    for cell in cells:
        aid, bank = cell
        pool_cell = [s for s in banks[bank] if s.article_id == aid]
        if not pool_cell:
            continue
        n = quotas[cell]
        made = 0
        while made < n:
            rng.shuffle(pool_cell)
            for s in pool_cell:
                seeds.append(s)
                made += 1
                if made >= n:
                    break
    return seeds, quotas


def shingles(text: str, n: int = 5) -> set[str]:
    t = re.sub(r"\s+", "", text)
    return {t[i:i + n] for i in range(max(1, len(t) - n + 1))}


def make_prompt(seed: Seed, target: dict, k: int, nonce: int, tail: str) -> str:
    y, m = seed.ym
    names = ["王某", "李某", "张某", "赵某", "孙某", "周某", "吴某", "郑某",
             "冯某", "陈某", "褚某", "卫某", "蒋某", "沈某", "韩某", "杨某"]
    units = ["某投资管理公司", "某财富管理公司", "某商贸公司", "某科技公司",
             "某实业集团", "某咨询公司", "某制造企业", "某贸易公司"]
    scales = ["数百万元", "数千万元", "上亿元", "数亿元", "特别巨大"]
    rng = random.Random(nonce)
    angles = ["线下推介会", "网络广告", "亲友介绍", "电话推销", "社区宣讲",
              "理财讲座", "微信群推广", "门店宣传"]
    roles = ["公司实际控制人", "业务经理", "销售人员", "财务人员", "项目负责人",
             "门店店长", "客服主管"]
    hint = (f"可变元素建议（随机取用）：人物可姓{rng.choice(names)}；"
            f"单位可用{rng.choice(units)}；金额量级可用{rng.choice(scales)}；"
            f"角色可用{rng.choice(roles)}；场景可用{rng.choice(angles)}。")
    return (
        f"请写 {k} 段彼此不同的中文案情叙述，案情发生时间均为「{y}年{m}月」。\n"
        "要求：\n"
        "1. 每段 80–150 字，第三人称，像真实案件摘要；人物姓名、公司、金额、"
        "职业、交易方式等细节必须彼此不同；\n"
        f"2. 案情必须体现这些要素：{sanitize_hint(target['key_facts'])}；"
        f"并且达到「{sanitize_hint(target['tier_hint'])}」的程度；\n"
        f"3. 注意与下列情形区分：{sanitize_hint(target.get('avoid', '无'))}；\n"
        "4. 不要出现任何法条编号、法律文件名称、'第X条'字样，不要提'刑法'、'修正案'、"
        "'版本'、'从旧兼从轻'等词；\n"
        f"5. 时间只写一次，就写「{y}年{m}月」，不要写具体日期，也不要出现其他年份或月份；\n"
        "6. 不要照抄法条原句，用自然叙事表达；\n"
        f"7. {hint}\n"
        f"8. 最后只需要写案情正文，不要写提问句；系统会追加固定问句：{tail}\n"
        "9. 只输出 JSON 数组，形如 [\"案情一\", \"案情二\", ...]，不要输出其它文字。"
    )


def validate_narrative(text: str, seed: Seed) -> Optional[str]:
    t = text.strip()
    if not (60 <= len(t) <= 260):
        return "长度异常"
    for name, rx in FORBIDDEN_PATTERNS:
        if rx.search(t):
            return f"含{name}"
    anchors = re.findall(r"((?:19|20)\d{2})年(\d{1,2})月", t)
    if len(anchors) != 1:
        return "时间锚点不是恰好一个"
    y, m = seed.ym
    if (int(anchors[0][0]), int(anchors[0][1])) != (y, m):
        return "时间锚点与排期不符"
    # 其它年份出现（如"1998年"）也不允许
    if len(re.findall(r"(?:19|20)\d{2}年", t)) != 1:
        return "出现多个年份"
    if re.search(r"去年|前年|今年|次年|同年|上月|下月", t):
        return "出现第二个时间指代"
    return None


def interleave_by_cell(rows: list[dict], seed: int = 20261021) -> list[dict]:
    """让文件前缀也覆盖所有条文，而不是按并发批次聚集在一起。"""
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row["article_id"], []).append(row)
    rng = random.Random(seed)
    for group in groups.values():
        rng.shuffle(group)
    articles = sorted(groups)
    out: list[dict] = []
    level = 0
    while True:
        added = False
        for article_id in articles:
            group = groups[article_id]
            if level < len(group):
                out.append(group[level])
                added = True
        if not added:
            break
        level += 1
    return out


def call_llm(base_url: str, model: str, api_key: str, prompt: str,
             max_tokens: int, timeout: float) -> str:
    def post(mt: int) -> dict:
        return http_post(base_url.rstrip("/") + "/chat/completions", {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.8, "max_tokens": mt,
        }, api_key, timeout)

    resp = post(max_tokens)
    msg = (resp.get("choices") or [{}])[0].get("message") or {}
    text = msg.get("content") or ""
    if not text:            # 思考型模型可能把预算花在推理上：放大重试一次
        resp = post(min(max(max_tokens * 2, 8192), 16384))
        msg = (resp.get("choices") or [{}])[0].get("message") or {}
        text = msg.get("content") or ""
    return text


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--num", type=int, default=10000)
    ap.add_argument("--per-call", type=int, default=8, help="每次 LLM 请求产出几条案情")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--targets", type=Path, default=DATA / "query_targets.json")
    ap.add_argument("--versions", type=Path, default=DATA / "derived" / "article_versions.jsonl")
    ap.add_argument("--out", type=Path, default=DATA / "queries" / "candidates_10000.jsonl")
    ap.add_argument("--report", type=Path, default=None,
                    help="生成报告路径；默认与输出同目录的 generation_report.json")
    ap.add_argument("--base-url",
                    default=os.environ.get("QUERY_GEN_BASE_URL")
                    or os.environ.get("LLM_BASE_URL", "https://api.deepseek.com"))
    ap.add_argument("--model",
                    default=os.environ.get("QUERY_GEN_MODEL")
                    or os.environ.get("LLM_MODEL", "deepseek-chat"))
    ap.add_argument("--api-key-env", default="LLM_API_KEY")
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--plan-only", action="store_true")
    a = ap.parse_args()

    targets = json.loads(a.targets.read_text(encoding="utf-8"))
    versions: dict[str, list[ArticleVersion]] = {}
    for v in load_corpus(a.versions):
        versions.setdefault(v.article_id, []).append(v)
    for vs in versions.values():
        vs.sort(key=lambda x: x.effective_from)

    seeds, quotas = build_seeds(a.num, versions, targets)
    n_old = sum(1 for s in seeds if s.bank == "old")
    print(f"排期：共 {len(seeds)} 条（old {n_old} / new {len(seeds) - n_old}）；"
          f"每格配额：{sorted(set(quotas.values()))}")
    if a.plan_only:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        plan = a.out.with_name(a.out.stem + "_plan.jsonl")
        with open(plan, "w", encoding="utf-8") as f:
            for s in seeds:
                f.write(json.dumps({"article_id": s.article_id, "bank": s.bank,
                                    "ym": f"{s.ym[0]}年{s.ym[1]}月",
                                    "gold_version_id": s.gold_version_id,
                                    "gold_years": s.gold_years},
                                   ensure_ascii=False) + "\n")
        print(f"排期已写入 {plan}（未调用 API）")
        return

    key = os.environ.get(a.api_key_env, "")
    if not key:
        raise SystemExit(f"环境变量 {a.api_key_env} 为空；先在 .env 里配置")
    a.out.parent.mkdir(parents=True, exist_ok=True)

    # 续跑：读取已有输出
    accepted: list[dict] = []
    seen_hash: set[str] = set()
    seen_shingles: dict[str, list[set[str]]] = {}
    if a.out.exists():
        with open(a.out, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                accepted.append(r)
                seen_hash.add(hashlib.md5(r["query"].encode()).hexdigest())
                seen_shingles.setdefault(r["article_id"], []).append(shingles(r["query"]))
    cnt_cell: dict[tuple[str, str], int] = {c: 0 for c in quotas}
    for r in accepted:
        cell = (r["article_id"], r["bank"])
        cnt_cell[cell] = cnt_cell.get(cell, 0) + 1
    print(f"已有 {sum(cnt_cell.values())}，目标 {sum(quotas.values())}；开始生成…")

    write_lock = __import__("threading").Lock()

    def process_batch(seed: Seed) -> None:
        cell = (seed.article_id, seed.bank)
        if cnt_cell.get(cell, 0) >= quotas.get(cell, 0):
            return
        nonce = random.randrange(1 << 30)
        tail = TAIL_VARIANTS[nonce % len(TAIL_VARIANTS)]
        prompt = make_prompt(seed, targets[seed.article_id], a.per_call,
                             nonce, tail)
        try:
            raw = call_llm(a.base_url, a.model, key, prompt, a.max_tokens, a.timeout)
        except Exception as e:
            print(f"[警告] 批次失败 {seed.article_id}/{seed.bank}: {e}")
            return
        m = re.search(r"\[.*\]", raw, re.S)
        if not m:
            return
        try:
            texts = json.loads(m.group(0))
        except Exception:
            return
        for t in texts:
            if not isinstance(t, str):
                continue
            err = validate_narrative(t, seed)
            if err:
                continue
            full = (f"{t.strip()}。{tail}" if not t.strip().endswith("。")
                    else f"{t.strip()}{tail}")
            h = hashlib.md5(full.encode()).hexdigest()
            with write_lock:
                if h in seen_hash or cnt_cell.get(cell, 0) >= quotas.get(cell, 0):
                    continue
                sh = shingles(full)
                prev = seen_shingles.setdefault(seed.article_id, [])
                if any(len(sh & p) / max(1, len(sh | p)) > 0.8 for p in prev[-400:]):
                    continue
                seen_hash.add(h)
                prev.append(sh)
                cnt_cell[cell] += 1
                rec = {
                    "qid": f"Q-{seed.article_id}-{seed.bank}-{cnt_cell[cell]:05d}",
                    "article_id": seed.article_id, "bank": seed.bank,
                    "query": full,
                    "fact_date": f"{seed.ym[0]:04d}-{seed.ym[1]:02d}-15",
                    "gold_version_id": seed.gold_version_id,
                    "gold_years": seed.gold_years,
                    "gold_nuggets": {"article_id": seed.article_id,
                                     "years": seed.gold_years},
                    "scenario": "行为时法（旧版）" if seed.bank == "old" else "行为时法（现行）",
                }
                with open(a.out, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        done_n = sum(cnt_cell.values())
        print(f"  进度 {done_n}/{sum(quotas.values())}（本格 "
              f"{seed.article_id}/{seed.bank}: {cnt_cell[cell]}/{quotas[cell]}）")

    rng = random.Random(20261020)
    order = seeds[:]
    rng.shuffle(order)
    with ThreadPoolExecutor(max_workers=max(1, a.workers)) as ex:
        pending = []
        idx = 0
        while any(cnt_cell.get(c, 0) < quotas[c] for c in quotas):
            while len(pending) < a.workers and idx < len(order):
                s = order[idx]
                idx += 1
                if cnt_cell.get((s.article_id, s.bank), 0) < quotas.get((s.article_id, s.bank), 0):
                    pending.append(ex.submit(process_batch, s))
            if not pending:
                if idx >= len(order):
                    idx = 0
                    rng.shuffle(order)
                    continue
                break
            done = next(as_completed(pending))
            pending.remove(done)
            if idx >= len(order) and any(cnt_cell.get(c, 0) < quotas[c] for c in quotas):
                idx = 0
                rng.shuffle(order)
    by_cell = {f"{a_}/{b}": n for (a_, b), n in sorted(cnt_cell.items())}
    print(f"\n完成：{sum(cnt_cell.values())} 条 -> {a.out}")
    print("每格:", by_cell)

    # 读取当前文件，并按条文 round-robin 重写。并发生成时同一批次的
    # 8 条会连续落盘；不重排会让抽样看到的前几十条只覆盖少数条文。
    rows = []
    with open(a.out, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    rows = interleave_by_cell(rows)
    with open(a.out, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # 生成报告必须反映当前文件，而不是上一次六条文试跑的结果。
    bad_regex = sum(
        1 for r in rows
        if re.search(r"第[0-9一二三四五六七八九十百千零两]+条|"
                    r"《[^》]*》|\d{4}[-/.]\d{1,2}([-/.]\d{1,2})?|"
                    r"刑法|修正案|司法解释", r["query"])
    )
    bad_anchor = sum(
        1 for r in rows
        if len(re.findall(r"(?:19|20)\d{2}年\d{1,2}月", r["query"])) != 1
    )
    bad_gold = 0
    for r in rows:
        v = next((x for x in versions.get(r["article_id"], [])
                  if x.version_id == r["gold_version_id"]), None)
        expected = target_gold_years(targets[r["article_id"]], v) if v else None
        if v is None or expected != int(r["gold_years"]):
            bad_gold += 1
    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": a.model, "api": a.base_url,
        "total": len(rows),
        "articles": len({r["article_id"] for r in rows}),
        "bank": dict(sorted(Counter(r["bank"] for r in rows).items())),
        "per_article_bank": {
            f"{aid}|{bank}": sum(
                1 for r in rows if r["article_id"] == aid and r["bank"] == bank)
            for aid in sorted({r["article_id"] for r in rows})
            for bank in ("old", "new")
        },
        "regex_violations": bad_regex,
        "anchor_violations": bad_anchor,
        "gold_violations": bad_gold,
        "exact_duplicates": len(rows) - len({r["query"] for r in rows}),
        "avg_length": round(sum(len(r["query"]) for r in rows) / max(1, len(rows)), 1),
        "min_length": min((len(r["query"]) for r in rows), default=0),
        "max_length": max((len(r["query"]) for r in rows), default=0),
    }
    report_path = a.report or a.out.with_name("generation_report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    print(f"报告 -> {report_path}")


if __name__ == "__main__":
    main()
