"""零依赖统计工具：配对差值 + 按条文聚类的 bootstrap 置信区间。

用途：实验脚本比较两个条件（同题配对）时，给出
  差值点估计、95% 置信区间、双尾 bootstrap p 值。
聚类 bootstrap 按「条文」重抽样（同一批题来自同一条文时避免把观测当独立），
与设计书里「按条文聚类」的判定口径一致。
"""
from __future__ import annotations

import random
from collections import defaultdict
from typing import Callable, Optional


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def accuracy(records: list[dict], where: Optional[Callable[[dict], bool]] = None) -> float:
    rs = [r for r in records if where is None or where(r)]
    return mean([1.0 if r.get("correct") else 0.0 for r in rs]) if rs else 0.0


def group_accuracy(records: list[dict]) -> dict[str, float]:
    out: dict[str, float] = {}
    groups = sorted({r.get("group") or "?" for r in records})
    for g in groups:
        out[g] = accuracy(records, where=lambda r, g=g: (r.get("group") or "?") == g)
    return out


def paired_cluster_bootstrap(
        base: list[dict], treat: list[dict], value: str = "correct",
        where: Optional[Callable[[dict], bool]] = None,
        cluster: str = "article_id", n_boot: int = 4000, seed: int = 0,
) -> dict:
    """比较两个条件下的配对差值：delta = mean(treat) - mean(base)。

    返回 {"n", "delta", "lo", "hi", "p"}；p 为双尾 bootstrap p 值（≈）。
    """
    b = {r["qid"]: r for r in base}
    t = {r["qid"]: r for r in treat}
    deltas: dict[str, list[float]] = defaultdict(list)
    for qid, rb in b.items():
        rt = t.get(qid)
        if rt is None:
            continue
        if where is not None and not (where(rb) and where(rt)):
            continue
        v = (1.0 if rt.get(value) else 0.0) - (1.0 if rb.get(value) else 0.0)
        deltas[rb.get(cluster, "?")].append(v)
    if not deltas:
        return {"n": 0, "delta": 0.0, "lo": 0.0, "hi": 0.0, "p": 1.0}

    point = mean([v for vs in deltas.values() for v in vs])
    clusters = list(deltas.values())
    rng = random.Random(seed)
    boots: list[float] = []
    for _ in range(n_boot):
        sample = [clusters[rng.randrange(len(clusters))] for _ in clusters]
        boots.append(mean([v for vs in sample for v in vs]))
    boots.sort()
    lo = boots[int(0.025 * len(boots))]
    hi = boots[min(int(0.975 * len(boots)), len(boots) - 1)]
    le = sum(1 for x in boots if x <= 0) / len(boots)
    ge = sum(1 for x in boots if x >= 0) / len(boots)
    p = min(1.0, 2 * min(le, ge))
    return {"n": sum(len(vs) for vs in deltas.values()),
            "delta": point, "lo": lo, "hi": hi, "p": p}


def fmt(x: float, nd: int = 3) -> str:
    return f"{x:.{nd}f}"


def deq_bootstrap(cells: dict[str, list[dict]], value: str = "correct",
                  cluster: str = "article_id", n_boot: int = 4000,
                  seed: int = 0) -> dict:
    """四格 DEQ 的聚类 bootstrap：DEQ = (err_B - err_A) - (err_D - err_C)。

    cells: {"A": records, "B": records, "C": records, "D": records}
    """
    by = {c: {r["qid"]: r for r in recs} for c, recs in cells.items()}
    qids = set(by["A"]) & set(by["B"]) & set(by["C"]) & set(by["D"])
    if not qids:
        return {"n": 0, "deq": 0.0, "lo": 0.0, "hi": 0.0, "p": 1.0}
    per_cluster: dict[str, list[float]] = defaultdict(list)
    for qid in qids:
        e = {c: 1.0 - (1.0 if by[c][qid].get(value) else 0.0) for c in "ABCD"}
        d = (e["B"] - e["A"]) - (e["D"] - e["C"])
        per_cluster[by["A"][qid].get(cluster, "?")].append(d)
    point = mean([v for vs in per_cluster.values() for v in vs])
    clusters = list(per_cluster.values())
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        sample = [clusters[rng.randrange(len(clusters))] for _ in clusters]
        boots.append(mean([v for vs in sample for v in vs]))
    boots.sort()
    lo = boots[int(0.025 * len(boots))]
    hi = boots[min(int(0.975 * len(boots)), len(boots) - 1)]
    le = sum(1 for x in boots if x <= 0) / len(boots)
    ge = sum(1 for x in boots if x >= 0) / len(boots)
    return {"n": sum(len(vs) for vs in per_cluster.values()),
            "deq": point, "lo": lo, "hi": hi, "p": min(1.0, 2 * min(le, ge))}
