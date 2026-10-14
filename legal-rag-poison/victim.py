"""Victim RAG：分块 -> 向量 -> 检索 -> 生成。

后端可插拔，全部支持「不联网也能自检」：
  向量后端 embed_backend : local（sentence-transformers）/ api（OpenAI 兼容
                          /embeddings）/ mock（离线哈希向量，仅用于冒烟自检）
  生成后端 generator     : api（OpenAI 兼容 /chat/completions）/ mock（离线自检）

工程约定：
  * 只用标准库发 HTTP 请求（urllib），API 模式不依赖 openai 包；
  * 每个请求都有超时、重试与指数退避；
  * LLM 请求默认带 JSONL 缓存（key = 模型 + prompt + 采样参数），
    重跑同一批实验不会重复花钱，缓存与结果都落在本地 outdir 下。
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

from common import ArticleVersion, cn2int, penalty_max_months


# ---------------------------------------------------------------- HTTP 工具

def http_post_json(url: str, payload: dict, headers: Optional[dict] = None,
                   timeout: float = 120.0, retries: int = 3,
                   backoff: float = 2.0) -> dict:
    """POST JSON，带重试与指数退避。429/5xx 与网络错误重试。"""
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    hdrs = {"Content-Type": "application/json"}
    if headers:
        hdrs.update(headers)
    last_err = "unknown error"
    for attempt in range(retries + 1):
        req = urllib.request.Request(url, data=body, headers=hdrs, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            try:
                detail = e.read().decode("utf-8", "ignore")[:400]
            except Exception:
                detail = ""
            last_err = f"HTTP {e.code}: {detail}"
            if e.code in (408, 409, 429, 500, 502, 503, 504) and attempt < retries:
                time.sleep(backoff * (2 ** attempt))
                continue
            raise RuntimeError(f"API 请求失败 {url} -> {last_err}")
        except Exception as e:  # URLError / timeout / 连接重置
            last_err = repr(e)
            if attempt < retries:
                time.sleep(backoff * (2 ** attempt))
                continue
            raise RuntimeError(f"API 请求失败 {url} -> {last_err}")
    raise RuntimeError(f"API 请求失败 {url} -> {last_err}")


def sha1_key(*parts: str) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\x1f")
    return h.hexdigest()


class JsonlCache:
    """append-only JSONL 缓存：重跑实验不重复调用 API。"""

    def __init__(self, path: Optional[Path]):
        self.path = Path(path) if path else None
        self.data: dict[str, str] = {}
        if self.path and self.path.exists():
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                        self.data[row["key"]] = row["value"]
                    except Exception:
                        continue

    def get(self, key: str) -> Optional[str]:
        return self.data.get(key)

    def set(self, key: str, value: str) -> None:
        self.data[key] = value
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"key": key, "value": value},
                                   ensure_ascii=False) + "\n")


# ---------------------------------------------------------------- 向量后端

def _hash_embed(text: str, dim: int = 512) -> list[float]:
    """离线 mock 向量：字符二元组哈希。只为跑通流程，不用于论文。"""
    t = re.sub(r"\s+", "", text)
    grams = [t[i:i + 2] for i in range(max(len(t) - 1, 1))] or [t]
    vec = [0.0] * dim
    for g in grams:
        h = int(hashlib.md5(g.encode("utf-8")).hexdigest()[:8], 16)
        vec[h % dim] += 1.0 if (h >> 31) & 1 else -1.0
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def _matvec(vecs: list[list[float]], q: list[float]) -> list[float]:
    try:  # 有 numpy 就快一点，没有也能跑（本项目规模纯 Python 足够）
        import numpy as np  # type: ignore
        return (np.asarray(vecs, dtype=float) @ np.asarray(q, dtype=float)).tolist()
    except Exception:
        return [sum(a * b for a, b in zip(v, q)) for v in vecs]


class Embedder:
    """向量化。backend: local / api / mock。

    * local: 需要 pip install sentence-transformers（CPU 可跑 bge-small-zh）
    * api  : 任意 OpenAI 兼容 /embeddings 接口（如硅基流动 bge-m3、DashScope）
    * mock : 离线自检用
    """

    def __init__(self, backend: str = "local",
                 model: str = "BAAI/bge-small-zh-v1.5",
                 base_url: Optional[str] = None,
                 api_key: Optional[str] = None,
                 batch_size: int = 32,
                 cache_dir: Optional[Path] = None,
                 timeout: float = 120.0,
                 retries: int = 3):
        self.backend = backend
        self.model = model
        self.base_url = (base_url or "").rstrip("/")
        self.api_key = api_key
        self.batch_size = max(int(batch_size), 1)
        self.timeout = timeout
        self.retries = retries
        self._cache: dict[str, list[float]] = {}
        self._cache_path = (Path(cache_dir) / "embeddings.jsonl") if cache_dir else None
        self._st = None
        if self._cache_path and self._cache_path.exists():
            with open(self._cache_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        row = json.loads(line)
                        self._cache[row["k"]] = row["v"]
                    except Exception:
                        continue

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        """批量编码，自动命中缓存、分批调用后端。"""
        keys = [sha1_key(self.backend, self.model, t) for t in texts]
        todo = [i for i, k in enumerate(keys) if k not in self._cache]
        for s in range(0, len(todo), self.batch_size):
            idxs = todo[s:s + self.batch_size]
            vecs = self._encode_batch([texts[i] for i in idxs])
            for i, v in zip(idxs, vecs):
                self._cache[keys[i]] = v
        self._save_cache()
        return [self._cache[k] for k in keys]

    def _encode_batch(self, texts: list[str]) -> list[list[float]]:
        if self.backend == "mock":
            return [_hash_embed(t) for t in texts]
        if self.backend == "local":
            if self._st is None:
                from sentence_transformers import SentenceTransformer  # 延迟导入
                self._st = SentenceTransformer(self.model)
            vecs = self._st.encode(list(texts), normalize_embeddings=True,
                                   show_progress_bar=False)
            return [[float(x) for x in v] for v in vecs]
        if self.backend == "api":
            if not self.base_url:
                raise RuntimeError("embed_backend=api 需要 --embed-base-url")
            resp = http_post_json(
                f"{self.base_url}/embeddings",
                {"model": self.model, "input": list(texts)},
                headers={"Authorization": f"Bearer {self.api_key or ''}"},
                timeout=self.timeout, retries=self.retries)
            rows = sorted(resp["data"], key=lambda r: r.get("index", 0))
            return [list(map(float, r["embedding"])) for r in rows]
        raise ValueError(f"未知 embed backend: {self.backend}")

    def _save_cache(self) -> None:
        if not self._cache_path:
            return
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._cache_path, "w", encoding="utf-8") as f:
            for k, v in self._cache.items():
                f.write(json.dumps({"k": k, "v": v}, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------- 分块

def chunk_fixed(text: str, size: int = 512, overlap: int = 0) -> list[str]:
    step = max(size - overlap, 1)
    return [text[i:i + size] for i in range(0, len(text), step)] or [text]


def chunk_article(text: str, _size: int = 0, _overlap: int = 0) -> list[str]:
    """按条文边界：整条一整块（不切）。用于「缺陷已修」对照。"""
    return [text]


CHUNKERS: dict[str, Callable[..., list[str]]] = {
    "fixed512": lambda t: chunk_fixed(t, 512, 0),
    "fixed512o": lambda t: chunk_fixed(t, 512, 64),
    "article": chunk_article,
}


# ---------------------------------------------------------------- 索引

@dataclass
class Doc:
    """索引里的一个文档（一块）。injected=True 表示是攻击载荷。"""
    doc_id: str
    text: str
    version_id: str
    article_id: str
    effective_from: Optional[str]
    effective_to: Optional[str]
    injected: bool = False
    meta: dict = field(default_factory=dict)


class Index:
    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self.docs: list[Doc] = []
        self.vecs: list[list[float]] = []

    def build(self, versions: list[ArticleVersion], chunker: str = "fixed512") -> None:
        self.docs = []
        self.vecs = []
        self.add_versions(versions, chunker=chunker, background=False)

    def add_versions(self, versions: list[ArticleVersion], chunker: str = "fixed512",
                     background: bool = False) -> None:
        """把一个条文版本集合切块后加入索引。

        background=True 的文档只参与检索竞争（干扰项），不参与出题/gold，
        用于「全量现行刑法背景 + 焦点条文多版本」的语料设计。
        """
        fn = CHUNKERS[chunker]
        new_docs: list[Doc] = []
        for v in versions:
            for i, c in enumerate(fn(v.text)):
                meta = dict(v.meta)
                if background:
                    meta["background"] = True
                new_docs.append(Doc(
                    doc_id=f"{v.version_id}#{i}",
                    text=c,
                    version_id=v.version_id,
                    article_id=v.article_id,
                    effective_from=v.effective_from,
                    effective_to=v.effective_to,
                    injected=False,
                    meta=meta,
                ))
        self.vecs += self.embedder.encode([d.text for d in new_docs])
        self.docs += new_docs

    def add_docs(self, docs: list[Doc]) -> None:
        """注入攻击载荷（内容与真本同文，但 injected=True）。"""
        if not docs:
            return
        self.vecs += self.embedder.encode([d.text for d in docs])
        self.docs += docs

    def search(self, query: str, k: int = 5, as_of: Optional[str] = None,
               unknown_dates: str = "reject") -> list[tuple[Doc, float]]:
        """as_of 非空则启用版本感知过滤（D5）。

        unknown_dates: 对 effective_from 为空的文档（如 A1 载荷）
                       "reject"=拒收（严格），"accept"=放行（宽松）。
        """
        q = self.embedder.encode([query])[0]
        sims = _matvec(self.vecs, q)
        cand = list(enumerate(sims))
        if as_of is not None:
            cand = [(i, s) for i, s in cand
                    if self._covers(self.docs[i], as_of, unknown_dates)]
        cand.sort(key=lambda x: -x[1])
        return [(self.docs[i], float(s)) for i, s in cand[:k]]

    @staticmethod
    def _covers(d: Doc, date: str, unknown_dates: str = "reject") -> bool:
        if d.effective_from is None:
            return unknown_dates == "accept"
        if date < d.effective_from:
            return False
        if d.effective_to is not None and date > d.effective_to:
            return False
        return True


# ---------------------------------------------------------------- 生成

PROMPT = """你是中国法律助手。只能依据下列检索到的条文作答。

[检索到的条文]
{context}

[问题]
{query}

要求：
1. 先判断应当适用哪一个版本的条文；
2. 再给出最高可判处的主刑年限；
3. 最后一行严格按此格式输出，不要有其他内容：第X条｜Y年
"""


class ApiGenerator:
    """OpenAI 兼容 /chat/completions 生成后端。"""

    def __init__(self, model: str, base_url: str, api_key: str,
                 temperature: float = 0.0, max_tokens: int = 512,
                 timeout: float = 120.0, retries: int = 3,
                 cache: Optional[JsonlCache] = None, sleep: float = 0.0):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.retries = retries
        self.cache = cache
        self.sleep = sleep
        if not self.base_url:
            raise RuntimeError("generator=api 需要 --llm-base-url")

    def _key(self, prompt: str) -> str:
        return sha1_key(self.model, self.base_url, prompt,
                        str(self.temperature), str(self.max_tokens))

    def __call__(self, prompt: str) -> str:
        key = self._key(prompt)
        if self.cache is not None:
            hit = self.cache.get(key)
            if hit is not None:
                return hit
        resp = http_post_json(
            f"{self.base_url}/chat/completions",
            {"model": self.model,
             "messages": [{"role": "user", "content": prompt}],
             "temperature": self.temperature,
             "max_tokens": self.max_tokens},
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=self.timeout, retries=self.retries)
        text = resp["choices"][0]["message"]["content"]
        if self.cache is not None:
            self.cache.set(key, text)
        if self.sleep:
            time.sleep(self.sleep)
        return text


class MockGenerator:
    """离线自检：模拟「完全照抄 top-1 检索块」的模型，用于验证流程。"""

    BLOCK_RE = re.compile(
        r"\[1\]\s*第\s*([0-9零一两二三四五六七八九十]+)\s*条[^\n]*\n(.*?)"
        r"(?=\n\[2\]|\n\n\[问题\])", re.S)

    def __call__(self, prompt: str) -> str:
        m = self.BLOCK_RE.search(prompt)
        if not m:
            return "第0条｜0年"
        aid = str(cn2int(m.group(1))) if not m.group(1).isdigit() else m.group(1)
        months = penalty_max_months(m.group(2))
        years = (months or 0) // 12
        return f"第{aid}条｜{years}年"


# ---------------------------------------------------------------- 组装

class LegalRAG:
    def __init__(self, index: Index, gen: Callable[[str], str]):
        self.index = index
        self.gen = gen

    def answer(self, query: str, k: int = 5, as_of: Optional[str] = None,
               unknown_dates: str = "reject",
               context_metadata: str = "none") -> dict:
        hits = self.index.search(query, k=k, as_of=as_of,
                                 unknown_dates=unknown_dates)
        lines = []
        for i, (d, _) in enumerate(hits):
            head = f"[{i+1}] 第{d.article_id}条"
            if context_metadata == "version_id":
                head += f"（版本 {d.version_id}）"
            lines.append(f"{head}\n{d.text}")
        context = "\n\n".join(lines)
        prompt = PROMPT.format(context=context, query=query)
        out = self.gen(prompt)
        return {
            "answer": out,
            "prompt": prompt,
            "retrieved": [
                {"doc_id": d.doc_id, "version_id": d.version_id,
                 "article_id": d.article_id, "injected": d.injected,
                 "score": s}
                for d, s in hits
            ],
            "retrieved_versions": [d.version_id for d, _ in hits],
        }


# ---------------------------------------------------------------- 引用校验

ART_RE = re.compile(r"第\s*([0-9零一两二三四五六七八九十]+)\s*条")


def verify_citation(answer: str, known_articles: set[str]) -> bool:
    """引用校验 D1：答案援引的条号必须存在于语料。"""
    for m in ART_RE.finditer(answer):
        aid = m.group(1)
        if aid.isdigit():
            aid = str(int(aid))
        if aid not in known_articles:
            return False
    return True
