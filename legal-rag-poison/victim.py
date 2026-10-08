"""Victim RAG：分块 -> 向量 -> FAISS -> 生成。

两个可插拔后端：
  - Embedding: sentence-transformers (bge-small-zh-v1.5)
  - Generator : 本地 transformers / OpenAI 兼容 API

无 GPU 时把 Generator 换成 API 即可，检索部分 CPU 就能跑。
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from common import ArticleVersion


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


@dataclass
class Doc:
    """索引里的一个文档（一块）。"""
    doc_id: str
    text: str
    version_id: str
    article_id: str
    effective_from: str
    effective_to: Optional[str]
    meta: dict = field(default_factory=dict)


# ---------------------------------------------------------------- 索引

class Index:
    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.docs: list[Doc] = []
        self.vecs: Optional[np.ndarray] = None

    def build(self, versions: list[ArticleVersion], chunker: str = "fixed512") -> None:
        fn = CHUNKERS[chunker]
        self.docs = []
        for v in versions:
            for i, c in enumerate(fn(v.text)):
                self.docs.append(Doc(
                    doc_id=f"{v.version_id}#{i}",
                    text=c,
                    version_id=v.version_id,
                    article_id=v.article_id,
                    effective_from=v.effective_from,
                    effective_to=v.effective_to,
                    meta=dict(v.meta),
                ))
        texts = [d.text for d in self.docs]
        self.vecs = self.model.encode(texts, normalize_embeddings=True,
                                      show_progress_bar=True)

    def search(self, query: str, k: int = 5,
               as_of: Optional[str] = None) -> list[tuple[Doc, float]]:
        """as_of 非空则启用版本感知过滤（D5 修复）。"""
        q = self.model.encode([query], normalize_embeddings=True)[0]
        sims = self.vecs @ q

        cand = list(enumerate(sims))
        if as_of is not None:
            cand = [(i, s) for i, s in cand if self._covers(self.docs[i], as_of)]
        cand.sort(key=lambda x: -x[1])
        return [(self.docs[i], float(s)) for i, s in cand[:k]]

    @staticmethod
    def _covers(d: Doc, date: str) -> bool:
        if date < d.effective_from:
            return False
        if d.effective_to is not None and date > d.effective_to:
            return False
        return True


# ---------------------------------------------------------------- 生成

PROMPT = """你是中国刑事法律助手。只能依据下列检索到的条文作答。

[检索到的条文]
{context}

[问题]
{query}

要求：
1. 先判断应当适用哪一个版本的条文；
2. 再给出最高可判处的主刑年限；
3. 最后一行严格按此格式输出，不要有其他内容：第X条｜Y年
"""


class Generator:
    """backend='local' 用 transformers；backend='api' 用 OpenAI 兼容接口。"""

    def __init__(self, backend: str = "api", model: str = "deepseek-chat"):
        self.backend = backend
        self.model = model
        if backend == "local":
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            self.tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
            self.llm = AutoModelForCausalLM.from_pretrained(
                model, torch_dtype=torch.float16, device_map="auto",
                trust_remote_code=True)
        else:
            from openai import OpenAI
            self.client = OpenAI(
                api_key=os.environ.get("OPENAI_API_KEY", "sk-x"),
                base_url=os.environ.get("OPENAI_BASE_URL", "https://api.deepseek.com"))

    def __call__(self, prompt: str) -> str:
        if self.backend == "local":
            import torch
            inputs = self.tok(prompt, return_tensors="pt").to(self.llm.device)
            with torch.no_grad():
                out = self.llm.generate(**inputs, max_new_tokens=256,
                                        temperature=0.0, do_sample=False)
            return self.tok.decode(out[0][inputs.input_ids.shape[1]:],
                                   skip_special_tokens=True)
        r = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0, max_tokens=512)
        return r.choices[0].message.content


# ---------------------------------------------------------------- 组装

class LegalRAG:
    def __init__(self, index: Index, gen: Generator):
        self.index = index
        self.gen = gen

    def answer(self, query: str, k: int = 5,
               as_of: Optional[str] = None) -> dict:
        hits = self.index.search(query, k=k, as_of=as_of)
        context = "\n\n".join(
            f"[{i+1}] 第{d.article_id}条（版本 {d.version_id}）\n{d.text}"
            for i, (d, _) in enumerate(hits))
        prompt = PROMPT.format(context=context, query=query)
        out = self.gen(prompt)
        return {
            "answer": out,
            "retrieved": [d.doc_id for d, _ in hits],
            "retrieved_versions": [d.version_id for d, _ in hits],
            "context": context,
        }


# ---------------------------------------------------------------- 引用校验

ART_RE = re.compile(r'第\s*([0-9零一两二三四五六七八九十]+)\s*条')


def verify_citation(answer: str, known_articles: set[str]) -> bool:
    """引用校验 D1：答案援引的条号必须存在于语料。"""
    for m in ART_RE.finditer(answer):
        aid = m.group(1)
        if aid.isdigit():
            aid = str(int(aid))
        if aid not in known_articles:
            return False
    return True
