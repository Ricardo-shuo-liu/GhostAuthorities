"""ChromaDB 检索后端：与 victim.Index 同接口（build / add_versions / add_docs / search）。

要点：
* 向量由我们自己的 Embedder 计算后喂给 Chroma —— 与 memory 后端同向量、可复现；
* metadata 只用 Chroma 允许的类型（str/int/float，不用 None）：
    eff_from_i / eff_to_i : YYYYMMDD 整数，便于 $lte/$gte 过滤
    has_date              : 1/0（A1 载荷没有日期元数据）
    injected / background : 1/0 标签
* as_of 过滤在 Chroma 的 where 里做数值比较；unknown_dates 策略：
    "reject"：只查有日期且覆盖该日的文档；
    "accept"：额外查一份 has_date=0 的文档，再按相似度合并。
* 每次 build 重建 collection，避免跨运行残留。
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from .victim import CHUNKERS, ArticleVersion, Doc, Embedder


def _date_i(d: Optional[str], default: int) -> int:
    if not d:
        return default
    return int(d.replace("-", ""))


class ChromaIndex:
    def __init__(self, embedder: Embedder, path: Path,
                 collection_name: str = "legalrag", batch: int = 256):
        import chromadb  # 延迟导入：不用 chroma 时不要求安装
        Path(path).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(path))
        try:
            self.client.delete_collection(collection_name)
        except Exception:
            pass
        self.col = self.client.create_collection(
            collection_name, metadata={"hnsw:space": "cosine"})
        self.embedder = embedder
        self.batch = max(int(batch), 1)
        self.docs: list[Doc] = []
        self._by_id: dict[str, Doc] = {}

    # ------------------------------------------------------------ 写入
    def build(self, versions: list[ArticleVersion], chunker: str = "fixed512") -> None:
        self.docs = []
        self._by_id = {}
        self.add_versions(versions, chunker=chunker, background=False)

    def add_versions(self, versions: list[ArticleVersion], chunker: str = "fixed512",
                     background: bool = False) -> None:
        fn = CHUNKERS[chunker]
        docs: list[Doc] = []
        for v in versions:
            for i, c in enumerate(fn(v.text)):
                meta = dict(v.meta)
                if background:
                    meta["background"] = True
                docs.append(Doc(
                    doc_id=f"{v.version_id}#{i}", text=c, version_id=v.version_id,
                    article_id=v.article_id, effective_from=v.effective_from,
                    effective_to=v.effective_to, injected=False, meta=meta))
        self.add_docs(docs)

    def add_docs(self, docs: list[Doc]) -> None:
        if not docs:
            return
        embs = self.embedder.encode([d.text for d in docs])
        for s in range(0, len(docs), self.batch):
            chunk = docs[s:s + self.batch]
            self.col.add(
                ids=[d.doc_id for d in chunk],
                embeddings=embs[s:s + self.batch],
                documents=[d.text for d in chunk],
                metadatas=[self._meta(d) for d in chunk],
            )
        for d in docs:
            self._by_id[d.doc_id] = d
        self.docs += docs

    @staticmethod
    def _meta(d: Doc) -> dict:
        return {
            "doc_id": d.doc_id,
            "version_id": d.version_id,
            "article_id": d.article_id,
            "injected": 1 if d.injected else 0,
            "background": 1 if (d.meta or {}).get("background") else 0,
            "has_date": 1 if d.effective_from is not None else 0,
            "eff_from_i": _date_i(d.effective_from, 0),
            "eff_to_i": _date_i(d.effective_to, 99999999),
        }

    # ------------------------------------------------------------ 检索
    def count(self) -> int:
        return int(self.col.count())

    def search(self, query: str, k: int = 5, as_of: Optional[str] = None,
               unknown_dates: str = "reject") -> list[tuple[Doc, float]]:
        n = self.count()
        if n == 0:
            return []
        q = self.embedder.encode([query])[0]
        fetch = min(n, max(k, 1))
        if as_of is None:
            res = self.col.query(query_embeddings=[q], n_results=fetch,
                                 include=["distances"])
            return self._hits(res)[:k]

        di = _date_i(as_of, 0)
        where = {"$and": [
            {"has_date": 1},
            {"eff_from_i": {"$lte": di}},
            {"eff_to_i": {"$gte": di}},
        ]}
        hits = self._hits(self.col.query(
            query_embeddings=[q], n_results=fetch, where=where,
            include=["distances"]))
        if unknown_dates == "accept":
            hits += self._hits(self.col.query(
                query_embeddings=[q], n_results=fetch, where={"has_date": 0},
                include=["distances"]))
            hits.sort(key=lambda x: -x[1])
        return hits[:k]

    def _hits(self, res: dict) -> list[tuple[Doc, float]]:
        ids = (res.get("ids") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        out = []
        for i, dist in zip(ids, dists):
            d = self._by_id.get(i)
            if d is not None:
                out.append((d, 1.0 - float(dist)))   # cosine 距离 -> 相似度
        return out


def make_index(backend: str, embedder: Embedder, chroma_path: Optional[Path] = None,
               collection_name: str = "legalrag"):
    """按后端名创建索引对象（chroma / memory）。"""
    if backend == "chroma":
        if chroma_path is None:
            raise RuntimeError("index_backend=chroma 需要 chroma_path")
        return ChromaIndex(embedder, chroma_path, collection_name=collection_name)
    if backend == "memory":
        from .victim import Index
        return Index(embedder)
    raise ValueError(f"未知 index_backend: {backend}")
