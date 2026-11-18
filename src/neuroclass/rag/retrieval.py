"""Retrieval over a versioned index: lexical, dense, hybrid, filtered, reranked.

Variants (all evaluated in scripts/eval_rag.py):
  bm25      Okapi BM25 over title + section + chunk text
  dense     cosine similarity of fastembed embeddings
  hybrid    reciprocal-rank fusion (k=60) of bm25 and dense top-50
  routed    hybrid + deterministic metadata routing (questions about this
            project search project documents only) + at most 2 chunks/document
  rerank    routed candidates re-scored by a cross-encoder

Dense search goes through a `VectorStore`: `MemoryStore` (exact, NumPy) or
`PgVectorStore` (PostgreSQL + pgvector HNSW). Chunk text and metadata are held
in memory either way (7k chunks); only the vector search moves.
"""

from __future__ import annotations

import json
import math
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from ..schemas import Passage, SearchFilters

_TOKEN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
STOPWORDS = frozenset("""a an and are as at be by can do does for from has have how in is it its
of on or that the their this to was were what when which who why will with within without
about after also any been but if into more most no not only other our over so some such than
then there these they those through under up very we you your i me my""".split())

PROJECT_TERMS = re.compile(
    r"(?i)\b(neuroclass|this project|the project|deployed model|current model|the model's|"
    r"our model|ieee|icc-robins|original paper|the paper|the space|hugging face space|"
    r"parity model|previous model)\b")


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS and len(t) > 1]


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        self.n = len(docs)
        self.len = np.array([len(d) for d in docs], dtype="float64")
        self.avg = float(self.len.mean()) if self.n else 0.0
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, d in enumerate(docs):
            for term, tf in Counter(d).items():
                self.postings[term].append((i, tf))
        self.idf = {t: math.log(1 + (self.n - len(p) + 0.5) / (len(p) + 0.5))
                    for t, p in self.postings.items()}

    def scores(self, query: list[str]) -> np.ndarray:
        s = np.zeros(self.n)
        for term in set(query):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, tf in self.postings[term]:
                s[i] += idf * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
        return s


class VectorStore(Protocol):
    def search(self, query: np.ndarray, k: int, allowed: np.ndarray | None,
               filters: SearchFilters | None = None) -> list[tuple[int, float]]:
        """Top-k (row index, cosine similarity). `allowed` is the boolean row mask the
        filters produce; SQL backends apply `filters` directly instead."""


class MemoryStore:
    def __init__(self, embeddings: np.ndarray):
        self.emb = embeddings.astype("float32")

    def search(self, query, k, allowed=None, filters=None):
        sims = self.emb @ query.astype("float32")
        if allowed is not None:
            sims = np.where(allowed, sims, -np.inf)
        k = min(k, int(np.isfinite(sims).sum()))
        top = np.argpartition(-sims, k - 1)[:k] if k else np.array([], dtype=int)
        top = top[np.argsort(-sims[top])]
        return [(int(i), float(sims[i])) for i in top]


@dataclass
class Index:
    version: str
    manifest: dict
    chunks: list[dict]
    docs: dict[str, dict]
    embeddings: np.ndarray

    @classmethod
    def load(cls, path: Path) -> "Index":
        manifest = json.loads((path / "manifest.json").read_text())
        chunks = [json.loads(line) for line in (path / "chunks.jsonl").open(encoding="utf-8")]
        docs = {d["id"]: d for d in (json.loads(line) for line in
                                     (path / "documents.jsonl").open(encoding="utf-8"))}
        return cls(manifest["version"], manifest, chunks, docs, np.load(path / "embeddings.npy"))


def latest_index(root: Path, model_name: str | None = None) -> Path:
    candidates = sorted(p for p in root.glob("*") if (p / "manifest.json").exists()
                        and (model_name is None or
                             json.loads((p / "manifest.json").read_text())["embedding_model"] == model_name))
    if not candidates:
        raise FileNotFoundError(f"no retrieval index under {root}; run python -m neuroclass.rag.index build")
    return candidates[-1]


class Retriever:
    def __init__(self, index: Index, embedder=None, store: VectorStore | None = None,
                 reranker=None):
        self.index = index
        self.embedder = embedder
        self.store = store or MemoryStore(index.embeddings)
        self.reranker = reranker
        self.row_of = {c["id"]: i for i, c in enumerate(index.chunks)}
        self.doc_of_row = np.array([c["doc_id"] for c in index.chunks])
        self.source_type = np.array([index.docs[c["doc_id"]]["source_type"] for c in index.chunks])
        years = [index.docs[c["doc_id"]].get("year") or 0 for c in index.chunks]
        self.year = np.array(years)
        self.bm25 = BM25([tokenize(f"{index.docs[c['doc_id']]['title']} {c['section']} {c['text']}")
                          for c in index.chunks])

    # ------------------------------------------------------------- helpers
    def route(self, query: str, filters: SearchFilters | None) -> SearchFilters:
        """Deterministic metadata routing: project questions search project docs."""
        f = filters.model_copy() if filters else SearchFilters()
        if f.source_type is None and PROJECT_TERMS.search(query):
            f.source_type = "project_document"
        return f

    def _mask(self, filters: SearchFilters | None) -> np.ndarray | None:
        if not filters:
            return None
        mask = np.ones(len(self.index.chunks), dtype=bool)
        if filters.source_type:
            mask &= self.source_type == filters.source_type
        if filters.min_year:
            mask &= self.year >= filters.min_year
        if filters.topics:
            wanted = set(filters.topics)
            ok_docs = {d for d, doc in self.index.docs.items() if wanted & set(doc.get("topics", []))}
            mask &= np.isin(self.doc_of_row, list(ok_docs))
        return mask

    def _bm25(self, query: str, k: int, mask) -> list[tuple[int, float]]:
        s = self.bm25.scores(tokenize(query))
        if mask is not None:
            s = np.where(mask, s, -np.inf)
        top = np.argsort(-s)[:k]
        return [(int(i), float(s[i])) for i in top if np.isfinite(s[i]) and s[i] > 0]

    def _dense(self, query: str, k: int, mask, filters=None) -> list[tuple[int, float]]:
        return self.store.search(self.embedder.query(query), k, mask, filters)

    @staticmethod
    def _rrf(*rankings: list[tuple[int, float]], k: int = 60) -> list[tuple[int, float]]:
        fused: dict[int, float] = defaultdict(float)
        for ranking in rankings:
            for rank, (i, _) in enumerate(ranking):
                fused[i] += 1.0 / (k + rank + 1)
        return sorted(fused.items(), key=lambda x: -x[1])

    @staticmethod
    def _diversify(ranked: list[tuple[int, float]], doc_of_row, per_doc: int) -> list[tuple[int, float]]:
        seen: Counter = Counter()
        out = []
        for i, s in ranked:
            d = doc_of_row[i]
            if seen[d] < per_doc:
                out.append((i, s))
                seen[d] += 1
        return out

    # ---------------------------------------------------------------- main
    def search(self, query: str, k: int = 6, mode: str = "routed",
               filters: SearchFilters | None = None, candidates: int = 50) -> tuple[list[Passage], dict]:
        t0 = time.perf_counter()
        timings: dict[str, float] = {}
        if mode in ("routed", "rerank"):
            filters = self.route(query, filters)
        mask = self._mask(filters)
        if mode == "bm25":
            ranked = self._bm25(query, k, mask)
        elif mode == "dense":
            ranked = self._dense(query, k, mask, filters)
        else:
            t = time.perf_counter()
            lex = self._bm25(query, candidates, mask)
            timings["bm25_ms"] = (time.perf_counter() - t) * 1000
            t = time.perf_counter()
            den = self._dense(query, candidates, mask, filters)
            timings["dense_ms"] = (time.perf_counter() - t) * 1000
            ranked = self._rrf(lex, den)
            if mode in ("routed", "rerank"):
                ranked = self._diversify(ranked, self.doc_of_row, per_doc=2)
            if mode == "rerank" and self.reranker is not None:
                t = time.perf_counter()
                pool = ranked[:30]
                texts = [self._rerank_text(i) for i, _ in pool]
                scores = list(self.reranker.rerank(query, texts))
                ranked = sorted(zip([i for i, _ in pool], scores), key=lambda x: -x[1])
                timings["rerank_ms"] = (time.perf_counter() - t) * 1000
        passages = [self._passage(i, s) for i, s in ranked[:k]]
        timings["total_ms"] = (time.perf_counter() - t0) * 1000
        return passages, {"mode": mode, "filters": filters.model_dump() if filters else None,
                          "timings": timings, "index_version": self.index.version}

    def _rerank_text(self, i: int) -> str:
        c = self.index.chunks[i]
        return f"{self.index.docs[c['doc_id']]['title']}. {c['text']}"

    def _passage(self, i: int, score: float) -> Passage:
        from .corpus import clean_text

        c = self.index.chunks[i]
        d = self.index.docs[c["doc_id"]]
        # clean_text is idempotent; indexes built before it removed empty citation
        # brackets still display clean text.
        return Passage(chunk_id=c["id"], doc_id=c["doc_id"], title=d["title"], section=c["section"],
                       text=clean_text(c["text"]), source_url=d["source_url"], license=d["license"],
                       year=d.get("year"), journal=d.get("journal"), source_type=d["source_type"],
                       score=round(float(score), 6))

    def passage_by_id(self, chunk_id: str) -> Passage | None:
        i = self.row_of.get(chunk_id)
        return None if i is None else self._passage(i, 0.0)


class CrossEncoderReranker:
    def __init__(self, model_name: str, cache_dir: Path):
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self.model = TextCrossEncoder(model_name=model_name, cache_dir=str(cache_dir))

    def rerank(self, query: str, texts: list[str]) -> list[float]:
        return [float(s) for s in self.model.rerank(query, texts)]
