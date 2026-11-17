"""Chunking, embedding and the versioned on-disk index.

Chunks never cross a section: paragraphs of one section are packed up to
~TARGET_WORDS (long paragraphs split at sentence boundaries). Each chunk is
embedded with its document title and section heading prepended, so a passage
like "the model achieved 94%" is retrievable by what it is about.

An index version is a directory data/rag_index/<version>/ holding
chunks.jsonl, documents.jsonl, embeddings.npy (L2-normalised float32) and a
manifest with checksums. `neuroclass.rag.store` loads it into memory or into
PostgreSQL/pgvector.

Usage:
    python -m neuroclass.rag.index build [--model BAAI/bge-small-en-v1.5]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..settings import get_settings
from .corpus import DOCS_PATH

TARGET_WORDS = 220
MAX_WORDS = 320
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(])")


@dataclass
class ChunkRecord:
    id: str
    doc_id: str
    section: str
    ordinal: int
    text: str


def split_paragraph(p: str) -> list[str]:
    if len(p.split()) <= MAX_WORDS:
        return [p]
    out, buf = [], []
    for s in _SENT.split(p):
        if buf and len(" ".join(buf + [s]).split()) > TARGET_WORDS:
            out.append(" ".join(buf))
            buf = []
        buf.append(s)
    if buf:
        out.append(" ".join(buf))
    # No usable sentence boundaries (lists, run-on text): fall back to word windows.
    final = []
    for piece in out:
        words = piece.split()
        if len(words) <= MAX_WORDS:
            final.append(piece)
        else:
            final += [" ".join(words[i:i + TARGET_WORDS]) for i in range(0, len(words), TARGET_WORDS)]
    return final


def chunk_document(doc: dict) -> list[ChunkRecord]:
    chunks: list[ChunkRecord] = []
    for sec in doc["sections"]:
        buf: list[str] = []
        pieces = [x for p in sec["paragraphs"] for x in split_paragraph(p)]
        for piece in pieces:
            if buf and len(" ".join(buf + [piece]).split()) > TARGET_WORDS:
                chunks.append(ChunkRecord("", doc["id"], sec["title"], 0, " ".join(buf)))
                buf = []
            buf.append(piece)
        if buf:
            chunks.append(ChunkRecord("", doc["id"], sec["title"], 0, " ".join(buf)))
    for i, c in enumerate(chunks):
        c.ordinal = i
        c.id = f"{doc['id']}#{i}"
    return chunks


def embedding_text(doc: dict, chunk: ChunkRecord) -> str:
    return f"{doc['title']}. {chunk.section}. {chunk.text}"


def load_documents(path: Path = DOCS_PATH) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


class Embedder:
    """fastembed (ONNX, CPU) wrapper with model-specific query/passage handling."""

    def __init__(self, model_name: str, cache_dir: Path | None = None, threads: int | None = None):
        from fastembed import TextEmbedding

        self.model_name = model_name
        self.model = TextEmbedding(model_name=model_name,
                                   cache_dir=str(cache_dir or get_settings().embedding_cache_dir),
                                   threads=threads)

    def passages(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        vecs = np.asarray(list(self.model.passage_embed(texts, batch_size=batch_size)),
                          dtype="float32")
        return _normalise(vecs)

    def query(self, text: str) -> np.ndarray:
        return _normalise(np.asarray(list(self.model.query_embed(text)), dtype="float32"))[0]


def _normalise(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=1, keepdims=True).clip(min=1e-12)


def build(model_name: str, out_root: Path | None = None) -> Path:
    docs = load_documents()
    by_id = {d["id"]: d for d in docs}
    chunks = [c for d in docs for c in chunk_document(d)]
    t0 = time.time()
    emb = Embedder(model_name).passages([embedding_text(by_id[c.doc_id], c) for c in chunks])
    seconds = time.time() - t0
    corpus_sha = hashlib.sha256(DOCS_PATH.read_bytes()).hexdigest()
    version = f"{time.strftime('%Y%m%d')}-{model_name.split('/')[-1]}-{corpus_sha[:8]}"
    out = (out_root or get_settings().index_dir) / version
    out.mkdir(parents=True, exist_ok=True)
    with (out / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c.__dict__, ensure_ascii=False) + "\n")
    with (out / "documents.jsonl").open("w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps({k: v for k, v in d.items() if k != "sections"},
                               ensure_ascii=False) + "\n")
    np.save(out / "embeddings.npy", emb)
    manifest = {
        "version": version, "embedding_model": model_name, "dim": int(emb.shape[1]),
        "n_documents": len(docs), "n_chunks": len(chunks), "corpus_sha256": corpus_sha,
        "chunking": {"target_words": TARGET_WORDS, "max_words": MAX_WORDS,
                     "section_aware": True, "embedding_prefix": "title. section."},
        "embed_seconds": round(seconds, 1),
        "files": {n: hashlib.sha256((out / n).read_bytes()).hexdigest()
                  for n in ("chunks.jsonl", "documents.jsonl", "embeddings.npy")},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["build"])
    parser.add_argument("--model", default=get_settings().embedding_model)
    args = parser.parse_args()
    out = build(args.model)
    print(json.loads((out / "manifest.json").read_text()))


if __name__ == "__main__":
    main()
