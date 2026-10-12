"""Persistent hybrid index: dense vectors (cosine) + BM25, fused with RRF."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from ..ingestion.chunker import Chunk
from ..ingestion.metadata import MovieMeta
from .embeddings import Embedder, get_embedder

TOKEN_RE = re.compile(r"[a-z0-9']+")
STOP = set("the a an and or of to in on at is are was were be it that this for with as by i you he she we they".split())


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOP]


@dataclass
class Hit:
    chunk: Chunk
    score: float
    dense: float = 0.0
    bm25: float = 0.0


class SubtitleIndex:
    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self.chunks: list[Chunk] = []
        self.movies: dict[str, MovieMeta] = {}
        self.vectors: np.ndarray | None = None
        self.bm25: BM25Okapi | None = None

    # ---------- build / persist ----------
    def build(self, movies: list[MovieMeta], chunks: list[Chunk]) -> None:
        self.movies = {m.movie_id: m for m in movies}
        self.chunks = chunks
        texts = [self._index_text(c) for c in chunks]
        self.embedder.fit(texts)
        self.vectors = self.embedder.embed(texts) if texts else np.zeros((0, 1), np.float32)
        self._build_bm25()

    @staticmethod
    def _index_text(c: Chunk) -> str:
        speakers = " ".join(c.speakers)
        return f"{c.title}. {speakers}. " + " ".join(
            f"{ln['speaker'] + ': ' if ln.get('speaker') else ''}{ln['text']}" for ln in c.lines)

    def _build_bm25(self):
        self.bm25 = BM25Okapi([tokenize(self._index_text(c)) or ["_"] for c in self.chunks]) if self.chunks else None

    def save(self, index_dir: Path) -> None:
        index_dir.mkdir(parents=True, exist_ok=True)
        (index_dir / "chunks.json").write_text(json.dumps([c.to_dict() for c in self.chunks]), encoding="utf-8")
        (index_dir / "movies.json").write_text(json.dumps([m.to_dict() for m in self.movies.values()], indent=1), encoding="utf-8")
        np.save(index_dir / "vectors.npy", self.vectors)
        (index_dir / "meta.json").write_text(json.dumps({"embedder": self.embedder.name}), encoding="utf-8")
        self.embedder.save(index_dir / "embedder.pkl")

    @classmethod
    def load(cls, index_dir: Path, embedder_kind: str = "auto", st_model: str = "") -> "SubtitleIndex":
        meta = json.loads((index_dir / "meta.json").read_text())
        emb = get_embedder(meta["embedder"] if embedder_kind == "auto" else embedder_kind, st_model or
                           "sentence-transformers/all-MiniLM-L6-v2")
        if (index_dir / "embedder.pkl").exists():
            emb.load(index_dir / "embedder.pkl")
        idx = cls(emb)
        idx.chunks = [Chunk.from_dict(d) for d in json.loads((index_dir / "chunks.json").read_text())]
        idx.movies = {d["movie_id"]: MovieMeta(**d) for d in json.loads((index_dir / "movies.json").read_text())}
        idx.vectors = np.load(index_dir / "vectors.npy")
        idx._build_bm25()
        return idx

    # ---------- search ----------
    def search(self, query: str, k: int = 6, movie_ids: list[str] | None = None, rrf_k: int = 60) -> list[Hit]:
        if not self.chunks:
            return []
        allowed = np.array([movie_ids is None or c.movie_id in movie_ids for c in self.chunks])
        qv = self.embedder.embed([query])[0]
        dense = self.vectors @ qv
        bm = self.bm25.get_scores(tokenize(query)) if self.bm25 else np.zeros(len(self.chunks))
        dense = np.where(allowed, dense, -np.inf)
        bm = np.where(allowed, bm, -np.inf)
        d_rank = {i: r for r, i in enumerate(np.argsort(-dense))}
        b_rank = {i: r for r, i in enumerate(np.argsort(-bm))}
        qnorm = _norm(query)
        fused = []
        for i in np.where(allowed)[0]:
            s = 1 / (rrf_k + d_rank[i]) + 1 / (rrf_k + b_rank[i])
            # exact-quote boost: verbatim phrase match is the strongest signal for quotes
            if len(qnorm) > 12 and qnorm in _norm(self.chunks[i].text):
                s += 0.05
            fused.append((s, i))
        fused.sort(reverse=True)
        return [Hit(self.chunks[i], float(s), float(dense[i]), float(bm[i])) for s, i in fused[:k]]

    def movie_chunks(self, movie_id: str) -> list[Chunk]:
        return [c for c in self.chunks if c.movie_id == movie_id]


def _norm(s: str) -> str:
    return " ".join(TOKEN_RE.findall(s.lower()))
