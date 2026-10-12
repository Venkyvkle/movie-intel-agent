"""Pluggable text embedders: sentence-transformers, OpenAI, or TF-IDF (offline)."""
from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


def _normalize(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return (m / n).astype(np.float32)


class Embedder:
    name = "base"

    def fit(self, texts: list[str]) -> None:  # only needed for corpus-fitted models
        pass

    def embed(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    def save(self, path: Path) -> None:
        pass

    def load(self, path: Path) -> None:
        pass


class SentenceTransformerEmbedder(Embedder):
    name = "sentence-transformers"

    def __init__(self, model: str):
        from sentence_transformers import SentenceTransformer  # noqa: WPS433

        self.model_name = model
        self.model = SentenceTransformer(model)

    def embed(self, texts):
        return _normalize(np.asarray(self.model.encode(texts, batch_size=64, show_progress_bar=False)))


class OpenAIEmbedder(Embedder):
    name = "openai"

    def __init__(self, model: str = "text-embedding-3-small"):
        from openai import OpenAI

        self.client = OpenAI()
        self.model = model

    def embed(self, texts):
        out = []
        for i in range(0, len(texts), 256):
            resp = self.client.embeddings.create(model=self.model, input=texts[i:i + 256])
            out.extend(d.embedding for d in resp.data)
        return _normalize(np.asarray(out))


class TfidfEmbedder(Embedder):
    """Offline sparse->dense embedder (word + char n-grams, LSA-free for exactness)."""

    name = "tfidf"

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1,
                                   strip_accents="unicode", lowercase=True)

    def fit(self, texts):
        self.vec.fit(texts)

    def embed(self, texts):
        return _normalize(self.vec.transform(texts).toarray())

    def save(self, path: Path):
        path.write_bytes(pickle.dumps(self.vec))

    def load(self, path: Path):
        self.vec = pickle.loads(path.read_bytes())


def get_embedder(kind: str = "auto", st_model: str = "sentence-transformers/all-MiniLM-L6-v2") -> Embedder:
    kind = (kind or "auto").lower()
    if kind in ("auto", "sentence-transformers"):
        try:
            return SentenceTransformerEmbedder(st_model)
        except Exception as exc:  # noqa: BLE001
            if kind != "auto":
                raise
            log.info("sentence-transformers unavailable (%s)", exc.__class__.__name__)
    if kind == "openai" or (kind == "auto" and os.getenv("OPENAI_API_KEY") and os.getenv("USE_OPENAI_EMBEDDINGS") == "true"):
        return OpenAIEmbedder()
    return TfidfEmbedder()
