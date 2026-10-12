"""Helpers to (re)build or load the persisted index."""
from __future__ import annotations

from pathlib import Path

from ..config import settings
from ..ingestion.pipeline import ingest_directory
from .embeddings import get_embedder
from .vector_store import SubtitleIndex


def build_index(subtitle_dir: Path | None = None, index_dir: Path | None = None) -> SubtitleIndex:
    movies, chunks = ingest_directory(subtitle_dir)
    idx = SubtitleIndex(get_embedder(settings.embedder, settings.st_model))
    idx.build(movies, chunks)
    idx.save(Path(index_dir or settings.index_dir))
    return idx


def load_or_build(index_dir: Path | None = None) -> SubtitleIndex:
    index_dir = Path(index_dir or settings.index_dir)
    if (index_dir / "meta.json").exists():
        return SubtitleIndex.load(index_dir, settings.embedder, settings.st_model)
    return build_index(index_dir=index_dir)
