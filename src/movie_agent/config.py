"""Central configuration loaded from environment variables (.env supported)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


def _path(name: str, default: str) -> Path:
    p = Path(os.getenv(name, default))
    return p if p.is_absolute() else PROJECT_ROOT / p


@dataclass
class Settings:
    subtitle_dir: Path = field(default_factory=lambda: _path("SUBTITLE_DIR", "data/subtitles"))
    index_dir: Path = field(default_factory=lambda: _path("INDEX_DIR", "index"))
    outbox_dir: Path = field(default_factory=lambda: _path("OUTBOX_DIR", "outbox"))
    max_movies: int = int(os.getenv("MAX_MOVIES", "100"))

    # chunking
    chunk_max_chars: int = int(os.getenv("CHUNK_MAX_CHARS", "700"))
    chunk_max_seconds: float = float(os.getenv("CHUNK_MAX_SECONDS", "75"))
    scene_gap_seconds: float = float(os.getenv("SCENE_GAP_SECONDS", "6"))
    chunk_overlap_cues: int = int(os.getenv("CHUNK_OVERLAP_CUES", "2"))

    # retrieval
    embedder: str = os.getenv("EMBEDDER", "auto")
    st_model: str = os.getenv("ST_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    top_k: int = int(os.getenv("TOP_K", "6"))

    # llm
    llm_provider: str = os.getenv("LLM_PROVIDER", "auto")
    anthropic_model: str = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


settings = Settings()
