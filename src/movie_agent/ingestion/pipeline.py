"""End-to-end ingestion: .srt files -> cues -> chunks -> persisted index."""
from __future__ import annotations

import logging
from pathlib import Path

from ..config import settings
from .chunker import Chunk, chunk_cues
from .metadata import MovieMeta, load_manifest, slugify, title_from_filename
from .srt_parser import ms_to_timecode, parse_srt_file

log = logging.getLogger(__name__)


def discover_srt_files(subtitle_dir: Path, limit: int) -> list[Path]:
    files = sorted(p for p in subtitle_dir.rglob("*") if p.suffix.lower() == ".srt")
    if len(files) > limit:
        log.warning("Found %d subtitle files; only the first %d are ingested.", len(files), limit)
    return files[:limit]


def ingest_directory(subtitle_dir: Path | None = None, limit: int | None = None
                     ) -> tuple[list[MovieMeta], list[Chunk]]:
    subtitle_dir = Path(subtitle_dir or settings.subtitle_dir)
    limit = limit or settings.max_movies
    manifest = load_manifest(subtitle_dir)
    movies: list[MovieMeta] = []
    chunks: list[Chunk] = []
    seen: set[str] = set()
    for path in discover_srt_files(subtitle_dir, limit):
        title, year = title_from_filename(path)
        override = manifest.get(path.name, {})
        title, year = override.get("title", title), override.get("year", year)
        movie_id = slugify(f"{title}-{year}" if year else title)
        if movie_id in seen:  # duplicate subtitle file for same movie
            movie_id = f"{movie_id}-{len(seen)}"
        seen.add(movie_id)
        try:
            cues = parse_srt_file(path)
        except Exception as exc:  # noqa: BLE001
            log.error("Failed to parse %s: %s", path, exc)
            continue
        if not cues:
            log.warning("No dialogue cues in %s", path)
            continue
        mchunks = chunk_cues(
            cues, movie_id, title, year,
            max_chars=settings.chunk_max_chars, max_seconds=settings.chunk_max_seconds,
            scene_gap_seconds=settings.scene_gap_seconds, overlap_cues=settings.chunk_overlap_cues,
        )
        speakers = sorted({c.speaker for c in cues if c.speaker})
        movies.append(MovieMeta(movie_id=movie_id, title=title, year=year, source_file=path.name,
                                num_cues=len(cues), duration=ms_to_timecode(cues[-1].end_ms),
                                speakers=speakers or None))
        chunks.extend(mchunks)
        log.info("Ingested %s: %d cues -> %d chunks", title, len(cues), len(mchunks))
    return movies, chunks
