"""Movie-level metadata extraction from file names and optional manifest."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path

JUNK = re.compile(
    r"\b(480p|720p|1080p|2160p|4k|bluray|blu-ray|brrip|bdrip|webrip|web-dl|web|hdrip|dvdrip|"
    r"x264|x265|h264|hevc|aac|dts|yify|yts|rarbg|eng|english|en|subs?|srt|hi|sdh|extended|"
    r"remastered|unrated|directors\.?cut|proper|repack)\b",
    re.I,
)
YEAR = re.compile(r"[\[(. _-]((?:19|20)\d{2})(?:[\])\. _-]|$)")


@dataclass
class MovieMeta:
    movie_id: str
    title: str
    year: int | None
    source_file: str
    num_cues: int = 0
    duration: str = ""
    speakers: list[str] | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def label(self) -> str:
        return f"{self.title} ({self.year})" if self.year else self.title


def slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def title_from_filename(path: Path) -> tuple[str, int | None]:
    stem = path.stem
    year = None
    ym = YEAR.search(" " + stem + " ")
    if ym:
        year = int(ym.group(1))
        stem = stem[: ym.start()] if ym.start() > 2 else stem.replace(ym.group(1), " ")
    stem = re.sub(r"[._]+", " ", stem)
    stem = JUNK.sub(" ", stem)
    stem = re.sub(r"[\[\]()\-]+", " ", stem)
    title = re.sub(r"\s+", " ", stem).strip()
    small = {"a", "an", "the", "of", "and", "in", "on", "at", "to", "for", "or"}
    words = title.split(" ")
    title = " ".join(w if (i and w.lower() in small) else (w[:1].upper() + w[1:])
                     for i, w in enumerate(words))
    return title or path.stem, year


def load_manifest(subtitle_dir: Path) -> dict[str, dict]:
    """Optional movies.json: {"file.srt": {"title": "...", "year": 1999}}."""
    mf = subtitle_dir / "movies.json"
    if mf.exists():
        return json.loads(mf.read_text(encoding="utf-8"))
    return {}
