"""Citation objects with chunk-level and line-level (precise) timestamps."""
from __future__ import annotations

from dataclasses import dataclass, asdict

from rapidfuzz import fuzz

from ..ingestion.chunker import Chunk


@dataclass
class Citation:
    ref: str              # "S1"
    movie_id: str
    title: str
    year: int | None
    start: str            # chunk range
    end: str
    best_line: str        # most relevant line within the chunk
    best_line_start: str  # precise timecode of that line
    best_line_end: str
    speaker: str | None
    excerpt: str          # timestamped dialogue for display

    @property
    def label(self) -> str:
        t = f"{self.title} ({self.year})" if self.year else self.title
        return f"{t}, {self.start}–{self.end}"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["label"] = self.label
        return d


def best_line(chunk: Chunk, query: str) -> dict:
    q = query.lower()
    return max(chunk.lines, key=lambda ln: (fuzz.partial_ratio(q, ln["text"].lower()) * 0.6
                                             + fuzz.token_set_ratio(q, ln["text"].lower()) * 0.4))


def make_citation(ref: str, chunk: Chunk, query: str) -> Citation:
    bl = best_line(chunk, query)
    return Citation(ref=ref, movie_id=chunk.movie_id, title=chunk.title, year=chunk.year,
                    start=chunk.start, end=chunk.end, best_line=bl["text"],
                    best_line_start=bl["start"], best_line_end=bl["end"], speaker=bl.get("speaker"),
                    excerpt=chunk.display_text())
