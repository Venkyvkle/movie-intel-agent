"""Scene-aware chunking of subtitle cues with overlap and rich metadata."""
from __future__ import annotations

from dataclasses import dataclass, asdict, field

from .srt_parser import Cue, ms_to_timecode


@dataclass
class Chunk:
    chunk_id: str
    movie_id: str
    title: str
    year: int | None
    start_ms: int
    end_ms: int
    cue_start: int
    cue_end: int
    text: str                      # plain dialogue used for embedding / BM25
    lines: list[dict] = field(default_factory=list)  # per-cue: start, end, speaker, text
    speakers: list[str] = field(default_factory=list)
    scene_index: int = 0

    @property
    def start(self) -> str:
        return ms_to_timecode(self.start_ms)

    @property
    def end(self) -> str:
        return ms_to_timecode(self.end_ms)

    @property
    def citation(self) -> str:
        label = f"{self.title} ({self.year})" if self.year else self.title
        return f"{label}, {self.start}–{self.end}"

    def display_text(self) -> str:
        out = []
        for ln in self.lines:
            who = f"{ln['speaker']}: " if ln.get("speaker") else ""
            out.append(f"[{ln['start']}] {who}{ln['text']}")
        return "\n".join(out)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Chunk":
        return cls(**d)


def chunk_cues(
    cues: list[Cue],
    movie_id: str,
    title: str,
    year: int | None,
    max_chars: int = 700,
    max_seconds: float = 75,
    scene_gap_seconds: float = 6,
    overlap_cues: int = 2,
) -> list[Chunk]:
    """Group consecutive cues into chunks.

    A new chunk starts when the text budget or time budget is exceeded, or when a
    silence gap longer than ``scene_gap_seconds`` suggests a scene change. The last
    ``overlap_cues`` cues are carried into the next chunk (within the same scene) so
    that dialogue exchanges are not cut in half.
    """
    chunks: list[Chunk] = []
    if not cues:
        return chunks

    def emit(group: list[Cue], scene_idx: int) -> None:
        lines = [{"start": c.start, "end": c.end, "start_ms": c.start_ms, "end_ms": c.end_ms,
                  "speaker": c.speaker, "text": c.text, "cue": c.index} for c in group]
        speakers = sorted({c.speaker for c in group if c.speaker})
        chunks.append(Chunk(
            chunk_id=f"{movie_id}::{len(chunks):04d}",
            movie_id=movie_id, title=title, year=year,
            start_ms=group[0].start_ms, end_ms=group[-1].end_ms,
            cue_start=group[0].index, cue_end=group[-1].index,
            text=" ".join(c.text for c in group),
            lines=lines, speakers=speakers, scene_index=scene_idx,
        ))

    scene = 0
    group: list[Cue] = []
    for cue in cues:
        if group:
            gap = (cue.start_ms - group[-1].end_ms) / 1000
            chars = sum(len(c.text) + 1 for c in group) + len(cue.text)
            span = (cue.end_ms - group[0].start_ms) / 1000
            new_scene = gap > scene_gap_seconds
            if new_scene or chars > max_chars or span > max_seconds:
                emit(group, scene)
                if new_scene:
                    scene += 1
                    group = []
                else:
                    group = group[-overlap_cues:] if overlap_cues else []
        group.append(cue)
    if group:
        emit(group, scene)
    return chunks
