"""Robust .srt parser: timecodes, encodings, and dialogue cleaning."""
from __future__ import annotations

import html
import re
from dataclasses import dataclass, asdict
from pathlib import Path

TIMECODE_RE = re.compile(
    r"(?P<h1>\d{1,2}):(?P<m1>\d{1,2}):(?P<s1>\d{1,2})[,.:](?P<ms1>\d{1,3})\s*-->\s*"
    r"(?P<h2>\d{1,2}):(?P<m2>\d{1,2}):(?P<s2>\d{1,2})[,.:](?P<ms2>\d{1,3})"
)
TAG_RE = re.compile(r"<[^>]+>")                     # <i>, <font color=..>
ASS_RE = re.compile(r"\{\\[^}]*\}")                 # {\an8}
SOUND_RE = re.compile(r"[\[(][^\])]{1,60}[\])]")     # [door slams], (laughs)
MUSIC_RE = re.compile(r"[♪♫#]+")
SPEAKER_RE = re.compile(r"^\s*-?\s*([A-Z][A-Z0-9 .'\-]{1,30}):\s+(.*)$")
AD_PATTERNS = re.compile(
    r"(opensubtitles|subscene|yify|yts\.|addic7ed|subtitles? (by|downloaded|synced)|"
    r"sync(ed)? (and|&) corrected|www\.|\.com\b|support us and become vip)",
    re.I,
)


@dataclass
class Cue:
    index: int
    start_ms: int
    end_ms: int
    text: str                 # cleaned dialogue (speaker prefix removed)
    raw: str                  # original text block
    speaker: str | None = None
    sound_cues: list[str] | None = None

    @property
    def start(self) -> str:
        return ms_to_timecode(self.start_ms)

    @property
    def end(self) -> str:
        return ms_to_timecode(self.end_ms)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update(start=self.start, end=self.end)
        return d


def ms_to_timecode(ms: int, with_ms: bool = False) -> str:
    h, rem = divmod(int(ms), 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, msr = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{msr:03d}" if with_ms else f"{h:02d}:{m:02d}:{s:02d}"


def timecode_to_ms(h: str, m: str, s: str, ms: str) -> int:
    ms = (ms + "00")[:3]  # "5" -> 500, "50" -> 500
    return ((int(h) * 60 + int(m)) * 60 + int(s)) * 1000 + int(ms)


def read_text(path: Path) -> str:
    data = path.read_bytes()
    if data[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return data.decode("utf-16")
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def clean_line(line: str) -> tuple[str, list[str]]:
    """Return cleaned text plus any removed sound/hearing-impaired cues."""
    line = html.unescape(line)
    line = ASS_RE.sub("", line)
    line = TAG_RE.sub("", line)
    sounds = [s.strip("[]() ") for s in SOUND_RE.findall(line)]
    line = SOUND_RE.sub("", line)
    line = MUSIC_RE.sub("", line)
    line = re.sub(r"\s+", " ", line).strip(" -–—")
    return line.strip(), sounds


def parse_srt_text(text: str) -> list[Cue]:
    text = text.replace("\r\n", "\n").replace("\r", "\n").lstrip("\ufeff")
    blocks = re.split(r"\n\s*\n", text.strip())
    cues: list[Cue] = []
    for block in blocks:
        lines = [l for l in block.split("\n") if l.strip()]
        tc_idx = next((i for i, l in enumerate(lines) if TIMECODE_RE.search(l)), None)
        if tc_idx is None:
            continue
        m = TIMECODE_RE.search(lines[tc_idx])
        start = timecode_to_ms(m["h1"], m["m1"], m["s1"], m["ms1"])
        end = timecode_to_ms(m["h2"], m["m2"], m["s2"], m["ms2"])
        if end < start:
            start, end = end, start
        raw_lines = lines[tc_idx + 1:]
        raw = "\n".join(raw_lines)
        if not raw or AD_PATTERNS.search(raw):
            continue
        speaker = None
        parts, sounds = [], []
        for rl in raw_lines:
            cl, snd = clean_line(rl)
            sounds += snd
            sm = SPEAKER_RE.match(cl)
            if sm:
                speaker = speaker or sm.group(1).strip().title()
                cl = sm.group(2)
            if cl:
                parts.append(cl)
        cleaned = " ".join(parts).strip()
        if not cleaned:
            continue
        cues.append(Cue(index=len(cues) + 1, start_ms=start, end_ms=end, text=cleaned,
                        raw=raw, speaker=speaker, sound_cues=sounds or None))
    cues.sort(key=lambda c: c.start_ms)
    for i, c in enumerate(cues, 1):
        c.index = i
    return cues


def parse_srt_file(path: str | Path) -> list[Cue]:
    return parse_srt_text(read_text(Path(path)))
