"""Entity extraction & resolution: movie titles, recipients, ordinal choices."""
from __future__ import annotations

import re
from dataclasses import dataclass

from rapidfuzz import fuzz

from ..ingestion.metadata import MovieMeta

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
MOVIE_PHRASE_RE = re.compile(
    r"\b(?:movie|film)\s+(?:called\s+|named\s+)?[\"'“]?([A-Z0-9][\w:'’ \-]{1,50}?)[\"'”]?(?=[?.,!]|\s+(?:and|to|with|for|about|where|when|who|what)\b|$)")
MOVIE_PHRASE_RE2 = re.compile(r"\b(?:in|from|of|about)\s+(?:the\s+)?([A-Z][\w:'’\-]*(?:\s+[A-Z0-9][\w:'’\-]*){0,5})\s+(?:movie|film)\b")
ORDINALS = {"first": 0, "1st": 0, "one": 0, "second": 1, "2nd": 1, "two": 1, "third": 2, "3rd": 2,
            "three": 2, "fourth": 3, "4th": 3, "last": -1, "latter": -1, "former": 0, "original": 0}
REFERENTIAL_RE = re.compile(r"\b(that|this|the same|previous|above|it|its|that one)\b\s*(movie|film|scene|answer|quote|summary|one|breakdown)?", re.I)


def _n(s: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", s.lower()).split())


@dataclass
class MovieResolution:
    resolved: list[MovieMeta]          # confidently identified movies
    ambiguous: list[list[MovieMeta]]   # groups needing a choice
    unknown: list[str]                 # mentioned titles not in the dataset

    @property
    def needs_clarification(self) -> bool:
        return bool(self.ambiguous or self.unknown)


def extract_emails(text: str) -> list[str]:
    return list(dict.fromkeys(EMAIL_RE.findall(text)))


def resolve_movies(text: str, catalog: list[MovieMeta], llm_mentions: list[str] | None = None) -> MovieResolution:
    t = _n(EMAIL_RE.sub(" ", text))
    year_mentions = {int(y) for y in re.findall(r"\b(19\d{2}|20\d{2})\b", text)}
    resolved: dict[str, MovieMeta] = {}
    ambiguous: list[list[MovieMeta]] = []
    unknown: list[str] = []

    # 1) direct / fuzzy title hits inside the message
    hits = []
    for m in catalog:
        title = _n(m.title)
        if not title:
            continue
        if re.search(rf"\b{re.escape(title)}\b", t):
            hits.append((m, 100))
        elif len(title) >= 6 and fuzz.partial_ratio(title, t) >= 90:
            hits.append((m, 90))
    # Prefer the longest title if one hit contains another ("starfall" vs "starfall rising")
    hit_titles = {_n(m.title) for m, _ in hits}
    for m, score in hits:
        title = _n(m.title)
        if any(other != title and title in other for other in hit_titles):
            continue  # superseded by a longer explicit match
        family = [o for o in catalog if o.movie_id != m.movie_id and title in _n(o.title)]
        if family and score == 100:
            group = [m] + family
            by_year = [g for g in group if g.year in year_mentions]
            if len(by_year) == 1:
                resolved[by_year[0].movie_id] = by_year[0]
            elif re.search(r"\b(first|original|1st)\b", text, re.I):
                resolved[m.movie_id] = m
            else:
                ambiguous.append(group)
            continue
        resolved[m.movie_id] = m

    # 2) mentions from the LLM or "the movie X" phrasing that didn't match yet
    mentions = list(llm_mentions or []) + MOVIE_PHRASE_RE.findall(text) + MOVIE_PHRASE_RE2.findall(text)
    for mention in mentions:
        mn = _n(mention)
        if not mn or any(mn in _n(m.title) or _n(m.title) in mn for m in list(resolved.values())
                         + [x for g in ambiguous for x in g]):
            continue
        scored = sorted(((fuzz.WRatio(mn, _n(m.title)), m) for m in catalog), key=lambda x: -x[0])
        good = [m for s, m in scored if s >= 86]
        if len(good) == 1:
            resolved[good[0].movie_id] = good[0]
        elif len(good) > 1:
            ambiguous.append(good)
        else:
            unknown.append(mention.strip())
    return MovieResolution(list(resolved.values()), ambiguous, list(dict.fromkeys(unknown)))


def pick_from_candidates(text: str, candidates: list[MovieMeta]) -> MovieMeta | None:
    low = text.lower().strip()
    m = re.fullmatch(r"\s*#?(\d{1,2})\s*\.?\s*", low)
    if m and 1 <= int(m.group(1)) <= len(candidates):
        return candidates[int(m.group(1)) - 1]
    for y in re.findall(r"\b(19\d{2}|20\d{2})\b", low):
        ys = [c for c in candidates if c.year == int(y)]
        if len(ys) == 1:
            return ys[0]
    best = max(candidates, key=lambda c: fuzz.WRatio(_n(c.title), _n(low)))
    # exact title match wins (handles "Starfall" vs "Starfall Rising")
    exact = [c for c in candidates if _n(c.title) == _n(low)]
    if exact:
        return exact[0]
    if fuzz.WRatio(_n(best.title), _n(low)) >= 80 and _n(low):
        return best
    for word, idx in ORDINALS.items():
        if re.search(rf"\b{word}\b", low):
            return candidates[idx]
    return None
