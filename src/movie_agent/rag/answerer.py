"""Retrieval-augmented answering with enforced source citations."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..agent.llm import LLM
from ..retrieval.vector_store import Hit, SubtitleIndex
from .citations import Citation, make_citation

SYSTEM = """You are a film analyst answering questions strictly from subtitle excerpts.
Rules:
- Use ONLY the provided sources. If they do not contain the answer, say so plainly.
- After every factual sentence add the supporting source tag(s) like [S1] or [S2][S3].
- When quoting dialogue, quote it verbatim and name the speaker when known.
- Be concise: 2-6 sentences, or a short bullet list for breakdowns."""


@dataclass
class RAGAnswer:
    answer: str
    citations: list[Citation] = field(default_factory=list)
    hits: list[Hit] = field(default_factory=list)
    grounded: bool = True

    def to_dict(self) -> dict:
        return {"answer": self.answer, "citations": [c.to_dict() for c in self.citations],
                "grounded": self.grounded}


def format_context(hits: list[Hit]) -> str:
    blocks = []
    for i, h in enumerate(hits, 1):
        c = h.chunk
        blocks.append(f"[S{i}] Movie: {c.title} ({c.year}) | Time: {c.start}-{c.end}\n{c.display_text()}")
    return "\n\n".join(blocks)


class RAGEngine:
    def __init__(self, index: SubtitleIndex, llm: LLM, top_k: int = 6):
        self.index, self.llm, self.top_k = index, llm, top_k

    def retrieve(self, query: str, movie_ids: list[str] | None = None, k: int | None = None) -> list[Hit]:
        return self.index.search(query, k=k or self.top_k, movie_ids=movie_ids)

    def answer(self, query: str, movie_ids: list[str] | None = None, hits: list[Hit] | None = None) -> RAGAnswer:
        hits = hits if hits is not None else self.retrieve(query, movie_ids)
        if not hits:
            return RAGAnswer("I couldn't find anything relevant in the subtitle dataset.", grounded=False)
        all_cits = [make_citation(f"S{i}", h.chunk, query) for i, h in enumerate(hits, 1)]
        if self.llm.available:
            text = self.llm.complete(SYSTEM, f"Sources:\n{format_context(hits)}\n\nQuestion: {query}")
            used = sorted({int(n) for n in re.findall(r"\[S(\d+)\]", text) if 0 < int(n) <= len(hits)})
            cits = [all_cits[n - 1] for n in used] or all_cits[:2]
            text = self._render_inline(text, all_cits)
            return RAGAnswer(text, cits, hits, grounded=bool(used))
        return self._extractive(query, hits, all_cits)

    @staticmethod
    def _render_inline(text: str, cits: list[Citation]) -> str:
        """Replace [S#] tags with human-readable (Movie, start–end) citations."""
        def rep(m):
            n = int(m.group(1))
            if 0 < n <= len(cits):
                c = cits[n - 1]
                return f" [{c.title}, {c.start}–{c.end}]"
            return ""
        out = re.sub(r"\s?\[S(\d+)\]", rep, text)
        return re.sub(r"(\[[^\]]+\])(\s?\1)+", r"\1", out)  # collapse duplicates

    @staticmethod
    def _extractive(query: str, hits: list[Hit], cits: list[Citation]) -> RAGAnswer:
        top = cits[:3]
        lines = ["Here are the most relevant moments from the subtitles (offline mode, no LLM):"]
        for c in top:
            who = f"{c.speaker}: " if c.speaker else ""
            lines.append(f"- \"{who}{c.best_line}\" — {c.title} ({c.year}), "
                         f"{c.best_line_start}–{c.best_line_end} (scene {c.start}–{c.end})")
        return RAGAnswer("\n".join(lines), top, hits, grounded=True)
