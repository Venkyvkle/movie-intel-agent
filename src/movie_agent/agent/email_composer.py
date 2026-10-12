"""Turns RAG results into an email (subject + body) with a citation appendix."""
from __future__ import annotations

import re

from ..rag.answerer import RAGAnswer, format_context
from .llm import LLM

STYLE_GUIDE = {
    "scene_breakdown": "a scene breakdown: setting, characters present, beat-by-beat events with timestamps, and why the scene matters",
    "dialogue_summary": "a dialogue summary: who speaks, what is discussed, key lines quoted verbatim with timestamps",
    "quote_analysis": "a quote analysis: the exact quote, speaker, timestamp, context before/after, and its meaning/themes",
    "general": "a concise, well-organized summary of the requested information",
}

SYSTEM = """You write clear, friendly emails sharing movie insights drawn ONLY from subtitle sources.
Write {style}. Use plain text (no markdown headings), short paragraphs or "-" bullets.
Mention timestamps like (00:12:03) when referencing moments. Do not invent facts.
Do NOT include a sources list; it is appended automatically. Sign off as "Movie Intelligence Assistant".
Output format, exactly:
SUBJECT: <one-line subject>
BODY:
<email body>"""


def strip_markdown(text: str) -> str:
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"\1", text)
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.M)
    return text


def citation_appendix(answer: RAGAnswer) -> str:
    lines = ["", "Sources", "-------"]
    for c in answer.citations:
        who = f"{c.speaker}: " if c.speaker else ""
        lines.append(f"- {c.title} ({c.year}), {c.start}–{c.end} | key line {c.best_line_start}: \"{who}{c.best_line}\"")
    return "\n".join(lines)


def compose_email(llm: LLM, request: str, style: str, answer: RAGAnswer, recipient: str) -> dict:
    style = style if style in STYLE_GUIDE else "general"
    if llm.available and answer.hits:
        try:
            raw = llm.complete(
                SYSTEM.format(style=STYLE_GUIDE[style]),
                f"Recipient: {recipient}\nRequest: {request}\n\nSources:\n{format_context(answer.hits[:5])}\n\n"
                f"Grounded answer already produced:\n{answer.answer}", max_tokens=1500)
            m = re.search(r"SUBJECT:\s*(.+?)\s*\n+\s*BODY:\s*\n?(.*)", raw, re.S | re.I)
            if m and m.group(2).strip():
                subject, body = m.group(1).strip(), strip_markdown(m.group(2).strip())
                return {"subject": subject, "body": body + "\n" + citation_appendix(answer), "style": style}
        except Exception:  # noqa: BLE001 — fall back to template
            pass
    movie = answer.citations[0].title if answer.citations else "your movie question"
    title = {"scene_breakdown": "Scene breakdown", "dialogue_summary": "Dialogue summary",
             "quote_analysis": "Quote analysis"}.get(style, "Movie notes")
    body = (f"Hi,\n\nHere is the {title.lower()} you asked about ({request}).\n\n{strip_markdown(answer.answer)}\n\n"
            "Relevant dialogue:\n" + "\n\n".join(c.excerpt for c in answer.citations[:2]) +
            "\n\nBest,\nMovie Intelligence Assistant\n" + citation_appendix(answer))
    return {"subject": f"{title}: {movie}", "body": body, "style": style}
