"""The Movie Intelligence agent: routing, clarification, RAG and MCP tool execution.

Routes
  question  -> RAG engine
  email     -> RAG engine + MCP email tool (send_email)
  unclear / missing slots / ambiguous entities -> clarification question (never guess)
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from rapidfuzz import fuzz

from ..config import settings
from ..ingestion.metadata import MovieMeta
from ..mcp_email.client import MCPEmailClient
from ..rag.answerer import RAGAnswer, RAGEngine
from ..retrieval.vector_store import SubtitleIndex, tokenize
from .email_composer import compose_email
from .entities import extract_emails, pick_from_candidates, resolve_movies
from .llm import LLM
from .router import IntentRouter, RouteDecision, heuristic_route

log = logging.getLogger(__name__)

HELP = ("I can answer questions about the movies in the subtitle dataset (plots, scenes, characters, "
        "who said a line, and when) with exact timestamp citations, and I can email a scene breakdown, "
        "dialogue summary, or quote analysis to someone. Try: \"Who said 'the light never went dark'?\" or "
        "\"Email a scene breakdown of the storm in The Last Lighthouse to alex@example.com\".")


@dataclass
class AgentResponse:
    kind: str                     # answer | clarification | email_sent | email_failed | smalltalk
    message: str
    citations: list[dict] = field(default_factory=list)
    trace: list[str] = field(default_factory=list)
    route: dict = field(default_factory=dict)
    email: dict | None = None
    options: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class Pending:
    intent: str
    request: str
    style: str = "general"
    movie_ids: list[str] | None = None
    recipients: list[str] = field(default_factory=list)
    missing: str = ""                       # recipient | movie | content | details
    candidates: list[MovieMeta] = field(default_factory=list)
    use_previous: bool = False
    is_quote: bool = False


@dataclass
class SessionState:
    pending: Pending | None = None
    last_movie_ids: list[str] | None = None
    last_answer: RAGAnswer | None = None
    last_request: str = ""
    history: list[tuple[str, str]] = field(default_factory=list)

    def history_text(self, n: int = 4) -> str:
        return "\n".join(f"{r}: {t[:300]}" for r, t in self.history[-n:])


class MovieAgent:
    def __init__(self, index: SubtitleIndex, llm: LLM | None = None, email_client: MCPEmailClient | None = None):
        self.index = index
        self.llm = llm or LLM()
        self.rag = RAGEngine(index, self.llm, settings.top_k)
        self.router = IntentRouter(self.llm)
        self.email = email_client or MCPEmailClient()

    @property
    def catalog(self) -> list[MovieMeta]:
        return list(self.index.movies.values())

    # ------------------------------------------------------------------ entry
    def handle(self, message: str, state: SessionState) -> AgentResponse:
        message = message.strip()
        state.history.append(("user", message))
        if state.pending:
            resp = self._resume(message, state)
            if resp is not None:
                state.history.append(("assistant", resp.message))
                return resp
        resp = self._fresh(message, state)
        state.history.append(("assistant", resp.message))
        return resp

    # ------------------------------------------------------------- new turns
    def _fresh(self, message: str, state: SessionState) -> AgentResponse:
        trace: list[str] = []
        route = self.router.route(message, [m.label for m in self.catalog], state.history_text())
        trace.append(f"router[{route.source}] -> intent={route.intent}")
        if route.intent == "smalltalk":
            return AgentResponse("smalltalk", HELP, trace=trace, route=route.to_dict())
        if route.intent == "unclear":
            return self._ask(Pending("question", message, missing="details"), state, trace, route,
                             "I'm not sure what you'd like. Are you asking about a movie (which one?), "
                             "or would you like me to email something?")

        is_email = route.intent == "email"
        request = route.content_request or ("" if is_email else message)
        use_prev = False
        if is_email and (not request or route.refers_to_previous) and state.last_answer:
            use_prev = not request or bool(re.search(r"\b(it|this|that|above|previous|same)\b", message, re.I))
        pending = Pending(intent=route.intent, request=request or state.last_request if use_prev else request,
                          style=route.email_style or "general", recipients=route.recipient_emails,
                          use_previous=use_prev, is_quote=route.is_quote_lookup)

        # ---- movie resolution (never guess between candidates)
        res = resolve_movies(message, self.catalog, route.movie_mentions)
        trace.append(f"movies resolved={[m.label for m in res.resolved]} ambiguous="
                     f"{[[m.label for m in g] for g in res.ambiguous]} unknown={res.unknown}")
        if res.unknown and not res.resolved:
            titles = ", ".join(m.label for m in self.catalog[:15])
            pending.missing = "movie"
            pending.candidates = self.catalog
            return self._ask(pending, state, trace, route,
                             f"I couldn't find \"{res.unknown[0]}\" in the subtitle dataset. Which movie did you "
                             f"mean? Available: {titles}.", options=[m.label for m in self.catalog[:15]])
        if res.ambiguous:
            group = res.ambiguous[0]
            pending.missing, pending.candidates = "movie", group
            pending.movie_ids = [m.movie_id for m in res.resolved] or None
            opts = [m.label for m in group]
            return self._ask(pending, state, trace, route,
                             "That title matches more than one movie. Which one do you mean?\n" +
                             "\n".join(f"{i}. {o}" for i, o in enumerate(opts, 1)), options=opts)
        if res.resolved:
            pending.movie_ids = [m.movie_id for m in res.resolved]
        elif route.refers_to_previous and state.last_movie_ids and not use_prev:
            pending.movie_ids = state.last_movie_ids
            trace.append("movie from conversation context")

        # ---- email slot checks
        if is_email:
            if route.recipient_names and not pending.recipients:
                pending.missing = "recipient"
                return self._ask(pending, state, trace, route,
                                 f"What is {route.recipient_names[0]}'s email address?")
            if not pending.recipients:
                pending.missing = "recipient"
                return self._ask(pending, state, trace, route, "Who should I send it to? Please share the "
                                 "recipient's email address.")
            if not pending.request and not use_prev:
                pending.missing = "content"
                return self._ask(pending, state, trace, route, "What should the email contain? For example "
                                 "a scene breakdown, a dialogue summary, or a quote analysis, and from which movie?")
        else:
            # vague informational query without anchor
            if len(tokenize(request)) < 2 and not pending.movie_ids:
                pending.missing = "details"
                return self._ask(pending, state, trace, route, "Could you add a bit more detail? Which movie, "
                                 "scene, character or line are you asking about?")
            if route.refers_to_previous and not pending.movie_ids and not state.last_movie_ids \
                    and re.search(r"\b(that|this)\s+(movie|film|scene)\b", message, re.I):
                pending.missing = "movie"
                pending.candidates = self.catalog
                return self._ask(pending, state, trace, route, "Which movie are you referring to?",
                                 options=[m.label for m in self.catalog[:15]])
        return self._execute(pending, state, trace, route)

    # --------------------------------------------------------- clarification
    def _ask(self, pending: Pending, state: SessionState, trace: list[str], route: RouteDecision | None,
             question: str, options: list[str] | None = None) -> AgentResponse:
        state.pending = pending
        trace.append(f"clarify: missing={pending.missing}")
        return AgentResponse("clarification", question, trace=trace,
                             route=route.to_dict() if route else {}, options=options or [])

    def _resume(self, message: str, state: SessionState) -> AgentResponse | None:
        p = state.pending
        trace = [f"resume pending intent={p.intent} missing={p.missing}"]
        if re.fullmatch(r"\s*(cancel|never ?mind|stop|forget it)\s*[.!]?\s*", message, re.I):
            state.pending = None
            return AgentResponse("smalltalk", "Okay, cancelled.", trace=trace)
        if p.missing == "recipient":
            emails = extract_emails(message)
            if emails:
                p.recipients = emails
                state.pending = None
                return self._continue(p, state, trace)
            if len(message.split()) <= 4:
                return self._ask(p, state, trace, None, "I still need a valid email address (e.g. name@example.com).")
            state.pending = None
            return None  # treat as a new request
        if p.missing == "movie":
            choice = pick_from_candidates(message, p.candidates)
            if choice:
                p.movie_ids = list(dict.fromkeys((p.movie_ids or []) + [choice.movie_id]))
                trace.append(f"user chose {choice.label}")
                state.pending = None
                return self._continue(p, state, trace)
            if len(message.split()) <= 4:
                return self._ask(p, state, trace, None, "Sorry, which one? Reply with the number or title.",
                                 options=[c.label for c in p.candidates])
            state.pending = None
            return None
        if p.missing in ("content", "details"):
            state.pending = None
            if p.intent == "email":
                p.request = message
                hs = heuristic_route("email " + message).email_style
                if p.style == "general" and hs:
                    p.style = hs
                res = resolve_movies(message, self.catalog)
                if res.resolved:
                    p.movie_ids = [m.movie_id for m in res.resolved]
                return self._continue(p, state, trace)
            return None  # re-route the clarified question from scratch
        state.pending = None
        return None

    def _continue(self, p: Pending, state: SessionState, trace: list[str]) -> AgentResponse:
        if p.intent == "email" and not p.recipients:
            p.missing = "recipient"
            return self._ask(p, state, trace, None, "Who should I send it to? Please share an email address.")
        if p.intent == "email" and not p.request and not p.use_previous:
            p.missing = "content"
            return self._ask(p, state, trace, None, "What should the email contain? For example a scene breakdown, "
                             "a dialogue summary, or a quote analysis, and from which movie?")
        return self._execute(p, state, trace, None)

    # ------------------------------------------------------------ execution
    def _quote_ambiguity(self, query: str, answer_hits) -> list[tuple[MovieMeta, dict]]:
        """If a quoted line appears near-verbatim in several movies, return them."""
        m = re.search(r"[\"“']([^\"”']{6,})[\"”']", query)
        phrase = (m.group(1) if m else re.sub(r"^(who|which character|where|when)\s+(said|says|say)\s+", "",
                                                   query, flags=re.I)).lower().strip(" ?.!")
        per_movie: dict[str, tuple[float, dict]] = {}
        for h in answer_hits:
            for ln in h.chunk.lines:
                s = fuzz.partial_ratio(phrase, ln["text"].lower()) if len(phrase) >= 6 else 0
                if s >= 88 and s > per_movie.get(h.chunk.movie_id, (0, {}))[0]:
                    per_movie[h.chunk.movie_id] = (s, ln)
        return [(self.index.movies[mid], ln) for mid, (s, ln) in per_movie.items()]

    def _execute(self, p: Pending, state: SessionState, trace: list[str], route: RouteDecision | None) -> AgentResponse:
        route_d = route.to_dict() if route else {}
        if p.use_previous and state.last_answer:
            answer = state.last_answer
            trace.append("reusing previous answer for email")
        else:
            hits = self.rag.retrieve(p.request, p.movie_ids, k=max(settings.top_k, 10) if p.is_quote else None)
            trace.append(f"RAG retrieve k={len(hits)} filter={p.movie_ids}")
            if p.is_quote and not p.movie_ids:
                matches = self._quote_ambiguity(p.request, hits)
                if len(matches) > 1:
                    p.missing, p.candidates = "movie", [m for m, _ in matches]
                    lines = [f"{i}. {m.label} at {ln['start']}: \"{ln['text']}\"" for i, (m, ln) in enumerate(matches, 1)]
                    return self._ask(p, state, trace, route, "That line appears in more than one movie:\n"
                                     + "\n".join(lines) + "\nWhich one do you mean?", options=[m.label for m, _ in matches])
            hits = hits[:settings.top_k]
            answer = self.rag.answer(p.request, p.movie_ids, hits=hits)
            trace.append(f"RAG answer grounded={answer.grounded} citations={len(answer.citations)}")
        state.last_answer, state.last_request = answer, p.request
        state.last_movie_ids = list(dict.fromkeys(c.movie_id for c in answer.citations)) or p.movie_ids
        cits = [c.to_dict() for c in answer.citations]

        if p.intent != "email":
            return AgentResponse("answer", answer.answer, cits, trace, route_d)

        # ---- action: compose + dispatch via MCP
        to = ", ".join(p.recipients)
        draft = compose_email(self.llm, p.request, p.style, answer, to)
        trace.append(f"compose email style={draft['style']}")
        trace.append("MCP call: send_email")
        try:
            receipt = self.email.send_email(to=to, subject=draft["subject"], body=draft["body"])
        except Exception as exc:  # noqa: BLE001
            receipt = {"status": "error", "error": f"MCP server unavailable: {exc}"}
        trace.append(f"MCP result: {receipt.get('status')}")
        email = {**draft, "to": p.recipients, "receipt": receipt}
        if receipt.get("status") in ("sent", "dry_run"):
            mode = "sent" if receipt["status"] == "sent" else "dispatched (dry-run outbox)"
            msg = f"Email {mode} to {to} with subject \"{draft['subject']}\".\n\nSummary of what I shared:\n{answer.answer}"
            return AgentResponse("email_sent", msg, cits, trace, route_d, email)
        return AgentResponse("email_failed", f"I couldn't send the email: {receipt.get('error')}", cits, trace,
                             route_d, email)
