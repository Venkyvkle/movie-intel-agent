"""Intent classification: LLM-first with a deterministic heuristic fallback."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from .entities import EMAIL_RE, REFERENTIAL_RE
from .llm import LLM

log = logging.getLogger(__name__)

INTENTS = ("question", "email", "smalltalk", "unclear")

ROUTER_SYSTEM = """You route requests for a movie-subtitle assistant. Classify the user's message.
Return JSON with keys:
  intent: "question" (wants info about movie plots, dialogue, quotes, characters, scenes),
          "email" (wants something sent/emailed/shared/forwarded to someone),
          "smalltalk" (greeting/thanks/help about the assistant), or
          "unclear" (cannot tell what they want).
  movie_mentions: list of movie titles explicitly named (verbatim; [] if none).
  recipient_emails: list of email addresses present.
  recipient_names: list of people named as recipients WITHOUT an email address (e.g. "send it to John").
  content_request: what information is wanted, rewritten as a standalone search query
                   (for email: what the email should be about; "" if not stated).
  email_style: one of "scene_breakdown","dialogue_summary","quote_analysis","general" (email only, else "").
  refers_to_previous: true if the message points at earlier context ("that scene", "this answer", "it").
  is_quote_lookup: true if the user is asking who said a line / where a line appears.
Catalog of available movies: {catalog}"""


@dataclass
class RouteDecision:
    intent: str
    movie_mentions: list[str] = field(default_factory=list)
    recipient_emails: list[str] = field(default_factory=list)
    recipient_names: list[str] = field(default_factory=list)
    content_request: str = ""
    email_style: str = "general"
    refers_to_previous: bool = False
    is_quote_lookup: bool = False
    source: str = "heuristic"

    def to_dict(self) -> dict:
        return self.__dict__.copy()


EMAIL_VERB = re.compile(r"\b(e-?mail|mail|send|forward|share|shoot|drop)\b", re.I)
IMPERATIVE = re.compile(r"^\s*(please\s+)?((can|could|would) you\s+)?(please\s+)?(e-?mail|mail|send|forward|share)\b", re.I)
SMALLTALK = re.compile(r"^\s*(hi|hello|hey|thanks|thank you|help|what can you do|who are you)\b[\s!?.]*$", re.I)
QUOTE_HINT = re.compile(r"(\"[^\"]{4,}\"|“[^”]{4,}”|'[^']{8,}'|\bwho (said|says)\b|\bthe line\b|\bquote\b|\bsays?\b.*\bwhere\b)", re.I)
NAME_RECIPIENT = re.compile(r"\b(?:to|for)\s+(?:my\s+)?([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)\b(?!@)")
STYLE_MAP = [("scene_breakdown", r"scene|breakdown|beat"), ("quote_analysis", r"quote|line|analy"),
             ("dialogue_summary", r"dialog|summar|conversation|exchange")]


def heuristic_route(message: str) -> RouteDecision:
    emails = EMAIL_RE.findall(message)
    is_email = bool(EMAIL_VERB.search(message)) and (bool(emails) or re.search(r"\b(e-?mail|mail)\b", message, re.I)
                                                     or re.search(r"\b(send|forward|share)\b.*\bto\b", message, re.I)
                                                     or IMPERATIVE.match(message))
    if SMALLTALK.match(message):
        intent = "smalltalk"
    elif is_email:
        intent = "email"
    elif len(re.findall(r"\w+", message)) == 0:
        intent = "unclear"
    else:
        intent = "question"
    names = []
    if intent == "email" and not emails:
        names = [n for n in NAME_RECIPIENT.findall(message)
                 if n.lower() not in {"the", "me", "him", "her", "them", "my"}]
    content = message
    if intent == "email":
        content = EMAIL_RE.sub("", message)
        content = re.sub(r"\b(please|can you|could you|e-?mail|mail|send|forward|share)\b", " ", content, flags=re.I)
        content = re.sub(r"\b(to|for)\s+(?:my\s+)?[A-Z][a-z]+(\s[A-Z][a-z]+)?\b", " ", content)
        content = re.sub(r"\s+", " ", content).strip(" ,.")
        content = re.sub(r"\s+(to|for)$", "", content).strip(" ,.")
        if len(re.findall(r"\w+", content)) <= 2 or re.fullmatch(r"(it|this|that|the (answer|summary))?\s*(to)?", content, re.I):
            content = ""
    style = "general"
    for name, pat in STYLE_MAP:
        if re.search(pat, message, re.I):
            style = name
            break
    return RouteDecision(
        intent=intent, recipient_emails=emails, recipient_names=names, content_request=content,
        email_style=style if intent == "email" else "",
        refers_to_previous=bool(REFERENTIAL_RE.search(message)),
        is_quote_lookup=bool(QUOTE_HINT.search(message)), source="heuristic")


class IntentRouter:
    def __init__(self, llm: LLM):
        self.llm = llm

    def route(self, message: str, catalog_titles: list[str], history: str = "") -> RouteDecision:
        heur = heuristic_route(message)
        if not self.llm.available:
            return heur
        try:
            sys = ROUTER_SYSTEM.replace("{catalog}", "; ".join(catalog_titles))
            user = (f"Recent conversation:\n{history}\n\n" if history else "") + f"Message: {message}"
            d = self.llm.complete_json(sys, user)
            intent = d.get("intent") if d.get("intent") in INTENTS else heur.intent
            emails = [e for e in d.get("recipient_emails") or [] if EMAIL_RE.fullmatch(e)]
            # never trust the model to invent addresses: keep only those literally present
            emails = [e for e in emails if e in message] or heur.recipient_emails
            return RouteDecision(
                intent=intent,
                movie_mentions=[m for m in d.get("movie_mentions") or [] if isinstance(m, str)],
                recipient_emails=emails,
                recipient_names=[n for n in d.get("recipient_names") or [] if isinstance(n, str)],
                content_request=(d.get("content_request") or "").strip(),
                email_style=d.get("email_style") or ("general" if intent == "email" else ""),
                refers_to_previous=bool(d.get("refers_to_previous")),
                is_quote_lookup=bool(d.get("is_quote_lookup")) or heur.is_quote_lookup,
                source="llm",
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("LLM routing failed, using heuristic: %s", exc)
            return heur
