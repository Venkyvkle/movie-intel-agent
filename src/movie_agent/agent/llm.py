"""Thin provider-agnostic LLM client (Anthropic / OpenAI) with graceful offline mode."""
from __future__ import annotations

import json
import logging
import os
import re

from ..config import settings

log = logging.getLogger(__name__)


class LLM:
    def __init__(self, provider: str | None = None):
        self.provider = self._resolve(provider or settings.llm_provider)
        self._client = None
        if self.provider == "anthropic":
            from anthropic import Anthropic
            self._client = Anthropic()
            self.model = settings.anthropic_model
        elif self.provider == "openai":
            from openai import OpenAI
            self._client = OpenAI()
            self.model = settings.openai_model
        else:
            self.model = "none"

    @staticmethod
    def _resolve(p: str) -> str:
        p = (p or "auto").lower()
        if p != "auto":
            return p
        if os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN"):
            return "anthropic"
        if os.getenv("OPENAI_API_KEY"):
            return "openai"
        return "none"

    @property
    def available(self) -> bool:
        return self.provider in ("anthropic", "openai")

    def complete(self, system: str, user: str, max_tokens: int = 900, temperature: float = 0.2) -> str:
        if not self.available:
            raise RuntimeError("No LLM configured")
        if self.provider == "anthropic":
            kwargs = dict(model=self.model, max_tokens=max_tokens, system=system,
                          messages=[{"role": "user", "content": user}])
            try:
                msg = self._client.messages.create(temperature=temperature, **kwargs)
            except TypeError:  # some gateways / newer models reject sampling params
                msg = self._client.messages.create(**kwargs)
            return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip()
        resp = self._client.chat.completions.create(
            model=self.model, temperature=temperature, max_tokens=max_tokens,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
        return resp.choices[0].message.content.strip()

    def complete_json(self, system: str, user: str, max_tokens: int = 500) -> dict:
        raw = self.complete(system + "\nRespond with a single JSON object only, no prose.", user, max_tokens, 0.0)
        m = re.search(r"\{.*\}", raw, re.S)
        if not m:
            raise ValueError(f"LLM did not return JSON: {raw[:200]}")
        return json.loads(m.group(0))
