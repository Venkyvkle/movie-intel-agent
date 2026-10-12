"""Flask web app: chat UI + JSON API for the Movie Intelligence agent.

Run:  python app.py            (http://localhost:5000)
"""
from __future__ import annotations

import logging
import os
import re
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from flask import Flask, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename

from movie_agent.agent.agent import MovieAgent, SessionState
from movie_agent.agent.llm import LLM
from movie_agent.config import settings
from movie_agent.mcp_email.client import MCPEmailClient
from movie_agent.retrieval.builder import build_index, load_or_build

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
WEB_DIR = Path(__file__).parent / "web"

app = Flask(__name__, static_folder=None)
_lock = threading.Lock()
_sessions: dict[str, SessionState] = {}
_llm = LLM()
_email = MCPEmailClient()
agent = MovieAgent(load_or_build(), _llm, _email)


@app.after_request
def cors(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    resp.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return resp


def _session() -> tuple[str, SessionState]:
    body = request.get_json(silent=True) or {}
    sid = f"{request.headers.get('X-Visitor-Id', 'local')}:{body.get('session_id') or request.args.get('session_id', 'default')}"
    return sid, _sessions.setdefault(sid, SessionState())


@app.get("/")
def index():
    return send_from_directory(WEB_DIR, "index.html")


@app.get("/api/health")
def health():
    idx = agent.index
    email_mode = "dry_run" if (os.getenv("EMAIL_DRY_RUN", "true").lower() == "true" or not os.getenv("SMTP_HOST")) else "smtp"
    return jsonify(movies=len(idx.movies), chunks=len(idx.chunks), embedder=idx.embedder.name,
                   llm=_llm.provider, llm_model=_llm.model, email_mode=email_mode,
                   max_movies=settings.max_movies)


@app.get("/api/movies")
def movies():
    return jsonify(sorted((m.to_dict() | {"label": m.label} for m in agent.index.movies.values()),
                          key=lambda d: d["title"]))


@app.route("/api/chat", methods=["POST", "OPTIONS"])
def chat():
    if request.method == "OPTIONS":
        return "", 204
    msg = ((request.get_json(silent=True) or {}).get("message") or "").strip()
    if not msg:
        return jsonify(error="empty message"), 400
    _, state = _session()
    try:
        resp = agent.handle(msg, state)
    except Exception as exc:  # noqa: BLE001
        logging.exception("agent failure")
        return jsonify(kind="error", message=f"Something went wrong: {exc}", citations=[], trace=[]), 500
    return jsonify(resp.to_dict())


@app.route("/api/reset", methods=["POST", "OPTIONS"])
def reset():
    if request.method == "OPTIONS":
        return "", 204
    sid, _ = _session()
    _sessions[sid] = SessionState()
    return jsonify(ok=True)


@app.route("/api/upload", methods=["POST", "OPTIONS"])
def upload():
    """Upload .srt files then rebuild the index."""
    if request.method == "OPTIONS":
        return "", 204
    files = request.files.getlist("files")
    existing = len(list(settings.subtitle_dir.glob("*.srt")))
    saved, skipped = [], []
    for f in files:
        name = secure_filename(f.filename or "")
        if not name.lower().endswith(".srt"):
            skipped.append(f.filename)
            continue
        if existing + len(saved) >= settings.max_movies:
            skipped.append(f"{name} (limit {settings.max_movies})")
            continue
        data = f.read()
        if len(data) > 5_000_000:
            skipped.append(f"{name} (too large)")
            continue
        (settings.subtitle_dir / name).write_bytes(data)
        saved.append(name)
    if saved:
        _rebuild()
    return jsonify(saved=saved, skipped=skipped, movies=len(agent.index.movies))


@app.route("/api/reindex", methods=["POST", "OPTIONS"])
def reindex():
    if request.method == "OPTIONS":
        return "", 204
    _rebuild()
    return jsonify(movies=len(agent.index.movies), chunks=len(agent.index.chunks))


def _rebuild():
    with _lock:
        agent.index = build_index()
        agent.rag.index = agent.index


@app.get("/api/outbox")
def outbox():
    try:
        return jsonify(_email.list_outbox(limit=int(request.args.get("limit", 10))))
    except Exception as exc:  # noqa: BLE001
        return jsonify(error=str(exc)), 500


@app.get("/api/tools")
def tools():
    return jsonify(_email.list_tools())


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), threaded=True)
