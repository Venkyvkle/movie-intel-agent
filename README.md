# Movie Intelligence & Follow-up Assistant

An AI agent that ingests `.srt` subtitle files for up to **100 movies**, answers questions about plots,
dialogue, characters and quotes with **precise citations (movie title + timestamp range)**, and executes
follow-up actions — sending scene breakdowns, dialogue summaries and quote analyses by email through a
**Model Context Protocol (MCP) email server**. When a request is ambiguous (unclear title, a quote that
appears in several movies, missing recipient) it **asks instead of guessing**.

```
            ┌──────────────┐   question    ┌──────────────┐
 user ───▶  │ Intent Router│ ────────────▶ │  RAG Engine  │ ──▶ answer + [Movie, 00:12:03–00:12:41]
  (Flask)   │ LLM + rules  │               └──────────────┘
            │              │   email       ┌──────────────┐   compose   ┌───────────────────────┐
            │              │ ────────────▶ │  RAG Engine  │ ──────────▶ │ MCP client ──stdio──▶ │
            │              │               └──────────────┘             │ MCP email server      │
            │              │   ambiguous / missing slot                 │ send_email (SMTP/eml) │
            │              │ ────────────▶ clarification question       └───────────────────────┘
            └──────────────┘   (pending state, resumed on next turn)
```

## Features

| Area | What it does |
|---|---|
| **Ingestion** | Robust SRT parser: BOM/CRLF/UTF‑16/cp1252, `,` or `.` millisecond separators, out‑of‑order cues, HTML/ASS tag stripping (`<i>`, `{\an8}`), hearing‑impaired cues (`[door slams]`) kept as metadata, `♪`, ad/credit lines removed, `NAME:` speaker extraction. Title + year parsed from release‑style file names (`The.Matrix.1999.1080p.BluRay.srt`) or overridden in `data/subtitles/movies.json`. |
| **Chunking** | Scene‑aware: a silence gap > 6 s starts a new scene; chunks are capped at ~700 chars / 75 s with a 2‑cue overlap so exchanges aren't cut. Each chunk keeps movie id/title/year, start/end ms, cue range, speakers, scene index and per‑line timestamps. |
| **Retrieval** | Hybrid: dense vectors (sentence‑transformers if installed, OpenAI, or offline TF‑IDF) + BM25, fused with Reciprocal Rank Fusion, verbatim‑quote boost, optional movie filter. Index persisted to `index/`. |
| **Answering** | LLM answers only from the retrieved excerpts and must tag each claim `[S#]`; tags are rewritten to `[Movie, HH:MM:SS–HH:MM:SS]`. Each citation also carries the **best‑matching line and its exact timecode**. Works offline with an extractive fallback. |
| **Agent routing** | `question → RAG`, `email → RAG + MCP send_email`, `smalltalk → help`, `unclear → clarify`. LLM classifier with a deterministic heuristic fallback; recipient addresses are only accepted if literally present in the user's text (no hallucinated emails). |
| **Ambiguity handling** | Asks when: a title matches several movies (*Starfall* vs *Starfall Rising*); a title isn't in the dataset (*Titanic*); a quote appears near‑verbatim in several movies; the recipient is missing or only a name ("send it to John"); the email content is missing; a query is too vague. The pending request is stored and resumed when the user answers (number, title, year, or an address). |
| **MCP email tool** | `src/movie_agent/mcp_email/server.py` — a standard MCP server (official Python SDK, stdio transport) exposing `send_email`, `validate_email`, `list_outbox`. Uses SMTP when configured, otherwise writes RFC‑822 `.eml` files to `outbox/` (dry‑run). Any MCP client (Claude Desktop, MCP Inspector) can use it. |
| **UI** | Flask app with chat, clickable clarification options, expandable citations with the exact dialogue excerpt, routing trace, `.srt` upload (re‑indexes), and an outbox viewer. |

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # add ANTHROPIC_API_KEY or OPENAI_API_KEY (optional)

python scripts/make_sample_data.py   # 5 original sample films (or drop your own .srt files in data/subtitles/)
python scripts/ingest.py             # build the index
python app.py                        # http://localhost:5000
# or: python scripts/cli.py          # terminal chat
```

Without an API key everything still runs (heuristic routing + extractive answers with citations).
For better semantic retrieval, `pip install sentence-transformers` — it's picked up automatically on re-ingest.

### Real email delivery

Set in `.env`:

```
EMAIL_DRY_RUN=false
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=you@gmail.com
SMTP_PASSWORD=<gmail app password>
SMTP_FROM=you@gmail.com
```

### Using the MCP server elsewhere

```bash
python -m movie_agent.mcp_email.server            # stdio MCP server (PYTHONPATH=src)
npx @modelcontextprotocol/inspector python -m movie_agent.mcp_email.server
```

Claude Desktop config:

```json
{ "mcpServers": { "movie-email": {
    "command": "python", "args": ["-m", "movie_agent.mcp_email.server"],
    "env": { "PYTHONPATH": "/path/to/movie-intel-agent/src", "EMAIL_DRY_RUN": "true" } } } }
```

Set `MCP_EMAIL_COMMAND` to point the agent at a different MCP email server.

## Example session

```
you> Who said "we're not done yet"?
[clarification] That line appears in more than one movie:
  1. Starfall Rising (2024) at 00:02:41 ...  2. Starfall (2021) at 00:02:45 ...  3. Code of Silence (2022) at 00:02:25 ...
you> the first Starfall
[answer] Reyes said it: "We're not done yet. Without that shield, we need another way home." [Starfall, 00:02:36–00:03:10]

you> Email a dialogue summary of the diner confession
[clarification] Who should I send it to? Please share the recipient's email address.
you> sam@example.com
[email_sent] Email dispatched to sam@example.com ...   (route: RAG → compose → MCP send_email)
```

## Project layout

```
app.py                         Flask API + serves web/index.html
web/index.html                 chat UI
src/movie_agent/
  config.py                    env-driven settings
  ingestion/  srt_parser.py    timecodes, encodings, cleaning, speakers
              chunker.py       scene-aware chunks with overlap + metadata
              metadata.py      title/year extraction, manifest override
              pipeline.py      directory -> movies + chunks (max 100)
  retrieval/  embeddings.py    sentence-transformers | openai | tfidf
              vector_store.py  dense + BM25 hybrid index (RRF), persistence
              builder.py       build / load index
  rag/        answerer.py      grounded answers with [S#] -> timestamp citations
              citations.py     chunk + line-level citation objects
  agent/      router.py        intent classification (LLM + heuristics)
              entities.py      movie resolution, emails, choice parsing
              agent.py         orchestrator, clarification state machine
              email_composer.py RAG result -> email (subject/body + sources)
              llm.py           Anthropic / OpenAI / offline
  mcp_email/  server.py        MCP server: send_email, validate_email, list_outbox
              client.py        MCP stdio client
scripts/  make_sample_data.py, ingest.py, cli.py
tests/    parser, chunking/retrieval, routing & ambiguity, MCP (incl. stdio round-trip)
```

## Tests

```bash
pytest -q
```

## Notes

The bundled sample films are original works written for this project, so the repo contains no copyrighted
subtitles. Point `SUBTITLE_DIR` at your own folder of up to 100 `.srt` files to use a real dataset.
