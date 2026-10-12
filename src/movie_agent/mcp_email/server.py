"""MCP email server (stdio transport).

Exposes tools:
  * send_email(to, subject, body, cc?, html_body?)  -> delivery receipt
  * validate_email(address)                          -> {valid, normalized}
  * list_outbox(limit?)                               -> recent dry-run messages
Delivery uses SMTP when configured; otherwise (or when EMAIL_DRY_RUN=true) the
message is rendered to a standards-compliant .eml file in the outbox directory.

Run standalone:  python -m movie_agent.mcp_email.server
Inspect:         npx @modelcontextprotocol/inspector python -m movie_agent.mcp_email.server
"""
from __future__ import annotations

import json
import os
import re
import smtplib
import ssl
import uuid
from datetime import datetime, timezone
from email.message import EmailMessage
from email.utils import formatdate, make_msgid
from pathlib import Path

from mcp.server.fastmcp import FastMCP

EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$")

mcp = FastMCP("movie-email")


def _outbox() -> Path:
    root = Path(__file__).resolve().parents[3]
    p = Path(os.getenv("OUTBOX_DIR", "outbox"))
    p = p if p.is_absolute() else root / p
    p.mkdir(parents=True, exist_ok=True)
    return p


def _dry_run() -> bool:
    return os.getenv("EMAIL_DRY_RUN", "true").lower() in {"1", "true", "yes"} or not os.getenv("SMTP_HOST") \
        or not os.getenv("SMTP_USER")


def _split(addrs: str | list[str] | None) -> list[str]:
    if not addrs:
        return []
    if isinstance(addrs, str):
        addrs = re.split(r"[,;\s]+", addrs)
    return [a.strip() for a in addrs if a and a.strip()]


@mcp.tool()
def validate_email(address: str) -> dict:
    """Check whether an email address is syntactically valid."""
    a = address.strip()
    return {"valid": bool(EMAIL_RE.match(a)), "normalized": a.lower()}


@mcp.tool()
def send_email(to: str, subject: str, body: str, cc: str | None = None, html_body: str | None = None) -> dict:
    """Send an email. `to`/`cc` accept one or more comma-separated addresses.

    Returns a receipt with status ("sent" | "dry_run" | "error"), message_id and recipients.
    """
    recipients, ccs = _split(to), _split(cc)
    bad = [a for a in recipients + ccs if not EMAIL_RE.match(a)]
    if not recipients:
        return {"status": "error", "error": "No recipient provided."}
    if bad:
        return {"status": "error", "error": f"Invalid email address(es): {', '.join(bad)}"}
    if not subject.strip() or not body.strip():
        return {"status": "error", "error": "Subject and body are required."}

    sender = os.getenv("SMTP_FROM") or os.getenv("SMTP_USER") or "movie-agent@localhost"
    msg = EmailMessage()
    msg["From"], msg["To"], msg["Subject"] = sender, ", ".join(recipients), subject
    if ccs:
        msg["Cc"] = ", ".join(ccs)
    msg["Date"] = formatdate(localtime=True)
    msg["Message-ID"] = make_msgid(domain="movie-agent.local")
    msg.set_content(body)
    if html_body:
        msg.add_alternative(html_body, subtype="html")

    if _dry_run():
        fname = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S}_{uuid.uuid4().hex[:6]}.eml"
        path = _outbox() / fname
        path.write_bytes(bytes(msg))
        return {"status": "dry_run", "message_id": msg["Message-ID"], "to": recipients, "cc": ccs,
                "subject": subject, "saved_to": str(path),
                "note": "SMTP not configured or EMAIL_DRY_RUN=true; message saved to outbox."}
    try:
        host, port = os.environ["SMTP_HOST"], int(os.getenv("SMTP_PORT", "587"))
        if port == 465:
            server = smtplib.SMTP_SSL(host, port, context=ssl.create_default_context(), timeout=30)
        else:
            server = smtplib.SMTP(host, port, timeout=30)
            if os.getenv("SMTP_STARTTLS", "true").lower() == "true":
                server.starttls(context=ssl.create_default_context())
        with server:
            server.login(os.environ["SMTP_USER"], os.environ["SMTP_PASSWORD"])
            server.send_message(msg)
        return {"status": "sent", "message_id": msg["Message-ID"], "to": recipients, "cc": ccs, "subject": subject}
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}


@mcp.tool()
def list_outbox(limit: int = 10) -> list[dict]:
    """List the most recent messages saved in the dry-run outbox."""
    from email import policy
    from email.parser import BytesParser

    out = []
    for p in sorted(_outbox().glob("*.eml"), reverse=True)[:limit]:
        m = BytesParser(policy=policy.default).parsebytes(p.read_bytes())
        body = m.get_body(preferencelist=("plain",))
        out.append({"file": p.name, "to": m["To"], "subject": m["Subject"], "date": m["Date"],
                    "body": body.get_content() if body else ""})
    return out


if __name__ == "__main__":
    mcp.run(transport="stdio")
