import os

from movie_agent.mcp_email import server


def test_server_tools_dry_run(tmp_path, monkeypatch):
    monkeypatch.setenv("OUTBOX_DIR", str(tmp_path))
    monkeypatch.setenv("EMAIL_DRY_RUN", "true")
    r = server.send_email("a@example.com", "Subj", "Body")
    assert r["status"] == "dry_run" and os.path.exists(r["saved_to"])
    assert server.send_email("not-an-email", "s", "b")["status"] == "error"
    assert server.validate_email("x@y.io")["valid"]
    assert server.list_outbox(5)[0]["subject"] == "Subj"


def test_stdio_roundtrip(tmp_path, monkeypatch):
    """Spawn the real MCP server over stdio and call it through the client."""
    monkeypatch.setenv("OUTBOX_DIR", str(tmp_path))
    from movie_agent.mcp_email.client import MCPEmailClient
    c = MCPEmailClient()
    assert {"send_email", "validate_email", "list_outbox"} <= set(c.list_tools())
    r = c.send_email("b@example.com", "Hello", "World")
    assert r["status"] == "dry_run"
