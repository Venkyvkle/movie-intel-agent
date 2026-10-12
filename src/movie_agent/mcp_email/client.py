"""MCP client that launches the email server over stdio and calls its tools."""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SRC_DIR = Path(__file__).resolve().parents[2]


def _server_params() -> StdioServerParameters:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(SRC_DIR), env.get("PYTHONPATH")]))
    cmd = os.getenv("MCP_EMAIL_COMMAND")  # allow pointing at any external MCP email server
    if cmd:
        parts = cmd.split()
        return StdioServerParameters(command=parts[0], args=parts[1:], env=env)
    return StdioServerParameters(command=sys.executable, args=["-m", "movie_agent.mcp_email.server"], env=env)


def _parse(result) -> object:
    if getattr(result, "structuredContent", None):
        sc = result.structuredContent
        return sc.get("result", sc) if isinstance(sc, dict) and set(sc) == {"result"} else sc
    texts = [c.text for c in result.content if getattr(c, "type", "") == "text"]
    if len(texts) == 1:
        try:
            return json.loads(texts[0])
        except json.JSONDecodeError:
            return texts[0]
    out = []
    for t in texts:
        try:
            out.append(json.loads(t))
        except json.JSONDecodeError:
            out.append(t)
    return out


class MCPEmailClient:
    async def _call(self, tool: str, args: dict) -> object:
        async with stdio_client(_server_params()) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(tool, args)
                if result.isError:
                    return {"status": "error", "error": " ".join(getattr(c, "text", "") for c in result.content)}
                return _parse(result)

    async def _list(self) -> list[str]:
        async with stdio_client(_server_params()) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                return [t.name for t in (await session.list_tools()).tools]

    # sync facade used by the agent / web app
    def call(self, tool: str, **args) -> object:
        return asyncio.run(self._call(tool, {k: v for k, v in args.items() if v is not None}))

    def list_tools(self) -> list[str]:
        return asyncio.run(self._list())

    def send_email(self, to: str, subject: str, body: str, cc: str | None = None) -> dict:
        return self.call("send_email", to=to, subject=subject, body=body, cc=cc)

    def list_outbox(self, limit: int = 10) -> list[dict]:
        r = self.call("list_outbox", limit=limit)
        return r if isinstance(r, list) else [r]
