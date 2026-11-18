"""MCP server: read-only surface, authorization, and the real stdio protocol."""

import asyncio
import json
import os
import sys

import pytest

from neuroclass import mcp_server

from .conftest import REPO, SERVING

EXPECTED = {"get_benchmarks", "get_model_info", "get_reproducibility", "search_literature", "get_analysis"}


def _text(result) -> dict:
    return json.loads(result.content[0].text)


def test_tool_surface_is_read_only():
    tools = asyncio.run(mcp_server.server.list_tools())
    assert {t.name for t in tools} == EXPECTED
    for t in tools:
        assert t.annotations.read_only_hint is True and t.annotations.destructive_hint is False
        assert t.annotations.model_dump(by_alias=True)["readOnlyHint"] is True       # on the wire
    # nothing that writes, uploads or touches files
    assert not any(w in t.name for t in tools for w in ("create", "upload", "delete", "review", "file"))


def test_benchmarks_keep_paper_and_measured_apart():
    data = _text(asyncio.run(mcp_server.server.call_tool("get_benchmarks", {})))
    assert data["paper_reported"]["status"].startswith("reported by the paper")
    assert data["measured"]["current_model"]["test_accuracy"] > 0.9


def test_analysis_access_requires_operator_configuration(monkeypatch):
    monkeypatch.delenv("NEUROCLASS_MCP_TOKEN", raising=False)
    data = _text(asyncio.run(mcp_server.server.call_tool("get_analysis", {"analysis_id": "x"})))
    assert "not configured" in data["error"]
    monkeypatch.setenv("NEUROCLASS_MCP_TOKEN", "forged.token.value")
    data = _text(asyncio.run(mcp_server.server.call_tool("get_analysis", {"analysis_id": "x"})))
    assert "invalid" in data["error"]


def test_search_rejects_unknown_source_type():
    data = asyncio.run(mcp_server.server.call_tool("search_literature",
                                                   {"query": "x", "source_type": "everything"}))
    assert "error" in data.content[0].text


@pytest.mark.skipif(not (SERVING / "manifest.json").exists(), reason="serving manifest not present")
def test_stdio_protocol_end_to_end():
    from mcp.client.session import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    async def run():
        params = StdioServerParameters(command=sys.executable, args=["-m", "neuroclass.mcp_server"],
                                       env={**os.environ, "PYTHONPATH": str(REPO / "src")})
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                names = {t.name for t in (await session.list_tools()).tools}
                info = await session.call_tool("get_model_info", {})
                return names, json.loads(info.content[0].text)

    names, info = asyncio.run(run())
    assert names == EXPECTED
    assert info["model_version"] == "2026.10-finetune"
    assert info["verification"]["argmax_disagreements_vs_tensorflow"] == 0
