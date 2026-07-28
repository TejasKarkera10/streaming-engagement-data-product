"""Tests that the MCP server exposes the expected tools and that calling
them returns real, grounded data (not stub/mock output)."""

from __future__ import annotations

import asyncio
import json

import src.mcp_server as mcp_server

EXPECTED_TOOLS = {
    "churn_rate_by_segment",
    "top_churn_drivers",
    "completion_rate_by_genre",
    "engagement_trend_over_time",
    "answer_business_question",
}


def _call(tool_name: str, args: dict) -> dict:
    async def run():
        result = await mcp_server.mcp.call_tool(tool_name, args)
        return result

    result = asyncio.run(run())
    # FastMCP returns a list of content blocks; tools returning dict/list are
    # serialized as JSON text content.
    text = result[0].text if isinstance(result, list) else result
    return json.loads(text)


def test_all_expected_tools_registered():
    async def run():
        return await mcp_server.mcp.list_tools()

    tools = asyncio.run(run())
    names = {t.name for t in tools}
    assert EXPECTED_TOOLS <= names


def test_churn_rate_tool_returns_real_data():
    payload = _call("churn_rate_by_segment", {"segment": "plan_tier"})
    assert 0.0 <= payload["overall_churn_rate"] <= 1.0
    assert len(payload["by_group"]) == 3


def test_top_churn_drivers_tool_returns_real_data():
    payload = _call("top_churn_drivers", {"top_n": 3})
    assert len(payload["drivers"]) == 3
    assert payload["model_roc_auc"] > 0.6


def test_answer_business_question_tool_is_grounded():
    payload = _call("answer_business_question", {"question": "Which genre has the highest completion rate?"})
    assert payload["supported"] is True
    assert payload["grounded_data"]["by_genre"]
