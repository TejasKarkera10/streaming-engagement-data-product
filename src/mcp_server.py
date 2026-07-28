"""MCP server exposing this project's analytics tools over the Model
Context Protocol.

Wraps the same grounded computations used by the Streamlit app
(src.insight_agent) so any MCP-compatible client (Claude Desktop, an MCP
inspector, another agent framework) can call churn_rate_by_segment /
top_churn_drivers / completion_rate_by_genre / answer_business_question
directly, independent of this app's own Streamlit UI.

Run with: mcp dev src/mcp_server.py    (or `python -m src.mcp_server` for stdio)
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from src.insight_agent import (
    answer_business_question as _answer_business_question,
)
from src.insight_agent import (
    churn_rate_by_segment as _churn_rate_by_segment,
)
from src.insight_agent import (
    completion_rate_by_genre as _completion_rate_by_genre,
)
from src.insight_agent import (
    engagement_trend_over_time as _engagement_trend_over_time,
)
from src.insight_agent import (
    top_churn_drivers as _top_churn_drivers,
)

mcp = FastMCP("streaming-engagement-data-product")


@mcp.tool()
def churn_rate_by_segment(segment: str = "plan_tier") -> dict:
    """Real churn rate grouped by 'plan_tier' or 'region', computed with
    DuckDB over data/subscribers.csv. Returns overall rate plus per-group
    rate and subscriber count."""
    return _churn_rate_by_segment(segment)


@mcp.tool()
def top_churn_drivers(top_n: int = 5) -> dict:
    """Top churn-driving features from the trained model's precomputed SHAP
    attribution (models/metrics.json). Includes the model's ROC-AUC."""
    return _top_churn_drivers(top_n)


@mcp.tool()
def completion_rate_by_genre(top_n: int = 10) -> dict:
    """Average watch-completion rate per MovieLens genre, computed with
    DuckDB over data/sessions.csv joined to the real movie catalog."""
    return _completion_rate_by_genre(top_n)


@mcp.tool()
def engagement_trend_over_time(months: int = 6) -> dict:
    """Monthly session volume and average completion rate over the most
    recent N months, computed with DuckDB over data/sessions.csv. This is
    an engagement trend, not a churn trend - churn is a static per-subscriber
    label in this dataset, not observed per period."""
    return _engagement_trend_over_time(months)


@mcp.tool()
def answer_business_question(question: str) -> dict:
    """Answer a natural-language business question by routing it to one of
    the four supported analyses above and computing real numbers - never
    hallucinated. If the question doesn't map to a supported analysis,
    returns supported=False with an explicit explanation instead of
    guessing."""
    return _answer_business_question(question)


if __name__ == "__main__":
    mcp.run()
