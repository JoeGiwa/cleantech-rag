"""Brave web search tool. Never raises to the agent; returns no article IDs (empty artifact)."""

from __future__ import annotations

from typing import Any

import requests
from langchain_core.tools import BaseTool, StructuredTool

from cleantech_rag.config import Settings, get_settings

DISABLED_MESSAGE = "Web search disabled: BRAVE_API_KEY is not set."


def _format_results(results: list[dict[str, Any]]) -> str:
    blocks = []
    for i, r in enumerate((r for r in results if isinstance(r, dict)), start=1):
        blocks.append(
            f"[{i}] Title: {r.get('title') or 'N/A'}\n"
            f"URL: {r.get('url') or 'N/A'}\n"
            f"{r.get('description') or ''}"
        )
    return "\n\n".join(blocks)


def brave_search(query: str, settings: Settings | None = None) -> str:
    """Run a Brave web search and return formatted text or a clean error/disabled message."""
    settings = settings or get_settings()
    if settings.brave_api_key is None or not settings.brave_api_key.get_secret_value():
        return DISABLED_MESSAGE
    headers = {
        "Accept": "application/json",
        "X-Subscription-Token": settings.brave_api_key.get_secret_value(),
    }
    params = {"q": query, "count": settings.brave_result_count}
    try:
        resp = requests.get(
            settings.brave_endpoint,
            headers=headers,
            params=params,
            timeout=settings.brave_timeout,
        )
    except requests.Timeout:
        return f"Web search failed: request timed out after {settings.brave_timeout}s."
    except requests.RequestException as exc:
        return f"Web search failed: {type(exc).__name__}."
    if resp.status_code != 200:
        return f"Web search failed: Brave API returned HTTP {resp.status_code}."
    try:
        data = resp.json()
    except ValueError:
        return "Web search failed: invalid response from Brave API."
    web = data.get("web") if isinstance(data, dict) else None
    results = web.get("results") if isinstance(web, dict) else None
    if not isinstance(results, list):
        results = []
    results = [r for r in results if isinstance(r, dict)]
    if not results:
        return "No web results found."
    return _format_results(results[: settings.brave_result_count])


def make_web_search_tool(settings: Settings | None = None) -> BaseTool:
    """Build the `web_search` tool. Returns (content, {"article_ids": [], "documents": []})."""

    def web_search(query: str) -> tuple[str, dict[str, Any]]:
        return brave_search(query, settings), {"article_ids": [], "documents": []}

    return StructuredTool.from_function(
        func=web_search,
        name="web_search",
        description=(
            "Search the web (Brave) for current information not in the article database. "
            "Returns result titles, URLs and snippets."
        ),
        response_format="content_and_artifact",
    )
