"""Brave web search tool. Never raises to the agent; returns no article IDs (empty artifact)."""

from __future__ import annotations

import logging
from typing import Any

import requests
from langchain_core.tools import BaseTool, StructuredTool

from cleantech_rag.config import Settings, get_settings

logger = logging.getLogger(__name__)

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


def _has_api_key(settings: Settings) -> bool:
    return settings.brave_api_key is not None and bool(settings.brave_api_key.get_secret_value())


def brave_search(query: str, settings: Settings | None = None) -> str:
    """Run a Brave web search and return formatted text or a clean error/disabled message."""
    return _search(query, settings or get_settings())[0]


def _search(query: str, settings: Settings) -> tuple[str, bool]:
    """(text, ok). ok is False when the search could not run or failed; "no results" is ok."""
    if not _has_api_key(settings):
        return DISABLED_MESSAGE, False
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
        return f"Web search failed: request timed out after {settings.brave_timeout}s.", False
    except requests.RequestException as exc:
        return f"Web search failed: {type(exc).__name__}.", False
    if resp.status_code != 200:
        return f"Web search failed: Brave API returned HTTP {resp.status_code}.", False
    try:
        data = resp.json()
    except ValueError:
        return "Web search failed: invalid response from Brave API.", False
    web = data.get("web") if isinstance(data, dict) else None
    results = web.get("results") if isinstance(web, dict) else None
    if not isinstance(results, list):
        results = []
    results = [r for r in results if isinstance(r, dict)]
    if not results:
        return "No web results found.", True
    return _format_results(results[: settings.brave_result_count]), True


def make_web_search_tool(settings: Settings | None = None) -> BaseTool:
    """Build the `web_search` tool. Returns (content, {article_ids, documents, ok})."""
    settings = settings or get_settings()
    if not _has_api_key(settings):
        logger.warning("BRAVE_API_KEY is not set; web_search tool is disabled")

    def web_search(query: str) -> tuple[str, dict[str, Any]]:
        text, ok = _search(query, settings)
        # "ok": False lets the agent report a failed search in tools_used.
        return text, {"article_ids": [], "documents": [], "ok": ok}

    return StructuredTool.from_function(
        func=web_search,
        name="web_search",
        description=(
            "Search the web (Brave) for current information not in the article database. "
            "Returns result titles, URLs and snippets."
        ),
        response_format="content_and_artifact",
    )
