"""LLM-as-judge: scores an answer on four 1-5 criteria with honest failure handling.

A judge failure (model error, malformed JSON, out-of-range score) is recorded as ``None`` plus an
error message, never as 0, and aggregates exclude failures and report ``n_failed``.

The judge model is injected: anything with ``invoke(prompt)`` returning an object with a
``content`` string (a LangChain chat model) or a plain string. ``build_judge_model`` creates the
real one lazily.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from cleantech_rag.config import Settings, get_settings

CRITERIA: tuple[str, ...] = ("relevance", "accuracy", "completeness", "clarity")
MIN_SCORE = 1
MAX_SCORE = 5

JUDGE_PROMPT = """You are an expert evaluator for question-answering systems.

Evaluate the following response on a scale of 1-5 for each criterion:

Question: {query}

System Response: {prediction}

Reference Answer: {reference}

Rate the system response on:
1. Relevance (1-5): How well does it address the question?
2. Accuracy (1-5): How factually correct is the information?
3. Completeness (1-5): How thoroughly does it answer the question?
4. Clarity (1-5): How clear and well-structured is the response?

Provide scores in JSON format:
{{
  "relevance": <score>,
  "accuracy": <score>,
  "completeness": <score>,
  "clarity": <score>
}}

Only output the JSON, no other text."""


class JudgeError(ValueError):
    """Raised internally when a judge response cannot be turned into valid scores."""


@dataclass
class JudgeResult:
    """Scores for one answer. On failure every score is ``None`` and ``error`` is set."""

    scores: dict[str, int | None] = field(default_factory=lambda: dict.fromkeys(CRITERIA))
    overall: float | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


def build_judge_model(settings: Settings | None = None) -> Any:
    """Create the real judge chat model (lazy import; never called from tests)."""
    from langchain_openai import ChatOpenAI

    settings = settings or get_settings()
    return ChatOpenAI(
        model=settings.judge_model,
        temperature=0.0,
        api_key=settings.openai_api_key,
    )


def _extract_json(text: str) -> dict[str, Any]:
    text = re.sub(r"```(?:json)?", "", text).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise JudgeError(f"malformed JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise JudgeError("judge response is not a JSON object")
    return data


def _validate_scores(data: dict[str, Any]) -> dict[str, int | None]:
    scores: dict[str, int | None] = {}
    for name in CRITERIA:
        if name not in data:
            raise JudgeError(f"missing criterion: {name}")
        value = data[name]
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise JudgeError(f"non-numeric score for {name}: {value!r}")
        if value != int(value) or not MIN_SCORE <= value <= MAX_SCORE:
            raise JudgeError(f"score for {name} out of range {MIN_SCORE}-{MAX_SCORE}: {value!r}")
        scores[name] = int(value)
    return scores


class LLMJudge:
    """Judge an answer with an injected chat model."""

    def __init__(self, model: Any | None = None, settings: Settings | None = None) -> None:
        self.model = model if model is not None else build_judge_model(settings)

    def evaluate(self, query: str, prediction: str, reference: str) -> JudgeResult:
        prompt = JUDGE_PROMPT.format(query=query, prediction=prediction, reference=reference)
        try:
            response = self.model.invoke(prompt)
            content = response if isinstance(response, str) else response.content
            if not isinstance(content, str):
                raise JudgeError("judge response content is not text")
            scores = _validate_scores(_extract_json(content))
        except Exception as exc:  # noqa: BLE001 - any failure must become an honest None result
            return JudgeResult(error=f"{type(exc).__name__}: {exc}")
        overall = sum(scores.values()) / len(CRITERIA)  # type: ignore[arg-type]
        return JudgeResult(scores=scores, overall=overall)


def aggregate_judge_results(results: Iterable[JudgeResult]) -> dict[str, Any]:
    """Mean per criterion and overall over successful results only; report ``n_failed``."""
    results = list(results)
    ok = [r for r in results if r.ok]
    summary: dict[str, Any] = {
        "n_total": len(results),
        "n_scored": len(ok),
        "n_failed": len(results) - len(ok),
    }
    for name in CRITERIA:
        vals = [r.scores[name] for r in ok if r.scores[name] is not None]
        summary[name] = sum(vals) / len(vals) if vals else None
    overalls = [r.overall for r in ok if r.overall is not None]
    summary["overall"] = sum(overalls) / len(overalls) if overalls else None
    return summary
