"""LLM-as-judge: scores an answer on four 1-5 criteria with honest failure handling.

The judge model is a LangChain chat model used through ``with_structured_output(JudgeScores)``,
so the scores come back already validated against a Pydantic schema. A judge failure (model
error, schema validation error, unexpected return) is recorded as ``None`` plus an error message,
never as 0, and aggregates exclude failures and report ``n_failed``.

The judge model is injected; ``build_judge_model`` creates the real one lazily.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field, StrictInt, ValidationError

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

Provide an integer score from 1 to 5 for each criterion."""


class JudgeScores(BaseModel):
    """Structured judge output: four strict integer scores, each 1-5."""

    relevance: StrictInt = Field(ge=MIN_SCORE, le=MAX_SCORE, description="Addresses the question")
    accuracy: StrictInt = Field(ge=MIN_SCORE, le=MAX_SCORE, description="Factual correctness")
    completeness: StrictInt = Field(ge=MIN_SCORE, le=MAX_SCORE, description="Thoroughness")
    clarity: StrictInt = Field(ge=MIN_SCORE, le=MAX_SCORE, description="Clarity and structure")


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


def _describe_error(exc: Exception) -> str:
    """Error text for a failure; names the failing criterion(s) for validation errors."""
    if isinstance(exc, ValidationError):
        parts = []
        for err in exc.errors():
            name = ".".join(str(x) for x in err["loc"]) or "response"
            parts.append(f"{name}: {err['msg']} (input={err.get('input')!r})")
        return "ValidationError: " + "; ".join(parts)
    return f"{type(exc).__name__}: {exc}"


class LLMJudge:
    """Judge an answer with an injected chat model via structured output."""

    def __init__(self, model: Any | None = None, settings: Settings | None = None) -> None:
        self.model = model if model is not None else build_judge_model(settings)

    def evaluate(self, query: str, prediction: str, reference: str) -> JudgeResult:
        prompt = JUDGE_PROMPT.format(query=query, prediction=prediction, reference=reference)
        try:
            structured = self.model.with_structured_output(JudgeScores)
            response = structured.invoke(prompt)
            if not isinstance(response, JudgeScores):
                raise TypeError(f"unexpected judge response: {type(response).__name__}")
            scores = {name: getattr(response, name) for name in CRITERIA}
        except Exception as exc:  # noqa: BLE001 - any failure must become an honest None result
            return JudgeResult(error=_describe_error(exc))
        overall = sum(scores.values()) / len(CRITERIA)
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
