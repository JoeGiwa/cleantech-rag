"""Turn an evaluation summary JSON into a markdown table and PNG charts.

Needs matplotlib (the ``eval`` extra: ``uv sync --extra eval``); it is imported lazily with the
non-interactive Agg backend, so the markdown part works without it.

Missing values (``None``, e.g. a metric no query could be scored on) are shown as ``n/a`` in the
table and skipped in the charts.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

RETRIEVAL_LABELS = {
    "precision_at_k": "Precision@k",
    "precision_at_k_retrieved": "Precision@k (retrieved denominator)",
    "recall_at_k": "Recall@k",
    "mrr": "MRR",
    "ndcg_at_k": "NDCG@k",
}
RESPONSE_LABELS = {
    "rouge1": "ROUGE-1",
    "rouge2": "ROUGE-2",
    "rougeL": "ROUGE-L",
    "bleu": "BLEU",
    "semantic_similarity": "Semantic similarity",
}
JUDGE_CRITERIA = ("relevance", "accuracy", "completeness", "clarity", "overall")
VIEWS = (("all_queries", "All queries"), ("retrieval_only", "Retrieval-only"))


def load_summary(path: Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a summary JSON (expected an object).")
    return data


def _section(data: Any, key: str) -> dict[str, Any]:
    """``data[key]`` if it is an object, else ``{}`` (wrong-shaped sections are tolerated)."""
    value = data.get(key) if isinstance(data, dict) else None
    return value if isinstance(value, dict) else {}


def _num(value: Any) -> float | None:
    """The value as a finite float, or None for anything non-numeric (None, str, bool, NaN)."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value) if math.isfinite(value) else None


def _fmt_n(value: Any) -> str:
    number = _num(value)
    return "0" if number is None else f"{number:g}"


def _fmt(value: Any, digits: int = 3) -> str:
    number = _num(value)
    return "n/a" if number is None else f"{number:.{digits}f}"


def _ci(cis: dict[str, Any], key: str) -> tuple[float, float] | None:
    """(low, high) for a metric from a ``confidence_intervals`` section, if present and finite."""
    entry = _section(cis, key)
    low, high = _num(entry.get("ci_low")), _num(entry.get("ci_high"))
    return None if low is None or high is None else (low, high)


def _fmt_ci(value: Any, cis: dict[str, Any], key: str, digits: int = 3) -> str:
    """``mean [low, high]`` when an interval exists, else just the mean."""
    text = _fmt(value, digits)
    ci = _ci(cis, key)
    if ci is None or _num(value) is None:
        return text
    return f"{text} [{ci[0]:.{digits}f}, {ci[1]:.{digits}f}]"


def render_markdown(summary: dict[str, Any]) -> str:
    """Markdown tables (config, counts, retrieval, response, judge) for a summary dict."""
    config, counts = _section(summary, "config"), _section(summary, "counts")
    k = config.get("k", "k")
    lines = ["## Run", ""]
    lines += [f"- {key}: {config[key]}" for key in sorted(config)]
    lines += ["", "## Counts", ""]
    lines += [f"- {key}: {counts[key]}" for key in counts]

    all_cis = _section(summary, "confidence_intervals")
    ci_retrieval, ci_response = _section(all_cis, "retrieval"), _section(all_cis, "response")
    ci_judge = _section(all_cis, "judge")
    if all_cis:
        level = _num(all_cis.get("confidence")) or 0.95
        lines += [
            "",
            f"Cells show `mean [low, high]`: {level:.0%} {all_cis.get('method', 'bootstrap CI')} "
            f"({_fmt_n(all_cis.get('n_resamples'))} resamples, "
            f"seed {_fmt_n(all_cis.get('seed'))}).",
        ]

    retrieval = _section(summary, "retrieval")
    lines += ["", f"## Retrieval (k={k})", ""]
    header = [
        "Metric",
        *(f"{label} (n={_fmt_n(_section(retrieval, v).get('n'))})" for v, label in VIEWS),
    ]
    lines += ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for key, label in RETRIEVAL_LABELS.items():
        cells = [
            _fmt_ci(_section(retrieval, v).get(key), _section(ci_retrieval, v), key)
            for v, _ in VIEWS
        ]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    response = _section(summary, "response")
    lines += ["", "## Response quality", "", "| Metric | Mean |", "|---|---|"]
    lines += [
        f"| {label} | {_fmt_ci(response.get(key), ci_response, key)} |"
        for key, label in RESPONSE_LABELS.items()
    ]

    judge = _section(summary, "judge") if summary.get("judge") is not None else None
    lines += ["", "## LLM judge (1-5)", ""]
    if judge is None:
        lines.append("Judge was not run.")
    else:
        lines += ["| Criterion | Mean |", "|---|---|"]
        lines += [
            f"| {c.capitalize()} | {_fmt_ci(judge.get(c), ci_judge, c, 2)} |"
            for c in JUDGE_CRITERIA
        ]
        lines += [
            "",
            f"Scored {_fmt_n(judge.get('n_scored'))} of {_fmt_n(judge.get('n_total'))}; "
            f"failed {_fmt_n(judge.get('n_failed'))}, pending {_fmt_n(judge.get('n_pending'))}.",
        ]
    lines += _usage_lines(_section(summary, "usage"))
    return "\n".join(lines) + "\n"


def _usage_lines(usage: dict[str, Any]) -> list[str]:
    """Token and estimated-cost table; empty for summaries from before usage was recorded."""
    tokens = _section(usage, "tokens")
    if not tokens:
        return []
    cost = _section(usage, "cost_usd")
    by_model = _section(cost, "by_model")
    lines = ["", "## Token usage and estimated cost", ""]
    lines += ["| Model | Input tokens | Output tokens | Total tokens | Est. cost (USD) |"]
    lines += ["|---|---|---|---|---|"]
    for model in sorted(tokens):
        t = _section(tokens, model)
        c = _num(by_model.get(model))
        cost_s = "n/a (no price)" if c is None else f"${c:.4f}"
        lines.append(
            f"| {model} | {_fmt_n(t.get('input_tokens'))} | {_fmt_n(t.get('output_tokens'))} "
            f"| {_fmt_n(t.get('total_tokens'))} | {cost_s} |"
        )
    total = _num(cost.get("total"))
    unpriced = [m for m in sorted(tokens) if _num(by_model.get(m)) is None]
    if total is None or len(unpriced) == len(tokens):
        lines += ["", "Estimated total: n/a (no model has a price)."]
    elif unpriced:
        missing = ", ".join(
            f"{m} ({_fmt_n(_section(tokens, m).get('total_tokens'))} tokens)" for m in unpriced
        )
        lines += ["", f"Estimated total: ${total:.4f} (partial; no price for {missing})."]
    else:
        lines += ["", f"Estimated total: ${total:.4f}."]
    return lines


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _bar_chart(
    series: dict[str, dict[str, Any]],
    labels: dict[str, str],
    title: str,
    ylim: float,
    path: Path,
    cis: dict[str, dict[str, Any]] | None = None,
) -> Path:
    """Grouped bars: one group per metric, one bar per series; None values are skipped.

    ``cis`` (series name -> metric -> CI entry) adds error bars where an interval exists.
    """
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    width = 0.8 / max(len(series), 1)
    for s, (name, values) in enumerate(series.items()):
        xs, ys, lows, highs = [], [], [], []
        for i, key in enumerate(labels):
            number = _num(values.get(key))
            if number is not None:
                xs.append(i + s * width)
                ys.append(number)
                ci = _ci((cis or {}).get(name, {}), key)
                lows.append(max(number - ci[0], 0.0) if ci else 0.0)
                highs.append(max(ci[1] - number, 0.0) if ci else 0.0)
        has_err = any(lows) or any(highs)
        bars = ax.bar(xs, ys, width, label=name, yerr=[lows, highs] if has_err else None, capsize=3)
        ax.bar_label(bars, fmt="%.2f", fontsize=8)
    ax.set_xticks([i + width * (len(series) - 1) / 2 for i in range(len(labels))])
    ax.set_xticklabels(list(labels.values()), rotation=20, ha="right")
    ax.set_ylim(0, ylim)
    ax.set_title(title)
    if len(series) > 1:
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def recompute_confidence_intervals(summary_path: Path, run_log: Path | None = None) -> Path:
    """Add bootstrap CIs to an existing summary JSON from its run log (no API calls).

    Only the ``confidence_intervals`` key is written; every other value is left untouched. The
    log defaults to ``config.run_log`` next to the summary. Refuses a log whose question counts
    do not match the summary.
    """
    from cleantech_rag.evaluation.runner import confidence_intervals, load_records

    summary_path = Path(summary_path)
    summary = load_summary(summary_path)
    config = _section(summary, "config")
    if run_log is None:
        name = config.get("run_log")
        if not isinstance(name, str) or not name:
            raise ValueError("summary has no config.run_log; pass --run-log.")
        run_log = summary_path.parent / name
    run_log = Path(run_log)
    if not run_log.is_file():
        raise FileNotFoundError(f"run log {run_log} not found (it is gitignored; pass --run-log).")
    _, records = load_records(run_log)
    n_queries = _section(summary, "counts").get("n_queries")
    if n_queries is not None and n_queries != len(records):
        raise ValueError(f"run log has {len(records)} questions but the summary says {n_queries}.")
    summary["confidence_intervals"] = confidence_intervals(
        records, bool(config.get("judge_enabled", summary.get("judge") is not None))
    )
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary_path


def generate_report(summary_path: Path, out_dir: Path) -> list[Path]:
    """Write ``<stem>.md`` and PNG charts next to each other in ``out_dir``; return the paths."""
    summary = load_summary(summary_path)
    _pyplot()  # fail on a missing matplotlib before anything is written
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(summary_path).stem
    k = _section(summary, "config").get("k", "k")

    md_path = out_dir / f"{stem}.md"
    md_path.write_text(render_markdown(summary), encoding="utf-8")
    written = [md_path]

    retrieval = _section(summary, "retrieval")
    all_cis = _section(summary, "confidence_intervals")
    series = {label: _section(retrieval, view) for view, label in VIEWS if view in retrieval}
    retrieval_cis = {
        label: _section(_section(all_cis, "retrieval"), view)
        for view, label in VIEWS
        if view in retrieval
    }
    written.append(
        _bar_chart(
            series,
            RETRIEVAL_LABELS,
            f"Retrieval metrics (k={k})",
            1.0,
            out_dir / f"{stem}_retrieval.png",
            retrieval_cis,
        )
    )
    written.append(
        _bar_chart(
            {"Mean": _section(summary, "response")},
            RESPONSE_LABELS,
            "Response quality",
            1.0,
            out_dir / f"{stem}_response.png",
            {"Mean": _section(all_cis, "response")},
        )
    )
    if summary.get("judge") is not None:
        judge = _section(summary, "judge")
        written.append(
            _bar_chart(
                {"Mean": judge},
                {c: c.capitalize() for c in JUDGE_CRITERIA},
                "LLM judge (1-5)",
                5.0,
                out_dir / f"{stem}_judge.png",
                {"Mean": _section(all_cis, "judge")},
            )
        )
    return written
