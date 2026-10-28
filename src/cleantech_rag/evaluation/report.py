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


def render_markdown(summary: dict[str, Any]) -> str:
    """Markdown tables (config, counts, retrieval, response, judge) for a summary dict."""
    config, counts = _section(summary, "config"), _section(summary, "counts")
    k = config.get("k", "k")
    lines = ["## Run", ""]
    lines += [f"- {key}: {config[key]}" for key in sorted(config)]
    lines += ["", "## Counts", ""]
    lines += [f"- {key}: {counts[key]}" for key in counts]

    retrieval = _section(summary, "retrieval")
    lines += ["", f"## Retrieval (k={k})", ""]
    header = [
        "Metric",
        *(f"{label} (n={_fmt_n(_section(retrieval, v).get('n'))})" for v, label in VIEWS),
    ]
    lines += ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for key, label in RETRIEVAL_LABELS.items():
        cells = [_fmt(_section(retrieval, v).get(key)) for v, _ in VIEWS]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    response = _section(summary, "response")
    lines += ["", "## Response quality", "", "| Metric | Mean |", "|---|---|"]
    lines += [f"| {label} | {_fmt(response.get(key))} |" for key, label in RESPONSE_LABELS.items()]

    judge = _section(summary, "judge") if summary.get("judge") is not None else None
    lines += ["", "## LLM judge (1-5)", ""]
    if judge is None:
        lines.append("Judge was not run.")
    else:
        lines += ["| Criterion | Mean |", "|---|---|"]
        lines += [f"| {c.capitalize()} | {_fmt(judge.get(c), 2)} |" for c in JUDGE_CRITERIA]
        lines += [
            "",
            f"Scored {_fmt_n(judge.get('n_scored'))} of {_fmt_n(judge.get('n_total'))}; "
            f"failed {_fmt_n(judge.get('n_failed'))}, pending {_fmt_n(judge.get('n_pending'))}.",
        ]
    return "\n".join(lines) + "\n"


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _bar_chart(
    series: dict[str, dict[str, Any]], labels: dict[str, str], title: str, ylim: float, path: Path
) -> Path:
    """Grouped bars: one group per metric, one bar per series; None values are skipped."""
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    width = 0.8 / max(len(series), 1)
    for s, (name, values) in enumerate(series.items()):
        xs, ys = [], []
        for i, key in enumerate(labels):
            number = _num(values.get(key))
            if number is not None:
                xs.append(i + s * width)
                ys.append(number)
        bars = ax.bar(xs, ys, width, label=name)
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
    series = {label: _section(retrieval, view) for view, label in VIEWS if view in retrieval}
    written.append(
        _bar_chart(
            series,
            RETRIEVAL_LABELS,
            f"Retrieval metrics (k={k})",
            1.0,
            out_dir / f"{stem}_retrieval.png",
        )
    )
    written.append(
        _bar_chart(
            {"Mean": _section(summary, "response")},
            RESPONSE_LABELS,
            "Response quality",
            1.0,
            out_dir / f"{stem}_response.png",
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
            )
        )
    return written
