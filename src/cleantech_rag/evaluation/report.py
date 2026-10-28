"""Turn an evaluation summary JSON into a markdown table and PNG charts.

Needs matplotlib (the ``eval`` extra: ``uv sync --extra eval``); it is imported lazily with the
non-interactive Agg backend, so the markdown part works without it.

Missing values (``None``, e.g. a metric no query could be scored on) are shown as ``n/a`` in the
table and skipped in the charts.
"""

from __future__ import annotations

import json
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


def _fmt(value: Any, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def render_markdown(summary: dict[str, Any]) -> str:
    """Markdown tables (config, counts, retrieval, response, judge) for a summary dict."""
    config, counts = summary.get("config", {}), summary.get("counts", {})
    k = config.get("k", "k")
    lines = ["## Run", ""]
    lines += [f"- {key}: {config[key]}" for key in sorted(config)]
    lines += ["", "## Counts", ""]
    lines += [f"- {key}: {counts[key]}" for key in counts]

    retrieval = summary.get("retrieval", {})
    lines += ["", f"## Retrieval (k={k})", ""]
    header = ["Metric", *(f"{label} (n={retrieval.get(v, {}).get('n', 0)})" for v, label in VIEWS)]
    lines += ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for key, label in RETRIEVAL_LABELS.items():
        cells = [_fmt(retrieval.get(v, {}).get(key)) for v, _ in VIEWS]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    response = summary.get("response", {})
    lines += ["", "## Response quality", "", "| Metric | Mean |", "|---|---|"]
    lines += [f"| {label} | {_fmt(response.get(key))} |" for key, label in RESPONSE_LABELS.items()]

    judge = summary.get("judge")
    lines += ["", "## LLM judge (1-5)", ""]
    if judge is None:
        lines.append("Judge was not run.")
    else:
        lines += ["| Criterion | Mean |", "|---|---|"]
        lines += [f"| {c.capitalize()} | {_fmt(judge.get(c), 2)} |" for c in JUDGE_CRITERIA]
        lines += [
            "",
            f"Scored {judge.get('n_scored', 0)} of {judge.get('n_total', 0)}; "
            f"failed {judge.get('n_failed', 0)}, pending {judge.get('n_pending', 0)}.",
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
            if values.get(key) is not None:
                xs.append(i + s * width)
                ys.append(values[key])
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
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(summary_path).stem
    k = summary.get("config", {}).get("k", "k")

    md_path = out_dir / f"{stem}.md"
    md_path.write_text(render_markdown(summary), encoding="utf-8")
    written = [md_path]

    retrieval = summary.get("retrieval", {})
    series = {label: retrieval.get(view, {}) for view, label in VIEWS if view in retrieval}
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
            {"Mean": summary.get("response", {})},
            RESPONSE_LABELS,
            "Response quality",
            1.0,
            out_dir / f"{stem}_response.png",
        )
    )
    judge = summary.get("judge")
    if judge is not None:
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
