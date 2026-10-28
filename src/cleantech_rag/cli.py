"""Command line interface: ``ask``, ``chat``, ``build-index`` and ``evaluate`` commands."""

from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal

import typer

from cleantech_rag.agent import create_agent
from cleantech_rag.config import Settings, get_settings
from cleantech_rag.data import chunk_documents, load_articles, to_documents
from cleantech_rag.evaluation.datasets import (
    load_complex_qa,
    load_simple_qa,
    load_url_index,
    map_article_ids,
)
from cleantech_rag.evaluation.metrics import missing_eval_extras
from cleantech_rag.evaluation.runner import (
    EvalItem,
    ResumeMismatchError,
    RunLog,
    evaluate_items,
    summarize,
    write_outputs,
)
from cleantech_rag.vectorstore import build_index, estimate_cost, index_exists, load_index

app = typer.Typer(help="CleanTech agentic RAG.", no_args_is_help=True, add_completion=False)

NOISY_LOGGERS = (
    "httpx",
    "httpcore",
    "openai",
    "urllib3",
    "huggingface_hub",
    "sentence_transformers",
    "transformers",
    "filelock",
)


def configure_logging(verbose: bool = False) -> None:
    """INFO messages always; third-party HTTP/download chatter only with ``--verbose``."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    level = logging.INFO if verbose else logging.WARNING
    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(level)
    if not verbose:
        # Read when huggingface_hub is imported (lazily, on first use of the encoder).
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")


@app.callback()
def _root(
    verbose: bool = typer.Option(
        False,
        "--verbose",
        help="Show HTTP request and model-download logs (put before the command).",
    ),
) -> None:
    configure_logging(verbose)


CORPUS_FILE = "cleantech_media_dataset_v3_2024-10-28.csv"
SIMPLE_QA_FILE = "cleantech_rag_evaluation_data_2024-09-20.csv"
COMPLEX_QA_FILE = "CleanTech-50answers.txt"
CHARS_PER_TOKEN = 4  # documented heuristic (English text); avoids tiktoken's network download


def _fail(message: str, code: int = 1) -> typer.Exit:
    typer.secho(f"Error: {message}", fg=typer.colors.RED, err=True)
    return typer.Exit(code)


def _require_key(settings: Settings, name: str) -> None:
    secret = getattr(settings, name)
    if secret is None or not secret.get_secret_value().strip():
        env = name.upper()
        raise _fail(f"{env} is not set. Put it in your dotenv file or export it in the shell.")


def _make_llm(settings: Settings):
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=settings.agent_model,
        temperature=settings.temperature,
        api_key=settings.openai_api_key,
    )


def gold_article_ids(raw_dir: Path) -> list[str]:
    """Gold article IDs (str) from the complex and simple eval sets; missing files are skipped."""
    ids: set[str] = set()
    complex_path = raw_dir / COMPLEX_QA_FILE
    if complex_path.exists():
        ids.update(i for q in load_complex_qa(complex_path) for i in q.article_ids)
    simple_path = raw_dir / SIMPLE_QA_FILE
    corpus_path = raw_dir / CORPUS_FILE
    if simple_path.exists() and corpus_path.exists():
        items, _ = map_article_ids(load_simple_qa(simple_path), load_url_index(corpus_path))
        ids.update(i for q in items for i in q.article_ids)
    return sorted(ids)


def estimate_full_corpus(df, settings: Settings) -> tuple[int, float | None]:
    """Offline estimate: chars / 4 tokens, inflated by the chunk overlap ratio."""
    chars = int(df["content"].astype(str).str.len().sum() + df["title"].astype(str).str.len().sum())
    overlap_factor = 1 + settings.chunk_overlap / max(settings.chunk_size, 1)
    tokens = int(chars / CHARS_PER_TOKEN * overlap_factor)
    return tokens, estimate_cost(tokens, settings.embedding_model)


NO_INDEX_MESSAGE = "No index found. Run build-index --limit 200 first."


def _setup_agent(settings: Settings):
    """Shared by ask/chat: index check, key check, Brave warning, then build the agent."""
    _require_key(settings, "openai_api_key")
    if not index_exists(settings):
        raise _fail(NO_INDEX_MESSAGE)
    if settings.brave_api_key is None:
        typer.secho(
            "Warning: BRAVE_API_KEY is not set; web search will be unavailable.",
            fg=typer.colors.YELLOW,
            err=True,
        )
    store = load_index(settings)
    return create_agent(store, _make_llm(settings), settings)


def _echo_sources(result) -> None:
    ids = list(getattr(result, "retrieved_ids", None) or [])
    if not ids:
        typer.echo("Sources: no articles retrieved this turn.")
        return
    by_id = {s["id"]: s for s in (getattr(result, "sources", None) or [])}
    typer.echo("Sources:")
    for n, article_id in enumerate(ids, start=1):
        src = by_id.get(article_id, {})
        line = f"  [{n}] {src.get('title') or '(untitled)'} (ID {article_id})"
        if src.get("url"):
            line += f"\n      {src['url']}"
        typer.echo(line)


@app.command()
def ask(question: str = typer.Argument(..., help="Question to ask the agent.")) -> None:
    """Answer one question (single-shot, no memory); use `chat` for a conversation."""
    agent = _setup_agent(get_settings())
    result = agent.run(question)
    typer.echo(result.answer)
    _echo_sources(result)


@app.command()
def chat() -> None:
    """Interactive chat with memory across turns. Type 'exit' (or Ctrl-D) to quit."""
    agent = _setup_agent(get_settings())
    thread_id = uuid.uuid4().hex
    typer.echo("Chat started. Type 'exit' to quit.")
    while True:
        try:
            line = input("You> ")
        except (EOFError, KeyboardInterrupt):
            typer.echo("")
            break
        question = line.strip()
        if not question:
            continue
        if question.lower() == "exit":
            break
        result = agent.run(question, thread_id=thread_id)
        typer.echo(result.answer)
        _echo_sources(result)


@app.command("build-index")
def build_index_cmd(
    limit: int | None = typer.Option(
        None, "--limit", min=1, help="Index N sampled articles plus all eval gold articles."
    ),
    yes_full_corpus: bool = typer.Option(
        False, "--yes-full-corpus", help="Confirm embedding the entire corpus (costs money)."
    ),
) -> None:
    """Chunk the corpus and embed it into the Chroma index."""
    settings = get_settings()
    corpus_path = settings.raw_data_dir / CORPUS_FILE
    if not corpus_path.exists():
        raise _fail(f"Corpus CSV not found at {corpus_path}.")

    if limit is None:
        df = load_articles(corpus_path)
        tokens, cost = estimate_full_corpus(df, settings)
        cost_s = f"~${cost:.2f}" if cost is not None else "unknown (no price for model)"
        typer.echo(
            f"Full corpus: {len(df)} articles, ~{tokens:,} tokens, "
            f"estimated embedding cost {cost_s} ({settings.embedding_model})."
        )
        if not yes_full_corpus:
            raise _fail("refusing to embed the full corpus; pass --yes-full-corpus to confirm.", 2)
    else:
        gold = gold_article_ids(settings.raw_data_dir)
        df = load_articles(corpus_path, limit=limit, include_ids=gold)
        typer.echo(f"Indexing {len(df)} articles ({limit} sampled + eval gold articles).")

    _require_key(settings, "openai_api_key")
    chunks = chunk_documents(to_documents(df), settings.chunk_size, settings.chunk_overlap)
    build_index(chunks, settings)
    typer.echo(f"Indexed {len(chunks)} chunks into {settings.chroma_dir}.")


DEFAULT_MAX_QUERIES = 3
MAX_QUERIES_WITHOUT_CONFIRM = 10  # larger runs need --yes-full-eval


def load_eval_items(raw_dir: Path, dataset: str) -> list[EvalItem]:
    """Load the chosen eval set as EvalItems (simple QA ids are mapped via the corpus URLs)."""
    if dataset == "complex":
        path = raw_dir / COMPLEX_QA_FILE
        if not path.exists():
            raise _fail(f"Eval file not found at {path}.")
        return [EvalItem(q.question, q.answer, q.article_ids) for q in load_complex_qa(path)]
    simple_path, corpus_path = raw_dir / SIMPLE_QA_FILE, raw_dir / CORPUS_FILE
    for p in (simple_path, corpus_path):
        if not p.exists():
            raise _fail(f"Eval file not found at {p}.")
    mapped, _ = map_article_ids(load_simple_qa(simple_path), load_url_index(corpus_path))
    return [EvalItem(q.question, q.answer, q.article_ids) for q in mapped]


@app.command()
def evaluate(
    dataset: Literal["complex", "simple"] = typer.Option("complex", "--dataset"),
    max_queries: int | None = typer.Option(
        None,
        "--max-queries",
        min=1,
        help=(
            f"Queries to run (default {DEFAULT_MAX_QUERIES}; more than "
            f"{MAX_QUERIES_WITHOUT_CONFIRM} needs --yes-full-eval, which alone runs all)."
        ),
    ),
    no_judge: bool = typer.Option(False, "--no-judge", help="Skip the LLM judge."),
    judge_model: str | None = typer.Option(None, "--judge-model", help="Override judge model."),
    yes_full_eval: bool = typer.Option(
        False, "--yes-full-eval", help="Confirm a full run over every query (costs money)."
    ),
    output_dir: Annotated[Path, typer.Option("--output-dir")] = Path("results"),
    resume: Annotated[
        Path | None,
        typer.Option("--resume", help="Continue a run log (.jsonl): skip completed questions."),
    ] = None,
) -> None:
    """Evaluate the agent on a dataset; writes a summary JSON and a per-query CSV."""
    if max_queries is None and not yes_full_eval:
        max_queries = DEFAULT_MAX_QUERIES
    if max_queries is not None and max_queries > MAX_QUERIES_WITHOUT_CONFIRM and not yes_full_eval:
        raise _fail(
            f"--max-queries {max_queries} exceeds {MAX_QUERIES_WITHOUT_CONFIRM}; "
            "pass --yes-full-eval to confirm (this costs money).",
            2,
        )

    missing = missing_eval_extras()
    if missing:  # before any paid call
        raise _fail(
            f"evaluation needs {', '.join(missing)}. Install the extras: `uv sync --extra eval`."
        )

    settings = get_settings()
    if judge_model:
        settings = settings.model_copy(update={"judge_model": judge_model})
    items = load_eval_items(settings.raw_data_dir, dataset)
    if max_queries is not None:
        items = items[:max_queries]

    agent = _setup_agent(settings)
    judge = None
    if not no_judge:
        from cleantech_rag.evaluation.judge import LLMJudge

        judge = LLMJudge(settings=settings)

    now = datetime.now()
    config = {
        "dataset": dataset,
        "k": settings.top_k,
        "agent_model": settings.agent_model,
        "embedding_model": settings.embedding_model,
        "judge_enabled": judge is not None,
        "judge_model": settings.judge_model if judge is not None else None,
    }
    log_path = resume or output_dir / f"run_{now.strftime('%Y%m%dT%H%M%S')}.jsonl"
    try:
        log = RunLog(log_path, config, resume=resume is not None)
    except (ResumeMismatchError, FileNotFoundError, FileExistsError) as exc:
        raise _fail(str(exc)) from exc

    if resume is not None:
        try:
            remaining = sum(
                not log.completed(i, q.question, judge is not None) for i, q in enumerate(items)
            )
        except ResumeMismatchError as exc:
            raise _fail(str(exc)) from exc
        typer.echo(f"Resuming {log_path}: {remaining} of {len(items)} questions remaining.")
    typer.echo(f"Evaluating {len(items)} {dataset} queries (log: {log_path})...")
    try:
        records = evaluate_items(items, agent, judge, k=settings.top_k, log=log)
    except ResumeMismatchError as exc:
        raise _fail(str(exc)) from exc
    except BaseException as exc:  # Ctrl-C or a fatal (e.g. judge auth) error
        typer.secho(
            f"Run interrupted ({type(exc).__name__}). Completed rows are saved; "
            f"continue with: --resume {log_path}",
            fg=typer.colors.YELLOW,
            err=True,
        )
        if isinstance(exc, KeyboardInterrupt):
            raise typer.Exit(130) from exc
        raise _fail(f"{type(exc).__name__}: {exc}") from exc
    config = {
        **config,
        "n_queries_selected": len(items),
        "max_queries": max_queries,
        "run_log": log_path.name,
        "timestamp": now.isoformat(timespec="seconds"),
    }
    summary = summarize(records, config, judge_enabled=judge is not None)
    summary_path, csv_path = write_outputs(records, summary, output_dir, now)
    counts = summary["counts"]
    n_new = len(log.written)
    typer.echo(
        f"Summary covers {len(records)} questions "
        f"({n_new} from this run, {len(records) - n_new} from resumed log)."
    )
    typer.echo(f"Done: {counts['n_evaluated']} evaluated, {counts['n_errors']} errors.")
    typer.echo(f"Summary: {summary_path}\nPer-query CSV: {csv_path}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
