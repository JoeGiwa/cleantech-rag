"""Command line interface: ``cleantech-rag ask`` and ``cleantech-rag build-index``."""

from __future__ import annotations

import logging
from pathlib import Path

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
from cleantech_rag.vectorstore import build_index, estimate_cost, load_index

app = typer.Typer(help="CleanTech agentic RAG.", no_args_is_help=True, add_completion=False)

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


@app.command()
def ask(
    question: str = typer.Argument(..., help="Question to ask the agent."),
    thread_id: str | None = typer.Option(
        None, "--thread-id", help="Conversation thread (memory is in-process only)."
    ),
) -> None:
    """Answer a question using the agent (retrieval, web search, reasoning)."""
    settings = get_settings()
    _require_key(settings, "openai_api_key")
    if settings.brave_api_key is None:
        typer.secho(
            "Warning: BRAVE_API_KEY is not set; web search will be unavailable.",
            fg=typer.colors.YELLOW,
            err=True,
        )
    store = load_index(settings)
    agent = create_agent(store, _make_llm(settings), settings)
    result = agent.run(question, thread_id=thread_id)
    typer.echo(result.answer)


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


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    app()


if __name__ == "__main__":
    main()
