"""Command line interface: ``ask``, ``chat`` and ``build-index`` commands."""

from __future__ import annotations

import logging
import uuid
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
from cleantech_rag.vectorstore import build_index, estimate_cost, index_exists, load_index

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


@app.command()
def ask(question: str = typer.Argument(..., help="Question to ask the agent.")) -> None:
    """Answer one question (single-shot, no memory); use `chat` for a conversation."""
    agent = _setup_agent(get_settings())
    result = agent.run(question)
    typer.echo(result.answer)


def _echo_sources(result) -> None:
    ids = list(getattr(result, "retrieved_ids", None) or [])
    if ids:
        typer.echo(f"Sources (article IDs): {', '.join(ids)}")
    else:
        typer.echo("Sources: no articles retrieved this turn.")


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


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    app()


if __name__ == "__main__":
    main()
