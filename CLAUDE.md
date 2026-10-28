# CleanTech Agentic RAG

## What this is
Agentic RAG system that answers questions about clean technology using ~20k CleanTech news articles
(ChromaDB + OpenAI embeddings), plus web search (Brave), a reasoning-chain tool, and conversation memory.
Includes an evaluation suite (Precision/Recall/MRR/NDCG@5, ROUGE/BLEU, semantic similarity, LLM-as-judge).
Originally a single Colab notebook (`notebooks/agentic_rag_system.ipynb`, kept as reference, do not edit it).

## Goal of the current work
Turn the notebook into a clean, tested, portfolio-quality Python package that runs locally:
1. Structured package under `src/cleantech_rag/`, no Colab/Drive code, config via env vars
2. Migrate agent orchestration from legacy LangChain `AgentExecutor` to **LangGraph**
3. Fix the known bugs below and make evaluation trustworthy
4. Tests (mocked, offline), README with architecture + real eval results, simple CLI
Deployment (Docker / GCP Cloud Run) is a LATER phase. Don't add it unless an issue asks for it.

## Target layout
```
src/cleantech_rag/
  config.py          # pydantic-settings / env vars, model names, paths
  data.py            # load + clean CSV → Documents (chunking here)
  vectorstore.py     # build / load Chroma
  tools/             # retriever.py, web_search.py, reasoning.py
  memory.py          # conversation memory
  agent.py           # LangGraph agent wiring
  cli.py             # `uv run cleantech-rag ask "..."`, `build-index`, `evaluate`
  evaluation/        # metrics.py, judge.py, datasets.py (parsers), runner.py
tests/               # mirrors src layout
data/raw/            # CSVs + CleanTech-50answers.txt (GITIGNORED, never commit)
data/chroma/         # vector DB (GITIGNORED, rebuilt by `build-index`)
results/             # eval outputs (commit only summary JSON, .md tables, .png charts)
```

## Data (not in git)
- `data/raw/cleantech_media_dataset_v3_2024-10-28.csv`: 20,111 articles; columns: Unnamed: 0 (→ id), title, date, author, content, domain, url
- `data/raw/cleantech_rag_evaluation_data_2024-09-20.csv`: 23 simple Qs, `;`-delimited
- `data/raw/CleanTech-50answers.txt`: 50 complex Qs with reference answers + article IDs

## Stack
- Python 3.11+, `uv`. LangGraph + langchain-openai + langchain-chroma, chromadb, sentence-transformers, rouge-score, nltk
- Models: `gpt-4o-mini` (agent), `gpt-4o` (judge), `text-embedding-3-small`. All configurable.
- Secrets in `.env` only: `OPENAI_API_KEY`, `BRAVE_API_KEY`. Load with python-dotenv / pydantic-settings.

## Commands
- Install: `uv sync`
- Test: `uv run pytest -q`
- Lint: `uv run ruff check . && uv run ruff format --check .`

## Conventions
- Article IDs are str everywhere (loader, retriever, parsers, eval).

## COST RULES (these calls cost real money)
- NEVER embed the full corpus, run the full 50-question eval, or call gpt-4o yourself. Ask Joe first.
- For dev runs use a sample: `--limit 200` articles, `--max-queries 3`.
- Tests must NEVER hit the network: mock OpenAI, Brave, and embeddings (use a fake embedding class).

## Known bugs in the original notebook (fix via issues, don't silently change behavior elsewhere)
1. **Memory is frozen**: conversation context is baked into the prompt with `.partial()` at agent creation, so memory never updates between turns.
2. **Eval retrieval IDs leak across queries**: `last_retrieved_ids` isn't reset per query; if the agent uses web_search/reasoning_chain (which bypass RetrieverTool), metrics use the previous query's IDs.
3. **No chunking**: `chunk_size`/`chunk_overlap` are configured but unused; whole articles are embedded (long ones get truncated).
4. **Duplicate retrieval**: `summarize_and_answer` re-runs retrieval; the agent often retrieves twice.
5. **Verify**: the 50-answers parser returned the same article IDs (`47488, 47487`) for the first 3 queries; check the parser/file.
6. Precision@k divides by k even when fewer docs are retrieved; BLEU is 0.0 for every query (check tokenization).

## Baseline results (original notebook, 50 complex Qs) for before/after comparison
precision@5 0.092 · recall@5 0.310 · MRR 0.357 · NDCG@5 0.290 · ROUGE-1 0.278 · semantic sim 0.676 · LLM-judge overall 4.12/5
(Note: retrieval numbers are suspect because of bugs 2 and 5.)

## Workflow rules (all agents)
- One GitHub issue = one branch = one PR. Stay inside the issue's scope; note other findings under "Follow-ups".
- Never commit to or push `main`. Never merge PRs. Never force-push. (A hook enforces this.)
- Never read, print, or commit `.env`, data files, or API keys.
- Run tests before claiming anything works. Never report a test as passing unless it ran.
- Conventional Commits: `feat(scope): ...`, `fix: ...`, `test: ...`, `refactor: ...`.
- Unsure about a design decision? Put it under "Questions for Joe" in the PR and pick the most reversible option.
- New dependency? One-line justification in the PR.
