# STATUS: cleantech-rag

Audit date: 2026-10-27. Audited against the goal in `CLAUDE.md` (package refactor, LangGraph migration,
bugs 1-6, offline tests, README with real eval results). No network calls or API spend during the audit;
`.env` and `data/` were not read.

## Works (verified by running)

| Check | Result |
|---|---|
| `uv sync --all-groups` | OK. Installs only `pytest` + `ruff` (no runtime deps declared) |
| `uv run cleantech-rag` | Prints `Hello from cleantech-rag!` (stub `main()` in `src/cleantech_rag/__init__.py`) |
| `uv run pytest -q` | `no tests ran`, exit code 5 (no `tests/` directory). CI tolerates exit 5 |
| Agent kit | `.claude/` agents (planner, implementer, tester, reviewer), commands, `guard.sh` and `lint.sh` hooks present; `agent-ready` label exists on GitHub |
| Notebook secrets | API keys in the notebook are `REDACTED`; no `sk-`/`BSA` key patterns in the cells or the 40 stored outputs |

## Broken

1. **CI is red on `main`** (the last 2 runs both failed at the Lint step). `uv run ruff check .` reports 86 errors
   and `ruff format --check .` reports 2 files that need reformatting. **All of them come from the reference files**
   `notebooks/agentic_rag_system.ipynb` (43) and `notebooks/evaluation.py` (43). There is no `[tool.ruff]` config,
   so ruff 0.17's defaults lint the notebooks. Every PR will fail CI until this is fixed.
2. `pyproject.toml`: `dependencies = []`, placeholder description, and the entry point `cleantech_rag:main` points
   at the hello-world stub.

## Missing (vs. target layout in CLAUDE.md)

Everything under `src/cleantech_rag/` except an empty `__init__.py`:
`config.py`, `data.py`, `vectorstore.py`, `tools/{retriever,summarizer,web_search,reasoning}.py`, `memory.py`,
`agent.py` (LangGraph), `cli.py`, `evaluation/{metrics,judge,datasets,runner}.py`. Also missing:
`tests/` (with offline fakes for embeddings, the chat model, and Brave), `results/`, `.env.example`, and README
(0 bytes). `.DS_Store` is not gitignored.

## What the notebook contains (source to port)

| Notebook cell(s) | Port to | Notes |
|---|---|---|
| 7 CONFIG | `config.py` | Hardcoded keys (redacted), `./chroma_database_v2` path |
| 9-11 load + preprocess | `data.py` | Colab Drive mount. Renames `Unnamed: 0` to `id`. **Extra bug:** metadata reads `row['published']` but the column is `date`, so `published_date` is always `'nan'`. `author`/`domain` are dropped |
| 14 vector DB | `vectorstore.py` | `langchain_community.vectorstores.Chroma` (deprecated) with batches of 100 and `sleep(2)`. No chunking (bug 3). No stable doc IDs |
| 16-17 Retriever/Summarizer | `tools/` | Retriever stores `last_retrieved_ids` on the instance (bug 2). The summarizer lambda calls `retriever.retrieve()` again (bug 4) |
| 19 base ReAct agent | (drop) | Superseded by the enhanced agent |
| 24 WebSearchTool (Brave) | `tools/web_search.py` | `requests` with string error returns |
| 26 ReasoningChainTool | `tools/reasoning.py` | Calls `vectorstore.similarity_search` directly, so its IDs are never tracked (contributes to bug 2) |
| 28 ConversationMemorySystem | `memory.py` | Fine as a store, but wired in with `.partial()` (bug 1) |
| 30 enhanced agent | `agent.py` | Legacy `AgentExecutor` + OpenAI tools parser, 5 tools, `max_iterations=10` |
| 34 evaluators + runner | `evaluation/` | Precision divides by k (bug 6). BLEU wrapped in a bare `except: return 0.0` (bug 6). Judge failures score 0 and drag averages down |
| 36 50-answers parser | `evaluation/datasets.py` | Stored output shows `['47488', '47487']` for queries 1-3 (bug 5) |
| 40-44 charts/stats/export | `evaluation/` + README | matplotlib/seaborn, xlsx export |

`notebooks/evaluation.py` is an older, alternative evaluator. It pulls IDs out of agent text with a regex and uses
BLEU method7. Keep it for reference only and don't port it.

## Known bugs: audit notes

1. **Frozen memory**: confirmed in cell 30 (`prompt.partial(conversation_context=memory.get_context_string())` runs once at creation).
2. **Leaking IDs**: confirmed. The runner reads `retriever.last_retrieved_ids` after `invoke`, and the attribute is never reset. `reasoning_chain` retrieves without updating it. With multiple retrieve calls, only the last one counts.
3. **No chunking**: confirmed. `chunk_size`/`chunk_overlap` are in CONFIG but never used.
4. **Duplicate retrieval**: confirmed (`summarize_and_answer = summarize(q, retriever.retrieve(q))`).
5. **Parser IDs**: the stored output confirms identical IDs for queries 1-3. The regex
   `\*\*(\d+)\.\s+Query:\*\*\s+(.*?)\s*\*\*Desired Output:\*\*\s+(.*?)\s*\*\*Referenced Articles:\*\*\s+([\d,\s]+)`
   uses lazy `.*?` with DOTALL. If an entry's format differs (e.g. a missing or differently formatted
   "Referenced Articles" line), a match can run on into later entries and take another entry's IDs. The file
   itself has to be checked, and I did not read it (data rule).
6. **Metrics**: precision@k divides by `k` regardless of how many docs were retrieved. BLEU = 0.0000 on all 50
   queries even though ROUGE-1 = 0.278, which points to an exception swallowed by the bare `except`. One
   hypothesis to verify: nltk 3.8.1's `Fraction(..., _normalize=False)` fails on Python 3.12+. Tokenization is
   plain `str.split()` (punctuation is attached to words), which lowers scores but would not zero them.

## Risks / unknowns

- **The baseline is not reproducible as-is.** The old Chroma DB isn't in the repo, and chunking (bug 3) changes the
  index anyway. Before/after numbers need a fresh full-corpus embed plus a full eval run (**paid, needs Joe's approval**).
- **Dev-sample retrieval metrics are meaningless.** A `--limit 200` index almost certainly lacks the gold articles
  for the eval questions, so retrieval scores will be about 0 unless the sample deliberately includes the gold IDs.
- **ID space**: gold IDs such as 47488 exceed the row count (20,111). This is fine if `Unnamed: 0` holds original
  IDs, but nobody has checked that every gold ID exists in the corpus.
- **Heavy deps**: `sentence-transformers` pulls in torch (slow CI, large install) and downloads a model on first use.
  nltk `punkt` also downloads. Tests must inject fakes or avoid downloads.
- **LangGraph API churn**: `langgraph.prebuilt.create_react_agent` vs. LangChain v1 `create_agent` vs. a hand-written
  `StateGraph`. Offline tests need a fake chat model that supports `bind_tools` and emits tool calls; the stock
  fakes do not do this well, so a small custom fake is needed.
- **Parallel PRs and `uv.lock`**: issues that add dependencies will conflict on the lockfile and must rebase and re-run `uv lock`.
- **Column schema of the 23-question CSV** is known only partially (`question_id`, `question` from the notebook).
- The judge model (`gpt-4o`) must never run in agent dev loops. Dev runs need `--no-judge` or a cheaper judge override.
