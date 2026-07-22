# Project Workflow & Interview Prep

A reference doc explaining how the Multi-Agent Research Assistant actually works end-to-end, and a bank of interview questions (with answers grounded in the real implementation) you can expect if you discuss this project.

---

## 1. High-Level Architecture

```
                              ┌─────────────────────────────┐
                              │        Streamlit UI          │
                              │   (app_ui.py) / CLI (main.py)│
                              └──────────────┬───────────────┘
                                             │ query + history
                                             ▼
                              ┌─────────────────────────────┐
                     ┌───────▶│      🤖 Orchestrator         │◀───────┐
                     │        │  (structured-output router)  │        │
                     │        └──────────────┬───────────────┘        │
                     │                       │ next_agent             │
                     │        ┌──────────────┼───────────────┐        │
                     │        ▼               ▼               ▼       │
                     │  🔍 Researcher    📊 Analyst       ✍️ Writer     │
                     │  Tavily + RAG    synthesizes     drafts final   │
                     │        │               │               │       │
                     └────────┴───────────────┴───────────────┘       │
                                       (loops back to Orchestrator) ───┘
                                             │ FINISH
                                             ▼
                              ┌─────────────────────────────┐
                              │  db.py → threads.db (SQLite) │
                              │  persists the completed turn  │
                              └─────────────────────────────┘
```

The whole pipeline is a **LangGraph `StateGraph`** (`graph.py`) — a supervisor/worker pattern where a single `orchestrator` node decides which worker runs next, and every worker routes back through the orchestrator until it signals `FINISH`.

---

## 2. The State Object

Everything agents read/write flows through one `TypedDict`, `ResearchState` (`state.py`):

| Field | Type | Purpose |
|---|---|---|
| `query` | `str` | Current turn's question |
| `plan` | `str` | Orchestrator's latest reasoning |
| `research_data` | `list[str]` (reducer: `operator.add`) | Accumulates across researcher passes *within one turn* |
| `analysis` | `str` | Analyst's structured notes |
| `final_report` | `str` | Writer's polished output |
| `next_agent` | `str` | Orchestrator's routing decision |
| `iteration_count` | `int` | Loop-guard counter |
| `history` | `list[dict]` | Prior turns (`{query, final_report}`) in this thread — plain field, **not** accumulated by LangGraph, set fresh by the caller on every `invoke()` |

The `research_data` reducer (`Annotated[list[str], operator.add]`) is the one subtlety: LangGraph merges *any* update to that key with what's already there, even the initial input on a fresh `invoke()`. That's why every new turn is run as an **independent `graph.invoke()`/`graph.stream()` call** with a brand-new state dict (not a resumed LangGraph checkpoint) — sidestepping the accumulation gotcha entirely. Cross-turn memory is handled by our own SQLite layer instead of LangGraph's built-in checkpointer.

---

## 3. Step-by-Step Request Flow

1. **User submits a query** (Streamlit `st.chat_input` or CLI prompt).
2. The UI/CLI builds a fresh `ResearchState`: `query` = new question, everything else reset to empty, `history` = prior turns of the active thread (loaded from `threads.db`, or `[]` for a new thread).
3. **Orchestrator** (`agents/orchestrator.py`) inspects state (not the query text — the *shape* of progress: is `research_data` empty? is `analysis` empty? is `final_report` empty?) and asks an LLM (Groq `llama-3.3-70b-versatile`, structured output via Pydantic `OrchestratorDecision`) to pick the next node: `researcher | analyst | writer | FINISH`. A hard iteration cap (`MAX_ITERATIONS = 10`) forces `FINISH` to prevent infinite loops.
4. **Researcher** (`agents/researcher.py`):
   - If `history` is non-empty, first asks an LLM to rewrite the (possibly context-dependent) query into a standalone search query — e.g. "what about its downsides?" → "downsides of quantum computing".
   - Runs that query against **Tavily** (live web search) **and** the **local Chroma RAG index** (`tools/rag.py`) in parallel, formats both result sets with explicit provenance labels, and asks the LLM to extract only factual bullet points, tagged by source — no analysis, no prose.
5. **Analyst** (`agents/analyst.py`) takes all accumulated `research_data` and synthesizes it into a fixed structure: Key Themes → Evidence Summary → Trends & Patterns → Open Questions. No new facts, no new searches.
6. **Writer** (`agents/writer.py`) takes the analyst's notes and drafts the final polished Markdown report: Executive Summary → Key Findings → Analysis & Implications → Conclusion.
7. Control returns to the **Orchestrator** after every worker; once `final_report` is populated, it signals `FINISH` and the graph ends.
8. The UI/CLI persists the completed turn via `db.add_turn(...)` — creating a new thread row on the first turn, appending a turn row every time — and refreshes the sidebar/turn list.

---

## 4. Memory: Threads & Turns (SQLite)

- **`db.py`** owns two tables: `threads` (id, title = first query, created/updated timestamps) and `turns` (thread_id, turn_index, query, plan, research_data, analysis, final_report, iteration_count, timestamp).
- A **thread** = one resumable conversation. A **turn** = one query/response cycle inside it.
- Reopening a thread reloads all its turns; the next query's `history` is built from those turns and threaded into the Researcher's context-resolution step (see §3.4).
- **Known limitation**: on Streamlit Community Cloud, the filesystem is ephemeral — `threads.db` survives while the app instance is warm, but resets on redeploy/sleep/wake. Documented in the README as a tradeoff, not fixed, since it wasn't required for local/demo use.

---

## 5. RAG: Hybrid Retrieval

- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` via `langchain-huggingface` — runs locally, no API key, no per-call cost.
- **Vector store**: `Chroma`, persisted to `./chroma_db`.
- **Ingestion** (`ingest.py` → `tools/rag.build_index()`): walks `./data` for `.txt/.md/.pdf`, chunks with `RecursiveCharacterTextSplitter` (1000 chars, 150 overlap), embeds, and rebuilds the collection from scratch each run (`store.reset_collection()` — simple, no incremental-update complexity).
- **Retrieval** (`tools/rag.get_relevant_docs()`): top-k (`k=3`) similarity search; returns `[]` gracefully if the index doesn't exist yet or is empty, so the pipeline degrades to pure web search with no crash.
- **Fusion point**: inside `researcher_node`, not a separate graph node — RAG and Tavily results are merged into one prompt with labeled sections ("Web Search Results" vs "Internal Knowledge Base Results"), and a single LLM call extracts facts from both, explicitly allowed to note when they conflict.

---

## 6. Deliberate Design Decisions (the "why")

- **Supervisor/worker over a fixed pipeline**: lets the orchestrator skip/repeat steps (e.g., re-research if analysis reveals gaps) instead of a rigid linear chain — at the cost of needing a loop guard.
- **Structured output (Pydantic) for routing**: guarantees the orchestrator's decision is always one of the four valid literals — never a hallucinated free-text route.
- **Strict role prompts per agent**: keeps failure modes isolated and debuggable (if the report is bad, you know whether it's a research gap, an analysis gap, or a writing gap) rather than one mega-prompt doing everything.
- **RAG fused into Researcher rather than a new graph node**: keeps the state machine's shape unchanged (Orchestrator still only reasons about "is research done / analysis done / report done") and treats internal docs as just another fact source, not a different pipeline stage.
- **Own SQLite layer instead of LangGraph's checkpointer for threads**: avoids the `operator.add` reducer accumulating stale `research_data` across turns; trades away "true" resumable graph execution for a much simpler, fully-controlled persistence model tailored to how this app is actually used (one-shot pipeline per turn, not a long-lived stateful agent loop).
- **Local embeddings over hosted**: zero marginal cost and no new API key, at the cost of a heavier local install (`torch`) and slightly lower retrieval quality than e.g. OpenAI embeddings.

---

## 7. Interview Questions & Answers

### Architecture / Multi-Agent Design

**Q: Why LangGraph instead of a simple sequential script?**
A: The orchestrator needs to make a runtime decision about what happens next (and can loop back), not just run steps 1→2→3→4. LangGraph gives you a state machine with conditional edges (`graph.add_conditional_edges`) so routing is explicit and inspectable, plus built-in streaming of intermediate node outputs (`graph.stream(..., stream_mode="updates")`), which is what powers the live pipeline view in the UI.

**Q: What stops the orchestrator from looping forever?**
A: `MAX_ITERATIONS = 10` in `orchestrator_node` — once `iteration_count` hits the cap, it force-returns `FINISH` regardless of what the LLM would have decided.

**Q: How does the orchestrator know what to do next — does it re-read the query every time?**
A: No — it looks at *state*, not intent: whether `research_data` is empty, whether `analysis` is empty, whether `final_report` is empty. That's a deliberate simplification (deterministic-ish routing) rather than asking the LLM to freely reason about strategy each time.

**Q: What happens if the Analyst or Writer produces bad output — does anything catch that?**
A: Not currently — there's no validation/critique step. This is a known gap; a natural extension would be a "critic" node the orchestrator can route to, or a retry-with-feedback loop if the Writer's output fails a quality check.

### State Management

**Q: Why is `research_data` an `Annotated[list[str], operator.add]` but `analysis`/`final_report` are plain strings?**
A: Within a single turn, the orchestrator might send the query back to the Researcher more than once (e.g., to fill gaps), and each pass should *add* to the evidence pool rather than overwrite it. `analysis`/`final_report` are produced once per turn by a single agent, so plain overwrite semantics are correct and simpler.

**Q: You said LangGraph reducers apply even to the initial `invoke()` input — what does that mean in practice, and how did you work around it?**
A: If you call `graph.invoke()` on a thread whose state already has `research_data` populated (e.g., via a checkpointer), passing `research_data: []` as part of the new input doesn't reset it — the reducer concatenates `[] + existing`, so old data leaks into the new turn. The workaround here: don't use LangGraph's own checkpointer/thread persistence at all — every turn is a fully independent `invoke()` call with clean state, and cross-turn memory (the `history` field) is populated from our own SQLite store instead.

**Q: How would you support true mid-conversation resumability (e.g., pause and continue the same in-progress turn) instead of just resuming between completed turns?**
A: That would mean adopting LangGraph's checkpointer (e.g., `SqliteSaver`) keyed by `thread_id`, and giving `research_data` a custom reducer that supports an explicit reset sentinel (since `operator.add` can't distinguish "new turn" from "same turn, more data"). That's a materially bigger change than what this app needed, since it's a one-shot-per-turn pipeline, not a long-running interruptible agent loop.

### RAG

**Q: How do you decide whether to trust the web result or the internal document when they disagree?**
A: The system doesn't automatically resolve conflicts — the Researcher's prompt explicitly asks it to surface disagreements between sources rather than silently picking one, and both are labeled with provenance (URL vs filename) so a downstream reader (or the Analyst) can judge.

**Q: Why chunk at 1000 characters with 150 overlap? Why rebuild the whole index instead of upserting?**
A: 1000/150 is a reasonable default for prose-style docs — big enough to preserve context per chunk, with overlap so a fact split across a chunk boundary isn't lost. Full-rebuild-on-ingest (`reset_collection()`) trades incremental-update efficiency for simplicity: there's no dedup/staleness logic to get wrong, at the cost of re-embedding everything on every `ingest.py` run — acceptable for a personal/small-corpus knowledge base, not for a large ever-growing one.

**Q: What happens if no documents have been ingested yet?**
A: `get_relevant_docs()` checks whether the persisted Chroma directory exists and has any entries; if not, it returns `[]` immediately rather than erroring, so the Researcher just falls back to Tavily-only results.

**Q: Why local `sentence-transformers` embeddings instead of OpenAI/Cohere?**
A: No API key, no per-call cost, works offline — consistent with the project's "free-tier friendly" stack (Groq + Tavily also have generous free tiers). Tradeoff: a much heavier local dependency footprint (`torch`) and somewhat lower embedding quality than a modern hosted model.

**Q: How would you scale this RAG setup to thousands of documents / multiple users?**
A: Chroma's local persistence and single-process embedding model don't scale well past a personal knowledge base — you'd want a managed vector DB (e.g., pgvector, Pinecone, Weaviate), a separate embedding service instead of loading `sentence-transformers` in-process, and incremental ingestion (hash-based change detection) instead of full `reset_collection()` rebuilds.

### Memory / Threads

**Q: Why SQLite instead of a NoSQL store or just JSON files?**
A: The data is inherently relational (threads → turns, one-to-many) and needed simple ordering/filtering (list threads by `updated_at`, get turns by `thread_id` in order) — SQLite's stdlib `sqlite3` module gives that with zero new dependencies and one file to manage.

**Q: How does a follow-up question actually "know" about earlier turns?**
A: The caller (UI/CLI) loads the thread's prior turns from SQLite and passes them as `history` in the fresh `ResearchState`. The Researcher's `_resolve_search_query()` feeds that history plus the new question to an LLM that rewrites it into a standalone query before it ever reaches Tavily/Chroma. Nothing else in the pipeline directly consumes `history` currently — it's scoped narrowly to fixing search relevance, not, e.g., influencing the Writer's tone.

**Q: What's the biggest limitation of this memory design?**
A: Two: (1) on Streamlit Cloud, the SQLite file doesn't survive redeploys/sleep cycles — it's genuinely ephemeral storage, documented but not solved; (2) `history` only carries `query` + `final_report` per turn (not the full plan/analysis), so very long threads could still lose some nuance the Writer originally had access to, since only the end product is remembered.

### Prompt Engineering / LLM Usage

**Q: Why does the Orchestrator use structured output (Pydantic) but the other three agents don't?**
A: The Orchestrator's output is consumed programmatically (`next_agent` drives graph routing) — it must always be one of four valid literals, so structured output eliminates a whole class of parsing failures. The other agents produce free-form Markdown/text meant for a human reader, where structured output would add rigidity without benefit — their "structure" is enforced by prompt instructions instead (fixed headings), not schema validation.

**Q: How do you keep the Researcher from "editorializing" and the Writer from inventing facts?**
A: Explicit negative instructions in each system prompt ("DO NOT interpret... DO NOT perform analysis...", "DO NOT introduce new facts, statistics, or claims not present in the analyst's notes") — role isolation is enforced entirely through prompt design, there's no programmatic guardrail checking this.

### Production Readiness / Follow-ups You Might Get

**Q: What would break first if this went to production with real traffic?**
A: The SQLite thread store (single-writer file, not built for concurrent multi-process access) and the in-process embedding model (loaded once per Streamlit session via `@st.cache_resource`-style caching, not horizontally scalable) — both are fine for single-user/local/demo use and would need swapping for hosted equivalents under real load.

**Q: How would you add automated testing for the agent pipeline?**
A: Unit-test each node function directly (they're plain functions taking/returning dicts — as done ad hoc during development here) with mocked LLM/Tavily/Chroma calls, plus a small number of real end-to-end smoke tests using Streamlit's `AppTest` harness (used during manual verification of this feature) to catch integration issues without needing a browser.

**Q: What's one thing you'd change if you rebuilt this today?**
A: Probably give the Orchestrator visibility into *why* a step failed (e.g., empty research results) rather than just presence/absence of data, so it can make a smarter routing decision instead of mechanically following the fixed "research → analyze → write" progression it currently infers from state shape.
