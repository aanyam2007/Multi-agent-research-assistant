# 🔬 Multi-Agent Research Assistant

A production-ready, supervisor-worker multi-agent system that researches any topic and returns a polished, structured report — powered by **LangGraph**, **Groq**, and **Tavily**.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)

---

## How It Works

The system runs a 4-agent pipeline orchestrated by a supervisor:

```
User Query
    │
    ▼
🤖 Orchestrator  ── plans & routes tasks
    │
    ├──▶ 🔍 Researcher  ── searches the web (Tavily)
    │
    ├──▶ 📊 Analyst     ── synthesizes & structures findings
    │
    └──▶ ✍️  Writer      ── drafts the final report
```

Each agent has strict role boundaries enforced by its system prompt — the Researcher only gathers facts, the Analyst only synthesizes, the Writer only drafts. The Orchestrator uses structured JSON outputs (Pydantic) to route between agents and signals `FINISH` when the report is ready.

---

## Features

- **Real-time streaming UI** — watch each agent complete live in the browser
- **Supervisor routing** — Orchestrator uses structured outputs to decide the next step; never hallucinates a route
- **Role-bounded agents** — explicit system prompts prevent agents from overstepping their responsibilities
- **Loop guard** — iteration cap (10) prevents runaway pipelines
- **LangSmith tracing** — every run is automatically traced for debugging
- **Download report** — export the final output as a Markdown file

---

## Tech Stack

| Layer | Technology |
|---|---|
| Agent framework | [LangGraph](https://github.com/langchain-ai/langgraph) |
| LLM | [Groq](https://groq.com) — `llama-3.3-70b-versatile` |
| Web search | [Tavily](https://tavily.com) |
| Observability | [LangSmith](https://smith.langchain.com) |
| UI | [Streamlit](https://streamlit.io) |
| Package manager | [uv](https://github.com/astral-sh/uv) |

---

## Project Structure

```
reserch_agent/
├── app_ui.py            # Streamlit UI (entry point for deployment)
├── graph.py             # LangGraph compilation — nodes, edges, routing
├── state.py             # ResearchState TypedDict
├── prompts.py           # System prompts for all 4 agents
├── main.py              # CLI entry point (local use)
├── api.py               # FastAPI REST wrapper (optional, local use)
├── agents/
│   ├── orchestrator.py  # Supervisor — structured-output routing
│   ├── researcher.py    # Tavily search + LLM fact extraction
│   ├── analyst.py       # Synthesis into structured framework
│   └── writer.py        # Final report drafting
└── tools/
    └── search.py        # TavilySearch factory
```

---

## Running Locally

**Prerequisites:** Python 3.12+, [uv](https://github.com/astral-sh/uv)

### 1. Clone & install

```bash
git clone https://github.com/aanyam2007/Multi-agent-research-assistant.git
cd Multi-agent-research-assistant
uv sync
```

### 2. Set up environment variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=researchagent
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
```

### 3. Run the Streamlit UI

```bash
uv run streamlit run app_ui.py
```

Open [http://localhost:8501](http://localhost:8501)

### 3a. Or run from the CLI

```bash
uv run python main.py
```

---

## Deploying to Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select your repo, set **Main file path** to `app_ui.py`
4. Go to **Settings → Secrets** and add:

```toml
GROQ_API_KEY = "your_groq_api_key"
TAVILY_API_KEY = "your_tavily_api_key"
LANGSMITH_API_KEY = "your_langsmith_api_key"
LANGSMITH_TRACING = "true"
LANGSMITH_PROJECT = "researchagent"
LANGSMITH_ENDPOINT = "https://api.smith.langchain.com"
```

5. Click **Deploy** — the app reads secrets directly and the pipeline runs inside the Streamlit process (no separate server needed).

---

## Agent Pipeline Details

| Agent | Input | Output | Tools |
|---|---|---|---|
| **Orchestrator** | Current state | `next_agent` routing decision | Pydantic structured output |
| **Researcher** | Query | Extracted facts with sources | Tavily Search |
| **Analyst** | Research data | Key themes, evidence, trends, open questions | — |
| **Writer** | Analyst notes | Formatted report (Summary → Findings → Implications → Conclusion) | — |

---

## Getting API Keys

| Service | Free tier | Link |
|---|---|---|
| Groq | Yes — fast LLM inference | [console.groq.com](https://console.groq.com) |
| Tavily | Yes — 1000 searches/month | [app.tavily.com](https://app.tavily.com) |
| LangSmith | Yes — tracing & observability | [smith.langchain.com](https://smith.langchain.com) |
