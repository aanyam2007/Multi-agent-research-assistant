import os

import streamlit as st
from dotenv import load_dotenv

# Load .env locally; on Streamlit Cloud push st.secrets into os.environ
# so LangChain/Groq/Tavily libraries can read their API keys.
load_dotenv()
try:
    for _k, _v in st.secrets.items():
        os.environ.setdefault(_k, str(_v))
except st.errors.StreamlitSecretNotFoundError:
    pass  # no secrets.toml locally — fine, .env already loaded above

import db  # noqa: E402 — must come after env setup
from graph import build_graph  # noqa: E402 — must come after env setup
from state import ResearchState  # noqa: E402
from tools.rag import DATA_DIR, INDEXABLE_SUFFIXES, build_index  # noqa: E402

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Research Agent",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

db.init_db()

# ── Cache the compiled graph (built once per session) ─────────────────────────

@st.cache_resource
def get_graph():
    return build_graph()

# ── Constants ─────────────────────────────────────────────────────────────────

AGENT_ICONS = {
    "orchestrator": "🤖",
    "researcher":   "🔍",
    "analyst":      "📊",
    "writer":       "✍️",
}

AGENT_LABELS = {
    "orchestrator": "Orchestrator",
    "researcher":   "Researcher",
    "analyst":      "Analyst",
    "writer":       "Writer",
}

# ── Session state ─────────────────────────────────────────────────────────────

if "thread_id" not in st.session_state:
    st.session_state.thread_id = None
if "turns" not in st.session_state:
    st.session_state.turns = []


def _load_thread(thread_id: str | None) -> None:
    st.session_state.thread_id = thread_id
    st.session_state.turns = db.get_turns(thread_id) if thread_id else []


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🔬 Research Agent")

    if st.button("➕ New thread", use_container_width=True):
        _load_thread(None)
        st.rerun()

    st.divider()
    st.caption("Threads")

    threads = db.list_threads()
    if not threads:
        st.caption("No saved threads yet.")
    for t in threads:
        is_active = t["id"] == st.session_state.thread_id
        col_select, col_delete = st.columns([5, 1])
        with col_select:
            if st.button(
                ("📌 " if is_active else "") + t["title"],
                key=f"thread_{t['id']}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
            ):
                _load_thread(t["id"])
                st.rerun()
        with col_delete:
            if st.button("🗑️", key=f"delete_{t['id']}"):
                db.delete_thread(t["id"])
                if is_active:
                    _load_thread(None)
                st.rerun()

    st.divider()
    st.caption("Knowledge Base (RAG)")

    uploaded_files = st.file_uploader(
        "Add documents",
        type=[s.lstrip(".") for s in INDEXABLE_SUFFIXES],
        accept_multiple_files=True,
        key="rag_uploader",
        label_visibility="collapsed",
    )

    if uploaded_files and st.button("📚 Add to knowledge base", use_container_width=True):
        DATA_DIR.mkdir(exist_ok=True)
        for uploaded in uploaded_files:
            (DATA_DIR / uploaded.name).write_bytes(uploaded.getvalue())
        with st.spinner("Indexing documents…"):
            chunk_count = build_index()
        st.success(f"Indexed {chunk_count} chunk(s).")

    indexed_files = sorted(
        p.name for p in DATA_DIR.glob("*") if p.suffix.lower() in INDEXABLE_SUFFIXES
    )
    if indexed_files:
        with st.expander(f"📄 Indexed documents ({len(indexed_files)})", expanded=False):
            for name in indexed_files:
                st.caption(name)
    else:
        st.caption("No documents indexed yet — the Researcher will use web search only.")

    st.divider()
    st.markdown(
        "A **supervisor-worker** multi-agent pipeline:\n\n"
        "1. 🤖 **Orchestrator** plans & routes\n"
        "2. 🔍 **Researcher** searches the web + your docs\n"
        "3. 📊 **Analyst** synthesizes findings\n"
        "4. ✍️ **Writer** drafts the report"
    )
    st.caption("Powered by LangGraph · Groq · Tavily")

# ── Header ────────────────────────────────────────────────────────────────────

st.title("🔬 Multi-Agent Research Assistant")
st.caption("Ask a question, then keep the conversation going — earlier turns are remembered.")
st.divider()

# ── Conversation history ─────────────────────────────────────────────────────

for turn in st.session_state.turns:
    with st.chat_message("user"):
        st.write(turn["query"])
    with st.chat_message("assistant"):
        st.markdown(turn["final_report"] or "_No report was generated for this turn._")
        with st.expander("Pipeline details", expanded=False):
            if turn["plan"]:
                st.caption(f"Orchestrator reasoning: {turn['plan']}")
            if turn["analysis"]:
                st.caption("**Analysis**")
                st.caption(turn["analysis"])
        if turn["final_report"]:
            st.download_button(
                label="📥 Download Report as Markdown",
                data=turn["final_report"],
                file_name="research_report.md",
                mime="text/markdown",
                key=f"download_{turn['id']}",
            )

# ── Query input ───────────────────────────────────────────────────────────────

query = st.chat_input("Ask a research question…")

if query:
    graph = get_graph()

    history = [
        {"query": t["query"], "final_report": t["final_report"]}
        for t in st.session_state.turns
    ]

    initial_state: ResearchState = {
        "query": query,
        "plan": "",
        "current_step": "",
        "research_data": [],
        "analysis": "",
        "final_report": "",
        "next_agent": "",
        "iteration_count": 0,
        "history": history,
    }

    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        report_area = st.empty()
        report_area.info("Waiting for the Writer agent to finish…")

        with st.status("Running research pipeline…", expanded=True) as pipeline_status:
            final_report = ""
            plan = ""
            analysis = ""
            research_data: list[str] = []
            iteration_count = 0

            try:
                for chunk in graph.stream(initial_state, stream_mode="updates"):
                    node_name, updates = next(iter(chunk.items()))

                    icon  = AGENT_ICONS.get(node_name, "⚙️")
                    label = AGENT_LABELS.get(node_name, node_name.capitalize())

                    if node_name == "orchestrator":
                        next_a    = updates.get("next_agent", "")
                        reasoning = updates.get("plan", "")
                        iteration = updates.get("iteration_count", 0)
                        iteration_count = iteration
                        if reasoning:
                            plan = reasoning
                        st.write(
                            f"{icon} **{label}** — iteration {iteration} "
                            f"→ routing to `{next_a}`"
                        )
                        if reasoning:
                            with st.expander("Reasoning", expanded=False):
                                st.caption(reasoning)

                    elif node_name == "researcher":
                        chunks = updates.get("research_data", [])
                        research_data.extend(chunks)
                        st.write(f"{icon} **{label}** — web research gathered")
                        if chunks:
                            with st.expander("Preview", expanded=False):
                                st.caption(chunks[0][:300] + "…")

                    elif node_name == "analyst":
                        analysis = updates.get("analysis", "")
                        st.write(f"{icon} **{label}** — findings synthesized")
                        if analysis:
                            with st.expander("Preview", expanded=False):
                                st.caption(analysis[:300] + "…")

                    elif node_name == "writer":
                        final_report = updates.get("final_report", "")
                        st.write(f"{icon} **{label}** — report drafted")
                        report_area.markdown(final_report)

                    else:
                        st.write(f"{icon} **{label}** — complete")

                pipeline_status.update(
                    label=f"Research complete ✅  ({iteration_count} iterations)",
                    state="complete",
                    expanded=False,
                )

            except Exception as exc:  # noqa: BLE001
                st.error(f"Pipeline error: {exc}")
                pipeline_status.update(label="Pipeline error ❌", state="error")

    if st.session_state.thread_id is None:
        st.session_state.thread_id = db.create_thread(query)

    db.add_turn(
        st.session_state.thread_id,
        query=query,
        plan=plan,
        research_data=research_data,
        analysis=analysis,
        final_report=final_report,
        iteration_count=iteration_count,
    )
    st.session_state.turns = db.get_turns(st.session_state.thread_id)
    st.rerun()
