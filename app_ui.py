import os

import streamlit as st
from dotenv import load_dotenv

# Load .env locally; on Streamlit Cloud push st.secrets into os.environ
# so LangChain/Groq/Tavily libraries can read their API keys.
load_dotenv()
for _k, _v in st.secrets.items():
    os.environ.setdefault(_k, str(_v))

from graph import build_graph  # noqa: E402 — must come after env setup
from state import ResearchState

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Research Agent",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

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

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("ℹ️ About")
    st.markdown(
        "A **supervisor-worker** multi-agent pipeline:\n\n"
        "1. 🤖 **Orchestrator** plans & routes\n"
        "2. 🔍 **Researcher** searches the web\n"
        "3. 📊 **Analyst** synthesizes findings\n"
        "4. ✍️ **Writer** drafts the report"
    )
    st.divider()
    st.caption("Powered by LangGraph · Groq · Tavily")

# ── Header ────────────────────────────────────────────────────────────────────

st.title("🔬 Multi-Agent Research Assistant")
st.caption("Enter a query and watch the agent pipeline work in real time.")
st.divider()

# ── Query input ───────────────────────────────────────────────────────────────

query = st.text_area(
    "Enter your research query",
    placeholder="e.g. What are the latest breakthroughs in nuclear fusion energy?",
    height=110,
)

run = st.button(
    "🚀 Start Research",
    type="primary",
    disabled=not query.strip(),
    use_container_width=True,
)

# ── Pipeline execution ────────────────────────────────────────────────────────

if run and query.strip():
    graph = get_graph()

    initial_state: ResearchState = {
        "query": query,
        "plan": "",
        "current_step": "",
        "research_data": [],
        "analysis": "",
        "final_report": "",
        "next_agent": "",
        "iteration_count": 0,
    }

    st.divider()
    col_left, col_right = st.columns([2, 3], gap="large")

    with col_right:
        st.subheader("📄 Final Report")
        report_area = st.empty()
        report_area.info("Waiting for the Writer agent to finish…")

    with col_left:
        st.subheader("🔄 Agent Pipeline")

        with st.status("Running research pipeline…", expanded=True) as pipeline_status:
            final_report = ""
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
                        st.write(
                            f"{icon} **{label}** — iteration {iteration} "
                            f"→ routing to `{next_a}`"
                        )
                        if reasoning:
                            with st.expander("Reasoning", expanded=False):
                                st.caption(reasoning)

                    elif node_name == "researcher":
                        chunks = updates.get("research_data", [])
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

    if final_report:
        st.divider()
        st.download_button(
            label="📥 Download Report as Markdown",
            data=final_report,
            file_name="research_report.md",
            mime="text/markdown",
            use_container_width=True,
        )
