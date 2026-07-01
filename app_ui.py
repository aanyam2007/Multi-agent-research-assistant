import json
import requests
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Research Agent",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Settings")
    api_base = st.text_input("API Base URL", value="http://localhost:8000")

    st.divider()
    st.markdown("**Start the API server:**")
    st.code("uv run uvicorn api:app --reload", language="bash")

    st.divider()
    # Health-check badge
    try:
        r = requests.get(f"{api_base}/health", timeout=2)
        if r.ok:
            st.success("API server is online ✅")
        else:
            st.warning("API responded with an error")
    except requests.exceptions.ConnectionError:
        st.error("API server is offline ❌")

    st.divider()
    st.caption("Built with LangGraph · Groq · Tavily")

# ── Header ────────────────────────────────────────────────────────────────────

st.title("🔬 Multi-Agent Research Assistant")
st.caption(
    "A supervisor-worker pipeline: **Orchestrator** routes tasks to "
    "**Researcher → Analyst → Writer**"
)

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

# ── SSE parser ────────────────────────────────────────────────────────────────

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


def _parse_sse_line(line: str) -> dict | None:
    if line.startswith("data: "):
        try:
            return json.loads(line[6:])
        except json.JSONDecodeError:
            return None
    return None


# ── Main execution ────────────────────────────────────────────────────────────

if run and query.strip():
    st.divider()

    # Two-column layout: pipeline log left, report right
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
                with requests.post(
                    f"{api_base}/research/stream",
                    json={"query": query},
                    stream=True,
                    timeout=180,
                ) as resp:
                    resp.raise_for_status()

                    for raw_line in resp.iter_lines(decode_unicode=True):
                        event = _parse_sse_line(raw_line)
                        if event is None:
                            continue

                        etype = event.get("event")

                        if etype == "node_complete":
                            node = event.get("node", "")
                            icon = AGENT_ICONS.get(node, "⚙️")
                            label = AGENT_LABELS.get(node, node.capitalize())

                            if node == "orchestrator":
                                next_a = event.get("next_agent", "")
                                iteration = event.get("iteration", "")
                                reasoning = event.get("reasoning", "")
                                iteration_count = iteration
                                st.write(
                                    f"{icon} **{label}** — iteration {iteration} "
                                    f"→ routing to `{next_a}`"
                                )
                                if reasoning:
                                    with st.expander("Reasoning", expanded=False):
                                        st.caption(reasoning)

                            elif node == "researcher":
                                st.write(f"{icon} **{label}** — web research gathered")
                                preview = event.get("preview", "")
                                if preview:
                                    with st.expander("Preview", expanded=False):
                                        st.caption(preview)

                            elif node == "analyst":
                                st.write(f"{icon} **{label}** — findings synthesized")
                                preview = event.get("preview", "")
                                if preview:
                                    with st.expander("Preview", expanded=False):
                                        st.caption(preview)

                            elif node == "writer":
                                st.write(f"{icon} **{label}** — report drafted")
                                final_report = event.get("final_report", "")
                                report_area.markdown(final_report)

                            else:
                                st.write(f"{icon} **{label}** — complete")

                        elif etype == "error":
                            msg = event.get("message", "Unknown error")
                            st.error(f"Pipeline error: {msg}")
                            pipeline_status.update(label="Pipeline error ❌", state="error")

                        elif etype == "done":
                            pipeline_status.update(
                                label=f"Research complete ✅  ({iteration_count} iterations)",
                                state="complete",
                                expanded=False,
                            )

            except requests.exceptions.ConnectionError:
                st.error(
                    f"Cannot reach the API server at **{api_base}**. "
                    "Make sure it's running."
                )
                pipeline_status.update(label="Connection failed ❌", state="error")

            except requests.exceptions.Timeout:
                st.error("Request timed out. The pipeline took too long.")
                pipeline_status.update(label="Timeout ❌", state="error")

            except requests.exceptions.HTTPError as exc:
                st.error(f"API error: {exc}")
                pipeline_status.update(label="API error ❌", state="error")

    # Download button below the columns
    if final_report:
        st.divider()
        st.download_button(
            label="📥 Download Report as Markdown",
            data=final_report,
            file_name="research_report.md",
            mime="text/markdown",
            use_container_width=True,
        )
