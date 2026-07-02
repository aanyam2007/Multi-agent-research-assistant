import logging

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from state import ResearchState, format_history
from prompts import RESEARCHER_PROMPT
from tools.search import get_search_tool

logger = logging.getLogger(__name__)


def _resolve_search_query(query: str, history: list[dict]) -> str:
    """Rewrite a follow-up question into a standalone search query using prior turns."""
    if not history:
        return query

    llm = ChatGroq(model="llama-3.3-70b-versatile")
    response = llm.invoke([
        SystemMessage(content=(
            "Rewrite the follow-up question into a standalone search query using the "
            "conversation history for context. Respond with ONLY the rewritten query — "
            "no explanation, no quotes."
        )),
        HumanMessage(
            content=f"Conversation history:\n{format_history(history)}\n\n"
                    f"Follow-up question: {query}\n\nStandalone query:"
        ),
    ])
    return response.content.strip()


def _format_results(raw) -> str:
    if isinstance(raw, list):
        parts = []
        for r in raw:
            if isinstance(r, dict):
                parts.append(
                    f"Source: {r.get('url', 'unknown')}\n"
                    f"Title: {r.get('title', '')}\n"
                    f"{r.get('content', '')}"
                )
            else:
                parts.append(str(r))
        return "\n\n".join(parts)
    return str(raw)


def researcher_node(state: ResearchState) -> dict:
    query = state["query"]
    search_query = _resolve_search_query(query, state.get("history", []))
    logger.info("[RESEARCHER] Searching Tavily for: \"%s\"", search_query)

    search_tool = get_search_tool(max_results=5)
    raw_results = search_tool.invoke({"query": search_query})
    formatted = _format_results(raw_results)

    llm = ChatGroq(model="llama-3.3-70b-versatile")
    response = llm.invoke([
        SystemMessage(content=RESEARCHER_PROMPT),
        HumanMessage(content=f"Query: {query}\n\nSearch Results:\n{formatted}"),
    ])

    logger.info("[RESEARCHER] Complete — extracted %d chars", len(response.content))

    return {
        "research_data": [response.content],
        "current_step": "research_complete",
    }
