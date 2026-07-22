import logging

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from state import ResearchState, format_history
from prompts import RESEARCHER_PROMPT
from tools.search import get_search_tool
from tools.rag import get_relevant_docs

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


def _format_web_results(raw) -> str:
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


def _format_internal_docs(docs: list[dict]) -> str:
    if not docs:
        return "No matching internal documents."
    return "\n\n".join(
        f"Source: {d['source']} (internal knowledge base)\n{d['content']}" for d in docs
    )


def researcher_node(state: ResearchState) -> dict:
    query = state["query"]
    search_query = _resolve_search_query(query, state.get("history", []))
    logger.info("[RESEARCHER] Searching Tavily for: \"%s\"", search_query)

    search_tool = get_search_tool(max_results=5)
    raw_results = search_tool.invoke({"query": search_query})
    formatted_web = _format_web_results(raw_results)

    internal_docs = get_relevant_docs(search_query, k=3)
    logger.info("[RESEARCHER] Internal knowledge base: %d matching chunk(s)", len(internal_docs))
    formatted_internal = _format_internal_docs(internal_docs)

    llm = ChatGroq(model="llama-3.3-70b-versatile")
    response = llm.invoke([
        SystemMessage(content=RESEARCHER_PROMPT),
        HumanMessage(
            content=f"Query: {query}\n\n"
                    f"Web Search Results:\n{formatted_web}\n\n"
                    f"Internal Knowledge Base Results:\n{formatted_internal}"
        ),
    ])

    logger.info("[RESEARCHER] Complete — extracted %d chars", len(response.content))

    return {
        "research_data": [response.content],
        "current_step": "research_complete",
    }
