import logging

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from state import ResearchState
from prompts import RESEARCHER_PROMPT
from tools.search import get_search_tool

logger = logging.getLogger(__name__)


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
    logger.info("[RESEARCHER] Searching Tavily for: \"%s\"", query)

    search_tool = get_search_tool(max_results=5)
    raw_results = search_tool.invoke({"query": query})
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
