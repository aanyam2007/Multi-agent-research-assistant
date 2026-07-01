import logging

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from state import ResearchState
from prompts import ANALYST_PROMPT

logger = logging.getLogger(__name__)


def analyst_node(state: ResearchState) -> dict:
    research_data = state.get("research_data", [])
    query = state["query"]

    logger.info("[ANALYST] Synthesizing %d research chunk(s)", len(research_data))

    combined = "\n\n---\n\n".join(research_data)

    llm = ChatGroq(model="llama-3.3-70b-versatile")
    response = llm.invoke([
        SystemMessage(content=ANALYST_PROMPT),
        HumanMessage(content=f"Original Query: {query}\n\nResearch Data:\n{combined}"),
    ])

    logger.info("[ANALYST] Analysis complete — %d chars", len(response.content))

    return {
        "analysis": response.content,
        "current_step": "analysis_complete",
    }
