import logging

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from state import ResearchState
from prompts import WRITER_PROMPT

logger = logging.getLogger(__name__)


def writer_node(state: ResearchState) -> dict:
    analysis = state.get("analysis", "")
    query = state["query"]

    logger.info("[WRITER] Drafting final report")

    llm = ChatGroq(model="llama-3.3-70b-versatile")
    response = llm.invoke([
        SystemMessage(content=WRITER_PROMPT),
        HumanMessage(content=f"Original Query: {query}\n\nAnalyst's Structured Notes:\n{analysis}"),
    ])

    logger.info("[WRITER] Report complete — %d chars", len(response.content))

    return {
        "final_report": response.content,
        "current_step": "writing_complete",
    }
