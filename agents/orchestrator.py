import logging
from typing import Literal

from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from state import ResearchState
from prompts import ORCHESTRATOR_PROMPT

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 10


class OrchestratorDecision(BaseModel):
    reasoning: str
    next_agent: Literal["researcher", "analyst", "writer", "FINISH"]


def orchestrator_node(state: ResearchState) -> dict:
    iteration = state.get("iteration_count", 0)

    if iteration >= MAX_ITERATIONS:
        logger.info("[ORCHESTRATOR] Iteration cap reached → forcing FINISH")
        return {"next_agent": "FINISH", "iteration_count": iteration + 1}

    llm = ChatGroq(model="llama-3.3-70b-versatile").with_structured_output(OrchestratorDecision)

    status = (
        f"Query: {state.get('query', '')}\n"
        f"Research data collected: {len(state.get('research_data', []))} chunk(s)\n"
        f"Analysis complete: {'Yes' if state.get('analysis') else 'No'}\n"
        f"Final report complete: {'Yes' if state.get('final_report') else 'No'}\n"
        f"Iteration: {iteration}"
    )

    decision: OrchestratorDecision = llm.invoke([
        SystemMessage(content=ORCHESTRATOR_PROMPT),
        HumanMessage(content=status),
    ])

    logger.info(
        "[ORCHESTRATOR] Iteration %d → next: %s | Reasoning: \"%s\"",
        iteration + 1,
        decision.next_agent,
        decision.reasoning,
    )

    return {
        "next_agent": decision.next_agent,
        "plan": decision.reasoning,
        "current_step": "orchestrating",
        "iteration_count": iteration + 1,
    }
