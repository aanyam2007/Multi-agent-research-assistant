from typing import TypedDict, Annotated
import operator


class ResearchState(TypedDict):
    query: str
    plan: str
    current_step: str
    research_data: Annotated[list[str], operator.add]
    analysis: str
    final_report: str
    next_agent: str
    iteration_count: int
