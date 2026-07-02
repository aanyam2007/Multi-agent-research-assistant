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
    # Prior turns in this thread: [{"query": ..., "final_report": ...}, ...].
    # Set once per invoke by the caller (not mutated by nodes) so follow-up
    # queries can be resolved against earlier conversation context.
    history: list[dict]


def format_history(history: list[dict]) -> str:
    if not history:
        return ""
    parts = [
        f"Turn {i} — Q: {turn['query']}\nA: {turn['final_report']}"
        for i, turn in enumerate(history, start=1)
    ]
    return "\n\n".join(parts)
