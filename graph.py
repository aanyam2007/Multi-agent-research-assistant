from langgraph.graph import StateGraph, END

from state import ResearchState
from agents.orchestrator import orchestrator_node
from agents.researcher import researcher_node
from agents.analyst import analyst_node
from agents.writer import writer_node


def _route(state: ResearchState) -> str:
    next_agent = state.get("next_agent", "researcher")
    return END if next_agent == "FINISH" else next_agent


def build_graph():
    graph = StateGraph(ResearchState)

    graph.add_node("orchestrator", orchestrator_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("analyst", analyst_node)
    graph.add_node("writer", writer_node)

    graph.set_entry_point("orchestrator")

    graph.add_conditional_edges(
        "orchestrator",
        _route,
        {
            "researcher": "researcher",
            "analyst": "analyst",
            "writer": "writer",
            END: END,
        },
    )

    graph.add_edge("researcher", "orchestrator")
    graph.add_edge("analyst", "orchestrator")
    graph.add_edge("writer", "orchestrator")

    return graph.compile()
