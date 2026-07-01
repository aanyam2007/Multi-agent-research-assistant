import logging
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)

from graph import build_graph  # noqa: E402 — must come after load_dotenv


def main() -> None:
    graph = build_graph()

    query = input("Enter your research query: ").strip()
    if not query:
        print("No query provided.")
        return

    print(f"\nStarting research on: {query}\n{'=' * 60}\n")

    result = graph.invoke({
        "query": query,
        "plan": "",
        "current_step": "",
        "research_data": [],
        "analysis": "",
        "final_report": "",
        "next_agent": "",
        "iteration_count": 0,
    })

    print("\n" + "=" * 60)
    print("FINAL REPORT")
    print("=" * 60 + "\n")
    print(result.get("final_report", "No report was generated."))


if __name__ == "__main__":
    main()
