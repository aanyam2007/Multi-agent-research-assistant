import logging
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)

import db  # noqa: E402 — must come after load_dotenv
from graph import build_graph  # noqa: E402 — must come after load_dotenv


def _pick_thread() -> tuple[str | None, list[dict]]:
    threads = db.list_threads()
    if not threads:
        return None, []

    print("\nExisting threads:")
    for i, t in enumerate(threads, start=1):
        print(f"  {i}. {t['title']}  (updated {t['updated_at']})")
    print("  0. Start a new thread\n")

    choice = input("Resume a thread (number) or press Enter for new: ").strip()
    if not choice or choice == "0":
        return None, []

    try:
        thread = threads[int(choice) - 1]
    except (ValueError, IndexError):
        print("Invalid choice — starting a new thread.")
        return None, []

    turns = db.get_turns(thread["id"])
    print(f"\nResumed thread: {thread['title']} ({len(turns)} turn(s))\n")
    return thread["id"], turns


def main() -> None:
    db.init_db()
    graph = build_graph()
    thread_id, turns = _pick_thread()

    while True:
        query = input("Enter your research query (or 'exit'): ").strip()
        if not query or query.lower() in {"exit", "quit"}:
            break

        print(f"\nStarting research on: {query}\n{'=' * 60}\n")

        history = [{"query": t["query"], "final_report": t["final_report"]} for t in turns]

        result = graph.invoke({
            "query": query,
            "plan": "",
            "current_step": "",
            "research_data": [],
            "analysis": "",
            "final_report": "",
            "next_agent": "",
            "iteration_count": 0,
            "history": history,
        })

        print("\n" + "=" * 60)
        print("FINAL REPORT")
        print("=" * 60 + "\n")
        print(result.get("final_report", "No report was generated."))
        print()

        if thread_id is None:
            thread_id = db.create_thread(query)

        db.add_turn(
            thread_id,
            query=query,
            plan=result.get("plan", ""),
            research_data=result.get("research_data", []),
            analysis=result.get("analysis", ""),
            final_report=result.get("final_report", ""),
            iteration_count=result.get("iteration_count", 0),
        )
        turns = db.get_turns(thread_id)


if __name__ == "__main__":
    main()
