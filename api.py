import asyncio
import json
import logging
import queue
import threading
from typing import AsyncGenerator

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from graph import build_graph

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Research Agent API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_graph = build_graph()


# ── Request / Response models ────────────────────────────────────────────────

class ResearchRequest(BaseModel):
    query: str


class ResearchResponse(BaseModel):
    final_report: str
    analysis: str
    iterations: int


# ── Helpers ──────────────────────────────────────────────────────────────────

def _initial_state(query: str) -> dict:
    return {
        "query": query,
        "plan": "",
        "current_step": "",
        "research_data": [],
        "analysis": "",
        "final_report": "",
        "next_agent": "",
        "iteration_count": 0,
    }


def _build_sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


async def _stream_graph(query: str) -> AsyncGenerator[str, None]:
    """
    Runs graph.stream() in a background thread and yields SSE events
    for each completed node without blocking the async event loop.
    """
    event_queue: queue.Queue = queue.Queue()

    def _run() -> None:
        try:
            for chunk in _graph.stream(_initial_state(query), stream_mode="updates"):
                event_queue.put(chunk)
        except Exception as exc:  # noqa: BLE001
            event_queue.put(exc)
        finally:
            event_queue.put(None)  # sentinel — signals end of stream

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    while True:
        try:
            item = event_queue.get_nowait()
        except queue.Empty:
            await asyncio.sleep(0.05)
            continue

        if item is None:
            yield _build_sse({"event": "done"})
            break

        if isinstance(item, Exception):
            yield _build_sse({"event": "error", "message": str(item)})
            break

        # Each chunk: {node_name: {state_updates}}
        node_name, updates = next(iter(item.items()))
        event: dict = {"event": "node_complete", "node": node_name}

        if node_name == "orchestrator":
            event["next_agent"] = updates.get("next_agent", "")
            event["reasoning"] = updates.get("plan", "")
            event["iteration"] = updates.get("iteration_count", 0)

        elif node_name == "researcher":
            chunks = updates.get("research_data", [])
            event["message"] = "Research gathered from web"
            event["preview"] = chunks[0][:300] + "…" if chunks else ""

        elif node_name == "analyst":
            event["message"] = "Analysis complete"
            event["preview"] = (updates.get("analysis", "")[:300] + "…")

        elif node_name == "writer":
            event["message"] = "Report drafted"
            event["final_report"] = updates.get("final_report", "")

        yield _build_sse(event)


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/research", response_model=ResearchResponse)
async def research(req: ResearchRequest) -> ResearchResponse:
    """Blocking endpoint — waits for the full pipeline to finish."""
    if not req.query.strip():
        raise HTTPException(status_code=422, detail="Query must not be empty.")

    logger.info("[API] /research query=%r", req.query)
    result = await asyncio.to_thread(_graph.invoke, _initial_state(req.query))

    return ResearchResponse(
        final_report=result.get("final_report", ""),
        analysis=result.get("analysis", ""),
        iterations=result.get("iteration_count", 0),
    )


@app.post("/research/stream")
async def research_stream(req: ResearchRequest) -> StreamingResponse:
    """SSE streaming endpoint — emits an event after each agent completes."""
    if not req.query.strip():
        raise HTTPException(status_code=422, detail="Query must not be empty.")

    logger.info("[API] /research/stream query=%r", req.query)
    return StreamingResponse(
        _stream_graph(req.query),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
