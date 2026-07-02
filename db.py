"""SQLite persistence for research threads.

A "thread" is a resumable conversation: a sequence of turns (query → final
report) that share context. Threads and their turns are stored in a local
SQLite database so the UI/CLI can list past threads and reopen any of them.
"""

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).parent / "threads.db"

TITLE_MAX_LEN = 80


@contextmanager
def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS threads (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS turns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id TEXT NOT NULL REFERENCES threads(id) ON DELETE CASCADE,
                turn_index INTEGER NOT NULL,
                query TEXT NOT NULL,
                plan TEXT,
                research_data TEXT,
                analysis TEXT,
                final_report TEXT,
                iteration_count INTEGER,
                created_at TEXT NOT NULL
            )
            """
        )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_thread(first_query: str) -> str:
    thread_id = uuid.uuid4().hex
    title = first_query.strip().replace("\n", " ")
    if len(title) > TITLE_MAX_LEN:
        title = title[: TITLE_MAX_LEN - 1].rstrip() + "…"
    now = _now()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO threads (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (thread_id, title, now, now),
        )
    return thread_id


def add_turn(
    thread_id: str,
    query: str,
    plan: str,
    research_data: list[str],
    analysis: str,
    final_report: str,
    iteration_count: int,
) -> None:
    with _connect() as conn:
        turn_index = conn.execute(
            "SELECT COUNT(*) FROM turns WHERE thread_id = ?", (thread_id,)
        ).fetchone()[0]
        now = _now()
        conn.execute(
            """
            INSERT INTO turns
                (thread_id, turn_index, query, plan, research_data, analysis, final_report, iteration_count, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                thread_id,
                turn_index,
                query,
                plan,
                json.dumps(research_data),
                analysis,
                final_report,
                iteration_count,
                now,
            ),
        )
        conn.execute("UPDATE threads SET updated_at = ? WHERE id = ?", (now, thread_id))


def list_threads() -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, title, created_at, updated_at FROM threads ORDER BY updated_at DESC"
        ).fetchall()
    return [dict(row) for row in rows]


def get_turns(thread_id: str) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM turns WHERE thread_id = ? ORDER BY turn_index ASC",
            (thread_id,),
        ).fetchall()
    turns = []
    for row in rows:
        turn = dict(row)
        turn["research_data"] = json.loads(turn["research_data"] or "[]")
        turns.append(turn)
    return turns


def delete_thread(thread_id: str) -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM turns WHERE thread_id = ?", (thread_id,))
        conn.execute("DELETE FROM threads WHERE id = ?", (thread_id,))
