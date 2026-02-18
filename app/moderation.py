import json
import os
import sqlite3
from typing import Any, Dict, List, Optional

DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "moderation.db"))


def _connect():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=5)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = _connect()
    with conn:
        conn.execute(
            """
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts DATETIME DEFAULT CURRENT_TIMESTAMP,
            event_type TEXT,
            user_id TEXT,
            message TEXT,
            reason TEXT,
            metadata TEXT
        )
        """
        )
    conn.close()


def log_event(event_type: str, user_id: Optional[str], message: str, reason: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> int:
    conn = _connect()
    meta_json = json.dumps(metadata or {})
    with conn:
        cur = conn.execute(
            "INSERT INTO events (event_type, user_id, message, reason, metadata) VALUES (?, ?, ?, ?, ?)",
            (event_type, user_id, message, reason, meta_json),
        )
        event_id = cur.lastrowid
    conn.close()
    return event_id


def get_events(limit: int = 100) -> List[Dict[str, Any]]:
    conn = _connect()
    with conn:
        cur = conn.execute("SELECT * FROM events ORDER BY ts DESC LIMIT ?", (limit,))
        rows = cur.fetchall()
    conn.close()
    results: List[Dict[str, Any]] = []
    for r in rows:
        meta = {}
        try:
            meta = json.loads(r["metadata"] or "{}")
        except Exception:
            meta = {}
        results.append(
            {
                "id": r["id"],
                "ts": r["ts"],
                "event_type": r["event_type"],
                "user_id": r["user_id"],
                "message": r["message"],
                "reason": r["reason"],
                "metadata": meta,
            }
        )
    return results
