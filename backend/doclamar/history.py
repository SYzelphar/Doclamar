"""Chat sessions and messages (SQLite). Upgrades the original chats.db in place."""
from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, List, Optional

_SESSION_COLUMNS = {
    "username": "TEXT DEFAULT 'guest'",
    "mode": "TEXT DEFAULT 'folder'",   # folder | file
    "folder": "TEXT",
    "file_path": "TEXT",
    "updated_at": "TEXT",
}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


class ChatHistory:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._db() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS sessions ("
                " id TEXT PRIMARY KEY, title TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS messages ("
                " id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, role TEXT, content TEXT,"
                " created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,"
                " FOREIGN KEY (session_id) REFERENCES sessions (id))"
            )
            existing = {r[1] for r in conn.execute("PRAGMA table_info(sessions)")}
            for column, ddl in _SESSION_COLUMNS.items():
                if column not in existing:
                    conn.execute(f"ALTER TABLE sessions ADD COLUMN {column} {ddl}")
            if "citations" not in {r[1] for r in conn.execute("PRAGMA table_info(messages)")}:
                conn.execute("ALTER TABLE messages ADD COLUMN citations TEXT")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(username)")

    @contextmanager
    def _db(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def create_session(self, username: str, mode: str, title: str,
                       folder: Optional[str] = None, file_path: Optional[str] = None) -> dict:
        session_id = str(uuid.uuid4())
        with self._db() as conn:
            conn.execute(
                "INSERT INTO sessions (id, title, username, mode, folder, file_path, created_at, updated_at)"
                " VALUES (?,?,?,?,?,?,?,?)",
                (session_id, title, username, mode, folder, file_path, _now(), _now()),
            )
        return self.get_session(session_id)

    def get_session(self, session_id: str) -> Optional[dict]:
        with self._db() as conn:
            row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
        if not row:
            return None
        session = dict(row)
        session["mode"] = session.get("mode") or "folder"
        return session

    def set_folder(self, session_id: str, folder: str) -> None:
        with self._db() as conn:
            conn.execute("UPDATE sessions SET folder = ? WHERE id = ?", (folder, session_id))

    def list_sessions(self, username: str) -> List[dict]:
        with self._db() as conn:
            rows = conn.execute(
                "SELECT id, title, mode, folder, file_path, created_at,"
                " COALESCE(updated_at, created_at) AS updated_at FROM sessions"
                " WHERE username = ? ORDER BY COALESCE(updated_at, created_at) DESC",
                (username,),
            ).fetchall()
        return [dict(r) for r in rows]

    def delete_session(self, session_id: str, username: str) -> bool:
        with self._db() as conn:
            owned = conn.execute(
                "SELECT 1 FROM sessions WHERE id = ? AND username = ?", (session_id, username)
            ).fetchone()
            if not owned:
                return False
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        return True

    def add_exchange(self, session_id: str, question: str, answer: str, citations: list) -> None:
        now = _now()
        with self._db() as conn:
            conn.execute(
                "INSERT INTO messages (session_id, role, content, created_at) VALUES (?,?,?,?)",
                (session_id, "user", question, now),
            )
            # Stored as 'ai' (the original app's convention) so older builds can still read it.
            conn.execute(
                "INSERT INTO messages (session_id, role, content, created_at, citations) VALUES (?,?,?,?,?)",
                (session_id, "ai", answer, now, json.dumps(citations)),
            )
            conn.execute("UPDATE sessions SET updated_at = ? WHERE id = ?", (now, session_id))

    def messages(self, session_id: str) -> List[dict]:
        with self._db() as conn:
            rows = conn.execute(
                "SELECT role, content, created_at, citations FROM messages"
                " WHERE session_id = ? ORDER BY id", (session_id,),
            ).fetchall()
        out = []
        for r in rows:
            role = "assistant" if r["role"] in ("ai", "assistant") else "user"
            out.append({
                "role": role,
                "content": r["content"],
                "created_at": r["created_at"],
                "citations": json.loads(r["citations"]) if r["citations"] else [],
            })
        return out

    def recent_turns(self, session_id: str, max_turns: int = 3) -> List[dict]:
        msgs = self.messages(session_id)[-2 * max_turns:]
        return [{"role": m["role"], "content": m["content"]} for m in msgs]
