import sqlite3

import pytest
from fastapi.testclient import TestClient

from api import TOKEN_HEADER, create_app
from doclamar.history import ChatHistory
from doclamar.indexer import IndexJob

TOKEN = "test-token"


@pytest.fixture
def client(engine):
    app = create_app(engine, token=TOKEN, extra_hosts=["testserver"])
    with TestClient(app, headers={TOKEN_HEADER: TOKEN}) as c:
        yield c


def test_token_is_required(engine):
    app = create_app(engine, token=TOKEN, extra_hosts=["testserver"])
    with TestClient(app) as anon:
        assert anon.get("/health").status_code == 200  # health stays open for readiness polling
        assert anon.get("/sessions").status_code == 401
        assert anon.get("/sessions", headers={TOKEN_HEADER: "wrong"}).status_code == 401
        assert anon.post("/chat", json={"question": "x", "folder": "C:/"}).status_code == 401


def test_cors_allows_the_app_but_not_other_sites(client):
    preflight = {"Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": TOKEN_HEADER}
    ok = client.options("/chat", headers={"Origin": "http://localhost:5173", **preflight})
    assert ok.headers.get("access-control-allow-origin") == "http://localhost:5173"
    evil = client.options("/chat", headers={"Origin": "https://evil.example", **preflight})
    assert "access-control-allow-origin" not in evil.headers


def test_foreign_host_header_is_rejected(engine):
    app = create_app(engine, token=TOKEN)
    with TestClient(app, base_url="http://attacker.example") as c:
        assert c.get("/health").status_code == 400


def test_index_then_chat_then_history(client, docs):
    status = client.post("/index", json={"folder": str(docs)}).json()
    assert status["job"]["state"] in ("queued", "scanning", "indexing", "done")
    for _ in range(200):
        status = client.get("/index/status", params={"folder": str(docs)}).json()
        if status["job"]["state"] == "done":
            break
    assert status["files_indexed"] == 3

    r = client.post("/chat", json={"question": "Which loss function?", "folder": str(docs), "username": "shlok"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "answered" and body["citations"][0]["file"] == "notes.txt"
    sid = body["session_id"]

    r2 = client.post("/chat", json={"question": "and the learning rate?", "session_id": sid, "username": "shlok"})
    assert r2.json()["session_id"] == sid

    sessions = client.get("/sessions", params={"username": "shlok"}).json()["sessions"]
    assert [s["id"] for s in sessions] == [sid] and sessions[0]["folder"] == str(docs)
    messages = client.get(f"/sessions/{sid}", params={"username": "shlok"}).json()["messages"]
    assert [m["role"] for m in messages] == ["user", "assistant", "user", "assistant"]
    assert messages[1]["citations"][0]["file"] == "notes.txt"  # citations survive a reload

    # other users can't read or delete it
    assert client.get(f"/sessions/{sid}", params={"username": "someone"}).status_code == 404
    assert client.delete(f"/sessions/{sid}", params={"username": "someone"}).status_code == 404
    assert client.delete(f"/sessions/{sid}", params={"username": "shlok"}).status_code == 200
    assert client.get("/sessions", params={"username": "shlok"}).json()["sessions"] == []


def test_file_chat_is_scoped_to_that_file(client, docs):
    r = client.post("/sessions/file", json={"file_path": str(docs / "notes.txt"), "username": "u"})
    assert r.status_code == 200
    session = r.json()["session"]
    assert session["mode"] == "file" and r.json()["file"]["chunks"] >= 1
    body = client.post("/chat", json={"question": "pancakes flour eggs", "session_id": session["id"],
                                      "username": "u"}).json()
    assert all(c["file"] == "notes.txt" for c in body["citations"])


def test_file_chat_errors(client, docs, tmp_path):
    assert client.post("/sessions/file", json={"file_path": str(docs / "nope.pdf")}).status_code == 404
    assert client.post("/sessions/file", json={"file_path": str(docs / "data.xlsx")}).status_code == 415
    empty = tmp_path / "empty.txt"
    empty.write_text("   ", encoding="utf-8")
    r = client.post("/sessions/file", json={"file_path": str(empty)})
    assert r.status_code == 422 and "No extractable text" in r.json()["detail"]


def test_chat_while_first_index_is_running(client, engine, docs):
    from doclamar.store import path_key

    engine.indexer._jobs[path_key(str(docs))] = IndexJob(folder=str(docs), state="indexing", to_index=3)
    body = client.post("/chat", json={"question": "anything", "folder": str(docs)}).json()
    assert body["status"] == "indexing" and "still indexing" in body["answer"]


def test_chat_validation(client, docs):
    assert client.post("/chat", json={"question": "hi"}).status_code == 400  # no folder
    assert client.post("/chat", json={"question": "hi", "folder": str(docs / "missing")}).status_code == 404
    assert client.post("/chat", json={"question": "", "folder": str(docs)}).status_code == 422
    assert client.post("/chat", json={"question": "hi", "session_id": "nope"}).status_code == 404


def test_llm_config_endpoint(client):
    r = client.post("/config/llm", json={"provider": "groq", "api_key": "k", "validate_key": False})
    assert r.status_code == 200 and r.json()["configured"]
    assert client.get("/health").json()["llm"]["configured"]


def test_old_chat_database_is_upgraded(tmp_path):
    db = tmp_path / "chats.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY, title TEXT,"
                 " created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, username TEXT DEFAULT 'guest')")
    conn.execute("CREATE TABLE messages (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, role TEXT,"
                 " content TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)")
    conn.execute("INSERT INTO sessions (id, title, username) VALUES ('old', 'Folder: docs', 'shlok')")
    conn.executemany("INSERT INTO messages (session_id, role, content, created_at) VALUES (?,?,?,?)",
                     [("old", "user", "q", "2026-01-01 10:00:00"), ("old", "ai", "a", "2026-01-01 10:00:00")])
    conn.commit()
    conn.close()

    history = ChatHistory(db)
    assert history.get_session("old")["mode"] == "folder"
    assert [m["role"] for m in history.messages("old")] == ["user", "assistant"]
    assert history.list_sessions("shlok")[0]["id"] == "old"
