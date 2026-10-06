"""DocLAMAR HTTP API, run as a local sidecar by the Electron app.

Security model: the server binds to 127.0.0.1 only, rejects Host headers other
than localhost (DNS-rebinding), and requires a random per-launch token
(X-Doclamar-Token) that Electron generates and hands to both this process and
the renderer. Other websites open in the user's browser therefore cannot read
files or chats through it.
"""
from __future__ import annotations

import logging
import os
import secrets
import sys
import threading
from contextlib import asynccontextmanager
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from doclamar import __version__
from doclamar.config import Settings
from doclamar.embeddings import ModelUnavailable
from doclamar.engine import Engine
from doclamar.llm import LLMError
from doclamar.parsing import SUPPORTED_EXTENSIONS
from doclamar.store import Scope

logger = logging.getLogger("doclamar.api")

TOKEN_HEADER = "X-Doclamar-Token"
ALLOWED_ORIGINS = r"^(null|file://.*|https?://(localhost|127\.0\.0\.1)(:\d+)?)$"


def setup_logging(settings: Settings) -> None:
    settings.ensure_dirs()
    handlers: List[logging.Handler] = [
        RotatingFileHandler(settings.log_file, maxBytes=5 * 1024 * 1024, backupCount=2, encoding="utf-8")
    ]
    if sys.stdout is not None:
        handlers.append(logging.StreamHandler(sys.stdout))
    # force=True: the old app's log file stayed empty because an earlier
    # basicConfig() call (in main.py) silently won.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=handlers, force=True)


# ----------------------------------------------------------------- schemas

class LLMSettingsRequest(BaseModel):
    provider: str
    api_key: str
    model: Optional[str] = None
    validate_key: bool = True


class FolderRequest(BaseModel):
    folder: str


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    username: str = "guest"
    session_id: Optional[str] = None
    folder: Optional[str] = None


class FileSessionRequest(BaseModel):
    file_path: str
    username: str = "guest"


# --------------------------------------------------------------------- app

def create_app(engine: Optional[Engine] = None, token: Optional[str] = None,
               extra_hosts: Optional[List[str]] = None) -> FastAPI:
    engine = engine or Engine()
    token = token if token is not None else os.environ.get("DOCLAMAR_TOKEN") or ""
    if not token:
        logger.warning("DOCLAMAR_TOKEN is not set: the API is unauthenticated (development only).")

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        engine.close()

    app = FastAPI(title="DocLAMAR", version=__version__, lifespan=lifespan)
    app.state.engine = engine

    @app.middleware("http")
    async def require_token(request: Request, call_next):
        if token and request.url.path != "/health":
            supplied = request.headers.get(TOKEN_HEADER, "")
            if not secrets.compare_digest(supplied.encode(), token.encode()):
                return JSONResponse({"detail": "Unauthorized"}, status_code=401)
        return await call_next(request)

    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", *(extra_hosts or [])])
    # Added last = outermost, so preflight requests are answered before the token check.
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=ALLOWED_ORIGINS,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type", TOKEN_HEADER],
    )

    @app.exception_handler(ModelUnavailable)
    async def model_unavailable(_: Request, exc: ModelUnavailable):
        return JSONResponse({"detail": str(exc)}, status_code=503)

    # ------------------------------------------------------------ helpers

    def folder_status(folder: str) -> dict:
        job = engine.indexer.job(folder)
        return {"folder": os.path.abspath(folder), "job": job.to_dict() if job else None,
                **engine.store.folder_summary(folder)}

    def owned_session(session_id: str, username: str) -> dict:
        session = engine.history.get_session(session_id)
        if not session or (session.get("username") or "guest") != username:
            raise HTTPException(404, "Chat not found.")
        return session

    # ---------------------------------------------------------- endpoints
    # Plain `def` endpoints run in FastAPI's thread pool, so a slow answer never
    # blocks /health or /sessions (the old async endpoints froze the whole server).

    @app.get("/health")
    def health():
        return {"status": "ready", "version": __version__, "llm": engine.llm.describe(),
                "ocr": {"enabled": engine.ocr is not None}}

    @app.post("/config/llm")
    def configure_llm(req: LLMSettingsRequest):
        try:
            return engine.llm.configure(req.provider, req.api_key, req.model, validate=req.validate_key)
        except LLMError as e:
            raise HTTPException(400, str(e))

    @app.post("/index")
    def start_index(req: FolderRequest):
        if not os.path.isdir(req.folder):
            raise HTTPException(404, f"Folder not found: {req.folder}")
        engine.indexer.start(req.folder)
        return folder_status(req.folder)

    @app.get("/index/status")
    def index_status(folder: str = Query(...)):
        if not os.path.isdir(folder):
            raise HTTPException(404, f"Folder not found: {folder}")
        return folder_status(folder)

    @app.post("/sessions/file")
    def open_file_session(req: FileSessionRequest):
        path = os.path.abspath(req.file_path)
        if not os.path.isfile(path):
            raise HTTPException(404, f"File not found: {req.file_path}")
        if Path(path).suffix.lstrip(".").lower() not in SUPPORTED_EXTENSIONS:
            raise HTTPException(415, f"Unsupported file type. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
        info = engine.indexer.index_file(path)
        if info["status"] != "indexed":
            raise HTTPException(422, info.get("error") or f"Could not index this file ({info['status']}).")
        session = engine.history.create_session(req.username, "file", title=info["name"], file_path=path)
        return {"session": session, "file": {"name": info["name"], "pages": info["num_pages"],
                                             "chunks": info["num_chunks"]}}

    @app.post("/chat")
    def chat(req: ChatRequest):
        question = req.question.strip()
        if not question:
            raise HTTPException(400, "Question is empty.")

        if req.session_id:
            session = owned_session(req.session_id, req.username)
        else:
            if not req.folder:
                raise HTTPException(400, "Choose a folder first.")
            title = question if len(question) <= 60 else question[:57] + "…"
            session = engine.history.create_session(req.username, "folder", title=title,
                                                    folder=os.path.abspath(req.folder))

        if session["mode"] == "file":
            if not os.path.isfile(session["file_path"] or ""):
                raise HTTPException(404, "The file for this chat no longer exists.")
            engine.indexer.index_file(session["file_path"])  # picks up edits since last time
            scope = Scope(files=[session["file_path"]])
        else:
            folder = session.get("folder") or req.folder
            if not folder:
                raise HTTPException(400, "Choose a folder first.")
            if not session.get("folder"):  # chats saved by the old app had no folder
                engine.history.set_folder(session["id"], os.path.abspath(folder))
            if not os.path.isdir(folder):
                raise HTTPException(404, f"Folder not found: {folder}")
            summary = engine.store.folder_summary(folder)
            if summary["synced_at"] is None and not engine.indexer.is_busy(folder):
                engine.indexer.start(folder)
            if summary["files_indexed"] == 0 and engine.indexer.is_busy(folder):
                job = engine.indexer.job(folder)
                return {"session_id": session["id"], "status": "indexing", "citations": [],
                        "answer": f"I'm still indexing this folder ({job.processed}/{job.to_index or '?'} files). "
                                  "Ask again in a moment.",
                        "index": folder_status(folder)}
            scope = Scope(folder=folder)

        history = engine.history.recent_turns(session["id"])
        result = engine.answer(question, scope, history)
        engine.history.add_exchange(session["id"], question, result["answer"], result["citations"])
        return {"session_id": session["id"], **result}

    @app.get("/sessions")
    def list_sessions(username: str = "guest"):
        return {"sessions": engine.history.list_sessions(username)}

    @app.get("/sessions/{session_id}")
    def get_session(session_id: str, username: str = "guest"):
        session = owned_session(session_id, username)
        return {"session": session, "messages": engine.history.messages(session_id)}

    @app.delete("/sessions/{session_id}")
    def delete_session(session_id: str, username: str = "guest"):
        if not engine.history.delete_session(session_id, username):
            raise HTTPException(404, "Chat not found.")
        return {"status": "deleted"}

    return app


def _exit_when_parent_dies() -> None:
    """Electron keeps our stdin open; EOF means the app is gone, so don't linger as an orphan."""
    def watch():
        try:
            while sys.stdin.read(1024):
                pass
        except Exception:
            pass
        logger.info("Parent process closed stdin; exiting.")
        os._exit(0)

    threading.Thread(target=watch, daemon=True, name="parent-watchdog").start()


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv()  # development convenience; the packaged app gets settings from Electron
    settings = Settings()
    if getattr(sys, "frozen", False) and any(settings.model_dir.glob("models--*")):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")  # bundled models: never phone home
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    setup_logging(settings)
    if os.environ.get("DOCLAMAR_EXIT_ON_STDIN_EOF") == "1" and sys.stdin is not None:
        _exit_when_parent_dies()

    import uvicorn

    port = int(os.environ.get("DOCLAMAR_PORT", "8000"))
    logger.info("DocLAMAR %s starting on 127.0.0.1:%d (data: %s)", __version__, port, settings.home)
    uvicorn.run(create_app(Engine(settings)), host="127.0.0.1", port=port, log_config=None)


if __name__ == "__main__":
    main()
