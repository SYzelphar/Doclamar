from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="DocLAMAR API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_graph = None
_chat_sessions: Dict[str, Any] = {}


def get_graph():
    global _graph
    if _graph is None:
        from graph.builder import build_graph
        _graph = build_graph()
    return _graph


class QueryRequest(BaseModel):
    query: str
    root_path: str
    top_k: int = 5


class QueryResponse(BaseModel):
    query: str
    answer: str
    source_files: List[str]
    citations: Optional[List[Dict]] = None
    search_stats: Optional[Dict[str, Any]] = None
    evaluation: Optional[Dict[str, Any]] = None
    node_timings: Optional[Dict[str, float]] = None
    retry_count: int
    error: Optional[str] = None


class ChatLoadRequest(BaseModel):
    file_path: str


class ChatMessageRequest(BaseModel):
    session_id: str
    message: str
    top_k: int = 5


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    citations: List[Dict]
    chunks_used: int


@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    if not request.root_path.strip():
        raise HTTPException(status_code=400, detail="root_path cannot be empty.")

    graph = get_graph()
    initial_state = {
        "query": request.query,
        "root_path": request.root_path,
        "top_k": request.top_k,
        "routing_plan": None,
        "candidate_documents": None,
        "parsed_chunks": None,
        "retrieved_chunks": None,
        "reranked_chunks": None,
        "final_answer": None,
        "source_files": None,
        "citations": None,
        "search_stats": None,
        "evaluation": None,
        "error": None,
        "retry_count": 0,
        "node_timings": {},
    }

    result = graph.invoke(initial_state)

    return QueryResponse(
        query=request.query,
        answer=result.get("final_answer") or "No answer generated.",
        source_files=result.get("source_files") or [],
        citations=result.get("citations"),
        search_stats=result.get("search_stats"),
        evaluation=result.get("evaluation"),
        node_timings=result.get("node_timings"),
        retry_count=result.get("retry_count", 0),
        error=result.get("error"),
    )


@app.post("/chat/load")
async def load_document(request: ChatLoadRequest):
    from chat.document_chat import load_document as _load
    import uuid
    try:
        session = _load(request.file_path)
        session_id = str(uuid.uuid4())
        _chat_sessions[session_id] = session
        return {
            "session_id": session_id,
            "file_name": session.file_name,
            "file_size_kb": session.file_size_kb,
            "total_chunks": session.total_chunks,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/chat/message", response_model=ChatResponse)
async def chat_message(request: ChatMessageRequest):
    from chat.document_chat import chat as _chat
    session = _chat_sessions.get(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found. Load a document first.")
    result = _chat(session, request.message, top_k=request.top_k)
    return ChatResponse(
        session_id=request.session_id,
        answer=result["answer"],
        citations=result["citations"],
        chunks_used=result["chunks_used"],
    )


@app.delete("/chat/{session_id}")
async def end_session(session_id: str):
    if session_id in _chat_sessions:
        del _chat_sessions[session_id]
    return {"status": "session ended"}


@app.get("/health")
async def health():
    return {"status": "ok", "version": "2.0.0"}
