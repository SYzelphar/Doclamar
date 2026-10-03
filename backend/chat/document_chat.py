from __future__ import annotations
import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from schemas.models import ParsedChunkSchema
from vectorstore.faiss_store import FAISSStore
from utils.llm_provider import get_llm

logger = logging.getLogger(__name__)

CHUNK_SIZE = 600
CHUNK_OVERLAP = 100
MAX_CONTEXT_CHARS = 5000
MAX_HISTORY_TURNS = 6


@dataclass
class ChatMessage:
    role: str
    content: str


@dataclass
class DocumentChatSession:
    file_path: str
    file_name: str
    file_type: str
    chunks: List[ParsedChunkSchema] = field(default_factory=list)
    store: Optional[FAISSStore] = None
    history: List[ChatMessage] = field(default_factory=list)
    total_chunks: int = 0
    file_size_kb: float = 0.0


def _read_file(file_path: str) -> str:
    ext = Path(file_path).suffix.lstrip(".").lower()
    try:
        if ext == "pdf":
            import pypdf
            text = ""
            with open(file_path, "rb") as fh:
                reader = pypdf.PdfReader(fh)
                for page in reader.pages:
                    text += (page.extract_text() or "")
            return text.strip()
        elif ext == "txt":
            with open(file_path, "r", encoding="utf-8", errors="ignore") as fh:
                return fh.read()
        elif ext == "docx":
            from docx import Document
            doc = Document(file_path)
            return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        else:
            raise ValueError(f"Unsupported file type: {ext}")
    except Exception as e:
        raise RuntimeError(f"Failed to read {file_path}: {e}")


def _chunk_text(text: str, file_path: str, file_name: str, file_type: str) -> List[ParsedChunkSchema]:
    chunks = []
    stride = CHUNK_SIZE - CHUNK_OVERLAP
    idx = 0
    for i in range(0, len(text), stride):
        chunk_text = text[i: i + CHUNK_SIZE].strip()
        if not chunk_text:
            continue
        chunks.append(ParsedChunkSchema(
            content=chunk_text,
            metadata={
                "file_path": file_path,
                "file_name": file_name,
                "file_type": file_type,
                "chunk_index": str(idx),
                "chunk_start": str(i),
            }
        ))
        idx += 1
    return chunks


def _build_context(results: List[Tuple[ParsedChunkSchema, float]]) -> Tuple[str, List[Dict]]:
    parts = []
    citations = []
    total = 0

    for i, (chunk, score) in enumerate(results):
        ref_id = i + 1
        chunk_idx = chunk.metadata.get("chunk_index", "?")
        block = f"[{ref_id}] (section {chunk_idx}, relevance {score:.3f})\n{chunk.content}"

        if total + len(block) > MAX_CONTEXT_CHARS:
            remaining = MAX_CONTEXT_CHARS - total
            if remaining > 150:
                parts.append(block[:remaining])
            break

        parts.append(block)
        total += len(block)
        citations.append({
            "ref_id": ref_id,
            "chunk_index": chunk_idx,
            "score": round(score, 4),
            "preview": chunk.content[:120].replace("\n", " "),
        })

    return "\n\n---\n\n".join(parts), citations


def _format_history(history: List[ChatMessage]) -> str:
    recent = history[-(MAX_HISTORY_TURNS * 2):]
    lines = []
    for msg in recent:
        role = "User" if msg.role == "user" else "Assistant"
        lines.append(f"{role}: {msg.content}")
    return "\n".join(lines)


def load_document(file_path: str) -> DocumentChatSession:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    file_name = Path(file_path).name
    file_type = Path(file_path).suffix.lstrip(".").lower()
    file_size_kb = round(os.path.getsize(file_path) / 1024, 2)

    logger.info(f"[DocumentChat] Loading: {file_name} ({file_size_kb} KB)")

    text = _read_file(file_path)
    if not text:
        raise ValueError(f"No text could be extracted from {file_name}")

    chunks = _chunk_text(text, file_path, file_name, file_type)
    store = FAISSStore()
    store.build(chunks)

    session = DocumentChatSession(
        file_path=file_path,
        file_name=file_name,
        file_type=file_type,
        chunks=chunks,
        store=store,
        total_chunks=len(chunks),
        file_size_kb=file_size_kb,
    )

    logger.info(f"[DocumentChat] Ready — {len(chunks)} chunks indexed from {file_name}")
    return session


def chat(session: DocumentChatSession, user_message: str, top_k: int = 5) -> Dict:
    results = session.store.search(user_message, top_k=top_k)

    if not results:
        reply = "I could not find relevant content in this document to answer your question."
        session.history.append(ChatMessage(role="user", content=user_message))
        session.history.append(ChatMessage(role="assistant", content=reply))
        return {"answer": reply, "citations": [], "chunks_used": 0}

    context, citations = _build_context(results)
    history_text = _format_history(session.history)

    system = (
        f"You are a document assistant helping the user understand: {session.file_name}. "
        "Answer strictly from the document content provided. "
        "Do NOT use external knowledge. "
        "Cite sections inline using [N] reference numbers. "
        "Maintain context from the conversation history."
    )

    history_block = f"Conversation history:\n{history_text}\n\n" if history_text else ""

    prompt = (
        f"{history_block}"
        f"Document excerpts from {session.file_name}:\n"
        f"{context}\n\n"
        f"User: {user_message}\n\n"
        "Instructions:\n"
        "- Answer using only the document content above\n"
        "- Cite inline using [N] references\n"
        "- Be conversational but precise\n"
        "- If the document does not cover the question, say so clearly\n\n"
        "Assistant:"
    )

    llm = get_llm()
    answer = llm.generate(prompt, system=system).strip()

    session.history.append(ChatMessage(role="user", content=user_message))
    session.history.append(ChatMessage(role="assistant", content=answer))

    return {
        "answer": answer,
        "citations": citations,
        "chunks_used": len(results),
    }


def clear_history(session: DocumentChatSession):
    session.history = []
    logger.info(f"[DocumentChat] History cleared for {session.file_name}")
