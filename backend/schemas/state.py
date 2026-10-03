from __future__ import annotations
from typing import Annotated, Any, Dict, List, Optional
from typing_extensions import TypedDict
import operator


class DoclamarState(TypedDict):
    query: str
    root_path: str
    top_k: int
    routing_plan: Optional[Dict[str, Any]]
    candidate_documents: Optional[List[Dict]]
    parsed_chunks: Optional[List[Dict]]
    retrieved_chunks: Optional[List[Dict]]
    reranked_chunks: Optional[List[Dict]]
    final_answer: Optional[str]
    source_files: Optional[List[str]]
    citations: Optional[List[Dict]]
    search_stats: Optional[Dict[str, Any]]
    evaluation: Optional[Dict[str, Any]]
    error: Optional[str]
    retry_count: Annotated[int, operator.add]
    node_timings: Optional[Dict[str, float]]
