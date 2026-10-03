from pydantic import BaseModel, Field
from typing import Dict, List, Optional


class DocumentSchema(BaseModel):
    file_path: str
    file_name: str
    file_type: str
    size_kb: Optional[float] = None


class ParsedChunkSchema(BaseModel):
    content: str
    metadata: Dict[str, str]
    score: Optional[float] = None


class RoutingPlanSchema(BaseModel):
    keywords: List[str]
    allowed_extensions: List[str]
    max_files: int
    use_content_preview: bool
    reasoning: str


class FinalResultSchema(BaseModel):
    answer: str
    source_files: List[str]
    chunks_used: int
    retrieval_scores: List[float] = Field(default_factory=list)
