from pydantic import BaseModel
from typing import Optional, List


class QuerySchema(BaseModel):
    query: str
    root_path: str
    file_types: Optional[List[str]] = None  # e.g. ["pdf", "docx"]
