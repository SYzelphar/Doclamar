from pydantic import BaseModel
from typing import Optional


class DocumentSchema(BaseModel):
    file_path: str
    file_name: str
    file_type: str
    size_kb: Optional[float] = None
