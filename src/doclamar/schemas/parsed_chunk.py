from pydantic import BaseModel
from typing import Dict


class ParsedChunkSchema(BaseModel):
    content: str
    metadata: Dict[str, str]
