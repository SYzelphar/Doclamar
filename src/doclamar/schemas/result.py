from pydantic import BaseModel
from typing import List


class FinalResultSchema(BaseModel):
    answer: str
    source_files: List[str]
