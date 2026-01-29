from crewai.tools import BaseTool
from typing import List
import PyPDF2

from doclamar.schemas.document import DocumentSchema
from doclamar.schemas.parsed_chunk import ParsedChunkSchema


class ParsingTool(BaseTool):
    name: str = "parsing_tool"
    description: str = "Extract and chunk text from documents"

    def _run(
        self,
        documents: List[DocumentSchema],
        chunk_size: int = 500
    ) -> List[ParsedChunkSchema]:

        parsed_chunks: List[ParsedChunkSchema] = []

        for doc in documents:
            text = ""

            if doc.file_type == "pdf":
                try:
                    with open(doc.file_path, "rb") as f:
                        reader = PyPDF2.PdfReader(f)
                        for page in reader.pages:
                            text += page.extract_text() or ""
                except Exception:
                    continue

            elif doc.file_type == "txt":
                try:
                    with open(doc.file_path, "r", encoding="utf-8", errors="ignore") as f:
                        text = f.read()
                except Exception:
                    continue
            else:
                continue

            for i in range(0, len(text), chunk_size):
                chunk = text[i:i + chunk_size].strip()
                if chunk:
                    parsed_chunks.append(
                        ParsedChunkSchema(
                            content=chunk,
                            metadata={
                                "file_path": doc.file_path,
                                "file_name": doc.file_name,
                                "file_type": doc.file_type
                            }
                        )
                    )

        return parsed_chunks
