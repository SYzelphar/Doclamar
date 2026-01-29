from crewai.tools import BaseTool
from typing import List

from doclamar.schemas.parsed_chunk import ParsedChunkSchema
from doclamar.schemas.result import FinalResultSchema
from doclamar.llm.llm_provider import get_llm


class SummarizerTool(BaseTool):
    name: str = "summarizer_tool"
    description: str = "Generate a concise summary from ranked document chunks"

    def _run(
        self,
        query: str,
        chunks: List[ParsedChunkSchema]
    ) -> FinalResultSchema:

        llm = get_llm()

        context = "\n\n".join(
            f"Source ({chunk.metadata.get('file_name')}):\n{chunk.content}"
            for chunk in chunks
        )

        prompt = f"""
You are a document analysis assistant.

User Query:
{query}

Relevant Document Content:
{context}

Task:
Provide a concise, accurate, and context-aware answer to the user's query.
Do not hallucinate information. Base your answer only on the provided content.
"""

        answer = llm.generate(prompt)

        source_files = list({
            chunk.metadata.get("file_path")
            for chunk in chunks
        })

        return FinalResultSchema(
            answer=answer,
            source_files=source_files
        )
