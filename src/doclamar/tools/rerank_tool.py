from crewai.tools import BaseTool
from typing import List

from doclamar.schemas.parsed_chunk import ParsedChunkSchema
from doclamar.vectorstore.chroma_manager import ChromaManager


class ReRankTool(BaseTool):
    name: str = "rerank_tool"
    description: str = "Rank document chunks based on semantic similarity"

    def _run(
        self,
        query: str,
        chunks: List[ParsedChunkSchema],
        top_k: int = 5
    ) -> List[ParsedChunkSchema]:

        chroma = ChromaManager()
        chroma.add_chunks(chunks)

        results = chroma.query(query, top_k=top_k)
        ranked_chunks: List[ParsedChunkSchema] = []

        if results and "documents" in results:
            for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
                ranked_chunks.append(
                    ParsedChunkSchema(
                        content=doc,
                        metadata=meta
                    )
                )

        return ranked_chunks
