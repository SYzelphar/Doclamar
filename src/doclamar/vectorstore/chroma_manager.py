import chromadb
from chromadb.config import Settings
from typing import List
from doclamar.schemas.parsed_chunk import ParsedChunkSchema


class ChromaManager:
    def __init__(self, collection_name: str = "documents"):
        self.client = chromadb.Client(
            Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name
        )

    def add_chunks(self, chunks: List[ParsedChunkSchema]):
        for idx, chunk in enumerate(chunks):
            self.collection.add(
                documents=[chunk.content],
                metadatas=[chunk.metadata],
                ids=[f"doc_{idx}"]
            )

    def query(self, query: str, top_k: int = 5):
        return self.collection.query(
            query_texts=[query],
            n_results=top_k
        )
