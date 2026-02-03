from crewai.tools import BaseTool
from typing import List, Optional
import os

from doclamar.schemas.document import DocumentSchema


class RoutingTool(BaseTool):
    name: str = "routing_tool"
    description: str = "Traverse directories and return candidate files using semantic hints"

    def _run(
        self,
        root_path= "C:\\Users",
        semantic_hints: Optional[List[str]] = None,
        allowed_extensions: Optional[List[str]] = None
    ) -> List[DocumentSchema]:

        results: List[DocumentSchema] = []

        for root, _, files in os.walk(root_path):
            for file in files:
                ext = file.split(".")[-1].lower()

                if allowed_extensions and ext not in allowed_extensions:
                    continue

                full_path = os.path.join(root, file)
                searchable_text = f"{full_path.lower()} {file.lower()}"

                if semantic_hints:
                    if not any(hint in searchable_text for hint in semantic_hints):
                        continue

                size_kb = round(os.path.getsize(full_path) / 1024, 2)

                results.append(
                    DocumentSchema(
                        file_path=full_path,
                        file_name=file,
                        file_type=ext,
                        size_kb=size_kb
                    )
                )

        return results
