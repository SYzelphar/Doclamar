from crewai.tools import BaseTool
from typing import List
from doclamar.llm.llm_provider import get_llm


class QueryHintTool(BaseTool):
    name: str = "query_hint_tool"
    description: str = "Extract semantic keywords and topics from a user query"

    def _run(self, query: str) -> List[str]:
        llm = get_llm()

        prompt = f"""
Extract important keywords and related concepts from the following user query.
Return them as a comma-separated list.
Do NOT explain anything.

Query:
{query}
"""

        response = llm.generate(prompt)

        keywords = [
            k.strip().lower()
            for k in response.split(",")
            if k.strip()
        ]

        return keywords
