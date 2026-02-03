from crewai import Agent
from doclamar.tools.rerank_tool import ReRankTool


def create_rerank_agent():
    return Agent(
        role="Re-Ranking Agent",
        goal="Rank parsed document content based on relevance to the user query",
        backstory=(
            "You prioritize extracted document content using semantic similarity. "
            "You do not summarize or generate new information."
        ),
        llm=None,
        tools=[ReRankTool()],
        verbose=True,
    )
