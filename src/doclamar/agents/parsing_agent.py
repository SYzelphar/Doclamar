from crewai import Agent
from doclamar.tools.parsing_tool import ParsingTool


def create_parsing_agent():
    return Agent(
        role="Parsing Agent",
        goal="Extract and structure content from documents",
        backstory=(
            "You extract meaningful text from documents identified by the routing agent. "
            "You do not rank or summarize content."
        ),
        llm=None,
        tools=[ParsingTool()],
        verbose=True,
    )
