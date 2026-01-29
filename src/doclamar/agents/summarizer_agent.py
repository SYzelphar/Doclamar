from crewai import Agent
from doclamar.tools.summarizer_tool import SummarizerTool


def create_summarizer_agent():
    return Agent(
        role="Summarizing Agent",
        goal="Generate concise and accurate summaries from ranked document content",
        backstory=(
            "You synthesize relevant document information into a clear, "
            "user-friendly response."
        ),
        tools=[SummarizerTool()],
        verbose=True,
    )
