from crewai import Agent
from doclamar.tools.routing_tool import RoutingTool
from doclamar.tools.query_hint_tool import QueryHintTool


def create_routing_agent():
    return Agent(
        role="Routing Agent",
        goal="Identify candidate files using semantic understanding of the user query",
        backstory=(
            "You analyze the user query to understand the intent and key concepts, "
            "then use tools to efficiently route to relevant files."
        ),
        tools=[
            QueryHintTool(),
            RoutingTool()
        ],
        
        verbose=True,
    )
