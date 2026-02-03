'''
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

@CrewBase
class Doclamar():
    """Doclamar crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
    
    # If you would like to add tools to your agents, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools
    @agent
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['researcher'], # type: ignore[index]
            verbose=True
        )

    @agent
    def reporting_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['reporting_analyst'], # type: ignore[index]
            verbose=True
        )

    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task
    @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config['research_task'], # type: ignore[index]
        )

    @task
    def reporting_task(self) -> Task:
        return Task(
            config=self.tasks_config['reporting_task'], # type: ignore[index]
            output_file='report.md'
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Doclamar crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )'''


from crewai import Crew, Task

from doclamar.agents.routing_agent import create_routing_agent
from doclamar.agents.parsing_agent import create_parsing_agent
from doclamar.agents.rerank_agent import create_rerank_agent
from doclamar.agents.summarizer_agent import create_summarizer_agent


def create_crew():
    # Agents
    routing_agent = create_routing_agent()
    parsing_agent = create_parsing_agent()
    rerank_agent = create_rerank_agent()
    summarizer_agent = create_summarizer_agent()

    # Tasks
    routing_task = Task(
        description=(
            "Traverse the directory provided by the user and return candidate files "
            "that may contain relevant information."
        ),
        expected_output="A list of candidate document file paths",
        agent=routing_agent,
    )

    parsing_task = Task(
        description=(
            "Extract and chunk textual content from the routed documents "
            "in a structured format."
        ),
        expected_output="A list of parsed document chunks",
        agent=parsing_agent,
    )

    rerank_task = Task(
        description=(
            "Rank the parsed document chunks based on relevance to the user query "
            "and select the most relevant ones."
        ),
        expected_output="Top-k ranked document chunks",
        agent=rerank_agent,
    )

    summarization_task = Task(
        description=(
            "Generate a concise and accurate summary based on the ranked document chunks."
        ),
        expected_output="Final summarized answer with source references",
        agent=summarizer_agent,
    )

    # Crew
    crew = Crew(
        agents=[
            routing_agent,
            parsing_agent,
            rerank_agent,
            summarizer_agent,
        ],
        tasks=[
            routing_task,
            parsing_task,
            rerank_task,
            summarization_task,
        ],
        verbose=True,
        max_rpm=10
    )

    return crew
