import os

from crewai import LLM, Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import SerperDevTool
from pydantic import BaseModel, Field
from typing import List
from .tools.push_tool import PushNotificationTool

# Force LiteLLM routing for Bedrock. CrewAI 1.14.1's native Bedrock Converse
# provider has a tool-use serialization bug that causes Claude models to emit
# empty `{}` as the `input` for every tool call. LiteLLM's bedrock handler
# serializes the tool schema correctly, so Claude fills in the required args.
_BEDROCK_REGION = os.getenv("AWS_REGION", "ap-southeast-1")

WORKER_LLM = LLM(
    model="bedrock/apac.anthropic.claude-3-haiku-20240307-v1:0",
    aws_region_name=_BEDROCK_REGION,
    is_litellm=True,
)

REASONING_LLM = LLM(
    model="bedrock/apac.anthropic.claude-3-5-sonnet-20241022-v2:0",
    aws_region_name=_BEDROCK_REGION,
    is_litellm=True,
)

class TrendingCompany(BaseModel):
    """ A company that is in the news and attracting attention """
    name: str = Field(description="Company name")
    ticker: str = Field(description="Stock ticker symbol")
    reason: str = Field(description="Reason this company is trending in the news")

class TrendingCompanyList(BaseModel):
    """ List of multiple trending companies that are in the news """
    companies: List[TrendingCompany] = Field(description="List of companies trending in the news")

class TrendingCompanyResearch(BaseModel):
    """ Detailed research on a company """
    name: str = Field(description="Company name")
    market_position: str = Field(description="Current market position and competitive analysis")
    future_outlook: str = Field(description="Future outlook and growth prospects")
    investment_potential: str = Field(description="Investment potential and suitability for investment")

class TrendingCompanyResearchList(BaseModel):
    """ A list of detailed research on all the companies """
    research_list: List[TrendingCompanyResearch] = Field(description="Comprehensive research on all trending companies")

@CrewBase
class StockPicker1():
    """StockPicker1 crew"""

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'
   

    
    @agent
    def trending_company_finder(self) -> Agent:
        return Agent(config=self.agents_config['trending_company_finder'],
                     tools=[SerperDevTool()],
                     llm=WORKER_LLM)

    @agent
    def financial_researcher(self) -> Agent:
        return Agent(config=self.agents_config['financial_researcher'],
                     tools=[SerperDevTool()],
                     llm=WORKER_LLM)

    @agent
    def stock_picker(self) -> Agent:
        return Agent(config=self.agents_config['stock_picker'],
                     tools=[PushNotificationTool()],
                     llm=REASONING_LLM)

    @task
    def find_trending_companies(self) -> Task:
        return Task(config=self.tasks_config['find_trending_companies'])

    @task
    def research_trending_companies(self) -> Task:
        return Task(config=self.tasks_config['research_trending_companies'])

    @task
    def pick_best_company(self) -> Task:
        return Task(config=self.tasks_config['pick_best_company'])

    @crew
    def crew(self) -> Crew:
        """Creates the StockPicker1 crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            verbose=True,
            process=Process.sequential,
            memory=False,
        )
