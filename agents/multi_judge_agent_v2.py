import os
from dotenv import load_dotenv
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core.models import UserMessage
from autogen_ext.models.azure import AzureAIChatCompletionClient
from azure.core.credentials import AzureKeyCredential
from autogen_core.tools import FunctionTool
from autogen_core import CancellationToken

from tools.arxiv_search_tool import query_arxiv, query_web

load_dotenv()

azure_api_key = os.getenv("GITHUB_TOKEN")

# Azure GPT-4o Model client
client = AzureAIChatCompletionClient(
    model="gpt-4o",
    endpoint="https://models.inference.ai.azure.com",
    credential=AzureKeyCredential(azure_api_key),
    model_info={
        "json_output": True,
        "function_calling": True,
        "vision": False,
        "family": "unknown",
    },
)

# Tool Wrappers
arxiv_tool = FunctionTool(query_arxiv, description="Searches arXiv for research papers.")
web_tool = FunctionTool(query_web, description="Searches the web for relevant academic content.")

# Define Judge Agents (all evaluate across Relevance, Novelty, and Impact)
def create_judge_agent(name):
    return AssistantAgent(
        name=name,
        model_client=client,
        tools=[arxiv_tool, web_tool],
        system_message=(
            f"You are a judge agent named {name}. For each paper, you will provide three scores (1-10):\n"
            "- Relevance to the user's query\n"
            "- Novelty of the work\n"
            "- Impact potential (citation likelihood, significance)\n"
            "You must justify each score with 1-2 sentences. Use tools like arxiv or web search if needed."
        ),
        reflect_on_tool_use=True,
    )

judge1 = create_judge_agent("Judge_1")
judge2 = create_judge_agent("Judge_2")
judge3 = create_judge_agent("Judge_3")

# Define Final Judge
final_judge = AssistantAgent(
    name="Final_Judge",
    model_client=client,
    system_message="""
You are the final judge. You will receive multiple papers and their evaluations from 3 agents.
Each agent provides scores and justifications for Relevance, Novelty, and Impact.
Your job is to:
1. Aggregate the scores for each paper.
2. Highlight areas of agreement or disagreement.
3. Output a ranked list of papers based on overall quality and fit for recommendation.
""",
)

# Async runner for multi-agent evaluation
async def run_multi_judge_agents(papers: list[str], user_query: str):
    judge_results = {}
    for paper in papers:
        evaluations = {}
        for judge in [judge1, judge2, judge3]:
            response = await judge.on_messages(
                [TextMessage(content=f"User query: {user_query}\nPaper: {paper}", source="user")],
                cancellation_token=CancellationToken()
            )
            evaluations[judge.name] = response[-1].content
        judge_results[paper] = evaluations

    final_prompt = """You will receive evaluations for multiple papers. Each paper has been scored by three judges in three dimensions:
- Relevance (1-10)
- Novelty (1-10)
- Impact (1-10)

Each judge also provides justifications. Please:
1. Aggregate the scores.
2. Analyze agreement between judges.
3. Rank the papers and explain your final ranking."""

    final_input = str(judge_results)
    result = await final_judge.on_messages(
        [TextMessage(content=final_prompt + "\n\n" + final_input, source="user")],
        cancellation_token=CancellationToken()
    )
    return result[-1].content