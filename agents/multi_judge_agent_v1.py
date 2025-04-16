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

# Define Multi-Judge Agents with tool access
judge_relevance = AssistantAgent(
    name="Judge_Relevance",
    model_client=client,
    tools=[arxiv_tool, web_tool],
    system_message="You are a judge who evaluates how relevant a paper is to the user's query. You may search for supporting information using the provided tools. Score from 1-10 and explain your reasoning.",
    reflect_on_tool_use=True,
)

judge_novelty = AssistantAgent(
    name="Judge_Novelty",
    model_client=client,
    tools=[arxiv_tool, web_tool],
    system_message="You are a judge who evaluates the novelty of a paper. You may search for similar or related work using the provided tools. Score from 1-10 and justify based on originality of idea and method.",
    reflect_on_tool_use=True,
)

judge_impact = AssistantAgent(
    name="Judge_Impact",
    model_client=client,
    tools=[arxiv_tool, web_tool],
    system_message="You are a judge who evaluates the research impact of a paper. Consider citation potential and field relevance. Use tools to look up context if needed. Score from 1-10 and justify.",
    reflect_on_tool_use=True,
)

# Define Final Judge
final_judge = AssistantAgent(
    name="Final_Judge",
    model_client=client,
    system_message="You are a final judge. Given scores and reasons from three agents, aggregate them into a ranked list of papers. Highlight consensus and summarize key strengths.",
)

# Async runner for multi-agent evaluation
async def run_multi_judge_agents(papers: list[str], user_query: str):
    # Step 1: Judge agents independently evaluate
    judge_results = {}
    for paper in papers:
        evaluations = {}
        for judge in [judge_relevance, judge_novelty, judge_impact]:
            response = await judge.on_messages(
                [TextMessage(content=f"User query: {user_query}\nPaper: {paper}", source="user")],
                cancellation_token=CancellationToken()
            )
            evaluations[judge.name] = response[-1].content  # Latest message
        judge_results[paper] = evaluations

    # Step 2: Final judge aggregates
    final_prompt = """You will receive multiple papers with evaluations from 3 judges: Judge_Relevance, Judge_Novelty, and Judge_Impact.
Each judge provides a score (1-10) and a justification. Your job is to output a ranked list of papers with explanations.

Evaluation format:
{
  "paper_title": {
    "Judge_Relevance": "...",
    "Judge_Novelty": "...",
    "Judge_Impact": "..."
  },
  ...
}

Now aggregate the evaluations and rank the papers, provide a summary for the users."""
    final_input = str(judge_results)
    result = await final_judge.on_messages(
        [TextMessage(content=final_prompt + "\n\n" + final_input, source="user")],
        cancellation_token=CancellationToken()
    )
    return result[-1].content
