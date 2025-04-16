import asyncio
import os
import json
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
    model="Llama-3.3-70B-Instruct",
    endpoint="https://models.inference.ai.azure.com",
    credential=AzureKeyCredential(azure_api_key),
    max_tokens=4096,
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
async def run_multi_judge_agents(user_query: str):
    # Each judge independently retrieves and evaluates papers using their internal prompt and tools
    judge_agents = [judge1, judge2, judge3]
    judge_outputs = {}

    for judge in judge_agents:
        # Each agent is expected to handle its own tool calling and reasoning
        print(f"Invoking {judge.name}...")
        result = await judge.on_messages(
            [TextMessage(content=user_query, source="user")],
            cancellation_token=CancellationToken()
        )
        # The last message should contain a structured list of paper evaluations
        judge_outputs[judge.name] = result.chat_message.content

        # Add delay to avoid triggering rate limit
        await asyncio.sleep(65)  # Retry-After: 60 + buffer

    # Prepare final aggregation prompt for Final Judge
    aggregation_prompt = (
        "You are the final judge aggregating independent evaluations from three judge agents.\n"
        "Each judge returns a list of evaluated papers with scores in Relevance, Novelty, and Impact (1-10), "
        "and short justifications.\n\n"
        "Your tasks:\n"
        "1. Identify and merge duplicate papers across judges (if any).\n"
        "2. Aggregate scores (average or weighted).\n"
        "3. Summarize agreement/disagreement.\n"
        "4. Rank papers by overall quality and fitness for recommendation.\n\n"
        f"Here is the input JSON:\n{json.dumps(judge_outputs)}"
    )

    print(f"Invoking Final Judge...")
    final_result = await final_judge.on_messages(
        [TextMessage(content=aggregation_prompt, source="user")],
        cancellation_token=CancellationToken()
    )
    return final_result.chat_message.content

