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

# Azure GitHub Model client
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

# Wrap arxiv/web search tools
arxiv_tool = FunctionTool(query_arxiv, description="Searches arXiv for research papers.")
web_tool = FunctionTool(query_web, description="Searches the web for relevant academic content.")

# Define agent
literature_assistant = AssistantAgent(
    name="LiteratureCollectionAgent",
    model_client=client,
    tools=[arxiv_tool, web_tool],
    system_message="You are a research assistant who can search academic databases and summarize results for the user.",
    reflect_on_tool_use=True,
)

# Async runner wrapper
async def run_literature_agent(user_input: str):
    return await literature_assistant.on_messages(
        [TextMessage(content=user_input, source="user")],
        cancellation_token=CancellationToken()
    )
