# ============ agents/paper_review_agent.py ============
import os
from dotenv import load_dotenv
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core.models import UserMessage
from autogen_ext.models.azure import AzureAIChatCompletionClient
from azure.core.credentials import AzureKeyCredential
from autogen_core.tools import FunctionTool
from autogen_core import CancellationToken

from tools.review_tools import review_dispatcher, summarize_pdf, enhanced_summary_web, visualize_summary
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient

from autogen_agentchat.base import Response
from autogen_agentchat.messages import ModelClientStreamingChunkEvent, ToolCallRequestEvent

load_dotenv()

azure_api_key = os.getenv("GITHUB_TOKEN")
api_key = os.getenv("OAI_KEY")
api_endpoint = os.getenv("OAI_ENDPOINT")

review_modes = ["rapid", "academic", "visual", "enhanced"]

# Azure GitHub Model client
client = AzureOpenAIChatCompletionClient(
    #endpoint="https://models.inference.ai.azure.com",
    azure_endpoint=api_endpoint,
    model = "gpt-4o",
    #credential=AzureKeyCredential(azure_api_key),
    api_key=api_key,
    api_version="2024-05-13",
    max_tokens= 4096,
    model_info={
        "json_output": True,
        "function_calling": True,
        "vision": False,
        "family": "unknown"}
)

# Register individual tools (optional if directly accessible)
summarize_tool = FunctionTool(
    summarize_pdf,
    description="Summarizes a given PDF file in either rapid or academic format."
)

enhance_tool = FunctionTool(
    enhanced_summary_web,
    description="Enhances the summary using relevant web results."
)

visualize_tool = FunctionTool(
    visualize_summary,
    description="Visualizes keyword frequency from summary content."
)

# Dispatcher tool that selects review mode
review_mode_tool = FunctionTool(
    review_dispatcher,
    description="Dispatcher tool to summarize, visualize, or enhance the review of a research paper."
)

# Define Paper Review Agent
paper_review_agent = AssistantAgent(
    name="PaperReviewAgent",
    model_client=client,
    tools=[review_mode_tool, summarize_tool, enhance_tool, visualize_tool],
    system_message="You are a research assistant that summarizes and analyzes academic papers using multiple review modes including visual and enhanced summaries.",
    reflect_on_tool_use=True,
    model_client_stream=True,
)

# Async wrapper for Chainlit/other UI
async def run_review_agent(user_input: str):
    return await paper_review_agent.on_messages(
        [TextMessage(content=user_input, source="user")],
        cancellation_token=CancellationToken()
    )

async def run_review_agent_stream(user_input: str):
    return paper_review_agent.on_messages_stream(
        [TextMessage(content=user_input, source="user")],
        cancellation_token=CancellationToken()
    )
