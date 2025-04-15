# ============ agents/qa_agent.py ============
import os
from dotenv import load_dotenv
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core.models import UserMessage
from autogen_ext.models.azure import AzureAIChatCompletionClient
from azure.core.credentials import AzureKeyCredential
from autogen_core.tools import FunctionTool
from autogen_core import CancellationToken

from tools.qa_tools import answer_from_context, explain_concept

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

# Define tools for Q&A
context_answer_tool = FunctionTool(
    answer_from_context,
    description="Answer questions based on previously reviewed paper content."
)

concept_explanation_tool = FunctionTool(
    explain_concept,
    description="Explain a specific academic or technical concept in detail."
)

# Define Q&A Agent
qa_agent = AssistantAgent(
    name="QAAssistantAgent",
    model_client=client,
    tools=[context_answer_tool, concept_explanation_tool],
    system_message="You are a Q&A assistant that answers questions based on prior paper reviews and explains technical concepts clearly.",
    reflect_on_tool_use=True,
)

# Async wrapper for UI integration
async def run_qa_agent(user_input: str):
    return await qa_agent.on_messages(
        [TextMessage(content=user_input, source="user")],
        cancellation_token=CancellationToken()
    )
