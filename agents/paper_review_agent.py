# ============ agents/paper_review_agent.py ============
import os
from dotenv import load_dotenv
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core.models import UserMessage
from autogen_ext.models.azure import AzureAIChatCompletionClient
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from azure.core.credentials import AzureKeyCredential
from autogen_core.tools import FunctionTool
from autogen_core import CancellationToken

from tools.review_tools import review_dispatcher, summarize_pdf, enhanced_summary_web, visualize_summary

load_dotenv()

azure_api_key = os.getenv("GITHUB_TOKEN")
review_modes = ["rapid", "academic", "visual", "enhanced"]

# Azure GitHub Model client
# client = AzureAIChatCompletionClient(
#     endpoint="https://models.inference.ai.azure.com",
#     model = "Llama-3.3-70B-Instruct",
#     credential=AzureKeyCredential(azure_api_key),
#     max_tokens= 4096,
#     model_info={
#         "json_output": True,
#         "function_calling": True,
#         "vision": False,
#         "family": "unknown"}
# )

api_key = os.getenv("OAI_KEY")
api_endpoint = os.getenv("OAI_ENDPOINT")

# Azure GitHub Model client
client = AzureOpenAIChatCompletionClient(
    api_key=api_key,
    azure_endpoint=api_endpoint,
    model="gpt-4o",
    api_version="2024-05-13",
    model_info={
        "json_output": True,
        "function_calling": True,
        "vision": False,
        "family": "unknown",
    },
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
    system_message="""You are a research assistant that summarizes and analyzes academic papers using multiple review modes including visual and enhanced summaries.
    
    IMPORTANT: For every paper review task, use a structured Chain of Thought (CoT) reasoning approach:
    
    1. UNDERSTAND: First, identify the type of paper and what aspects the user needs analyzed.
    2. PLAN: Determine which review mode(s) would be most appropriate (rapid, academic, visual, enhanced).
    3. ANALYZE: Describe your approach to breaking down the paper (sections, methodology, results, implications).
    4. EXTRACT: Explain what key information you're looking for and why it matters.
    5. SYNTHESIZE: Show how you're combining different elements into a coherent review.
    6. REFLECT: Consider limitations of your analysis and alternative interpretations.
    
    Begin each review with "💭 Review Approach: [your chain of thought]" before providing the actual review.
    """,
    reflect_on_tool_use=True,
)

# Async wrapper for Chainlit/other UI
async def run_review_agent(user_input: str):
    return await paper_review_agent.on_messages(
        [TextMessage(content=user_input, source="user")],
        cancellation_token=CancellationToken()
    )
