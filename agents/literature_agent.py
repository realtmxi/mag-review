import os
from typing import AsyncGenerator
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core.models import UserMessage
from autogen_ext.models.azure import AzureAIChatCompletionClient
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from azure.core.credentials import AzureKeyCredential
from autogen_core.tools import FunctionTool
from autogen_core import CancellationToken
from tools.arxiv_search_tool import query_arxiv, query_web



load_dotenv()

azure_api_key = os.getenv("GITHUB_TOKEN") 
azure_endpoint = os.getenv("AZURE_INFERENCE_ENDPOINT", "https://models.inference.ai.azure.com") # e.g., "https://models.inference.ai.azure.com"
model_name = os.getenv("LITERATURE_AGENT_MODEL", "gpt-4o") # Default to gpt-4o if not set

# # Azure GitHub Model client
# client = AzureAIChatCompletionClient(
#     model="gpt-4o",
#     endpoint="https://models.inference.ai.azure.com",
#     credential=AzureKeyCredential(azure_api_key),
#     model_info={
#         "json_output": True,
#         "function_calling": True,
#         "vision": False,
#         "family": "unknown",
#         "structured_output": True
#     },
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
# Wrap arxiv/web search tools
arxiv_tool = FunctionTool(query_arxiv, description="Searches arXiv for research papers.")
web_tool = FunctionTool(query_web, description="Searches the web for relevant academic content.")

# Define agent
literature_assistant = AssistantAgent(
    name="LiteratureCollectionAgent",
    model_client=client,
    tools=[arxiv_tool, web_tool],
    system_message="""You are a research assistant who can search academic databases and summarize results for the user.
    
    IMPORTANT: For every query, use a structured Chain of Thought (CoT) reasoning approach:
    
    1. UNDERSTAND: First, explicitly interpret what the user is asking for. Define key search terms and objectives.
    2. PLAN: Outline a clear research strategy - which tools to use (arxiv_tool, web_tool), in what order, and why.
    3. SEARCH: Execute searches with well-formulated queries, explaining your choice of search parameters.
    4. ANALYZE: Examine search results critically, explaining how you're evaluating relevance and quality.
    5. SYNTHESIZE: Combine and structure your findings into a coherent response.
    6. CONCLUDE: Summarize key takeaways and suggest potential next steps for deeper research.
    
    Keep your reasoning transparent and numbered throughout each step. Format this as "💭 Reasoning: [your chain of thought]" before providing the final result.
    """,
    reflect_on_tool_use=True,
    model_client_stream=True
)

# Async runner wrapper with proper token streaming
async def run_literature_agent_stream(user_input: str) -> AsyncGenerator[str, None]:
    stream = literature_assistant.on_messages_stream(
        [TextMessage(content=user_input, source="user")],
        cancellation_token=CancellationToken()
    )
    
    # Yield a loader indicator
    yield "⏳ Thinking..."
    
    # Track the tools being used
    announced_tools = set()  
    result_shown = False
    
    async for chunk_event in stream:
        # If it's a string (direct content chunk)
        if isinstance(chunk_event, str):
            yield chunk_event
            
        # If it has a content attribute
        elif hasattr(chunk_event, 'content'):
            # Check for tool calls
            if isinstance(chunk_event.content, list):
                for function_call in chunk_event.content:
                    if hasattr(function_call, 'name'):
                        tool_name = function_call.name
                        
                        # Only announce a tool if we haven't announced it yet
                        if tool_name not in announced_tools:
                            announced_tools.add(tool_name)
                            
                            # Announce the tool being used
                            yield f"\n\n🔍 **Using tool: {tool_name}**\n"
                            
                            # Add function arguments display
                            if hasattr(function_call, 'arguments') and function_call.arguments:
                                try:
                                    import json
                                    # Parse the arguments if they're a string containing JSON
                                    if isinstance(function_call.arguments, str):
                                        try:
                                            args_obj = json.loads(function_call.arguments)
                                            if args_obj == {} or not args_obj:
                                                continue
                                            args_formatted = json.dumps(args_obj, indent=2)
                                        except:
                                            args_formatted = json.dumps(function_call.arguments, indent=2)
                                    else:
                                        args_formatted = json.dumps(function_call.arguments, indent=2)
                                    
                                    yield f"\n\n📋 **Tool call arguments:**\n\n```json\n{args_formatted}\n```\n\n"
                                except Exception as e:
                                     yield f"\n\n❌ **Error displaying arguments:** {str(e)}\n\n"
                            else:
                                # Add an extra newline for consistent spacing
                                yield "\n"
                        
            # Handle final response
            elif isinstance(chunk_event.content, str):
                # If we used tools and haven't shown results marker yet
                if announced_tools and not result_shown:
                    yield f"\n\n✅ **Results:**\n\n"
                    result_shown = True
                    
                # Yield the final content
                yield chunk_event.content