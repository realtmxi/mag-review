import os
from typing import AsyncGenerator
from dotenv import load_dotenv
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen_core import CancellationToken
import json

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

# === Judge Agent Factory ===
def create_judge_agent(name, model_client, dimension_prompt):
    return AssistantAgent(
        name=name,
        model_client=model_client,
        system_message=dimension_prompt,
        reflect_on_tool_use=True
    )

# === Individual prompts
judge_relevance_prompt = """
You are a semantic relevance expert.

User query: {query}

Step 1: You must use your tools to retrieve 5–10 papers that are closely aligned with the user's query.

Step 2: Score each paper on its semantic relevancy (1–10), and explain briefly.

Return only a valid JSON list (no explanation, no markdown block).
"""

judge_impact_prompt = """
You are a scientific impact expert.

User query: {query}

Step 1: You must use your tools to retrieve 5–10 papers that are highly cited, influential, or from reputable venues.

Step 2: Score each paper on its scientific impact (1–10), and explain briefly.

Return only a valid JSON list (no explanation, no markdown block).
"""

judge_novelty_prompt = """
You are a novelty and originality expert.

User query: {query}

Step 1: You must use your tools to retrieve 5–10 papers that contain new ideas, novel methods, or unique perspectives.

Step 2: Score each paper (1–10) based on innovation, and provide a one-line explanation.

Return only a valid JSON list (no explanation, no markdown block).
"""

# === Final Judge Prompt
final_judge_prompt = """
You are the final evaluator.

You are given 3 sets of papers, each selected and scored by a different expert agent:
- Relevance Expert
- Impact Expert
- Novelty Expert

Each expert returned 5–10 papers relevant to their dimension, including score and reason.

Your task is to review their evaluations and produce a clean, structured recommendation output for the user.
DO NOT include internal reasoning, calculations, or judge disagreements in your output.
ONLY include the final result: a ranked list of top recommended papers and a summary paragraph.

Please follow this output format:
1. A list of recommended papers, ranked from most to least relevant.
   For each paper, provide:
     - Title
     - Abstract
     - Link (if available)
2. A concluding summary paragraph that synthesizes the overall recommendation set—what kinds of papers are included, what trends or strengths are evident, and why they are suited to the user's query.
"""

# === Final Judge
final_judge = AssistantAgent(
    name="Final_Judge",
    model_client=client,
    system_message=final_judge_prompt,
    model_client_stream=True
)


# Async runner wrapper with proper token streaming
async def run_multi_judge_agents(user_input: str) -> AsyncGenerator[str, None]:
    judge1 = create_judge_agent("Judge_Relevance", client, judge_relevance_prompt)
    judge2 = create_judge_agent("Judge_Impact", client, judge_impact_prompt)
    judge3 = create_judge_agent("Judge_Novelty", client, judge_novelty_prompt)

    judge_agents = [judge1, judge2, judge3]
    judge_outputs = {}

    for judge in judge_agents:
        print(f"🧠 Invoking {judge.name}...")
        result = await judge.on_messages(
            [TextMessage(content=user_input, source="user")],
            cancellation_token=CancellationToken()
        )

        content = getattr(result.chat_message, "content", str(result))
        judge_outputs[judge.name] = content

        # Create aggregation prompt for Final Judge (no JSON parsing)
    aggregation_prompt = (
        "You are the final judge aggregating independent evaluations from three judge agents.\n\n"
        "Each judge has independently reviewed a set of papers based on a specific dimension: semantic relevance, impact, or novelty.\n"
        "Their outputs may include paper titles, abstracts, scores (1–10), and short reasons.\n\n"
        "Your task is to:\n"
        "1. Read all three agents' evaluations.\n"
        "2. Select and rank the top 5 papers across all outputs.\n"
        "3. Provide for each selected paper:\n"
        "   - Title\n"
        "   - Abstract\n"
        "   - Link (if available)\n"
        "4. Write a final summary paragraph describing the selection trends and how these papers relate to the user's query.\n\n"
        "DO NOT include any judge disagreements, internal calculations, or tool usage logs.\n"
        "ONLY include the final output.\n\n"
        f"Here is the input JSON containing the three judges' evaluations:\n{json.dumps(judge_outputs)}"
    )

    print("🏁 Invoking Final Judge...")
    stream = final_judge.on_messages_stream(
        [TextMessage(content=aggregation_prompt, source="user")],
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
