import os
import json
import asyncio
import re
from dotenv import load_dotenv
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_ext.models.azure import AzureAIChatCompletionClient
from azure.core.credentials import AzureKeyCredential
from autogen_core.tools import FunctionTool
from autogen_core import CancellationToken

from tools.arxiv_search_tool import query_arxiv, query_web

load_dotenv()

# === Azure client setup ===
def create_azure_client(key_env):
    return AzureAIChatCompletionClient(
        model="gpt-4o-mini",
        endpoint="https://models.inference.ai.azure.com",
        credential=AzureKeyCredential(os.getenv(key_env)),
        model_info={"json_output": True, "function_calling": True, "vision": False, "family": "unknown",},
    )

client1 = create_azure_client("GITHUB_TOKEN")
client2 = create_azure_client("GITHUB_TOKEN_2")

# === Tools: arxiv + web
arxiv_tool = FunctionTool(query_arxiv, description="Searches arXiv for academic papers.")
web_tool = FunctionTool(query_web, description="Searches the web for supporting resources beyond formal papers — including methods, datasets, results, discussions, and more.")

# === Judge Agent Factory ===
def create_judge_agent(name, model_client, dimension_prompt):
    return AssistantAgent(
        name=name,
        model_client=model_client,
        tools=[arxiv_tool, web_tool],
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
    model_client=client2,
    system_message=final_judge_prompt
)

# === Main function
async def run_multi_judge_agents(user_query: str):
    judge1 = create_judge_agent("Judge_Relevance", client1, judge_relevance_prompt)
    judge2 = create_judge_agent("Judge_Impact", client1, judge_impact_prompt)
    judge3 = create_judge_agent("Judge_Novelty", client1, judge_novelty_prompt)

    judge_agents = [judge1, judge2, judge3]
    judge_outputs = {}

    for judge in judge_agents:
        print(f"🧠 Invoking {judge.name}...")
        result = await judge.on_messages(
            [TextMessage(content=user_query, source="user")],
            cancellation_token=CancellationToken()
        )

        content = getattr(result.chat_message, "content", str(result))
        judge_outputs[judge.name] = content

        await asyncio.sleep(65)  # avoid rate limit

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
    final_result = await final_judge.on_messages(
        [TextMessage(content=aggregation_prompt, source="user")],
        cancellation_token=CancellationToken()
    )

    return final_result.chat_message.content