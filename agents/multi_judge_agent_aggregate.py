import os
import json
import re
import asyncio
from dotenv import load_dotenv
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_ext.models.azure import AzureAIChatCompletionClient
from azure.core.credentials import AzureKeyCredential
from autogen_core import CancellationToken

from agents.retriever_agent import run_retriever_assistant  # Your retriever wrapper

load_dotenv()

# === Azure clients (use your 3 API keys) ===
client1 = AzureAIChatCompletionClient(
    model="gpt-4o-mini",
    endpoint="https://models.inference.ai.azure.com",
    credential=AzureKeyCredential(os.environ["GITHUB_TOKEN_1"]),
    model_info={
        "json_output": True,
        "function_calling": True,
        "vision": False,
        "family": "unknown",
    },
)

client2 = AzureAIChatCompletionClient(
    model="gpt-4o-mini",
    endpoint="https://models.inference.ai.azure.com",
    credential=AzureKeyCredential(os.environ["GITHUB_TOKEN_2"]),
    model_info={
        "json_output": True,
        "function_calling": True,
        "vision": False,
        "family": "unknown",
    },
)


# === Judge Agents (each focused on ONE dimension) ===
judge_relevance = AssistantAgent(
    name="Judge_Relevance",
    model_client=client1,
    system_message="You are a semantic relevance expert. For each paper, assign a score from 1–10 and a short reason explaining how well it matches the user's query."
)

judge_impact = AssistantAgent(
    name="Judge_Impact",
    model_client=client1,
    system_message="You are an academic impact expert. For each paper, score its expected influence (1–10) and briefly explain based on citations, venue, or reputation."
)

judge_novelty = AssistantAgent(
    name="Judge_Novelty",
    model_client=client1,
    system_message="You are a novelty expert. For each paper, score its originality (1–10) and briefly explain the novelty of its ideas or methods."
)

final_judge = AssistantAgent(
    name="Final_Judge",
    model_client=client2,
    system_message="""
You are the final evaluator. You will receive relevance, impact, and novelty scores for 20 papers.
Your tasks:
1. Select and rank the Top 5 papers overall.
2. For each selected paper, provide:
   - Title
   - Abstract
   - Link
3. Write a brief summary explaining why these papers were chosen based on the aggregated evaluations.
Do not include score tables.
"""
)

# === Parse helper (extract JSON list from LLM output) ===
def parse_json_from_agent(text):
    cleaned = text.strip("`json\n").strip("`")
    match = re.search(r"\[\s*{.*?}\s*]", cleaned, re.DOTALL)
    if not match:
        raise ValueError("No valid JSON list found.")
    return json.loads(match.group(0))

# === Ask a judge to score all papers in their dimension ===
async def score_papers(agent, papers, user_query, dimension):
    prompt = (
        f"User query: {user_query}\n\n"
        f"You are evaluating the following papers based on {dimension}.\n"
        "For each paper, return:\n"
        '{ "title": ..., "score": ..., "reason": ... }\n'
        "Only output a JSON list of 20 entries. No intro, no explanation.\n\n"
    )
    for p in papers:
        prompt += f"Title: {p['title']}\nAbstract: {p['abstract']}\n\n"

    response = await agent.on_messages(
        [TextMessage(content=prompt, source="user")],
        cancellation_token=CancellationToken()
    )

    return parse_json_from_agent(response.chat_message.content)

# === Main orchestration function ===
async def run_multi_judge_agent_aggregate(user_query: str):
    # Step 1: Retrieve papers
    papers_response = await run_retriever_assistant(user_query)
    try:
        papers = json.loads(papers_response.chat_message.content.strip("`json\n").strip("`"))
    except Exception as e:
        print("Failed to parse paper list:", e)
        return "Retrieval agent returned malformed data."

    if not isinstance(papers, list) or len(papers) == 0:
        return "No papers were retrieved or parsed correctly."

    selected_papers = papers[:20]

    # Step 2: Get scores from each judge
    relevance_scores = await score_papers(judge_relevance, selected_papers, user_query, "semantic relevance")
    impact_scores = await score_papers(judge_impact, selected_papers, user_query, "scientific impact")
    novelty_scores = await score_papers(judge_novelty, selected_papers, user_query, "novelty and originality")

    # Step 3: Final Judge aggregates everything
    final_input = {
        "papers": selected_papers,
        "relevance_scores": relevance_scores,
        "impact_scores": impact_scores,
        "novelty_scores": novelty_scores
    }

    final_prompt = (
        "You are given scores and explanations from 3 judges.\n"
        "Your job is to rank the best 5 papers overall.\n"
        f"{json.dumps(final_input)}"
    )

    result = await final_judge.on_messages(
        [TextMessage(content=final_prompt, source="user")],
        cancellation_token=CancellationToken()
    )

    return result.chat_message.content
