import os
import json
import asyncio
from dotenv import load_dotenv
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.base import TaskResult
from autogen_ext.models.azure import AzureAIChatCompletionClient
from azure.core.credentials import AzureKeyCredential
from autogen_core import CancellationToken

from agents.retriever_agent import run_retriever_assistant

load_dotenv()

# === Azure OpenAI client configuration ===
client = AzureAIChatCompletionClient(
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

# === Define the three expert agents with distinct areas of evaluation ===
judge_semantic = AssistantAgent(
    name="Judge_Semantic",
    model_client=client,
    system_message=(
        "You are an expert in semantic relevance. Your role is to determine how well each paper aligns "
        "with the user's research interest. Focus strictly on the topical and contextual match between the paper content "
        "and the user's query. Do not consider novelty or citation impact."
    )
)

judge_impact = AssistantAgent(
    name="Judge_Impact",
    model_client=client,
    system_message=(
        "You are a bibliometric expert evaluating scientific impact. Assess papers based on expected or proven academic influence, "
        "including citation potential, venue, and author reputation. Ignore novelty and topic relevance."
    )
)

judge_novelty = AssistantAgent(
    name="Judge_Novelty",
    model_client=client,
    system_message=(
        "You are a scientific innovation expert. Evaluate the originality of the paper’s ideas, methods, and contributions. "
        "Focus on novelty, not on popularity or semantic alignment."
    )
)

# === Final Judge Agent: Aggregates discussion into top 5 papers + summary ===
final_judge = AssistantAgent(
    name="Final_Judge",
    model_client=client,
    system_message=(
        "You are the final evaluator. You have access to a transcript of a discussion between three domain experts "
        "(semantic relevance, scientific impact, and novelty). They reviewed 10 academic papers.\n\n"
        "Your tasks:\n"
        "1. Select and rank the top 5 papers based on their overall merit across all three dimensions.\n"
        "2. For each paper, output:\n"
        "   - Title\n"
        "   - Abstract\n"
        "   - Link\n"
        "3. Then write a short paragraph summarizing why these papers stood out.\n"
        "Avoid including raw discussion or internal reasoning.\n"
    )
)

# === RoundRobinGroupChat termination: 1 rounds of expert rotation ===
termination = MaxMessageTermination(3) # 1 rounds of 3 messages each (1 per judge)

# === Full pipeline ===
async def run_multi_judge_round_robin(user_query: str):
    # Step 1: Use the Retrieval Agent to obtain a list of 10 candidate papers
    papers_response = await run_retriever_assistant(user_query)
    try:
        papers = json.loads(papers_response.chat_message.content.strip("`json\n").strip("`"))
    except Exception as e:
        print("Failed to parse paper list:", e)
        return "Retrieval agent returned malformed data."

    if not isinstance(papers, list) or len(papers) == 0:
        return "No papers were retrieved or parsed correctly."

    selected_papers = papers[:20]

    # Step 2: Format a rich, rigorous task prompt for the judges
    intro = (
        f"The user is researching the topic: **{user_query}**.\n\n"
        "Below is a list of 10 academic papers retrieved from arXiv or similar sources. "
        "Each paper includes a title, abstract, and optional link.\n\n"
        "Your task as a panel of expert reviewers is to collaboratively evaluate these papers from three perspectives:\n"
        "- Judge_Semantic: Semantic relevance to the user's query\n"
        "- Judge_Impact: Academic/scientific impact and citation potential\n"
        "- Judge_Novelty: Novelty and originality of research ideas or methods\n\n"
        "Please refer to papers by their assigned number (e.g., Paper 1, Paper 2).\n"
        "For each paper, provide your assessment clearly and concisely.\n"
        "You may challenge or refine previous comments in your specialty.\n"
        "Avoid vague or general statements; be specific, and focus only on your evaluation dimension.\n"
        "The discussion will proceed in three total rounds of alternating responses.\n\n"
    )

    for i, p in enumerate(selected_papers):
        intro += (
            f"Paper {i+1}:\n"
            f"Title: {p['title']}\n"
            f"Abstract: {p['abstract']}\n"
            f"Link: {p.get('link', 'N/A')}\n\n"
        )

    # Step 3: Initialize the groupchat team
    team = RoundRobinGroupChat(
        [judge_semantic, judge_impact, judge_novelty],
        termination_condition=termination
    )

    # Step 4: Run the group discussion in a round robin fashion
    discussion_transcript = ""
    async for message in team.run_stream(task=intro):
        if isinstance(message, TaskResult):
            print("Discussion complete:", message.stop_reason)
        elif isinstance(message, TextMessage):
            print(f"[Message] {message.content}")  # Avoid accessing `sender`
            discussion_transcript += f"{message.content}\n"
        else:
            print("Unknown message type:", type(message))

    # Step 5: Send the transcript to the Final Judge for top-5 recommendations
    final_prompt = (
        "You are the final judge. You have been given a transcript of a round robin discussion among three expert reviewers. "
        "They evaluated 10 research papers from three perspectives: relevance, impact, and novelty.\n\n"
        "DO NOT include internal reasoning, calculations, or judge disagreements in your output.\n"
        "ONLY include the final result: a ranked list of top recommended papers and a summary paragraph.\n\n"
        "Do not include the discussion transcript or raw scores—just the structured result.\n\n"
        "Your task is as follows:\n"
        "1. Rank and recommend the top 5 papers based on combined strength across all three dimensions.\n"
        "2. For each paper, provide the following fields:\n"
        "   - Title\n"
        "   - Abstract\n"
        "   - Link (if available)\n"
        "3. A concluding summary paragraph that synthesizes the overall recommendation set—what kinds of papers are included, what trends or strengths are evident, and why they are suited to the user's query.\n"

        f"Discussion transcript:\n\n{discussion_transcript}"
    )

    # Step 6: Run the Final Judge Agent
    result = await final_judge.on_messages(
        [TextMessage(content=final_prompt, source="user")],
        cancellation_token=CancellationToken()
    )
    return result.chat_message.content
