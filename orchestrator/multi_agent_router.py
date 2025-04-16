# ============ orchestrator/multi_agent_router.py ============
import asyncio
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken
from agents.literature_agent import run_literature_agent
from agents.paper_review_agent import run_review_agent
from agents.qa_agent import run_qa_agent
from agents.multi_judge_agent import run_multi_judge_agents
from tools.qa_tools import load_context


# async def multi_agent_dispatch(user_input: str):
#     """
#     Routes input to the appropriate agent based on detected intent.
#     """
#     lowered = user_input.lower()

#     if any(x in lowered for x in ["find papers", "search", "recommend", "literature"]):
#         return await run_literature_agent(user_input)

#     elif any(x in lowered for x in ["review", "summarize", "analyze", "academic", "rapid", "enhanced", "visual"]):
#         result = await run_review_agent(user_input)

#         # Store result into Q&A context
#         try:
#             summary_text = result.chat_message.content
#             load_context(summary_text)
#         except Exception:
#             pass

#         return result

#     elif any(x in lowered for x in ["question", "what is", "explain", "how does", "why"]):
#         return await run_qa_agent(user_input)

#     else:
#         return await run_literature_agent(user_input)  # default fallback

async def multi_agent_dispatch(user_input: str):
    # return await run_multi_judge_agent_v1(user_input)
    return await run_multi_judge_agents(user_input)
