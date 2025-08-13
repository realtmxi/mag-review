import asyncio
from typing import AsyncGenerator
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken
from agents.literature_agent import run_literature_agent_stream
from agents.enhanced_web_crawling_agent import run_simple_agent
# from agents.paper_review_agent import run_review_agent_stream
# from agents.qa_agent import run_qa_agent_stream
from tools.qa_tools import load_context

async def multi_agent_dispatch_stream(user_input: str) -> AsyncGenerator[str, None]:
    """
    Routes input to the appropriate agent based on detected intent.
    """
    lowered = user_input.lower()

    if any(x in lowered for x in ["find papers", "search", "recommend", "literature"]):
        async for token in run_literature_agent_stream(user_input):
            yield token

    elif any(x in lowered for x in ["crawl", "web crawl", "scrape", "extract", "ontology", "structure data", "url", "urls", "website", "curl"]):
        # Use simplified agent for more reliable execution
        result = await run_simple_agent(user_input)
        yield result

    # elif any(x in lowered for x in ["review", "summarize", "analyze", "academic", "rapid", "enhanced", "visual"]):
    #     async for token in run_review_agent_stream(user_input):
    #         yield token

    # elif any(x in lowered for x in ["question", "what is", "explain", "how does", "why"]):
    #     async for token in run_qa_agent_stream(user_input):
    #         yield token

    else:
        # Default to literature agent if no specific intent is detected
        async for token in run_literature_agent_stream(user_input):
            yield token