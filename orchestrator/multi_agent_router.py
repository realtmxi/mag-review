# ============ orchestrator/multi_agent_router.py ============
import asyncio
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken
from agents.literature_agent import run_literature_agent, run_literature_agent_stream
from agents.paper_review_agent import run_review_agent, run_review_agent_stream
from agents.qa_agent import run_qa_agent, run_qa_agent_stream
from tools.qa_tools import load_context


async def multi_agent_dispatch(user_input: str):
    """
    Routes input to the appropriate agent based on detected intent.
    """
    lowered = user_input.lower()

    if any(x in lowered for x in ["find papers", "search", "recommend", "literature"]):
        return await run_literature_agent(user_input)

    elif any(x in lowered for x in ["review", "summarize", "analyze", "academic", "rapid", "enhanced", "visual"]):
        result = await run_review_agent(user_input)

        # Store result into Q&A context
        try:
            summary_text = result.chat_message.content
            load_context(summary_text)
        except Exception:
            pass

        return result

    elif any(x in lowered for x in ["question", "what is", "explain", "how does", "why"]):
        return await run_qa_agent(user_input)

    else:
        return await run_literature_agent(user_input)  # default fallback


async def multi_agent_dispatch_stream(user_input: str):
    """
    Routes input to the appropriate agent and returns a streaming response.
    Uses the same routing logic as multi_agent_dispatch.
    """
    lowered = user_input.lower()

    if any(x in lowered for x in ["find papers", "search", "recommend", "literature"]):
        return await run_literature_agent_stream(user_input)

    elif any(x in lowered for x in ["review", "summarize", "analyze", "academic", "rapid", "enhanced", "visual"]):
        # For review agent, we still need to store the final result in context
        # The stream will handle this once complete
        stream = await run_review_agent_stream(user_input)
        
        # We'll wrap the stream to extract the final result for context
        async def process_stream():
            final_result = None
            async for chunk in stream:
                # Pass along each chunk
                yield chunk
                # If it's the final Response, store it
                from autogen_agentchat.base import Response
                if isinstance(chunk, Response):
                    final_result = chunk
            
            # Now store the final result in context if available
            if final_result and hasattr(final_result.chat_message, 'content'):
                try:
                    summary_text = final_result.chat_message.content
                    load_context(summary_text)
                except Exception:
                    pass
        
        return process_stream()

    elif any(x in lowered for x in ["question", "what is", "explain", "how does", "why"]):
        return await run_qa_agent_stream(user_input)

    else:
        return await run_literature_agent_stream(user_input)  # default fallback