import chainlit as cl
from orchestrator.multi_agent_router import multi_agent_dispatch, multi_agent_dispatch_stream
from autogen_agentchat.base import Response
from autogen_agentchat.messages import ModelClientStreamingChunkEvent, ToolCallRequestEvent

@cl.on_chat_start
async def start():
    cl.user_session.set("history", [])
    await cl.Message(content="Welcome! Ask me anything — I can find papers, review PDFs, or answer scientific questions.").send()

@cl.on_message
async def handle_message(message: cl.Message):
    user_input = message.content.strip()
    cl.user_session.get("history").append(("user", user_input))

    # Create a streaming message
    msg = cl.Message(content="")
    await msg.send()
    
    # Get the streaming response
    stream = await multi_agent_dispatch_stream(user_input)
    
    # Process different types of streaming content
    final_content = ""
    tool_calls_seen = False
    
    async for chunk in stream:
        if isinstance(chunk, ModelClientStreamingChunkEvent):
            # Stream normal text chunks
            await msg.stream_token(chunk.content)
            final_content += chunk.content
        elif isinstance(chunk, ToolCallRequestEvent) and not tool_calls_seen:
            # Notify when tools are being used
            tool_calls_seen = True
            await msg.stream_token("\n\n*Searching for information...*\n\n")
        elif isinstance(chunk, Response):
            # When we get the final complete response
            if hasattr(chunk.chat_message, 'content'):
                final_content = chunk.chat_message.content
                # For Chainlit Message objects, we create a new message with the final content
                # instead of trying to update the existing one
                await msg.update()  # This will complete the current message
                await cl.Message(content=final_content).send()  # Send a new message with the final content
                cl.user_session.get("history").append(("assistant", final_content))
                return
    
    # If we didn't get a final Response object but have accumulated content
    if final_content:
        # Create a new message with the final content
        await msg.update()  # Close the streaming message
        await cl.Message(content=final_content).send()
        cl.user_session.get("history").append(("assistant", final_content))