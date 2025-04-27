import chainlit as cl
from orchestrator.sk_router_planner import multi_agent_dispatch

@cl.on_chat_start
async def start():
    cl.user_session.set("history", [])
    await cl.Message(content="Welcome! Ask me anything — I can find papers, review PDFs, or answer scientific questions.").send()

@cl.on_message
async def handle_message(message: cl.Message):
    user_input = message.content.strip()
    cl.user_session.get("history").append(("user", user_input))

    response = await multi_agent_dispatch(user_input)
    # await cl.Message(content=response.chat_message.content).send()
    await cl.Message(content=response).send()


