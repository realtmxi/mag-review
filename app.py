import chainlit as cl
from orchestrator.multi_agent_router import multi_agent_dispatch_stream

@cl.on_chat_start
async def start():
    cl.user_session.set("history", [])
    await cl.Message(content="Welcome! Ask me anything — I can find papers, review PDFs, or answer scientific questions.").send()

@cl.on_message
async def handle_message(message: cl.Message):
    # Process the user input
    user_input = message.content.strip()
    
    # Update conversation history
    history = cl.user_session.get("history")
    history.append(("user", user_input))
    
    # Initialize message for streaming with "Thinking..."
    msg = cl.Message(content="Thinking...")
    await msg.send()
    
    try:
        full_response = ""
        
        # Stream tokens from the appropriate agent
        async for token in multi_agent_dispatch_stream(user_input):
            if token:
                # Skip the loader token
                if token == "⏳ Thinking...":
                    continue
                
                # For the first real token, update the existing message
                if full_response == "":
                    # Clear the "Thinking..." text
                    msg.content = ""
                    await msg.update()
                
                # Add token to full response
                full_response += token
                await msg.stream_token(token)
        
        # Update history with assistant's response
        if full_response:
            history.append(("assistant", full_response))
    
    except Exception as e:
        # Handle any errors during streaming
        error_msg = cl.Message(content=f"Sorry, I encountered an error: {str(e)}")
        await error_msg.send()
        print(f"Error: {str(e)}")
        return