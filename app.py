import chainlit as cl
from orchestrator.multi_agent_router import multi_agent_dispatch_stream
from typing import Optional
import json

# Constants for agent types
SEARCH_AGENT = "search"
DOCUMENT_AGENT = "document"

@cl.set_chat_profiles
async def chat_profiles(current_user: cl.User):
    return [
        cl.ChatProfile(
            name="Search Agent",
            markdown_description="🔎 **Academic Research Explorer**\n\nAccess cutting-edge research papers and scholarly articles from arXiv, academic databases, and trusted web sources. Perfect for comprehensive literature reviews, citation analysis, and staying current with the latest developments in your field.",
            icon="https://cdn-icons-png.flaticon.com/512/7641/7641727.png",
        ),
        cl.ChatProfile(
            name="Document Analysis",
            markdown_description="📑 **Document Intelligence System**\n\nUpload research papers, technical documents, and academic PDFs for in-depth analysis. Extract key insights, visualize data, identify main findings, and get comprehensive answers to your specific questions about the document content.",
            icon="https://cdn-icons-png.flaticon.com/512/4725/4725970.png",
        ),
    ]

@cl.on_chat_start
async def start():
    # Initialize session variables
    cl.user_session.set("history", [])
    cl.user_session.set("active_documents", [])
    
    # Get the selected chat profile
    chat_profile = cl.user_session.get("chat_profile")
    
    # Set the current agent based on the selected profile without sending a welcome message
    if chat_profile == "Search Agent":
        cl.user_session.set("current_agent", SEARCH_AGENT)
    elif chat_profile == "Document Analysis":
        cl.user_session.set("current_agent", DOCUMENT_AGENT)


@cl.on_message
async def main(message: cl.Message):
    # Get current active agent
    current_agent = cl.user_session.get("current_agent")
    
    history = cl.user_session.get("history")
    history.append(("user", message.content))
    
    if current_agent == SEARCH_AGENT:
        await handle_search_message(message)
    elif current_agent == DOCUMENT_AGENT:
        # placeholder, work in progress
        await handle_search_message(message)


@cl.on_message
async def handle_search_message(message: cl.Message):
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