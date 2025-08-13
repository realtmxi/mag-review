import chainlit as cl
from orchestrator.multi_agent_router import multi_agent_dispatch_stream
from typing import Optional
import json

# Constants for agent types
SEARCH_AGENT = "search"
WEB_CRAWLING_AGENT = "web_crawling"
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
            name="Web Crawling & Data Extraction",
            markdown_description="🕷️ **Enhanced Web Crawling Agent**\n\nCrawl websites and extract structured data using ontology-based transformations. Convert unstructured web content into organized (URL, content) pairs with customizable schemas and few-shot learning. Perfect for data collection, content analysis, and information structuring.",
            icon="https://cdn-icons-png.flaticon.com/512/2966/2966327.png",
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
    elif chat_profile == "Web Crawling & Data Extraction":
        cl.user_session.set("current_agent", WEB_CRAWLING_AGENT)
    elif chat_profile == "Document Analysis":
        cl.user_session.set("current_agent", DOCUMENT_AGENT)


@cl.on_message
async def main(message: cl.Message):
    # Get current active agent
    current_agent = cl.user_session.get("current_agent")
    
    history = cl.user_session.get("history")
    history.append(("user", message.content))
    
    # Always use the multi-agent router for intelligent routing
    # This ensures web crawling and other agents work properly
    await handle_universal_message(message)


async def handle_universal_message(message: cl.Message):
    """
    Universal message handler that routes to appropriate agents based on content.
    Supports: Literature Search, Web Crawling, Document Analysis, and more.
    """
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
        first_content_received = False
        
        # Stream tokens from the appropriate agent
        async for token in multi_agent_dispatch_stream(user_input):
            if token and token.strip():  # Ensure token has content
                print(f"DEBUG: Received token: {repr(token)}")  # Debug output - show full token
                
                # Skip various loader tokens
                if token.strip() in ["⏳ Thinking...", "🚀 Initializing Enhanced Web Crawling Agent..."]:
                    continue
                
                # For the first real token, clear "Thinking..." and start fresh
                if not first_content_received:
                    msg.content = ""
                    await msg.update()
                    first_content_received = True
                
                # Add token to full response and stream it
                full_response += token
                await msg.stream_token(token)
        
        # If no streaming content was received, get the complete response
        if not full_response.strip():
            # Fallback: get complete response for web crawling agent
            print("No streaming content received, using fallback method...")
            
            # Import the web crawling agent directly
            from agents.enhanced_web_crawling_agent import run_enhanced_web_crawling_agent_simple
            
            try:
                complete_response = await run_enhanced_web_crawling_agent_simple(user_input)
                msg.content = complete_response
                await msg.update()
                full_response = complete_response
            except Exception as fallback_error:
                error_text = f"Error during crawling: {str(fallback_error)}"
                msg.content = error_text
                await msg.update()
                full_response = error_text
        
        # Update history with assistant's response
        if full_response:
            history.append(("assistant", full_response))
    
    except Exception as e:
        # Handle any errors during streaming
        error_msg = cl.Message(content=f"Sorry, I encountered an error: {str(e)}")
        await error_msg.send()
        print(f"Error: {str(e)}")
        return