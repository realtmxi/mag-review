import os
import asyncio
from autogen_core import CancellationToken
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.ui import Console

async def main():
    # Get API credentials from environment variables
    api_key = os.getenv("OAI_KEY")
    api_endpoint = os.getenv("OAI_ENDPOINT")
    
    if not api_key or not api_endpoint:
        raise ValueError("Please set OAI_KEY and OAI_ENDPOINT environment variables")
    
    # Configure the Azure OpenAI client
    model_client = AzureOpenAIChatCompletionClient(
        api_key=api_key,
        azure_endpoint=api_endpoint,
        api_version="2024-05-13",
        model="gpt-4o",  # Replace with your actual model deployment name
    )
    
    # Create an assistant agent
    agent = AssistantAgent(
        name="assistant",
        model_client=model_client,
        description="A helpful assistant.",
        system_message="You are a helpful AI assistant. Answer questions concisely and accurately.",
    )
    
    # Test a simple query
    print("Running single response test...")
    response = await agent.on_messages(
        [TextMessage(content="What is the capital of France?", source="user")],
        CancellationToken()
    )
    print(f"Response: {response.chat_message.content}")
    
    # Test streaming mode
    print("\nRunning streaming test...")
    streaming_agent = AssistantAgent(
        name="streaming_assistant",
        model_client=model_client,
        model_client_stream=True,
        description="A helpful assistant with streaming enabled.",
    )
    
    print("Streaming response (you'll see tokens as they arrive):")
    stream = streaming_agent.on_messages_stream(
        [TextMessage(content="Name three famous landmarks in Paris.", source="user")],
        CancellationToken()
    )
    
    await Console(stream)

# Add a simple tool function to test function calling capabilities
async def get_current_time():
    """Returns the current time."""
    from datetime import datetime
    return f"The current time is {datetime.now().strftime('%H:%M:%S')}."

async def test_with_tool():
    # Get API credentials from environment variables
    api_key = os.getenv("OAI_KEY")
    api_endpoint = os.getenv("OAI_ENDPOINT")
    
    # Configure the Azure OpenAI client
    model_client = AzureOpenAIChatCompletionClient(
        api_key=api_key,
        azure_endpoint=api_endpoint,
        api_version="2024-05-13",
        model="gpt-4o",  # Replace with your actual model deployment name
    )
    
    # Create an assistant agent with tools
    agent = AssistantAgent(
        name="tool_assistant",
        model_client=model_client,
        tools=[get_current_time],
        description="A helpful assistant that can use tools.",
    )
    
    print("\nRunning tool test...")
    response = await agent.on_messages(
        [TextMessage(content="What time is it now?", source="user")],
        CancellationToken()
    )
    print(f"Response with tool: {response.chat_message.content}")

if __name__ == "__main__":
    # Run tests
    asyncio.run(main())
    
    # Uncomment below to test function calling
    asyncio.run(test_with_tool())