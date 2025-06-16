import os
from typing import Dict, List, Optional, Any, AsyncGenerator
from dotenv import load_dotenv
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen_core.tools import FunctionTool
from autogen_core import CancellationToken
import json
from tools.web_crawling_tools import web_crawling_tool

load_dotenv()

def crawl_website_wrapper(url: str, max_depth: int = 2, max_links_per_page: int = 20) -> str:
    """
    Wrapper function for web crawling that can be used as a tool.
    
    Args:
        url: The URL to crawl
        max_depth: Maximum depth for recursive crawling (default: 2)
        max_links_per_page: Maximum number of links to follow per page (default: 20)
    
    Returns:
        JSON string with crawl results
    """
    return web_crawling_tool(url, max_depth)

def structure_crawled_data(
    crawled_data: str, 
    ontology_schema: str, 
    few_shot_examples: str = ""
) -> str:
    """
    Structure crawled data according to a given ontology using few-shot learning examples.
    
    Args:
        crawled_data: Raw crawled data in JSON format
        ontology_schema: The target ontology/schema to structure data into
        few_shot_examples: Few-shot learning examples showing input-output pairs
    
    Returns:
        Structured data according to the ontology
    """
    # This function will be enhanced by the LLM agent with the few-shot examples
    # The actual structuring logic will be handled by the agent's reasoning
    return f"Raw data to be structured:\n{crawled_data}\n\nTarget schema:\n{ontology_schema}\n\nExamples:\n{few_shot_examples}"

# Azure client configuration following the literature agent pattern
api_key = os.getenv("OAI_KEY")
api_endpoint = os.getenv("OAI_ENDPOINT")

# Azure OpenAI client
client = AzureOpenAIChatCompletionClient(
    api_key=api_key,
    azure_endpoint=api_endpoint,
    model="gpt-4o",
    api_version="2024-05-13",
    model_info={
        "json_output": True,
        "function_calling": True,
        "vision": False,
        "family": "unknown",
    },
)

# Create function tools
crawl_tool = FunctionTool(
    crawl_website_wrapper, 
    description="Crawls a website and extracts content from multiple pages with configurable depth and link limits."
)

structure_tool = FunctionTool(
    structure_crawled_data,
    description="Structures raw crawled data according to a given ontology using few-shot learning examples."
)

# Define the Web Crawling Agent
web_crawling_agent = AssistantAgent(
    name="WebCrawlingAgent",
    model_client=client,
    tools=[crawl_tool, structure_tool],
    system_message="""You are a specialized web crawling and data structuring agent. Your role is to:

1. **Web Crawling**: Use the crawl_website_wrapper tool to extract content from websites
   - Configure appropriate depth and link limits based on the task
   - Handle various content types (HTML, PDF, DOCX, etc.)
   - Extract both text content and metadata

2. **Data Structuring**: Transform unstructured crawled data into structured formats using ontologies
   - Apply few-shot learning techniques when provided with examples
   - Follow the specified ontology schema precisely
   - Maintain data quality and completeness during transformation

3. **Chain of Thought Processing**: For each task, follow this approach:
   - ANALYZE: Understand the target website and desired ontology structure
   - CRAWL: Execute web crawling with optimal parameters
   - STRUCTURE: Apply the ontology transformation using examples
   - VALIDATE: Ensure the output matches the required schema

When given few-shot examples, use them as templates to understand:
- Input data format and structure
- Expected output format according to the ontology
- Transformation patterns and logic
- Field mappings and data relationships

Always provide clear explanations of your structuring decisions and any assumptions made during the transformation process.
""",
    reflect_on_tool_use=True,
)

# Async runner for the agent
async def run_web_crawling_agent(user_input: str) -> AsyncGenerator[str, None]:
    """
    Run the web crawling agent with streaming support.
    
    Args:
        user_input: User query containing crawling instructions and ontology requirements
    
    Yields:
        Streaming response from the agent
    """
    stream = web_crawling_agent.on_messages_stream(
        [TextMessage(content=user_input, source="user")],
        cancellation_token=CancellationToken()
    )
    
    yield "🕷️ Initializing web crawling agent..."
    
    announced_tools = set()
    result_shown = False
    
    async for chunk_event in stream:
        if isinstance(chunk_event, str):
            yield chunk_event
            
        elif hasattr(chunk_event, 'content'):
            if isinstance(chunk_event.content, list):
                for function_call in chunk_event.content:
                    if hasattr(function_call, 'name'):
                        tool_name = function_call.name
                        
                        if tool_name not in announced_tools:
                            announced_tools.add(tool_name)
                            
                            if tool_name == "crawl_website_wrapper":
                                yield f"\n\n🕷️ **Crawling website...**\n"
                            elif tool_name == "structure_crawled_data":
                                yield f"\n\n🏗️ **Structuring data according to ontology...**\n"
                            
                            if hasattr(function_call, 'arguments'):
                                try:
                                    import json
                                    if isinstance(function_call.arguments, str):
                                        try:
                                            args_obj = json.loads(function_call.arguments)
                                            if args_obj:
                                                args_formatted = json.dumps(args_obj, indent=2)
                                                yield f"\n📋 **Parameters:**\n```json\n{args_formatted}\n```\n\n"
                                        except:
                                            pass
                                except Exception:
                                    pass
                        
            elif isinstance(chunk_event.content, str):
                if announced_tools and not result_shown:
                    yield f"\n\n✅ **Results:**\n\n"
                    result_shown = True
                    
                yield chunk_event.content

# Synchronous runner for simple use cases
async def run_web_crawling_agent_simple(user_input: str) -> str:
    """
    Run the web crawling agent and return the complete response.
    
    Args:
        user_input: User query containing crawling instructions and ontology requirements
    
    Returns:
        Complete response from the agent
    """
    response = await web_crawling_agent.on_messages(
        [TextMessage(content=user_input, source="user")],
        cancellation_token=CancellationToken()
    )
    return response.chat_message.content

# Helper function for ontology-based crawling workflow
def create_ontology_crawling_prompt(
    urls: List[str],
    ontology_schema: str,
    few_shot_examples: str = "",
    max_depth: int = 2,
    max_links_per_page: int = 20
) -> str:
    """
    Create a structured prompt for ontology-based web crawling.
    
    Args:
        urls: List of URLs to crawl
        ontology_schema: Target ontology/schema definition
        few_shot_examples: Few-shot learning examples
        max_depth: Crawling depth
        max_links_per_page: Links per page limit
    
    Returns:
        Formatted prompt for the agent
    """
    urls_text = "\n".join([f"- {url}" for url in urls])
    
    prompt = f"""I need you to crawl the following URLs and structure the extracted data according to the specified ontology:

**URLs to Crawl:**
{urls_text}

**Crawling Parameters:**
- Max Depth: {max_depth}
- Max Links per Page: {max_links_per_page}

**Target Ontology Schema:**
```json
{ontology_schema}
```

**Few-Shot Learning Examples:**
{few_shot_examples}

Please:
1. Crawl each URL to extract content and metadata
2. Structure the extracted data according to the ontology schema
3. Use the few-shot examples as templates for the transformation
4. Provide the final structured data in JSON format
"""
    
    return prompt 