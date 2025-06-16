import os
import json
from typing import Dict, List, Optional, Any, AsyncGenerator
from dotenv import load_dotenv
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen_core.tools import FunctionTool
from autogen_core import CancellationToken
from tools.web_crawling_tools import web_crawling_tool

load_dotenv()

def crawl_urls_batch(urls: List[str], max_depth: int = 2, max_links_per_page: int = 20) -> str:
    """
    Crawl multiple URLs and return structured results.
    
    Args:
        urls: List of URLs to crawl
        max_depth: Maximum depth for recursive crawling
        max_links_per_page: Maximum number of links to follow per page
    
    Returns:
        JSON string with batch crawl results
    """
    batch_results = {
        'batch_summary': {
            'total_urls_requested': len(urls),
            'max_depth': max_depth,
            'max_links_per_page': max_links_per_page
        },
        'url_results': []
    }
    
    for url in urls:
        try:
            crawl_result = web_crawling_tool(url, max_depth, max_links_per_page)
            crawl_data = json.loads(crawl_result)
            batch_results['url_results'].append({
                'requested_url': url,
                'status': 'success',
                'data': crawl_data
            })
        except Exception as e:
            batch_results['url_results'].append({
                'requested_url': url,
                'status': 'error',
                'error': str(e),
                'data': None
            })
    
    return json.dumps(batch_results, indent=2)

def apply_ontology_structuring(
    raw_crawl_data: str,
    ontology_schema: str,
    few_shot_examples: str = "",
    transformation_instructions: str = ""
) -> str:
    """
    Apply ontology structuring to crawled data using few-shot learning.
    
    Args:
        raw_crawl_data: Raw crawled data in JSON format
        ontology_schema: Target ontology/schema definition
        few_shot_examples: Few-shot learning examples showing input-output pairs
        transformation_instructions: Additional instructions for transformation
    
    Returns:
        Structured data according to the ontology
    """
    # This function serves as a structured input to the LLM agent
    # The actual transformation logic will be handled by the agent's reasoning
    structuring_request = {
        "task": "ontology_structuring",
        "raw_data": raw_crawl_data,
        "target_schema": ontology_schema,
        "few_shot_examples": few_shot_examples,
        "transformation_instructions": transformation_instructions,
        "instructions": """
        Transform the raw crawled data according to the provided ontology schema.
        
        Process:
        1. Parse the raw crawled data
        2. Analyze the target ontology schema
        3. Use the few-shot examples to understand the transformation pattern
        4. Apply the transformation to create structured output
        5. Validate the output against the schema
        
        Return the structured data in the exact format specified by the ontology.
        """
    }
    
    return json.dumps(structuring_request, indent=2)

def extract_content_pairs(crawled_data: str) -> str:
    """
    Extract (URL, content) pairs from crawled data in both raw and structured formats.
    
    Args:
        crawled_data: JSON string containing crawled data
    
    Returns:
        JSON string with extracted pairs
    """
    try:
        data = json.loads(crawled_data)
        pairs = {
            "unstructured_pairs": [],
            "metadata": {
                "extraction_timestamp": data.get('crawl_summary', {}).get('crawl_timestamp'),
                "total_pages": data.get('crawl_summary', {}).get('total_pages_crawled', 0)
            }
        }
        
        # Extract unstructured pairs
        if 'crawled_data' in data:
            for item in data['crawled_data']:
                pairs["unstructured_pairs"].append({
                    "url": item.get('url', ''),
                    "content": {
                        "title": item.get('title', ''),
                        "text": item.get('content', ''),
                        "metadata": item.get('metadata', {}),
                        "content_type": item.get('content_type', ''),
                        "crawl_depth": item.get('crawl_depth', 0)
                    }
                })
        
        # Handle batch results
        elif 'url_results' in data:
            for url_result in data['url_results']:
                if url_result['status'] == 'success' and url_result['data']:
                    crawl_data = url_result['data'].get('crawled_data', [])
                    for item in crawl_data:
                        pairs["unstructured_pairs"].append({
                            "url": item.get('url', ''),
                            "content": {
                                "title": item.get('title', ''),
                                "text": item.get('content', ''),
                                "metadata": item.get('metadata', {}),
                                "content_type": item.get('content_type', ''),
                                "crawl_depth": item.get('crawl_depth', 0)
                            }
                        })
        
        return json.dumps(pairs, indent=2)
        
    except Exception as e:
        return json.dumps({
            "error": f"Failed to extract content pairs: {str(e)}",
            "unstructured_pairs": [],
            "metadata": {}
        }, indent=2)

# Azure client configuration
api_key = os.getenv("OAI_KEY")
api_endpoint = os.getenv("OAI_ENDPOINT")

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

# Create enhanced function tools
batch_crawl_tool = FunctionTool(
    crawl_urls_batch,
    description="Crawls multiple URLs in batch and extracts content with configurable depth and link limits."
)

ontology_structuring_tool = FunctionTool(
    apply_ontology_structuring,
    description="Structures raw crawled data according to a given ontology using few-shot learning examples and transformation instructions."
)

content_pairs_tool = FunctionTool(
    extract_content_pairs,
    description="Extracts (URL, content) pairs from crawled data in unstructured format."
)

# Enhanced Web Crawling Agent
enhanced_web_crawling_agent = AssistantAgent(
    name="EnhancedWebCrawlingAgent",
    model_client=client,
    tools=[batch_crawl_tool, ontology_structuring_tool, content_pairs_tool],
    system_message="""You are an advanced web crawling and ontology structuring agent. Your specialized capabilities include:

## PRIMARY FUNCTIONS:

### 1. **Intelligent Web Crawling**
- Process single URLs or batches of URLs efficiently
- Extract both unstructured content and metadata
- Handle various content types (HTML, PDF, text)
- Respect crawling parameters (depth, links per page)

### 2. **Ontology-Based Data Structuring**
- Transform unstructured crawled data into structured formats
- Apply target ontology schemas with precision
- Utilize few-shot learning examples for pattern recognition
- Maintain data fidelity during transformation

### 3. **Few-Shot Learning Integration**
- Analyze provided examples to understand transformation patterns
- Extract implicit rules from input-output example pairs
- Apply learned patterns to new data consistently
- Adapt to different ontology formats and structures

## WORKFLOW METHODOLOGY:

For each request, follow this systematic approach:

**PHASE 1: ANALYSIS**
- Parse input requirements (URLs, ontology, examples)
- Identify content types and expected complexity
- Plan optimal crawling strategy

**PHASE 2: CRAWLING**
- Execute batch crawling with appropriate parameters
- Extract unstructured (URL, content) pairs
- Preserve metadata and structural information

**PHASE 3: LEARNING**
- Analyze few-shot examples to understand:
  - Input data patterns and formats
  - Target ontology structure and requirements
  - Transformation rules and mappings
  - Field relationships and constraints

**PHASE 4: STRUCTURING**
- Apply learned patterns to crawled data
- Transform content according to ontology schema
- Generate structured (URL, content) pairs
- Validate output against schema requirements

**PHASE 5: VALIDATION**
- Verify structured data matches ontology
- Check completeness and consistency
- Provide transformation explanations

## OUTPUT FORMATS:

Always provide both:
1. **Unstructured Pairs**: `[(URL, raw_content), ...]`
2. **Structured Pairs**: `[(URL, ontology_structured_content), ...]`

Where structured content follows the exact ontology schema provided.

## QUALITY PRINCIPLES:
- Maintain transparency in transformation decisions
- Preserve original data integrity
- Explain any assumptions or interpretations
- Handle errors gracefully with clear error reporting
- Optimize for both accuracy and completeness

Use your tools strategically and explain your reasoning for each step of the process.
""",
    reflect_on_tool_use=True,
)

# Async runner with enhanced streaming
async def run_enhanced_web_crawling_agent(user_input: str) -> AsyncGenerator[str, None]:
    """
    Run the enhanced web crawling agent with streaming support.
    
    Args:
        user_input: User query containing crawling instructions and ontology requirements
    
    Yields:
        Streaming response from the agent
    """
    yield "🚀 Initializing Enhanced Web Crawling Agent...\n\n"
    
    try:
        # Use the simple runner for more reliable results
        response = await run_enhanced_web_crawling_agent_simple(user_input)
        
        # Yield the complete response
        yield response
        
    except Exception as e:
        error_message = f"❌ **Error during web crawling:**\n\n{str(e)}\n\nPlease try with a different URL or check your network connection."
        yield error_message

# Synchronous runner
async def run_enhanced_web_crawling_agent_simple(user_input: str) -> str:
    """
    Run the enhanced web crawling agent and return the complete response.
    """
    response = await enhanced_web_crawling_agent.on_messages(
        [TextMessage(content=user_input, source="user")],
        cancellation_token=CancellationToken()
    )
    return response.chat_message.content

# Helper function for creating comprehensive ontology crawling requests
def create_enhanced_ontology_crawling_prompt(
    urls: List[str],
    ontology_schema: str,
    few_shot_examples: str = "",
    transformation_instructions: str = "",
    max_depth: int = 2,
    max_links_per_page: int = 20,
    output_format: str = "both"
) -> str:
    """
    Create a comprehensive prompt for ontology-based web crawling with few-shot learning.
    
    Args:
        urls: List of URLs to crawl
        ontology_schema: Target ontology/schema definition
        few_shot_examples: Few-shot learning examples
        transformation_instructions: Additional transformation guidance
        max_depth: Crawling depth
        max_links_per_page: Links per page limit
        output_format: "unstructured", "structured", or "both"
    
    Returns:
        Formatted prompt for the enhanced agent
    """
    urls_text = "\n".join([f"- {url}" for url in urls])
    
    prompt = f"""I need you to crawl the following URLs and structure the extracted data according to the specified ontology using few-shot learning:

## **URLs to Crawl:**
{urls_text}

## **Crawling Configuration:**
- Max Depth: {max_depth}
- Max Links per Page: {max_links_per_page}
- Output Format: {output_format}

## **Target Ontology Schema:**
```json
{ontology_schema}
```

## **Few-Shot Learning Examples:**
{few_shot_examples}

## **Additional Transformation Instructions:**
{transformation_instructions}

## **Required Deliverables:**

1. **Unstructured Content Pairs**: Extract (URL, raw_content) pairs from all crawled pages
2. **Structured Content Pairs**: Transform raw content to (URL, structured_content) pairs following the ontology
3. **Transformation Analysis**: Explain how you applied the few-shot examples to structure the data
4. **Quality Assessment**: Validate that structured output matches the ontology schema

Please process this request systematically and provide both formats as specified.
"""
    
    return prompt

# Workflow integration helper
def integrate_with_workflow(workflow_config: Dict[str, Any]) -> str:
    """
    Helper function to integrate enhanced web crawling with existing workflows.
    
    Args:
        workflow_config: Configuration for workflow integration
    
    Returns:
        Integration prompt
    """
    return f"""
    Workflow Integration Configuration:
    {json.dumps(workflow_config, indent=2)}
    
    This configuration enables the enhanced web crawling agent to integrate with your existing workflow system.
    The agent will provide structured outputs that can be consumed by downstream workflow steps.
    """ 