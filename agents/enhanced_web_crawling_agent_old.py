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
from tools.bedeo_ontology_tool import load_bedeo_ontology, get_bedeo_template, validate_rdf_against_bedeo

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

def crawl_single_url(url: str, max_depth: int = 1, max_links_per_page: int = 5) -> str:
    """
    Crawl a single URL and return structured results.
    
    Args:
        url: URL to crawl
        max_depth: Maximum depth for recursive crawling
        max_links_per_page: Maximum number of links to follow per page
    
    Returns:
        JSON string with crawl results
    """
    try:
        print(f"🕷️ Starting to crawl: {url}")
        print(f"   Max depth: {max_depth}, Max links: {max_links_per_page}")
        crawl_result = web_crawling_tool(url, max_depth, max_links_per_page)
        print(f"✅ Crawling completed, data size: {len(crawl_result)} characters")
        return crawl_result
    except Exception as e:
        print(f"❌ Crawling failed: {str(e)}")
        return json.dumps({
            "error": f"Failed to crawl URL: {str(e)}",
            "url": url
        }, indent=2)

def get_bedeo_ontology_schema() -> str:
    """
    Get the BEDEO ontology schema and template for structuring RDF data.
    
    Returns:
        String containing BEDEO template and available classes/properties
    """
    try:
        ontology = load_bedeo_ontology()
        template = get_bedeo_template()
        
        result = f"""BEDEO ONTOLOGY REFERENCE
        
Template to follow:
{template}

Available BEDEO Classes (USE ONLY THESE):
{', '.join(ontology['classes'][:20])}... ({ontology['total_classes']} total)

Available BEDEO Properties (USE ONLY THESE):
{', '.join(ontology['object_properties'][:20])}... ({ontology['total_properties']} total)

CRITICAL: You MUST use ONLY the classes and properties from BEDEO ontology.
DO NOT invent new properties like cmhc:minimumAffordableUnits.
"""
        return result
    except Exception as e:
        return f"Error loading BEDEO ontology: {str(e)}"

# Azure client configuration
api_key = os.getenv("OAI_KEY")
api_endpoint = os.getenv("OAI_ENDPOINT")

client = AzureOpenAIChatCompletionClient(
    api_key=api_key,
    azure_endpoint=api_endpoint,
    model="o4-mini",
    api_version="2024-05-13",
    model_info={
        "json_output": True,
        "function_calling": True,
        "vision": False,
        "family": "unknown",
    },
)

# Create enhanced function tools
single_crawl_tool = FunctionTool(
    crawl_single_url,
    description="REQUIRED STEP 2: Crawl the URL from the curl command. Call this with the full URL as the 'url' parameter."
)

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

bedeo_ontology_tool = FunctionTool(
    get_bedeo_ontology_schema,
    description="REQUIRED STEP 1: Get the BEDEO ontology schema with classes, properties, and RDF/Turtle template."
)

# Enhanced Web Crawling Agent
enhanced_web_crawling_agent = AssistantAgent(
    name="EnhancedWebCrawlingAgent",
    model_client=client,
    tools=[single_crawl_tool, batch_crawl_tool, ontology_structuring_tool, content_pairs_tool, bedeo_ontology_tool],
    system_message="""You are a web crawling agent that structures data into RDF/Turtle format using BEDEO ontology.

When user gives "curl [URL]" command:
1. First call get_bedeo_ontology_schema() to get BEDEO template
2. Then IMMEDIATELY call crawl_single_url(url="THE_URL_HERE") to crawl the page
3. Finally structure the crawled content as RDF/Turtle using BEDEO classes and properties

IMPORTANT: You MUST actually call the crawl_single_url() function. Do not just describe what you will do.

Output format:
1. Brief summary of crawled content
2. Complete RDF/Turtle representation using BEDEO ontology with:
   - Proper @prefix declarations
   - Organization → Opportunity → RealEstateAsset → Address structure
   - Only use BEDEO classes and properties (no invented properties)
   - Proper data types (^^xsd:string, ^^xsd:decimal)

Remember: ALWAYS call the tools, don't just describe your intent.
""",
    reflect_on_tool_use=False,  # o1 model works better without reflection
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
    yield "📋 Step 1: Loading BEDEO ontology template...\n"
    
    try:
        # Use the simple runner for more reliable results
        print("🔄 Agent is processing... (this may take 30-60 seconds)")
        response = await run_enhanced_web_crawling_agent_simple(user_input)
        
        # Yield the complete response
        yield "\n✅ Processing complete!\n\n"
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
    
    prompt = f"""I need you to crawl the following URLs and structure the extracted data according to the specified RDF/Turtle ontology using few-shot learning:

## **URLs to Crawl:**
{urls_text}

## **Crawling Configuration:**
- Max Depth: {max_depth}
- Max Links per Page: {max_links_per_page}
- Output Format: {output_format}

## **Target RDF/Turtle Ontology Schema:**
```turtle
{ontology_schema}
```

## **Few-Shot Learning Examples (RDF/Turtle format):**
{few_shot_examples}

## **Additional Transformation Instructions:**
{transformation_instructions}

## **Required Deliverables:**

1. **Crawled Data Summary**: Brief overview of what was crawled and extracted
2. **RDF/Turtle Structured Data**: Transform the crawled content into valid RDF/Turtle format following the ontology pattern from the examples

**Important Requirements:**
- Output must be valid RDF/Turtle syntax
- Use proper @prefix declarations
- Follow the exact URI patterns and property relationships from the examples
- Generate appropriate URIs for entities based on the crawled data
- Include proper data types for literals (e.g., ^^xsd:string)
- Use rdfs:label properties where shown in examples

Please process this request systematically and provide the final RDF/Turtle output that matches the ontology structure.
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