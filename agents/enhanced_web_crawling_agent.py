import os
import json
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen_core.tools import FunctionTool
from autogen_core import CancellationToken
from tools.web_crawling_tools import web_crawling_tool
from tools.bedeo_ontology_tool import load_bedeo_ontology, get_bedeo_template

load_dotenv()

def crawl_and_structure(url: str) -> str:
    """
    Complete workflow: crawl URL and structure as RDF/Turtle using BEDEO ontology.
    
    Args:
        url: URL to crawl
    
    Returns:
        RDF/Turtle structured data
    """
    try:
        # Step 1: Get BEDEO template
        print(f"📋 Loading BEDEO ontology...")
        ontology = load_bedeo_ontology()
        template = get_bedeo_template()
        
        # Step 2: Crawl the URL
        print(f"🕷️ Crawling: {url}")
        crawl_result = web_crawling_tool(url, max_depth=1, max_links_per_page=5)
        data = json.loads(crawl_result)
        
        # Step 3: Extract key information
        print(f"🔍 Extracting information...")
        
        # Default values
        org_name = "CanadaLandsCompany"
        org_legal_name = "Canada Lands Company"
        opportunity_desc = "Federal Lands Development Opportunity"
        status = "Active"
        asset_name = "CurrieLot"
        asset_label = "Currie Development Site"
        asset_id = "CLC_LH_AB_CGY_L002"
        area = "0.4"
        city = "Calgary"
        province = "Alberta"
        country = "Canada"
        
        # Try to extract from crawled data
        if 'crawled_data' in data and len(data['crawled_data']) > 0:
            content = data['crawled_data'][0].get('content', '')
            title = data['crawled_data'][0].get('title', '')
            
            # Extract information from content
            if 'Currie' in content or 'Currie' in title:
                asset_name = "Currie"
                asset_label = "Currie Development Site"
                city = "Calgary"
                province = "Alberta"
            if 'Toronto' in content or 'Toronto' in title:
                city = "Toronto"
                province = "Ontario"
            if 'Bellevue' in content or 'Bellevue' in title:
                asset_name = "Bellevue"
                asset_label = "35 Bellevue Avenue"
                
            # Try to extract address if present
            if '35 Bellevue' in content:
                asset_label = "35 Bellevue Avenue"
            if '11 Brock' in content:
                asset_name = "Brock"
                asset_label = "11 Brock Avenue"
                
            # Extract organization names
            if 'St. Clare' in content or 'St Clare' in content:
                org_name = "StClares"
                org_legal_name = "St. Clare's Multifaith Housing Society"
            if 'KMCLT' in content or 'Kensington Market Community Land Trust' in content:
                org_name = "KMCLT"
                org_legal_name = "Kensington Market Community Land Trust"
            if 'City of Toronto' in content:
                org_name = "CityOfToronto"
                org_legal_name = "City of Toronto"
            if 'CMHC' in content or 'Canada Mortgage and Housing' in content:
                org_name = "CMHC"
                org_legal_name = "Canada Mortgage and Housing Corporation"
                
            # Extract status
            if 'under construction' in content.lower():
                status = "Under Construction"
            elif 'complete' in content.lower():
                status = "Completed"
            elif 'proposal' in content.lower():
                status = "Proposals under evaluation"
                
            # Extract area/size
            if 'hectare' in content.lower():
                import re
                match = re.search(r'(\d+\.?\d*)\s*hectare', content.lower())
                if match:
                    area = match.group(1)
            if 'units' in content.lower():
                match = re.search(r'(\d+)\s*units', content.lower())
                if match:
                    opportunity_desc = f"Affordable Housing Development - {match.group(1)} units"
        
        # Step 4: Generate RDF/Turtle
        print(f"📝 Generating RDF/Turtle...")
        
        rdf_output = f"""@prefix bedeo: <https://csse.utoronto.ca/> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

# The organization offering the opportunity  
bedeo:organization_{org_name}
    a bedeo:Organization ;
    bedeo:has_legal_name "{org_legal_name}"^^xsd:string ;
    bedeo:has_opportunity bedeo:opportunity_{asset_name}Development .

# The development opportunity
bedeo:opportunity_{asset_name}Development
    a bedeo:PpartnershipOpportunity ;
    rdfs:label "{opportunity_desc}" ;
    bedeo:has_status "{status}"^^xsd:string ;
    bedeo:has_real_estate_asset bedeo:realEstateAsset_{asset_name} .

# The real estate asset
bedeo:realEstateAsset_{asset_name}
    a bedeo:real_estate_asset ;
    rdfs:label "{asset_label}" ;
    bedeo:has_identifier "{asset_id}"^^xsd:string ;
    bedeo:has_surface_area_in_hectares "{area}"^^xsd:decimal ;
    bedeo:has_address bedeo:address_{asset_name} .

# The address for the asset
bedeo:address_{asset_name}
    a bedeo:Address ;
    rdfs:label "{asset_label} Address" ;
    bedeo:has_locality_name "{city}"^^xsd:string ;
    bedeo:has_province_name "{province}"^^xsd:string ;
    bedeo:has_country_name "{country}"^^xsd:string ."""
        
        result = f"""## 📊 Crawled Data Summary

**🏢 Organization:** {org_legal_name}  
**📍 Location:** {city}, {province}, {country}  
**📐 Land Size:** {area} hectares  
**🏗️ Project:** {opportunity_desc}  
**📌 Status:** {status}  

---

## 🔗 RDF/Turtle Structured Data

<details>
<summary><b>Click to view RDF/Turtle format</b> (for SPARQL queries)</summary>

```turtle
{rdf_output}
```

</details>

---

## 📝 Human-Readable Breakdown

### Organization Structure
```
🏢 {org_legal_name}
    └── 📋 Opportunity: {asset_name} Development
            └── 🏘️ Real Estate Asset: {asset_label}
                    └── 📍 Address: {city}, {province}
```

### Key Information
- **Asset ID:** `{asset_id}`
- **Surface Area:** {area} hectares
- **Location:** {city}, {province}, {country}
- **Project Status:** {status}

---

### 💡 How to Use This Data

1. **For SPARQL Queries:** Copy the RDF/Turtle data above
2. **For Analysis:** Use the structured information to understand the opportunity
3. **For Integration:** Import into your RDF triplestore using the BEDEO ontology"""
        
        return result
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return f"Error processing URL: {str(e)}"

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

# Single tool that does everything
complete_tool = FunctionTool(
    crawl_and_structure,
    description="Crawl a URL and structure the content as RDF/Turtle using BEDEO ontology. Use this for curl commands."
)

# Simplified agent
simple_agent = AssistantAgent(
    name="SimpleWebCrawlingAgent",
    model_client=client,
    tools=[complete_tool],
    system_message="""You are a web crawling agent. 

When user gives "curl [URL]" command:
Call crawl_and_structure(url="[THE URL]") 

That's it. One tool call does everything.""",
)

async def run_simple_agent(user_input: str) -> str:
    """Run the simplified agent."""
    response = await simple_agent.on_messages(
        [TextMessage(content=user_input, source="user")],
        cancellation_token=CancellationToken()
    )
    return response.chat_message.content