import requests
from duckduckgo_search import DDGS
from typing import Dict, Any, List, Optional, Union

def query_arxiv(topic: str, max_results: int = 5, sort_by: str = "relevance", sort_order: str = "descending"):
    """
    Query the arXiv API for papers on a specific topic with sorting options.
    
    Parameters:
    - topic: Search topic
    - max_results: Maximum number of results to return (default: 5)
    - sort_by: Sorting field ("relevance" or "submittedDate") (default: "relevance")
    - sort_order: Sorting direction ("ascending" or "descending") (default: "descending")
    
    Returns:
    - XML response as string, or error message
    """
    valid_sort_by = ["relevance", "submittedDate"]
    valid_sort_order = ["ascending", "descending"]
    
    if sort_by not in valid_sort_by:
        return f"Invalid sort_by parameter. Must be one of: {', '.join(valid_sort_by)}"
    
    if sort_order not in valid_sort_order:
        return f"Invalid sort_order parameter. Must be one of: {', '.join(valid_sort_order)}"
    
    url = f"http://export.arxiv.org/api/query?search_query=all:{topic}&start=0&max_results={max_results}"
    url += f"&sortBy={sort_by}&sortOrder={sort_order}"
    
    response = requests.get(url)
    if response.status_code != 200:
        return "Failed to fetch arXiv data"
    
    return response.text

def query_web(
    query: str,
    max_results: int = 5,
    search_type: str = "text",
    time_filter: str = None,
    site_specific: str = None,
    file_type: str = None,
    exclude_terms: list = None,
    include_keywords: list = None,
    return_full_results: bool = False
):
    """
    Simplified web search function using DuckDuckGo.
    
    Args:
        query (str): Main search query
        max_results (int): Maximum number of results to return
        search_type (str): Type of search - "text", "news", "images", or "videos"
        time_filter (str): Time filter - "d" (day), "w" (week), "m" (month), "y" (year)
        site_specific (str): Limit search to specific site (e.g., "arxiv.org")
        file_type (str): Filter by file type (e.g., "pdf", "doc")
        exclude_terms (list): Terms to exclude from search
        include_keywords (list): Additional keywords to include
        return_full_results (bool): Return full result objects instead of just URLs
        
    Returns:
        list: List of URLs or full result objects
        
    Examples:
        # Basic search
        results = query_web("machine learning")
        
        # Academic search
        papers = query_web(
            "transformer architecture",
            site_specific="arxiv.org", 
            file_type="pdf",
            max_results=10
        )
        
        # News search from past week
        news = query_web(
            "climate change",
            search_type="news",
            time_filter="w"
        )
    """
    advanced_query = query
    
    if site_specific:
        advanced_query += f" site:{site_specific}"
    
    if file_type:
        advanced_query += f" filetype:{file_type}"
    
    if include_keywords and isinstance(include_keywords, list):
        advanced_query += " " + " ".join(include_keywords)
    
    if exclude_terms and isinstance(exclude_terms, list):
        advanced_query += " " + " ".join([f"-{term}" for term in exclude_terms])
    
    search_params = {
        "keywords": advanced_query,
        "region": "wt-wt",  # Default worldwide region
        "safesearch": "moderate",  # Always use moderate safe search
        "max_results": max_results
    }
    
    # Add time filter if specified
    if time_filter in ["d", "w", "m", "y"]:
        search_params["timelimit"] = time_filter
    
    # Initialize DuckDuckGo search client
    ddgs = DDGS()
    
    # Select the appropriate search method
    if search_type == "images":
        search_method = ddgs.images
    elif search_type == "news":
        search_method = ddgs.news
    elif search_type == "videos":
        search_method = ddgs.videos
    else:  # Default to text search
        search_method = ddgs.text
    
    try:
        results = list(search_method(**search_params))
        
        if return_full_results:
            return results
        else:
            # Handle different result formats
            if search_type == "text" or search_type == "news":
                return [r.get("href", "") for r in results]
            elif search_type == "images" or search_type == "videos":
                return [r.get("image", r.get("url", "")) for r in results]
            
    except Exception as e:
        print(f"Search error: {str(e)}")
        return []
    
    return []