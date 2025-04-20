import requests
from duckduckgo_search import DDGS


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


def query_web(topic: str):
    results = DDGS().text(topic, max_results=5)
    return [r["href"] for r in results]
