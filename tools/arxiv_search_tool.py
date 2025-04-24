import requests
from duckduckgo_search import DDGS


def query_arxiv(topic: str, max_results: int = 10):
    url = f"http://export.arxiv.org/api/query?search_query=all:{topic}&start=0&max_results={max_results}"
    response = requests.get(url)
    if response.status_code != 200:
        return "Failed to fetch arXiv data"
    return response.text


def query_web(topic: str):
    results = DDGS().text(topic, max_results=10)
    return [r["href"] for r in results]
