import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import json
from typing import Dict, List, Optional, Any
import time
import re
from dataclasses import dataclass
import PyPDF2
from io import BytesIO

@dataclass
class CrawledContent:
    """Data structure for crawled content."""
    url: str
    title: str
    content: str
    metadata: Dict[str, Any]
    content_type: str
    links: List[str]
    crawl_depth: int

def extract_pdf_content(pdf_url: str) -> str:
    """Extract text content from PDF URLs."""
    try:
        response = requests.get(pdf_url, timeout=30)
        response.raise_for_status()
        
        pdf_file = BytesIO(response.content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        
        return text.strip()
    except Exception as e:
        return f"Error extracting PDF content: {str(e)}"

def extract_html_content(html_content: str, url: str) -> tuple:
    """Extract structured content from HTML."""
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Remove script and style elements
    for script in soup(["script", "style"]):
        script.decompose()
    
    # Extract title
    title = ""
    if soup.title:
        title = soup.title.string.strip()
    
    # Extract main content
    content = soup.get_text()
    content = re.sub(r'\s+', ' ', content).strip()
    
    # Extract links
    links = []
    for link in soup.find_all('a', href=True):
        href = link['href']
        full_url = urljoin(url, href)
        if full_url.startswith(('http://', 'https://')):
            links.append(full_url)
    
    # Extract metadata
    metadata = {
        'description': '',
        'keywords': '',
        'author': '',
        'published_date': ''
    }
    
    # Meta description
    meta_desc = soup.find('meta', attrs={'name': 'description'})
    if meta_desc:
        metadata['description'] = meta_desc.get('content', '')
    
    # Meta keywords
    meta_keywords = soup.find('meta', attrs={'name': 'keywords'})
    if meta_keywords:
        metadata['keywords'] = meta_keywords.get('content', '')
    
    # Author
    meta_author = soup.find('meta', attrs={'name': 'author'})
    if meta_author:
        metadata['author'] = meta_author.get('content', '')
    
    return title, content, links, metadata

def crawl_single_url(url: str, depth: int = 0) -> CrawledContent:
    """Crawl a single URL and extract content."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        content_type = response.headers.get('content-type', '').lower()
        
        if 'application/pdf' in content_type:
            # Handle PDF content
            pdf_content = extract_pdf_content(url)
            return CrawledContent(
                url=url,
                title=f"PDF Document: {urlparse(url).path.split('/')[-1]}",
                content=pdf_content,
                metadata={'content_type': 'application/pdf'},
                content_type='pdf',
                links=[],
                crawl_depth=depth
            )
        
        elif 'text/html' in content_type:
            # Handle HTML content
            title, content, links, metadata = extract_html_content(response.text, url)
            return CrawledContent(
                url=url,
                title=title,
                content=content,
                metadata=metadata,
                content_type='html',
                links=links,
                crawl_depth=depth
            )
        
        else:
            # Handle other text content
            return CrawledContent(
                url=url,
                title=f"Document: {urlparse(url).path.split('/')[-1]}",
                content=response.text,
                metadata={'content_type': content_type},
                content_type='text',
                links=[],
                crawl_depth=depth
            )
            
    except Exception as e:
        return CrawledContent(
            url=url,
            title="Error",
            content=f"Error crawling {url}: {str(e)}",
            metadata={'error': str(e)},
            content_type='error',
            links=[],
            crawl_depth=depth
        )

def web_crawling_tool(base_url: str, max_depth: int = 2, max_links_per_page: int = 20) -> str:
    """
    Main web crawling function that crawls websites recursively.
    
    Args:
        base_url: Starting URL to crawl
        max_depth: Maximum depth for recursive crawling
        max_links_per_page: Maximum number of links to follow per page
    
    Returns:
        JSON string containing crawling results
    """
    crawled_urls = set()
    results = []
    urls_to_crawl = [(base_url, 0)]  # (url, depth)
    
    while urls_to_crawl:
        current_url, current_depth = urls_to_crawl.pop(0)
        
        if current_url in crawled_urls or current_depth > max_depth:
            continue
            
        crawled_urls.add(current_url)
        
        print(f"Crawling: {current_url} (depth: {current_depth})")
        
        # Crawl the current URL
        crawled_content = crawl_single_url(current_url, current_depth)
        results.append({
            'url': crawled_content.url,
            'title': crawled_content.title,
            'content': crawled_content.content[:5000],  # Limit content length
            'metadata': crawled_content.metadata,
            'content_type': crawled_content.content_type,
            'crawl_depth': crawled_content.crawl_depth,
            'links_found': len(crawled_content.links)
        })
        
        # Add links to crawl queue if we haven't reached max depth
        if current_depth < max_depth and crawled_content.content_type == 'html':
            # Limit the number of links to follow
            links_to_add = crawled_content.links[:max_links_per_page]
            for link in links_to_add:
                if link not in crawled_urls:
                    urls_to_crawl.append((link, current_depth + 1))
        
        # Add a small delay to be respectful
        time.sleep(1)
    
    return json.dumps({
        'crawl_summary': {
            'total_pages_crawled': len(results),
            'base_url': base_url,
            'max_depth_used': max_depth,
            'crawl_timestamp': time.time()
        },
        'crawled_data': results
    }, indent=2)

def crawl_website(url: str, max_depth: int = 1) -> Dict[str, Any]:
    """
    Simple wrapper for backward compatibility.
    """
    result = web_crawling_tool(url, max_depth)
    return json.loads(result) 