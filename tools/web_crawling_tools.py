import requests
import time
import hashlib
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Union, Any
from urllib.parse import urljoin, urlparse, urlunparse
from dataclasses import dataclass, asdict
import re
from bs4 import BeautifulSoup
import mimetypes

# PDF processing
try:
    import PyPDF2
    from io import BytesIO
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# Document processing
try:
    from docx import Document
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False


@dataclass
class CrawlResult:
    """Data structure for crawl results"""
    url: str
    depth: int
    content_type: str
    metadata: Dict[str, Any]
    extracted_text: str
    links: List['CrawlResult']
    error: Optional[str] = None


class WebCrawler:
    """
    Web crawler tool for extracting housing-related content from websites.
    Handles recursive crawling with depth limits and various content types.
    """
    
    def __init__(self, max_depth: int = 2, timeout: int = 30, max_links_per_page: int = 50):
        self.max_depth = max_depth
        self.timeout = timeout
        self.max_links_per_page = max_links_per_page
        self.visited_urls = set()
        self.session = requests.Session()
        
        # Configure session with headers
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def normalize_url(self, url: str) -> str:
        """Normalize URL by removing fragments and ensuring consistency"""
        parsed = urlparse(url)
        # Remove fragment and normalize
        normalized = urlunparse((
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            parsed.path,
            parsed.params,
            parsed.query,
            ''  # Remove fragment
        ))
        return normalized
    
    def is_valid_url(self, url: str, base_domain: str = None) -> bool:
        """Check if URL is valid and within scope"""
        try:
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                return False
            
            # Stay within the same domain if base_domain is specified
            if base_domain and parsed.netloc.lower() != base_domain.lower():
                return False
                
            # Filter out common non-content URLs
            excluded_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.css', '.js', '.ico', '.svg'}
            if any(parsed.path.lower().endswith(ext) for ext in excluded_extensions):
                return False
                
            return True
        except Exception:
            return False
    
    def extract_text_from_html(self, html_content: str) -> tuple[str, Dict[str, Any]]:
        """Extract text and metadata from HTML content"""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.extract()
            
            # Extract metadata
            metadata = {}
            
            # Title
            title_tag = soup.find('title')
            if title_tag:
                metadata['title'] = title_tag.get_text().strip()
            
            # Description
            desc_tag = soup.find('meta', attrs={'name': 'description'})
            if desc_tag:
                metadata['description'] = desc_tag.get('content', '').strip()
            
            # Keywords
            keywords_tag = soup.find('meta', attrs={'name': 'keywords'})
            if keywords_tag:
                keywords = keywords_tag.get('content', '').strip()
                metadata['keywords'] = [k.strip() for k in keywords.split(',') if k.strip()]
            
            # Extract main text content
            text = soup.get_text()
            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text, metadata
            
        except Exception as e:
            self.logger.error(f"Error extracting text from HTML: {e}")
            return "", {}
    
    def extract_text_from_pdf(self, content: bytes) -> str:
        """Extract text from PDF content"""
        if not PDF_AVAILABLE:
            return "PDF processing not available - PyPDF2 not installed"
        
        try:
            pdf_file = BytesIO(content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            
            return text.strip()
            
        except Exception as e:
            self.logger.error(f"Error extracting text from PDF: {e}")
            return f"Error processing PDF: {str(e)}"
    
    def extract_text_from_docx(self, content: bytes) -> str:
        """Extract text from DOCX content"""
        if not DOCX_AVAILABLE:
            return "DOCX processing not available - python-docx not installed"
        
        try:
            doc_file = BytesIO(content)
            doc = Document(doc_file)
            
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            
            return text.strip()
            
        except Exception as e:
            self.logger.error(f"Error extracting text from DOCX: {e}")
            return f"Error processing DOCX: {str(e)}"
    
    def extract_links_from_html(self, html_content: str, base_url: str) -> List[str]:
        """Extract all links from HTML content"""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            links = []
            
            for link in soup.find_all('a', href=True):
                href = link['href']
                absolute_url = urljoin(base_url, href)
                links.append(absolute_url)
            
            return links
            
        except Exception as e:
            self.logger.error(f"Error extracting links: {e}")
            return []
    
    def fetch_content(self, url: str) -> tuple[str, str, bytes]:
        """Fetch content from URL and return content type, text content, and raw bytes"""
        try:
            response = self.session.get(url, timeout=self.timeout, stream=True)
            response.raise_for_status()
            
            content_type = response.headers.get('content-type', '').lower()
            
            # Read content
            content_bytes = response.content
            
            return content_type, content_bytes.decode('utf-8', errors='ignore'), content_bytes
            
        except Exception as e:
            self.logger.error(f"Error fetching {url}: {e}")
            raise
    
    def process_content(self, url: str, content_type: str, text_content: str, raw_content: bytes) -> tuple[str, Dict[str, Any], List[str]]:
        """Process content based on its type and extract text, metadata, and links"""
        extracted_text = ""
        metadata = {}
        links = []
        
        try:
            if 'text/html' in content_type:
                extracted_text, metadata = self.extract_text_from_html(text_content)
                links = self.extract_links_from_html(text_content, url)
                
            elif 'application/pdf' in content_type:
                extracted_text = self.extract_text_from_pdf(raw_content)
                metadata = {'title': f"PDF document from {url}"}
                
            elif 'application/vnd.openxmlformats-officedocument.wordprocessingml.document' in content_type:
                extracted_text = self.extract_text_from_docx(raw_content)
                metadata = {'title': f"DOCX document from {url}"}
                
            elif 'text/' in content_type:
                extracted_text = text_content
                metadata = {'title': f"Text document from {url}"}
                
            else:
                extracted_text = f"Unsupported content type: {content_type}"
                metadata = {'title': f"Unsupported content from {url}"}
            
        except Exception as e:
            extracted_text = f"Error processing content: {str(e)}"
            metadata = {'title': f"Error processing {url}"}
        
        return extracted_text, metadata, links
    
    def crawl_recursive(self, url: str, current_depth: int = 0, base_domain: str = None) -> CrawlResult:
        """Recursively crawl URLs up to max_depth"""
        
        # Normalize URL
        normalized_url = self.normalize_url(url)
        
        # Check if already visited
        if normalized_url in self.visited_urls:
            return CrawlResult(
                url=normalized_url,
                depth=current_depth,
                content_type="",
                metadata={},
                extracted_text="Already visited",
                links=[],
                error="Already visited"
            )
        
        # Mark as visited
        self.visited_urls.add(normalized_url)
        
        try:
            self.logger.info(f"Crawling depth {current_depth}: {normalized_url}")
            
            # Fetch content
            content_type, text_content, raw_content = self.fetch_content(normalized_url)
            
            # Process content
            extracted_text, metadata, found_links = self.process_content(
                normalized_url, content_type, text_content, raw_content
            )
            
            # Initialize result
            result = CrawlResult(
                url=normalized_url,
                depth=current_depth,
                content_type=content_type,
                metadata=metadata,
                extracted_text=extracted_text,
                links=[]
            )
            
            # Recursively crawl links if we haven't reached max depth
            if current_depth < self.max_depth and found_links:
                
                # Set base domain from first URL if not set
                if base_domain is None:
                    base_domain = urlparse(normalized_url).netloc
                
                # Filter and limit links
                valid_links = []
                for link in found_links:
                    if self.is_valid_url(link, base_domain):
                        valid_links.append(link)
                        if len(valid_links) >= self.max_links_per_page:
                            break
                
                # Crawl each valid link
                for link in valid_links:
                    try:
                        link_result = self.crawl_recursive(link, current_depth + 1, base_domain)
                        result.links.append(link_result)
                        
                        # Add small delay to be respectful
                        time.sleep(0.5)
                        
                    except Exception as e:
                        self.logger.error(f"Error crawling link {link}: {e}")
                        error_result = CrawlResult(
                            url=link,
                            depth=current_depth + 1,
                            content_type="",
                            metadata={},
                            extracted_text="",
                            links=[],
                            error=str(e)
                        )
                        result.links.append(error_result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error crawling {normalized_url}: {e}")
            return CrawlResult(
                url=normalized_url,
                depth=current_depth,
                content_type="",
                metadata={},
                extracted_text="",
                links=[],
                error=str(e)
            )


def crawl_website(start_url: str, max_depth: int = 2, timeout: int = 30, max_links_per_page: int = 50) -> Dict[str, Any]:
    """
    Main function to crawl a website and return structured data.
    
    Args:
        start_url: The URL to start crawling from
        max_depth: Maximum recursion depth (0 = only start URL, 1 = start + direct links, etc.)
        timeout: Request timeout in seconds
        max_links_per_page: Maximum number of links to follow per page
    
    Returns:
        Structured crawl data as specified in the requirements
    """
    
    # Generate unique crawl ID
    crawl_id = f"crawl_{int(time.time())}_{hashlib.md5(start_url.encode()).hexdigest()[:8]}"
    
    # Initialize crawler
    crawler = WebCrawler(max_depth=max_depth, timeout=timeout, max_links_per_page=max_links_per_page)
    
    # Start crawling
    crawl_result = crawler.crawl_recursive(start_url)
    
    # Convert to the specified format
    def convert_crawl_result(result: CrawlResult) -> Dict[str, Any]:
        """Convert CrawlResult to the specified JSON format"""
        converted = {
            "url": result.url,
            "depth": result.depth,
            "content_type": result.content_type,
            "metadata": result.metadata,
            "extracted_text": result.extracted_text,
            "links": [convert_crawl_result(link) for link in result.links]
        }
        
        if result.error:
            converted["error"] = result.error
            
        return converted
    
    # Build final response
    response = {
        "crawl_id": crawl_id,
        "start_url": start_url,
        "crawl_timestamp": datetime.utcnow().isoformat() + "Z",
        "max_depth": max_depth,
        "data": convert_crawl_result(crawl_result)
    }
    
    return response


# Tool function for agent calling
def web_crawling_tool(url: str, max_depth: int = 2) -> str:
    """
    Web crawling tool for agent use.
    
    Args:
        url: The URL to crawl
        max_depth: Maximum depth for recursive crawling (default: 2)
    
    Returns:
        JSON string with crawl results
    """
    try:
        result = crawl_website(url, max_depth=max_depth)
        return json.dumps(result, indent=2)
    except Exception as e:
        error_result = {
            "error": f"Crawling failed: {str(e)}",
            "url": url,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        return json.dumps(error_result, indent=2)


if __name__ == "__main__":
    # Test the crawler
    test_url = "https://www.cmhc-schl.gc.ca"
    print("Testing web crawler...")
    result = web_crawling_tool(test_url, max_depth=1)
    print(result)
