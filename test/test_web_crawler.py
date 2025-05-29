#!/usr/bin/env python3
"""
Test script for the web crawling tool.
This script tests the web crawler and saves results to files for examination.
"""

import sys
import json
import os
from datetime import datetime

# Add current directory to path to import our tools
sys.path.append('.')

from tools.web_crawling_tools import web_crawling_tool, crawl_website

def test_crawler():
    """Test the web crawler with different URLs and save results"""
    
    # Create results directory if it doesn't exist
    os.makedirs('crawler_test_results', exist_ok=True)
    
    # Test URLs - start with simple ones
    test_urls = [
        # {
        #     'url': 'https://httpbin.org/html',
        #     'max_depth': 0,
        #     'description': 'Simple HTML test page'
        # },
        {
            'url': 'https://example.com',
            'max_depth': 1,
            'description': 'Example.com with depth 1'
        }
        # {
        #     'url': 'https://www.cmhc-schl.gc.ca',
        #     'max_depth': 1,
        #     'description': 'CMHC-SCHL.GC.CA with depth 1'
        # }
    ]
    
    print("Testing Web Crawler...")
    print("=" * 50)
    
    for i, test_case in enumerate(test_urls, 1):
        print(f"\nTest {i}: {test_case['description']}")
        print(f"URL: {test_case['url']}")
        print(f"Max Depth: {test_case['max_depth']}")
        
        try:
            # Run the crawler
            result = web_crawling_tool(test_case['url'], max_depth=test_case['max_depth'])
            
            # Parse result to check for errors
            data = json.loads(result)
            
            # Create filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"crawler_test_results/test_{i}_{timestamp}.json"
            
            # Save result to file
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(result)
            
            print(f"✓ Success! Results saved to: {filename}")
            
            # Print summary
            if 'error' not in data:
                crawl_data = data.get('data', {})
                print(f"  - Content type: {crawl_data.get('content_type', 'N/A')}")
                print(f"  - Title: {crawl_data.get('metadata', {}).get('title', 'N/A')}")
                print(f"  - Text length: {len(crawl_data.get('extracted_text', ''))}")
                print(f"  - Links found: {len(crawl_data.get('links', []))}")
            else:
                print(f"  - Error: {data.get('error')}")
                
        except Exception as e:
            print(f"✗ Error: {str(e)}")
            # Save error to file
            error_data = {
                "error": str(e),
                "test_case": test_case,
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"crawler_test_results/error_{i}_{timestamp}.json"
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(error_data, f, indent=2)
            print(f"  Error details saved to: {filename}")
    
    print("\n" + "=" * 50)
    print("Test completed! Check the 'crawler_test_results' directory for output files.")

if __name__ == "__main__":
    test_crawler() 