import os
import re
import requests
from datetime import datetime
from typing import List

PDF_KNOWLEDGE_BASE_PATH = "/Users/smu_cs_dsi/Documents/multi-agent-assistant/knowledge_base"  # change this to your path

# This cache should be filled after recommendations are made
RECOMMENDATION_CACHE = []  # Each item: {"title": ..., "url": ...}

def list_local_pdfs(base_path: str = PDF_KNOWLEDGE_BASE_PATH) -> dict:
    '''
    Recursively lists all PDF files in the user's local knowledge base directory, including subfolders.

    Returns:
    - dict: A dictionary with relative folder paths as keys, and list of PDFs as values.
    '''
    file_tree = {}
    for root, dirs, files in os.walk(base_path):
        rel_path = os.path.relpath(root, base_path)
        pdf_files = [f for f in files if f.lower().endswith('.pdf')]
        if pdf_files:
            file_tree[rel_path if rel_path != '.' else '/'] = pdf_files
    return file_tree


def download_pdf_to_local(url: str, base_path: str = PDF_KNOWLEDGE_BASE_PATH) -> str:
    '''
    Downloads a PDF file from an arXiv paper URL and saves it to the local knowledge base directory.

    Parameters:
    - url (str): The arXiv paper URL (e.g., "https://arxiv.org/abs/2001.12345")
    - base_path (str): The local folder path where PDFs should be saved

    Returns:
    - str: A confirmation message with the local save path or an error message
    '''
    if "arxiv.org" not in url:
        return "Only arXiv links are supported for now."

    match = re.search(r'arxiv.org/(abs|pdf)/([0-9\.]+)', url)
    if not match:
        return "Invalid arXiv link format."

    arxiv_id = match.group(2)
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

    response = requests.get(pdf_url)
    if response.status_code != 200:
        return f"Failed to download PDF from {pdf_url}"

    os.makedirs(base_path, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"{arxiv_id}_{timestamp}.pdf"
    save_path = os.path.join(base_path, filename)

    with open(save_path, "wb") as f:
        f.write(response.content)

    return f"PDF saved to: {save_path}"


def resolve_user_selection_and_download(user_reply: str) -> List[str]:
    '''
    Interprets a user's save instruction (e.g., "save the 1st paper", "save all") 
    and downloads the corresponding PDFs from the current recommendation cache.

    Parameters:
    - user_reply (str): A user command referring to one or more papers (e.g., "save 2", "save all")

    Returns:
    - List[str]: A list of confirmation messages or error messages for each downloaded paper
    '''

    lowered = user_reply.lower()
    selected_indices = []

    if "all" in lowered or "save all" in lowered:
        selected_indices = list(range(len(RECOMMENDATION_CACHE)))

    import re
    matches = re.findall(r"(\d+)", lowered)
    if matches:
        selected_indices += [int(idx) - 1 for idx in matches]

    results = []
    for i in selected_indices:
        if 0 <= i < len(RECOMMENDATION_CACHE):
            paper = RECOMMENDATION_CACHE[i]
            result = download_pdf_to_local(paper["url"])
            results.append(result)

    return results
