# ============ tools/review_tools.py ============
import fitz  # PyMuPDF
from duckduckgo_search import DDGS
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from collections import Counter


def summarize_pdf(file_path: str, mode: str = "rapid") -> str:
    """
    Extracts text from a PDF and summarizes it based on the chosen mode.
    Modes:
    - "rapid": 3-4 bullet points
    - "academic": paragraph-style summary
    """
    doc = fitz.open(file_path)
    full_text = "\n".join(page.get_text() for page in doc)
    doc.close()

    if mode == "rapid":
        return "\n".join([f"- {line.strip()}" for line in full_text.split(". ")[:4]])
    else:
        return f"Summary:\n{full_text[:1000]}..."


def enhanced_summary_web(content: str) -> str:
    """
    Searches the web for supporting content based on keywords from the summary.
    """
    keywords = " ".join(content.split()[:6])
    results = DDGS().text(keywords, max_results=3)
    return "\n".join([f"- {r['href']}" for r in results])


def visualize_summary(content: str) -> str:
    """
    Generates a simple visualization of keyword frequencies in the summary.
    Returns a base64-encoded PNG image.
    """
    words = [w.strip(".,()[]{}") for w in content.lower().split() if len(w) > 4]
    freq = Counter(words)
    top_words = freq.most_common(5)

    words, counts = zip(*top_words)
    fig, ax = plt.subplots()
    ax.bar(words, counts)
    ax.set_title("Top Keywords in Summary")
    ax.set_ylabel("Frequency")

    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def review_dispatcher(file_path: str, mode: str = "academic") -> str:
    """
    Dispatcher function that calls different review modes.
    """
    base_summary = summarize_pdf(file_path, mode="academic")

    if mode == "rapid":
        return summarize_pdf(file_path, mode="rapid")
    elif mode == "academic":
        return base_summary
    elif mode == "visual":
        image_data = visualize_summary(base_summary)
        return f"![summary](data:image/png;base64,{image_data})"
    elif mode == "enhanced":
        web_result = enhanced_summary_web(base_summary)
        return f"Summary with Web Enhancement:\n{base_summary}\n\nAdditional Links:\n{web_result}"
    else:
        return "Unknown review mode specified."
