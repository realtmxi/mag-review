#  Multi-Agent Research Assistant

A modular research assistant built with **AutoGen AgentChat**, **Chainlit UI**, and **GitHub-hosted LLMs** on Azure Inference.

It simulates a full AI research pipeline:
- 📚 **Literature Agent** – searches papers from arXiv & web
- 🧾 **Paper Review Agent** – summarizes, visualizes, and enhances PDFs
- ❓ **Q&A Agent** – answers follow-up questions from reviewed content

---

## 🚀 Features

- 🔍 Dynamic academic search with arXiv + DuckDuckGo
- 📄 Multi-mode PDF review: "rapid", "academic", "visual", "enhanced"
- 🧠 Q&A powered by stored context and external search
- 🗂 Modular agent architecture with AutoGen + Chainlit
- 🧩 Pluggable LLMs: `gpt-4o`, `LLaMA`, `Mistral`, or custom deployments

---

## 🛠️ Tech Stack

| Layer | Tool |
|-------|------|
| Agents | `autogen-agentchat` |
| Tools  | `autogen-core`, `FunctionTool` |
| LLMs   | GitHub-hosted models on Azure Inference |
| Frontend | `Chainlit` for conversational UI |
| PDF & Web | `PyMuPDF`, `duckduckgo-search`, `matplotlib` |

---

## 📂 Project Structure

```
project_root/
├── agents/
│   ├── literature_agent.py
│   ├── paper_review_agent.py
│   └── qa_agent.py
├── tools/
│   ├── review_tools.py
│   └── qa_tools.py
├── orchestrator/
│   └── multi_agent_router.py
├── .env
├── app.py (Chainlit entry)
├── requirements.txt
├── environment.yml
└── README.md
```

---

## ⚙️ Setup

### 📦 Option 1: Conda Environment (Recommended)

```bash
conda env create -f environment.yml
conda activate agent
```

### 💡 Option 2: Pip Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\activate
pip install -r requirements.txt
```

---

## 🔐 Configure Environment
Create a `.env` file:
```env
GITHUB_TOKEN=ghp_XXXXXXXXXXXXXXXXXXXXX
```

---

##  Run the Agent System

```bash
chainlit run app.py
```
Then open [http://localhost:8000](http://localhost:8000)

---

## Example Queries

- "Search for top papers on temporal graph neural networks."
- "Review this PDF in enhanced mode."( to be implemented)
- "Give me a visual summary of this paper."
- "What is a temporal point process?"

---

## Limitations

- Azure Inference requires correct PAT and model permissions
- Some models do not support auto tool calling (manual fix required)
- Large PDFs are chunked to avoid context overflows

---

## Credits

- Microsoft AutoGen
- GitHub Models on Azure Inference
- Chainlit.io

---

## 📜 License

MIT License. Use freely, modify creatively, contribute collaboratively.

---

## 🤖 Future Work

- [ ] Add memory and persistent context
- [ ] Integrate PubMed, Semantic Scholar APIs
- [ ] Stream output for long responses
- [ ] Full PDF upload pipeline via Chainlit