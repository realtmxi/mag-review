# Multi-Agent Research Assistant

An advanced research assistant powered by **AutoGen AgentChat**, **Microsoft Azure Semantic Kernel**, and **Chainlit UI**, simulating the full academic workflow from literature search to document analysis.

> ⚙️ **Built for the Microsoft AI Agent Hackathon** — this system demonstrates **intelligent agent routing**, **modular architecture**, **semantic reasoning**, and **tool-enhanced AI interactions**, all using **Azure-native services**.

---

## 🧩 Overview

![image](https://github.com/user-attachments/assets/d7a519e2-2137-4558-9971-78cf928ac052)


The system supports **two core tabs** in the UI:

- **🔎 Search Assistant**
  - Handles general or thematic paper queries.
  - Routes intelligently via **Semantic Kernel planner**:
    - ➤ For classic/foundational questions → **Multi-Judge Agent**
    - ➤ For real-time or code-related queries (e.g., “2024”, “benchmark”, “with code”) → **Literature Review Agent** with Chain-of-Thought (CoT) and external tools.

- **📄 QA Assistant**
  - Supports document-based Q&A using a **robust RAG pipeline**.
  - Allows **uploading up to 10 files**, supporting `PDF`, `DOC`, `TXT`, etc.
  - Retrieves context using **chunking + cosine similarity** and enriches answers with **web search + citation**.

---

## ⚙️ Multi-Judge Recommendation System

Instead of relying on a single LLM, this system **concurrently dispatches three expert judge agents** with distinct scoring dimensions:

| Judge Agent         | Responsibility                                             |
|---------------------|-------------------------------------------------------------|
| 🧠 Semantic Judge    | Evaluates **semantic relevance** to user query              |
| 🌱 Innovation Judge | Scores **novelty and originality** of contributions         |
| 📈 Impact Judge     | Assesses **influence**, e.g., citation potential or impact  |

Each judge **independently retrieves** papers using arXiv/Web tools and returns scores and justifications.

A **Final Judge** agent aggregates the results through **reasoned synthesis**, producing a diverse, stable, and query-aligned ranked list.  
> This panel-style evaluation increases **robustness**, encourages **diversity**, and mirrors how academic peer review committees operate.

---

## 🚀 Features & Highlights

| Category               | Description                                                                 |
|------------------------|-----------------------------------------------------------------------------|
| 🧠 Semantic Routing     | SK-powered `FunctionCallingStepwisePlanner` dynamically selects the best agent |
| ⚖️ Multi-Judge Voting  | 3 judges + 1 aggregator create a robust, diverse ranking system               |
| 🔍 Real-Time Retrieval | Uses arXiv + DuckDuckGo via **NER-enhanced** keyword extraction               |
| 📚 RAG Q&A             | Upload up to 10 files, chunked + embedded via LangChain + Chroma             |
| 🌐 External Search     | Web results are retrieved, cited, and appended to Q&A results                 |
| 🧩 Modular Architecture| Fully decoupled agents for ease of customization and future integration       |
| ☁️ Azure Native        | Uses `AzureChatCompletion`, integrated with GitHub-hosted GPT-4o models       |

---

## 🛠️ Tech Stack

| Layer             | Tools Used                                                                 |
|------------------|-----------------------------------------------------------------------------|
| Agent Framework   | `AutoGen AgentChat`, `Semantic Kernel`, `FunctionTool`                     |
| LLM Backend       | Azure OpenAI (`AzureChatCompletion`, GPT-4o-mini)                          |
| Routing           | `FunctionCallingStepwisePlanner` for intelligent skill invocation          |
| Retrieval         | `arxiv.org API`, `DuckDuckGo`, `NER-based keyword extraction`              |
| Document QA       | `LangChain`, `Chroma`, `PyMuPDF`, `unstructured`, `cosine similarity`      |
| UI & UX           | `Chainlit` interface with interactive tabs                                 |
| Data Security     | `.env`, `dotenv`, Azure key + PAT protection                               |

---

## 📂 Codebase Structure

```
project_root/
├── agents/
│   ├── multi_judge_agent.py         # Concurrent judge execution
│   ├── literature_agent.py          # Web/arXiv retriever
│   └── document_agent.py            # Chroma + LangChain RAG
├── orchestrator/
│   └── sk_router_planner.py         # SK-based semantic router
├── tools/
│   ├── arxiv_search_tool.py         # With NER topic extraction
│   ├── qa_tools.py
│   └── review_tools.py
├── prompts/
│   └── prompt_template.py
├── public/                          # Chainlit UI
├── app.py                           # Entry point
├── environment.yml / requirements.txt
├── session_manager.py / test.py
└── README.md
```

---

## ⚙️ Setup Instructions

### 📦 Conda

```bash
conda env create -f environment.yml
conda activate agent
```

### 💡 Virtualenv

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 🔐 Environment Configuration

```env
AZURE_OPENAI_KEY=your-azure-key
GITHUB_TOKEN=ghp_XXXXXXXXXXXXXXXXX
```

---

## ▶️ Run the System

```bash
chainlit run app.py
```

Then open [http://localhost:8000](http://localhost:8000)

---

## 💬 Sample Prompts

- "Recommend classic deep learning papers."
- "Find the latest 2024 papers on Graph Transformers with code."
- "Upload and summarize these three PDFs."
- "Compare the novelty of these two LLM papers."

---

## ✅ Hackathon Evaluation Alignment

| Judging Area           | How This Project Aligns                                                                                  |
|------------------------|----------------------------------------------------------------------------------------------------------|
| 💡 Innovation           | Panel-based multi-agent scoring + SK planner + NER + CoT pipeline + RAG + tool-calling                  |
| 🌍 Impact               | Supports researchers, educators, and students in discovering, comparing, and understanding literature   |
| 🧰 Usability            | Real-world file support, Human-in-the-Loop citations, CoT, RAG, external data sources                    |
| 🧠 Solution Quality     | Full modular repo, semantic planner, test suite, robust chunking, clear separation of agents            |
| ☁️ MS Tech Focus        | Azure-hosted LLMs, Semantic Kernel routing, AutoGen agents, GitHub model integration                     |

---

## 📈 Future Work

- [ ] Stream CoT steps to front-end
- [ ] Add PubMed + Semantic Scholar APIs
- [ ] Visual summary support for uploaded PDFs
- [ ] Score calibration via user feedback loop

---

## 🙌 Acknowledgements

- Microsoft AutoGen Team
- Semantic Kernel & Azure AI Agent Team
- Chainlit.io contributors
- All open-source tools & research APIs used
