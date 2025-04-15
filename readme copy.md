# 📚 Literature Collection Agent using GitHub Models + Chainlit

This project is an AI-powered research assistant that helps users find and summarize academic papers on any topic. It uses **GitHub Models on Azure AI Inference**, AutoGen's latest `agentchat` framework, and a conversational UI powered by **Chainlit**.

---

## 🔧 Features

- 🌐 **Search arXiv API** and general web sources (via DuckDuckGo)
- 🤖 **LLM-based summarization** of search results via Azure-hosted `gpt-4o` (or any model you configure)
- 🛠️ **Tool-calling support** with the latest `autogen-agentchat`
- 💬 **Interactive chat interface** using Chainlit
- 🔁 Easily extendable to support file uploads, PubMed, citation graphs, etc.

---

## 🗂️ Project Structure

```
project_root/
├── tools/
│   └── arxiv_search_tool.py         # Contains search functions for arXiv + web
├── agents/
│   └── literature_agent.py          # Agent definition + Azure client
├── .env                             # Secrets and config
└── app.py                           # Chainlit entrypoint
```

---

## ⚙️ Requirements

```bash
pip install chainlit autogen-agentchat autogen-core autogen-ext python-dotenv duckduckgo-search requests
```

---

## 🔐 Setup Environment

Create a `.env` file in the root directory:

```env
GITHUB_TOKEN=your_github_pat_token_here

```

>  You can find your GitHub token in GitHub → Settings → Developer Settings → Personal Access Tokens  

---

## 🚀 Run the App

```bash
chainlit run app.py
```

Then open your browser at [http://localhost:8000](http://localhost:8000)

---

## 💡 How It Works

- **Chainlit** provides the UI.
- User enters a research query.
- The assistant agent:
  - Calls `query_arxiv()` and `query_web()` via `FunctionTool`
  - Uses `AzureAIChatCompletionClient` to process and summarize
- Results are displayed in the chat as a clean, concise answer.

---

## 📌 Sample Query

> _"Search for top papers on: multimodal foundation models in biology. Summarize the best 5."_

---

##  Powered By

- [Microsoft AutoGen Core](https://github.com/microsoft/autogen-core)
- [GitHub Models via Azure AI Inference](https://github.com/orgs/github-models)
- [Chainlit](https://www.chainlit.io/)
- [arXiv API](https://arxiv.org/help/api/index)
- [DuckDuckGo Search API](https://pypi.org/project/duckduckgo-search/)

