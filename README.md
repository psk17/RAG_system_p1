# RAG Knowledge Retrieval System

![CI](https://github.com/psk17/rag_system_p1/actions/workflows/ci.yml/badge.svg)

A production-ready, object-oriented Q&A system for private documents using **FastAPI**, **LangChain**, and **ChromaDB**. Features a high-performance minimalist Nike-style frontend interface with real-time markdown rendering and token streaming.

---

## 📁 Project Structure

```
rag_system/
├── src/
│   └── rag_system/
│       ├── api/
│       │   ├── routes/              # FastAPI endpoints (health, upload, query, sessions)
│       │   ├── schemas/             # Pydantic request/response validation schemas
│       │   ├── static/              # Nike-style HTML/JS/CSS frontend
│       │   ├── app.py               # FastAPI server startup & routing setup
│       │   ├── auth.py              # Header token authentication logic
│       │   ├── dependencies.py      # Core service initialization (Chroma, Redis)
│       │   └── middleware.py        # CORS config
│       ├── core/
│       │   ├── config/
│       │   │   └── settings.py      # Validated configuration settings (Pydantic)
│       │   └── interfaces/
│       │       ├── document_processor.py # Base document processing ABC & Chunk Dataclass
│       │       └── vector_store.py  # Base vector store ABC & SearchResult contract
│       ├── ingestion/
│       │   ├── chunking_service.py  # Recursive splitting & document hashing logic
│       │   ├── ingestion_service.py # Directory & file batch ingestion pipeline coordinator
│       │   ├── markdown_processor.py# Markdown-specific splitters (header preservation)
│       │   ├── pdf_processor.py     # PDF text extractor (PyMuPDF layout-aware engine)
│       │   └── vector_store_chroma.py # Chroma DB Adapter with custom embeddings factory
│       ├── memory/
│       │   └── redis_memory.py      # Redis-backed session memory store
│       └── rag/
│           ├── chain_manager.py     # LangChain generation Orchestrator
│           ├── factories.py         # LLM builder factory (OpenAI, Anthropic, Ollama, Fake)
│           ├── grounding.py         # Context grounding validation rules
│           ├── models.py            # Typed context & answer outputs
│           ├── prompts.py           # System Prompt Templates
│           └── retriever.py         # Vector similarity search retriever wrapper
├── tests/
│   ├── unit/                        # Processor, Settings, and Chroma Unit tests
│   └── integration/                 # End-to-end ingestion and query pipeline tests
├── scripts/
│   └── ingest.py                    # Bulk directory ingestion CLI tool
├── pyproject.toml                   # Poetry environment & project dependencies
├── START.bat                        # One-click clean, build, & start script (Windows)
└── .env.example                     # Environment variables configuration template
```

---

## ⚡ Core Features

- **Object-Oriented Design**: Pure abstract base classes (`BaseDocumentProcessor`, `BaseVectorStore`) for simple extension to other formats or databases.
- **Multimodal Ingestion**: Fully handles `.pdf` (layout-preserving parsing), `.md` (structured headers splits), and `.txt` files.
- **Pluggable Vector DB**: Ships with ChromaDB default adapter. Supports easy swap-ins for PGVector or Qdrant.
- **Flexible LLMs & Embeddings**: Native support for **OpenAI**, **Anthropic**, and local offline **Ollama** instances (both chat generation and vector embeddings).
- **API Security Hardening**: Restricts CORS wildcards in production, enforces file upload size limits (`max_upload_mb`), and secures operational endpoints (query, stream, sessions, metrics) using Bearer token authentication.
- **Nike-Style UI**: A premium, minimalist interface (`#0a0a0a` background with `#e5000a` red details) showcasing:
  - Token-by-token text streaming (SSE) with abort control.
  - Full Markdown rendering with syntax-highlighted code blocks.
  - Inline hover footnote citations displaying context, document origin, page, and confidence percentage.
  - Document library with delete capabilities to clean up indexed collections.

---

## 🚀 Setup & Execution

### Prerequisites
* Python **3.10+** (default AppData directory or in `PATH`)
* (Optional) [Ollama](https://ollama.com/) running locally for 100% private offline execution.

### Quick Start (Windows)
1. Copy the environment template to create your `.env` file:
   ```bash
   cp .env.example .env
   ```

## 📚 Community

- [CONTRIBUTING](./CONTRIBUTING.md)
- [CODE_OF_CONDUCT](./CODE_OF_CONDUCT.md)

2. Double-click the launcher script:
   ```
   START.bat
   ```
   *The script automatically cleans cache files, initializes a `.venv`, upgrades pip, installs all required packages, and starts the FastAPI server.*

3. Open your browser and navigate to:
   - **Frontend UI**: [http://localhost:8080/](http://localhost:8080/)
   - **Swagger Docs**: [http://localhost:8080/docs](http://localhost:8080/docs)

---

## ⚙️ Model Configurations

Edit your `.env` file to swap between models:

### Local Offline Mode (Ollama Embeddings + Ollama LLM)
Use a local Ollama instance for both generating RAG answers and vectorizing document text offline:
```ini
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3
EMBEDDING_PROVIDER=ollama
```
*Note: Make sure Ollama is running (`ollama run llama3`) before sending queries or uploading files.*

### Local Offline Mode (HuggingFace Embeddings + Ollama LLM)
Use local HuggingFace embeddings (`all-MiniLM-L6-v2`) and a local Ollama instance for LLM responses:
```ini
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3
EMBEDDING_PROVIDER=huggingface
```

### OpenAI Mode (Cloud-based)
```ini
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-your-openai-api-key
OPENAI_MODEL=gpt-4o-mini
EMBEDDING_PROVIDER=openai
```

---

## 🧪 Testing

Execute the test suites via pytest to ensure everything compiles and functions:
```bash
poetry run pytest tests/ -v
```
*(Over 50+ unit and integration tests covering extraction, vector ingestion, score normalization, auth middleware, and prompt grounding validation)*
