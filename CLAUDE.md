# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Setup and Installation
- **Install dependencies**: `uv sync`
- **Start development server**: `./run.sh` or manually `cd backend && uv run uvicorn app:app --reload --port 8000`
- **Environment setup**: Create `.env` file in root with `ANTHROPIC_API_KEY=your_key_here`

### Application Access
- **Web interface**: http://localhost:8000
- **API documentation**: http://localhost:8000/docs

### Code Quality
- **Format code**: `./scripts/format.sh` (runs Black + isort)
- **Lint code**: `./scripts/lint.sh` (runs flake8)
- **All quality checks**: `./scripts/quality.sh` (format + lint + tests)
- **Manual formatting**: `uv run black backend/ main.py`
- **Manual import sorting**: `uv run isort backend/ main.py`
- **Manual linting**: `uv run flake8 backend/ main.py --max-line-length=88 --extend-ignore=E203,W503`

## Architecture Overview

This is a **Retrieval-Augmented Generation (RAG) chatbot** system that answers questions about course materials using semantic search and AI-powered responses.

### Tech Stack
- **Backend**: FastAPI + Python 3.13+
- **Vector Database**: ChromaDB with persistent storage (`./backend/chroma_db/`)
- **AI Model**: Anthropic Claude Sonnet 4 (`claude-sonnet-4-20250514`)
- **Embeddings**: SentenceTransformer (`all-MiniLM-L6-v2`)
- **Frontend**: Vanilla HTML/CSS/JavaScript
- **Package Manager**: uv

### Core Components Architecture

The system follows a **tool-based RAG approach** where Claude decides autonomously whether to search for information:

#### 1. RAGSystem (`backend/rag_system.py`)
Main orchestrator that coordinates all components:
- **DocumentProcessor**: Chunking and course extraction from text files
- **VectorStore**: ChromaDB vector operations and semantic search
- **AIGenerator**: Claude API integration with tool support
- **SessionManager**: In-memory conversation history (2 exchanges max)
- **ToolManager**: Manages search tools available to Claude

#### 2. Query Processing Flow
```
User Query → FastAPI → RAGSystem.query() → AIGenerator (Claude API) 
    ↓
Claude decides: Use search tool OR answer directly
    ↓
If search: CourseSearchTool → ChromaDB → Vector results → Claude generates response
    ↓
Response + Sources → Frontend
```

#### 3. Document Structure
- **Course**: Title, lessons, instructor metadata
- **CourseChunk**: Text chunks with course/lesson attribution for vector search
- **Document Processing**: Sentence-based chunking (800 chars, 100 overlap)

#### 4. Vector Storage Strategy
- **Two ChromaDB collections**:
  - `course_metadata`: Course-level information for course name matching
  - `course_content`: Text chunks for semantic content search
- **Smart course name matching**: Partial matches work (e.g., "MCP" finds "MCP Course")
- **Lesson filtering**: Can search within specific lesson numbers

#### 5. AI Integration
- **System prompt** in `AIGenerator` defines tool usage patterns
- **One search per query maximum** to avoid API overhead
- **Direct answers** for general knowledge, **search-first** for course-specific questions
- **Tool-based approach**: Claude autonomously decides when to search vs. direct answer

### Data Flow
Documents (`.txt` files in `docs/`) → DocumentProcessor → Course + CourseChunks → VectorStore → Available for semantic search via tools → Claude generates contextual responses

### Session Management
- **In-memory sessions** with unique IDs
- **2-exchange history limit** (configurable in `config.py`)
- **Auto-cleanup** when reaching history limits

### Configuration
Key settings in `backend/config.py`:
- `CHUNK_SIZE`: 800 characters
- `CHUNK_OVERLAP`: 100 characters  
- `MAX_RESULTS`: 5 search results
- `MAX_HISTORY`: 2 conversation exchanges
- `ANTHROPIC_MODEL`: claude-sonnet-4-20250514

This architecture enables intelligent document Q&A where Claude can leverage both its training knowledge and course-specific content through semantic search when appropriate.
- always use uv package manager when operating