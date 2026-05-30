# 🎓 AI Academic Advisor — MCP Memory System

A production-grade AI Academic Advisor with persistent, context-aware memory using the **Memory, Control, and Process (MCP)** architecture pattern. Combines SQLite (structured data) and ChromaDB (semantic vector search) for hybrid long-term memory.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Docker Compose                             │
│  ┌────────────────┐      HTTP       ┌──────────────────────┐    │
│  │   LLM Agent    │ ←─────────────→ │    MCP Server        │    │
│  │ (Claude/Ollama)│   Tool Calls    │    (FastAPI :8000)   │    │
│  └────────────────┘                 └──────────────────────┘    │
│                                          │           │          │
│                                    ┌─────▼──┐    ┌────▼──────┐  │
│                                    │ SQLite │    │ ChromaDB  │  │
│                                    │Structured│  │ Vectors   │  │
│                                    └─────────┘   └───────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

See `docs/memory_architecture.png` for the full architecture diagram.

---

## 📋 Features

- **Persistent Memory**: Conversations, preferences, and milestones survive across sessions
- **Semantic RAG**: ChromaDB vector search finds contextually relevant past memories
- **Structured Queries**: SQLite for exact lookups (last N turns, all milestones, etc.)
- **Hybrid Memory**: Best of both worlds — relational + vector databases
- **Pydantic v2 Validation**: All data strictly validated before persistence
- **Idempotent Writes**: Duplicate data handled gracefully (upsert pattern)
- **Docker-first**: Single `docker-compose up` starts everything
- **LLM Flexible**: Works with Claude API or Ollama (local LLMs)

---

## 📁 Project Structure

```
ai-academic-advisor/
├── docker-compose.yml          # Orchestrates all services
├── .env.example                # Environment variable template
├── submission.json             # Test data for evaluation
├── README.md                   # This file
├── docs/
│   └── memory_architecture.png # System architecture diagram
├── data/                       # Persisted data (created at runtime)
│   ├── advisor_memory.db       # SQLite database
│   └── chroma_db/              # ChromaDB vector store
├── mcp_server/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── app/
│       ├── __init__.py
│       ├── main.py             # FastAPI application & endpoints
│       ├── memory_schemas.py   # Pydantic v2 models
│       ├── database.py         # SQLAlchemy ORM + CRUD operations
│       ├── vector_store.py     # ChromaDB integration
│       └── tools.py            # MCP tool implementations
└── agent/
    ├── Dockerfile
    ├── requirements.txt
    └── agent.py                # AI agent with tool-calling loop
```

---

## 🚀 Quick Start

### Prerequisites

- Docker Desktop (or Docker Engine + Docker Compose)
- 4GB+ RAM recommended (for embedding model)
- Optional: Anthropic API key for Claude-powered responses

### Step 1: Clone / Download the Project

```bash
# If from a git repo:
git clone <repo-url>
cd ai-academic-advisor

# If from zip:
unzip ai-academic-advisor.zip
cd ai-academic-advisor
```

### Step 2: Configure Environment

```bash
cp .env.example .env
```

Edit `.env` if needed. To use Claude API:
```ini
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

### Step 3: Start All Services

```bash
docker-compose up --build
```

This will:
1. Build the MCP server image (downloads embedding model ~90MB)
2. Build the agent image
3. Start both services
4. Initialize SQLite tables and ChromaDB collection
5. Wait for health check to pass before starting the agent

**First build takes 3-8 minutes** due to PyTorch and embedding model download.
Subsequent starts are much faster (< 30 seconds).

### Step 4: Interact with the Agent

The agent runs interactively in the Docker container. Access it via:

```bash
docker-compose logs -f agent
# or for interactive mode:
docker attach ai-academic-advisor_agent_1
```

Or run the agent locally (with MCP server running in Docker):

```bash
cd agent
pip install -r requirements.txt
MCP_SERVER_URL=http://localhost:8000 python agent.py
```

---

## 🔌 MCP Server API Reference

Base URL: `http://localhost:8000`

### Health Check
```
GET /health
→ {"status": "ok"}
```

### List Available Tools
```
GET /tools
→ {"tools": [{"name": "memory_write", ...}, ...]}
```

### memory_write — Save a Memory
```http
POST /invoke/memory_write
Content-Type: application/json

{
  "memory_type": "conversation",
  "data": {
    "user_id": "student_001",
    "turn_id": 1,
    "role": "user",
    "content": "I want to major in computer science."
  }
}

→ 201 Created
{"status": "success", "memory_id": "conv_student_001_turn_1"}
```

Supported `memory_type` values:
- `"conversation"` → saves to `conversations` table + ChromaDB
- `"preference"` → saves to `user_preferences` table + ChromaDB
- `"milestone"` → saves to `milestones` table + ChromaDB

### memory_read — Structured Query
```http
POST /invoke/memory_read
Content-Type: application/json

{
  "user_id": "student_001",
  "query_type": "last_n_turns",
  "params": {"n": 5}
}

→ 200 OK
{"results": [...]}
```

Supported `query_type` values:
- `"last_n_turns"` — last N conversation turns (params: `{"n": 5}`)
- `"all_preferences"` — all user preferences
- `"all_milestones"` — all academic milestones

### memory_retrieve_by_context — Semantic Search
```http
POST /invoke/memory_retrieve_by_context
Content-Type: application/json

{
  "user_id": "student_001",
  "query_text": "What subjects does the student find interesting?",
  "top_k": 3
}

→ 200 OK
{
  "results": [
    {
      "content": "I really enjoy machine learning and data analysis.",
      "metadata": {"user_id": "student_001", "role": "user", ...},
      "score": 0.8912
    }
  ]
}
```

### Debug Endpoints
```
GET /debug/vector-count  → {"total_vectors": 42}
GET /debug/db-stats      → {"conversations": 20, "user_preferences": 3, "milestones": 5}
```

### Interactive API Docs
```
http://localhost:8000/docs    (Swagger UI)
http://localhost:8000/redoc   (ReDoc)
```

---

## 🧪 Testing the System

### Quick Smoke Test (curl)

```bash
# 1. Health check
curl http://localhost:8000/health

# 2. Write a conversation
curl -X POST http://localhost:8000/invoke/memory_write \
  -H "Content-Type: application/json" \
  -d '{
    "memory_type": "conversation",
    "data": {
      "user_id": "test_user",
      "turn_id": 1,
      "role": "user",
      "content": "I am interested in quantum computing."
    }
  }'

# 3. Write another conversation
curl -X POST http://localhost:8000/invoke/memory_write \
  -H "Content-Type: application/json" \
  -d '{
    "memory_type": "conversation",
    "data": {
      "user_id": "test_user",
      "turn_id": 2,
      "role": "user",
      "content": "I dislike waking up early for 8 AM classes."
    }
  }'

# 4. Read last 5 turns
curl -X POST http://localhost:8000/invoke/memory_read \
  -H "Content-Type: application/json" \
  -d '{"user_id": "test_user", "query_type": "last_n_turns", "params": {"n": 5}}'

# 5. Semantic search
curl -X POST http://localhost:8000/invoke/memory_retrieve_by_context \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "test_user",
    "query_text": "What topics does this student find interesting?",
    "top_k": 1
  }'
```

### Load Test Data (submission.json)

```bash
python3 - << 'EOF'
import json, httpx

with open("submission.json") as f:
    data = json.load(f)

base_url = "http://localhost:8000"
user_id = data["testUserId"]

for i, turn in enumerate(data["testConversation"]):
    resp = httpx.post(f"{base_url}/invoke/memory_write", json={
        "memory_type": "conversation",
        "data": {
            "user_id": user_id,
            "turn_id": i + 1,
            "role": turn["role"],
            "content": turn["content"]
        }
    })
    print(f"Turn {i+1}: {resp.status_code} - {resp.json()}")

print("\nVector count:", httpx.get(f"{base_url}/debug/vector-count").json())
EOF
```

---

## 🔧 Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `DB_PATH` | `/app/data/advisor_memory.db` | SQLite database path |
| `CHROMA_DB_PATH` | `/app/data/chroma_db` | ChromaDB persistence directory |
| `EMBEDDING_MODEL_NAME` | `all-MiniLM-L6-v2` | Sentence transformer model |
| `ANTHROPIC_API_KEY` | *(empty)* | Claude API key (optional) |
| `MCP_SERVER_URL` | `http://mcp_server:8000` | Agent → MCP server URL |

---

## 🗄️ Database Schema

### conversations
| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | Auto-increment |
| user_id | TEXT | User identifier (indexed) |
| turn_id | INTEGER | Conversation turn number |
| role | TEXT | 'user' or 'assistant' |
| content | TEXT | Message content |
| timestamp | DATETIME | When the turn occurred |

### user_preferences
| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | Auto-increment |
| user_id | TEXT UNIQUE | User identifier |
| preferences | TEXT | JSON-encoded preferences dict |

### milestones
| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | Auto-increment |
| user_id | TEXT | User identifier |
| milestone_id | TEXT UNIQUE | Milestone UUID |
| description | TEXT | Milestone description |
| status | TEXT | 'planned', 'in-progress', 'completed' |
| date_achieved | DATETIME | Completion date (nullable) |

---

## 🐛 Troubleshooting

**Build takes too long?**
The first build downloads PyTorch (~200MB) and the embedding model (~90MB). Subsequent builds use Docker layer cache.

**Agent can't connect to MCP server?**
```bash
# Check MCP server is running
docker-compose ps
# Check logs
docker-compose logs mcp_server
```

**Embedding model download fails?**
The model downloads during `docker build`. Ensure you have internet access. The model is cached in the Docker image after first build.

**Out of memory?**
Increase Docker Desktop memory to at least 4GB in Settings → Resources.

**Reset all data?**
```bash
docker-compose down
rm -rf data/
docker-compose up
```

---

## 🛠️ Development

Run services locally (without Docker):

```bash
# Terminal 1: Start MCP server
cd mcp_server
pip install -r requirements.txt
DB_PATH=./data/advisor.db CHROMA_DB_PATH=./data/chroma uvicorn app.main:app --reload

# Terminal 2: Run agent
cd agent
pip install -r requirements.txt
MCP_SERVER_URL=http://localhost:8000 python agent.py
```

---

## 📐 Architecture Decisions

**Why SQLite + ChromaDB?**
SQLite excels at structured queries (give me the last 5 turns for user X). ChromaDB excels at semantic similarity (what did this user say about quantum computing?). Together they provide complementary query capabilities.

**Why sentence-transformers all-MiniLM-L6-v2?**
It's small (90MB), fast on CPU, and produces high-quality 384-dimensional embeddings. No GPU required.

**Why FastAPI?**
Async-ready, automatic OpenAPI docs, excellent Pydantic integration, and high performance for REST APIs.

**Why upsert over insert?**
Idempotent writes prevent duplicate data if the agent retries a failed tool call — a common scenario in agentic systems.
