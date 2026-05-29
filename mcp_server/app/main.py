"""
main.py
-------
FastAPI application for the AI Academic Advisor MCP Server.
Exposes RESTful endpoints for memory tools and health checks.
"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from .database import init_db, get_db
from .vector_store import get_vector_store
from .memory_schemas import (
    HealthResponse,
    ToolsListResponse,
    MemoryWriteRequest,
    MemoryWriteResponse,
    MemoryReadRequest,
    MemoryReadResponse,
    MemoryRetrieveByContextRequest,
    MemoryRetrieveByContextResponse,
)
from .tools import (
    tool_memory_write,
    tool_memory_read,
    tool_memory_retrieve_by_context,
    TOOL_DEFINITIONS,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ─── Lifespan: startup/shutdown tasks ─────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize databases on startup."""
    logger.info("Starting AI Academic Advisor MCP Server...")

    # Initialize SQLite tables
    init_db()
    logger.info("SQLite database initialized")

    # Initialize ChromaDB (eager load embedding model)
    vs = get_vector_store()
    logger.info(f"ChromaDB initialized with {vs.count()} vectors")

    logger.info("MCP Server ready!")
    yield
    logger.info("MCP Server shutting down...")


# ─── FastAPI App ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="AI Academic Advisor MCP Server",
    description=(
        "Memory, Control, and Process server providing persistent, context-aware "
        "memory for the AI Academic Advisor agent. Integrates SQLite (structured data) "
        "and ChromaDB (vector search) for hybrid memory retrieval."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Health Check ─────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check endpoint. Returns 200 OK when the service is running."""
    return HealthResponse(status="ok")


# ─── Tools Listing ────────────────────────────────────────────────────────────

@app.get("/tools", response_model=ToolsListResponse, tags=["MCP"])
async def list_tools():
    """List all available MCP memory tools with their descriptions and schemas."""
    return ToolsListResponse(tools=TOOL_DEFINITIONS)


# ─── Tool: memory_write ───────────────────────────────────────────────────────

@app.post(
    "/invoke/memory_write",
    response_model=MemoryWriteResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["MCP Tools"],
)
async def invoke_memory_write(
    request: MemoryWriteRequest,
    db: Session = Depends(get_db),
):
    """
    Write a memory to the persistent store.

    Persists structured data to SQLite and embeds text content in ChromaDB
    for semantic retrieval. Supports conversation turns, user preferences,
    and academic milestones.
    """
    try:
        result = tool_memory_write(request, db)
        return MemoryWriteResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"memory_write failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


# ─── Tool: memory_read ────────────────────────────────────────────────────────

@app.post(
    "/invoke/memory_read",
    response_model=MemoryReadResponse,
    status_code=status.HTTP_200_OK,
    tags=["MCP Tools"],
)
async def invoke_memory_read(
    request: MemoryReadRequest,
    db: Session = Depends(get_db),
):
    """
    Read structured memories from SQLite.

    Query types:
    - last_n_turns: retrieve the N most recent conversation turns
    - all_preferences: retrieve all user preferences
    - all_milestones: retrieve all academic milestones
    """
    try:
        result = tool_memory_read(request, db)
        return MemoryReadResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"memory_read failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


# ─── Tool: memory_retrieve_by_context ────────────────────────────────────────

@app.post(
    "/invoke/memory_retrieve_by_context",
    response_model=MemoryRetrieveByContextResponse,
    status_code=status.HTTP_200_OK,
    tags=["MCP Tools"],
)
async def invoke_memory_retrieve_by_context(
    request: MemoryRetrieveByContextRequest,
    db: Session = Depends(get_db),
):
    """
    Semantic memory retrieval using vector similarity search.

    Embeds the query text and returns the most semantically similar memories
    for the specified user from ChromaDB.
    """
    try:
        result = tool_memory_retrieve_by_context(request, db)
        return MemoryRetrieveByContextResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"memory_retrieve_by_context failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


# ─── Debug Endpoints (optional, useful for development) ───────────────────────

@app.get("/debug/vector-count", tags=["Debug"])
async def debug_vector_count():
    """Return the total number of vectors in ChromaDB."""
    vs = get_vector_store()
    return {"total_vectors": vs.count()}


@app.get("/debug/db-stats", tags=["Debug"])
async def debug_db_stats(db: Session = Depends(get_db)):
    """Return row counts for all SQLite tables."""
    from .database import ConversationModel, UserPreferencesModel, MilestoneModel
    return {
        "conversations": db.query(ConversationModel).count(),
        "user_preferences": db.query(UserPreferencesModel).count(),
        "milestones": db.query(MilestoneModel).count(),
    }
