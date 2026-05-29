"""
tools.py
--------
MCP tool implementations for memory_write, memory_read, and memory_retrieve_by_context.
Each tool validates input, persists to the appropriate database(s), and returns results.
"""

import logging
import uuid
from datetime import datetime
from typing import Any, Dict

from sqlalchemy.orm import Session

from .database import (
    ConversationCRUD,
    UserPreferencesCRUD,
    MilestoneCRUD,
)
from .memory_schemas import (
    Conversation,
    UserPreferences,
    Milestone,
    MemoryWriteRequest,
    MemoryReadRequest,
    MemoryRetrieveByContextRequest,
)
from .vector_store import get_vector_store

logger = logging.getLogger(__name__)


# ─── memory_write ─────────────────────────────────────────────────────────────

def tool_memory_write(request: MemoryWriteRequest, db: Session) -> Dict[str, Any]:
    """
    Persist a memory object to SQLite and, for text content, to ChromaDB.

    Supported memory_types: 'conversation', 'preference', 'milestone'
    Returns: {'status': 'success', 'memory_id': str}
    """
    memory_type = request.memory_type
    data = request.data
    vs = get_vector_store()

    try:
        if memory_type == "conversation":
            # Validate with Pydantic
            conv = Conversation(**data)

            record = ConversationCRUD.upsert(
                db=db,
                user_id=conv.user_id,
                turn_id=conv.turn_id,
                role=conv.role,
                content=conv.content,
                timestamp=conv.timestamp,
            )

            # Index text content in ChromaDB for semantic search
            doc_id = f"conv_{conv.user_id}_turn_{conv.turn_id}"
            vs.add(
                doc_id=doc_id,
                text=conv.content,
                metadata={
                    "user_id": conv.user_id,
                    "turn_id": str(conv.turn_id),
                    "role": conv.role,
                    "memory_type": "conversation",
                    "timestamp": conv.timestamp.isoformat(),
                },
            )

            memory_id = doc_id
            logger.info(f"Written conversation: user={conv.user_id} turn={conv.turn_id}")

        elif memory_type == "preference":
            # Validate with Pydantic
            pref = UserPreferences(**data)

            UserPreferencesCRUD.upsert(
                db=db,
                user_id=pref.user_id,
                preferences=pref.preferences,
            )

            # Index preference summary in ChromaDB
            pref_text = f"User preferences: {pref.preferences}"
            doc_id = f"pref_{pref.user_id}"
            vs.add(
                doc_id=doc_id,
                text=pref_text,
                metadata={
                    "user_id": pref.user_id,
                    "memory_type": "preference",
                    "timestamp": datetime.utcnow().isoformat(),
                },
            )

            memory_id = doc_id
            logger.info(f"Written preferences: user={pref.user_id}")

        elif memory_type == "milestone":
            # Validate with Pydantic
            ms = Milestone(**data)

            MilestoneCRUD.upsert(
                db=db,
                user_id=ms.user_id,
                milestone_id=ms.milestone_id,
                description=ms.description,
                status=ms.status,
                date_achieved=ms.date_achieved,
            )

            # Index milestone in ChromaDB
            ms_text = f"Milestone: {ms.description} (status: {ms.status})"
            doc_id = f"milestone_{ms.milestone_id}"
            vs.add(
                doc_id=doc_id,
                text=ms_text,
                metadata={
                    "user_id": ms.user_id,
                    "milestone_id": ms.milestone_id,
                    "status": ms.status,
                    "memory_type": "milestone",
                    "timestamp": datetime.utcnow().isoformat(),
                },
            )

            memory_id = doc_id
            logger.info(f"Written milestone: user={ms.user_id} id={ms.milestone_id}")

        else:
            raise ValueError(f"Unknown memory_type: {memory_type}")

        return {"status": "success", "memory_id": memory_id}

    except Exception as e:
        logger.error(f"memory_write error: {e}", exc_info=True)
        raise


# ─── memory_read ──────────────────────────────────────────────────────────────

def tool_memory_read(request: MemoryReadRequest, db: Session) -> Dict[str, Any]:
    """
    Retrieve structured data from SQLite.

    Supported query_types:
    - 'last_n_turns': retrieves last N conversation turns (params: {'n': int})
    - 'all_preferences': retrieves all preferences for a user
    - 'all_milestones': retrieves all milestones for a user
    """
    user_id = request.user_id
    query_type = request.query_type
    params = request.params

    try:
        if query_type == "last_n_turns":
            n = int(params.get("n", 10))
            records = ConversationCRUD.get_last_n_turns(db, user_id, n)
            results = [r.to_dict() for r in records]

        elif query_type == "all_preferences":
            record = UserPreferencesCRUD.get(db, user_id)
            results = [record.to_dict()] if record else []

        elif query_type == "all_milestones":
            records = MilestoneCRUD.get_all(db, user_id)
            results = [r.to_dict() for r in records]

        elif query_type == "all_conversations":
            from .database import ConversationCRUD
            records = ConversationCRUD.get_all(db, user_id)
            results = [r.to_dict() for r in records]

        else:
            raise ValueError(f"Unknown query_type: {query_type}")

        logger.info(f"memory_read: {query_type} for user={user_id}, {len(results)} results")
        return {"results": results}

    except Exception as e:
        logger.error(f"memory_read error: {e}", exc_info=True)
        raise


# ─── memory_retrieve_by_context ───────────────────────────────────────────────

def tool_memory_retrieve_by_context(
    request: MemoryRetrieveByContextRequest,
    db: Session,
) -> Dict[str, Any]:
    """
    Perform semantic similarity search in ChromaDB.

    Takes a natural language query and returns the most semantically similar
    memories for the given user.
    """
    vs = get_vector_store()

    try:
        results = vs.query(
            query_text=request.query_text,
            user_id=request.user_id,
            top_k=request.top_k,
        )

        logger.info(
            f"memory_retrieve_by_context: query='{request.query_text[:50]}...' "
            f"user={request.user_id}, {len(results)} results"
        )
        return {"results": results}

    except Exception as e:
        logger.error(f"memory_retrieve_by_context error: {e}", exc_info=True)
        raise


# ─── Tool Registry ────────────────────────────────────────────────────────────

TOOL_DEFINITIONS = [
    {
        "name": "memory_write",
        "description": (
            "Persist a memory object (conversation turn, user preference, or academic milestone) "
            "to the relational database (SQLite). Text-based memories are also embedded and stored "
            "in the vector database (ChromaDB) for semantic search. Use this after each interaction "
            "to maintain long-term context."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "memory_type": {
                    "type": "string",
                    "enum": ["conversation", "preference", "milestone"],
                    "description": "The type of memory to write",
                },
                "data": {
                    "type": "object",
                    "description": "The memory payload matching the schema for the chosen memory_type",
                },
            },
            "required": ["memory_type", "data"],
        },
    },
    {
        "name": "memory_read",
        "description": (
            "Retrieve structured data from the relational database (SQLite). "
            "Supports querying the last N conversation turns, user preferences, "
            "or all academic milestones for a given user."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "user_id": {"type": "string", "description": "The user to query memories for"},
                "query_type": {
                    "type": "string",
                    "enum": ["last_n_turns", "all_preferences", "all_milestones", "all_conversations"],
                    "description": "Type of structured query to perform",
                },
                "params": {
                    "type": "object",
                    "description": "Query parameters, e.g. {'n': 5} for last_n_turns",
                },
            },
            "required": ["user_id", "query_type"],
        },
    },
    {
        "name": "memory_retrieve_by_context",
        "description": (
            "Perform semantic similarity search (RAG) in the vector database to find "
            "the most contextually relevant memories for a given query. Use this to "
            "surface relevant past interactions before generating a response."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "user_id": {"type": "string", "description": "Filter results to this user"},
                "query_text": {"type": "string", "description": "The natural language query to search by"},
                "top_k": {"type": "integer", "description": "Number of top results to return (default: 3)"},
            },
            "required": ["user_id", "query_text"],
        },
    },
]
