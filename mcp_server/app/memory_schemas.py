"""
memory_schemas.py
-----------------
Pydantic v2 models for the AI Academic Advisor memory system.
Defines the core data structures: Conversation, UserPreferences, Milestone.
"""

from pydantic import BaseModel, Field, field_validator
from datetime import datetime
from typing import List, Dict, Any, Optional
import uuid


class Conversation(BaseModel):
    """Represents a single turn in a conversation between user and assistant."""
    user_id: str = Field(..., description="Unique identifier for the user")
    turn_id: int = Field(..., description="Sequential turn number in the conversation")
    role: str = Field(..., description="Speaker role: 'user' or 'assistant'")
    content: str = Field(..., description="The text content of the conversation turn")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When this turn occurred")

    @field_validator("role")
    @classmethod
    def validate_role(cls, v: str) -> str:
        allowed = {"user", "assistant", "system"}
        if v not in allowed:
            raise ValueError(f"role must be one of {allowed}, got '{v}'")
        return v

    @field_validator("content")
    @classmethod
    def validate_content(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("content cannot be empty")
        return v.strip()

    model_config = {"from_attributes": True}


class UserPreferences(BaseModel):
    """Stores user-specific preferences and settings."""
    user_id: str = Field(..., description="Unique identifier for the user")
    preferences: Dict[str, Any] = Field(
        default_factory=dict,
        description="Key-value map of user preferences, e.g. {'favorite_subjects': ['Math', 'Art']}"
    )

    model_config = {"from_attributes": True}


class Milestone(BaseModel):
    """Tracks academic milestones and goals for a user."""
    user_id: str = Field(..., description="Unique identifier for the user")
    milestone_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this milestone"
    )
    description: str = Field(..., description="Description of the milestone")
    status: str = Field(..., description="Current status: 'completed', 'in-progress', or 'planned'")
    date_achieved: Optional[datetime] = Field(None, description="When the milestone was achieved (if completed)")

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str) -> str:
        allowed = {"completed", "in-progress", "planned"}
        if v not in allowed:
            raise ValueError(f"status must be one of {allowed}, got '{v}'")
        return v

    model_config = {"from_attributes": True}


# ─── Request / Response schemas for the API ────────────────────────────────────

class MemoryWriteRequest(BaseModel):
    """Request body for the memory_write tool endpoint."""
    memory_type: str = Field(..., description="Type of memory: 'conversation', 'preference', or 'milestone'")
    data: Dict[str, Any] = Field(..., description="The memory data payload matching the appropriate schema")

    @field_validator("memory_type")
    @classmethod
    def validate_memory_type(cls, v: str) -> str:
        allowed = {"conversation", "preference", "milestone"}
        if v not in allowed:
            raise ValueError(f"memory_type must be one of {allowed}, got '{v}'")
        return v


class MemoryWriteResponse(BaseModel):
    """Response from the memory_write tool."""
    status: str = "success"
    memory_id: str


class MemoryReadRequest(BaseModel):
    """Request body for the memory_read tool endpoint."""
    user_id: str = Field(..., description="The user to read memories for")
    query_type: str = Field(..., description="Type of query: 'last_n_turns', 'all_preferences', 'all_milestones'")
    params: Dict[str, Any] = Field(default_factory=dict, description="Query parameters, e.g. {'n': 5}")


class MemoryReadResponse(BaseModel):
    """Response from the memory_read tool."""
    results: List[Dict[str, Any]]


class MemoryRetrieveByContextRequest(BaseModel):
    """Request body for the memory_retrieve_by_context tool endpoint."""
    user_id: str = Field(..., description="The user to search memories for")
    query_text: str = Field(..., description="The semantic query text to search by")
    top_k: int = Field(default=3, ge=1, le=20, description="Number of top results to return")


class MemoryRetrieveByContextResponse(BaseModel):
    """Response from the memory_retrieve_by_context tool."""
    results: List[Dict[str, Any]]


class ToolInfo(BaseModel):
    """Describes an available MCP tool."""
    name: str
    description: str
    input_schema: Dict[str, Any] = Field(default_factory=dict)


class ToolsListResponse(BaseModel):
    """Response listing all available MCP tools."""
    tools: List[ToolInfo]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "ok"
