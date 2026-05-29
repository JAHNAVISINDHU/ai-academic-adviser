"""
database.py
-----------
SQLAlchemy ORM setup for the AI Academic Advisor memory system.
Defines table models and provides session management.
"""

import os
import json
import logging
from datetime import datetime
from typing import Generator

from sqlalchemy import (
    create_engine, Column, String, Integer, Text, DateTime,
    Index, event
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from sqlalchemy.pool import StaticPool

logger = logging.getLogger(__name__)

# ─── Database Configuration ────────────────────────────────────────────────────

DB_PATH = os.getenv("DB_PATH", "/app/data/advisor_memory.db")
DATABASE_URL = f"sqlite:///{DB_PATH}"

# Ensure the data directory exists
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
    echo=False,
)

# Enable WAL mode for better concurrent read performance
@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# ─── ORM Models ───────────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


class ConversationModel(Base):
    """SQLAlchemy model for conversation turns."""
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(255), nullable=False, index=True)
    turn_id = Column(Integer, nullable=False)
    role = Column(String(50), nullable=False)
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Composite unique index to prevent duplicate turns
    __table_args__ = (
        Index("ix_conversations_user_turn", "user_id", "turn_id", unique=True),
        Index("ix_conversations_timestamp", "timestamp"),
    )

    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "turn_id": self.turn_id,
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }


class UserPreferencesModel(Base):
    """SQLAlchemy model for user preferences."""
    __tablename__ = "user_preferences"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(255), nullable=False, unique=True, index=True)
    preferences = Column(Text, nullable=False, default="{}")  # JSON stored as text

    def get_preferences(self) -> dict:
        try:
            return json.loads(self.preferences)
        except (json.JSONDecodeError, TypeError):
            return {}

    def set_preferences(self, prefs: dict):
        self.preferences = json.dumps(prefs)

    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "preferences": self.get_preferences(),
        }


class MilestoneModel(Base):
    """SQLAlchemy model for academic milestones."""
    __tablename__ = "milestones"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(255), nullable=False, index=True)
    milestone_id = Column(String(255), nullable=False, unique=True)
    description = Column(Text, nullable=False)
    status = Column(String(50), nullable=False)
    date_achieved = Column(DateTime, nullable=True)

    __table_args__ = (
        Index("ix_milestones_user_id", "user_id"),
        Index("ix_milestones_status", "status"),
    )

    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "milestone_id": self.milestone_id,
            "description": self.description,
            "status": self.status,
            "date_achieved": self.date_achieved.isoformat() if self.date_achieved else None,
        }


# ─── Database Initialization ──────────────────────────────────────────────────

def init_db():
    """Create all tables if they don't exist."""
    Base.metadata.create_all(bind=engine)
    logger.info(f"Database initialized at {DB_PATH}")


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency: yields a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ─── CRUD Operations ──────────────────────────────────────────────────────────

class ConversationCRUD:
    """CRUD operations for conversations."""

    @staticmethod
    def upsert(db: Session, user_id: str, turn_id: int, role: str, content: str,
               timestamp: datetime = None) -> ConversationModel:
        """Insert or update a conversation turn (idempotent)."""
        existing = db.query(ConversationModel).filter_by(
            user_id=user_id, turn_id=turn_id
        ).first()

        if existing:
            existing.role = role
            existing.content = content
            if timestamp:
                existing.timestamp = timestamp
            db.commit()
            db.refresh(existing)
            return existing

        record = ConversationModel(
            user_id=user_id,
            turn_id=turn_id,
            role=role,
            content=content,
            timestamp=timestamp or datetime.utcnow(),
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        return record

    @staticmethod
    def get_last_n_turns(db: Session, user_id: str, n: int) -> list[ConversationModel]:
        """Retrieve the last N conversation turns for a user."""
        return (
            db.query(ConversationModel)
            .filter(ConversationModel.user_id == user_id)
            .order_by(ConversationModel.timestamp.desc())
            .limit(n)
            .all()[::-1]  # Reverse to chronological order
        )

    @staticmethod
    def get_all(db: Session, user_id: str) -> list[ConversationModel]:
        return (
            db.query(ConversationModel)
            .filter(ConversationModel.user_id == user_id)
            .order_by(ConversationModel.timestamp.asc())
            .all()
        )


class UserPreferencesCRUD:
    """CRUD operations for user preferences."""

    @staticmethod
    def upsert(db: Session, user_id: str, preferences: dict) -> UserPreferencesModel:
        """Insert or update preferences for a user."""
        existing = db.query(UserPreferencesModel).filter_by(user_id=user_id).first()
        if existing:
            # Merge preferences
            current = existing.get_preferences()
            current.update(preferences)
            existing.set_preferences(current)
            db.commit()
            db.refresh(existing)
            return existing

        record = UserPreferencesModel(user_id=user_id)
        record.set_preferences(preferences)
        db.add(record)
        db.commit()
        db.refresh(record)
        return record

    @staticmethod
    def get(db: Session, user_id: str) -> UserPreferencesModel | None:
        return db.query(UserPreferencesModel).filter_by(user_id=user_id).first()


class MilestoneCRUD:
    """CRUD operations for milestones."""

    @staticmethod
    def upsert(db: Session, user_id: str, milestone_id: str, description: str,
               status: str, date_achieved: datetime = None) -> MilestoneModel:
        """Insert or update a milestone."""
        existing = db.query(MilestoneModel).filter_by(milestone_id=milestone_id).first()
        if existing:
            existing.description = description
            existing.status = status
            existing.date_achieved = date_achieved
            db.commit()
            db.refresh(existing)
            return existing

        record = MilestoneModel(
            user_id=user_id,
            milestone_id=milestone_id,
            description=description,
            status=status,
            date_achieved=date_achieved,
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        return record

    @staticmethod
    def get_all(db: Session, user_id: str) -> list[MilestoneModel]:
        return (
            db.query(MilestoneModel)
            .filter(MilestoneModel.user_id == user_id)
            .order_by(MilestoneModel.status)
            .all()
        )
