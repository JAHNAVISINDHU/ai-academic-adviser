"""
vector_store.py
---------------
ChromaDB vector store integration for semantic memory retrieval.
Uses sentence-transformers to embed text and supports similarity search.
"""

import os
import logging
from typing import List, Dict, Any

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "/app/data/chroma_db")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
COLLECTION_NAME = "academic_advisor_memories"


class VectorStore:
    """
    Manages ChromaDB vector storage for semantic memory retrieval.
    Uses a single shared collection with user_id metadata for filtering.
    """

    def __init__(self):
        os.makedirs(CHROMA_DB_PATH, exist_ok=True)

        logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

        logger.info(f"Initializing ChromaDB at: {CHROMA_DB_PATH}")
        self.client = chromadb.PersistentClient(
            path=CHROMA_DB_PATH,
            settings=Settings(anonymized_telemetry=False),
        )

        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            f"ChromaDB ready. Collection '{COLLECTION_NAME}' has "
            f"{self.collection.count()} vectors."
        )

    def _embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        embeddings = self.embedding_model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()

    def add(
        self,
        doc_id: str,
        text: str,
        metadata: Dict[str, Any],
    ) -> None:
        """
        Embed and store a text document in ChromaDB.

        Args:
            doc_id: Unique identifier for this document (e.g. "conv_user1_turn5")
            text: The text content to embed and store
            metadata: Additional metadata (must include 'user_id')
        """
        # Ensure all metadata values are strings/numbers (ChromaDB requirement)
        safe_metadata = {
            k: str(v) if not isinstance(v, (str, int, float, bool)) else v
            for k, v in metadata.items()
        }

        embeddings = self._embed([text])

        # Upsert to handle idempotent writes
        self.collection.upsert(
            ids=[doc_id],
            embeddings=embeddings,
            documents=[text],
            metadatas=[safe_metadata],
        )
        logger.debug(f"Stored vector for doc_id={doc_id}")

    def query(
        self,
        query_text: str,
        user_id: str,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic similarity search filtered by user_id.

        Args:
            query_text: The query to search for semantically similar memories
            user_id: Filter results to this user only
            top_k: Number of top results to return

        Returns:
            List of result dicts with 'content', 'metadata', and 'score'
        """
        query_embedding = self._embed([query_text])

        total = self.collection.count()
        if total == 0:
            return []

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=min(top_k, total),
            where={"user_id": user_id} if user_id else None,
            include=["documents", "metadatas", "distances"],
        )

        output = []
        if results and results.get("ids") and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                distance = results["distances"][0][i]
                # Convert cosine distance to similarity score (0-1, higher is better)
                score = float(1 - distance)
                output.append({
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "score": round(score, 4),
                })

        # Sort by score descending
        output.sort(key=lambda x: x["score"], reverse=True)
        return output

    def count(self) -> int:
        """Return total number of vectors in the collection."""
        return self.collection.count()

    def count_for_user(self, user_id: str) -> int:
        """Return number of vectors for a specific user."""
        try:
            results = self.collection.get(where={"user_id": user_id})
            return len(results["ids"]) if results and results.get("ids") else 0
        except Exception:
            return 0


# Singleton instance
_vector_store: VectorStore | None = None


def get_vector_store() -> VectorStore:
    """Get or create the singleton VectorStore instance."""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store
