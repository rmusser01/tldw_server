"""
Pytest fixtures and configuration for RAG tests.

Provides shared fixtures, mocks, and utilities for testing RAG functionality.
"""

import asyncio
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from uuid import uuid4
from typing import Dict, List, Any, Optional

import pytest
from unittest.mock import Mock, AsyncMock, MagicMock

from tldw_Server_API.app.core.DB_Management.Media_DB_v2 import MediaDatabase
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User
from tldw_Server_API.app.core.RAG.rag_service.types import DataSource, Document
from tldw_Server_API.app.core.RAG.rag_service.config import RAGConfig


# Test user fixtures

@pytest.fixture
def test_user():
    """Create a standard test user."""
    return User(
        id=12345,
        username="test_user",
        email="test@example.com",
        is_active=True
    )


@pytest.fixture
def test_users():
    """Create multiple test users."""
    return [
        User(id=1, username="user1", email="user1@test.com", is_active=True),
        User(id=2, username="user2", email="user2@test.com", is_active=True),
        User(id=3, username="user3", email="user3@test.com", is_active=False)
    ]


# Database fixtures

@pytest.fixture
def temp_db_dir():
    """Create a temporary directory for test databases."""
    temp_dir = tempfile.mkdtemp(prefix="rag_test_")
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_media_db():
    """Create a mock MediaDatabase with test data."""
    db = Mock(spec=MediaDatabase)
    
    # Mock search methods
    db.search_media_items = Mock(return_value=[
        {
            "id": 1,
            "title": "Test Document 1",
            "content": "This is a test document about RAG and machine learning.",
            "url": "https://example.com/doc1",
            "type": "article",
            "created_at": "2024-01-01T00:00:00Z",
            "media_metadata": {"author": "Test Author"}
        },
        {
            "id": 2,
            "title": "Test Video",
            "content": "Transcript of a video about vector databases.",
            "url": "https://example.com/video1",
            "type": "video",
            "created_at": "2024-02-01T00:00:00Z",
            "media_metadata": {"duration": "10:30"}
        }
    ])
    
    db.get_media_item = Mock(side_effect=lambda id: {
        "id": id,
        "title": f"Media Item {id}",
        "content": f"Content of media item {id}",
        "url": f"https://example.com/item{id}"
    })
    
    return db


@pytest.fixture
def mock_chacha_db():
    """Create a mock CharactersRAGDB with test data."""
    db = Mock(spec=CharactersRAGDB)
    
    # Mock note methods
    db.get_all_notes = Mock(return_value=[
        {
            "id": "note1",
            "title": "RAG Implementation Notes",
            "content": "Important notes about implementing RAG systems.",
            "created_at": "2024-01-15T00:00:00Z",
            "tags": ["rag", "implementation"]
        },
        {
            "id": "note2",
            "title": "Meeting Notes",
            "content": "Discussed vector database options for RAG.",
            "created_at": "2024-02-01T00:00:00Z",
            "tags": ["meeting", "vectors"]
        }
    ])
    
    # Mock character methods
    db.get_all_character_cards = Mock(return_value=[
        {
            "id": 1,
            "name": "AI Assistant",
            "description": "Helpful AI assistant",
            "tags": json.dumps(["ai", "helpful"])
        }
    ])
    
    # Mock conversation methods
    db.get_all_conversations = Mock(return_value=[
        {
            "conversation_id": str(uuid4()),
            "character_id": 1,
            "conversation_name": "Test Conversation"
        }
    ])
    
    return db


# RAG Service fixtures

@pytest.fixture
def mock_rag_config():
    """Create a mock RAG configuration."""
    config = RAGConfig()
    
    # Set test-friendly values
    config.batch_size = 10
    config.num_workers = 2
    config.log_level = "INFO"
    config.log_performance_metrics = True
    
    # Cache config
    config.cache.enable_cache = True
    config.cache.max_cache_size = 100
    config.cache.cache_ttl = 300
    
    # Retriever config
    config.retriever.fts_top_k = 5
    config.retriever.vector_top_k = 5
    config.retriever.hybrid_alpha = 0.5
    
    # Processor config
    config.processor.enable_reranking = True
    config.processor.max_context_length = 2048
    
    # Generator config
    config.generator.default_model = "gpt-3.5-turbo"
    config.generator.enable_streaming = True
    
    return config


@pytest.fixture
def mock_rag_service():
    """Create a mock RAG service."""
    service = AsyncMock()
    
    # Mock search method
    async def mock_search(query, sources=None, filters=None, **kwargs):
        return [
            Mock(
                source=DataSource.MEDIA_DB,
                documents=[
                    Document(
                        id="doc1",
                        content=f"Search result for: {query}",
                        metadata={"title": "Result 1"},
                        source=DataSource.MEDIA_DB,
                        relevance_score=0.95
                    )
                ]
            )
        ]
    
    service.search = AsyncMock(side_effect=mock_search)
    
    # Mock generate_answer method
    async def mock_generate(query, **kwargs):
        return {
            "answer": f"Generated answer for: {query}",
            "sources": [
                {
                    "id": "src1",
                    "source": "MEDIA_DB",
                    "title": "Source Document",
                    "score": 0.9,
                    "snippet": "Relevant snippet...",
                    "metadata": {}
                }
            ],
            "context_size": 500,
            "metadata": {"model": "gpt-3.5-turbo"}
        }
    
    service.generate_answer = AsyncMock(side_effect=mock_generate)
    
    # Mock streaming method
    async def mock_stream(query, **kwargs):
        yield {"type": "content", "content": "Streaming "}
        yield {"type": "content", "content": "answer"}
        yield {"type": "citation", "citation": {"id": "1", "source": "test"}}
    
    service.generate_answer_stream = mock_stream
    
    return service


# Sample data generators

@pytest.fixture
def sample_media_items():
    """Generate sample media items for testing."""
    return [
        {
            "id": i,
            "title": f"Document {i}",
            "content": f"This is the content of document {i}. It contains information about topic {i % 3}.",
            "url": f"https://example.com/doc{i}",
            "type": "article" if i % 2 == 0 else "video",
            "created_at": f"2024-01-{i:02d}T00:00:00Z",
            "media_metadata": {
                "author": f"Author {i % 5}",
                "tags": [f"tag{i}", f"category{i % 3}"]
            }
        }
        for i in range(1, 11)
    ]


@pytest.fixture
def sample_notes():
    """Generate sample notes for testing."""
    topics = ["RAG", "Machine Learning", "Vectors", "Embeddings", "Search"]
    return [
        {
            "id": f"note_{i}",
            "title": f"{topics[i % len(topics)]} Notes {i}",
            "content": f"These are notes about {topics[i % len(topics)]}. Important point number {i}.",
            "created_at": f"2024-02-{i:02d}T00:00:00Z",
            "tags": [topics[i % len(topics)].lower(), "notes"]
        }
        for i in range(1, 16)
    ]


@pytest.fixture
def sample_conversations():
    """Generate sample conversations for testing."""
    conversations = []
    for i in range(1, 6):
        conv_id = str(uuid4())
        messages = []
        for j in range(1, 4):
            messages.append((
                f"Question {j} in conversation {i}",
                f"Answer {j} in conversation {i}"
            ))
        
        conversations.append({
            "conversation_id": conv_id,
            "character_id": i % 3 + 1,
            "conversation_name": f"Conversation {i}",
            "messages": messages
        })
    
    return conversations


# Mock LLM fixtures

@pytest.fixture
def mock_llm_handler():
    """Create a mock LLM handler."""
    handler = AsyncMock()
    
    async def generate_response(prompt, **kwargs):
        return {
            "response": f"LLM response to: {prompt[:50]}...",
            "model": kwargs.get("model", "gpt-3.5-turbo"),
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150
            }
        }
    
    handler.generate = AsyncMock(side_effect=generate_response)
    
    async def stream_response(prompt, **kwargs):
        words = f"Streaming response to: {prompt}".split()
        for word in words:
            yield word + " "
    
    handler.stream = stream_response
    
    return handler


# Utility functions

@pytest.fixture
def create_test_embeddings():
    """Create test embeddings for documents."""
    def _create_embeddings(texts: List[str]) -> List[List[float]]:
        """Create fake embeddings based on text hash."""
        embeddings = []
        for text in texts:
            # Create a deterministic "embedding" based on text
            hash_val = hash(text) % 1000000
            embedding = [
                float((hash_val + i) % 1000) / 1000
                for i in range(384)  # Standard embedding size
            ]
            embeddings.append(embedding)
        return embeddings
    
    return _create_embeddings


@pytest.fixture
def async_context():
    """Provide an async context for tests."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()


# Mock API response fixtures

@pytest.fixture
def mock_search_response():
    """Create a mock search API response."""
    return {
        "querystring_echo": "test query",
        "results": [
            {
                "id": "result1",
                "score": 0.95,
                "title": "Test Result 1",
                "snippet": "This is a test result...",
                "metadata": {"source": "MEDIA_DB"}
            }
        ],
        "total_results": 1,
        "debug_info": {
            "search_mode_used": "advanced",
            "sources_searched": ["MEDIA_DB", "NOTES"]
        }
    }


@pytest.fixture
def mock_agent_response():
    """Create a mock agent API response."""
    return {
        "conversation_id": str(uuid4()),
        "response_message": {
            "role": "assistant",
            "content": "This is a generated response based on the retrieved context."
        },
        "citations": [
            {
                "source_name": "Test Document",
                "document_id": "doc1",
                "content": "Relevant excerpt...",
                "metadata": {"page": 1}
            }
        ],
        "debug_info": {
            "mode": "rag",
            "sources_used": ["MEDIA_DB", "NOTES"],
            "context_size": 1500
        }
    }


# Environment setup

@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch, temp_db_dir):
    """Set up test environment variables."""
    monkeypatch.setenv("USER_DB_BASE_DIR", str(temp_db_dir))
    monkeypatch.setenv("APP_MODE", "multi")
    monkeypatch.setenv("JWT_SECRET_KEY", "test_secret_key")
    
    # Clear any cached services
    from tldw_Server_API.app.api.v1.endpoints.rag import _user_rag_services
    _user_rag_services.clear()


# Performance testing fixtures

@pytest.fixture
def performance_timer():
    """Utility for timing operations."""
    import time
    
    class Timer:
        def __init__(self):
            self.times = {}
        
        def start(self, name: str):
            self.times[name] = {"start": time.time()}
        
        def stop(self, name: str):
            if name in self.times:
                self.times[name]["end"] = time.time()
                self.times[name]["duration"] = self.times[name]["end"] - self.times[name]["start"]
        
        def get_duration(self, name: str) -> Optional[float]:
            if name in self.times and "duration" in self.times[name]:
                return self.times[name]["duration"]
            return None
        
        def get_all_durations(self) -> Dict[str, float]:
            return {
                name: data.get("duration", 0)
                for name, data in self.times.items()
                if "duration" in data
            }
    
    return Timer()


if __name__ == "__main__":
    print("RAG test fixtures loaded successfully")