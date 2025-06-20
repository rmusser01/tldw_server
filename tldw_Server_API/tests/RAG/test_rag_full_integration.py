# test_rag_full_integration.py
# Comprehensive integration tests for the RAG pipeline
import asyncio
import json
import os
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
from unittest.mock import patch, Mock
import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient, ASGITransport
import sqlite3

from tldw_Server_API.app.main import app
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User
from tldw_Server_API.app.core.config import settings
from tldw_Server_API.app.core.DB_Management.Media_DB_v2 import MediaDatabase
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.RAG.rag_service.integration import RAGService
from tldw_Server_API.app.api.v1.endpoints.rag import _user_rag_services


class TestRAGFullIntegration:
    """Full integration tests for the RAG pipeline."""
    
    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Setup and cleanup for each test."""
        # Create temporary directories
        self.temp_dir = tempfile.mkdtemp()
        self.user_base_dir = Path(self.temp_dir) / "users"
        self.user_base_dir.mkdir(exist_ok=True)
        
        # Clear the RAG service cache
        _user_rag_services.clear()
        
        # Patch settings - settings is a dict-like object, so we need to mock differently
        self.original_user_db_dir = settings.get("USER_DB_BASE_DIR", "/tmp/users")
        settings["USER_DB_BASE_DIR"] = str(self.user_base_dir)
        
        yield
        
        # Cleanup
        settings["USER_DB_BASE_DIR"] = self.original_user_db_dir
        _user_rag_services.clear()
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def test_users(self):
        """Create test users."""
        return [
            User(id=1, username="user1", email="user1@test.com", is_active=True),
            User(id=2, username="user2", email="user2@test.com", is_active=True)
        ]
    
    @pytest.fixture
    def mock_auth(self, test_users):
        """Mock authentication to return test users."""
        async def get_user(user_id: int):
            return next((u for u in test_users if u.id == user_id), None)
        return get_user
    
    def create_test_media_db(self, user_id: int) -> Path:
        """Create a test media database with sample data."""
        user_dir = self.user_base_dir / str(user_id)
        user_dir.mkdir(exist_ok=True)
        db_path = user_dir / "user_media_library.sqlite"
        
        # Create database and add test data
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Create media table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS Media (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT,
                content TEXT,
                transcription_model TEXT,
                creation_date TEXT,
                media_type TEXT,
                author TEXT,
                ingestion_date TEXT
            )
        """)
        
        # Add test media entries
        test_data = [
            ("Introduction to RAG", "Retrieval-Augmented Generation (RAG) is a powerful technique that combines retrieval and generation.", "test-model", "2024-01-01", "article", "AI Expert", datetime.now().isoformat()),
            ("Machine Learning Basics", "Machine learning is a subset of artificial intelligence that enables systems to learn from data.", "test-model", "2024-01-02", "video", "ML Teacher", datetime.now().isoformat()),
            ("Python Programming Guide", "Python is a versatile programming language used for web development, data science, and automation.", "test-model", "2024-01-03", "document", "Code Master", datetime.now().isoformat()),
            ("Deep Learning Tutorial", "Deep learning uses neural networks with multiple layers to progressively extract higher-level features.", "test-model", "2024-02-01", "video", "DL Researcher", datetime.now().isoformat()),
            ("Natural Language Processing", "NLP enables computers to understand, interpret, and generate human language in valuable ways.", "test-model", "2024-02-15", "article", "NLP Expert", datetime.now().isoformat())
        ]
        
        cursor.executemany(
            "INSERT INTO Media (title, content, transcription_model, creation_date, media_type, author, ingestion_date) VALUES (?, ?, ?, ?, ?, ?, ?)",
            test_data
        )
        
        conn.commit()
        conn.close()
        
        return db_path
    
    def create_test_chachanotes_db(self, user_id: int) -> Path:
        """Create a test ChaChaNotes database with sample data."""
        user_dir = self.user_base_dir / str(user_id)
        chacha_dir = user_dir / "chachanotes_user_dbs"
        chacha_dir.mkdir(exist_ok=True)
        db_path = chacha_dir / "user_chacha_notes_rag.sqlite"
        
        # Create database and add test data
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Create notes table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS Notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT,
                content TEXT,
                created_at TEXT,
                updated_at TEXT,
                keywords TEXT
            )
        """)
        
        # Add test notes
        test_notes = [
            ("RAG Implementation Notes", "Important: RAG requires both a retriever and a generator component. The retriever finds relevant documents.", datetime.now().isoformat(), datetime.now().isoformat(), "rag,implementation,retriever"),
            ("Meeting Notes: AI Strategy", "Discussed implementing RAG for our search system. Key benefits: better context, more accurate responses.", datetime.now().isoformat(), datetime.now().isoformat(), "meeting,ai,strategy,rag"),
            ("Research: Vector Databases", "ChromaDB and Pinecone are popular choices for vector storage in RAG systems. ChromaDB is open source.", datetime.now().isoformat(), datetime.now().isoformat(), "research,vector,database,chromadb")
        ]
        
        cursor.executemany(
            "INSERT INTO Notes (title, content, created_at, updated_at, keywords) VALUES (?, ?, ?, ?, ?)",
            test_notes
        )
        
        conn.commit()
        conn.close()
        
        return db_path
    
    def setup_user_environment(self, user_id: int):
        """Set up complete test environment for a user."""
        # Create databases
        media_db_path = self.create_test_media_db(user_id)
        chacha_db_path = self.create_test_chachanotes_db(user_id)
        
        # Create ChromaDB directory
        chroma_dir = self.user_base_dir / str(user_id) / "chroma"
        chroma_dir.mkdir(exist_ok=True)
        
        return {
            "media_db": media_db_path,
            "chacha_db": chacha_db_path,
            "chroma_dir": chroma_dir
        }
    
    @pytest.mark.asyncio
    async def test_search_endpoint_full_pipeline(self, test_users):
        """Test the search endpoint with real data."""
        # Setup environment for user1
        user1_env = self.setup_user_environment(test_users[0].id)
        
        # Mock authentication
        with patch('tldw_Server_API.app.core.AuthNZ.User_DB_Handling.get_request_user') as mock_auth:
            mock_auth.return_value = test_users[0]
            
            # Mock database dependencies
            with patch('tldw_Server_API.app.api.v1.API_Deps.DB_Deps.get_media_db_for_user') as mock_media_db:
                with patch('tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps.get_chacha_db_for_user') as mock_chacha_db:
                    mock_media_db.return_value = Mock(db_path=str(user1_env["media_db"]))
                    mock_chacha_db.return_value = Mock(db_path=str(user1_env["chacha_db"]))
                    
                    # Create test client
                    transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
                        # Test 1: Basic search
                        response = await client.post(
                            "/api/v1/retrieval_agent/search",
                            json={
                                "querystring": "RAG retrieval",
                                "search_mode": "basic",
                                "limit": 5
                            }
                        )
                        
                        assert response.status_code == 200
                        data = response.json()
                        assert "results" in data
                        assert len(data["results"]) > 0
                        assert any("RAG" in result["title"] or "RAG" in result["snippet"] for result in data["results"])
                        
                        # Test 2: Search with database selection
                        response = await client.post(
                            "/api/v1/retrieval_agent/search",
                            json={
                                "querystring": "implementation",
                                "search_databases": ["media_db", "notes"],
                                "search_mode": "custom",
                                "use_semantic_search": True
                            }
                        )
                        
                        assert response.status_code == 200
                        data = response.json()
                        assert len(data["results"]) > 0
                        
                        # Test 3: Search with date filters
                        response = await client.post(
                            "/api/v1/retrieval_agent/search",
                            json={
                                "querystring": "machine learning",
                                "date_range_start": "2024-01-01T00:00:00",
                                "date_range_end": "2024-01-31T23:59:59",
                                "search_mode": "advanced"
                            }
                        )
                        
                        assert response.status_code == 200
                        data = response.json()
                        # Should find entries from January 2024
                        assert len(data["results"]) > 0
    
    @pytest.mark.asyncio
    async def test_agent_endpoint_full_pipeline(self, test_users):
        """Test the agent endpoint with real RAG generation."""
        # Setup environment for user1
        user1_env = self.setup_user_environment(test_users[0].id)
        
        # Mock LLM response
        async def mock_llm_generate(*args, **kwargs):
            return {
                "content": "Based on the retrieved information about RAG, it combines retrieval and generation for better results.",
                "usage": {"total_tokens": 50}
            }
        
        with patch('tldw_Server_API.app.core.AuthNZ.User_DB_Handling.get_request_user') as mock_auth:
            mock_auth.return_value = test_users[0]
            
            with patch('tldw_Server_API.app.api.v1.API_Deps.DB_Deps.get_media_db_for_user') as mock_media_db:
                with patch('tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps.get_chacha_db_for_user') as mock_chacha_db:
                    mock_media_db.return_value = Mock(db_path=str(user1_env["media_db"]))
                    mock_chacha_db.return_value = Mock(db_path=str(user1_env["chacha_db"]))
                    
                    # Mock the LLM handler
                    with patch('tldw_Server_API.app.core.RAG.rag_service.generation.LLMHandler') as mock_llm_class:
                        mock_llm = Mock()
                        mock_llm.generate = mock_llm_generate
                        mock_llm_class.return_value = mock_llm
                        
                        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
                            # Test 1: Basic RAG generation
                            response = await client.post(
                                "/api/v1/retrieval_agent/agent",
                                json={
                                    "message": {
                                        "role": "user",
                                        "content": "What is RAG and how does it work?"
                                    },
                                    "mode": "rag",
                                    "search_mode": "custom",
                                    "rag_generation_config": {
                                        "model": "test-model",
                                        "temperature": 0.7,
                                        "max_tokens_to_sample": 500
                                    }
                                }
                            )
                            
                            assert response.status_code == 200
                            data = response.json()
                            assert "response_message" in data
                            assert data["response_message"]["role"] == "assistant"
                            assert len(data["response_message"]["content"]) > 0
                            assert "conversation_id" in data
                            
                            # Test 2: Research mode
                            response = await client.post(
                                "/api/v1/retrieval_agent/agent",
                                json={
                                    "message": {
                                        "role": "user",
                                        "content": "Research the benefits of using vector databases in RAG systems"
                                    },
                                    "mode": "research",
                                    "search_settings": {
                                        "search_databases": ["media_db", "notes"],
                                        "limit": 10
                                    }
                                }
                            )
                            
                            assert response.status_code == 200
                            data = response.json()
                            assert "response_message" in data
    
    @pytest.mark.asyncio
    async def test_multi_user_isolation(self, test_users):
        """Test that multiple users have isolated RAG services."""
        # Setup environments for both users
        user1_env = self.setup_user_environment(test_users[0].id)
        user2_env = self.setup_user_environment(test_users[1].id)
        
        # Add unique data to user2's database
        conn = sqlite3.connect(str(user2_env["media_db"]))
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO Media (title, content, transcription_model, creation_date, media_type, author, ingestion_date) VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("User2 Specific Document", "This document belongs only to user2 and should not appear in user1 searches.", "test-model", "2024-03-01", "document", "User2", datetime.now().isoformat())
        )
        conn.commit()
        conn.close()
        
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # Search as user1
            with patch('tldw_Server_API.app.core.AuthNZ.User_DB_Handling.get_request_user') as mock_auth:
                mock_auth.return_value = test_users[0]
                
                with patch('tldw_Server_API.app.api.v1.API_Deps.DB_Deps.get_media_db_for_user') as mock_media_db:
                    with patch('tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps.get_chacha_db_for_user') as mock_chacha_db:
                        mock_media_db.return_value = Mock(db_path=str(user1_env["media_db"]))
                        mock_chacha_db.return_value = Mock(db_path=str(user1_env["chacha_db"]))
                        
                        response = await client.post(
                            "/api/v1/retrieval_agent/search",
                            json={
                                "querystring": "User2 Specific Document",
                                "search_mode": "basic"
                            }
                        )
                        
                        assert response.status_code == 200
                        data = response.json()
                        # User1 should not find user2's document
                        assert not any("User2 Specific" in result["title"] for result in data["results"])
            
            # Search as user2
            with patch('tldw_Server_API.app.core.AuthNZ.User_DB_Handling.get_request_user') as mock_auth:
                mock_auth.return_value = test_users[1]
                
                with patch('tldw_Server_API.app.api.v1.API_Deps.DB_Deps.get_media_db_for_user') as mock_media_db:
                    with patch('tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps.get_chacha_db_for_user') as mock_chacha_db:
                        mock_media_db.return_value = Mock(db_path=str(user2_env["media_db"]))
                        mock_chacha_db.return_value = Mock(db_path=str(user2_env["chacha_db"]))
                        
                        response = await client.post(
                            "/api/v1/retrieval_agent/search",
                            json={
                                "querystring": "User2 Specific Document",
                                "search_mode": "basic"
                            }
                        )
                        
                        assert response.status_code == 200
                        data = response.json()
                        # User2 should find their own document
                        assert any("User2 Specific" in result["title"] for result in data["results"])
    
    @pytest.mark.asyncio
    async def test_streaming_response(self, test_users):
        """Test streaming response functionality."""
        # Setup environment
        user1_env = self.setup_user_environment(test_users[0].id)
        
        # Mock streaming LLM response
        async def mock_llm_stream(*args, **kwargs):
            chunks = [
                {"type": "content", "content": "Based on "},
                {"type": "content", "content": "the retrieved information, "},
                {"type": "content", "content": "RAG combines retrieval and generation."},
                {"type": "citation", "citation": {"source_name": "Introduction to RAG", "content": "RAG is a powerful technique"}}
            ]
            for chunk in chunks:
                yield chunk
        
        with patch('tldw_Server_API.app.core.AuthNZ.User_DB_Handling.get_request_user') as mock_auth:
            mock_auth.return_value = test_users[0]
            
            with patch('tldw_Server_API.app.api.v1.API_Deps.DB_Deps.get_media_db_for_user') as mock_media_db:
                with patch('tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps.get_chacha_db_for_user') as mock_chacha_db:
                    mock_media_db.return_value = Mock(db_path=str(user1_env["media_db"]))
                    mock_chacha_db.return_value = Mock(db_path=str(user1_env["chacha_db"]))
                    
                    # Mock the streaming generation
                    with patch('tldw_Server_API.app.core.RAG.rag_service.integration.RAGService.generate_answer_stream') as mock_stream:
                        mock_stream.return_value = mock_llm_stream()
                        
                        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
                            response = await client.post(
                                "/api/v1/retrieval_agent/agent",
                                json={
                                    "message": {
                                        "role": "user",
                                        "content": "What is RAG?"
                                    },
                                    "mode": "rag",
                                    "rag_generation_config": {
                                        "stream": True,
                                        "model": "test-model"
                                    }
                                }
                            )
                            
                            assert response.status_code == 200
                            assert response.headers["content-type"] == "text/event-stream"
                            
                            # Parse SSE events
                            events = []
                            for line in response.iter_lines():
                                if line.startswith("data: "):
                                    events.append(json.loads(line[6:]))
                            
                            # Verify we got all event types
                            assert any(e["type"] == "start" for e in events)
                            assert any(e["type"] == "content" for e in events)
                            assert any(e["type"] == "citation" for e in events)
                            assert any(e["type"] == "end" for e in events)
    
    @pytest.mark.asyncio
    async def test_error_handling(self, test_users):
        """Test error handling throughout the pipeline."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            with patch('tldw_Server_API.app.core.AuthNZ.User_DB_Handling.get_request_user') as mock_auth:
                mock_auth.return_value = test_users[0]
                
                # Test 1: Empty message
                response = await client.post(
                    "/retrieval/agent",
                    json={
                        "message": {
                            "role": "user",
                            "content": ""
                        }
                    }
                )
                assert response.status_code == 422
                
                # Test 2: Invalid search mode
                response = await client.post(
                    "/retrieval/search",
                    json={
                        "querystring": "test",
                        "search_mode": "invalid_mode"
                    }
                )
                assert response.status_code == 422
                
                # Test 3: Database error simulation
                with patch('tldw_Server_API.app.api.v1.API_Deps.DB_Deps.get_media_db_for_user') as mock_media_db:
                    mock_media_db.side_effect = Exception("Database connection failed")
                    
                    response = await client.post(
                        "/retrieval/search",
                        json={
                            "querystring": "test",
                            "search_mode": "basic"
                        }
                    )
                    assert response.status_code == 500
    
    @pytest.mark.asyncio
    async def test_conversation_context(self, test_users):
        """Test conversation context handling."""
        user1_env = self.setup_user_environment(test_users[0].id)
        
        with patch('tldw_Server_API.app.core.AuthNZ.User_DB_Handling.get_request_user') as mock_auth:
            mock_auth.return_value = test_users[0]
            
            with patch('tldw_Server_API.app.api.v1.API_Deps.DB_Deps.get_media_db_for_user') as mock_media_db:
                with patch('tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps.get_chacha_db_for_user') as mock_chacha_db:
                    mock_media_db.return_value = Mock(db_path=str(user1_env["media_db"]))
                    mock_chacha_db.return_value = Mock(db_path=str(user1_env["chacha_db"]))
                    
                    # Mock LLM to check conversation history
                    conversation_history_checked = False
                    
                    async def mock_llm_generate(*args, **kwargs):
                        nonlocal conversation_history_checked
                        if "conversation_history" in kwargs and kwargs["conversation_history"]:
                            conversation_history_checked = True
                        return {"content": "Response with context", "usage": {"total_tokens": 50}}
                    
                    with patch('tldw_Server_API.app.core.RAG.rag_service.generation.LLMHandler') as mock_llm_class:
                        mock_llm = Mock()
                        mock_llm.generate = mock_llm_generate
                        mock_llm_class.return_value = mock_llm
                        
                        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
                            # First message
                            response1 = await client.post(
                                "/retrieval/agent",
                                json={
                                    "message": {
                                        "role": "user",
                                        "content": "Tell me about RAG"
                                    },
                                    "mode": "rag"
                                }
                            )
                            
                            assert response1.status_code == 200
                            data1 = response1.json()
                            conversation_id = data1["conversation_id"]
                            
                            # Second message with conversation ID
                            response2 = await client.post(
                                "/retrieval/agent",
                                json={
                                    "message": {
                                        "role": "user",
                                        "content": "What are its benefits?"
                                    },
                                    "conversation_id": conversation_id,
                                    "mode": "rag"
                                }
                            )
                            
                            assert response2.status_code == 200
                            data2 = response2.json()
                            assert data2["conversation_id"] == conversation_id


class TestRAGPerformance:
    """Performance tests for the RAG pipeline."""
    
    @pytest.fixture
    def large_dataset_env(self):
        """Create a large dataset for performance testing."""
        temp_dir = tempfile.mkdtemp()
        user_dir = Path(temp_dir) / "1"
        user_dir.mkdir(exist_ok=True)
        db_path = user_dir / "user_media_library.sqlite"
        
        # Create database with many entries
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS Media (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT,
                content TEXT,
                transcription_model TEXT,
                creation_date TEXT,
                media_type TEXT,
                author TEXT,
                ingestion_date TEXT
            )
        """)
        
        # Add 1000 test entries
        test_data = []
        for i in range(1000):
            test_data.append((
                f"Document {i}: {['RAG', 'ML', 'AI', 'Python', 'Data'][i % 5]} Guide",
                f"This is test content for document {i}. It contains information about {['retrieval', 'machine learning', 'artificial intelligence', 'programming', 'data science'][i % 5]}.",
                "test-model",
                f"2024-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}",
                ["article", "video", "document"][i % 3],
                f"Author {i % 10}",
                datetime.now().isoformat()
            ))
        
        cursor.executemany(
            "INSERT INTO Media (title, content, transcription_model, creation_date, media_type, author, ingestion_date) VALUES (?, ?, ?, ?, ?, ?, ?)",
            test_data
        )
        
        conn.commit()
        conn.close()
        
        yield {
            "temp_dir": temp_dir,
            "db_path": db_path,
            "user_dir": user_dir
        }
        
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.mark.asyncio
    async def test_search_performance(self, large_dataset_env):
        """Test search performance with large dataset."""
        import time
        
        user = User(id=1, username="perftest", email="perf@test.com", is_active=True)
        
        with patch('tldw_Server_API.app.core.AuthNZ.User_DB_Handling.get_request_user') as mock_auth:
            mock_auth.return_value = user
            
            with patch('tldw_Server_API.app.api.v1.API_Deps.DB_Deps.get_media_db_for_user') as mock_media_db:
                with patch('tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps.get_chacha_db_for_user') as mock_chacha_db:
                    # Temporarily set the user database directory
                    original_user_db_dir = settings.get("USER_DB_BASE_DIR", "/tmp/users")
                    settings["USER_DB_BASE_DIR"] = str(large_dataset_env["temp_dir"])
                    try:
                        mock_media_db.return_value = Mock(db_path=str(large_dataset_env["db_path"]))
                        mock_chacha_db.return_value = Mock(db_path=str(large_dataset_env["user_dir"] / "chachanotes_user_dbs" / "user_chacha_notes_rag.sqlite"))
                        
                        transport = ASGITransport(app=app)
                        async with AsyncClient(transport=transport, base_url="http://test") as client:
                            # Measure search time
                            start_time = time.time()
                            
                            response = await client.post(
                                "/retrieval/search",
                                json={
                                    "querystring": "machine learning",
                                    "search_mode": "basic",
                                    "limit": 50
                                }
                            )
                            
                            end_time = time.time()
                            search_time = end_time - start_time
                            
                            assert response.status_code == 200
                            data = response.json()
                            assert len(data["results"]) > 0
                            
                            # Performance assertion - search should complete within 2 seconds
                            assert search_time < 2.0, f"Search took {search_time:.2f} seconds, expected < 2.0s"
                            
                            print(f"Search performance: {search_time:.2f}s for {len(data['results'])} results")
                    finally:
                        settings["USER_DB_BASE_DIR"] = original_user_db_dir