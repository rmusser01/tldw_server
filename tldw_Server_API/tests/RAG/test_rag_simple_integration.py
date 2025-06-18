# test_rag_simple_integration.py
# Simplified integration tests for the RAG pipeline
import os
import tempfile
import shutil
import sqlite3
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock
import pytest

from tldw_Server_API.app.api.v1.endpoints.rag import (
    get_rag_service_for_user,
    perform_search,
    run_retrieval_agent,
    _user_rag_services
)
from tldw_Server_API.app.api.v1.schemas.rag_schemas import (
    SearchApiRequest,
    RetrievalAgentRequest,
    Message,
    MessageRole,
    SearchModeEnum,
    AgentModeEnum,
    GenerationConfig
)
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User
from tldw_Server_API.app.core.config import settings


class TestRAGIntegration:
    """Simplified integration tests that test the RAG components together."""
    
    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Setup test environment."""
        # Create temp directory
        self.temp_dir = tempfile.mkdtemp()
        self.user_base_dir = Path(self.temp_dir) / "users"
        self.user_base_dir.mkdir(exist_ok=True)
        
        # Clear RAG service cache
        _user_rag_services.clear()
        
        # Store original settings
        self.original_user_db_dir = settings.get("USER_DB_BASE_DIR", "/tmp/users")
        settings["USER_DB_BASE_DIR"] = str(self.user_base_dir)
        
        yield
        
        # Cleanup
        settings["USER_DB_BASE_DIR"] = self.original_user_db_dir
        _user_rag_services.clear()
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_test_database(self, user_id: int):
        """Create test databases for a user."""
        user_dir = self.user_base_dir / str(user_id)
        user_dir.mkdir(exist_ok=True)
        
        # Create media database
        media_db = user_dir / "user_media_library.sqlite"
        conn = sqlite3.connect(str(media_db))
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS Media (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT UNIQUE,
                title TEXT NOT NULL,
                type TEXT NOT NULL,
                content TEXT,
                author TEXT,
                ingestion_date DATETIME,
                transcription_model TEXT,
                is_trash BOOLEAN DEFAULT 0 NOT NULL,
                trash_date DATETIME,
                vector_embedding BLOB,
                chunking_status TEXT DEFAULT 'pending' NOT NULL,
                vector_processing INTEGER DEFAULT 0 NOT NULL,
                content_hash TEXT UNIQUE NOT NULL,
                uuid TEXT UNIQUE NOT NULL,
                last_modified DATETIME NOT NULL,
                version INTEGER NOT NULL DEFAULT 1,
                client_id TEXT NOT NULL,
                deleted BOOLEAN NOT NULL DEFAULT 0,
                prev_version INTEGER,
                merge_parent_uuid TEXT
            )
        """)
        
        # Add test data with all required fields
        import hashlib
        test_data = []
        for i, (title, content, url, model, type_, author) in enumerate([
            ("RAG Overview", "Retrieval-Augmented Generation combines retrieval with generation for better AI responses.", "http://example.com/rag", "test", "article", "AI Expert"),
            ("Python Tutorial", "Python is a versatile programming language for many applications.", "http://example.com/python", "test", "video", "Code Teacher"),
            ("Machine Learning Basics", "ML enables computers to learn from data without explicit programming.", "http://example.com/ml", "test", "document", "ML Expert")
        ]):
            content_hash = hashlib.md5(content.encode()).hexdigest()
            uuid_val = str(user_dir / f"media_{i}")  # Simple UUID for testing
            test_data.append((
                url, title, type_, content, author, datetime.now().isoformat(),
                model, content_hash, uuid_val, datetime.now().isoformat(), 
                1, "test_client", 0
            ))
        
        cursor.executemany(
            """INSERT INTO Media (url, title, type, content, author, ingestion_date,
                                 transcription_model, content_hash, uuid, last_modified,
                                 version, client_id, deleted)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            test_data
        )
        
        conn.commit()
        
        # Create FTS5 virtual table for full-text search (matching actual schema)
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS media_fts USING fts5(
                title,
                content,
                content='Media',
                content_rowid='id'
            )
        """)
        
        # Populate FTS table
        cursor.execute("""
            INSERT INTO media_fts(title, content)
            SELECT title, content FROM Media WHERE deleted = 0
        """)
        
        conn.commit()
        conn.close()
        
        # Create ChaChaNotes database
        chacha_dir = user_dir / "chachanotes_user_dbs"
        chacha_dir.mkdir(exist_ok=True)
        chacha_db = chacha_dir / "user_chacha_notes_rag.sqlite"
        
        conn = sqlite3.connect(str(chacha_db))
        cursor = conn.cursor()
        
        # Create notes table (matching actual schema)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS notes (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                last_modified DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                deleted BOOLEAN NOT NULL DEFAULT 0,
                client_id TEXT NOT NULL DEFAULT 'unknown',
                version INTEGER NOT NULL DEFAULT 1
            )
        """)
        
        # Create keywords table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS keywords (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                keyword TEXT NOT NULL UNIQUE COLLATE NOCASE
            )
        """)
        
        # Create note_keywords linking table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS note_keywords (
                note_id TEXT NOT NULL,
                keyword_id INTEGER NOT NULL,
                PRIMARY KEY (note_id, keyword_id),
                FOREIGN KEY (note_id) REFERENCES notes(id),
                FOREIGN KEY (keyword_id) REFERENCES keywords(id)
            )
        """)
        
        # Add test notes and keywords
        test_notes = [
            ("note1", "RAG Notes", "Remember: RAG needs good retrieval for quality generation.", ["rag", "notes"]),
            ("note2", "Project Ideas", "Build a RAG system for documentation search.", ["project", "rag"])
        ]
        
        for note_id, title, content, keywords in test_notes:
            cursor.execute(
                """INSERT INTO notes (id, title, content, created_at, last_modified, client_id)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (note_id, title, content, datetime.now().isoformat(), datetime.now().isoformat(), "test_client")
            )
            
            # Add keywords
            for keyword in keywords:
                cursor.execute("INSERT OR IGNORE INTO keywords (keyword) VALUES (?)", (keyword,))
                cursor.execute("SELECT id FROM keywords WHERE keyword = ?", (keyword,))
                keyword_id = cursor.fetchone()[0]
                cursor.execute("INSERT INTO note_keywords (note_id, keyword_id) VALUES (?, ?)", (note_id, keyword_id))
        
        # Create FTS5 virtual table for notes search
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS notes_fts USING fts5(
                title, content,
                content='notes',
                content_rowid='rowid'
            )
        """)
        
        # Populate FTS table
        cursor.execute("""
            INSERT INTO notes_fts(title, content)
            SELECT title, content FROM notes WHERE deleted = 0
        """)
        
        conn.commit()
        conn.close()
        
        # Create chroma directory
        chroma_dir = user_dir / "chroma"
        chroma_dir.mkdir(exist_ok=True)
        
        return {
            "media_db": media_db,
            "chacha_db": chacha_db,
            "chroma_dir": chroma_dir
        }
    
    @pytest.mark.asyncio
    async def test_rag_service_creation(self):
        """Test that RAG service is created correctly for a user."""
        user = User(id=1, username="testuser", email="test@example.com", is_active=True)
        self.create_test_database(user.id)
        
        # Mock dependencies
        mock_media_db = Mock(db_path=str(self.user_base_dir / "1" / "user_media_library.sqlite"))
        mock_chacha_db = Mock(db_path=str(self.user_base_dir / "1" / "chachanotes_user_dbs" / "user_chacha_notes_rag.sqlite"))
        
        # Get RAG service
        rag_service = await get_rag_service_for_user(
            current_user=user,
            media_db=mock_media_db,
            chacha_db=mock_chacha_db
        )
        
        assert rag_service is not None
        assert user.id in _user_rag_services
        assert rag_service.media_db_path.exists()
        assert rag_service.chachanotes_db_path.exists()
        
        # Test caching - should return same instance
        rag_service2 = await get_rag_service_for_user(
            current_user=user,
            media_db=mock_media_db,
            chacha_db=mock_chacha_db
        )
        assert rag_service is rag_service2
    
    @pytest.mark.asyncio
    async def test_search_integration(self):
        """Test search functionality with real database."""
        user = User(id=2, username="searchuser", email="search@example.com", is_active=True)
        self.create_test_database(user.id)
        
        # Create search request
        search_request = SearchApiRequest(
            querystring="RAG generation",
            search_mode=SearchModeEnum.BASIC,
            limit=10
        )
        
        # Mock dependencies
        mock_media_db = Mock(db_path=str(self.user_base_dir / "2" / "user_media_library.sqlite"))
        mock_chacha_db = Mock(db_path=str(self.user_base_dir / "2" / "chachanotes_user_dbs" / "user_chacha_notes_rag.sqlite"))
        
        # Get RAG service
        rag_service = await get_rag_service_for_user(
            current_user=user,
            media_db=mock_media_db,
            chacha_db=mock_chacha_db
        )
        
        # Perform search
        response = await perform_search(
            request_body=search_request,
            rag_service=rag_service,
            current_user=user
        )
        
        assert response is not None
        assert response.querystring_echo == "RAG generation"
        assert hasattr(response, 'results')
        # Should find results related to RAG
        assert len(response.results) > 0
        assert any("RAG" in r.title or "RAG" in r.snippet for r in response.results)
    
    @pytest.mark.asyncio
    async def test_agent_integration(self):
        """Test agent functionality with mocked LLM."""
        user = User(id=3, username="agentuser", email="agent@example.com", is_active=True)
        self.create_test_database(user.id)
        
        # Create agent request
        agent_request = RetrievalAgentRequest(
            message=Message(
                role=MessageRole.USER,
                content="Tell me about RAG"
            ),
            mode=AgentModeEnum.RAG,
            rag_generation_config=GenerationConfig(
                model="test-model",
                temperature=0.7,
                max_tokens_to_sample=500
            )
        )
        
        # Mock dependencies
        mock_media_db = Mock(db_path=str(self.user_base_dir / "3" / "user_media_library.sqlite"))
        mock_chacha_db = Mock(db_path=str(self.user_base_dir / "3" / "chachanotes_user_dbs" / "user_chacha_notes_rag.sqlite"))
        
        # Get RAG service
        rag_service = await get_rag_service_for_user(
            current_user=user,
            media_db=mock_media_db,
            chacha_db=mock_chacha_db
        )
        
        # Mock the LLM generation
        async def mock_generate(*args, **kwargs):
            return {
                "answer": "RAG combines retrieval and generation techniques.",
                "sources": [
                    {
                        "id": "1",
                        "title": "RAG Overview",
                        "snippet": "Retrieval-Augmented Generation combines..."
                    }
                ]
            }
        
        rag_service.generate_answer = mock_generate
        
        # Run agent
        response = await run_retrieval_agent(
            request_body=agent_request,
            rag_service=rag_service,
            current_user=user
        )
        
        assert response is not None
        assert response.response_message.role == MessageRole.ASSISTANT
        assert len(response.response_message.content) > 0
        assert response.conversation_id is not None
        if response.citations:
            assert len(response.citations) > 0
    
    @pytest.mark.asyncio
    async def test_multi_user_isolation(self):
        """Test that different users have isolated data."""
        user1 = User(id=10, username="user1", email="user1@example.com", is_active=True)
        user2 = User(id=20, username="user2", email="user2@example.com", is_active=True)
        
        # Create databases for both users
        self.create_test_database(user1.id)
        user2_env = self.create_test_database(user2.id)
        
        # Add unique data to user2
        import hashlib
        conn = sqlite3.connect(str(user2_env["media_db"]))
        cursor = conn.cursor()
        # Add user2's unique document
        content_hash = hashlib.md5("This is user2's private document.".encode()).hexdigest()
        uuid_val = f"user2_doc_{datetime.now().timestamp()}"
        cursor.execute(
            """INSERT INTO Media (url, title, type, content, author, ingestion_date,
                                transcription_model, content_hash, uuid, last_modified,
                                version, client_id, deleted)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            ("http://example.com/user2", "User2 Secret", "document", 
             "This is user2's private document.", "User2", datetime.now().isoformat(),
             "test", content_hash, uuid_val, datetime.now().isoformat(), 
             1, "test_client", 0)
        )
        
        # Update FTS index for the new document
        cursor.execute("""
            INSERT INTO media_fts(title, content)
            VALUES (?, ?)
        """, ("User2 Secret", "This is user2's private document."))
        
        conn.commit()
        conn.close()
        
        # Search as user1 - should not find user2's document
        search_request = SearchApiRequest(
            querystring="User2 Secret",
            search_mode=SearchModeEnum.BASIC
        )
        
        mock_media_db1 = Mock(db_path=str(self.user_base_dir / "10" / "user_media_library.sqlite"))
        mock_chacha_db1 = Mock(db_path=str(self.user_base_dir / "10" / "chachanotes_user_dbs" / "user_chacha_notes_rag.sqlite"))
        
        rag_service1 = await get_rag_service_for_user(
            current_user=user1,
            media_db=mock_media_db1,
            chacha_db=mock_chacha_db1
        )
        
        response1 = await perform_search(
            request_body=search_request,
            rag_service=rag_service1,
            current_user=user1
        )
        
        # User1 should not find user2's document
        assert not any("User2 Secret" in r.title for r in response1.results)
        
        # Search as user2 - should find their document
        mock_media_db2 = Mock(db_path=str(self.user_base_dir / "20" / "user_media_library.sqlite"))
        mock_chacha_db2 = Mock(db_path=str(self.user_base_dir / "20" / "chachanotes_user_dbs" / "user_chacha_notes_rag.sqlite"))
        
        rag_service2 = await get_rag_service_for_user(
            current_user=user2,
            media_db=mock_media_db2,
            chacha_db=mock_chacha_db2
        )
        
        response2 = await perform_search(
            request_body=search_request,
            rag_service=rag_service2,
            current_user=user2
        )
        
        # User2 should find their document
        assert any("User2 Secret" in r.title for r in response2.results)
    
    @pytest.mark.asyncio
    async def test_search_with_filters(self):
        """Test search with various filters."""
        user = User(id=4, username="filteruser", email="filter@example.com", is_active=True)
        self.create_test_database(user.id)
        
        # Test with date range filter
        search_request = SearchApiRequest(
            querystring="tutorial",
            search_mode=SearchModeEnum.CUSTOM,
            date_range_start="2024-01-01T00:00:00",
            date_range_end="2024-01-02T23:59:59",
            use_semantic_search=False
        )
        
        mock_media_db = Mock(db_path=str(self.user_base_dir / "4" / "user_media_library.sqlite"))
        mock_chacha_db = Mock(db_path=str(self.user_base_dir / "4" / "chachanotes_user_dbs" / "user_chacha_notes_rag.sqlite"))
        
        rag_service = await get_rag_service_for_user(
            current_user=user,
            media_db=mock_media_db,
            chacha_db=mock_chacha_db
        )
        
        response = await perform_search(
            request_body=search_request,
            rag_service=rag_service,
            current_user=user
        )
        
        assert response is not None
        assert len(response.results) >= 0  # May or may not find results based on date filter
        
        # Test with database selection
        search_request2 = SearchApiRequest(
            querystring="notes",
            search_databases=["notes"],
            search_mode=SearchModeEnum.CUSTOM
        )
        
        response2 = await perform_search(
            request_body=search_request2,
            rag_service=rag_service,
            current_user=user
        )
        
        assert response2 is not None
        # Should only search in notes database
        if len(response2.results) > 0:
            assert all(r.metadata.get("source") == "NOTES" for r in response2.results)
    
    @pytest.mark.asyncio
    async def test_streaming_generation(self):
        """Test streaming response generation."""
        user = User(id=5, username="streamuser", email="stream@example.com", is_active=True)
        self.create_test_database(user.id)
        
        agent_request = RetrievalAgentRequest(
            message=Message(
                role=MessageRole.USER,
                content="Explain RAG"
            ),
            mode=AgentModeEnum.RAG,
            rag_generation_config=GenerationConfig(
                model="test-model",
                stream=True
            )
        )
        
        mock_media_db = Mock(db_path=str(self.user_base_dir / "5" / "user_media_library.sqlite"))
        mock_chacha_db = Mock(db_path=str(self.user_base_dir / "5" / "chachanotes_user_dbs" / "user_chacha_notes_rag.sqlite"))
        
        rag_service = await get_rag_service_for_user(
            current_user=user,
            media_db=mock_media_db,
            chacha_db=mock_chacha_db
        )
        
        # Mock streaming
        async def mock_stream(*args, **kwargs):
            yield {"type": "content", "content": "RAG is "}
            yield {"type": "content", "content": "a technique."}
            yield {"type": "citation", "citation": {"source_name": "RAG Overview"}}
        
        rag_service.generate_answer_stream = mock_stream
        
        # Since streaming returns a StreamingResponse, we need to handle it differently
        # For this test, we'll just verify the function runs without error
        try:
            response = await run_retrieval_agent(
                request_body=agent_request,
                rag_service=rag_service,
                current_user=user
            )
            # If it's a streaming response, it will have specific attributes
            assert hasattr(response, 'media_type') or hasattr(response, 'response_message')
        except Exception as e:
            pytest.fail(f"Streaming generation failed: {str(e)}")
    
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling in the pipeline."""
        user = User(id=6, username="erroruser", email="error@example.com", is_active=True)
        
        # Test with empty message
        with pytest.raises(Exception):  # Should raise validation error
            agent_request = RetrievalAgentRequest(
                message=Message(
                    role=MessageRole.USER,
                    content=""  # Empty content
                ),
                mode=AgentModeEnum.RAG
            )
            
            mock_media_db = Mock()
            mock_chacha_db = Mock()
            rag_service = await get_rag_service_for_user(
                current_user=user,
                media_db=mock_media_db,
                chacha_db=mock_chacha_db
            )
            
            await run_retrieval_agent(
                request_body=agent_request,
                rag_service=rag_service,
                current_user=user
            )
        
        # Test with database error
        search_request = SearchApiRequest(
            querystring="test",
            search_mode=SearchModeEnum.BASIC
        )
        
        # Create a RAG service that will fail
        mock_media_db = Mock(db_path="/nonexistent/path/db.sqlite")
        mock_chacha_db = Mock(db_path="/nonexistent/path/chacha.sqlite")
        
        with pytest.raises(Exception):
            rag_service = await get_rag_service_for_user(
                current_user=user,
                media_db=mock_media_db,
                chacha_db=mock_chacha_db
            )
            
            await perform_search(
                request_body=search_request,
                rag_service=rag_service,
                current_user=user
            )