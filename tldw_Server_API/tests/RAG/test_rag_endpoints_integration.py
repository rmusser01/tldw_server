"""
Integration tests for RAG endpoints.

Tests the full integration of RAG endpoints with real database connections
and the RAG service pipeline.
"""

import json
import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from uuid import uuid4

from fastapi.testclient import TestClient
from httpx import AsyncClient

from tldw_Server_API.app.main import app
from tldw_Server_API.app.core.DB_Management.Media_DB_v2 import MediaDatabase
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User
from tldw_Server_API.app.api.v1.endpoints.rag import _user_rag_services


@pytest.fixture(scope="module")
def test_user():
    """Create a test user."""
    return User(
        id=9999,
        username="rag_test_user",
        email="ragtest@example.com",
        is_active=True
    )


@pytest.fixture(scope="module")
def test_db_dir():
    """Create a temporary directory for test databases."""
    temp_dir = tempfile.mkdtemp(prefix="rag_test_")
    yield Path(temp_dir)
    # Cleanup after tests
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(scope="module")
def media_db(test_db_dir, test_user):
    """Create a test media database with sample data."""
    db_path = test_db_dir / "media.sqlite"
    db = MediaDatabase(str(db_path), client_id=str(test_user.id))
    
    # Add sample media items
    sample_media = [
        {
            "title": "Introduction to RAG",
            "content": "Retrieval-Augmented Generation (RAG) combines the power of retrieval systems with generative models. It allows AI systems to access external knowledge bases to provide more accurate and up-to-date information.",
            "url": "https://example.com/rag-intro",
            "type": "article",
            "media_metadata": {"author": "AI Expert", "date": "2024-01-15"}
        },
        {
            "title": "Machine Learning Basics",
            "content": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It includes supervised, unsupervised, and reinforcement learning.",
            "url": "https://example.com/ml-basics",
            "type": "article",
            "media_metadata": {"author": "ML Teacher", "date": "2024-02-01"}
        },
        {
            "title": "Vector Databases Explained",
            "content": "Vector databases are specialized databases designed to store and query high-dimensional vector embeddings. They are essential for semantic search and RAG applications, enabling efficient similarity search.",
            "url": "https://example.com/vector-db",
            "type": "video",
            "media_metadata": {"duration": "15:30", "date": "2024-03-10"}
        }
    ]
    
    for item in sample_media:
        db.add_media_item(**item)
    
    yield db
    db.close_connection()


@pytest.fixture(scope="module")
def chacha_db(test_db_dir, test_user):
    """Create a test ChaChaNotes database with sample data."""
    db_path = test_db_dir / "chacha.sqlite"
    db = CharactersRAGDB(str(db_path), client_id=str(test_user.id))
    
    # Add sample notes
    note1_id = db.create_note(
        title="RAG Implementation Notes",
        content="Key points for RAG implementation: 1) Choose appropriate embedding model, 2) Optimize chunk size, 3) Implement proper re-ranking",
        is_private=False
    )
    
    note2_id = db.create_note(
        title="Meeting Notes - AI Strategy",
        content="Discussed moving to RAG-based system for better accuracy. Need to evaluate different vector databases and embedding models.",
        is_private=False
    )
    
    # Add a character card
    char_id = db.add_character_card({
        'name': 'AI Assistant',
        'description': 'A helpful AI assistant specialized in RAG and machine learning',
        'personality': 'Professional and knowledgeable',
        'tags': json.dumps(['AI', 'RAG', 'helpful']),
        'creator': 'Test',
        'client_id': str(test_user.id)
    })
    
    # Add chat history
    conv_id = str(uuid4())
    db.save_chat_history(
        conversation_id=conv_id,
        character_id=char_id,
        conversation_name="RAG Discussion",
        chat_history=[
            ("What is RAG?", "RAG stands for Retrieval-Augmented Generation..."),
            ("How does it work?", "RAG works by first retrieving relevant documents...")
        ]
    )
    
    yield db
    db.close_connection()


@pytest.fixture
def auth_headers(test_user):
    """Create authentication headers for the test user."""
    # In a real scenario, you'd generate a proper JWT token
    # For testing, we'll mock the authentication
    return {"Authorization": f"Bearer test_token_for_user_{test_user.id}"}


@pytest.fixture
async def async_client():
    """Create an async test client."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client


class TestRAGSearchIntegration:
    """Integration tests for the search endpoint."""
    
    @pytest.mark.asyncio
    async def test_search_across_databases(self, async_client, auth_headers, media_db, chacha_db, test_user):
        """Test searching across multiple databases."""
        # Clear any cached services
        _user_rag_services.clear()
        
        # Mock the database dependencies
        with pytest.mock.patch('tldw_Server_API.app.api.v1.endpoints.rag.get_media_db_for_user', return_value=media_db):
            with pytest.mock.patch('tldw_Server_API.app.api.v1.endpoints.rag.get_chacha_db_for_user', return_value=chacha_db):
                with pytest.mock.patch('tldw_Server_API.app.api.v1.endpoints.rag.get_request_user', return_value=test_user):
                    
                    response = await async_client.post(
                        "/retrieval/search",
                        json={
                            "querystring": "RAG implementation",
                            "search_mode": "advanced",
                            "search_databases": ["media_db", "notes"],
                            "limit": 10
                        },
                        headers=auth_headers
                    )
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure
        assert "results" in data
        assert "total_results" in data
        assert data["querystring_echo"] == "RAG implementation"
        
        # Should find results from both databases
        results = data["results"]
        assert len(results) > 0
        
        # Check that we have results from different sources
        sources = {r["metadata"]["source"] for r in results}
        assert len(sources) > 1  # Should have multiple sources
    
    @pytest.mark.asyncio
    async def test_search_with_filters(self, async_client, auth_headers, media_db, chacha_db, test_user):
        """Test search with filters applied."""
        _user_rag_services.clear()
        
        with pytest.mock.patch('tldw_Server_API.app.api.v1.endpoints.rag.get_media_db_for_user', return_value=media_db):
            with pytest.mock.patch('tldw_Server_API.app.api.v1.endpoints.rag.get_chacha_db_for_user', return_value=chacha_db):
                with pytest.mock.patch('tldw_Server_API.app.api.v1.endpoints.rag.get_request_user', return_value=test_user):
                    
                    response = await async_client.post(
                        "/retrieval/search",
                        json={
                            "querystring": "machine learning",
                            "search_mode": "custom",
                            "filters": {"root": {"type": "article"}},
                            "date_range_start": "2024-01-01T00:00:00",
                            "date_range_end": "2024-12-31T23:59:59",
                            "limit": 5
                        },
                        headers=auth_headers
                    )
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify filters were applied
        assert data["debug_info"]["filters_provided"] is True
        
        # Results should only include articles
        for result in data["results"]:
            if "type" in result["metadata"]:
                assert result["metadata"]["type"] == "article"
    
    @pytest.mark.asyncio
    async def test_hybrid_search(self, async_client, auth_headers, media_db, chacha_db, test_user):
        """Test hybrid search combining keyword and semantic search."""
        _user_rag_services.clear()
        
        with pytest.mock.patch('tldw_Server_API.app.api.v1.endpoints.rag.get_media_db_for_user', return_value=media_db):
            with pytest.mock.patch('tldw_Server_API.app.api.v1.endpoints.rag.get_chacha_db_for_user', return_value=chacha_db):
                with pytest.mock.patch('tldw_Server_API.app.api.v1.endpoints.rag.get_request_user', return_value=test_user):
                    
                    response = await async_client.post(
                        "/retrieval/search",
                        json={
                            "querystring": "vector embeddings",
                            "search_mode": "advanced",
                            "use_hybrid_search": True,
                            "use_semantic_search": True,
                            "limit": 10
                        },
                        headers=auth_headers
                    )
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify hybrid search was used
        assert data["debug_info"]["hybrid_search_flag"] is True
        
        # Should find the vector database article
        results = data["results"]
        vector_results = [r for r in results if "vector" in r["title"].lower()]
        assert len(vector_results) > 0


class TestRAGAgentIntegration:
    """Integration tests for the agent endpoint."""
    
    @pytest.mark.asyncio
    async def test_rag_generation_basic(self, async_client, auth_headers, media_db, chacha_db, test_user):
        """Test basic RAG answer generation."""
        _user_rag_services.clear()
        
        # Mock LLM response
        async def mock_generate(*args, **kwargs):
            return {
                "answer": "Based on the retrieved information, RAG (Retrieval-Augmented Generation) is a technique that combines retrieval systems with generative models to provide more accurate responses.",
                "sources": [],
                "context_size": 500
            }
        
        with pytest.mock.patch('tldw_Server_API.app.api.v1.endpoints.rag.get_media_db_for_user', return_value=media_db):
            with pytest.mock.patch('tldw_Server_API.app.api.v1.endpoints.rag.get_chacha_db_for_user', return_value=chacha_db):
                with pytest.mock.patch('tldw_Server_API.app.api.v1.endpoints.rag.get_request_user', return_value=test_user):
                    # Mock the RAG service's generate_answer method
                    with pytest.mock.patch('tldw_Server_API.app.core.RAG.rag_service.integration.RAGService.generate_answer', side_effect=mock_generate):
                        
                        response = await async_client.post(
                            "/retrieval/agent",
                            json={
                                "message": {
                                    "role": "user",
                                    "content": "What is RAG?"
                                },
                                "mode": "rag",
                                "search_mode": "advanced"
                            },
                            headers=auth_headers
                        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure
        assert "response_message" in data
        assert data["response_message"]["role"] == "assistant"
        assert "RAG" in data["response_message"]["content"]
        assert "conversation_id" in data
    
    @pytest.mark.asyncio
    async def test_research_mode(self, async_client, auth_headers, media_db, chacha_db, test_user):
        """Test research mode with different sources."""
        _user_rag_services.clear()
        
        async def mock_generate(*args, **kwargs):
            # Verify research mode sources were used
            sources = kwargs.get('sources', [])
            # Research mode shouldn't include chat history
            from tldw_Server_API.app.core.RAG.rag_service.types import DataSource
            assert DataSource.CHAT_HISTORY not in sources
            
            return {
                "answer": "Research findings on machine learning...",
                "sources": [{
                    "id": "1",
                    "source": "MEDIA_DB",
                    "title": "ML Research",
                    "score": 0.9,
                    "snippet": "Machine learning research...",
                    "metadata": {}
                }],
                "context_size": 300
            }
        
        with pytest.mock.patch('tldw_Server_API.app.api.v1.endpoints.rag.get_media_db_for_user', return_value=media_db):
            with pytest.mock.patch('tldw_Server_API.app.api.v1.endpoints.rag.get_chacha_db_for_user', return_value=chacha_db):
                with pytest.mock.patch('tldw_Server_API.app.api.v1.endpoints.rag.get_request_user', return_value=test_user):
                    with pytest.mock.patch('tldw_Server_API.app.core.RAG.rag_service.integration.RAGService.generate_answer', side_effect=mock_generate):
                        
                        response = await async_client.post(
                            "/retrieval/agent",
                            json={
                                "message": {
                                    "role": "user",
                                    "content": "Research machine learning applications"
                                },
                                "mode": "research"
                            },
                            headers=auth_headers
                        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["debug_info"]["mode"] == "research"
        assert len(data["citations"]) > 0
    
    @pytest.mark.asyncio
    async def test_streaming_response(self, async_client, auth_headers, media_db, chacha_db, test_user):
        """Test streaming response generation."""
        _user_rag_services.clear()
        
        async def mock_stream(*args, **kwargs):
            yield {"type": "content", "content": "Streaming response: "}
            yield {"type": "content", "content": "RAG is great!"}
            yield {"type": "citation", "citation": {"id": "1", "source": "test", "title": "Test Source"}}
        
        with pytest.mock.patch('tldw_Server_API.app.api.v1.endpoints.rag.get_media_db_for_user', return_value=media_db):
            with pytest.mock.patch('tldw_Server_API.app.api.v1.endpoints.rag.get_chacha_db_for_user', return_value=chacha_db):
                with pytest.mock.patch('tldw_Server_API.app.api.v1.endpoints.rag.get_request_user', return_value=test_user):
                    with pytest.mock.patch('tldw_Server_API.app.core.RAG.rag_service.integration.RAGService.generate_answer_stream', return_value=mock_stream()):
                        
                        # Use the synchronous test client for streaming
                        with TestClient(app) as client:
                            response = client.post(
                                "/retrieval/agent",
                                json={
                                    "message": {
                                        "role": "user",
                                        "content": "Stream this response"
                                    },
                                    "mode": "rag",
                                    "rag_generation_config": {
                                        "stream": True
                                    }
                                },
                                headers=auth_headers,
                                stream=True
                            )
                            
                            assert response.status_code == 200
                            assert response.headers["content-type"] == "text/event-stream; charset=utf-8"
                            
                            # Read streaming chunks
                            chunks = []
                            for line in response.iter_lines():
                                if line.startswith("data: "):
                                    chunk_data = json.loads(line[6:])
                                    chunks.append(chunk_data)
                            
                            # Verify we got different chunk types
                            content_chunks = [c for c in chunks if c.get("type") == "content"]
                            citation_chunks = [c for c in chunks if c.get("type") == "citation"]
                            
                            assert len(content_chunks) > 0
                            assert len(citation_chunks) > 0
    
    @pytest.mark.asyncio
    async def test_conversation_context(self, async_client, auth_headers, media_db, chacha_db, test_user):
        """Test conversation with history context."""
        _user_rag_services.clear()
        
        async def mock_generate(*args, **kwargs):
            # Verify conversation history was passed
            history = kwargs.get('conversation_history', [])
            assert len(history) == 2
            assert history[0]['content'] == "What is machine learning?"
            
            return {
                "answer": "As I mentioned earlier, supervised learning is one type of machine learning...",
                "sources": [],
                "context_size": 200
            }
        
        with pytest.mock.patch('tldw_Server_API.app.api.v1.endpoints.rag.get_media_db_for_user', return_value=media_db):
            with pytest.mock.patch('tldw_Server_API.app.api.v1.endpoints.rag.get_chacha_db_for_user', return_value=chacha_db):
                with pytest.mock.patch('tldw_Server_API.app.api.v1.endpoints.rag.get_request_user', return_value=test_user):
                    with pytest.mock.patch('tldw_Server_API.app.core.RAG.rag_service.integration.RAGService.generate_answer', side_effect=mock_generate):
                        
                        response = await async_client.post(
                            "/retrieval/agent",
                            json={
                                "messages": [
                                    {"role": "user", "content": "What is machine learning?"},
                                    {"role": "assistant", "content": "Machine learning is a type of AI..."},
                                    {"role": "user", "content": "What are the main types?"}
                                ],
                                "mode": "rag",
                                "conversation_id": str(uuid4())
                            },
                            headers=auth_headers
                        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Should reference earlier context
        assert "mentioned earlier" in data["response_message"]["content"]


class TestRAGServiceCaching:
    """Test caching behavior of RAG services."""
    
    @pytest.mark.asyncio
    async def test_service_caching_per_user(self, async_client, auth_headers, media_db, chacha_db):
        """Test that RAG services are cached per user."""
        _user_rag_services.clear()
        
        # Create two different users
        user1 = User(id=1001, username="user1", email="user1@test.com")
        user2 = User(id=1002, username="user2", email="user2@test.com")
        
        # Make requests for both users
        for user in [user1, user2]:
            with pytest.mock.patch('tldw_Server_API.app.api.v1.endpoints.rag.get_media_db_for_user', return_value=media_db):
                with pytest.mock.patch('tldw_Server_API.app.api.v1.endpoints.rag.get_chacha_db_for_user', return_value=chacha_db):
                    with pytest.mock.patch('tldw_Server_API.app.api.v1.endpoints.rag.get_request_user', return_value=user):
                        
                        response = await async_client.post(
                            "/retrieval/search",
                            json={"querystring": "test", "limit": 1},
                            headers={"Authorization": f"Bearer test_token_{user.id}"}
                        )
                        
                        assert response.status_code == 200
        
        # Verify both users have separate cached services
        assert len(_user_rag_services) == 2
        assert 1001 in _user_rag_services
        assert 1002 in _user_rag_services
        assert _user_rag_services[1001] != _user_rag_services[1002]


class TestErrorScenarios:
    """Test error handling scenarios."""
    
    @pytest.mark.asyncio
    async def test_database_connection_error(self, async_client, auth_headers, test_user):
        """Test handling of database connection errors."""
        _user_rag_services.clear()
        
        # Mock database error
        with pytest.mock.patch('tldw_Server_API.app.api.v1.endpoints.rag.get_media_db_for_user', side_effect=Exception("DB connection failed")):
            with pytest.mock.patch('tldw_Server_API.app.api.v1.endpoints.rag.get_request_user', return_value=test_user):
                
                response = await async_client.post(
                    "/retrieval/search",
                    json={"querystring": "test"},
                    headers=auth_headers
                )
                
                assert response.status_code == 500
    
    @pytest.mark.asyncio
    async def test_invalid_search_parameters(self, async_client, auth_headers):
        """Test validation of search parameters."""
        response = await async_client.post(
            "/retrieval/search",
            json={
                "querystring": "",  # Empty query
                "limit": -1,  # Invalid limit
                "offset": -10  # Invalid offset
            },
            headers=auth_headers
        )
        
        assert response.status_code == 422  # Validation error
    
    @pytest.mark.asyncio
    async def test_rag_service_initialization_error(self, async_client, auth_headers, test_user):
        """Test handling of RAG service initialization errors."""
        _user_rag_services.clear()
        
        with pytest.mock.patch('tldw_Server_API.app.api.v1.endpoints.rag.RAGService') as mock_service_class:
            mock_service = pytest.mock.AsyncMock()
            mock_service.initialize.side_effect = Exception("Failed to initialize")
            mock_service_class.return_value = mock_service
            
            with pytest.mock.patch('tldw_Server_API.app.api.v1.endpoints.rag.get_request_user', return_value=test_user):
                
                response = await async_client.post(
                    "/retrieval/agent",
                    json={
                        "message": {"role": "user", "content": "test"},
                        "mode": "rag"
                    },
                    headers=auth_headers
                )
                
                assert response.status_code == 500


if __name__ == "__main__":
    pytest.main([__file__, "-v"])