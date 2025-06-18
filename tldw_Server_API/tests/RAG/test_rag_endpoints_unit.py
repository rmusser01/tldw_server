"""
Unit tests for RAG endpoints.

Tests individual components and functions in isolation with mocked dependencies.
"""

import json
import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from uuid import uuid4
from pathlib import Path
from datetime import datetime

from fastapi import HTTPException
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.endpoints.rag import (
    get_rag_service_for_user,
    perform_search,
    run_retrieval_agent,
    _user_rag_services
)
from tldw_Server_API.app.api.v1.schemas.rag_schemas import (
    SearchApiRequest,
    SearchModeEnum,
    RetrievalAgentRequest,
    AgentModeEnum,
    Message,
    MessageRole,
    GenerationConfig
)
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User
from tldw_Server_API.app.core.RAG.rag_service.types import DataSource


class TestGetRAGServiceForUser:
    """Test the get_rag_service_for_user dependency function."""
    
    @pytest.fixture
    def mock_user(self):
        """Create a mock user."""
        return User(id=123, username="testuser", email="test@example.com")
    
    @pytest.fixture
    def mock_media_db(self):
        """Create a mock MediaDatabase."""
        return Mock()
    
    @pytest.fixture
    def mock_chacha_db(self):
        """Create a mock CharactersRAGDB."""
        return Mock()
    
    @pytest.mark.asyncio
    async def test_creates_new_service_for_user(self, mock_user, mock_media_db, mock_chacha_db):
        """Test that a new RAG service is created for a user."""
        # Clear any existing services
        _user_rag_services.clear()
        
        with patch('tldw_Server_API.app.api.v1.endpoints.rag.RAGService') as mock_rag_service_class:
            mock_service = AsyncMock()
            mock_rag_service_class.return_value = mock_service
            
            # Call the function
            service = await get_rag_service_for_user(mock_user, mock_media_db, mock_chacha_db)
            
            # Verify service was created with correct parameters
            mock_rag_service_class.assert_called_once()
            call_kwargs = mock_rag_service_class.call_args.kwargs
            
            assert call_kwargs['media_db_path'] == Path("/Users/appledev/Working/tldw_server/user_databases/123/user_media_library.sqlite")
            assert call_kwargs['chachanotes_db_path'] == Path("/Users/appledev/Working/tldw_server/user_databases/123/chachanotes_user_dbs/user_chacha_notes_rag.sqlite")
            assert call_kwargs['chroma_path'] == Path("/Users/appledev/Working/tldw_server/user_databases/123/chroma")
            
            # Verify service was initialized
            mock_service.initialize.assert_called_once()
            
            # Verify service was cached
            assert _user_rag_services[123] == mock_service
    
    @pytest.mark.asyncio
    async def test_returns_cached_service(self, mock_user, mock_media_db, mock_chacha_db):
        """Test that cached service is returned for existing user."""
        # Pre-populate cache
        mock_cached_service = Mock()
        _user_rag_services[123] = mock_cached_service
        
        # Call the function
        service = await get_rag_service_for_user(mock_user, mock_media_db, mock_chacha_db)
        
        # Verify cached service was returned
        assert service == mock_cached_service
    
    @pytest.mark.asyncio
    async def test_applies_config_from_settings(self, mock_user, mock_media_db, mock_chacha_db):
        """Test that RAG_SERVICE_CONFIG is properly applied."""
        _user_rag_services.clear()
        
        with patch('tldw_Server_API.app.api.v1.endpoints.rag.RAGService') as mock_rag_service_class:
            with patch('tldw_Server_API.app.api.v1.endpoints.rag.RAGConfig') as mock_config_class:
                mock_config = Mock()
                mock_config.cache = Mock()
                mock_config.retriever = Mock()
                mock_config.processor = Mock()
                mock_config.generator = Mock()
                mock_config_class.return_value = mock_config
                
                mock_service = AsyncMock()
                mock_rag_service_class.return_value = mock_service
                
                # Call the function
                await get_rag_service_for_user(mock_user, mock_media_db, mock_chacha_db)
                
                # Verify config was created and passed
                mock_config_class.assert_called_once()
                mock_rag_service_class.assert_called_once()
                assert mock_rag_service_class.call_args.kwargs['config'] == mock_config


class TestPerformSearch:
    """Test the perform_search endpoint function."""
    
    @pytest.fixture
    def mock_user(self):
        return User(id=123, username="testuser", email="test@example.com")
    
    @pytest.fixture
    def mock_rag_service(self):
        service = AsyncMock()
        service.search = AsyncMock()
        return service
    
    @pytest.fixture
    def search_request(self):
        return SearchApiRequest(
            querystring="test query",
            search_mode=SearchModeEnum.BASIC,
            limit=10,
            offset=0,
            use_semantic_search=False
        )
    
    @pytest.mark.asyncio
    async def test_basic_search(self, search_request, mock_rag_service, mock_user):
        """Test basic search functionality."""
        # Mock search results
        mock_results = [
            Mock(
                source=DataSource.MEDIA_DB,
                documents=[
                    Mock(
                        id="doc1",
                        content="Test document content",
                        relevance_score=0.95,
                        source_id="media1",
                        metadata={"title": "Test Document"}
                    )
                ]
            )
        ]
        mock_rag_service.search.return_value = mock_results
        
        # Call the function
        response = await perform_search(search_request, mock_rag_service, mock_user)
        
        # Verify search was called with correct parameters
        mock_rag_service.search.assert_called_once_with(
            query="test query",
            sources=[DataSource.MEDIA_DB, DataSource.NOTES],
            filters={},
            fts_top_k=10,
            vector_top_k=0,  # semantic search not enabled
            hybrid_alpha=0.0
        )
        
        # Verify response
        assert response.querystring_echo == "test query"
        assert len(response.results) == 1
        assert response.results[0].id == "doc1"
        assert response.results[0].title == "Test Document"
        assert response.results[0].score == 0.95
    
    @pytest.mark.asyncio
    async def test_search_with_filters(self, mock_rag_service, mock_user):
        """Test search with filters and date ranges."""
        request = SearchApiRequest(
            querystring="filtered search",
            search_mode=SearchModeEnum.ADVANCED,
            filters={"root": {"category": "video"}},
            date_range_start="2024-01-01T00:00:00",
            date_range_end="2024-12-31T00:00:00",
            use_semantic_search=True,
            use_hybrid_search=True
        )
        
        mock_rag_service.search.return_value = []
        
        await perform_search(request, mock_rag_service, mock_user)
        
        # Verify filters were passed correctly
        call_args = mock_rag_service.search.call_args
        assert call_args.kwargs['filters']['category'] == "video"
        assert call_args.kwargs['filters']['date_start'] == "2024-01-01T00:00:00"
        assert call_args.kwargs['filters']['date_end'] == "2024-12-31T00:00:00"
        assert call_args.kwargs['vector_top_k'] == 10  # semantic search enabled
        assert call_args.kwargs['hybrid_alpha'] == 0.5
    
    @pytest.mark.asyncio
    async def test_search_with_custom_databases(self, mock_rag_service, mock_user):
        """Test search with specific database selection."""
        request = SearchApiRequest(
            querystring="notes search",
            search_databases=["notes", "character_cards"]
        )
        
        mock_rag_service.search.return_value = []
        
        await perform_search(request, mock_rag_service, mock_user)
        
        # Verify correct sources were used
        call_args = mock_rag_service.search.call_args
        sources = call_args.kwargs['sources']
        assert DataSource.NOTES in sources
        assert DataSource.CHARACTER_CARDS in sources
        assert DataSource.MEDIA_DB not in sources
    
    @pytest.mark.asyncio
    async def test_search_error_handling(self, search_request, mock_rag_service, mock_user):
        """Test error handling in search."""
        mock_rag_service.search.side_effect = Exception("Search failed")
        
        with pytest.raises(HTTPException) as exc_info:
            await perform_search(search_request, mock_rag_service, mock_user)
        
        assert exc_info.value.status_code == 500
        assert "Search failed" in exc_info.value.detail


class TestRunRetrievalAgent:
    """Test the run_retrieval_agent endpoint function."""
    
    @pytest.fixture
    def mock_user(self):
        return User(id=123, username="testuser", email="test@example.com")
    
    @pytest.fixture
    def mock_rag_service(self):
        service = AsyncMock()
        service.generate_answer = AsyncMock()
        service.generate_answer_stream = AsyncMock()
        return service
    
    @pytest.fixture
    def agent_request(self):
        return RetrievalAgentRequest(
            message=Message(role=MessageRole.USER, content="What is RAG?"),
            mode=AgentModeEnum.RAG,
            conversation_id=str(uuid4())
        )
    
    @pytest.mark.asyncio
    async def test_basic_rag_generation(self, agent_request, mock_rag_service, mock_user):
        """Test basic RAG answer generation."""
        # Mock response
        mock_rag_service.generate_answer.return_value = {
            "answer": "RAG stands for Retrieval-Augmented Generation.",
            "sources": [
                {
                    "id": "src1",
                    "source": "MEDIA_DB",
                    "title": "RAG Overview",
                    "score": 0.92,
                    "snippet": "RAG combines retrieval and generation...",
                    "metadata": {"page": 1}
                }
            ],
            "context_size": 500
        }
        
        # Call the function
        response = await run_retrieval_agent(agent_request, mock_rag_service, mock_user)
        
        # Verify generate_answer was called correctly
        mock_rag_service.generate_answer.assert_called_once()
        call_kwargs = mock_rag_service.generate_answer.call_args.kwargs
        assert call_kwargs['query'] == "What is RAG?"
        assert DataSource.MEDIA_DB in call_kwargs['sources']
        assert DataSource.NOTES in call_kwargs['sources']
        assert DataSource.CHAT_HISTORY in call_kwargs['sources']
        
        # Verify response
        assert response.response_message.role == MessageRole.ASSISTANT
        assert "Retrieval-Augmented Generation" in response.response_message.content
        assert len(response.citations) == 1
        assert response.citations[0].source_name == "RAG Overview"
    
    @pytest.mark.asyncio
    async def test_research_mode(self, mock_rag_service, mock_user):
        """Test research mode uses different sources."""
        request = RetrievalAgentRequest(
            message=Message(role=MessageRole.USER, content="Research question"),
            mode=AgentModeEnum.RESEARCH
        )
        
        mock_rag_service.generate_answer.return_value = {"answer": "Research answer", "sources": []}
        
        await run_retrieval_agent(request, mock_rag_service, mock_user)
        
        # Verify research mode sources
        call_kwargs = mock_rag_service.generate_answer.call_args.kwargs
        sources = call_kwargs['sources']
        assert DataSource.MEDIA_DB in sources
        assert DataSource.NOTES in sources
        assert DataSource.CHAT_HISTORY not in sources  # Not included in research mode
    
    @pytest.mark.asyncio
    async def test_streaming_response(self, mock_rag_service, mock_user):
        """Test streaming response generation."""
        request = RetrievalAgentRequest(
            message=Message(role=MessageRole.USER, content="Stream this"),
            mode=AgentModeEnum.RAG,
            rag_generation_config=GenerationConfig(stream=True)
        )
        
        # Mock streaming response
        async def mock_stream():
            yield {"type": "content", "content": "Streaming "}
            yield {"type": "content", "content": "response"}
            yield {"type": "citation", "citation": {"id": "1", "source": "test", "title": "Test"}}
        
        mock_rag_service.generate_answer_stream.return_value = mock_stream()
        
        # Call the function
        response = await run_retrieval_agent(request, mock_rag_service, mock_user)
        
        # Verify streaming response
        assert hasattr(response, 'body_iterator')  # StreamingResponse
        assert response.media_type == "text/event-stream"
    
    @pytest.mark.asyncio
    async def test_conversation_history_handling(self, mock_rag_service, mock_user):
        """Test handling of conversation history."""
        request = RetrievalAgentRequest(
            messages=[
                Message(role=MessageRole.USER, content="First question"),
                Message(role=MessageRole.ASSISTANT, content="First answer"),
                Message(role=MessageRole.USER, content="Follow-up question")
            ],
            mode=AgentModeEnum.RAG
        )
        
        mock_rag_service.generate_answer.return_value = {"answer": "Follow-up answer", "sources": []}
        
        await run_retrieval_agent(request, mock_rag_service, mock_user)
        
        # Verify conversation history was passed
        call_kwargs = mock_rag_service.generate_answer.call_args.kwargs
        assert len(call_kwargs['conversation_history']) == 2
        assert call_kwargs['conversation_history'][0]['content'] == "First question"
        assert call_kwargs['query'] == "Follow-up question"
    
    @pytest.mark.asyncio
    async def test_api_config_handling(self, mock_rag_service, mock_user):
        """Test API configuration is passed to generation."""
        request = RetrievalAgentRequest(
            message=Message(role=MessageRole.USER, content="Test"),
            mode=AgentModeEnum.RAG,
            api_config={"api_provider": "openai", "api_key": "test-key"},
            rag_generation_config=GenerationConfig(
                model="gpt-4",
                temperature=0.5,
                max_tokens_to_sample=1000
            )
        )
        
        mock_rag_service.generate_answer.return_value = {"answer": "Test", "sources": []}
        
        await run_retrieval_agent(request, mock_rag_service, mock_user)
        
        # Verify generation config
        gen_config = mock_rag_service.generate_answer.call_args.kwargs['generation_config']
        assert gen_config['model'] == "gpt-4"
        assert gen_config['temperature'] == 0.5
        assert gen_config['max_tokens'] == 1000
        assert gen_config['api_provider'] == "openai"
        assert gen_config['api_key'] == "test-key"
    
    @pytest.mark.asyncio
    async def test_error_handling(self, agent_request, mock_rag_service, mock_user):
        """Test error handling in agent."""
        mock_rag_service.generate_answer.side_effect = Exception("Generation failed")
        
        with pytest.raises(HTTPException) as exc_info:
            await run_retrieval_agent(agent_request, mock_rag_service, mock_user)
        
        assert exc_info.value.status_code == 500
        assert "Generation failed" in exc_info.value.detail
    
    @pytest.mark.asyncio
    async def test_empty_message_handling(self, mock_rag_service, mock_user):
        """Test handling of empty messages."""
        request = RetrievalAgentRequest(
            message=Message(role=MessageRole.USER, content=""),
            mode=AgentModeEnum.RAG
        )
        
        with pytest.raises(HTTPException) as exc_info:
            await run_retrieval_agent(request, mock_rag_service, mock_user)
        
        assert exc_info.value.status_code == 422
        assert "Message content cannot be empty" in exc_info.value.detail


if __name__ == "__main__":
    pytest.main([__file__, "-v"])