"""
Unit tests for the RAG service integration module.

Tests the RAGService class and its methods in isolation.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from pathlib import Path

from tldw_Server_API.app.core.RAG.rag_service.integration import RAGService
from tldw_Server_API.app.core.RAG.rag_service.config import RAGConfig
from tldw_Server_API.app.core.RAG.rag_service.types import DataSource
from tldw_Server_API.app.core.RAG.rag_service.app import RAGApplication


class TestRAGServiceInitialization:
    """Test RAGService initialization and configuration."""
    
    def test_init_with_config_object(self, mock_rag_config):
        """Test initialization with a pre-configured RAGConfig object."""
        service = RAGService(
            config=mock_rag_config,
            media_db_path=Path("/test/media.db"),
            chachanotes_db_path=Path("/test/chacha.db")
        )
        
        assert service.config == mock_rag_config
        assert service.media_db_path == Path("/test/media.db")
        assert service.chachanotes_db_path == Path("/test/chacha.db")
        assert service.chroma_path == Path.home() / ".tldw_chatbook" / "chroma"
    
    def test_init_with_config_path(self, tmp_path):
        """Test initialization with a TOML config file path."""
        config_file = tmp_path / "config.toml"
        config_file.write_text("""
[general]
batch_size = 16

[cache]
enable_cache = true
max_cache_size = 500
        """)
        
        with patch('tldw_Server_API.app.core.RAG.rag_service.config.RAGConfig.from_toml') as mock_from_toml:
            mock_config = Mock()
            mock_from_toml.return_value = mock_config
            
            service = RAGService(config_path=config_file)
            
            mock_from_toml.assert_called_once_with(config_file)
            assert service.config == mock_config
    
    def test_custom_chroma_path(self):
        """Test setting a custom ChromaDB path."""
        custom_path = Path("/custom/chroma")
        service = RAGService(chroma_path=custom_path)
        
        assert service.chroma_path == custom_path
    
    @pytest.mark.asyncio
    async def test_initialize(self, mock_rag_config):
        """Test service initialization."""
        service = RAGService(
            config=mock_rag_config,
            media_db_path=Path("/test/media.db"),
            chachanotes_db_path=Path("/test/chacha.db")
        )
        
        with patch.object(service, '_setup_retrievers', new_callable=AsyncMock) as mock_setup_retrievers:
            with patch.object(service, '_setup_processor') as mock_setup_processor:
                with patch.object(service, '_setup_generator') as mock_setup_generator:
                    
                    await service.initialize()
                    
                    assert service._initialized
                    mock_setup_retrievers.assert_called_once()
                    mock_setup_processor.assert_called_once()
                    mock_setup_generator.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_initialize_idempotent(self, mock_rag_config):
        """Test that initialization is idempotent."""
        service = RAGService(config=mock_rag_config)
        
        with patch.object(service, '_setup_retrievers', new_callable=AsyncMock) as mock_setup:
            await service.initialize()
            await service.initialize()  # Second call
            
            # Should only be called once
            mock_setup.assert_called_once()


class TestRAGServiceRetrievers:
    """Test retriever setup and configuration."""
    
    @pytest.mark.asyncio
    async def test_setup_retrievers_with_media_db(self, mock_rag_config, tmp_path):
        """Test setting up retrievers with media database."""
        media_db_path = tmp_path / "media.db"
        media_db_path.touch()  # Create the file
        
        service = RAGService(
            config=mock_rag_config,
            media_db_path=media_db_path
        )
        
        with patch('tldw_Server_API.app.core.RAG.rag_service.integration.MediaDBRetriever') as mock_retriever_class:
            mock_retriever = Mock()
            mock_retriever_class.return_value = mock_retriever
            
            await service._setup_retrievers()
            
            mock_retriever_class.assert_called_once_with(media_db_path)
            service.app.register_retriever.assert_called()
    
    @pytest.mark.asyncio
    async def test_setup_retrievers_with_chacha_db(self, mock_rag_config, tmp_path):
        """Test setting up retrievers with ChaChaNotes database."""
        chacha_db_path = tmp_path / "chacha.db"
        chacha_db_path.touch()
        
        service = RAGService(
            config=mock_rag_config,
            chachanotes_db_path=chacha_db_path
        )
        
        with patch('tldw_Server_API.app.core.RAG.rag_service.integration.NotesRetriever') as mock_notes_retriever:
            with patch('tldw_Server_API.app.core.RAG.rag_service.integration.ChatHistoryRetriever') as mock_chat_retriever:
                
                await service._setup_retrievers()
                
                # Should create both notes and chat history retrievers
                mock_notes_retriever.assert_called()
                mock_chat_retriever.assert_called()
    
    @pytest.mark.asyncio
    async def test_hybrid_retriever_setup(self, mock_rag_config, tmp_path):
        """Test hybrid retriever setup when configured."""
        mock_rag_config.retriever.hybrid_alpha = 0.7
        mock_rag_config.retriever.vector_top_k = 10
        
        media_db_path = tmp_path / "media.db"
        media_db_path.touch()
        
        service = RAGService(
            config=mock_rag_config,
            media_db_path=media_db_path,
            chroma_path=tmp_path / "chroma"
        )
        
        with patch('tldw_Server_API.app.core.RAG.rag_service.integration.HybridRetriever') as mock_hybrid_class:
            with patch('tldw_Server_API.app.core.RAG.rag_service.integration.MediaDBRetriever'):
                with patch('tldw_Server_API.app.core.RAG.rag_service.integration.VectorRetriever'):
                    
                    await service._setup_retrievers()
                    
                    # Should create hybrid retriever with correct alpha
                    mock_hybrid_class.assert_called()
                    call_kwargs = mock_hybrid_class.call_args.kwargs
                    assert call_kwargs['alpha'] == 0.7


class TestRAGServiceProcessorGenerator:
    """Test processor and generator setup."""
    
    def test_setup_processor_default(self, mock_rag_config):
        """Test default processor setup."""
        service = RAGService(config=mock_rag_config)
        
        with patch('tldw_Server_API.app.core.RAG.rag_service.integration.DefaultProcessor') as mock_processor_class:
            mock_processor = Mock()
            mock_processor_class.return_value = mock_processor
            
            service._setup_processor()
            
            mock_processor_class.assert_called_once()
            service.app.register_processor.assert_called_with(mock_processor)
    
    def test_setup_processor_advanced(self, mock_rag_config):
        """Test advanced processor setup with reranking."""
        mock_rag_config.processor.enable_reranking = True
        service = RAGService(config=mock_rag_config)
        
        with patch('tldw_Server_API.app.core.RAG.rag_service.integration.AdvancedProcessor') as mock_processor_class:
            service._setup_processor()
            mock_processor_class.assert_called_once()
    
    def test_setup_generator_with_llm(self, mock_rag_config, mock_llm_handler):
        """Test generator setup with LLM handler."""
        service = RAGService(
            config=mock_rag_config,
            llm_handler=mock_llm_handler
        )
        
        with patch('tldw_Server_API.app.core.RAG.rag_service.integration.StreamingGenerator') as mock_gen_class:
            service._setup_generator()
            
            mock_gen_class.assert_called_once_with(
                mock_llm_handler,
                mock_rag_config.generator.__dict__
            )
    
    def test_setup_generator_fallback(self, mock_rag_config):
        """Test fallback generator when no LLM available."""
        service = RAGService(config=mock_rag_config)
        
        with patch('tldw_Server_API.app.core.RAG.rag_service.integration.FallbackGenerator') as mock_gen_class:
            service._setup_generator()
            mock_gen_class.assert_called_once()


class TestRAGServiceSearch:
    """Test search functionality."""
    
    @pytest.mark.asyncio
    async def test_search_basic(self, mock_rag_config):
        """Test basic search functionality."""
        service = RAGService(config=mock_rag_config)
        service._initialized = True
        
        mock_results = [
            Mock(documents=[
                Mock(id="1", content="Test", source=DataSource.MEDIA_DB, score=0.9, metadata={})
            ])
        ]
        
        service.app.search = AsyncMock(return_value=mock_results)
        
        results = await service.search("test query")
        
        service.app.search.assert_called_once_with(
            "test query", None, None
        )
        assert len(results) == 1
        assert results[0]["id"] == "1"
        assert results[0]["content"] == "Test"
    
    @pytest.mark.asyncio
    async def test_search_with_sources(self, mock_rag_config):
        """Test search with specific sources."""
        service = RAGService(config=mock_rag_config)
        service._initialized = True
        service.app.search = AsyncMock(return_value=[])
        
        await service.search(
            "test",
            sources=["MEDIA_DB", "NOTES"],
            filters={"category": "test"}
        )
        
        call_args = service.app.search.call_args
        assert DataSource.MEDIA_DB in call_args[0][1]
        assert DataSource.NOTES in call_args[0][1]
        assert call_args[0][2] == {"category": "test"}
    
    @pytest.mark.asyncio
    async def test_search_auto_initialize(self, mock_rag_config):
        """Test that search auto-initializes if needed."""
        service = RAGService(config=mock_rag_config)
        assert not service._initialized
        
        with patch.object(service, 'initialize', new_callable=AsyncMock) as mock_init:
            service.app.search = AsyncMock(return_value=[])
            
            await service.search("test")
            
            mock_init.assert_called_once()


class TestRAGServiceGeneration:
    """Test answer generation functionality."""
    
    @pytest.mark.asyncio
    async def test_generate_answer_basic(self, mock_rag_config):
        """Test basic answer generation."""
        service = RAGService(config=mock_rag_config)
        service._initialized = True
        
        mock_response = Mock(
            answer="Generated answer",
            sources=[Mock(id="1", source=DataSource.MEDIA_DB, metadata={"title": "Doc"}, score=0.9, content="Content")],
            context=Mock(combined_text="Context text"),
            metadata={"model": "gpt-3.5-turbo"}
        )
        
        service.app.generate = AsyncMock(return_value=mock_response)
        
        result = await service.generate_answer("What is RAG?")
        
        assert result["answer"] == "Generated answer"
        assert len(result["sources"]) == 1
        assert result["sources"][0]["title"] == "Doc"
        assert "context_preview" in result
        assert result["context_size"] == len("Context text")
    
    @pytest.mark.asyncio
    async def test_generate_answer_with_options(self, mock_rag_config):
        """Test answer generation with various options."""
        service = RAGService(config=mock_rag_config)
        service._initialized = True
        
        mock_response = Mock(
            answer="Answer",
            sources=[],
            context=Mock(combined_text=""),
            metadata={}
        )
        service.app.generate = AsyncMock(return_value=mock_response)
        
        await service.generate_answer(
            "Query",
            sources=["MEDIA_DB"],
            filters={"type": "article"},
            temperature=0.5,
            max_tokens=1000
        )
        
        call_kwargs = service.app.generate.call_args.kwargs
        assert "temperature" in call_kwargs
        assert call_kwargs["temperature"] == 0.5
        assert "max_tokens" in call_kwargs
        assert call_kwargs["max_tokens"] == 1000
    
    @pytest.mark.asyncio
    async def test_generate_answer_stream(self, mock_rag_config):
        """Test streaming answer generation."""
        service = RAGService(config=mock_rag_config)
        service._initialized = True
        
        async def mock_stream():
            yield Mock(content="Hello")
            yield Mock(content=" world")
            yield "Raw text"
        
        service.app.generate_stream = mock_stream
        
        chunks = []
        async for chunk in service.generate_answer_stream("Test"):
            chunks.append(chunk)
        
        assert len(chunks) == 3
        assert chunks[0]["type"] == "content"
        assert chunks[0]["content"] == "Hello"
        assert chunks[2]["content"] == "Raw text"


class TestRAGServiceUtilities:
    """Test utility methods."""
    
    @pytest.mark.asyncio
    async def test_embed_documents(self, mock_rag_config):
        """Test document embedding."""
        service = RAGService(config=mock_rag_config)
        service._initialized = True
        service.app.embed_documents = AsyncMock()
        
        documents = [
            {"id": "1", "content": "Test document", "metadata": {"title": "Test"}}
        ]
        
        await service.embed_documents("MEDIA_DB", documents)
        
        service.app.embed_documents.assert_called_once()
        call_args = service.app.embed_documents.call_args[0]
        assert len(call_args[0]) == 1
        assert call_args[0][0].id == "1"
        assert call_args[1] == DataSource.MEDIA_DB
    
    def test_get_stats(self, mock_rag_config):
        """Test getting service statistics."""
        service = RAGService(config=mock_rag_config)
        service._initialized = True
        service.app._cache = Mock(get_stats=Mock(return_value={"hits": 10, "misses": 5}))
        
        stats = service.get_stats()
        
        assert stats["initialized"] is True
        assert stats["config"]["cache_enabled"] is True
        assert stats["cache"]["hits"] == 10
        assert stats["cache"]["misses"] == 5
    
    @pytest.mark.asyncio
    async def test_clear_cache(self, mock_rag_config):
        """Test cache clearing."""
        service = RAGService(config=mock_rag_config)
        service._initialized = True
        service.app.clear_cache = AsyncMock()
        
        await service.clear_cache()
        
        service.app.clear_cache.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_close(self, mock_rag_config):
        """Test service cleanup."""
        service = RAGService(config=mock_rag_config)
        service._initialized = True
        
        mock_retriever = Mock(close=Mock())
        service.app._retrievers = {DataSource.MEDIA_DB: mock_retriever}
        
        await service.close()
        
        mock_retriever.close.assert_called_once()


class TestCompatibilityWrapper:
    """Test the compatibility wrapper for migration."""
    
    @pytest.mark.asyncio
    async def test_enhanced_rag_pipeline_wrapper(self, tmp_path):
        """Test the enhanced_rag_pipeline compatibility wrapper."""
        from tldw_Server_API.app.core.RAG.rag_service.integration import enhanced_rag_pipeline
        
        media_db = tmp_path / "media.db"
        chacha_db = tmp_path / "chacha.db"
        media_db.touch()
        chacha_db.touch()
        
        with patch('tldw_Server_API.app.core.RAG.rag_service.integration.RAGService') as mock_service_class:
            mock_service = AsyncMock()
            mock_service.generate_answer = AsyncMock(return_value={
                "answer": "Test answer",
                "context_preview": "Test context",
                "sources": []
            })
            mock_service_class.return_value = mock_service
            
            result = await enhanced_rag_pipeline(
                query="Test query",
                api_choice="openai",
                media_db_path=media_db,
                chachanotes_db_path=chacha_db,
                keywords="test,keywords",
                fts_top_k=20,
                vector_top_k=15,
                apply_re_ranking=False
            )
            
            # Verify service was configured correctly
            assert mock_service.config.retriever.fts_top_k == 20
            assert mock_service.config.retriever.vector_top_k == 15
            assert mock_service.config.processor.enable_reranking is False
            
            # Verify result format
            assert result["answer"] == "Test answer"
            assert result["context"] == "Test context"
            assert "source_documents" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])