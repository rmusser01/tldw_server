"""Tests for RAG configuration."""

import pytest
from pathlib import Path
import tempfile
import os

from ..config import RAGConfig, RetrieverConfig, ProcessorConfig, GeneratorConfig


class TestRAGConfig:
    """Test the RAGConfig class."""
    
    def test_default_config(self):
        """Test creating config with defaults."""
        config = RAGConfig()
        
        # Check defaults
        assert config.retriever.fts_top_k == 10
        assert config.retriever.vector_top_k == 10
        assert config.processor.enable_reranking == True
        assert config.generator.default_temperature == 0.7
        assert config.cache.enable_cache == True
        
    def test_config_validation(self):
        """Test configuration validation."""
        # Valid config
        config = RAGConfig()
        errors = config.validate()
        assert len(errors) == 0
        
        # Invalid retriever settings
        config.retriever.fts_top_k = 0
        config.retriever.hybrid_alpha = 1.5
        errors = config.validate()
        assert "retriever.fts_top_k must be >= 1" in errors
        assert "retriever.hybrid_alpha must be between 0 and 1" in errors
        
    def test_config_from_toml(self, tmp_path):
        """Test loading config from TOML file."""
        # Create test TOML file
        config_content = """
[rag]
batch_size = 64
num_workers = 8
use_gpu = false

[rag.retriever]
fts_top_k = 20
vector_top_k = 15
hybrid_alpha = 0.3

[rag.processor]
enable_reranking = false
reranker_top_k = 10

[rag.generator]
default_temperature = 0.5
max_tokens = 2048

[rag.cache]
enable_cache = false
cache_ttl = 7200
"""
        config_file = tmp_path / "test_config.toml"
        config_file.write_text(config_content)
        
        # Load config
        config = RAGConfig.from_toml(config_file)
        
        # Verify loaded values
        assert config.batch_size == 64
        assert config.num_workers == 8
        assert config.use_gpu == False
        assert config.retriever.fts_top_k == 20
        assert config.retriever.vector_top_k == 15
        assert config.retriever.hybrid_alpha == 0.3
        assert config.processor.enable_reranking == False
        assert config.generator.default_temperature == 0.5
        assert config.generator.max_tokens == 2048
        assert config.cache.enable_cache == False
        assert config.cache.cache_ttl == 7200
        
    def test_env_overrides(self, monkeypatch):
        """Test environment variable overrides."""
        # Set environment variables
        monkeypatch.setenv("RAG_FTS_TOP_K", "25")
        monkeypatch.setenv("RAG_ENABLE_RERANKING", "false")
        monkeypatch.setenv("RAG_DEFAULT_MODEL", "gpt-4")
        
        # Create config
        config = RAGConfig()
        
        # Check overrides were applied
        assert config.retriever.fts_top_k == 25
        assert config.processor.enable_reranking == False
        assert config.generator.default_model == "gpt-4"
        
    def test_config_to_dict(self):
        """Test converting config to dictionary."""
        config = RAGConfig()
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert "retriever" in config_dict
        assert "processor" in config_dict
        assert "generator" in config_dict
        assert config_dict["retriever"]["fts_top_k"] == 10