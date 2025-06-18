"""
Configuration management for the RAG service.

This module provides configuration loading and validation for the RAG service,
integrating with the existing TOML configuration system.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import tomli
from loguru import logger


@dataclass
class RetrieverConfig:
    """Configuration for retrieval components."""
    fts_top_k: int = 10
    vector_top_k: int = 10
    hybrid_alpha: float = 0.5  # Weight for hybrid search (0=FTS only, 1=vector only)
    chunk_size: int = 512
    chunk_overlap: int = 128
    
    # Collection names for different data types
    media_collection: str = "media_embeddings"
    chat_collection: str = "chat_embeddings"
    notes_collection: str = "notes_embeddings"
    character_collection: str = "character_embeddings"


@dataclass
class ProcessorConfig:
    """Configuration for document processing."""
    enable_reranking: bool = True
    reranker_model: Optional[str] = None  # If None, uses FlashRank
    reranker_top_k: int = 5
    deduplication_threshold: float = 0.85  # Similarity threshold for deduplication
    max_context_length: int = 4096
    
    # Result combination strategy
    combination_method: str = "weighted"  # "weighted", "round_robin", "score_based"


@dataclass
class GeneratorConfig:
    """Configuration for response generation."""
    default_model: Optional[str] = None
    default_temperature: float = 0.7
    max_tokens: int = 1024
    system_prompt_template: str = """You are a helpful AI assistant. Use the following context to answer the user's question.
If the context doesn't contain relevant information, say so clearly.

Context:
{context}

Question: {question}
Answer:"""
    
    # Streaming configuration
    enable_streaming: bool = True
    stream_chunk_size: int = 10  # tokens


@dataclass
class ChromaConfig:
    """Configuration for ChromaDB."""
    persist_directory: Optional[str] = None
    collection_prefix: str = "tldw_rag"
    embedding_model: str = "all-MiniLM-L6-v2"  # Default sentence-transformers model
    embedding_dimension: int = 384
    distance_metric: str = "cosine"  # "cosine", "euclidean", "ip"


@dataclass
class CacheConfig:
    """Configuration for caching."""
    enable_cache: bool = True
    cache_ttl: int = 3600  # 1 hour in seconds
    max_cache_size: int = 1000  # Maximum number of cached items
    cache_embedding_results: bool = True
    cache_search_results: bool = True
    cache_llm_responses: bool = False  # Usually want fresh responses


@dataclass
class RAGConfig:
    """Main configuration class for the RAG service."""
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    processor: ProcessorConfig = field(default_factory=ProcessorConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    chroma: ChromaConfig = field(default_factory=ChromaConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    
    # Performance settings
    batch_size: int = 32
    num_workers: int = 4
    use_gpu: bool = True
    
    # Logging
    log_level: str = "INFO"
    log_performance_metrics: bool = True
    
    @classmethod
    def from_toml(cls, config_path: Optional[Path] = None) -> 'RAGConfig':
        """
        Load configuration from TOML file.
        
        Args:
            config_path: Path to config file. If None, uses default location.
            
        Returns:
            RAGConfig instance
        """
        if config_path is None:
            # Try multiple default locations
            possible_paths = [
                Path.home() / ".config" / "tldw_cli" / "config.toml",
                Path("config.toml"),
                Path(__file__).parent.parent.parent / "config.toml"
            ]
            
            for path in possible_paths:
                if path.exists():
                    config_path = path
                    break
            else:
                logger.warning("No config file found, using defaults")
                return cls()
        
        logger.info(f"Loading RAG config from: {config_path}")
        
        try:
            with open(config_path, "rb") as f:
                toml_data = tomli.load(f)
            
            # Extract RAG-specific configuration
            rag_config = toml_data.get("rag", {})
            
            # Create nested configs
            config = cls(
                retriever=RetrieverConfig(**rag_config.get("retriever", {})),
                processor=ProcessorConfig(**rag_config.get("processor", {})),
                generator=GeneratorConfig(**rag_config.get("generator", {})),
                chroma=ChromaConfig(**rag_config.get("chroma", {})),
                cache=CacheConfig(**rag_config.get("cache", {}))
            )
            
            # Apply top-level RAG settings
            for key in ["batch_size", "num_workers", "use_gpu", "log_level", "log_performance_metrics"]:
                if key in rag_config:
                    setattr(config, key, rag_config[key])
            
            # Override with environment variables if present
            config._apply_env_overrides()
            
            return config
            
        except Exception as e:
            logger.error(f"Error loading config from {config_path}: {e}")
            logger.warning("Using default configuration")
            return cls()
    
    def _apply_env_overrides(self):
        """Apply environment variable overrides."""
        # Examples of environment variables that can override config
        env_mappings = {
            "RAG_FTS_TOP_K": ("retriever", "fts_top_k", int),
            "RAG_VECTOR_TOP_K": ("retriever", "vector_top_k", int),
            "RAG_ENABLE_RERANKING": ("processor", "enable_reranking", lambda x: x.lower() == "true"),
            "RAG_ENABLE_CACHE": ("cache", "enable_cache", lambda x: x.lower() == "true"),
            "RAG_DEFAULT_MODEL": ("generator", "default_model", str),
            "RAG_CHROMA_PERSIST_DIR": ("chroma", "persist_directory", str),
        }
        
        for env_var, (section, attr, converter) in env_mappings.items():
            value = os.environ.get(env_var)
            if value is not None:
                try:
                    section_obj = getattr(self, section)
                    setattr(section_obj, attr, converter(value))
                    logger.debug(f"Override from env: {env_var} -> {section}.{attr} = {value}")
                except Exception as e:
                    logger.warning(f"Failed to apply env override {env_var}: {e}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "retriever": self.retriever.__dict__,
            "processor": self.processor.__dict__,
            "generator": self.generator.__dict__,
            "chroma": self.chroma.__dict__,
            "cache": self.cache.__dict__,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "use_gpu": self.use_gpu,
            "log_level": self.log_level,
            "log_performance_metrics": self.log_performance_metrics
        }
    
    def validate(self) -> List[str]:
        """
        Validate configuration settings.
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Validate retriever settings
        if self.retriever.fts_top_k < 1:
            errors.append("retriever.fts_top_k must be >= 1")
        if self.retriever.vector_top_k < 1:
            errors.append("retriever.vector_top_k must be >= 1")
        if not 0 <= self.retriever.hybrid_alpha <= 1:
            errors.append("retriever.hybrid_alpha must be between 0 and 1")
        
        # Validate processor settings
        if self.processor.reranker_top_k < 1:
            errors.append("processor.reranker_top_k must be >= 1")
        if not 0 <= self.processor.deduplication_threshold <= 1:
            errors.append("processor.deduplication_threshold must be between 0 and 1")
        
        # Validate generator settings
        if self.generator.max_tokens < 1:
            errors.append("generator.max_tokens must be >= 1")
        if not 0 <= self.generator.default_temperature <= 2:
            errors.append("generator.default_temperature should be between 0 and 2")
        
        # Validate cache settings
        if self.cache.cache_ttl < 0:
            errors.append("cache.cache_ttl must be >= 0")
        if self.cache.max_cache_size < 1:
            errors.append("cache.max_cache_size must be >= 1")
        
        return errors