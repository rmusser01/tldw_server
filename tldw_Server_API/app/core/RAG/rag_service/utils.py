"""
Utility functions for the RAG service.
"""

import hashlib
import re
from typing import List, Dict, Any, Optional, Tuple
import tiktoken
from loguru import logger
import numpy as np
from collections import defaultdict


class TokenCounter:
    """Utility for counting tokens in text."""
    
    def __init__(self, model: str = "gpt-3.5-turbo"):
        """
        Initialize token counter.
        
        Args:
            model: Model name for tokenizer
        """
        try:
            self.encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            # Fallback to cl100k_base encoding
            self.encoding = tiktoken.get_encoding("cl100k_base")
            logger.warning(f"Model {model} not found, using cl100k_base encoding")
    
    def count(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.encoding.encode(text))
    
    def count_batch(self, texts: List[str]) -> List[int]:
        """Count tokens in multiple texts."""
        return [self.count(text) for text in texts]
    
    def truncate(self, text: str, max_tokens: int) -> str:
        """Truncate text to maximum token count."""
        tokens = self.encoding.encode(text)
        if len(tokens) <= max_tokens:
            return text
        
        truncated_tokens = tokens[:max_tokens]
        return self.encoding.decode(truncated_tokens)


def create_document_id(content: str, metadata: Dict[str, Any]) -> str:
    """
    Create a unique document ID from content and metadata.
    
    Args:
        content: Document content
        metadata: Document metadata
        
    Returns:
        Unique document ID
    """
    # Combine content hash with key metadata
    hasher = hashlib.md5()
    hasher.update(content.encode('utf-8'))
    
    # Add stable metadata to hash
    for key in sorted(metadata.keys()):
        if key in ['source_id', 'chunk_index', 'document_id']:
            hasher.update(f"{key}:{metadata[key]}".encode('utf-8'))
    
    return hasher.hexdigest()


def chunk_text(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 128,
    separator: str = "\n\n"
) -> List[Tuple[str, int]]:
    """
    Split text into overlapping chunks.
    
    Args:
        text: Text to chunk
        chunk_size: Target size of each chunk in characters
        chunk_overlap: Number of characters to overlap
        separator: Preferred separator for splitting
        
    Returns:
        List of (chunk_text, start_index) tuples
    """
    if not text:
        return []
    
    # Try to split on separator first
    if separator and separator in text:
        sections = text.split(separator)
        chunks = []
        current_chunk = ""
        current_start = 0
        
        for section in sections:
            if len(current_chunk) + len(section) + len(separator) <= chunk_size:
                if current_chunk:
                    current_chunk += separator + section
                else:
                    current_chunk = section
            else:
                if current_chunk:
                    chunks.append((current_chunk, current_start))
                current_chunk = section
                current_start = len(text) - len(separator.join(sections[sections.index(section):]))
        
        if current_chunk:
            chunks.append((current_chunk, current_start))
            
        return chunks
    
    # Fallback to simple overlapping chunks
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        
        # Try to break at word boundary
        if end < len(text) and not text[end].isspace():
            last_space = chunk.rfind(' ')
            if last_space > chunk_size * 0.8:  # Only break if we're not losing too much
                end = start + last_space
                chunk = text[start:end]
        
        chunks.append((chunk.strip(), start))
        start = end - chunk_overlap
    
    return chunks


def normalize_scores(scores: List[float], method: str = "minmax") -> List[float]:
    """
    Normalize scores to [0, 1] range.
    
    Args:
        scores: List of scores to normalize
        method: Normalization method ("minmax" or "zscore")
        
    Returns:
        Normalized scores
    """
    if not scores:
        return []
    
    scores_array = np.array(scores)
    
    if method == "minmax":
        min_score = scores_array.min()
        max_score = scores_array.max()
        if max_score == min_score:
            return [0.5] * len(scores)
        return ((scores_array - min_score) / (max_score - min_score)).tolist()
    
    elif method == "zscore":
        mean = scores_array.mean()
        std = scores_array.std()
        if std == 0:
            return [0.5] * len(scores)
        z_scores = (scores_array - mean) / std
        # Convert to [0, 1] using sigmoid
        return (1 / (1 + np.exp(-z_scores))).tolist()
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def combine_scores(
    scores_dict: Dict[str, List[float]],
    weights: Optional[Dict[str, float]] = None,
    normalize: bool = True
) -> List[float]:
    """
    Combine multiple score lists with optional weighting.
    
    Args:
        scores_dict: Dictionary mapping score type to score list
        weights: Optional weights for each score type
        normalize: Whether to normalize before combining
        
    Returns:
        Combined scores
    """
    if not scores_dict:
        return []
    
    # Get the length from first score list
    length = len(next(iter(scores_dict.values())))
    
    # Validate all lists have same length
    for name, scores in scores_dict.items():
        if len(scores) != length:
            raise ValueError(f"Score list '{name}' has different length")
    
    # Default weights if not provided
    if weights is None:
        weights = {name: 1.0 for name in scores_dict}
    
    # Normalize if requested
    if normalize:
        scores_dict = {
            name: normalize_scores(scores)
            for name, scores in scores_dict.items()
        }
    
    # Combine with weights
    combined = np.zeros(length)
    total_weight = sum(weights.get(name, 1.0) for name in scores_dict)
    
    for name, scores in scores_dict.items():
        weight = weights.get(name, 1.0) / total_weight
        combined += np.array(scores) * weight
    
    return combined.tolist()


def deduplicate_documents(
    documents: List[Any],
    key_func: callable,
    similarity_threshold: float = 0.85
) -> List[Any]:
    """
    Remove duplicate documents based on similarity.
    
    Args:
        documents: List of documents
        key_func: Function to extract comparison key from document
        similarity_threshold: Threshold for considering documents duplicate
        
    Returns:
        Deduplicated documents
    """
    if len(documents) <= 1:
        return documents
    
    seen = {}
    deduplicated = []
    
    for doc in documents:
        key = key_func(doc)
        is_duplicate = False
        
        # Check against seen documents
        for seen_key in seen:
            similarity = calculate_text_similarity(key, seen_key)
            if similarity >= similarity_threshold:
                is_duplicate = True
                # Keep the one with higher score
                if hasattr(doc, 'score') and hasattr(seen[seen_key], 'score'):
                    if doc.score > seen[seen_key].score:
                        # Replace with higher scoring document
                        deduplicated.remove(seen[seen_key])
                        del seen[seen_key]
                        seen[key] = doc
                        deduplicated.append(doc)
                break
        
        if not is_duplicate:
            seen[key] = doc
            deduplicated.append(doc)
    
    return deduplicated


def calculate_text_similarity(text1: str, text2: str) -> float:
    """
    Calculate simple text similarity using Jaccard index.
    
    Args:
        text1: First text
        text2: Second text
        
    Returns:
        Similarity score between 0 and 1
    """
    # Simple word-based Jaccard similarity
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if not words1 or not words2:
        return 0.0
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return len(intersection) / len(union)


def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """
    Extract keywords from text using simple frequency analysis.
    
    Args:
        text: Input text
        max_keywords: Maximum number of keywords to extract
        
    Returns:
        List of keywords
    """
    # Simple stopwords (extend as needed)
    stopwords = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
        'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'cannot'
    }
    
    # Extract words
    words = re.findall(r'\b\w+\b', text.lower())
    
    # Count frequencies
    word_freq = defaultdict(int)
    for word in words:
        if word not in stopwords and len(word) > 2:
            word_freq[word] += 1
    
    # Sort by frequency and return top keywords
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    return [word for word, _ in sorted_words[:max_keywords]]


def format_sources(documents: List[Any], max_sources: int = 5) -> str:
    """
    Format source documents for display.
    
    Args:
        documents: List of source documents
        max_sources: Maximum number of sources to show
        
    Returns:
        Formatted string of sources
    """
    if not documents:
        return "No sources available."
    
    sources = []
    for i, doc in enumerate(documents[:max_sources]):
        metadata = getattr(doc, 'metadata', {})
        source_type = metadata.get('source_type', 'Unknown')
        source_id = metadata.get('source_id', getattr(doc, 'id', f'doc_{i}'))
        
        source_str = f"{i+1}. [{source_type}] {source_id}"
        if 'title' in metadata:
            source_str += f" - {metadata['title']}"
        
        sources.append(source_str)
    
    if len(documents) > max_sources:
        sources.append(f"... and {len(documents) - max_sources} more sources")
    
    return "\n".join(sources)