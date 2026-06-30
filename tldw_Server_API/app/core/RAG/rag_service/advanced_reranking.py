"""
Advanced reranking strategies for RAG search results.

This module provides sophisticated reranking capabilities including:
- Cross-encoder based reranking
- LLM-based relevance scoring
- Diversity-aware reranking (MMR)
- Multi-criteria reranking
- Hybrid reranking strategies
"""

import asyncio
import math
import os
import time
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, Optional

import numpy as np
from loguru import logger

from tldw_Server_API.app.core.Utils.prompt_loader import load_prompt

from .types import DataSource, Document
from .utils import get_float_env as _get_float_env


class RerankingStrategy(Enum):
    """Available reranking strategies."""
    FLASHRANK = "flashrank"          # Fast neural reranking
    CROSS_ENCODER = "cross_encoder"  # Cross-encoder models
    LLM_SCORING = "llm_scoring"      # LLM-based relevance
    DIVERSITY = "diversity"          # MMR for diversity
    MULTI_CRITERIA = "multi_criteria" # Multiple ranking factors
    HYBRID = "hybrid"                # Combine strategies
    LLAMA_CPP = "llama_cpp"          # Embedding-based rerank via llama.cpp GGUF
    TWO_TIER = "two_tier"            # Cross-encoder prefilter then LLM rerank


_RERANKER_TRUST_REMOTE_CODE_PATTERNS = (
    "mixedbread-ai/mxbai-rerank*",
)


def _normalize_hf_patterns(patterns: Optional[Iterable[Any]]) -> list[str]:
    if not patterns:
        return []
    if isinstance(patterns, str):
        return [p.strip() for p in patterns.split(",") if p.strip()]
    out: list[str] = []
    for p in patterns:
        if p is None:
            continue
        s = str(p).strip()
        if s:
            out.append(s)
    return out


def _hf_trusts_remote_code(model_name: str, patterns: Optional[Iterable[Any]]) -> bool:
    if not model_name:
        return False
    pats = _normalize_hf_patterns(patterns)
    if not pats:
        return False
    model_l = str(model_name).lower()
    return any(fnmatch(model_name, pat) or fnmatch(model_l, str(pat).lower()) for pat in pats)


def _repo_root_dir() -> Path:
    """
    Resolve repository root from this module location.

    advanced_reranking.py is at:
    tldw_Server_API/app/core/RAG/rag_service/advanced_reranking.py
    so repo root is parents[5].
    """
    try:
        return Path(__file__).resolve().parents[5]
    except Exception:  # noqa: BLE001 - fallback to current working dir
        return Path.cwd()


def _resolve_flashrank_cache_dir(raw_value: Optional[str]) -> str:
    """Resolve flashrank cache dir, defaulting to repo-local models/flashrank."""
    repo_root = _repo_root_dir()
    default_dir = repo_root / "models" / "flashrank"
    if raw_value is None:
        return str(default_dir)

    normalized = str(raw_value).strip()
    if not normalized:
        return str(default_dir)
    candidate = Path(normalized).expanduser()
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    return str(candidate)


def _load_flashrank_defaults_from_config() -> tuple[str, Optional[str]]:
    """Load FlashRank model/cache defaults from env/config."""
    model_name = os.getenv("RAG_FLASHRANK_MODEL_NAME")
    cache_dir = os.getenv("RAG_FLASHRANK_CACHE_DIR")

    if model_name and cache_dir:
        return model_name, cache_dir

    try:
        from tldw_Server_API.app.core.config import load_comprehensive_config
        cp = load_comprehensive_config()
        if cp and hasattr(cp, "has_section") and cp.has_section("RAG"):
            if not model_name:
                model_name = cp.get("RAG", "flashrank_model_name", fallback=None)
            if not cache_dir:
                cache_dir = cp.get("RAG", "flashrank_cache_dir", fallback=None)
    except Exception:  # noqa: BLE001 - best effort lookup
        logger.debug("FlashRank config lookup failed; using defaults")

    return (model_name or "ms-marco-TinyBERT-L-2-v2"), cache_dir


class LlamaEmbeddingError(RuntimeError):
    """Raised when llama-embedding returns an error."""

    def __init__(self) -> None:
        super().__init__("llama-embedding error")


class LlamaEmbeddingParseError(ValueError):
    """Raised when no embeddings can be parsed from llama-embedding output."""

    def __init__(self) -> None:
        super().__init__("No embeddings parsed from llama-embedding output")


def _raise_llama_embedding_error() -> None:
    raise LlamaEmbeddingError()


def _raise_llama_embedding_parse_error() -> None:
    raise LlamaEmbeddingParseError()


@dataclass
class RerankingConfig:
    """Configuration for reranking."""
    strategy: RerankingStrategy = RerankingStrategy.FLASHRANK
    model_name: Optional[str] = None
    top_k: int = 10
    diversity_weight: float = 0.3
    relevance_weight: float = 0.7
    min_similarity_threshold: float = 0.3
    batch_size: int = 32
    use_gpu: bool = False
    criteria_weights: dict[str, float] = field(default_factory=lambda: {
        "relevance": 0.4,
        "recency": 0.2,
        "source_quality": 0.2,
        "length": 0.2
    })
    # llama.cpp (GGUF) reranker options
    llama_binary: Optional[str] = None  # path or program name (defaults to 'llama-embedding')
    llama_ngl: Optional[int] = None     # n_gpu_layers (-ngl)
    llama_embd_separator: str = "<#sep#>"
    llama_embd_output_format: str = "json+"
    llama_pooling: str = "last"
    llama_normalize: int = -1
    llama_max_doc_chars: int = 2000
    # Instruction/prefix formatting for instruct-style embedding models (e.g., BGE)
    llama_template_mode: Optional[str] = None  # 'auto' | 'bge' | 'jina' | 'none'
    llama_query_prefix: Optional[str] = None   # e.g., 'query: '
    llama_doc_prefix: Optional[str] = None     # e.g., 'passage: '
    # Transformers Cross-Encoder options
    transformers_device: Optional[str] = None  # 'auto' | 'cuda' | 'cpu'
    transformers_trust_remote_code: bool = False
    transformers_max_length: Optional[int] = None
    hf_revision: Optional[str] = None
    # Two-tier gating overrides (optional per-request)
    min_relevance_prob: Optional[float] = None
    sentinel_margin: Optional[float] = None


@dataclass
class ScoredDocument:
    """Document with detailed scoring information."""
    document: Document
    original_score: float
    rerank_score: float
    relevance_score: float = 0.0
    diversity_score: float = 0.0
    criteria_scores: dict[str, float] = field(default_factory=dict)
    explanation: Optional[str] = None

    @property
    def final_score(self) -> float:
        """Calculate final score."""
        return self.rerank_score


class BaseReranker(ABC):
    """Base class for all reranking strategies."""

    def __init__(self, config: RerankingConfig):
        """
        Initialize reranker.

        Args:
            config: Reranking configuration
        """
        self.config = config
        self._cache: dict[str, Any] = {}

    @abstractmethod
    async def rerank(
        self,
        query: str,
        documents: list[Document],
        original_scores: Optional[list[float]] = None
    ) -> list[ScoredDocument]:
        """
        Rerank documents based on query.

        Args:
            query: Search query
            documents: Documents to rerank
            original_scores: Original retrieval scores

        Returns:
            List of reranked documents with scores
        """
        pass

    def _normalize_scores(self, scores: list[float]) -> list[float]:
        """Normalize scores to [0, 1] range."""
        if not scores:
            return []

        min_score = min(scores)
        max_score = max(scores)

        if max_score == min_score:
            return [0.5] * len(scores)

        return [(s - min_score) / (max_score - min_score) for s in scores]


class FlashRankReranker(BaseReranker):
    """Fast neural reranking using FlashRank."""

    def __init__(self, config: RerankingConfig):
        """Initialize FlashRank reranker."""
        super().__init__(config)
        self._ranker = None
        default_model_name, default_cache_dir = _load_flashrank_defaults_from_config()
        self._flashrank_model_name = default_model_name
        self._flashrank_cache_dir = _resolve_flashrank_cache_dir(None)

        try:
            from flashrank import Ranker
            configured_model = (
                config.model_name
                or default_model_name
            )
            configured_cache = default_cache_dir
            resolved_cache_dir = _resolve_flashrank_cache_dir(configured_cache)
            os.makedirs(resolved_cache_dir, exist_ok=True)

            self._ranker = Ranker(
                model_name=str(configured_model),
                cache_dir=resolved_cache_dir
            )
            self._flashrank_model_name = str(configured_model)
            self._flashrank_cache_dir = resolved_cache_dir
            logger.info(
                "FlashRank reranker initialized (model='{}', cache_dir='{}')",
                self._flashrank_model_name,
                self._flashrank_cache_dir,
            )
        except ImportError:
            logger.warning("FlashRank not available")
        except Exception as e:  # noqa: BLE001 - fallback should not fail request
            logger.warning(
                "FlashRank initialization failed (model='{}', cache_dir='{}'): {}. "
                "Falling back to retrieval order.",
                self._flashrank_model_name,
                self._flashrank_cache_dir,
                e,
            )

    async def rerank(
        self,
        query: str,
        documents: list[Document],
        original_scores: Optional[list[float]] = None
    ) -> list[ScoredDocument]:
        """Rerank using FlashRank."""
        if not self._ranker or not documents:
            # Fallback to original scores
            return [
                ScoredDocument(
                    document=doc,
                    original_score=original_scores[i] if original_scores else doc.score,
                    rerank_score=original_scores[i] if original_scores else doc.score,
                    relevance_score=original_scores[i] if original_scores else doc.score
                )
                for i, doc in enumerate(documents)
            ]

        # Prepare passages for reranking
        passages = [
            {"id": i, "text": doc.content[:1000]}  # Limit content length
            for i, doc in enumerate(documents)
        ]

        # Rerank
        try:
            # Create rerank request - FlashRank rerank() takes a RerankRequest
            from flashrank import RerankRequest
            request = RerankRequest(query=query, passages=passages)
            results = self._ranker.rerank(request)

            # Create scored documents
            scored_docs = []
            for result in results:
                idx = result["id"]
                scored_docs.append(ScoredDocument(
                    document=documents[idx],
                    original_score=original_scores[idx] if original_scores else documents[idx].score,
                    rerank_score=result["score"],
                    relevance_score=result["score"]
                ))

            return scored_docs[:self.config.top_k]

        except Exception as e:  # noqa: BLE001 - fallback to original order
            logger.error("FlashRank reranking failed (error_type={})", type(e).__name__)
            # Fallback to original order
            return [
                ScoredDocument(
                    document=doc,
                    original_score=original_scores[i] if original_scores else doc.score,
                    rerank_score=original_scores[i] if original_scores else doc.score
                )
                for i, doc in enumerate(documents[:self.config.top_k])
            ]


class LlamaCppReranker(BaseReranker):
    """Embedding-based reranker using llama.cpp `llama-embedding` on GGUF models.

    Builds a single prompt with separators: [query] <sep> [doc1] <sep> [doc2] ...
    Parses per-segment embeddings and computes cosine similarity between the
    query embedding and each candidate to produce rerank scores.
    """

    def __init__(self, config: RerankingConfig):
        super().__init__(config)
        import os
        from shutil import which
        # Load env/config fallbacks lazily to avoid tight coupling
        try:
            from tldw_Server_API.app.core.config import load_and_log_configs
            _cfg = load_and_log_configs() or {}
        except Exception:  # noqa: BLE001 - config load best-effort
            _cfg = {}

        self.binary = (
            config.llama_binary
            or os.getenv("RAG_LLAMA_RERANKER_BIN")
            or _cfg.get("RAG_LLAMA_RERANKER_BIN")
            or "llama-embedding"
        )
        self.model_path = (
            config.model_name or os.getenv("RAG_LLAMA_RERANKER_MODEL") or _cfg.get("RAG_LLAMA_RERANKER_MODEL")
        )
        self.sep = (
            config.llama_embd_separator
            or os.getenv("RAG_LLAMA_RERANKER_SEP")
            or _cfg.get("RAG_LLAMA_RERANKER_SEP")
            or "<#sep#>"
        )
        try:
            self.ngl = (
                config.llama_ngl
                if config.llama_ngl is not None else (
                    int(os.getenv("RAG_LLAMA_RERANKER_NGL", "0")) if os.getenv("RAG_LLAMA_RERANKER_NGL") is not None else (
                        int(_cfg.get("RAG_LLAMA_RERANKER_NGL", 0)) if isinstance(_cfg.get("RAG_LLAMA_RERANKER_NGL"), (int, str)) else 0
                    )
                )
            )
        except (TypeError, ValueError):
            self.ngl = 0
        cfg_output = _cfg.get("RAG_LLAMA_RERANKER_OUTPUT", "json+")
        if not isinstance(cfg_output, str):
            cfg_output = str(cfg_output)
        env_output = os.getenv("RAG_LLAMA_RERANKER_OUTPUT")
        self.embd_format = config.llama_embd_output_format or env_output or cfg_output
        # Pooling may vary by model family; default later if none provided
        env_pooling = os.getenv("RAG_LLAMA_RERANKER_POOLING")
        cfg_pooling = _cfg.get("RAG_LLAMA_RERANKER_POOLING")
        cfg_pooling_str = cfg_pooling if isinstance(cfg_pooling, str) else ""
        self.pooling = config.llama_pooling or env_pooling or cfg_pooling_str
        try:
            self.normalize = (
                config.llama_normalize
                if config.llama_normalize is not None else (
                    int(os.getenv("RAG_LLAMA_RERANKER_NORMALIZE", "-1")) if os.getenv("RAG_LLAMA_RERANKER_NORMALIZE") is not None else (
                        int(_cfg.get("RAG_LLAMA_RERANKER_NORMALIZE", -1)) if isinstance(_cfg.get("RAG_LLAMA_RERANKER_NORMALIZE"), (int, str)) else -1
                    )
                )
            )
        except (TypeError, ValueError):
            self.normalize = -1
        try:
            self.max_doc_chars = (
                config.llama_max_doc_chars
                if config.llama_max_doc_chars is not None else (
                    int(os.getenv("RAG_LLAMA_RERANKER_MAX_DOC_CHARS", "2000")) if os.getenv("RAG_LLAMA_RERANKER_MAX_DOC_CHARS") is not None else (
                        int(_cfg.get("RAG_LLAMA_RERANKER_MAX_DOC_CHARS", 2000)) if isinstance(_cfg.get("RAG_LLAMA_RERANKER_MAX_DOC_CHARS"), (int, str)) else 2000
                    )
                )
            )
        except (TypeError, ValueError):
            self.max_doc_chars = 2000

        # Template mode and prefixes
        self.template_mode = (config.llama_template_mode or os.getenv("RAG_LLAMA_RERANKER_TEMPLATE_MODE") or _cfg.get("RAG_LLAMA_RERANKER_TEMPLATE_MODE") or "auto").lower()
        self.query_prefix = config.llama_query_prefix or os.getenv("RAG_LLAMA_RERANKER_QUERY_PREFIX") or _cfg.get("RAG_LLAMA_RERANKER_QUERY_PREFIX")
        self.doc_prefix = config.llama_doc_prefix or os.getenv("RAG_LLAMA_RERANKER_DOC_PREFIX") or _cfg.get("RAG_LLAMA_RERANKER_DOC_PREFIX")

        # Auto-detect defaults based on model name if not explicitly set
        model_l = (str(self.model_path or "").lower())
        is_bge = "bge" in model_l
        is_jina = "jina" in model_l
        is_qwen = "qwen" in model_l
        if (self.template_mode == "auto" or not self.template_mode) and is_bge:
            # BGE instruct convention
            self.query_prefix = self.query_prefix or "query: "
            self.doc_prefix = self.doc_prefix or "passage: "
        # Default pooling if still unset
        if not self.pooling:
            if is_bge or is_jina:
                self.pooling = "mean"
            elif is_qwen:
                self.pooling = "last"
            else:
                self.pooling = "mean"

        # Validate binary presence for logging only; graceful fallback happens in rerank()
        if which(self.binary) is None:
            logger.warning(f"llama-embedding binary '{self.binary}' not found on PATH; will fallback if invoked")
        if not self.model_path:
            logger.warning("RAG_LLAMA_RERANKER_MODEL not set; llama.cpp reranker will fallback")

    async def rerank(
        self,
        query: str,
        documents: list[Document],
        original_scores: Optional[list[float]] = None
    ) -> list[ScoredDocument]:
        import asyncio
        if not documents:
            return []

        # Do not cap input candidates here: must score all supplied docs and return top_k

        # Build prompt: query first, then candidate passages
        # Format query/documents (instruct-style prefixes when configured)
        def _fmt_q(txt: str) -> str:
            if self.query_prefix:
                return f"{self.query_prefix}{txt}"
            return txt
        def _fmt_d(txt: str) -> str:
            if self.doc_prefix:
                return f"{self.doc_prefix}{txt}"
            return txt

        texts = [_fmt_q(query)]
        for d in documents:
            t = (d.content or "")
            if self.max_doc_chars and self.max_doc_chars > 0:
                t = t[: self.max_doc_chars]
            texts.append(_fmt_d(t))
        prompt = self.sep.join(texts)

        # Prepare command
        cmd = [
            self.binary,
            "-m", str(self.model_path),
            "--embd-output-format", str(self.embd_format),
            "--embd-separator", str(self.sep),
            "-p", prompt,
            "--pooling", str(self.pooling),
            "--embd-normalize", str(self.normalize),
        ]
        if self.ngl and int(self.ngl) > 0:
            cmd.extend(["-ngl", str(int(self.ngl))])

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            out_b, err_b = await proc.communicate()
            if proc.returncode != 0:
                logger.error(f"llama-embedding failed (code {proc.returncode}): {err_b.decode('utf-8', 'ignore')[:500]}")
                _raise_llama_embedding_error()

            raw = out_b.decode("utf-8", "ignore")
            embs = self._parse_embeddings_output(raw)
            if not embs or len(embs) < 2:
                _raise_llama_embedding_parse_error()

            # Compute cosine similarity to first vector (query)
            q = np.asarray(embs[0], dtype=float)
            qn = np.linalg.norm(q) or 1.0
            scored = []
            for i, d in enumerate(documents):
                idx = i + 1  # because 0 is query
                if idx >= len(embs):
                    sim = float(getattr(d, "score", 0.0))
                else:
                    v = np.asarray(embs[idx], dtype=float)
                    vn = np.linalg.norm(v) or 1.0
                    sim = float(np.dot(q, v) / (qn * vn))
                    # normalize to [0,1] from [-1,1]
                    sim = (sim + 1.0) / 2.0
                scored.append(ScoredDocument(
                    document=d,
                    original_score=original_scores[i] if original_scores else d.score,
                    rerank_score=sim,
                    relevance_score=sim,
                    explanation="llama.cpp cosine(query, doc)"
                ))

            scored.sort(key=lambda x: x.rerank_score, reverse=True)
            return scored[: self.config.top_k]

        except Exception as e:  # noqa: BLE001 - fallback to original order
            logger.error("LlamaCppReranker failed (error_type={})", type(e).__name__)
            return [
                ScoredDocument(
                    document=doc,
                    original_score=original_scores[i] if original_scores else doc.score,
                    rerank_score=original_scores[i] if original_scores else doc.score,
                )
                for i, doc in enumerate(documents[: self.config.top_k])
            ]

    def _parse_embeddings_output(self, s: str) -> Optional[list[list[float]]]:
        """Parse embeddings from llama-embedding stdout.

        Supports a few common shapes:
        - {"embeddings": [[...], [...], ...]}
        - {"data": [{"embedding": [...]}, ...]}
        - A JSON array of arrays
        - Best-effort regex extraction of large float arrays
        """
        import json
        import re
        try:
            data = json.loads(s)
            if isinstance(data, dict):
                if "embeddings" in data and isinstance(data["embeddings"], list):
                    return [list(map(float, vec)) for vec in data["embeddings"] if isinstance(vec, (list, tuple))]
                if "data" in data and isinstance(data["data"], list):
                    out = []
                    for item in data["data"]:
                        emb = item.get("embedding") if isinstance(item, dict) else None
                        if isinstance(emb, (list, tuple)):
                            out.append(list(map(float, emb)))
                    if out:
                        return out
            if isinstance(data, list):
                # Could be [[...], [...]] or [{embedding:[...]}]
                out = []
                for item in data:
                    if isinstance(item, (list, tuple)):
                        out.append(list(map(float, item)))
                    elif isinstance(item, dict) and isinstance(item.get("embedding"), list):
                        out.append(list(map(float, item["embedding"])))
                if out:
                    return out
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

        # Fallback: regex extract arrays of floats; keep reasonably long ones
        arrays: list[list[float]] = []
        for m in re.finditer(r"\[(?:\s*-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\s*,\s*){8,}-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\s*\]", s):
            try:
                vec = json.loads(m.group(0))
                if isinstance(vec, list) and len(vec) >= 8:
                    arrays.append([float(x) for x in vec])
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
        return arrays if arrays else None


class TransformersCrossEncoderReranker(BaseReranker):
    """Cross-encoder reranking using HuggingFace/Sentence-Transformers models.

    Works with models like 'BAAI/bge-reranker-v2-m3' and other CE models.
    """

    def __init__(self, config: RerankingConfig):
        super().__init__(config)
        self._ce = None
        self._using_st = False
        self._device = (config.transformers_device or "auto").lower()
        self._max_length = config.transformers_max_length or None
        self._trust_remote_code = bool(config.transformers_trust_remote_code)
        self._hf_revision = (
            config.hf_revision
            or os.getenv("RAG_TRANSFORMERS_RERANKER_REVISION")
            or None
        )

        model_id = config.model_name or None
        if not model_id:
            # Attempt to load from global config
            try:
                from tldw_Server_API.app.core.config import load_and_log_configs
                _cfg = load_and_log_configs() or {}
                model_id = _cfg.get("RAG_TRANSFORMERS_RERANKER_MODEL")
            except Exception:  # noqa: BLE001 - config load best-effort
                model_id = None
        # Auto-enable trust_remote_code for known reranker families that require it,
        # unless explicitly overridden by request config.
        if model_id and not self._trust_remote_code:
            try:
                from tldw_Server_API.app.core.config import load_and_log_configs
                _cfg = load_and_log_configs() or {}
                allowlist = _cfg.get("TRUSTED_HF_REMOTE_CODE_MODELS") or []
            except Exception:  # noqa: BLE001 - config load best-effort
                allowlist = []
            allowlist = list(allowlist) + list(_RERANKER_TRUST_REMOTE_CODE_PATTERNS)
            if _hf_trusts_remote_code(model_id, allowlist):
                self._trust_remote_code = True

        if model_id:
            try:
                # Prefer sentence-transformers CrossEncoder if available
                try:
                    from sentence_transformers import CrossEncoder
                    self._ce = CrossEncoder(model_id, device=None if self._device == "auto" else self._device, trust_remote_code=self._trust_remote_code)
                    self._using_st = True
                    logger.info(f"Loaded CrossEncoder model via sentence-transformers: {model_id}")
                except Exception:  # noqa: BLE001 - fallback to transformers
                    # Fallback: raw transformers pipeline if provided
                    import torch
                    from transformers import AutoModelForSequenceClassification, AutoTokenizer
                    self._tokenizer = AutoTokenizer.from_pretrained(  # nosec B615
                        model_id,
                        revision=self._hf_revision,
                        trust_remote_code=self._trust_remote_code,
                    )
                    self._model = AutoModelForSequenceClassification.from_pretrained(  # nosec B615
                        model_id,
                        revision=self._hf_revision,
                        trust_remote_code=self._trust_remote_code,
                    )
                    self._model.eval()
                    self._torch = torch
                    if self._device != "auto":
                        self._model.to(self._device)
                    logger.info(f"Loaded cross-encoder model via transformers: {model_id}")
            except Exception as e:  # noqa: BLE001 - model loading best-effort
                logger.warning("Failed to load transformers reranker model (error_type={})", type(e).__name__)

    async def rerank(
        self,
        query: str,
        documents: list[Document],
        original_scores: Optional[list[float]] = None
    ) -> list[ScoredDocument]:
        if not documents:
            return []

        if self._ce is None and not hasattr(self, "_model"):
            logger.warning("TransformersCrossEncoderReranker not available; returning original order")
            return [
                ScoredDocument(
                    document=doc,
                    original_score=original_scores[i] if original_scores else doc.score,
                    rerank_score=original_scores[i] if original_scores else doc.score,
                    relevance_score=original_scores[i] if original_scores else doc.score,
                )
                for i, doc in enumerate(documents[: self.config.top_k])
            ]

        # Build pairs
        pairs = [(query, (d.content or "")[: self.config.llama_max_doc_chars if self.config.llama_max_doc_chars else None]) for d in documents]

        try:
            scores: list[float]
            if self._using_st and self._ce is not None:
                # sentence-transformers CrossEncoder
                scores = list(map(float, self._ce.predict(pairs, batch_size=self.config.batch_size)))
            else:
                # Raw transformers inference
                tok = self._tokenizer
                model = self._model
                torch = self._torch
                all_scores: list[float] = []
                bs = max(1, int(self.config.batch_size))
                for i in range(0, len(pairs), bs):
                    batch = pairs[i:i+bs]
                    texts1 = [a for a, _ in batch]
                    texts2 = [b for _, b in batch]
                    enc = tok(texts1, texts2, padding=True, truncation=True, max_length=self._max_length or 512, return_tensors="pt")
                    with torch.no_grad():
                        if self._device != "auto":
                            enc = {k: v.to(self._device) for k, v in enc.items()}
                        model_out = model(**enc)
                        logits = model_out.logits
                        # If single logit, apply sigmoid; if two logits, take softmax prob of class 1
                        if logits.shape[-1] == 1:
                            probs = torch.sigmoid(logits).squeeze(-1)
                        else:
                            probs = torch.softmax(logits, dim=-1)[..., -1]
                        all_scores.extend(probs.detach().cpu().tolist())
                scores = [float(s) for s in all_scores]

            # Normalize scores to [0,1]
            try:
                mn, mx = min(scores), max(scores)
                if mx > mn:
                    scores = [(s - mn) / (mx - mn) for s in scores]
            except (ValueError, TypeError):
                pass

            scored_out: list[ScoredDocument] = []
            for i, d in enumerate(documents):
                sc = scores[i] if i < len(scores) else float(getattr(d, "score", 0.0))
                scored_out.append(ScoredDocument(
                    document=d,
                    original_score=original_scores[i] if original_scores else d.score,
                    rerank_score=sc,
                    relevance_score=sc,
                    explanation="transformers cross-encoder"
                ))
            scored_out.sort(key=lambda x: x.rerank_score, reverse=True)
            return scored_out[: self.config.top_k]
        except Exception as e:  # noqa: BLE001 - fallback to original order
            logger.error("Transformers cross-encoder reranking failed (error_type={})", type(e).__name__)
            return [
                ScoredDocument(
                    document=doc,
                    original_score=original_scores[i] if original_scores else doc.score,
                    rerank_score=original_scores[i] if original_scores else doc.score,
                )
                for i, doc in enumerate(documents[: self.config.top_k])
            ]


class Qwen3CausalLMReranker(BaseReranker):
    """
    Qwen3 Transformers reranker that follows the official yes/no next-token
    judging prompt. It formats inputs exactly as specified and computes the
    probability of "yes" as the rerank score.

    Reference (requires transformers>=4.51.0):
    - AutoTokenizer / AutoModelForCausalLM
    - Left padding, ChatML-style system/user/assistant with <think> block
    - Final token distribution compared over token ids for "yes" and "no"
    """

    def __init__(self, config: RerankingConfig):
        super().__init__(config)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_id = config.model_name or "Qwen/Qwen3-Reranker-8B"
        model_revision = config.hf_revision or os.getenv("RAG_QWEN3_RERANKER_REVISION") or None

        # Mirror official example behavior
        self.tokenizer = AutoTokenizer.from_pretrained(  # nosec B615
            model_id,
            revision=model_revision,
            padding_side='left',
        )
        self.model = AutoModelForCausalLM.from_pretrained(  # nosec B615
            model_id,
            revision=model_revision,
        ).eval()
        self._torch = torch
        # Optional: honor configured device if provided
        self._device = (config.transformers_device or "auto").lower()
        if self._device != "auto":
            try:
                self.model.to(self._device)
            except Exception as device_move_error:  # noqa: BLE001 - device move best-effort
                logger.debug(
                    "LLM scoring reranker device move failed; keeping default device (error_type={})",
                    type(device_move_error).__name__,
                )

        # Yes/No token ids
        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")

        # Max length per official guidance
        self.max_length = self.config.transformers_max_length or 8192

        # Exact system/user/assistant prefix/suffix used by Qwen3-Reranker examples,
        # with system text editable via YAML key `qwen3_reranker_system`.
        default_system_text = (
            "Judge whether the Document meets the requirements based on the Query and the Instruct provided. "
            "Note that the answer can only be \"yes\" or \"no\"."
        )
        try:
            system_text = load_prompt("rag", "qwen3_reranker_system") or default_system_text
        except Exception:  # noqa: BLE001 - prompt load best-effort
            system_text = default_system_text
        self.system_text = system_text
        self.prefix = (
            "<|im_start|>system\n"
            f"{self.system_text}<|im_end|>\n"
            "<|im_start|>user\n"
        )
        self.suffix = (
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
            "<think>\n\n"
            "</think>\n\n"
        )
        self.prefix_tokens = self.tokenizer.encode(self.prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(self.suffix, add_special_tokens=False)

        # Default instruction text used inside the message body
        self.default_instruction = (
            "Given a web search query, retrieve relevant passages that answer the query"
        )

    # --- Formatting utilities (kept 1:1 with the reference example) ---
    def _format_instruction(self, instruction: Optional[str], query: str, doc: str) -> str:
        if instruction is None:
            instruction = self.default_instruction
        return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"

    def _process_inputs(self, pairs: list[str]):
        tok = self.tokenizer
        model = self.model

        inputs = tok(
            pairs,
            padding=False,
            truncation='longest_first',
            return_attention_mask=False,
            max_length=self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens),
        )
        for i, ele in enumerate(inputs['input_ids']):
            inputs['input_ids'][i] = self.prefix_tokens + ele + self.suffix_tokens
        inputs = tok.pad(inputs, padding=True, return_tensors="pt", max_length=self.max_length)
        for key in inputs:
            inputs[key] = inputs[key].to(model.device)
        return inputs

    def _compute_logits(self, inputs, **kwargs) -> list[float]:
        torch = self._torch
        with torch.no_grad():
            batch_scores = self.model(**inputs).logits[:, -1, :]
            true_vector = batch_scores[:, self.token_true_id]
            false_vector = batch_scores[:, self.token_false_id]
            batch_scores = torch.stack([false_vector, true_vector], dim=1)
            batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
            scores = batch_scores[:, 1].exp().tolist()
            return [float(s) for s in scores]

    async def rerank(
        self,
        query: str,
        documents: list[Document],
        original_scores: Optional[list[float]] = None,
    ) -> list[ScoredDocument]:
        if not documents:
            return []

        # Prefer unique YAML override for Qwen3 reranker, else fallback to default
        try:
            instruction_text = load_prompt("rag", "qwen3_reranker_instruction")
        except Exception:  # noqa: BLE001 - prompt load best-effort
            instruction_text = None
        if not instruction_text:
            instruction_text = None

        # Build string pairs that get wrapped by the Qwen system/user/assistant template
        pairs: list[str] = []
        for d in documents:
            txt = (getattr(d, 'content', '') or '')
            pairs.append(self._format_instruction(instruction_text, query, txt))

        # Batch if needed to control memory; otherwise single pass
        bs = max(1, int(self.config.batch_size))
        all_scores: list[float] = []
        for i in range(0, len(pairs), bs):
            batch_pairs = pairs[i:i+bs]
            inputs = self._process_inputs(batch_pairs)
            all_scores.extend(self._compute_logits(inputs))

        # Turn into ScoredDocument list and sort
        out: list[ScoredDocument] = []
        for i, d in enumerate(documents):
            sc = all_scores[i] if i < len(all_scores) else float(getattr(d, "score", 0.0))
            out.append(ScoredDocument(
                document=d,
                original_score=original_scores[i] if original_scores else d.score,
                rerank_score=sc,
                relevance_score=sc,
                explanation="qwen3 yes/no next-token",
            ))
        out.sort(key=lambda x: x.rerank_score, reverse=True)
        return out[: self.config.top_k]


# --- Compatibility helper for tests ---
def rerank_by_similarity(documents: list[Document], top_k: int = 10) -> list[Document]:
    """
    Simple similarity-based rerank placeholder to satisfy unit tests that patch
    this function. Sorts by `metadata.score` or `doc.score` if present.
    """
    def score_of(d: Document) -> float:
        try:
            return float(d.metadata.get('score', d.score))
        except (AttributeError, TypeError, ValueError):
            return getattr(d, 'score', 0.0)
    return sorted(documents, key=score_of, reverse=True)[:top_k]


class DiversityReranker(BaseReranker):
    """
    Maximal Marginal Relevance (MMR) reranking for diversity.

    Balances relevance with diversity to avoid redundant results.
    """

    def __init__(self, config: RerankingConfig):
        """Initialize diversity reranker."""
        super().__init__(config)
        self.lambda_param = config.diversity_weight

    async def rerank(
        self,
        query: str,
        documents: list[Document],
        original_scores: Optional[list[float]] = None
    ) -> list[ScoredDocument]:
        """
        Rerank using MMR algorithm.

        MMR = λ * Relevance(doc) - (1-λ) * max(Similarity(doc, selected))
        """
        if not documents:
            return []

        # Normalize original scores
        scores = original_scores if original_scores else [doc.score for doc in documents]
        norm_scores = self._normalize_scores(scores)

        # Initialize result set
        selected_indices = []
        selected_docs = []
        remaining_indices = list(range(len(documents)))

        # Select first document (highest relevance)
        first_idx = int(np.argmax(norm_scores))
        selected_indices.append(first_idx)
        selected_docs.append(ScoredDocument(
            document=documents[first_idx],
            original_score=scores[first_idx],
            rerank_score=norm_scores[first_idx],
            relevance_score=norm_scores[first_idx],
            diversity_score=0.0
        ))
        remaining_indices.remove(first_idx)

        # Select remaining documents
        while remaining_indices and len(selected_docs) < self.config.top_k:
            mmr_scores = []

            for idx in remaining_indices:
                # Relevance score
                relevance = norm_scores[idx]

                # Maximum similarity to selected documents
                max_sim = 0.0
                for selected_idx in selected_indices:
                    sim = self._compute_similarity(
                        documents[idx].content,
                        documents[selected_idx].content
                    )
                    max_sim = max(max_sim, sim)

                # MMR score
                mmr = self.lambda_param * relevance - (1 - self.lambda_param) * max_sim
                mmr_scores.append((idx, mmr, relevance, 1 - max_sim))

            # Select document with highest MMR
            if mmr_scores:
                best_idx, best_mmr, rel_score, div_score = max(mmr_scores, key=lambda x: x[1])
                best_idx_int = int(best_idx)
                selected_indices.append(best_idx_int)
                selected_docs.append(ScoredDocument(
                    document=documents[best_idx_int],
                    original_score=scores[best_idx_int],
                    rerank_score=best_mmr,
                    relevance_score=rel_score,
                    diversity_score=div_score
                ))
                remaining_indices.remove(best_idx_int)

        return selected_docs

    def _compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute similarity between two texts.

        Simple Jaccard similarity for demonstration.
        In production, use embeddings or more sophisticated methods.
        """
        # Simple word-based Jaccard similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1 & words2
        union = words1 | words2

        return len(intersection) / len(union) if union else 0.0


class MultiCriteriaReranker(BaseReranker):
    """
    Rerank based on multiple criteria.

    Combines various factors like relevance, recency, source quality, etc.
    """

    async def rerank(
        self,
        query: str,
        documents: list[Document],
        original_scores: Optional[list[float]] = None
    ) -> list[ScoredDocument]:
        """Rerank using multiple criteria."""
        if not documents:
            return []

        scored_docs = []

        for i, doc in enumerate(documents):
            # Calculate individual criteria scores
            criteria_scores = {}

            # Relevance (from original score)
            criteria_scores["relevance"] = original_scores[i] if original_scores else doc.score

            # Recency (based on metadata if available)
            criteria_scores["recency"] = self._calculate_recency_score(doc)

            # Source quality
            criteria_scores["source_quality"] = self._calculate_source_quality(doc)

            # Document length (prefer moderate length)
            criteria_scores["length"] = self._calculate_length_score(doc)

            # Normalize all scores
            for key in criteria_scores:
                criteria_scores[key] = max(0.0, min(1.0, criteria_scores[key]))

            # Calculate weighted final score
            final_score = sum(
                criteria_scores.get(criterion, 0.0) * weight
                for criterion, weight in self.config.criteria_weights.items()
            )

            scored_docs.append(ScoredDocument(
                document=doc,
                original_score=original_scores[i] if original_scores else doc.score,
                rerank_score=final_score,
                relevance_score=criteria_scores["relevance"],
                criteria_scores=criteria_scores
            ))

        # Sort by final score
        scored_docs.sort(key=lambda x: x.rerank_score, reverse=True)

        return scored_docs[:self.config.top_k]

    def _calculate_recency_score(self, doc: Document) -> float:
        """Calculate recency score based on document age."""
        # Check for timestamp in metadata
        if doc.metadata and "created_at" in doc.metadata:
            # Simple linear decay based on age
            # In production, use proper time-based scoring
            return 0.5  # Default for now
        return 0.5  # Neutral score if no timestamp

    def _calculate_source_quality(self, doc: Document) -> float:
        """Calculate source quality score."""
        # Source-based quality scoring
        source_scores = {
            DataSource.MEDIA_DB: 0.8,
            DataSource.CHAT_HISTORY: 0.7,
            DataSource.NOTES: 0.9,
            DataSource.CHARACTER_CARDS: 0.6,
            DataSource.KANBAN: 0.85,
        }
        return source_scores.get(doc.source, 0.5)

    def _calculate_length_score(self, doc: Document) -> float:
        """Calculate length score (prefer moderate length)."""
        content_length = len(doc.content)

        # Ideal length range (characters)
        ideal_min = 100
        ideal_max = 1000

        if ideal_min <= content_length <= ideal_max:
            return 1.0
        elif content_length < ideal_min:
            return content_length / ideal_min
        else:
            # Decay for very long documents
            return max(0.3, 1.0 - (content_length - ideal_max) / 10000)


class HybridReranker(BaseReranker):
    """
    Combines multiple reranking strategies.

    Uses voting or weighted combination of different rerankers.
    """

    def __init__(self, config: RerankingConfig, strategies: Optional[list[BaseReranker]] = None):
        """
        Initialize hybrid reranker.

        Args:
            config: Reranking configuration
            strategies: List of reranking strategies to combine
        """
        super().__init__(config)

        if strategies:
            self.strategies = strategies
        else:
            # Default combination
            self.strategies = [
                FlashRankReranker(config),
                DiversityReranker(config),
                MultiCriteriaReranker(config)
            ]

        # Strategy weights (could be configurable)
        self.strategy_weights = {
            "FlashRankReranker": 0.4,
            "DiversityReranker": 0.3,
            "MultiCriteriaReranker": 0.3
        }

    async def rerank(
        self,
        query: str,
        documents: list[Document],
        original_scores: Optional[list[float]] = None
    ) -> list[ScoredDocument]:
        """Rerank using multiple strategies and combine results."""
        if not documents:
            return []

        # Run all strategies in parallel
        rerank_tasks = [
            strategy.rerank(query, documents, original_scores)
            for strategy in self.strategies
        ]

        all_results = await asyncio.gather(*rerank_tasks)

        # Create document score map
        doc_scores: dict[int, dict[str, Any]] = {}

        for strategy_idx, results in enumerate(all_results):
            strategy_name = type(self.strategies[strategy_idx]).__name__
            weight = self.strategy_weights.get(strategy_name, 1.0)

            for rank, scored_doc in enumerate(results):
                doc_id = id(scored_doc.document)

                if doc_id not in doc_scores:
                    doc_scores[doc_id] = {
                        "document": scored_doc.document,
                        "original_score": float(scored_doc.original_score),
                        "weighted_scores": [],
                        "strategy_scores": {}
                    }

                # Use reciprocal rank for position-based scoring
                position_score = 1.0 / (rank + 1)
                weighted_score = position_score * weight

                entry = doc_scores[doc_id]
                weighted_scores = entry.get("weighted_scores")
                if isinstance(weighted_scores, list):
                    weighted_scores.append(float(weighted_score))
                strategy_scores = entry.get("strategy_scores")
                if isinstance(strategy_scores, dict):
                    strategy_scores[strategy_name] = float(scored_doc.rerank_score)

        # Calculate final scores
        final_scored_docs = []

        for doc_data in doc_scores.values():
            # Combine weighted scores
            weighted_scores = doc_data.get("weighted_scores")
            weights_list = weighted_scores if isinstance(weighted_scores, list) else []
            final_score = sum(weights_list) / len(self.strategies) if weights_list else 0.0
            doc = doc_data.get("document")
            if not isinstance(doc, Document):
                continue
            strategy_scores = doc_data.get("strategy_scores")
            criteria_scores = strategy_scores if isinstance(strategy_scores, dict) else {}

            final_scored_docs.append(ScoredDocument(
                document=doc,
                original_score=float(doc_data.get("original_score", 0.0)),
                rerank_score=final_score,
                criteria_scores=criteria_scores,
                explanation=f"Combined from {len(self.strategies)} strategies"
            ))

        # Sort by final score
        final_scored_docs.sort(key=lambda x: x.rerank_score, reverse=True)

        return final_scored_docs[:self.config.top_k]


class LLMReranker(BaseReranker):
    """
    LLM-based reranking for high-quality relevance scoring.

    Uses a language model to score query-document relevance.
    Note: This is expensive and should be used sparingly.
    """

    def __init__(self, config: RerankingConfig, llm_client=None):
        """
        Initialize LLM reranker.

        Args:
            config: Reranking configuration
            llm_client: LLM client for scoring
        """
        super().__init__(config)
        self.llm_client = llm_client

    async def rerank(
        self,
        query: str,
        documents: list[Document],
        original_scores: Optional[list[float]] = None
    ) -> list[ScoredDocument]:
        """Rerank using LLM for relevance scoring."""
        if not self.llm_client or not documents:
            # Fallback to original scores
            return [
                ScoredDocument(
                    document=doc,
                    original_score=original_scores[i] if original_scores else doc.score,
                    rerank_score=original_scores[i] if original_scores else doc.score
                )
                for i, doc in enumerate(documents[:self.config.top_k])
            ]

        # Score documents in batches
        scored_docs = []
        batch_size = self.config.batch_size

        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            batch_scores = await self._score_batch(query, batch)

            for j, doc in enumerate(batch):
                scored_docs.append(ScoredDocument(
                    document=doc,
                    original_score=original_scores[i + j] if original_scores else doc.score,
                    rerank_score=batch_scores[j],
                    relevance_score=batch_scores[j],
                    explanation="LLM relevance score"
                ))

        # Sort by score
        scored_docs.sort(key=lambda x: x.rerank_score, reverse=True)

        return scored_docs[:self.config.top_k]

    async def _score_batch(self, query: str, documents: list[Document]) -> list[float]:
        """Score a batch of documents using LLM.

        Attempts to call the shared analyze() function with a reranking instruction
        loaded from Prompts/rag. Falls back to naive scores on error.
        """
        sgl: Optional[Any] = None
        try:
            import tldw_Server_API.app.core.LLM_Calls.Summarization_General_Lib as sgl
        except ImportError:
            sgl = None

        instruction = load_prompt("rag", "reranking_instruction") or (
            "Rerank by how well the passage answers the query. Output a number between 0 and 1."
        )

        # Guardrails: cap docs, set per-call timeout and total budget
        def _inc_counter(name: str, value: int = 1) -> None:
            try:
                from .metrics_collector import get_metrics_collector  # lazy import to avoid heavy deps when unused
                get_metrics_collector().increment(name, value)
            except Exception as metrics_error:  # noqa: BLE001 - metrics best-effort
                logger.debug(
                    "LLM reranker local metrics increment failed (error_type={})",
                    type(metrics_error).__name__,
                )
            # Also export to central metrics registry (Prometheus/OTel) when available
            try:
                from tldw_Server_API.app.core.Metrics.metrics_manager import increment_counter
                mapping = {
                    "reranker.llm.timeouts": "rag_reranker_llm_timeouts_total",
                    "reranker.llm.exceptions": "rag_reranker_llm_exceptions_total",
                    "reranker.llm.budget_exhausted": "rag_reranker_llm_budget_exhausted_total",
                    "reranker.llm.docs_scored": "rag_reranker_llm_docs_scored_total",
                }
                metric_name = mapping.get(name)
                if metric_name:
                    increment_counter(metric_name, value, labels={"strategy": "llm_scoring"})
            except Exception as metrics_error:  # noqa: BLE001 - metrics best-effort
                logger.debug(
                    "LLM reranker central metrics increment failed (error_type={})",
                    type(metrics_error).__name__,
                )
        try:
            per_call_timeout = float(os.getenv("RAG_LLM_RERANK_TIMEOUT_SEC", "10"))
        except (TypeError, ValueError):
            per_call_timeout = 10.0
        try:
            total_budget = float(os.getenv("RAG_LLM_RERANK_TOTAL_BUDGET_SEC", "20"))
        except (TypeError, ValueError):
            total_budget = 20.0
        try:
            max_docs_env = int(os.getenv("RAG_LLM_RERANK_MAX_DOCS", "20"))
        except (TypeError, ValueError):
            max_docs_env = 20
        # Also respect config.top_k to avoid excessive LLM calls
        safe_limit = max(1, min(len(documents), self.config.top_k or len(documents), max_docs_env))

        scores: list[float] = []
        start_ts = time.time()
        for doc in documents[:safe_limit]:
            passage = getattr(doc, 'content', '') or ''
            # Build a compact prompt for scoring
            prompt = (
                f"Query:\n{query}\n\nPassage:\n{passage[:1500]}\n\n"
                f"Instruction:\n{instruction}\n\nOnly output a number between 0 and 1 with up to 3 decimals."
            )

            score_val: float = 0.5
            if self.llm_client and hasattr(self.llm_client, 'analyze'):
                try:
                    out = await asyncio.wait_for(asyncio.to_thread(self.llm_client.analyze, prompt), timeout=per_call_timeout)
                    score_val = _parse_float_score(out, default=0.5)
                except asyncio.TimeoutError:
                    _inc_counter("reranker.llm.timeouts")
                    score_val = 0.5
                except Exception:  # noqa: BLE001 - LLM scoring best-effort
                    _inc_counter("reranker.llm.exceptions")
                    score_val = 0.5
            elif sgl is not None:
                try:
                    # Use default provider from env/config inside analyze
                    out = await asyncio.wait_for(
                        asyncio.to_thread(
                            lambda _passage=passage, _prompt=prompt: sgl.analyze(
                                api_name='openai',
                                input_data=_passage,
                                prompt=_prompt,
                                context=None,
                                user_embedding_config=None,
                            )
                        ),
                        timeout=per_call_timeout,
                    )
                    score_val = _parse_float_score(out, default=0.5)
                except asyncio.TimeoutError:
                    _inc_counter("reranker.llm.timeouts")
                    score_val = 0.5
                except Exception:  # noqa: BLE001 - LLM scoring best-effort
                    _inc_counter("reranker.llm.exceptions")
                    score_val = 0.5
            scores.append(score_val)

            # Check total budget
            if (time.time() - start_ts) >= total_budget:
                _inc_counter("reranker.llm.budget_exhausted")
                break

        # Number of documents scored (for visibility)
        try:
            _inc_counter("reranker.llm.docs_scored", len(scores))
        except Exception as metrics_error:  # noqa: BLE001 - metrics best-effort
            logger.debug(
                "LLM reranker docs_scored metric failed (error_type={})",
                type(metrics_error).__name__,
            )

        # Normalize to [0,1]
        try:
            mn, mx = min(scores), max(scores)
            if mx > mn:
                scores = [(s - mn) / (mx - mn) for s in scores]
        except (ValueError, TypeError):
            pass
        return scores


def _parse_float_score(text: Any, default: float = 0.5) -> float:
    s = str(text).strip()
    # Extract first float-like token
    import re
    m = re.search(r"([01](?:\.\d+)?|0?\.\d+)", s)
    if m:
        try:
            val = float(m.group(1))
        except (TypeError, ValueError):
            return default
        # Clamp to [0,1]
        return max(0.0, min(1.0, val))
    return default


def create_reranker(strategy: RerankingStrategy, config: Optional[RerankingConfig] = None, llm_client=None) -> BaseReranker:
    """
    Factory function to create a reranker.

    Args:
        strategy: Reranking strategy to use
        config: Optional configuration

    Returns:
        Reranker instance
    """
    if config is None:
        config = RerankingConfig(strategy=strategy)

    if strategy == RerankingStrategy.FLASHRANK:
        return FlashRankReranker(config)
    elif strategy == RerankingStrategy.DIVERSITY:
        return DiversityReranker(config)
    elif strategy == RerankingStrategy.MULTI_CRITERIA:
        return MultiCriteriaReranker(config)
    elif strategy == RerankingStrategy.HYBRID:
        return HybridReranker(config)
    elif strategy == RerankingStrategy.LLM_SCORING:
        return LLMReranker(config, llm_client=llm_client)
    elif strategy == RerankingStrategy.CROSS_ENCODER:
        # Auto-detect Qwen3-Reranker generative models and route to the
        # specialized CausalLM-based reranker that uses the official prompt.
        model_l = str(config.model_name or "").lower()
        if ("qwen3" in model_l and "reranker" in model_l) or model_l.startswith("qwen/qwen3-reranker"):
            return Qwen3CausalLMReranker(config)
        return TransformersCrossEncoderReranker(config)
    elif strategy == RerankingStrategy.LLAMA_CPP:
        return LlamaCppReranker(config)
    elif strategy == RerankingStrategy.TWO_TIER:
        # Two-tier uses CE for fast pass and LLM for final scoring
        return TwoTierReranker(config, llm_client=llm_client)
    else:
        # Default to FlashRank
        return FlashRankReranker(config)


class TwoTierReranker(BaseReranker):
    """
    Two-tier reranking pipeline:
      1) Fast cross-encoder reranking (e.g., BAAI/bge-reranker) to select top N (default 50)
      2) LLM-based reranking on that shortlist to top K (default 10)

    Also injects a small sentinel "irrelevant" document to calibrate low-evidence
    scenarios and computes a calibrated probability of relevance via a logistic
    mapping over features (original, CE score, LLM score). The final rerank score
    is the calibrated probability.

    Exposes last run calibration in self.last_metadata for the pipeline to gate
    answer generation.
    """

    def __init__(self, config: RerankingConfig, llm_client=None, cross_reranker: Optional[BaseReranker] = None, llm_reranker: Optional[BaseReranker] = None):
        super().__init__(config)
        # Determine shortlist sizes (stage1 >> stage2)
        try:
            stage2_top_k = int(max(1, self.config.top_k))
        except (TypeError, ValueError):
            stage2_top_k = 10
        try:
            # default stage1 to max(50, 5x stage2) but bounded by 100
            stage1_top_k = min(100, max(50, stage2_top_k * 5))
        except (TypeError, ValueError):
            stage1_top_k = 50

        self.stage1_top_k = stage1_top_k
        self.stage2_top_k = stage2_top_k

        # Create internal rerankers if not injected (DI for tests)
        self._cross = cross_reranker
        self._llm = llm_reranker
        self.llm_client = llm_client

        if self._cross is None:
            # Auto-select cross-encoder model from config/env
            model_id = None
            try:
                from tldw_Server_API.app.core.config import load_and_log_configs
                cfg = load_and_log_configs() or {}
            except Exception:  # noqa: BLE001 - config load best-effort
                cfg = {}
            model_id = self.config.model_name or cfg.get("RAG_TRANSFORMERS_RERANKER_MODEL") or "BAAI/bge-reranker-v2-m3"
            ce_cfg = RerankingConfig(
                strategy=RerankingStrategy.CROSS_ENCODER,
                model_name=model_id,
                top_k=stage1_top_k,
                transformers_device=getattr(self.config, 'transformers_device', None),
                transformers_trust_remote_code=getattr(self.config, 'transformers_trust_remote_code', False),
                transformers_max_length=getattr(self.config, 'transformers_max_length', None),
            )
            self._cross = TransformersCrossEncoderReranker(ce_cfg)

        if self._llm is None:
            llm_cfg = RerankingConfig(
                strategy=RerankingStrategy.LLM_SCORING,
                top_k=stage2_top_k,
                batch_size=self.config.batch_size,
            )
            self._llm = LLMReranker(llm_cfg, llm_client=self.llm_client)

        # Place to store calibration/sentinel info for the caller
        self.last_metadata: dict[str, Any] = {}

    async def rerank(
        self,
        query: str,
        documents: list[Document],
        original_scores: Optional[list[float]] = None
    ) -> list[ScoredDocument]:
        if not documents:
            self.last_metadata = {"strategy": "two_tier", "reason": "no_documents"}
            return []

        cross = self._cross
        llm = self._llm
        if cross is None or llm is None:
            self.last_metadata = {"strategy": "two_tier", "reason": "missing_rerankers"}
            return [
                ScoredDocument(
                    document=doc,
                    original_score=original_scores[i] if original_scores else doc.score,
                    rerank_score=original_scores[i] if original_scores else doc.score,
                    relevance_score=original_scores[i] if original_scores else doc.score,
                )
                for i, doc in enumerate(documents[: self.config.top_k])
            ]

        # Inject sentinel doc for calibration
        sentinel = Document(
            id="sentinel:irrelevant",
            content=(
                "This passage is intentionally irrelevant to most queries. "
                "It contains generic filler text and should be scored as not relevant."
            ),
            metadata={"sentinel": True, "source": "synthetic"},
            source=DataSource.WEB_CONTENT,
            score=0.0,
        )

        pool_docs = list(documents)
        pool_docs.append(sentinel)
        # Stage 1: Cross-encoder fast scoring
        ce_t0 = time.time()
        ce_results: list[ScoredDocument] = await cross.rerank(query, pool_docs, original_scores=None)
        ce_dt = time.time() - ce_t0
        try:
            from tldw_Server_API.app.core.Metrics.metrics_manager import observe_histogram
            observe_histogram("rag_phase_duration_seconds", ce_dt, labels={"phase": "rerank_fast", "difficulty": "na"})
        except Exception as metrics_error:  # noqa: BLE001 - metrics best-effort
            logger.debug(
                "Two-tier reranker cross-encoder duration metric failed (error_type={})",
                type(metrics_error).__name__,
            )

        # Track CE scores in a map, and record sentinel CE score
        ce_scores: dict[str, float] = {}
        ce_sentinel_score: float = 0.0
        for sd in ce_results:
            did = getattr(sd.document, 'id', None) or str(id(sd.document))
            ce_scores[did] = float(sd.rerank_score)
            if getattr(sd.document, 'id', '') == sentinel.id:
                ce_sentinel_score = float(sd.rerank_score)

        # Sort desc by CE score and select top shortlist (excluding sentinel)
        ce_sorted = sorted(
            [sd for sd in ce_results if getattr(sd.document, 'id', '') != sentinel.id],
            key=lambda x: x.rerank_score,
            reverse=True,
        )
        shortlist_docs = [sd.document for sd in ce_sorted[: self.stage1_top_k]]

        # Stage 2: LLM reranking on shortlist + sentinel for calibration
        llm_input = list(shortlist_docs)
        llm_input.append(sentinel)
        llm_t0 = time.time()
        llm_results: list[ScoredDocument] = await llm.rerank(query, llm_input, original_scores=None)
        llm_dt = time.time() - llm_t0
        try:
            from tldw_Server_API.app.core.Metrics.metrics_manager import observe_histogram
            observe_histogram("rag_phase_duration_seconds", llm_dt, labels={"phase": "rerank_llm", "difficulty": "na"})
        except Exception as metrics_error:  # noqa: BLE001 - metrics best-effort
            logger.debug(
                "Two-tier reranker llm duration metric failed (error_type={})",
                type(metrics_error).__name__,
            )

        # Map LLM scores and capture sentinel
        llm_scores: dict[str, float] = {}
        llm_sentinel_score: float = 0.0
        llm_scored_docs: list[ScoredDocument] = []
        for sd in llm_results:
            did = getattr(sd.document, 'id', None) or str(id(sd.document))
            llm_scores[did] = float(sd.rerank_score)
            if did == sentinel.id:
                llm_sentinel_score = float(sd.rerank_score)
            else:
                llm_scored_docs.append(sd)

        # Calibrate final probability using logistic over (orig, CE, LLM)
        w0 = _get_float_env("RAG_RERANK_CALIB_BIAS", -1.5)
        w1 = _get_float_env("RAG_RERANK_CALIB_W_ORIG", 0.8)
        w2 = _get_float_env("RAG_RERANK_CALIB_W_CE", 2.5)
        w3 = _get_float_env("RAG_RERANK_CALIB_W_LLM", 3.0)

        def _logistic(x: float) -> float:
            try:
                return 1.0 / (1.0 + math.exp(-x))
            except OverflowError:
                # Safe fallback
                return 0.5

        final_scored: list[ScoredDocument] = []
        for sd in llm_scored_docs:
            did = getattr(sd.document, 'id', None) or str(id(sd.document))
            orig = float(getattr(sd.document, 'score', 0.0) or 0.0)
            ce = float(ce_scores.get(did, orig))
            llm_score = float(llm_scores.get(did, sd.rerank_score))
            # Simple bounded normalization for orig
            try:
                orig_n = max(0.0, min(1.0, orig))
            except (TypeError, ValueError):
                orig_n = 0.0
            logit = (w0 + (w1 * orig_n) + (w2 * ce) + (w3 * llm_score))
            prob = float(_logistic(logit))
            # Attach breakdown
            sd.criteria_scores = {
                **(sd.criteria_scores or {}),
                "orig": orig_n,
                "ce": ce,
                "llm": llm_score,
                "calibrated_prob": prob,
            }
            sd.rerank_score = prob
            sd.relevance_score = prob
            sd.explanation = "two_tier(logistic(orig,ce,llm))"
            final_scored.append(sd)

        # Sort by calibrated probability
        final_scored.sort(key=lambda x: x.rerank_score, reverse=True)
        out = final_scored[: self.stage2_top_k]

        # Compute sentinel calibrated probability too
        s_orig = 0.0
        s_ce = float(ce_sentinel_score)
        s_llm = float(llm_sentinel_score)
        s_prob = float(_logistic(w0 + (w1 * s_orig) + (w2 * s_ce) + (w3 * s_llm)))

        # Threshold gating
        # Per-request overrides take precedence over env defaults
        cfg_min = self.config.min_relevance_prob
        threshold = cfg_min if cfg_min is not None else _get_float_env("RAG_MIN_RELEVANCE_PROB", 0.35)
        cfg_margin = self.config.sentinel_margin
        margin_thr = cfg_margin if cfg_margin is not None else _get_float_env("RAG_SENTINEL_MARGIN", 0.10)
        top_prob = float(out[0].rerank_score) if out else 0.0
        prob_margin = top_prob - s_prob
        gated = (top_prob < threshold) or (prob_margin < margin_thr)

        # Persist metadata for unified pipeline to consume
        self.last_metadata = {
            "strategy": "two_tier",
            "cross_model": getattr(cross, 'config', None) and getattr(cross.config, 'model_name', None),
            "stage1_top_k": self.stage1_top_k,
            "stage2_top_k": self.stage2_top_k,
            "sentinel_scores": {"cross": s_ce, "llm": s_llm, "calibrated": s_prob},
            "top_doc_prob": top_prob,
            "threshold": threshold,
            "margin_threshold": margin_thr,
            "prob_margin": prob_margin,
            "gated": gated,
        }

        return out
