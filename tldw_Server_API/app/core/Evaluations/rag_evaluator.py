# rag_evaluator.py - RAG System Evaluation Module
"""
Evaluation module for RAG (Retrieval-Augmented Generation) systems.

Implements metrics:
- Relevance: How relevant is the response to the query
- Faithfulness: Is the response grounded in retrieved contexts
- Answer Similarity: Similarity to ground truth (if available)
- Context Precision: Precision of retrieved contexts
- Context Recall: Coverage of necessary information
"""

import asyncio
import contextlib
from collections.abc import Awaitable, Callable
from functools import partial
from typing import Any, Optional

import numpy as np
from loguru import logger

import tldw_Server_API.app.core.LLM_Calls.Summarization_General_Lib as sgl
from tldw_Server_API.app.core.AuthNZ.provider_credential_runtime import (
    ProviderCallCredentials,
)
from tldw_Server_API.app.core.Chat.bounded_daemon import (
    SYNC_ADAPTER_CALL_POOL,
    DaemonCapacityError,
    await_bounded_daemon_with_timeout,
    await_bounded_sync_call,
    await_owned_worker,
)
from tldw_Server_API.app.core.testing import is_test_mode

_RAG_EVAL_NONCRITICAL_EXCEPTIONS = (
    AssertionError,
    AttributeError,
    ConnectionError,
    FileNotFoundError,
    ImportError,
    IndexError,
    KeyError,
    LookupError,
    OSError,
    PermissionError,
    RuntimeError,
    TimeoutError,
    TypeError,
    UnicodeDecodeError,
    ValueError,
    np.linalg.LinAlgError,
)

RAG_EVALUATION_CALL_TIMEOUT_SECONDS = 60.0

# Safe import of embeddings helpers to avoid heavy deps during app import
try:
    from tldw_Server_API.app.core.Embeddings.Embeddings_Server.Embeddings_Create import (
        OpenAIModelCfg,
        create_embedding,
        get_embedding_config,
    )
    _RAG_EMBEDDINGS_AVAILABLE = True
except ImportError:
    _RAG_EMBEDDINGS_AVAILABLE = False
    def create_embedding(*args, **kwargs):  # type: ignore[misc]
        raise RuntimeError("Embeddings backend unavailable; install required dependencies")
    def get_embedding_config():  # type: ignore[misc]
        return {"embedding_config": {"default_model_id": ""}}
    class _FallbackOpenAIModelCfg(dict):
        """Lightweight stand-in when full embeddings stack is unavailable."""

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            for key, value in kwargs.items():
                setattr(self, key, value)

    OpenAIModelCfg = _FallbackOpenAIModelCfg  # type: ignore[assignment]

from tldw_Server_API.app.core.Claims_Extraction.claims_engine import ClaimsEngine
from tldw_Server_API.app.core.Evaluations.circuit_breaker import CircuitOpenError, llm_circuit_breaker
from tldw_Server_API.app.core.exceptions import raise_detached_error
from tldw_Server_API.app.core.LLM_Calls.Summarization_General_Lib import (
    SummaryProviderError,
)
from tldw_Server_API.app.core.RAG.rag_service.types import Document


# Module-level alias and helpers
def analyze(api_name: str, input_data: Any, custom_prompt_arg: Optional[str] = None, api_key: Optional[str] = None, system_message: Optional[str] = None, temp: Optional[float] = None, **kwargs) -> Any:
    """Alias wrapper for sgl.analyze to enable test monkeypatching.

    ClaimsEngine can supply a structured-output hint that the legacy SGL
    boundary does not accept. Its prompts already require JSON, so discard the
    unsupported hint instead of turning claim evaluation into a ``TypeError``.
    """
    kwargs.pop("response_format", None)
    kwargs.setdefault("raise_on_error", True)
    return sgl.analyze(api_name, input_data, custom_prompt_arg, api_key, system_message, temp, **kwargs)


async def _run_bounded_rag_analyze(*args: Any, **kwargs: Any) -> Any:
    """Run one sync evaluation call with process-local capacity and a deadline."""
    return await await_owned_worker(
        await_bounded_daemon_with_timeout(
            partial(analyze, *args, **kwargs),
            pool=SYNC_ADAPTER_CALL_POOL,
            name="rag-evaluation-analyze",
            timeout_seconds=RAG_EVALUATION_CALL_TIMEOUT_SECONDS,
            timeout_message="RAG evaluation provider call timed out",
            drain_after_timeout=True,
        )
    )


async def _run_circuit_bounded_rag_analyze(*args: Any, **kwargs: Any) -> Any:
    """Run a sync RAG adapter inside the breaker's single deadline."""
    return await await_bounded_sync_call(
        partial(analyze, *args, **kwargs),
        pool=SYNC_ADAPTER_CALL_POOL,
        exhaustion_message="RAG evaluation provider capacity is exhausted",
    )


def _simple_tokens(text: str) -> list[str]:
    """Lightweight tokenizer with basic stopword filtering for heuristics."""
    if not isinstance(text, str):
        text = str(text or "")
    text = text.lower()
    stop = {
        "the", "is", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with",
        "it", "this", "that", "as", "by", "at", "from", "are", "be"
    }
    cleaned = ''.join(ch if ch.isalnum() or ch.isspace() else ' ' for ch in text)
    tokens = [tok for tok in cleaned.split() if tok and tok not in stop and len(tok) > 2]
    return tokens


class RAGEvaluator:
    """Evaluator for RAG system performance"""

    def __init__(
        self,
        embedding_provider: Optional[str] = "openai",
        embedding_model: Optional[str] = "text-embedding-3-small",
        api_key: Optional[str] = None,
        *,
        app_config: Optional[dict[str, Any]] = None,
        credentials_resolved: bool = False,
        provider_credentials: ProviderCallCredentials | None = None,
    ):
        """
        Initialize RAG evaluator with production embeddings.

        Args:
            embedding_provider: Provider for embeddings (openai, huggingface, cohere)
            embedding_model: Model to use for embeddings
            api_key: Optional API key for the embedding provider
            app_config: Provider configuration captured with the API key
            credentials_resolved: Whether the key/config snapshot is authoritative
        """
        self.embedding_provider = embedding_provider
        self.embedding_model = embedding_model
        self.api_key = api_key
        self.app_config = app_config
        self.credentials_resolved = (
            provider_credentials is not None
            or credentials_resolved
        )
        self.provider_credentials = provider_credentials

        # If either provider or model is not specified (None/empty), treat embeddings as disabled.
        if not self.embedding_provider or not self.embedding_model:
            # Minimal config stub to avoid downstream KeyErrors when referenced
            self.embedding_config = {"embedding_config": {"default_model_id": ""}}
            # Explicitly mark as unavailable so property does not probe backends
            self._embedding_available = False
            logger.info("RAG evaluator initialized without embeddings (disabled by configuration)")
        else:
            # Get embedding configuration
            self.embedding_config = self._setup_embedding_config()

            # Lazy check - don't test embeddings on init, only when first used
            self._embedding_available = None  # Will be set on first use
            logger.info(f"RAG evaluator initialized with {embedding_provider}/{embedding_model} configuration")

    def _setup_embedding_config(self) -> dict[str, Any]:
        """Setup embedding configuration."""
        config = get_embedding_config()

        # Override model if specified
        if self.embedding_model:
            config["embedding_config"]["default_model_id"] = self.embedding_model

        # Add/override API key if provided - ensure target model has correct key
        if self.api_key and self.embedding_provider == "openai":
            models = config["embedding_config"].setdefault("models", {})
            existing = models.get(self.embedding_model)
            try:
                if existing is not None:
                    existing.api_key = self.api_key
                else:
                    models[self.embedding_model] = OpenAIModelCfg(
                        provider=self.embedding_provider,
                        model_name_or_path=self.embedding_model,
                        api_key=self.api_key
                    )
            except _RAG_EVAL_NONCRITICAL_EXCEPTIONS:
                models[self.embedding_model] = OpenAIModelCfg(
                    provider=self.embedding_provider,
                    model_name_or_path=self.embedding_model,
                    api_key=self.api_key
                )

            # Ensure batch embedding call can discover the API key: provide openai_api section
            # get_openai_embeddings_batch reads app_config['openai_api']['api_key']
            try:
                oa = config.setdefault("openai_api", {})
                oa.setdefault("api_key", self.api_key)
            except _RAG_EVAL_NONCRITICAL_EXCEPTIONS:
                pass

        return config

    @property
    def embedding_available(self) -> bool:
        """Check if embeddings are available (lazy evaluation)."""
        if self._embedding_available is None:
            self._embedding_available = self._test_embeddings()
            if not self._embedding_available:
                logger.warning(f"Embeddings not available for {self.embedding_provider}/{self.embedding_model}. Using LLM-based fallback.")
        return self._embedding_available

    @embedding_available.setter
    def embedding_available(self, value: bool):
        """Set embedding availability (for testing)."""
        self._embedding_available = value

    def _test_embeddings(self) -> bool:
        """Test if embeddings are available."""
        try:
            # Try to get a test embedding
            result = create_embedding("test", self.embedding_config, model_id_override=self.embedding_model)
            return result is not None and len(result) > 0
        except _RAG_EVAL_NONCRITICAL_EXCEPTIONS as e:
            logger.debug("Embeddings test failed error_type={}", type(e).__name__)
            return False

    async def evaluate(
        self,
        query: str,
        contexts: list[str],
        response: str,
        ground_truth: Optional[str] = None,
        metrics: Optional[list[str]] = None,
        api_name: str = "openai",
        metric_weights: Optional[dict[str, float]] = None,
        model: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Evaluate RAG system performance.

        Args:
            query: User query
            contexts: Retrieved context chunks
            response: Generated response
            ground_truth: Optional ground truth answer
            metrics: Metrics to compute
            api_name: LLM API to use

        Returns:
            Evaluation results with metrics and suggestions
        """
        caller_provided_metrics = metrics is not None
        if metrics is None:
            metrics = ["relevance", "faithfulness", "answer_similarity", "context_relevance"]

        results = {
            "metrics": {},
            "suggestions": []
        }

        # Evaluate each metric
        task_factories: list[Callable[[], Awaitable[tuple]]] = []
        metric_names = []  # Track which metrics we're actually evaluating

        if "relevance" in metrics or "answer_relevance" in metrics:
            task_factories.append(
                lambda: self._evaluate_relevance(query, response, api_name, model)
            )
            metric_names.append("relevance")

        if "faithfulness" in metrics or "answer_faithfulness" in metrics:
            task_factories.append(
                lambda: self._evaluate_faithfulness(response, contexts, api_name, model)
            )
            metric_names.append("faithfulness")

        if "answer_similarity" in metrics and ground_truth:
            task_factories.append(
                lambda: self._evaluate_answer_similarity(
                    response,
                    ground_truth,
                    api_name=api_name,
                    model=model,
                )
            )
            metric_names.append("answer_similarity")

        if "context_relevance" in metrics:
            task_factories.append(
                lambda: self._evaluate_context_relevance(query, contexts, api_name, model)
            )
            metric_names.append("context_relevance")

        if "context_precision" in metrics:
            task_factories.append(
                lambda: self._evaluate_context_precision(query, contexts, api_name, model)
            )
            metric_names.append("context_precision")

        if "context_recall" in metrics and ground_truth:
            task_factories.append(
                lambda: self._evaluate_context_recall(ground_truth, contexts, api_name, model)
            )
            metric_names.append("context_recall")

        # Optional: claim-level faithfulness using claim extraction + verification
        if "claim_faithfulness" in metrics:
            task_factories.append(
                lambda: self._evaluate_claim_faithfulness(response, contexts, api_name, model)
            )
            metric_names.append("claim_faithfulness")

        # Run evaluations in parallel with error handling
        try:
            metric_results: list[Any] = []
            if task_factories:
                # Prefer an adapter-backed metric as a single credential probe.
                # Answer similarity can complete heuristically without an LLM.
                probe_index = next(
                    (
                        index
                        for index, name in enumerate(metric_names)
                        if name != "answer_similarity"
                    ),
                    0,
                )
                probe_factory = task_factories.pop(probe_index)
                probe_name = metric_names.pop(probe_index)
                try:
                    probe_result: Any = await probe_factory()
                except (SummaryProviderError, DaemonCapacityError):
                    raise
                except Exception as exc:  # noqa: BLE001 - preserve per-metric partial results
                    probe_result = exc
                remaining = await asyncio.gather(
                    *(factory() for factory in task_factories),
                    return_exceptions=True,
                )
                metric_results = [probe_result, *remaining]
                metric_names = [probe_name, *metric_names]

            # Compile results and handle exceptions
            failed_metrics = []
            explicit_metrics = caller_provided_metrics
            # Provide aliases only when explicitly requested or when using defaults with ground truth
            set(metrics or []) & {"answer_relevance", "answer_faithfulness"}
            (
                isinstance(metric_weights, dict)
                and any(k in {"answer_relevance", "answer_faithfulness"} for k in metric_weights)
            )
            for i, result in enumerate(metric_results):
                if isinstance(result, Exception):
                    if isinstance(result, (SummaryProviderError, DaemonCapacityError)):
                        raise result
                    # Log the failure but continue with other metrics
                    metric_name = metric_names[i] if i < len(metric_names) else f"metric_{i}"
                    logger.error(
                        "Metric {} failed error_type={}",
                        metric_name,
                        type(result).__name__,
                    )
                    failed_metrics.append(metric_name)
                else:
                    metric_name, metric_result = result
                    # Always record canonical key
                    results["metrics"][metric_name] = metric_result

                    # Always include alias keys for relevance/faithfulness alongside canonical names
                    # Do not overwrite if already explicitly provided
                    if metric_name == "relevance":
                        results["metrics"].setdefault("answer_relevance", metric_result)
                    elif metric_name == "faithfulness":
                        results["metrics"].setdefault("answer_faithfulness", metric_result)

            # Add information about failed metrics
            if failed_metrics:
                results["failed_metrics"] = failed_metrics
                results["partial_results"] = True

            # For default metric set (caller did not explicitly request metrics),
            # prefer alias keys for relevance/faithfulness and drop canonical keys
            if not explicit_metrics:
                if "answer_relevance" in results["metrics"] and "relevance" in results["metrics"]:
                    try:
                        # Remove canonical to keep metric set compact and OpenAI-style
                        results["metrics"].pop("relevance", None)
                    except _RAG_EVAL_NONCRITICAL_EXCEPTIONS:
                        pass
                if "answer_faithfulness" in results["metrics"] and "faithfulness" in results["metrics"]:
                    with contextlib.suppress(_RAG_EVAL_NONCRITICAL_EXCEPTIONS):
                        results["metrics"].pop("faithfulness", None)
        except (SummaryProviderError, DaemonCapacityError):
            raise
        except _RAG_EVAL_NONCRITICAL_EXCEPTIONS as e:
            logger.error("Critical evaluation failure error_type={}", type(e).__name__)
            raise_detached_error(ValueError("Evaluation failed"))

        # Calculate overall score if we have metrics
        if results["metrics"]:
            results["overall_score"] = self._calculate_overall_score(results["metrics"], metric_weights)

        # Generate suggestions based on scores
        results["suggestions"] = self._generate_suggestions(results["metrics"])

        return results

    async def _evaluate_claim_faithfulness(self, response: str, contexts: list[str], api_name: str, model: Optional[str] = None) -> tuple:
        """Evaluate claim-level faithfulness by verifying extracted claims against contexts.

        Uses ClaimsEngine with APS-style extraction (gemma_aps) and hybrid verification.
        Returns a normalized score in [0,1].
        """
        try:
            # Wrap contexts into lightweight Document objects
            docs: list[Document] = []
            for i, ctx in enumerate(contexts or []):
                try:
                    docs.append(Document(id=f"ctx_{i+1}", content=str(ctx or ""), metadata={}))
                except _RAG_EVAL_NONCRITICAL_EXCEPTIONS:
                    # Minimal fallback if Document import shape changes
                    docs.append(Document(id=f"ctx_{i+1}", content=str(ctx or ""), metadata={}))

            def analyze_fn(
                _claims_provider,
                input_data,
                custom_prompt_arg,
                _claims_api_key=None,
                system_message=None,
                temp=None,
                streaming=False,
                recursive_summarization=False,
                chunked_summarization=False,
                chunk_options=None,
                model_override=None,
                **kwargs,
            ):
                """Bind ClaimsEngine calls to this evaluation's resolved snapshot."""
                del _claims_provider, _claims_api_key, model_override
                kwargs.pop("app_config", None)
                kwargs.pop("credentials_resolved", None)
                kwargs.pop("provider_credentials", None)
                kwargs.pop("_provider_call_credentials", None)
                return analyze(
                    api_name=api_name,
                    input_data=input_data,
                    custom_prompt_arg=custom_prompt_arg,
                    api_key=self.api_key,
                    system_message=system_message,
                    temp=temp,
                    streaming=streaming,
                    recursive_summarization=recursive_summarization,
                    chunked_summarization=chunked_summarization,
                    chunk_options=chunk_options,
                    model_override=model,
                    app_config=self.app_config,
                    credentials_resolved=self.credentials_resolved,
                    provider_credentials=self.provider_credentials,
                    **kwargs,
                )
            engine = ClaimsEngine(analyze_fn)
            claims_result = await engine.run(
                answer=response or "",
                query="",  # not needed for direct verification against contexts
                documents=docs,
                claim_extractor="aps",
                claim_verifier="hybrid",
                claims_top_k=5,
                claims_conf_threshold=0.7,
                claims_max=25,
                retrieve_fn=None,
                nli_model=None,
            )

            summary = claims_result.get("summary", {}) if isinstance(claims_result, dict) else {}
            supported = float(summary.get("supported", 0) or 0)
            total = float((summary.get("supported", 0) or 0) + (summary.get("refuted", 0) or 0) + (summary.get("nei", 0) or 0))
            score = (supported / total) if total > 0 else 0.0

            return ("claim_faithfulness", {
                "name": "claim_faithfulness",
                "score": float(max(0.0, min(1.0, score))),
                "raw_score": float(score * 5.0),  # map to 1-5 like other metrics if desired
                "explanation": "Fraction of claims supported by contexts (APS extraction + hybrid verification)"
            })
        except _RAG_EVAL_NONCRITICAL_EXCEPTIONS as e:
            logger.error("Claim faithfulness failed error_type={}", type(e).__name__)
            raise_detached_error(ValueError("Claim faithfulness evaluation failed"))

    async def _evaluate_relevance(self, query: str, response: str, api_name: str, model: Optional[str] = None) -> tuple:
        """Evaluate relevance of response to query"""
        prompt = f"""
        Evaluate how relevant the following response is to the given query.

        Query: {query}

        Response: {response}

        Rate the relevance on a scale of 1-5 where:
        1 = Completely irrelevant
        2 = Mostly irrelevant with minor relevant points
        3 = Partially relevant
        4 = Mostly relevant with minor gaps
        5 = Highly relevant and comprehensive

        Provide only the numeric score.
        """

        try:
            # Heuristic lexical overlap to dampen obvious mismatches
            q_tokens = set(_simple_tokens(query))
            r_tokens = set(_simple_tokens(response))
            union_tokens = q_tokens | r_tokens
            lexical_overlap = 0.0
            if q_tokens and r_tokens:
                lexical_overlap = len(q_tokens & r_tokens) / max(1, len(union_tokens))

            # Use circuit breaker for LLM call via local alias
            score_str = await llm_circuit_breaker.call_with_breaker(
                api_name,
                _run_circuit_bounded_rag_analyze,
                api_name,  # First param to analyze
                query,     # input_data
                prompt,    # custom_prompt_arg
                self.api_key,  # api_key (None to load from config)
                "You are an evaluation expert. Provide only numeric scores.",  # system_message
                0.1,       # temp
                model_override=model,
                app_config=self.app_config,
                credentials_resolved=self.credentials_resolved,
                provider_credentials=self.provider_credentials,
            )

            raw = float(score_str.strip())
            llm_score = raw / 5.0  # Normalize to 0-1

            # Clamp heuristics for obvious mismatches/strong matches to stabilize outputs
            # Only apply when we have enough lexical signal (>=3 distinct tokens)
            score = llm_score
            if len(union_tokens) >= 3:
                if lexical_overlap <= 0.10:
                    score = min(score, 0.35)
                elif lexical_overlap >= 0.60:
                    score = max(score, 0.75)

            return ("relevance", {
                "name": "relevance",
                "score": score,
                "raw_score": raw,
                "explanation": "Measures how well the response addresses the query"
            })

        except CircuitOpenError as e:
            logger.warning("Relevance circuit breaker open error_type={}", type(e).__name__)
            # Re-raise with more context
            raise_detached_error(
                ValueError("Service temporarily unavailable for relevance evaluation")
            )
        except DaemonCapacityError:
            raise
        except _RAG_EVAL_NONCRITICAL_EXCEPTIONS as e:
            logger.error("Relevance evaluation failed error_type={}", type(e).__name__)
            # Raise exception instead of returning 0.0
            raise_detached_error(ValueError("Relevance evaluation failed"))

    async def _evaluate_faithfulness(self, response: str, contexts: list[str], api_name: str, model: Optional[str] = None) -> tuple:
        """Evaluate if response is grounded in contexts"""
        combined_context = "\n\n".join(contexts)

        prompt = f"""
        Evaluate if the following response is faithful to the provided contexts.
        Check if all claims in the response are supported by the contexts.

        Contexts:
        {combined_context}

        Response:
        {response}

        Rate faithfulness on a scale of 1-5 where:
        1 = Contains many unsupported claims
        2 = Some major unsupported claims
        3 = Mix of supported and unsupported claims
        4 = Mostly supported with minor unsupported details
        5 = Fully supported by contexts

        Provide only the numeric score.
        """

        try:
            # Simple coverage heuristic: fraction of response tokens present in contexts
            ctx_tokens = set(_simple_tokens(combined_context))
            resp_tokens = set(_simple_tokens(response))
            coverage = 0.0
            if resp_tokens:
                coverage = len(resp_tokens & ctx_tokens) / len(resp_tokens)

            score_str = await llm_circuit_breaker.call_with_breaker(
                api_name,
                _run_circuit_bounded_rag_analyze,
                api_name,  # First param
                response,  # input_data
                prompt,    # custom_prompt_arg
                self.api_key,  # api_key (None to load from config)
                "You are an evaluation expert. Provide only numeric scores.",  # system_message
                0.1,       # temp
                model_override=model,
                app_config=self.app_config,
                credentials_resolved=self.credentials_resolved,
                provider_credentials=self.provider_credentials,
            )

            raw = float(score_str.strip())
            llm_score = raw / 5.0

            # Clamp for obvious hallucinations (very low coverage) / strong coverage
            score = llm_score
            if coverage <= 0.05:
                score = min(score, 0.55)
            elif coverage >= 0.70:
                score = max(score, 0.70)

            return ("faithfulness", {
                "name": "faithfulness",
                "score": score,
                "raw_score": raw,
                "explanation": "Measures if response is grounded in retrieved contexts"
            })

        except DaemonCapacityError:
            raise
        except _RAG_EVAL_NONCRITICAL_EXCEPTIONS as e:
            logger.error("Faithfulness evaluation failed error_type={}", type(e).__name__)
            # Raise exception instead of returning 0.0
            raise_detached_error(ValueError("Faithfulness evaluation failed"))

    async def _evaluate_answer_similarity(self, response: str, ground_truth: str, api_name: str = "openai", model: Optional[str] = None) -> tuple:
        """Evaluate similarity between response and ground truth"""
        # Imported here rather than at module scope: sklearn pulls scipy and
        # pandas (~0.9 s, ~1,300 modules) into every process that registers the
        # evaluations router, which startup does whether or not this runs.
        from sklearn.metrics.pairwise import cosine_similarity

        # Use embedding-based similarity if available
        if self.embedding_available:
            try:
                # Get embeddings for both texts (create_embedding is synchronous, so run in executor)
                loop = asyncio.get_event_loop()
                response_embedding = await loop.run_in_executor(
                    None, create_embedding, response, self.embedding_config, self.embedding_model
                )
                ground_truth_embedding = await loop.run_in_executor(
                    None, create_embedding, ground_truth, self.embedding_config, self.embedding_model
                )

                # Convert to numpy arrays and calculate cosine similarity
                response_embedding = np.array(response_embedding)
                ground_truth_embedding = np.array(ground_truth_embedding)

                # Reshape for sklearn if needed
                if response_embedding.ndim == 1:
                    response_embedding = response_embedding.reshape(1, -1)
                if ground_truth_embedding.ndim == 1:
                    ground_truth_embedding = ground_truth_embedding.reshape(1, -1)

                similarity = cosine_similarity(response_embedding, ground_truth_embedding)[0][0]

                # Convert similarity to 1-5 scale for consistency
                # Cosine similarity ranges from -1 to 1, but typically 0 to 1 for text
                # Map [0, 1] to [1, 5]
                raw_score = 1 + (similarity * 4)  # Maps 0->1, 0.5->3, 1->5
                score = similarity  # Keep normalized 0-1 score

                return ("answer_similarity", {
                    "name": "answer_similarity",
                    "score": score,
                    "raw_score": raw_score,
                    "explanation": "Embedding-based semantic similarity to ground truth answer",
                    "method": "embeddings"
                })

            except _RAG_EVAL_NONCRITICAL_EXCEPTIONS as e:
                logger.warning(
                    "Embedding-based similarity failed error_type={}; using fallback",
                    type(e).__name__,
                )
                # Only synthesize embeddings in TEST_MODE when an API key was explicitly provided
                if is_test_mode() and self.api_key and (self.embedding_provider == "openai"):
                    import hashlib
                    def _cheap_embed(text: str, dim: int = 128):
                        h = hashlib.sha256((text or "").encode("utf-8")).digest()
                        # Expand digest to desired dim deterministically
                        vals = []
                        while len(vals) < dim:
                            for b in h:
                                vals.append(((b / 255.0) - 0.5) * 2.0)
                                if len(vals) >= dim:
                                    break
                        return np.array(vals, dtype=float)
                    r_vec = _cheap_embed(response)
                    g_vec = _cheap_embed(ground_truth)
                    r_vec = r_vec.reshape(1, -1)
                    g_vec = g_vec.reshape(1, -1)
                    similarity = cosine_similarity(r_vec, g_vec)[0][0]
                    raw_score = 1 + (similarity * 4)
                    return ("answer_similarity", {
                        "name": "answer_similarity",
                        "score": float(similarity),
                        "raw_score": float(raw_score),
                        "explanation": "Synthetic embedding similarity (TEST_MODE)",
                        "method": "embeddings"
                    })
        # If embeddings are not available at all but we have api_key in TEST_MODE, synthesize
        else:
            if is_test_mode() and self.api_key and (self.embedding_provider == "openai"):
                import hashlib
                def _cheap_embed(text: str, dim: int = 128):
                    h = hashlib.sha256((text or "").encode("utf-8")).digest()
                    vals = []
                    while len(vals) < dim:
                        for b in h:
                            vals.append(((b / 255.0) - 0.5) * 2.0)
                            if len(vals) >= dim:
                                break
                    return np.array(vals, dtype=float)
                r_vec = _cheap_embed(response).reshape(1, -1)
                g_vec = _cheap_embed(ground_truth).reshape(1, -1)
                similarity = cosine_similarity(r_vec, g_vec)[0][0]
                raw_score = 1 + (similarity * 4)
                return ("answer_similarity", {
                    "name": "answer_similarity",
                    "score": float(similarity),
                    "raw_score": float(raw_score),
                    "explanation": "Synthetic embedding similarity (TEST_MODE)",
                    "method": "embeddings"
                })

        # Fallback to LLM-based similarity evaluation
        prompt = f"""
        Compare the similarity between the following response and ground truth answer.

        Response: {response}

        Ground Truth: {ground_truth}

        Rate similarity on a scale of 1-5 where:
        1 = Completely different meaning
        2 = Some shared concepts but mostly different
        3 = Moderately similar with key differences
        4 = Very similar with minor differences
        5 = Nearly identical meaning

        Provide only the numeric score.
        """

        # Heuristic fast-path for identical or near-identical texts
        import difflib
        r_norm = (response or "").strip().lower()
        g_norm = (ground_truth or "").strip().lower()
        if r_norm and g_norm:
            if r_norm == g_norm:
                return ("answer_similarity", {
                    "name": "answer_similarity",
                    "score": 1.0,
                    "raw_score": 5.0,
                    "explanation": "Identical texts",
                    "method": "heuristic"
                })
            ratio = difflib.SequenceMatcher(None, r_norm, g_norm).ratio()
            if ratio >= 0.95:
                return ("answer_similarity", {
                    "name": "answer_similarity",
                    "score": ratio,
                    "raw_score": 1 + 4 * ratio,
                    "explanation": "Near-identical texts",
                    "method": "heuristic"
                })

        try:
            # Use thread offload for deterministic unit-test mocking
            score_str = await _run_bounded_rag_analyze(
                api_name,  # api_name - first param
                response,   # input_data
                prompt,     # custom_prompt_arg
                self.api_key,  # api_key (None to load from config)
                "You are an evaluation expert. Provide only numeric scores.",  # system_message
                0.1,        # temp
                model_override=model,
                app_config=self.app_config,
                credentials_resolved=self.credentials_resolved,
                provider_credentials=self.provider_credentials,
            )

            score = float((score_str or "").strip()) / 5.0

            return ("answer_similarity", {
                "name": "answer_similarity",
                "score": score,
                "raw_score": float((score_str or "").strip() or 0.0),
                "explanation": "LLM-based semantic similarity to ground truth answer",
                "method": "llm"
            })

        except DaemonCapacityError:
            raise
        except _RAG_EVAL_NONCRITICAL_EXCEPTIONS as e:
            logger.error("Answer similarity failed error_type={}", type(e).__name__)
            # Raise exception instead of returning 0.0 (fixing error handling issue)
            raise_detached_error(ValueError("Answer similarity evaluation failed"))

    async def _evaluate_context_precision(self, query: str, contexts: list[str], api_name: str, model: Optional[str] = None) -> tuple:
        """Evaluate precision of retrieved contexts"""
        # Check each context for relevance
        relevance_scores = []

        for _i, context in enumerate(contexts):
            prompt = f"""
            Rate how relevant this context is to the query on a scale of 1-5.

            Query: {query}

            Context: {context}

            Provide only the numeric score.
            """

            try:
                score_str = await _run_bounded_rag_analyze(
                    api_name,  # First param
                    context,   # input_data
                    prompt,    # custom_prompt_arg
                    self.api_key,  # api_key (None to load from config)
                    "You are an evaluation expert. Provide only numeric scores.",  # system_message
                    0.1,       # temp
                    model_override=model,
                    app_config=self.app_config,
                    credentials_resolved=self.credentials_resolved,
                    provider_credentials=self.provider_credentials,
                )

                relevance_scores.append(float(score_str.strip()) / 5.0)

            except DaemonCapacityError:
                raise
            except _RAG_EVAL_NONCRITICAL_EXCEPTIONS as e:
                logger.debug(
                    "Context precision score failed error_type={}; defaulting to 0",
                    type(e).__name__,
                )
                relevance_scores.append(0.0)

        # Calculate precision as average relevance
        precision = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0
        # Calculate raw score on 1-5 scale
        raw_score = precision * 4 + 1 if precision > 0 else 1.0

        return ("context_precision", {
            "name": "context_precision",
            "score": precision,
            "raw_score": raw_score,
            "explanation": "Average relevance of retrieved contexts",
            "metadata": {"individual_scores": relevance_scores}
        })

    async def _evaluate_context_relevance(self, query: str, contexts: list[str], api_name: str, model: Optional[str] = None) -> tuple:
        """Evaluate relevance of retrieved contexts to the query"""
        # Check each context for relevance
        relevance_scores = []

        for context in contexts:
            prompt = f"""
            Rate how relevant this context is to the query on a scale of 1-5.

            Query: {query}

            Context: {context}

            Provide only the numeric score.
            """

            try:
                # Use thread offload for deterministic unit-test mocking
                score_str = await _run_bounded_rag_analyze(
                    api_name,  # First param
                    context,   # input_data
                    prompt,    # custom_prompt_arg
                    self.api_key,  # api_key (None to load from config)
                    "You are an evaluation expert. Provide only numeric scores.",  # system_message
                    0.1,       # temp
                    model_override=model,
                    app_config=self.app_config,
                    credentials_resolved=self.credentials_resolved,
                    provider_credentials=self.provider_credentials,
                )

                # Parse score and handle invalid responses
                try:
                    score = float((score_str or "").strip())
                    relevance_scores.append(score / 5.0)
                except (ValueError, AttributeError):
                    # Invalid response format - treat as 0.0
                    logger.warning("Invalid score format from evaluation provider")
                    relevance_scores.append(0.0)

            except DaemonCapacityError:
                raise
            except _RAG_EVAL_NONCRITICAL_EXCEPTIONS as e:
                logger.debug(
                    "Context relevance failed error_type={}; defaulting to 0",
                    type(e).__name__,
                )
                relevance_scores.append(0.0)

        # Calculate average relevance
        relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0
        raw_score = relevance * 4 + 1 if relevance > 0 else 1.0

        return ("context_relevance", {
            "name": "context_relevance",
            "score": relevance,
            "raw_score": raw_score,
            "explanation": "Average relevance of retrieved contexts to the query",
            "metadata": {"individual_scores": relevance_scores}
        })

    async def _evaluate_context_recall(self, ground_truth: str, contexts: list[str], api_name: str, model: Optional[str] = None) -> tuple:
        """Evaluate if contexts contain necessary information"""
        combined_context = "\n\n".join(contexts)

        prompt = f"""
        Evaluate if the provided contexts contain all the information needed to answer with the ground truth.

        Ground Truth Answer: {ground_truth}

        Contexts:
        {combined_context}

        Rate coverage on a scale of 1-5 where:
        1 = Missing most key information
        2 = Missing several important points
        3 = Contains about half the needed information
        4 = Contains most information with minor gaps
        5 = Contains all necessary information

        Provide only the numeric score.
        """

        try:
            score_str = await llm_circuit_breaker.call_with_breaker(
                api_name,
                _run_circuit_bounded_rag_analyze,
                api_name,  # First param
                combined_context,  # input_data
                prompt,    # custom_prompt_arg
                self.api_key,  # api_key (None to load from config)
                "You are an evaluation expert. Provide only numeric scores.",  # system_message
                0.1,       # temp
                model_override=model,
                app_config=self.app_config,
                credentials_resolved=self.credentials_resolved,
                provider_credentials=self.provider_credentials,
            )

            score = float(score_str.strip()) / 5.0

            return ("context_recall", {
                "name": "context_recall",
                "score": score,
                "raw_score": float(score_str.strip()),
                "explanation": "Coverage of necessary information in contexts"
            })

        except DaemonCapacityError:
            raise
        except _RAG_EVAL_NONCRITICAL_EXCEPTIONS as e:
            logger.error("Context recall failed error_type={}", type(e).__name__)
            # Raise exception instead of returning 0.0
            raise_detached_error(ValueError("Context recall evaluation failed"))

    def _generate_suggestions(self, metrics: dict[str, dict]) -> list[str]:
        """Generate improvement suggestions based on metrics"""
        suggestions = []

        # Check relevance
        if "relevance" in metrics and metrics["relevance"]["score"] < 0.7:
            suggestions.append("Consider improving query understanding or response generation to better address user queries")

        # Check faithfulness
        if "faithfulness" in metrics and metrics["faithfulness"]["score"] < 0.7:
            suggestions.append("Improve grounding of responses in retrieved contexts to reduce hallucinations")

        # Check context precision
        if "context_precision" in metrics and metrics["context_precision"]["score"] < 0.6:
            suggestions.append("Enhance retrieval system to fetch more relevant contexts")

        # Check context recall
        if "context_recall" in metrics and metrics["context_recall"]["score"] < 0.7:
            suggestions.append("Expand retrieval to capture more comprehensive information")

        # Check answer similarity
        if "answer_similarity" in metrics and metrics["answer_similarity"]["score"] < 0.6:
            suggestions.append("Fine-tune generation to produce more accurate responses")

        return suggestions

    def _normalize_score(self, score: float) -> float:
        """
        Normalize score from 1-5 scale to 0-1 scale.

        Args:
            score: Score on 1-5 scale

        Returns:
            Normalized score on 0-1 scale
        """
        # Clamp score to 1-5 range
        score = max(1, min(5, score))
        # Normalize to 0-1
        return (score - 1) / 4

    def _calculate_overall_score(self, metrics: dict[str, dict], weights: Optional[dict[str, float]] = None) -> float:
        """
        Calculate weighted overall score from individual metrics.

        Args:
            metrics: Dictionary of metric results with scores
            weights: Optional weights for each metric (defaults to equal weights)

        Returns:
            Weighted average score
        """
        if not metrics:
            return 0.0

        # Default to equal weights
        if weights is None:
            weights = dict.fromkeys(metrics.keys(), 1.0)

        total_weight = 0
        weighted_sum = 0

        for metric_name, metric_data in metrics.items():
            if "score" in metric_data and metric_name in weights:
                score = metric_data["score"]
                weight = weights[metric_name]
                weighted_sum += score * weight
                total_weight += weight

        if total_weight == 0:
            return 0.0

        result = weighted_sum / total_weight

        # Clamp to observed min/max to avoid tiny floating-point overshoots
        try:
            scores = [d.get("score", 0.0) for d in metrics.values() if isinstance(d, dict)]
            if scores:
                min_s, max_s = min(scores), max(scores)
                if result < min_s:
                    result = min_s
                elif result > max_s:
                    result = max_s
        except _RAG_EVAL_NONCRITICAL_EXCEPTIONS:
            pass

        return result

    def close(self):
        """Clean up resources."""
        # No resources to clean up with direct embeddings API
        pass
