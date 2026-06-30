"""
Post-generation verification and adaptive repair for RAG answers.

This module provides a lightweight interface to verify a generated answer
against retrieved documents, and optionally attempt a targeted repair pass
when evidence is insufficient.

Design goals:
- Pluggable: accept an injected claims runner for unit tests
- Minimal coupling: reuse existing ClaimsEngine when available
- Guarded by feature flags and strict time/attempt budgets
- Export central metrics for observability
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Optional, cast

from loguru import logger
from tldw_Server_API.app.core.testing import is_truthy

if TYPE_CHECKING:
    from .claims import ClaimsEngine as ClaimsEngineType
    from .database_retrievers import MultiDatabaseRetriever as MultiDatabaseRetrieverType
    from .generation import AnswerGenerator as AnswerGeneratorType
    from .types import DataSource, Document
    GenerateHypoFn = Callable[[str, Optional[str], Optional[str]], str]
    HydeEmbedFn = Callable[[str], Any]
    MultiStrategyExpansionFn = Callable[..., Any]
else:
    from dataclasses import dataclass as _dc

    @_dc
    class Document:  # pragma: no cover - fallback for isolated tests
        id: str
        content: str
        metadata: dict[str, Any]
        score: float = 0.0

    class DataSource:  # pragma: no cover - fallback for isolated tests
        MEDIA_DB = "media_db"

    ClaimsEngineType = Any
    MultiDatabaseRetrieverType = Any
    AnswerGeneratorType = Any
    GenerateHypoFn = Any
    HydeEmbedFn = Any
    MultiStrategyExpansionFn = Any

ClaimsEngine: type[ClaimsEngineType] | None = None
MultiDatabaseRetriever: type[MultiDatabaseRetrieverType] | None = None
AnswerGenerator: type[AnswerGeneratorType] | None = None
generate_hypothetical_answer: GenerateHypoFn | None = None
hyde_embed_text: HydeEmbedFn | None = None
multi_strategy_expansion: MultiStrategyExpansionFn | None = None

try:
    from . import claims as _claims_mod
    ClaimsEngine = cast(Optional[type[ClaimsEngineType]], getattr(_claims_mod, "ClaimsEngine", None))
except ImportError:
    ClaimsEngine = None

try:
    from . import database_retrievers as _db_mod
    MultiDatabaseRetriever = cast(Optional[type[MultiDatabaseRetrieverType]], getattr(_db_mod, "MultiDatabaseRetriever", None))
except ImportError:
    MultiDatabaseRetriever = None

try:
    from . import generation as _gen_mod
    AnswerGenerator = cast(Optional[type[AnswerGeneratorType]], getattr(_gen_mod, "AnswerGenerator", None))
except ImportError:
    AnswerGenerator = None

try:
    from . import hyde as _hyde_mod
    generate_hypothetical_answer = cast(Optional[GenerateHypoFn], getattr(_hyde_mod, "generate_hypothetical_answer", None))
    hyde_embed_text = cast(Optional[HydeEmbedFn], getattr(_hyde_mod, "embed_text", None))
except ImportError:
    generate_hypothetical_answer = None
    hyde_embed_text = None

try:
    from . import query_expansion as _qe_mod
    multi_strategy_expansion = cast(Optional[MultiStrategyExpansionFn], getattr(_qe_mod, "multi_strategy_expansion", None))
except ImportError:
    multi_strategy_expansion = None

try:
    # Central metrics registry (Prometheus/OTel)
    from tldw_Server_API.app.core.Metrics.metrics_manager import (
        increment_counter,
        observe_histogram,
    )
except Exception:  # noqa: BLE001 - metrics optional in test envs
    def increment_counter(metric_name: str, value: float = 1.0, labels: dict[str, str] | None = None) -> Any:
        return None
    def observe_histogram(metric_name: str, value: float, labels: dict[str, str] | None = None) -> Any:
        return None


@dataclass
class VerificationOutcome:
    unsupported_ratio: float
    total_claims: int
    unsupported_count: int
    fixed: bool
    reason: str = ""
    new_answer: str | None = None
    claims: list[dict[str, Any]] | None = None
    summary: dict[str, Any] | None = None


class PostGenerationVerifier:
    """Verify a generated answer and optionally perform an adaptive repair pass."""

    def __init__(
        self,
        claims_runner: Callable[..., Any] | None = None,
        max_retries: int = 1,
        unsupported_threshold: float = 0.15,
        max_claims: int = 20,
        time_budget_sec: float | None = None,
        use_advanced_rewrites: bool | None = None,
    ):
        self._claims_runner = claims_runner
        self._max_retries = max(0, int(max_retries or 0))
        try:
            self._threshold = float(unsupported_threshold)
        except (TypeError, ValueError):
            self._threshold = 0.15
        self._max_claims = max(1, int(max_claims or 1))
        self._time_budget = float(time_budget_sec) if time_budget_sec is not None else None
        # Toggle for advanced rewrites (HyDE + multi-strategy + diversity). Default enabled.
        if use_advanced_rewrites is None:
            try:
                env_val = os.getenv("RAG_ADAPTIVE_ADVANCED_REWRITES", "true").strip().lower()
                self._adv = is_truthy(env_val)
            except Exception:  # noqa: BLE001 - env parsing best-effort
                self._adv = True
        else:
            self._adv = bool(use_advanced_rewrites)

    async def verify_and_maybe_fix(
        self,
        query: str,
        answer: str | None,
        base_documents: list[Document],
        *,
        media_db_path: str | None = None,
        notes_db_path: str | None = None,
        character_db_path: str | None = None,
        user_id: str | None = None,
        generation_model: str | None = None,
        generation_provider: str | None = None,
        existing_claims: list[dict[str, Any]] | None = None,
        existing_summary: dict[str, Any] | None = None,
        search_mode: str = "hybrid",
        hybrid_alpha: float = 0.7,
        top_k: int = 10,
    ) -> VerificationOutcome:
        """Run post-generation verification and, if needed, a single repair attempt.

        Returns a VerificationOutcome with summary and optional new answer.
        """
        start_ts = time.time()
        outcome = VerificationOutcome(
            unsupported_ratio=0.0,
            total_claims=0,
            unsupported_count=0,
            fixed=False,
            reason="",
            new_answer=None,
            claims=None,
            summary=None,
        )

        if not answer or not isinstance(answer, str) or answer.strip() == "":
            observe_histogram("rag_postcheck_duration_seconds", time.time() - start_ts, labels={"outcome": "skipped"})
            outcome.reason = "empty_answer"
            return outcome

        # Run or reuse claims verification
        claims_payload: list[dict[str, Any]] | None = existing_claims
        summary_payload: dict[str, Any] | None = existing_summary
        try:
            if (claims_payload is None or summary_payload is None):
                if self._claims_runner is not None:
                    # Injectable runner (testing)
                    run = await _maybe_await(self._claims_runner(
                        query=query,
                        answer=answer,
                        documents=base_documents,
                        claims_max=self._max_claims,
                    ))
                    claims_payload = (run or {}).get("claims")
                    summary_payload = (run or {}).get("summary")
                elif ClaimsEngine is not None:
                    # Use default analyze function
                    import tldw_Server_API.app.core.LLM_Calls.Summarization_General_Lib as sgl
                    def _analyze(api_name: str, input_data: Any, custom_prompt_arg: str | None = None,
                                 api_key: str | None = None, system_message: str | None = None,
                                 temp: float | None = None, **kwargs):
                        return sgl.analyze(api_name, input_data, custom_prompt_arg, api_key, system_message, temp, **kwargs)

                    engine = ClaimsEngine(_analyze)

                    # Build a claim-level retrieval function
                    async def _retrieve_for_claim(c_text: str, top: int = 5):
                        try:
                            if MultiDatabaseRetriever is None:
                                return base_documents[:top]
                            db_paths = {}
                            if media_db_path:
                                db_paths["media_db"] = media_db_path
                            if notes_db_path:
                                db_paths["notes_db"] = notes_db_path
                            if character_db_path:
                                db_paths["character_cards_db"] = character_db_path
                            mdr = MultiDatabaseRetriever(
                                db_paths,
                                user_id=user_id or "0",
                            )
                            med = mdr.retrievers.get(DataSource.MEDIA_DB)
                            docs: list[Document] = []
                            if med is not None:
                                rh = getattr(med, 'retrieve_hybrid', None)
                                if rh is not None and asyncio.iscoroutinefunction(rh) and search_mode == "hybrid":
                                    docs.extend(await rh(query=c_text, alpha=hybrid_alpha))
                                else:
                                    docs.extend(await med.retrieve(query=c_text))
                            # Other sources could be added similarly
                            docs = sorted(docs, key=lambda d: getattr(d, 'score', 0.0), reverse=True)
                            return docs[:top]
                        except Exception:  # noqa: BLE001 - retrieval fallback should be safe
                            return base_documents[:top]

                    run = await engine.run(
                        answer=answer,
                        query=query,
                        documents=base_documents,
                        claim_extractor="auto",
                        claim_verifier="hybrid",
                        claims_top_k=5,
                        claims_conf_threshold=0.7,
                        claims_max=self._max_claims,
                        retrieve_fn=_retrieve_for_claim,
                        nli_model=os.getenv("RAG_NLI_MODEL") or os.getenv("RAG_NLI_MODEL_PATH"),
                        claims_concurrency=8,
                    )
                    claims_payload = (run or {}).get("claims")
                    summary_payload = (run or {}).get("summary")
        except Exception as e:  # noqa: BLE001 - claims verification best-effort
            logger.warning(f"Post-check claims verification failed: {_safe_exception_label(e)}")

        # Compute unsupported ratio
        unsupported = 0
        total = 0
        try:
            if isinstance(summary_payload, dict):
                supported = int(summary_payload.get("supported") or 0)
                refuted = int(summary_payload.get("refuted") or 0)
                nei = int(summary_payload.get("nei") or 0)
                total = max(0, supported + refuted + nei)
                unsupported = max(0, refuted + nei)
        except (TypeError, ValueError):
            pass
        ratio = (unsupported / total) if total else 0.0

        outcome.unsupported_ratio = ratio
        outcome.total_claims = total
        outcome.unsupported_count = unsupported
        outcome.claims = claims_payload
        outcome.summary = summary_payload

        # Record metrics for unsupported and total claims
        try:
            from tldw_Server_API.app.core.Claims_Extraction.monitoring import record_postcheck_metrics
            record_postcheck_metrics(total, unsupported)
        except Exception:  # noqa: BLE001 - metrics best-effort
            logger.debug("Post-generation verifier metrics recording failed")

        # Decide if we should attempt a repair
        if ratio <= self._threshold or self._max_retries <= 0:
            observe_histogram("rag_postcheck_duration_seconds", time.time() - start_ts, labels={"outcome": "ok"})
            outcome.reason = "threshold_not_exceeded" if ratio <= self._threshold else "no_retries"
            return outcome

        # Attempt a single adaptive retrieval + regeneration pass (bounded)
        retries = 0
        fixed = False
        new_answer: str | None = None
        while retries < self._max_retries:
            retries += 1
            increment_counter("rag_adaptive_retries_total", 1)

            if self._time_budget is not None and (time.time() - start_ts) > self._time_budget:
                outcome.reason = "time_budget_exhausted"
                break

            # Second-chance retrieval: use query rewrites (HyDE + multi-strategy) and apply diversity
            new_docs: list[Document] = base_documents[:]
            try:
                if MultiDatabaseRetriever is not None and media_db_path:
                    mdr = MultiDatabaseRetriever({"media_db": media_db_path}, user_id=user_id or "0")
                    med = mdr.retrievers.get(DataSource.MEDIA_DB)
                    if med is not None:
                        rh = getattr(med, 'retrieve_hybrid', None)
                        hybrid_supported = rh is not None and asyncio.iscoroutinefunction(rh)
                        rh_fn = rh if hybrid_supported else None
                        if not self._adv:
                            # Simple path: single-query retrieval only
                            if search_mode == "hybrid" and rh_fn is not None:
                                new_docs = await rh_fn(query=query, alpha=min(max(hybrid_alpha, 0.1), 0.9))
                            else:
                                new_docs = await med.retrieve(query=query)
                            new_docs = new_docs[: max(5, min(15, top_k))]
                        else:
                            # Advanced path: rewrites + HyDE + diversity
                            candidate_queries: list[str] = [query]
                            try:
                                if multi_strategy_expansion is not None:
                                    expand_fn = multi_strategy_expansion
                                    expanded = await expand_fn(query, strategies=["acronym", "synonym", "domain"])  # light expansion
                                    if isinstance(expanded, list):
                                        candidate_queries.extend([q for q in expanded if isinstance(q, str) and q.strip()])
                            except Exception:  # noqa: BLE001 - expansion best-effort
                                logger.debug("Adaptive fix query expansion failed; continuing")
                            # Optional HyDE vector for the base query
                            hyde_vector = None
                            try:
                                if generate_hypothetical_answer is not None and hyde_embed_text is not None:
                                    hypo = generate_hypothetical_answer(query, None, None)
                                    vec = await hyde_embed_text(hypo)
                                    if vec:
                                        hyde_vector = vec
                            except Exception:  # noqa: BLE001 - HyDE best-effort
                                hyde_vector = None

                            # Aggregate retrieval across queries
                            docs_union: dict[str, Document] = {}
                            for cq in list(dict.fromkeys(candidate_queries))[:4]:  # bound rewrites
                                try:
                                    cur_docs: list[Document]
                                    if search_mode == "hybrid" and rh_fn is not None:
                                        kwargs = {"query": cq, "alpha": min(max(hybrid_alpha, 0.1), 0.9)}
                                        if hyde_vector is not None and cq == query:
                                            kwargs["query_vector"] = hyde_vector
                                        cur_docs = await rh_fn(**kwargs)
                                    else:
                                        cur_docs = await med.retrieve(query=cq)
                                    for d in cur_docs or []:
                                        prev = docs_union.get(getattr(d, "id", ""))
                                        if prev is None or float(getattr(d, "score", 0.0)) > float(getattr(prev, "score", 0.0)):
                                            docs_union[getattr(d, "id", "")] = d
                                except Exception:  # noqa: BLE001 - per-query retrieval best-effort
                                    logger.debug("Adaptive per-query retrieval failed; continuing")

                            merged_docs = sorted(docs_union.values(), key=lambda x: getattr(x, "score", 0.0), reverse=True)
                            merged_docs = merged_docs[: max(5, min(30, top_k * 2))]
                            # Apply simple diversity filter to reduce near-duplicates
                            new_docs = _select_diverse(merged_docs, k=max(5, min(15, top_k)))
            except Exception as e:  # noqa: BLE001 - fallback to base docs
                logger.debug(f"Adaptive retrieval failed; using base docs. Reason: {_safe_exception_label(e)}")
                new_docs = base_documents[:]

            # Regenerate if possible
            try:
                if AnswerGenerator is not None:
                    gen_cls = AnswerGenerator
                    gen = gen_cls(
                        model=generation_model,
                        provider=generation_provider,
                    )
                    context = "\n\n".join([getattr(d, 'content', '') for d in new_docs[:5]])
                    maybe = await gen.generate(query=query, context=context, prompt_template=None, max_tokens=500)
                    new_answer = maybe.get("answer") if isinstance(maybe, dict) else str(maybe)
                else:
                    new_answer = None
            except Exception as e:  # noqa: BLE001 - regeneration best-effort
                logger.debug(f"Adaptive regeneration failed: {_safe_exception_label(e)}")
                new_answer = None

            # If we couldn't regenerate, stop
            if not new_answer:
                outcome.reason = "regen_failed"
                break

            # Re-check quickly with limited claims
            try:
                # Use same runner to compute a small verification sample
                if self._claims_runner is not None:
                    run2 = await _maybe_await(self._claims_runner(
                        query=query,
                        answer=new_answer,
                        documents=new_docs,
                        claims_max=max(5, min(10, self._max_claims)),
                    ))
                    sum2 = (run2 or {}).get("summary") or {}
                elif ClaimsEngine is not None:
                    import tldw_Server_API.app.core.LLM_Calls.Summarization_General_Lib as sgl
                    def _analyze2(api_name: str, input_data: Any, custom_prompt_arg: str | None = None,
                                   api_key: str | None = None, system_message: str | None = None,
                                   temp: float | None = None, **kwargs):
                        return sgl.analyze(api_name, input_data, custom_prompt_arg, api_key, system_message, temp, **kwargs)
                    eng2 = ClaimsEngine(_analyze2)
                    run2 = await eng2.run(
                        answer=new_answer,
                        query=query,
                        documents=new_docs,
                        claim_extractor="auto",
                        claim_verifier="hybrid",
                        claims_top_k=5,
                        claims_conf_threshold=0.7,
                        claims_max=max(5, min(10, self._max_claims)),
                        retrieve_fn=None,
                        nli_model=os.getenv("RAG_NLI_MODEL") or os.getenv("RAG_NLI_MODEL_PATH"),
                        claims_concurrency=4,
                    )
                    sum2 = (run2 or {}).get("summary") or {}
                else:
                    sum2 = {}
            except Exception as e:  # noqa: BLE001 - recheck best-effort
                logger.debug(f"Adaptive recheck failed: {_safe_exception_label(e)}")
                sum2 = {}

            try:
                s2_supported = int(sum2.get("supported") or 0)
                s2_refuted = int(sum2.get("refuted") or 0)
                s2_nei = int(sum2.get("nei") or 0)
                s2_total = max(0, s2_supported + s2_refuted + s2_nei)
                s2_ratio = ((s2_refuted + s2_nei) / s2_total) if s2_total else 0.0
            except (TypeError, ValueError):
                s2_ratio = 0.0

            if s2_ratio <= self._threshold:
                fixed = True
                break
            else:
                # Try next retry if allowed
                continue

        outcome.fixed = fixed
        if fixed and new_answer:
            outcome.new_answer = new_answer
            try:
                increment_counter("rag_adaptive_fix_success_total", 1)
            except Exception:  # noqa: BLE001 - metrics best-effort
                logger.debug("Post-generation adaptive-fix success metric failed")
            observe_histogram("rag_postcheck_duration_seconds", time.time() - start_ts, labels={"outcome": "fixed"})
        else:
            observe_histogram("rag_postcheck_duration_seconds", time.time() - start_ts, labels={"outcome": "unfixed"})

        return outcome


async def _maybe_await(value):  # pragma: no cover - trivial helper
    if asyncio.iscoroutine(value):
        return await value
    return value


def _safe_exception_label(exc: BaseException) -> str:
    return type(exc).__name__


def _jaccard(a: str, b: str) -> float:
    try:
        sa = set((a or "").lower().split())
        sb = set((b or "").lower().split())
        if not sa or not sb:
            return 0.0
        inter = len(sa & sb)
        union = len(sa | sb)
        return float(inter) / float(union) if union else 0.0
    except Exception:  # noqa: BLE001 - best-effort similarity
        return 0.0


def _select_diverse(docs: list[Document], k: int = 10, sim_threshold: float = 0.6) -> list[Document]:
    selected: list[Document] = []
    for d in docs:
        if len(selected) >= k:
            break
        # Keep if sufficiently different from all selected
        if all(_jaccard(getattr(d, 'content', ''), getattr(s, 'content', '')) < sim_threshold for s in selected):
            selected.append(d)
    # If selection too small, pad with top docs
    if len(selected) < min(k, len(docs)):
        seen = {getattr(x, 'id', '') for x in selected}
        for d in docs:
            if len(selected) >= k:
                break
            if getattr(d, 'id', '') not in seen:
                selected.append(d)
                seen.add(getattr(d, 'id', ''))
    return selected
