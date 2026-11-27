"""claims code moved from unified_pipeline.py (TASK-13950.24)."""

import asyncio
import sqlite3
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, Optional

from tldw_Server_API.app.core.RAG.rag_service import unified_pipeline as up

from .run_config import RagRunConfig

if TYPE_CHECKING:
    from .run_state import PipelineState


def prepare_claims_job(rcfg: RagRunConfig, state: "PipelineState") -> None:
    """Build the claims engine and its closures."""
    state.engine = state.claims_engine_type(state._analyze)

    async def _mark_claims_used() -> None:
        if (
            state.claims_handle is not None
            and state.claims_state["used"]
            and not state.claims_state["marked"]
        ):
            await rcfg.runtime.credential_runtime.mark_used(state.claims_handle)
            state.claims_state["marked"] = True
    state._mark_claims_used = _mark_claims_used

    async def _run_claims_engine(**run_kwargs: Any) -> dict[str, Any]:
        async def _run_and_mark() -> dict[str, Any]:
            try:
                claims_output: dict[str, Any] = await state.engine.run(**run_kwargs)
                return claims_output
            finally:
                await state._mark_claims_used()

        operation = _run_and_mark()
        if state.claims_handle is not None:
            return await up.await_owned_worker(operation)
        return await operation
    state._run_claims_engine = _run_claims_engine

    # Default NLI model from environment if not provided
    if not state.nli_model:
        import os as _os
        state.nli_model = _os.environ.get("RAG_NLI_MODEL") or _os.environ.get("RAG_NLI_MODEL_PATH")
    state.job_context = None
    if up.ClaimsJobContext is not None:
        user_id_val = None
        try:
            user_id_val = int(rcfg.scope.user_id) if rcfg.scope.user_id is not None else None
        except (TypeError, ValueError):
            user_id_val = None
        state.job_context = up.ClaimsJobContext(
            user_id=user_id_val,
            request_id=rcfg.observability.trace_id,
            endpoint="rag_claims",
        )
    state.job_budget = None
    if up.resolve_claims_job_budget is not None:
        settings_obj = None
        try:
            from tldw_Server_API.app.core.config import settings as _settings

            if isinstance(_settings, dict):
                settings_obj = _settings
            elif hasattr(_settings, "dict"):
                settings_obj = _settings.dict()
        except (ImportError, AttributeError, TypeError):
            settings_obj = None
        budget_usd = rcfg.verification.claims_budget_usd
        budget_tokens = rcfg.verification.claims_budget_tokens
        budget_strict = rcfg.verification.claims_budget_strict
        try:
            budget_usd = float(budget_usd) if budget_usd is not None else None
        except (TypeError, ValueError):
            budget_usd = None
        try:
            budget_tokens = int(budget_tokens) if budget_tokens is not None else None
        except (TypeError, ValueError):
            budget_tokens = None
        if isinstance(budget_strict, str):
            budget_strict = up._shared_is_truthy(budget_strict)
        state.job_budget = up.resolve_claims_job_budget(
            settings=settings_obj,
            max_cost_usd=budget_usd,
            max_tokens=budget_tokens,
            strict=budget_strict if isinstance(budget_strict, bool) else None,
        )

    # Build a per-claim retrieval that uses MultiDatabaseRetriever and hybrid search when available
    async def _retrieve_for_claim(c_text: str, top_k: int = 5):
        if state.latched_optional_provider_failure("per_claim") is not None:
            return state.result.documents[:top_k] if state.result.documents else []
        try:
            if up.MultiDatabaseRetriever and up.RetrievalConfig:
                db_paths = state.build_pipeline_db_paths()
                # Initialize multi retriever scoped to user's databases
                mdr = state.build_multi_retriever(db_paths)
                ds = list(state.resolved_data_sources)

                # For media_db, attempt hybrid; for others, simple retrieve
                docs: list[Any] = []
                # Media hybrid
                med = mdr.retrievers.get(up.DataSource.MEDIA_DB)
                if med is not None and up._scope_includes_data_source(
                    state.resolved_data_sources, up.DataSource.MEDIA_DB
                ):
                    rh = getattr(med, "retrieve_hybrid", None)
                    if rh is not None and asyncio.iscoroutinefunction(rh) and rcfg.retrieval.search_mode == "hybrid":
                        media_docs = await rh(query=c_text, alpha=state.hybrid_alpha)
                    else:
                        media_docs = await med.retrieve(query=c_text)
                    docs.extend(media_docs)
                # Other sources
                for src in ds:
                    if src == up.DataSource.MEDIA_DB:
                        continue
                    retr = mdr.retrievers.get(src)
                    if retr is not None:
                        try:
                            more = await retr.retrieve(query=c_text)
                            docs.extend(more)
                        except up._RAG_PROVIDER_FAILURES:
                            raise
                        except (
                            AttributeError,
                            ConnectionError,
                            OSError,
                            RuntimeError,
                            TypeError,
                            ValueError,
                            asyncio.TimeoutError,
                        ):
                            pass
                # Sort and cap
                docs = sorted(docs, key=lambda d: getattr(d, "score", 0.0), reverse=True)
                return docs[:top_k]
        except up._RAG_PROVIDER_FAILURES as exc:
            if rcfg.runtime.credential_runtime is None:
                raise
            state.latch_optional_provider_failure("per_claim", exc)
            return state.result.documents[:top_k] if state.result.documents else []
        except (
            AttributeError,
            ConnectionError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
            asyncio.TimeoutError,
        ) as e:
            up.logger.debug(f"Per-claim retrieval fallback to base docs due to error: {e}")
        return state.result.documents[:top_k] if state.result.documents else []
    state._retrieve_for_claim = _retrieve_for_claim


async def execute_claims(rcfg: RagRunConfig, state: "PipelineState") -> None:
    """Extract, verify and report claims."""
    # Prefer pre-extracted claims if available for current documents
    claims_out = None  # noqa: F841 - dead store moved from unified_pipeline.py
    try:
        pre_claims: list[str] = []
        if state.media_db_path and (state.result.documents or []):
            from tldw_Server_API.app.core.config import settings as _settings

            with up.managed_media_database(
                client_id=str(_settings.get("SERVER_CLIENT_ID", "SERVER_API_V1")),
                db_path=state.media_db_path,
                initialize=False,
                suppress_close_exceptions=(
                    AttributeError,
                    RuntimeError,
                    TypeError,
                    ValueError,
                    sqlite3.Error,
                ),
            ) as db:
                # Collect media IDs present in documents
                media_ids: list[int] = []
                for d in state.result.documents:
                    try:
                        mid = d.metadata.get("media_id") if isinstance(d.metadata, dict) else None
                        if mid is not None:
                            media_ids.append(int(mid))
                    except (TypeError, ValueError):
                        continue
                media_ids = list(dict.fromkeys(media_ids))[:5]
                if media_ids:
                    # Fetch a small number of claims per media
                    for mid in media_ids:
                        rows = db.execute_query(
                            "SELECT claim_text FROM Claims WHERE media_id = ? AND deleted = 0 LIMIT ?",
                            (int(mid), int(rcfg.verification.claims_max)),
                        ).fetchall()
                        pre_claims.extend([r[0] for r in rows])
        if pre_claims:
            # Verify these claims directly, skipping extraction
            from tldw_Server_API.app.core.Claims_Extraction.claims_engine import Claim as _Claim

            verifications = []
            try:
                for i, ctext in enumerate(pre_claims[:rcfg.verification.claims_max]):
                    cv = await state.engine.verifier.verify(
                        claim=_Claim(id=f"pc{i+1}", text=ctext),
                        query=state.query,
                        base_documents=state.result.documents or [],
                        retrieve_fn=state._retrieve_for_claim,
                        top_k=rcfg.verification.claims_top_k,
                        conf_threshold=rcfg.verification.claims_conf_threshold,
                        mode=(rcfg.verification.claim_verifier or "hybrid").strip().lower(),
                        budget=state.job_budget,
                        job_context=state.job_context,
                        doc_only_mode=rcfg.verification.doc_only_verification,
                        numeric_precision_mode=rcfg.verification.numeric_precision_mode,
                    )
                    verifications.append(cv)
            finally:
                await state._mark_claims_used()
            supported = sum(1 for v in verifications if v.label == "supported")
            refuted = sum(1 for v in verifications if v.label == "refuted")
            nei = sum(1 for v in verifications if v.label == "nei")
            total = max(1, len(verifications))
            precision = supported / total
            coverage = (supported + refuted) / total
            # Enhanced claim output with new fields
            state.claims_payload = [
                {
                    "id": v.claim.id,
                    "text": v.claim.text,
                    "span": list(v.claim.span) if v.claim.span else None,
                    "claim_type": (
                        v.claim.claim_type.value if hasattr(v.claim, "claim_type") else "general"
                    ),
                    "status": v.status.value if hasattr(v, "status") else v.label,
                    "label": v.label,
                    "confidence": v.confidence,
                    "match_level": (
                        v.match_level.value if hasattr(v, "match_level") else "interpretation"
                    ),
                    "source_authority": (
                        v.source_authority.value if hasattr(v, "source_authority") else 1
                    ),
                    "requires_external_knowledge": getattr(v, "requires_external_knowledge", False),
                    "evidence": [
                        {
                            "doc_id": e.doc_id,
                            "snippet": e.snippet,
                            "score": e.score,
                            "authority": e.authority.value if hasattr(e, "authority") else 1,
                        }
                        for e in v.evidence
                    ],
                    "citations": v.citations,
                    "rationale": v.rationale,
                }
                for v in verifications
            ]
            state.factuality_payload = {
                "supported": supported,
                "refuted": refuted,
                "nei": nei,
                "precision": precision,
                "coverage": coverage,
            }
        else:
            # Fall back to on-the-fly extraction from the generated answer
            claims_run = await state._run_claims_engine(
                answer=state.result.generated_answer,
                query=state.query,
                documents=state.result.documents or [],
                claim_extractor=rcfg.verification.claim_extractor,
                claim_verifier=rcfg.verification.claim_verifier,
                claims_top_k=rcfg.verification.claims_top_k,
                claims_conf_threshold=rcfg.verification.claims_conf_threshold,
                claims_max=rcfg.verification.claims_max,
                retrieve_fn=state._retrieve_for_claim,
                nli_model=state.nli_model,
                claims_concurrency=rcfg.verification.claims_concurrency,
                job_budget=state.job_budget,
                job_context=state.job_context,
            )
            state.claims_payload = claims_run.get("claims")
            state.factuality_payload = claims_run.get("summary")
            verifications = claims_run.get("verifications", [])
    except (
        AttributeError,
        ConnectionError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        asyncio.TimeoutError,
    ) as _eclaims:
        if rcfg.runtime.credential_runtime is not None:
            up.logger.debug(
                "Pre-extracted claims path failed; using on-the-fly fallback"
            )
        else:
            up.logger.debug(f"Pre-extracted claims path failed: {_eclaims}")
        claims_run = await state._run_claims_engine(
            answer=state.result.generated_answer,
            query=state.query,
            documents=state.result.documents or [],
            claim_extractor=rcfg.verification.claim_extractor,
            claim_verifier=rcfg.verification.claim_verifier,
            claims_top_k=rcfg.verification.claims_top_k,
            claims_conf_threshold=rcfg.verification.claims_conf_threshold,
            claims_max=rcfg.verification.claims_max,
            retrieve_fn=state._retrieve_for_claim,
            nli_model=state.nli_model,
            claims_concurrency=rcfg.verification.claims_concurrency,
            job_budget=state.job_budget,
            job_context=state.job_context,
        )
        state.claims_payload = claims_run.get("claims")
        state.factuality_payload = claims_run.get("summary")
        verifications = claims_run.get("verifications", [])
    # Also store in metadata for debugging/analytics
    await state._mark_claims_used()
    state.result.metadata["claims"] = state.claims_payload
    state.result.metadata["factuality"] = state.factuality_payload

    # Generate verification report if requested
    if rcfg.verification.generate_verification_report and verifications:
        try:
            from tldw_Server_API.app.core.Claims_Extraction.verification_report import (
                generate_verification_report as gen_report,
            )

            # Use the full VerificationReport class with raw verification objects
            report = gen_report(
                verifications=verifications,
                query=state.query,
                answer_text=state.result.generated_answer,
                metadata={"source": "unified_pipeline"},
            )
            state.result.metadata["verification_report"] = report.to_dict()
        except (ImportError, RuntimeError, TypeError, ValueError) as _ereport:
            up.logger.debug(f"Verification report generation failed: {_ereport}")


async def run_claims(rcfg: RagRunConfig, state: "PipelineState") -> None:
    """Claims and factuality stage."""
    if rcfg.verification.enable_claims and state.result.generated_answer and not state.skip_claims_due_to_groundedness:
        try:
            # Import shared analyze function for LLM calls
            import tldw_Server_API.app.core.LLM_Calls.Summarization_General_Lib as sgl

            state.claims_handle = None
            claims_provider = None
            state.claims_state = {"used": False, "marked": False}
            if rcfg.runtime.credential_runtime is not None:
                from tldw_Server_API.app.core.Claims_Extraction.claims_engine import (
                    _resolve_claims_llm_config,
                )

                claims_provider, claims_model, _ = _resolve_claims_llm_config()
                state.claims_handle = await rcfg.runtime.credential_runtime.resolve(
                    claims_provider,
                    model=claims_model,
                )

            def _analyze(api_name: str, input_data: Any, custom_prompt_arg: Optional[str] = None,
                         api_key: Optional[str] = None, system_message: Optional[str] = None,
                         temp: Optional[float] = None, **kwargs):
                kwargs["llm_surface"] = "rag.auxiliary"
                if state.claims_handle is None:
                    return sgl.analyze(
                        api_name,
                        input_data,
                        custom_prompt_arg,
                        api_key,
                        system_message,
                        temp,
                        **kwargs,
                    )
                for key in (
                    "app_config",
                    "credentials_resolved",
                    "provider_credentials",
                    "_provider_call_credentials",
                    "raise_on_error",
                ):
                    kwargs.pop(key, None)
                response = sgl.analyze(
                    claims_provider or state.claims_handle.provider,
                    input_data,
                    custom_prompt_arg,
                    state.claims_handle.api_key,
                    system_message,
                    temp,
                    app_config=state.claims_handle.app_config,
                    credentials_resolved=True,
                    provider_credentials=state.claims_handle,
                    raise_on_error=True,
                    **kwargs,
                )
                if isinstance(response, Iterator):
                    from ..generation import (
                        _classify_stream_content,
                        _extract_stream_text,
                    )

                    chunks = []
                    for chunk in response:
                        chunks.append(str(chunk))
                        has_content, has_error = _classify_stream_content(
                            _extract_stream_text(chunk)
                        )
                        if has_content:
                            state.claims_state["used"] = True
                        if has_error:
                            raise up.SummaryProviderError(
                                code="provider_failure",
                                provider=claims_provider or state.claims_handle.provider,
                            )
                    response = "".join(chunks)
                if isinstance(response, str) and response.startswith("Error:"):
                    raise up.SummaryProviderError(
                        code="provider_failure",
                        provider=claims_provider or state.claims_handle.provider,
                    )
                state.claims_state["used"] = True
                return response
            state._analyze = _analyze

            state.claims_engine_type = up._resolve_claims_engine()
            if state.claims_engine_type is not None:
                prepare_claims_job(rcfg, state)

                await execute_claims(rcfg, state)
        except (up.ByokResolutionError, up.SummaryProviderError) as exc:
            state.result.metadata["claims"] = {
                "failure_code": (
                    exc.code
                    if isinstance(exc, up.ByokResolutionError)
                    else "provider_unavailable"
                ),
                "verification_available": False,
            }
            up.logger.warning("Claims analysis provider unavailable")
        except (
            AttributeError,
            ConnectionError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
            asyncio.TimeoutError,
        ) as e:
            if rcfg.runtime.credential_runtime is not None:
                state.result.metadata["claims"] = {
                    "failure_code": "provider_unavailable",
                    "verification_available": False,
                }
                up.logger.warning("Claims analysis provider unavailable")
            else:
                state.result.errors.append(f"Claims analysis failed: {str(e)}")
                up.logger.error(f"Claims analysis error: {e}")
