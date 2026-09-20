import inspect
import re
import sys
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
from tldw_Server_API.app.core.RAG import (
    DataSource,
    apply_profile_to_kwargs,
    get_multi_tenant_safe_kwargs,
    get_profile,
    get_profile_kwargs,
    list_profiles,
    unified_rag_pipeline,
)
from tldw_Server_API.app.core.RAG.exceptions import RAGConfigurationError
from tldw_Server_API.app.core.RAG.rag_service import advanced_reranking, database_retrievers
from tldw_Server_API.app.core.RAG.rag_service import profiles as profiles_module
from tldw_Server_API.app.core.RAG.rag_service import unified_pipeline as unified_pipeline_module
from tldw_Server_API.app.core.RAG.rag_service.result_model import RAGResult
from tldw_Server_API.app.core.RAG.rag_service.types import Document

_SLIDES_PROFILE = "slides_source_retrieval_v1"


def _closed_slides_retrieval():
    return unified_pipeline_module.retrieve_slides_source_documents_v1


def _source_document(
    source: DataSource,
    document_id: str,
    score: float,
) -> Document:
    return Document(
        id=document_id,
        content=f"{document_id} body",
        metadata={
            "title": document_id,
            "_standalone_source_projection_truncated": False,
        },
        source=source,
        score=score,
    )


def _patch_closed_source_retrievers(monkeypatch):
    documents = {
        "media": [
            _source_document(DataSource.MEDIA_DB, "media-low", 0.4),
            _source_document(DataSource.MEDIA_DB, "media-tie", 0.8),
        ],
        "notes": [_source_document(DataSource.NOTES, "notes-high", 0.9)],
        "chats": [_source_document(DataSource.CHAT_HISTORY, "chat-tie", 0.8)],
    }
    retrievers = {}
    constructors = {}

    for name, class_name in (
        ("media", "MediaDBRetriever"),
        ("notes", "NotesDBRetriever"),
        ("chats", "ChatHistoryRetriever"),
    ):
        retriever = MagicMock(name=f"{name}_retriever")
        retriever.retrieve = AsyncMock(side_effect=AssertionError(f"generic {name} retrieve was called"))
        candidates = [
            Document(
                id=document.id,
                content="",
                metadata={
                    **document.metadata,
                    "_standalone_source_full_chars": len(document.content),
                    "_standalone_source_projection_key": document.id,
                    "_test_full_content": document.content,
                },
                source=document.source,
                score=document.score,
            )
            for document in documents[name]
        ]

        def project(*, projections, owner_user_id):
            assert owner_user_id == "42"
            projected = []
            for candidate, char_cap in projections:
                full_content = candidate.metadata["_test_full_content"]
                metadata = {key: value for key, value in candidate.metadata.items() if not key.startswith("_test_")}
                if len(full_content) > char_cap:
                    metadata["_standalone_source_projection_truncated"] = True
                projected.append(
                    Document(
                        id=candidate.id,
                        content=full_content[:char_cap],
                        metadata=metadata,
                        source=candidate.source,
                        score=candidate.score,
                    )
                )
            return projected

        retriever.retrieve_slides_source_documents_v1 = AsyncMock(
            side_effect=AssertionError(f"one-phase {name} retrieve was called")
        )
        retriever.retrieve_slides_source_candidates_v1 = AsyncMock(return_value=candidates)
        retriever.project_slides_source_documents_v1 = AsyncMock(side_effect=project)
        retrievers[name] = retriever
        constructor = MagicMock(return_value=retriever)
        constructors[name] = constructor
        monkeypatch.setattr(
            unified_pipeline_module,
            class_name,
            constructor,
            raising=False,
        )

    generic_phase = AsyncMock(side_effect=AssertionError("generic retrieval phase was called"))
    monkeypatch.setattr(
        unified_pipeline_module,
        "execute_retrieval_phase",
        generic_phase,
    )
    return retrievers, constructors, generic_phase


class _CapturedCursor:
    def __init__(self, rows):
        self._rows = rows
        self._index = 0
        self.description = [(column_name,) for column_name in rows[0]] if rows else []

    def fetchone(self):
        if self._index >= len(self._rows):
            return None
        row = self._rows[self._index]
        self._index += 1
        return row


class _SqlCaptureAdapter:
    backend_type = BackendType.POSTGRESQL

    def __init__(self, rows=(), *, row_batches=None, client_id="owner-42"):
        self.client_id = client_id
        self.rows = list(rows)
        self.row_batches = [list(batch) for batch in (row_batches or [])]
        self.calls = []
        self.options = []

    def execute_query(self, query, params=(), **kwargs):
        self.calls.append((query, tuple(params)))
        self.options.append(kwargs)
        rows = self.row_batches.pop(0) if self.row_batches else self.rows
        return _CapturedCursor(rows)


@pytest.mark.unit
class TestRAGProfiles:
    def test_profiles_are_registered(self):
        profiles = list_profiles()
        assert "production" in profiles
        assert "research" in profiles
        assert "cheap" in profiles
        assert "fast" in profiles
        assert "balanced" in profiles
        assert "accuracy" in profiles

    def test_switchable_profile_defaults_match_design_targets(self):
        fast = get_profile_kwargs("fast")
        balanced = get_profile_kwargs("balanced")
        accuracy = get_profile_kwargs("accuracy")

        assert fast["max_generation_tokens"] == 440
        assert fast["generation_prompt"] == "instruction_tuned"
        assert fast["enable_query_decomposition"] is False

        assert balanced["max_generation_tokens"] == 1000
        assert balanced["generation_prompt"] == "multi_hop_compact"
        assert balanced["enable_query_decomposition"] is True
        assert balanced["reranking_strategy"] == "hybrid"

        assert accuracy["max_generation_tokens"] == 2200
        assert accuracy["generation_prompt"] == "expert_synthesis"
        assert accuracy["enable_query_decomposition"] is True
        assert accuracy["reranking_strategy"] == "two_tier"

    def test_get_profile_kwargs_merges_overrides(self):

        base = get_profile_kwargs("cheap")
        assert base["search_mode"] in {"fts", "hybrid"}
        # Override a couple of knobs and ensure they take precedence
        overrides = {"top_k": 3, "enable_generation": False}
        merged = get_profile_kwargs("cheap", overrides=overrides)
        assert merged["top_k"] == 3
        assert merged["enable_generation"] is False

    def test_apply_profile_to_existing_kwargs(self):

        existing = {"search_mode": "vector", "top_k": 5}
        merged = apply_profile_to_kwargs("production", existing)
        # Existing keys should win over profile defaults
        assert merged["search_mode"] == "vector"
        assert merged["top_k"] == 5
        # And some known production default should still be present
        assert merged["enable_security_filter"] is True

    def test_multi_tenant_safe_kwargs_enforces_namespace_and_observability(self):

        ns = "tenant-xyz"
        kwargs = get_multi_tenant_safe_kwargs(ns)
        assert kwargs["index_namespace"] == ns
        assert kwargs["enable_observability"] is False
        # Monitoring should remain on for metrics
        assert kwargs["enable_monitoring"] is True

    def test_multi_tenant_safe_kwargs_requires_namespace(self):

        for bad in ("", "   ", None):
            with pytest.raises(ValueError):
                # type: ignore[arg-type]
                get_multi_tenant_safe_kwargs(bad)  # noqa: PT011

    @pytest.mark.asyncio
    async def test_profile_kwargs_drive_unified_pipeline_retrieval_config(self):
        """Exercise a profile through unified_rag_pipeline and validate mapped knobs."""
        with patch(
            "tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.MultiDatabaseRetriever"
        ) as mock_retriever:
            retriever_instance = MagicMock()
            retriever_instance.retrieve = AsyncMock(
                return_value=[
                    Document(
                        id="doc-1",
                        content="RAG content",
                        metadata={},
                        source=DataSource.MEDIA_DB,
                        score=0.9,
                    )
                ]
            )
            mock_retriever.return_value = retriever_instance

            with patch("tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.AnswerGenerator") as mock_generator:
                generator_instance = MagicMock()
                generator_instance.generate = AsyncMock(return_value={"answer": "Profile answer"})
                mock_generator.return_value = generator_instance

                kwargs = get_profile_kwargs(
                    "cheap",
                    overrides={
                        "enable_cache": False,
                        "enable_reranking": False,
                        "enable_security_filter": False,
                    },
                )
                result = await unified_rag_pipeline(query="What is RAG?", **kwargs)

                assert result.generated_answer == "Profile answer"
                assert retriever_instance.retrieve.await_count >= 1
                retrieve_kwargs = retriever_instance.retrieve.await_args.kwargs
                config = retrieve_kwargs["config"]
                assert config.max_results == kwargs["top_k"]
                assert config.use_fts is True
                assert config.use_vector is False


@pytest.mark.unit
class TestSlidesSourceRetrievalProfile:
    def test_profile_is_registered_with_closed_retrieval_only_defaults(self):
        defaults = get_profile(_SLIDES_PROFILE).defaults

        assert defaults["sources"] == ("media_db", "notes", "chats")
        assert defaults["search_mode"] == "fts"
        assert defaults["fts_level"] == "chunk"
        assert defaults["enable_reranking"] is False
        assert defaults["reranking_strategy"] == "none"
        assert defaults["reranking_model"] is None
        assert defaults["adaptive_max_retries"] == 0
        assert defaults["generation_provider"] is None
        assert defaults["generation_model"] is None
        assert defaults["generation_prompt"] is None
        assert defaults["search_depth_mode"] is None
        assert defaults["chat_history"] is None
        assert defaults["discussion_platforms"] is None
        assert defaults["rag_profile"] is None

        disabled_stages = {
            "enable_text_late_chunking",
            "adaptive_hybrid_weights",
            "enable_intent_routing",
            "auto_temporal_filters",
            "expand_query",
            "spell_check",
            "enable_prf",
            "enable_hyde",
            "enable_gap_analysis",
            "enable_cache",
            "adaptive_cache",
            "enable_table_processing",
            "enable_vlm_late_chunking",
            "enable_enhanced_chunking",
            "enable_parent_expansion",
            "include_sibling_chunks",
            "include_parent_document",
            "enable_multi_vector_passages",
            "enable_precomputed_spans",
            "enable_numeric_table_boost",
            "enable_learned_fusion",
            "enable_citations",
            "enable_chunk_citations",
            "enable_generation",
            "strict_extractive",
            "enable_pre_retrieval_clarification",
            "enable_abstention",
            "enable_multi_turn_synthesis",
            "enable_post_verification",
            "adaptive_rerun_on_low_confidence",
            "adaptive_rerun_include_generation",
            "enable_query_decomposition",
            "enable_graph_retrieval",
            "collect_feedback",
            "apply_feedback_boost",
            "enable_monitoring",
            "enable_observability",
            "enable_performance_analysis",
            "enable_streaming",
            "highlight_results",
            "track_cost",
            "debug_mode",
            "include_rerank_debug_documents",
            "enable_injection_filter",
            "enable_content_policy_filter",
            "enable_html_sanitizer",
            "require_hard_citations",
            "enable_numeric_fidelity",
            "enable_claims",
            "doc_only_verification",
            "generate_verification_report",
            "enable_dynamic_granularity",
            "enable_evidence_accumulation",
            "enable_evidence_chains",
            "enable_document_grading",
            "enable_query_rewriting_loop",
            "enable_web_fallback",
            "enable_knowledge_strips",
            "enable_fast_hallucination_check",
            "enable_utility_grading",
            "enable_batch",
            "enable_resilience",
            "enable_date_filter",
            "fallback_on_error",
            "enable_faithfulness_eval",
            "enable_query_classification",
            "enable_query_reformulation",
            "enable_research_loop",
            "enable_discussion_search",
            "search_url_scraping",
            "enable_research_progress",
            "enable_suggestions",
            "enable_structured_response",
            "enable_image_search",
            "enable_video_search",
        }
        assert all(defaults[name] is False for name in disabled_stages)

    def test_profile_defaults_cannot_be_mutated(self):
        defaults = get_profile(_SLIDES_PROFILE).defaults

        with pytest.raises(TypeError):
            defaults["enable_generation"] = True  # type: ignore[index]

        assert get_profile(_SLIDES_PROFILE).defaults["enable_generation"] is False

    @pytest.mark.parametrize(
        ("merge", "overrides"),
        (
            (get_profile_kwargs, {"enable_generation": True}),
            (apply_profile_to_kwargs, {"enable_web_fallback": True}),
        ),
    )
    def test_profile_rejects_every_generic_override_path(self, merge, overrides):
        assert get_profile(_SLIDES_PROFILE).name == _SLIDES_PROFILE
        with pytest.raises(ValueError):
            merge(_SLIDES_PROFILE, overrides)

    def test_profile_exposes_only_closed_local_reranking_strategies(self):
        assert frozenset({"none", "flashrank", "cross_encoder"}) == profiles_module.SLIDES_SOURCE_RERANKING_STRATEGIES


@pytest.mark.unit
class TestSlidesSourceRetrievalEntryPoint:
    def test_fts_only_retriever_does_not_initialize_vector_store(self, monkeypatch):
        vector_store_factory = MagicMock()
        monkeypatch.setattr(
            database_retrievers,
            "create_from_settings_for_user",
            vector_store_factory,
        )

        database_retrievers.MediaDBRetriever(
            None,
            config=database_retrievers.RetrievalConfig(
                max_results=3,
                use_fts=True,
                use_vector=False,
                fts_level="chunk",
            ),
            user_id="42",
            media_db=object(),
        )

        vector_store_factory.assert_not_called()

    def test_entry_point_has_no_generic_kwargs_or_profile_injection_seam(self):
        signature = inspect.signature(_closed_slides_retrieval())

        assert all(parameter.kind is not inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values())
        assert {
            "resolved_request",
            "retrieval_plan",
            "rag_profile",
            "profile",
            "overrides",
            "generation_provider",
            "generation_model",
            "reranking_model",
        }.isdisjoint(signature.parameters)
        for name in (
            "query",
            "owner_user_id",
            "top_k",
            "max_source_chars",
            "media_db",
            "chacha_db",
        ):
            assert signature.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
        assert signature.parameters["reranking_strategy"].default == "none"

    @pytest.mark.asyncio
    async def test_entry_point_calls_only_three_dedicated_owner_bounded_retrievers(
        self,
        monkeypatch,
    ):
        media_db = object()
        chacha_db = object()
        retrievers, constructors, generic_phase = _patch_closed_source_retrievers(monkeypatch)

        result = await _closed_slides_retrieval()(
            query="local evidence",
            owner_user_id="42",
            top_k=3,
            max_source_chars=1234,
            media_db=media_db,
            chacha_db=chacha_db,
            reranking_strategy="none",
        )

        for retriever in retrievers.values():
            retriever.retrieve_slides_source_candidates_v1.assert_awaited_once_with(
                query="local evidence",
                owner_user_id="42",
                top_k=3,
            )
            retriever.retrieve_slides_source_documents_v1.assert_not_awaited()
            retriever.retrieve.assert_not_awaited()
        generic_phase.assert_not_awaited()

        assert constructors["media"].call_args.kwargs["media_db"] is media_db
        assert constructors["notes"].call_args.kwargs["chacha_db"] is chacha_db
        assert constructors["chats"].call_args.kwargs["chacha_db"] is chacha_db
        assert [document.id for document in result.documents] == [
            "media-tie",
            "notes-high",
            "chat-tie",
        ]
        assert result.generated_answer is None

    @pytest.mark.asyncio
    async def test_entry_point_merge_is_deterministic_and_never_generates(
        self,
        monkeypatch,
    ):
        retrievers, _, _ = _patch_closed_source_retrievers(monkeypatch)

        results = []
        for _ in range(2):
            result = await _closed_slides_retrieval()(
                query="local evidence",
                owner_user_id="42",
                top_k=3,
                media_db=object(),
                chacha_db=object(),
                reranking_strategy="none",
            )
            results.append([document.id for document in result.documents])

        assert isinstance(result, RAGResult)
        assert results == [
            ["media-tie", "notes-high", "chat-tie"],
            ["media-tie", "notes-high", "chat-tie"],
        ]
        assert result.generated_answer is None
        assert all(retriever.retrieve.await_count == 0 for retriever in retrievers.values())

    @pytest.mark.asyncio
    async def test_entry_point_never_enters_generic_or_generative_external_stages(
        self,
        monkeypatch,
    ):
        closed_retrieval = _closed_slides_retrieval()
        _, _, generic_phase = _patch_closed_source_retrievers(monkeypatch)
        forbidden_calls = {}
        async_names = (
            "unified_rag_pipeline",
            "simple_search",
            "generate_hypothetical_answer",
            "classify_and_reformulate",
            "web_search_fallback",
            "fallback_to_web_search",
            "research_loop",
            "generate_suggestions",
        )
        sync_names = (
            "AnswerGenerator",
            "QueryRewriter",
            "DocumentGrader",
            "ClaimsEngine",
        )
        for name in async_names:
            forbidden_calls[name] = AsyncMock(side_effect=AssertionError(f"closed retrieval called {name}"))
            monkeypatch.setattr(
                unified_pipeline_module,
                name,
                forbidden_calls[name],
                raising=False,
            )
        for name in sync_names:
            forbidden_calls[name] = MagicMock(side_effect=AssertionError(f"closed retrieval called {name}"))
            monkeypatch.setattr(
                unified_pipeline_module,
                name,
                forbidden_calls[name],
                raising=False,
            )

        result = await closed_retrieval(
            query="local evidence",
            owner_user_id="42",
            top_k=3,
            media_db=object(),
            chacha_db=object(),
            reranking_strategy="none",
        )

        assert result.generated_answer is None
        assert all(mock.call_count == 0 for mock in forbidden_calls.values())
        generic_phase.assert_not_awaited()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("max_source_chars", (0, 200_001, True))
    async def test_entry_point_rejects_invalid_v1_character_ceiling_before_work(
        self,
        monkeypatch,
        max_source_chars,
    ):
        constructors = {
            name: MagicMock()
            for name in (
                "MediaDBRetriever",
                "NotesDBRetriever",
                "ChatHistoryRetriever",
            )
        }
        for name, constructor in constructors.items():
            monkeypatch.setattr(
                unified_pipeline_module,
                name,
                constructor,
                raising=False,
            )
        reranker_factory = MagicMock()
        monkeypatch.setattr(
            unified_pipeline_module,
            "create_preinstalled_local_reranker",
            reranker_factory,
        )

        with pytest.raises(ValueError):
            await _closed_slides_retrieval()(
                query="local evidence",
                owner_user_id="42",
                top_k=3,
                max_source_chars=max_source_chars,
                media_db=object(),
                chacha_db=object(),
            )

        assert all(constructor.call_count == 0 for constructor in constructors.values())
        reranker_factory.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "reranking_strategy",
        ("hybrid", "llama_cpp", "llm_scoring", "two_tier"),
    )
    async def test_entry_point_rejects_nonlocal_or_generative_rerankers_before_retrieval(
        self,
        monkeypatch,
        reranking_strategy,
    ):
        constructors = {
            name: MagicMock()
            for name in (
                "MediaDBRetriever",
                "NotesDBRetriever",
                "ChatHistoryRetriever",
            )
        }
        for name, constructor in constructors.items():
            monkeypatch.setattr(
                unified_pipeline_module,
                name,
                constructor,
                raising=False,
            )
        generic_phase = AsyncMock()
        monkeypatch.setattr(
            unified_pipeline_module,
            "execute_retrieval_phase",
            generic_phase,
        )

        with pytest.raises(ValueError):
            await _closed_slides_retrieval()(
                query="local evidence",
                owner_user_id="42",
                top_k=3,
                media_db=object(),
                chacha_db=object(),
                reranking_strategy=reranking_strategy,
            )

        assert all(constructor.call_count == 0 for constructor in constructors.values())
        generic_phase.assert_not_awaited()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("reranking_strategy", ("flashrank", "cross_encoder"))
    async def test_missing_local_reranker_is_rejected_before_retrieval(
        self,
        monkeypatch,
        tmp_path,
        reranking_strategy,
    ):
        missing_model = tmp_path / "missing-local-reranker"
        monkeypatch.setenv("RAG_FLASHRANK_MODEL_NAME", missing_model.name)
        monkeypatch.setenv("RAG_FLASHRANK_CACHE_DIR", str(tmp_path))
        monkeypatch.setenv("RAG_TRANSFORMERS_RERANKER_MODEL", str(missing_model))
        constructors = {
            name: MagicMock()
            for name in (
                "MediaDBRetriever",
                "NotesDBRetriever",
                "ChatHistoryRetriever",
            )
        }
        for name, constructor in constructors.items():
            monkeypatch.setattr(
                unified_pipeline_module,
                name,
                constructor,
                raising=False,
            )

        with pytest.raises(RAGConfigurationError):
            await _closed_slides_retrieval()(
                query="local evidence",
                owner_user_id="42",
                top_k=3,
                media_db=object(),
                chacha_db=object(),
                reranking_strategy=reranking_strategy,
            )

        assert all(constructor.call_count == 0 for constructor in constructors.values())


@pytest.mark.unit
class TestSlidesSourceDedicatedSqlRetrievers:
    @staticmethod
    def _build_retriever(source_name, adapter):
        config = database_retrievers.RetrievalConfig(
            max_results=3,
            use_fts=True,
            use_vector=False,
            fts_level="chunk",
        )
        if source_name == "media":
            return database_retrievers.MediaDBRetriever(
                None,
                config=config,
                user_id="owner-42",
                media_db=adapter,
            )
        if source_name == "notes":
            return database_retrievers.NotesDBRetriever(
                None,
                config=config,
                chacha_db=adapter,
            )
        return database_retrievers.ChatHistoryRetriever(
            None,
            config=config,
            chacha_db=adapter,
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("source_name", "active_fragments"),
        (
            ("media", ("deleted", "is_trash", "owner_user_id")),
            ("notes", ("deleted", "client_id")),
            ("chats", ("deleted", "client_id")),
        ),
    )
    async def test_candidate_sql_is_owner_scoped_active_and_metadata_only(
        self,
        monkeypatch,
        source_name,
        active_fragments,
    ):
        adapter = _SqlCaptureAdapter()
        retriever = self._build_retriever(source_name, adapter)
        generic_retrieve = AsyncMock(side_effect=AssertionError("generic retrieval fallback was called"))
        monkeypatch.setattr(retriever, "retrieve", generic_retrieve)

        documents = await retriever.retrieve_slides_source_documents_v1(
            query="owner evidence",
            owner_user_id="owner-42",
            max_source_chars=80,
            top_k=3,
        )

        assert documents == []
        generic_retrieve.assert_not_awaited()
        assert adapter.calls
        sql = "\n".join(query for query, _ in adapter.calls).lower()
        params = tuple(value for _, values in adapter.calls for value in values)
        assert "select *" not in sql
        assert "substr(" not in sql
        assert "source_text" not in sql
        assert "length(" in sql
        assert "_standalone_source_invalid_text" in sql
        assert "plainto_tsquery" in sql
        assert " like " not in sql
        assert re.search(r"\bdeleted\s*=\s*(?:0|false)\b", sql)
        assert all(fragment in sql for fragment in active_fragments)
        assert "owner-42" in params
        query_params = " ".join(str(value) for value in params)
        assert "owner" in query_params
        assert "evidence" in query_params
        assert not re.search(r"\burl\b", sql)
        assert "image_data" not in sql
        assert "image_mime" not in sql
        assert "message_images" not in sql
        assert ".metadata" not in sql
        if source_name == "media":
            assert adapter.options == [{"log_errors": False}]
        else:
            assert adapter.options == [
                {"log_params": False, "log_errors": False}
            ]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("source_name", "source"),
        (
            ("media", DataSource.MEDIA_DB),
            ("notes", DataSource.NOTES),
            ("chats", DataSource.CHAT_HISTORY),
        ),
    )
    async def test_dedicated_sql_preserves_stored_truncation_marker(
        self,
        source_name,
        source,
    ):
        candidate_rows = {
            "media": {
                "chunk_uuid": "source-1",
                "media_id": 1,
                "chunk_index": 0,
                "rank": 0.25,
            },
            "notes": {
                "id": "source-1",
                "updated_at": "2026-01-01",
                "rank": 0.25,
            },
            "chats": {
                "id": "source-1",
                "conversation_id": "conversation-1",
                "timestamp": "2026-01-01",
                "rank": 0.25,
            },
        }
        candidate_rows[source_name].update(
            {
                "_standalone_source_full_chars": 120,
                "_standalone_source_invalid_text": False,
            }
        )
        projection_row = {
            "projection_order": 0,
            "projection_cap": 80,
            "source_text": ("user: bounded body" if source_name == "chats" else "# Title\n\nbounded body"),
            "_standalone_source_projection_truncated": True,
            "_standalone_source_invalid_text": False,
        }
        adapter = _SqlCaptureAdapter(row_batches=([candidate_rows[source_name]], [projection_row]))
        retriever = self._build_retriever(source_name, adapter)

        documents = await retriever.retrieve_slides_source_documents_v1(
            query="bounded",
            owner_user_id="owner-42",
            max_source_chars=80,
            top_k=3,
        )

        assert len(documents) == 1
        assert documents[0].source is source
        assert documents[0].metadata["_standalone_source_projection_truncated"] is True
        assert documents[0].metadata["_standalone_source_preformatted"] is True
        assert "url" not in documents[0].metadata
        assert "image_data" not in documents[0].metadata
        assert "chunk_metadata" not in documents[0].metadata


@pytest.mark.unit
class TestSlidesSourceLocalRerankers:
    @staticmethod
    def _write_complete_flashrank_bundle(model_dir):
        model_dir.mkdir(parents=True)
        for name in (
            "config.json",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "flashrank-TinyBERT-L-2-v2.onnx",
        ):
            (model_dir / name).write_bytes(b"{}")

    def test_flashrank_rejects_incomplete_bundle_before_constructor(
        self,
        monkeypatch,
        tmp_path,
    ):
        model_name = "ms-marco-TinyBERT-L-2-v2"
        model_dir = tmp_path / model_name
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{}")
        constructor = MagicMock()
        monkeypatch.setenv("RAG_FLASHRANK_MODEL_NAME", model_name)
        monkeypatch.setenv("RAG_FLASHRANK_CACHE_DIR", str(tmp_path))
        monkeypatch.setattr(advanced_reranking, "FlashRankReranker", constructor)

        with pytest.raises(RAGConfigurationError):
            advanced_reranking.create_preinstalled_local_reranker(
                "flashrank",
                top_k=3,
            )

        constructor.assert_not_called()

    def test_flashrank_rejects_listwise_model_even_with_local_bundle(
        self,
        monkeypatch,
        tmp_path,
    ):
        model_name = "rank_zephyr_7b_v1_full"
        self._write_complete_flashrank_bundle(tmp_path / model_name)
        constructor = MagicMock()
        monkeypatch.setenv("RAG_FLASHRANK_MODEL_NAME", model_name)
        monkeypatch.setenv("RAG_FLASHRANK_CACHE_DIR", str(tmp_path))
        monkeypatch.setattr(advanced_reranking, "FlashRankReranker", constructor)

        with pytest.raises(RAGConfigurationError):
            advanced_reranking.create_preinstalled_local_reranker(
                "flashrank",
                top_k=3,
            )

        constructor.assert_not_called()

    def test_flashrank_accepts_complete_non_listwise_local_bundle(
        self,
        monkeypatch,
        tmp_path,
    ):
        model_name = "ms-marco-TinyBERT-L-2-v2"
        self._write_complete_flashrank_bundle(tmp_path / model_name)

        class FakeFlashRankReranker:
            def __init__(self, config, **kwargs):
                self.config = config
                self.local_model_dir = kwargs["local_model_dir"]
                self.required_local_files = kwargs["required_local_files"]
                self._ranker = object()

        monkeypatch.setenv("RAG_FLASHRANK_MODEL_NAME", model_name)
        monkeypatch.setenv("RAG_FLASHRANK_CACHE_DIR", str(tmp_path))
        monkeypatch.setattr(
            advanced_reranking,
            "FlashRankReranker",
            FakeFlashRankReranker,
        )

        reranker = advanced_reranking.create_preinstalled_local_reranker(
            "flashrank",
            top_k=3,
        )

        assert reranker.config.model_name == model_name
        assert reranker.config.fail_closed_on_error is True
        assert reranker.local_model_dir == (tmp_path / model_name).resolve()
        assert "flashrank-TinyBERT-L-2-v2.onnx" in reranker.required_local_files

    def test_cross_encoder_sentence_transformers_is_strictly_local(
        self,
        monkeypatch,
        tmp_path,
    ):
        model_dir = tmp_path / "cross-encoder"
        model_dir.mkdir()
        calls = []

        class FakeCrossEncoder:
            def __init__(self, model_id, **kwargs):
                calls.append((model_id, kwargs))

        monkeypatch.setitem(
            sys.modules,
            "sentence_transformers",
            types.SimpleNamespace(CrossEncoder=FakeCrossEncoder),
        )
        monkeypatch.setenv("RAG_TRANSFORMERS_RERANKER_MODEL", str(model_dir))

        reranker = advanced_reranking.create_preinstalled_local_reranker(
            "cross_encoder",
            top_k=3,
        )

        assert reranker is not None
        assert calls[0][0] == str(model_dir)
        assert calls[0][1]["local_files_only"] is True
        assert calls[0][1]["trust_remote_code"] is False
