import ast
import configparser
import inspect
import importlib
import sys
from types import ModuleType
from pathlib import Path

import pytest

from tldw_Server_API.app.core.DB_Management import DB_Manager
from tldw_Server_API.app.core.DB_Management import Users_DB
from tldw_Server_API.app.core.DB_Management import Prompts_DB as prompts_db_module
from tldw_Server_API.app.core.DB_Management import db_path_utils
from tldw_Server_API.app.core.AuthNZ import migrate_to_multiuser
from tldw_Server_API.app.api.v1.endpoints import rag_unified as rag_unified_endpoint
from tldw_Server_API.app.api.v1.endpoints import research
from tldw_Server_API.app.api.v1.endpoints import claims as claims_endpoint
from tldw_Server_API.app.api.v1.endpoints import chunking as chunking_endpoint
from tldw_Server_API.app.api.v1.endpoints import chunking_templates as chunking_templates_endpoint
from tldw_Server_API.app.api.v1.endpoints import embeddings_v5_production_enhanced as embeddings_v5_endpoint
from tldw_Server_API.app.api.v1.endpoints import paper_search as paper_search_endpoint
from tldw_Server_API.app.api.v1.endpoints.audio import audiobooks as audiobooks_endpoint
from tldw_Server_API.app.api.v1.endpoints import slides as slides_endpoint
from tldw_Server_API.app.api.v1.endpoints import text2sql as text2sql_endpoint
from tldw_Server_API.app.api.v1.endpoints import media_embeddings as media_embeddings_endpoint
from tldw_Server_API.app.api.v1.endpoints import email as email_endpoint
from tldw_Server_API.app.api.v1.endpoints import data_tables
from tldw_Server_API.app.api.v1.endpoints import items
from tldw_Server_API.app.api.v1.endpoints import quizzes as quizzes_endpoint
from tldw_Server_API.app.api.v1.endpoints import vector_stores_openai as vector_stores_endpoint
from tldw_Server_API.app.api.v1.endpoints import sync
from tldw_Server_API.app.api.v1.API_Deps import DB_Deps as media_db_deps
from tldw_Server_API.app.api.v1.endpoints.media import document_outline
from tldw_Server_API.app.api.v1.endpoints.media import document_insights
from tldw_Server_API.app.api.v1.endpoints.media import document_references
from tldw_Server_API.app.api.v1.endpoints.media import add as media_add_endpoint
from tldw_Server_API.app.api.v1.endpoints.media import document_annotations as media_document_annotations_endpoint
from tldw_Server_API.app.api.v1.endpoints.media import debug as media_debug_endpoint
from tldw_Server_API.app.api.v1.endpoints.media import document_figures as media_document_figures_endpoint
from tldw_Server_API.app.api.v1.endpoints.media import file as media_file_endpoint
from tldw_Server_API.app.api.v1.endpoints.media import ingest_web_content as media_ingest_web_content_endpoint
from tldw_Server_API.app.api.v1.endpoints.media import item as media_item
from tldw_Server_API.app.api.v1.endpoints.media import listing as media_listing_endpoint
from tldw_Server_API.app.api.v1.endpoints.media import navigation as media_navigation
from tldw_Server_API.app.api.v1.endpoints.media import process_audios as media_process_audios_endpoint
from tldw_Server_API.app.api.v1.endpoints.media import process_code as media_process_code_endpoint
from tldw_Server_API.app.api.v1.endpoints.media import process_ebooks as media_process_ebooks_endpoint
from tldw_Server_API.app.api.v1.endpoints.media import process_documents
from tldw_Server_API.app.api.v1.endpoints.media import process_emails as media_process_emails_endpoint
from tldw_Server_API.app.api.v1.endpoints.media import process_pdfs
from tldw_Server_API.app.api.v1.endpoints.media import process_videos as media_process_videos_endpoint
from tldw_Server_API.app.api.v1.endpoints.media import process_web_scraping as media_process_web_scraping_endpoint
from tldw_Server_API.app.api.v1.endpoints.media import reprocess as media_reprocess_endpoint
from tldw_Server_API.app.api.v1.endpoints.audio import audio_history as audio_history_endpoint
from tldw_Server_API.app.api.v1.endpoints.audio import audio_tts as audio_tts_endpoint
from tldw_Server_API.app.api.v1.utils import http_errors
from tldw_Server_API.app.api.v1.endpoints.media import versions as media_versions
from tldw_Server_API.app.core.Claims_Extraction import (
    claims_clustering,
    ingestion_claims,
    claims_notifications,
    claims_rebuild_service,
    claims_service,
    claims_utils,
    review_assignment,
)
from tldw_Server_API.app.core.Evaluations import embeddings_abtest_service
from tldw_Server_API.app.core.Evaluations import embeddings_abtest_jobs_worker
from tldw_Server_API.app.core.Embeddings.services import (
    jobs_worker as embeddings_jobs_worker,
    vector_compactor,
)
from tldw_Server_API.app.core.Embeddings import ChromaDB_Library
from tldw_Server_API.app.core.Chunking import template_initialization
from tldw_Server_API.app.core.Ingestion_Media_Processing import visual_ingestion
from tldw_Server_API.app.core.TTS import tts_jobs_worker
from tldw_Server_API.app.core.Workflows.adapters.knowledge import crud as knowledge_crud
from tldw_Server_API.app.core.Workflows.adapters.media import ingest as workflow_media_ingest
from tldw_Server_API.app.core.Ingestion_Media_Processing.Books import Book_Processing_Lib
from tldw_Server_API.app.core.Ingestion_Media_Processing import persistence as ingestion_persistence
from tldw_Server_API.app.core.Ingestion_Media_Processing import XML_Ingestion_Lib
from tldw_Server_API.app.core.Ingestion_Media_Processing.MediaWiki import Media_Wiki
from tldw_Server_API.app.core.Data_Tables import jobs_worker as data_tables_jobs_worker
from tldw_Server_API.app.core.External_Sources import sync_coordinator
from tldw_Server_API.app.core.RAG.rag_service import agentic_chunker
from tldw_Server_API.app.core.RAG.rag_service import database_retrievers
from tldw_Server_API.app.core.RAG.rag_service import unified_pipeline
from tldw_Server_API.app.core.Chatbooks import chatbook_service
from tldw_Server_API.app.core.MCP_unified.modules.implementations import media_module as media_module_impl
from tldw_Server_API.app.core.Sync import Sync_Client as sync_client_module
from tldw_Server_API.app.services import ingestion_sources_worker
from tldw_Server_API.app.core.MCP_unified.modules.implementations import quizzes_module
from tldw_Server_API.app.core.MCP_unified.modules.implementations import slides_module
from tldw_Server_API.app.core.Utils import metadata_utils
from tldw_Server_API.app.core.Web_Scraping import Article_Extractor_Lib
from tldw_Server_API.app.core.Watchlists import pipeline as watchlists_pipeline
from tldw_Server_API.app.core.DB_Management.media_db import api as media_db_api
from tldw_Server_API.app.core.DB_Management import media_db as media_db_package
from tldw_Server_API.app.core.DB_Management.media_db import legacy_backup as media_db_legacy_backup
from tldw_Server_API.app.core.DB_Management.media_db import errors as media_db_errors
from tldw_Server_API.app.core.DB_Management.media_db import native_class as media_db_native_class
from tldw_Server_API.app.core.DB_Management.media_db import legacy_identifiers as media_db_legacy_identifiers
from tldw_Server_API.app.core.DB_Management.media_db import legacy_document_artifacts
from tldw_Server_API.app.core.DB_Management.media_db import legacy_content_queries
from tldw_Server_API.app.core.DB_Management.media_db import legacy_maintenance
from tldw_Server_API.app.core.DB_Management.media_db import legacy_reads as media_db_legacy_reads
from tldw_Server_API.app.core.DB_Management.media_db import legacy_state as media_db_state
from tldw_Server_API.app.core.DB_Management.media_db import legacy_transcripts
from tldw_Server_API.app.core.DB_Management.media_db import legacy_wrappers
from tldw_Server_API.app.core.DB_Management.media_db.repositories import chunks_repository
from tldw_Server_API.app.core.DB_Management.media_db.repositories import document_versions_repository
from tldw_Server_API.app.core.DB_Management.media_db.repositories import keywords_repository
from tldw_Server_API.app.core.DB_Management.media_db.repositories import media_files_repository
from tldw_Server_API.app.core.DB_Management.media_db.repositories import media_repository
from tldw_Server_API.app.core.DB_Management.media_db.runtime import factory as media_db_runtime_factory
from tldw_Server_API.app.core.DB_Management.media_db.runtime import execution as media_db_runtime_execution
from tldw_Server_API.app.core.DB_Management.media_db.runtime import media_class as media_db_runtime_media_class
from tldw_Server_API.app.core.DB_Management.media_db.runtime import rows as media_db_runtime_rows
from tldw_Server_API.app.core.DB_Management.media_db.runtime import validation as media_db_validation
from tldw_Server_API.app.services import admin_bundle_service
from tldw_Server_API.app.services import media_files_cleanup_service
from tldw_Server_API.app.services import document_processing_service
from tldw_Server_API.app.services import claims_alerts_scheduler
from tldw_Server_API.app.services import claims_review_metrics_scheduler
from tldw_Server_API.app.services import connectors_worker
from tldw_Server_API.app.services import meetings_webhook_dlq_service
from tldw_Server_API.app.services import quiz_generator
from tldw_Server_API.app.services import quiz_source_resolver
from tldw_Server_API.app.services import audiobook_jobs_worker
from tldw_Server_API.app.services import enhanced_web_scraping_service
from tldw_Server_API.app.services import media_ingest_jobs_worker
from tldw_Server_API.app.services import outputs_purge_scheduler
from tldw_Server_API.app.services import storage_cleanup_service
from tldw_Server_API.app.services import tts_history_cleanup_service
from tldw_Server_API.app.services import web_scraping_service


class _LazyLegacyMediaDBProxy(ModuleType):
    """Module-like proxy that stays detached from the deleted legacy module."""

    _STUB_ATTRS = {
        "ConflictError": object(),
        "DatabaseError": object(),
        "InputError": object(),
        "SchemaError": object(),
        "MediaDatabase": object(),
        "check_media_exists": object(),
        "get_document_version": object(),
        "get_full_media_details": object(),
        "get_full_media_details_rich": object(),
        "create_automated_backup": object(),
    }

    def __init__(self) -> None:
        legacy_module_path = (
            Path(__file__).resolve().parents[2]
            / "app/core/DB_Management/Media_DB_v2.py"
        )
        super().__init__("tldw_Server_API.app.core.DB_Management.Media_DB_v2")
        object.__setattr__(self, "__file__", str(legacy_module_path))
        object.__setattr__(self, "__package__", "tldw_Server_API.app.core.DB_Management")
        for name, value in self._STUB_ATTRS.items():
            object.__setattr__(self, name, value)

    def __getattr__(self, name: str):
        value = object()
        object.__setattr__(self, name, value)
        return value


legacy_media_db = _LazyLegacyMediaDBProxy()


REPO_ROOT = Path(__file__).resolve().parents[3]


def _repo_path(relative_path: str) -> Path:
    return REPO_ROOT / relative_path


def _import_targets_from_source(source: str) -> set[str]:
    """Return fully qualified import targets found in source code."""
    targets: set[str] = set()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module is None:
                continue
            prefix = "." * node.level + node.module if node.level else node.module
            for alias in node.names:
                targets.add(f"{prefix}.{alias.name}")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                targets.add(alias.name)
    return targets


def _has_import_target_prefix(import_targets: set[str], forbidden_prefix: str) -> bool:
    """Return True when any parsed import target matches a forbidden prefix."""
    return any(
        target == forbidden_prefix or target.startswith(f"{forbidden_prefix}.")
        for target in import_targets
    )


def test_media_db_api_create_media_database_uses_runtime_factory(monkeypatch, tmp_path):
    captured = {}
    override_db_path = str(tmp_path / "override.db")
    runtime = media_db_runtime_factory.MediaDbRuntimeConfig(
        default_db_path=str(tmp_path / "api-media.db"),
        default_config=configparser.ConfigParser(),
        postgres_content_mode=True,
        backend_loader=lambda: "backend-sentinel",
    )

    def _fake_runtime_create_media_database(client_id, **kwargs):
        captured["client_id"] = client_id
        captured.update(kwargs)
        return "db-instance"

    cfg = configparser.ConfigParser()
    monkeypatch.setattr(
        media_db_api,
        "runtime_create_media_database",
        _fake_runtime_create_media_database,
        raising=False,
    )
    monkeypatch.setattr(
        media_db_api,
        "build_media_runtime_config",
        lambda: runtime,
        raising=False,
    )

    result = media_db_api.create_media_database(
        "client-api",
        db_path=override_db_path,
    )

    assert result == "db-instance"
    assert captured["client_id"] == "client-api"
    assert captured["db_path"] == override_db_path
    assert captured["runtime"] is runtime
    assert captured["runtime"].postgres_content_mode is True


def test_media_db_api_no_longer_mentions_db_manager_in_source() -> None:
    assert "DB_Manager" not in inspect.getsource(media_db_api)


def test_media_db_runtime_validation_exposes_protocol() -> None:
    assert getattr(media_db_validation.MediaDbLike, "_is_protocol", False) is True


def test_media_db_runtime_validation_exposes_read_protocol() -> None:
    assert getattr(media_db_validation.MediaDbReadLike, "_is_protocol", False) is True


def test_media_db_api_exposes_read_contract_functions() -> None:
    assert callable(getattr(media_db_api, "create_document_version", None))
    assert callable(getattr(media_db_api, "fetch_all_keywords", None))
    assert callable(getattr(media_db_api, "get_all_document_versions", None))
    assert callable(getattr(media_db_api, "list_chunking_templates", None))
    assert callable(getattr(media_db_api, "seed_builtin_templates", None))
    assert callable(getattr(media_db_api, "lookup_section_for_offset", None))
    assert callable(getattr(media_db_api, "lookup_section_by_heading", None))
    assert callable(getattr(media_db_api, "get_media_by_id", None))
    assert callable(getattr(media_db_api, "get_media_status_by_id", None))
    assert callable(getattr(media_db_api, "has_unvectorized_chunks", None))
    assert callable(getattr(media_db_api, "get_media_by_uuid", None))
    assert callable(getattr(media_db_api, "get_media_by_url", None))
    assert callable(getattr(media_db_api, "get_media_by_hash", None))
    assert callable(getattr(media_db_api, "get_media_by_title", None))
    assert callable(getattr(media_db_api, "get_distinct_media_types", None))
    assert callable(getattr(media_db_api, "get_unvectorized_chunk_count", None))
    assert callable(getattr(media_db_api, "get_unvectorized_anchor_index_for_offset", None))
    assert callable(getattr(media_db_api, "get_unvectorized_chunk_index_by_uuid", None))
    assert callable(getattr(media_db_api, "get_unvectorized_chunk_by_index", None))
    assert callable(getattr(media_db_api, "get_unvectorized_chunks_in_range", None))
    assert callable(getattr(media_db_api, "search_media", None))
    assert callable(getattr(media_db_api, "list_document_versions", None))
    assert callable(getattr(media_db_api, "soft_delete_document_version", None))
    assert callable(getattr(media_db_api, "soft_delete_keyword", None))
    assert callable(getattr(media_db_api, "update_keywords_for_media", None))
    assert callable(getattr(media_db_api, "get_full_media_details", None))
    assert callable(getattr(media_db_api, "get_full_media_details_rich", None))


def test_media_db_api_get_unvectorized_chunk_count_accepts_lightweight_read_double() -> None:
    class StubReader:
        def get_unvectorized_chunk_count(self, media_id: int):
            return media_id + 1

    result = media_db_api.get_unvectorized_chunk_count(StubReader(), 9)

    assert result == 10


def test_media_db_api_has_unvectorized_chunks_accepts_lightweight_read_double() -> None:
    class StubReader:
        def has_unvectorized_chunks(self, media_id: int) -> bool:
            return media_id == 9

    assert media_db_api.has_unvectorized_chunks(StubReader(), 9) is True
    assert media_db_api.has_unvectorized_chunks(StubReader(), 8) is False


def test_media_db_api_fetch_all_keywords_accepts_lightweight_read_double() -> None:
    class StubReader:
        def fetch_all_keywords(self) -> list[str]:
            return ["beta", "alpha"]

    assert media_db_api.fetch_all_keywords(StubReader()) == ["beta", "alpha"]


def test_media_db_api_create_document_version_accepts_lightweight_double() -> None:
    class StubDb:
        def create_document_version(
            self,
            media_id: int,
            content: str,
            prompt=None,
            analysis_content=None,
            safe_metadata=None,
        ):
            return {
                "media_id": media_id,
                "content": content,
                "prompt": prompt,
                "analysis_content": analysis_content,
                "safe_metadata": safe_metadata,
            }

    result = media_db_api.create_document_version(
        StubDb(),
        media_id=9,
        content="v2",
        prompt="p",
        analysis_content="a",
        safe_metadata="{}",
    )

    assert result["media_id"] == 9
    assert result["content"] == "v2"


def test_media_db_api_update_keywords_for_media_accepts_lightweight_double() -> None:
    class StubDb:
        def update_keywords_for_media(self, media_id: int, keywords: list[str], conn=None):
            return {"media_id": media_id, "keywords": keywords, "conn": conn}

    result = media_db_api.update_keywords_for_media(StubDb(), media_id=5, keywords=["x", "y"])

    assert result == {"media_id": 5, "keywords": ["x", "y"], "conn": None}


def test_media_db_api_soft_delete_keyword_accepts_lightweight_double() -> None:
    class StubDb:
        def soft_delete_keyword(self, keyword: str) -> bool:
            return keyword == "alpha"

    assert media_db_api.soft_delete_keyword(StubDb(), "alpha") is True
    assert media_db_api.soft_delete_keyword(StubDb(), "beta") is False


def test_media_db_api_soft_delete_keyword_accepts_partial_legacy_like_db() -> None:
    class _Cursor:
        def __init__(self, *, rows=None, rowcount: int = 0):
            self._rows = rows or []
            self.rowcount = rowcount

        def fetchall(self):
            return self._rows

    class PartialDb:
        client_id = "tenant-42"

        def __init__(self) -> None:
            self._fetchone_calls = 0
            self.logged_events: list[tuple[str, str]] = []
            self.deleted_fts: list[int] = []

        def transaction(self):
            class _Tx:
                def __enter__(self_inner):
                    return object()

                def __exit__(self_inner, exc_type, exc, tb):
                    return False

            return _Tx()

        def _fetchone_with_connection(self, connection, query: str, params=None):
            self._fetchone_calls += 1
            if self._fetchone_calls == 1:
                return {"id": 7, "uuid": "kw-uuid", "version": 2}
            raise AssertionError("unexpected extra fetchone call")

        def _execute_with_connection(self, connection, query: str, params=None):
            if query.startswith("UPDATE Keywords"):
                return _Cursor(rowcount=1)
            if query.startswith("SELECT mk.media_id"):
                return _Cursor(rows=[{"media_id": 3, "media_uuid": "media-uuid"}])
            if query.startswith("DELETE FROM MediaKeywords"):
                return _Cursor(rowcount=1)
            raise AssertionError(f"unexpected query: {query}")

        def _get_current_utc_timestamp_str(self) -> str:
            return "2026-03-20T12:00:00Z"

        def _log_sync_event(
            self,
            connection,
            table_name: str,
            entity_uuid: str,
            action: str,
            version: int,
            payload=None,
        ) -> None:
            self.logged_events.append((table_name, action))

        def _delete_fts_keyword(self, connection, keyword_id: int) -> None:
            self.deleted_fts.append(keyword_id)

    db = PartialDb()

    assert media_db_api.soft_delete_keyword(db, "Science") is True
    assert db.deleted_fts == [7]
    assert ("Keywords", "delete") in db.logged_events
    assert ("MediaKeywords", "unlink") in db.logged_events


def test_media_db_api_soft_delete_document_version_accepts_lightweight_double() -> None:
    class StubDb:
        def soft_delete_document_version(self, version_uuid: str) -> bool:
            return version_uuid == "v1"

    assert media_db_api.soft_delete_document_version(StubDb(), "v1") is True
    assert media_db_api.soft_delete_document_version(StubDb(), "v2") is False


def test_media_db_api_list_chunking_templates_accepts_lightweight_double() -> None:
    class StubDb:
        def list_chunking_templates(self, **kwargs):
            return [{"name": "stub", "kwargs": kwargs}]

    result = media_db_api.list_chunking_templates(
        StubDb(),
        include_builtin=False,
        include_custom=True,
        tags=["x"],
        user_id="u1",
        include_deleted=True,
    )

    assert result == [
        {
            "name": "stub",
            "kwargs": {
                "include_builtin": False,
                "include_custom": True,
                "tags": ["x"],
                "user_id": "u1",
                "include_deleted": True,
            },
        }
    ]


def test_media_db_api_lookup_section_for_offset_accepts_lightweight_double() -> None:
    class StubDb:
        def lookup_section_for_offset(self, media_id: int, char_offset: int):
            return {"media_id": media_id, "start_char": char_offset}

    assert media_db_api.lookup_section_for_offset(StubDb(), 9, 17) == {
        "media_id": 9,
        "start_char": 17,
    }


def test_media_db_api_lookup_section_by_heading_accepts_lightweight_double() -> None:
    class StubDb:
        def lookup_section_by_heading(self, media_id: int, heading: str):
            return (10, 20, heading.upper())

    assert media_db_api.lookup_section_by_heading(StubDb(), 4, "intro") == (10, 20, "INTRO")


def test_media_db_api_search_media_requires_read_contract() -> None:
    with pytest.raises(TypeError, match="read contract"):
        media_db_api.search_media(object(), "query")


def test_media_db_api_get_media_by_id_accepts_lightweight_read_double() -> None:
    class StubReader:
        def get_media_by_id(self, media_id: int, include_deleted: bool = False, include_trash: bool = False):
            return {
                "id": media_id,
                "include_deleted": include_deleted,
                "include_trash": include_trash,
            }

    result = media_db_api.get_media_by_id(
        StubReader(),
        9,
        include_deleted=True,
        include_trash=True,
    )

    assert result == {"id": 9, "include_deleted": True, "include_trash": True}


def test_media_db_api_get_media_status_by_id_accepts_lightweight_status_double() -> None:
    class StubReader:
        def get_media_status_by_id(
            self,
            media_id: int,
            include_deleted: bool = False,
            include_trash: bool = False,
        ):
            return {
                "id": media_id,
                "has_content": True,
                "include_deleted": include_deleted,
                "include_trash": include_trash,
            }

    result = media_db_api.get_media_status_by_id(
        StubReader(),
        9,
        include_deleted=True,
        include_trash=True,
    )

    assert result == {
        "id": 9,
        "has_content": True,
        "include_deleted": True,
        "include_trash": True,
    }


def test_media_db_api_search_media_accepts_lightweight_search_double() -> None:
    class StubSearcher:
        def search_media_db(self, search_query: str | None, **kwargs):
            return ([{"id": 1, "query": search_query, "page": kwargs["page"]}], 1)

    rows, total = media_db_api.search_media(
        StubSearcher(),
        "hello",
        page=2,
        results_per_page=10,
    )

    assert rows == [{"id": 1, "query": "hello", "page": 2}]
    assert total == 1


def test_media_db_api_get_paginated_files_prefers_direct_db_method() -> None:
    class StubDb:
        def get_paginated_files(self, page: int, results_per_page: int):
            return ([{"id": 1}], 7, page, results_per_page)

    rows, total_pages, current_page, total_items = media_db_api.get_paginated_files(
        StubDb(),
        page=3,
        results_per_page=12,
    )

    assert rows == [{"id": 1}]
    assert total_pages == 7
    assert current_page == 3
    assert total_items == 12


def test_media_db_api_get_paginated_trash_files_prefers_direct_db_method() -> None:
    class StubDb:
        def get_paginated_trash_list(self, page: int, results_per_page: int):
            return ([{"id": 2}], 5, page, results_per_page)

    rows, total_pages, current_page, total_items = media_db_api.get_paginated_trash_files(
        StubDb(),
        page=4,
        results_per_page=9,
    )

    assert rows == [{"id": 2}]
    assert total_pages == 5
    assert current_page == 4
    assert total_items == 9


def test_media_db_api_fetch_keywords_for_media_delegates_to_helper(monkeypatch) -> None:
    captured = {}

    def _fake_fetch_keywords_for_media(media_id: int, db_instance):
        captured["media_id"] = media_id
        captured["db_instance"] = db_instance
        return ["alpha", "beta"]

    monkeypatch.setattr(
        legacy_content_queries,
        "fetch_keywords_for_media",
        _fake_fetch_keywords_for_media,
    )

    sentinel_db = object()
    result = media_db_api.fetch_keywords_for_media(sentinel_db, 11)

    assert result == ["alpha", "beta"]
    assert captured == {"media_id": 11, "db_instance": sentinel_db}


def test_media_db_api_fetch_keywords_for_media_batch_delegates_to_helper(monkeypatch) -> None:
    captured = {}

    def _fake_fetch_keywords_for_media_batch(media_ids: list[int], db_instance):
        captured["media_ids"] = media_ids
        captured["db_instance"] = db_instance
        return {7: ["alpha"]}

    monkeypatch.setattr(
        legacy_content_queries,
        "fetch_keywords_for_media_batch",
        _fake_fetch_keywords_for_media_batch,
    )

    sentinel_db = object()
    result = media_db_api.fetch_keywords_for_media_batch(sentinel_db, [7, 8])

    assert result == {7: ["alpha"]}
    assert captured == {"media_ids": [7, 8], "db_instance": sentinel_db}


def test_media_db_api_check_media_exists_delegates_to_helper(monkeypatch) -> None:
    captured = {}

    def _fake_check_media_exists(db_instance, media_id=None, url=None, content_hash=None):
        captured["db_instance"] = db_instance
        captured["media_id"] = media_id
        captured["url"] = url
        captured["content_hash"] = content_hash
        return 42

    monkeypatch.setattr(
        media_db_state,
        "check_media_exists",
        _fake_check_media_exists,
    )

    sentinel_db = object()
    result = media_db_api.check_media_exists(
        sentinel_db,
        media_id=7,
        url="https://example.com",
        content_hash="abc123",
    )

    assert result == 42
    assert captured == {
        "db_instance": sentinel_db,
        "media_id": 7,
        "url": "https://example.com",
        "content_hash": "abc123",
    }


def test_media_db_api_permanently_delete_item_delegates_to_helper(monkeypatch) -> None:
    captured = {}

    def _fake_permanently_delete_item(db_instance, media_id: int):
        captured["db_instance"] = db_instance
        captured["media_id"] = media_id
        return True

    monkeypatch.setattr(
        legacy_maintenance,
        "permanently_delete_item",
        _fake_permanently_delete_item,
    )

    sentinel_db = object()
    result = media_db_api.permanently_delete_item(sentinel_db, 13)

    assert result is True
    assert captured == {"db_instance": sentinel_db, "media_id": 13}


def test_media_db_api_get_latest_transcription_delegates_to_helper(monkeypatch) -> None:
    captured = {}

    def _fake_get_latest_transcription(db_instance, media_id: int):
        captured["db_instance"] = db_instance
        captured["media_id"] = media_id
        return "latest transcription"

    monkeypatch.setattr(
        media_db_legacy_reads,
        "get_latest_transcription",
        _fake_get_latest_transcription,
    )

    sentinel_db = object()
    result = media_db_api.get_latest_transcription(sentinel_db, 17)

    assert result == "latest transcription"
    assert captured == {"db_instance": sentinel_db, "media_id": 17}


def test_media_db_api_get_media_transcripts_delegates_to_helper(monkeypatch) -> None:
    captured = {}

    def _fake_get_media_transcripts(db_instance, media_id: int):
        captured["db_instance"] = db_instance
        captured["media_id"] = media_id
        return [{"id": 1, "media_id": media_id}]

    monkeypatch.setattr(
        media_db_legacy_reads,
        "get_media_transcripts",
        _fake_get_media_transcripts,
    )

    sentinel_db = object()
    result = media_db_api.get_media_transcripts(sentinel_db, 12)

    assert result == [{"id": 1, "media_id": 12}]
    assert captured == {"db_instance": sentinel_db, "media_id": 12}


def test_media_db_api_get_media_transcripts_unwraps_wrapped_db(monkeypatch) -> None:
    captured = {}

    class StubDatabase:
        client_id = "wrapped-client"
        db_path_str = "/tmp/wrapped-media.db"

        def execute_query(self, query: str, params=None):
            return None

        def transaction(self):
            raise AssertionError("transaction should not be used in this helper test")

        def _fetchall_with_connection(self, connection, query: str, params=None):
            return []

        def _fetchone_with_connection(self, connection, query: str, params=None):
            return None

        def _execute_with_connection(self, connection, query: str, params=None):
            return None

        def _get_current_utc_timestamp_str(self) -> str:
            return "2026-03-19T00:00:00Z"

        def _log_sync_event(
            self,
            connection,
            table_name: str,
            entity_uuid: str,
            action: str,
            version: int,
            payload=None,
        ) -> None:
            return None

        def initialize_db(self) -> None:
            return None

        def close_connection(self) -> None:
            return None

    wrapped_db = StubDatabase()

    class StubWrapper:
        database = wrapped_db

    def _fake_get_media_transcripts(db_instance, media_id: int):
        captured["db_instance"] = db_instance
        captured["media_id"] = media_id
        return []

    monkeypatch.setattr(
        media_db_legacy_reads,
        "get_media_transcripts",
        _fake_get_media_transcripts,
    )

    result = media_db_api.get_media_transcripts(StubWrapper(), 21)

    assert result == []
    assert media_db_validation.is_media_database_like(wrapped_db) is True
    assert captured == {"db_instance": wrapped_db, "media_id": 21}


def test_media_db_api_get_media_prompts_delegates_to_helper(monkeypatch) -> None:
    captured = {}

    def _fake_get_media_prompts(db_instance, media_id: int):
        captured["db_instance"] = db_instance
        captured["media_id"] = media_id
        return [{"id": 1, "prompt": "hello"}]

    monkeypatch.setattr(
        media_db_legacy_reads,
        "get_media_prompts",
        _fake_get_media_prompts,
    )

    sentinel_db = object()
    result = media_db_api.get_media_prompts(sentinel_db, 12)

    assert result == [{"id": 1, "prompt": "hello"}]
    assert captured == {"db_instance": sentinel_db, "media_id": 12}


def test_media_db_api_get_document_version_delegates_to_helper(monkeypatch) -> None:
    captured = {}

    def _fake_get_document_version(db_instance, media_id: int, version_number=None, include_content: bool = True):
        captured["db_instance"] = db_instance
        captured["media_id"] = media_id
        captured["version_number"] = version_number
        captured["include_content"] = include_content
        return {"media_id": media_id, "version_number": version_number, "content": "latest"}

    sentinel_db = object()
    monkeypatch.setattr(media_db_api, "is_media_database_like", lambda _db: True)
    monkeypatch.setattr(legacy_wrappers, "get_document_version", _fake_get_document_version)
    result = media_db_api.get_document_version(
        sentinel_db,
        media_id=5,
        version_number=None,
        include_content=True,
    )

    assert result == {"media_id": 5, "version_number": None, "content": "latest"}
    assert captured == {
        "db_instance": sentinel_db,
        "media_id": 5,
        "version_number": None,
        "include_content": True,
    }


def test_media_db_api_get_full_media_details_accepts_lightweight_read_double() -> None:
    class StubDetails:
        def get_media_by_id(self, media_id: int, include_deleted: bool = False, include_trash: bool = False):
            return {"id": media_id, "title": "Doc", "content": "hello world"}

        def get_all_document_versions(
            self,
            media_id: int,
            include_content: bool = False,
            include_deleted: bool = False,
            limit=None,
            offset: int | None = 0,
        ):
            return [
                {
                    "media_id": media_id,
                    "version_number": 1,
                    "content": "hello world" if include_content else None,
                    "analysis_content": "summary",
                }
            ]

    details = media_db_api.get_full_media_details(
        StubDetails(),
        media_id=7,
        include_content=True,
    )

    assert details is not None
    assert details["media"]["id"] == 7
    assert details["latest_version"]["version_number"] == 1
    assert details["keywords"] == []


def test_media_db_api_get_full_media_details_rich_accepts_lightweight_read_double() -> None:
    class StubRichDetails:
        def get_media_by_id(self, media_id: int, include_deleted: bool = False, include_trash: bool = False):
            return {
                "id": media_id,
                "title": "Doc",
                "url": "https://example.com/doc",
                "type": "text",
                "content": "hello world",
                "timestamps": ["00:01"],
            }

        def get_all_document_versions(
            self,
            media_id: int,
            include_content: bool = False,
            include_deleted: bool = False,
            limit=None,
            offset: int | None = 0,
        ):
            return [
                {
                    "uuid": "v1",
                    "media_id": media_id,
                    "version_number": 1,
                    "analysis_content": "summary",
                    "prompt": "prompt",
                    "safe_metadata": '{"published_at": "2026-03-18"}',
                    "content": "hello world" if include_content else None,
                }
            ]

        def has_original_file(self, media_id: int) -> bool:
            return media_id == 7

    details = media_db_api.get_full_media_details_rich(
        StubRichDetails(),
        media_id=7,
        include_content=True,
        include_versions=True,
        include_version_content=True,
    )

    assert details is not None
    assert details["media_id"] == 7
    assert details["processing"]["analysis"] == "summary"
    assert details["versions"][0]["content"] == "hello world"
    assert details["has_original_file"] is True


@pytest.mark.parametrize(
    ("file_path", "forbidden_patterns"),
    [
        (
            "tldw_Server_API/app/api/v1/endpoints/media/item.py",
            ("get_full_media_details_rich2(", ".get_media_by_id("),
        ),
        (
            "tldw_Server_API/app/api/v1/endpoints/media/listing.py",
            (".search_media_db(",),
        ),
        (
            "tldw_Server_API/app/api/v1/endpoints/items.py",
            (".search_media_db(",),
        ),
        (
            "tldw_Server_API/app/api/v1/endpoints/outputs_templates.py",
            (".search_media_db(",),
        ),
        (
            "tldw_Server_API/app/api/v1/endpoints/vector_stores_openai.py",
            (".search_media_db(", ".get_media_by_id("),
        ),
        (
            "tldw_Server_API/app/api/v1/endpoints/media_embeddings.py",
            (".get_media_by_id(",),
        ),
        (
            "tldw_Server_API/app/core/RAG/rag_service/database_retrievers.py",
            (".search_media_db(",),
        ),
        (
            "tldw_Server_API/app/core/Chatbooks/chatbook_service.py",
            (".get_media_by_id(",),
        ),
        (
            "tldw_Server_API/app/core/MCP_unified/modules/implementations/media_module.py",
            (".search_media_db(", ".get_media_by_id("),
        ),
        (
            "tldw_Server_API/app/core/Data_Tables/jobs_worker.py",
            (".get_media_by_id(",),
        ),
        (
            "tldw_Server_API/app/core/Embeddings/services/jobs_worker.py",
            (".get_media_by_id(",),
        ),
    ],
)
def test_stage2_task4_production_callers_use_media_db_read_contract_in_source(
    file_path: str,
    forbidden_patterns: tuple[str, ...],
) -> None:
    source = _repo_path(file_path).read_text(encoding="utf-8")
    for pattern in forbidden_patterns:
        assert pattern not in source


def test_db_deps_still_returns_media_db_session_in_source() -> None:
    assert "MediaDbSession" in inspect.getsource(media_db_deps)


def test_media_db_api_no_longer_mentions_media_db_v2_in_source() -> None:
    assert "Media_DB_v2" not in inspect.getsource(media_db_api)


def test_media_versions_imports_rich_details_from_media_db_api() -> None:
    assert media_versions.get_full_media_details_rich is media_db_api.get_full_media_details_rich
    assert not hasattr(media_versions, "get_full_media_details_rich2")
    source = (
        REPO_ROOT
        / "tldw_Server_API/app/api/v1/endpoints/media/versions.py"
    ).read_text(encoding="utf-8")
    assert "get_full_media_details_rich2(" not in source
    assert "get_full_media_details_rich(\n            db_instance=" not in source


def test_shared_workspace_resolver_no_longer_mentions_media_db_v2_in_source() -> None:
    resolver_source = _repo_path(
        "tldw_Server_API/app/core/Sharing/shared_workspace_resolver.py"
    ).read_text(encoding="utf-8")
    assert "Media_DB_v2" not in resolver_source


def test_app_source_only_compat_hosts_mention_media_db_v2() -> None:
    app_root = _repo_path("tldw_Server_API/app")
    matching_files = sorted(
        str(path.relative_to(REPO_ROOT)).replace("\\", "/")
        for path in app_root.rglob("*.py")
        if "Media_DB_v2" in path.read_text(encoding="utf-8")
    )
    assert matching_files == [
        "tldw_Server_API/app/core/DB_Management/media_db/constants.py",
        "tldw_Server_API/app/core/DB_Management/media_db/legacy_identifiers.py",
        "tldw_Server_API/app/core/DB_Management/media_db/runtime/execution_ops.py",
    ]


def test_db_path_utils_no_longer_imports_legacy_identifiers_in_source() -> None:
    assert "legacy_identifiers" not in inspect.getsource(db_path_utils)


def test_media_db_constants_export_canonical_filename() -> None:
    try:
        media_db_constants = importlib.import_module(
            "tldw_Server_API.app.core.DB_Management.media_db.constants"
        )
    except ModuleNotFoundError as exc:  # pragma: no cover - red phase expectation
        pytest.fail(f"media_db.constants module missing: {exc}")

    assert media_db_constants.MEDIA_DB_FILENAME == "Media_DB_v2.db"


@pytest.mark.parametrize(
    ("relative_path", "forbidden_fragment"),
    [
        (
            "tldw_Server_API/app/api/v1/endpoints/media/item.py",
            "tldw_Server_API.app.core.DB_Management.media_db.legacy_content_queries",
        ),
        (
            "tldw_Server_API/app/api/v1/endpoints/media/item.py",
            "tldw_Server_API.app.core.DB_Management.media_db.legacy_maintenance",
        ),
        (
            "tldw_Server_API/app/api/v1/endpoints/media/listing.py",
            "tldw_Server_API.app.core.DB_Management.DB_Manager",
        ),
        (
            "tldw_Server_API/app/api/v1/endpoints/media/listing.py",
            "tldw_Server_API.app.core.DB_Management.media_db.legacy_content_queries",
        ),
        (
            "tldw_Server_API/app/api/v1/endpoints/media/listing.py",
            "tldw_Server_API.app.core.DB_Management.media_db.legacy_maintenance",
        ),
        (
            "tldw_Server_API/app/api/v1/endpoints/media/versions.py",
            "tldw_Server_API.app.core.DB_Management.media_db.legacy_state",
        ),
        (
            "tldw_Server_API/app/api/v1/endpoints/media/versions.py",
            "tldw_Server_API.app.core.DB_Management.media_db.legacy_wrappers",
        ),
        (
            "tldw_Server_API/app/api/v1/endpoints/media/document_insights.py",
            "tldw_Server_API.app.core.DB_Management.media_db.legacy_reads",
        ),
        (
            "tldw_Server_API/app/api/v1/endpoints/media/document_references.py",
            "tldw_Server_API.app.core.DB_Management.media_db.legacy_reads",
        ),
        (
            "tldw_Server_API/app/api/v1/endpoints/media/navigation.py",
            "tldw_Server_API.app.core.DB_Management.media_db.legacy_reads",
        ),
        (
            "tldw_Server_API/app/api/v1/endpoints/media/navigation.py",
            "tldw_Server_API.app.core.DB_Management.media_db.legacy_wrappers",
        ),
        (
            "tldw_Server_API/app/services/quiz_source_resolver.py",
            "tldw_Server_API.app.core.DB_Management.media_db.legacy_reads",
        ),
        (
            "tldw_Server_API/app/services/quiz_source_resolver.py",
            "tldw_Server_API.app.core.DB_Management.media_db.legacy_wrappers",
        ),
        (
            "tldw_Server_API/app/core/Data_Tables/jobs_worker.py",
            "tldw_Server_API.app.core.DB_Management.media_db.legacy_reads",
        ),
        (
            "tldw_Server_API/app/core/Data_Tables/jobs_worker.py",
            "tldw_Server_API.app.core.DB_Management.media_db.legacy_wrappers",
        ),
        (
            "tldw_Server_API/app/core/MCP_unified/modules/implementations/media_module.py",
            "tldw_Server_API.app.core.DB_Management.media_db.legacy_reads",
        ),
        (
            "tldw_Server_API/app/core/MCP_unified/modules/implementations/media_module.py",
            "tldw_Server_API.app.core.DB_Management.media_db.legacy_wrappers",
        ),
        (
            "tldw_Server_API/app/core/MCP_unified/modules/implementations/media_module.py",
            "tldw_Server_API.app.core.DB_Management.media_db.legacy_maintenance",
        ),
        (
            "tldw_Server_API/app/api/v1/endpoints/slides.py",
            "tldw_Server_API.app.core.DB_Management.media_db.legacy_reads",
        ),
        (
            "tldw_Server_API/app/core/Chatbooks/chatbook_service.py",
            "tldw_Server_API.app.core.DB_Management.media_db.legacy_reads",
        ),
        (
            "tldw_Server_API/app/core/Embeddings/services/jobs_worker.py",
            "tldw_Server_API.app.core.DB_Management.media_db.legacy_wrappers",
        ),
        (
            "tldw_Server_API/app/api/v1/endpoints/items.py",
            "tldw_Server_API.app.core.DB_Management.media_db.legacy_wrappers",
        ),
        (
            "tldw_Server_API/app/api/v1/endpoints/items.py",
            "tldw_Server_API.app.core.DB_Management.media_db.legacy_content_queries",
        ),
        (
            "tldw_Server_API/app/api/v1/endpoints/outputs_templates.py",
            "tldw_Server_API.app.core.DB_Management.media_db.legacy_wrappers",
        ),
        (
            "tldw_Server_API/app/api/v1/endpoints/outputs_templates.py",
            "tldw_Server_API.app.core.DB_Management.media_db.legacy_content_queries",
        ),
        (
            "tldw_Server_API/app/api/v1/endpoints/media_embeddings.py",
            "tldw_Server_API.app.core.DB_Management.media_db.legacy_wrappers",
        ),
        (
            "tldw_Server_API/app/core/Ingestion_Media_Processing/Media_Update_lib.py",
            "tldw_Server_API.app.core.DB_Management.media_db.legacy_state",
        ),
        (
            "tldw_Server_API/app/core/Ingestion_Media_Processing/Media_Update_lib.py",
            "tldw_Server_API.app.core.DB_Management.media_db.legacy_wrappers",
        ),
    ],
)
def test_selected_media_endpoint_sources_no_longer_import_compat_helpers(
    relative_path: str,
    forbidden_fragment: str,
) -> None:
    source = _repo_path(relative_path).read_text(encoding="utf-8")
    imported_targets = _import_targets_from_source(source)
    assert not _has_import_target_prefix(imported_targets, forbidden_fragment)


def test_media_db_api_exposes_tranche_facades() -> None:
    expected = [
        "get_paginated_files",
        "get_paginated_trash_files",
        "fetch_keywords_for_media",
        "fetch_keywords_for_media_batch",
        "get_document_version",
        "get_media_prompts",
        "get_media_transcripts",
        "check_media_exists",
        "permanently_delete_item",
        "get_latest_transcription",
    ]
    for name in expected:
        assert callable(getattr(media_db_api, name, None))


def test_media_db_media_database_exports_impl_class() -> None:
    from tldw_Server_API.app.core.DB_Management.media_db import media_database

    assert media_database.MediaDatabase.__module__.endswith("media_database_impl")


def test_media_db_native_class_exports_same_impl_class() -> None:
    from tldw_Server_API.app.core.DB_Management.media_db import media_database
    from tldw_Server_API.app.core.DB_Management.media_db import native_class

    assert native_class.MediaDatabase is media_database.MediaDatabase


@pytest.mark.parametrize(
    "relative_root",
    [
        "tldw_Server_API/tests/Claims",
        "tldw_Server_API/tests/RAG",
        "tldw_Server_API/tests/RAG_NEW",
        "tldw_Server_API/tests/TTS_NEW",
        "tldw_Server_API/tests/DataTables",
        "tldw_Server_API/tests/ChromaDB",
        "tldw_Server_API/tests/MediaDB2",
    ],
)
def test_domain_test_slice_no_longer_imports_media_db_v2(relative_root: str) -> None:
    offenders = []
    slice_root = _repo_path(relative_root)
    legacy_import = "from tldw_Server_API.app.core.DB_Management.Media_DB_v2 import"

    for path in slice_root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if legacy_import in text:
            offenders.append(str(path.relative_to(REPO_ROOT)).replace("\\", "/"))

    assert offenders == []


def test_test_utils_no_longer_imports_media_db_v2() -> None:
    source = _repo_path("tldw_Server_API/tests/test_utils.py").read_text(encoding="utf-8")
    assert "from tldw_Server_API.app.core.DB_Management.Media_DB_v2 import" not in source


def test_tests_conftest_no_longer_mentions_media_db_v2_in_source() -> None:
    assert "Media_DB_v2" not in Path(
        _repo_path("tldw_Server_API/tests/conftest.py")
    ).read_text(encoding="utf-8")


def test_chat_test_fixtures_no_longer_mentions_media_db_v2_in_source() -> None:
    assert "Media_DB_v2" not in Path(
        _repo_path("tldw_Server_API/tests/Chat/test_fixtures.py")
    ).read_text(encoding="utf-8")


def test_sync_coordinator_tests_no_longer_mentions_media_db_v2_in_source() -> None:
    assert "Media_DB_v2" not in Path(
        _repo_path("tldw_Server_API/tests/External_Sources/test_sync_coordinator.py")
    ).read_text(encoding="utf-8")


def test_media_reprocess_tests_no_longer_mentions_media_db_v2_in_source() -> None:
    assert "Media_DB_v2" not in Path(
        _repo_path("tldw_Server_API/tests/Media/test_media_reprocess_endpoint.py")
    ).read_text(encoding="utf-8")


def test_media_repository_no_longer_mentions_media_db_v2_in_source() -> None:
    assert "Media_DB_v2" not in inspect.getsource(media_repository)


def test_chunking_templates_no_longer_mentions_media_db_v2_in_source() -> None:
    assert "Media_DB_v2" not in inspect.getsource(chunking_templates_endpoint)


def test_embeddings_v5_endpoint_no_longer_mentions_media_db_v2_in_source() -> None:
    assert "Media_DB_v2" not in inspect.getsource(embeddings_v5_endpoint)


def test_prompts_db_no_longer_mentions_media_db_v2_in_source() -> None:
    assert "Media_DB_v2" not in inspect.getsource(prompts_db_module)


def test_migrate_to_multiuser_no_longer_mentions_media_db_v2_in_source() -> None:
    assert "Media_DB_v2" not in inspect.getsource(migrate_to_multiuser)


def test_db_path_utils_no_longer_mentions_media_db_v2_in_source() -> None:
    assert "Media_DB_v2" not in inspect.getsource(db_path_utils)


def test_media_db_errors_no_longer_mentions_media_db_v2_in_source() -> None:
    assert "Media_DB_v2" not in inspect.getsource(media_db_errors)


def test_media_db_package_no_longer_mentions_media_db_v2_in_source() -> None:
    assert "Media_DB_v2" not in inspect.getsource(media_db_package)


def test_media_db_runtime_rows_no_longer_mentions_media_db_v2_in_source() -> None:
    assert "Media_DB_v2" not in inspect.getsource(media_db_runtime_rows)


def test_media_db_runtime_execution_no_longer_mentions_media_db_v2_in_source() -> None:
    assert "Media_DB_v2" not in inspect.getsource(media_db_runtime_execution)


def test_media_db_runtime_media_class_no_longer_mentions_media_db_v2_in_source() -> None:
    assert "Media_DB_v2" not in inspect.getsource(media_db_runtime_media_class)


def test_media_db_native_class_no_longer_mentions_media_db_v2_in_source() -> None:
    assert "Media_DB_v2" not in inspect.getsource(media_db_native_class)


def test_media_db_native_class_exports_canonical_media_database() -> None:
    assert media_db_runtime_media_class.load_media_database_cls() is media_db_native_class.MediaDatabase


_MEDIA_DB_V2_DOC_BLOCKER_FRAGMENTS = (
    "tldw_Server_API/app/core/DB_Management/Media_DB_v2.py",
    "/core/DB_Management/Media_DB_v2.py",
    "from tldw_Server_API.app.core.DB_Management.Media_DB_v2 import",
    "Media_DB_v2 system",
    "Media_DB_v2 module",
    "Media_DB_v2 import",
)


def _media_db_active_doc_offenders() -> set[str]:
    """Return active code docs that still read like legacy module guidance."""

    doc_paths = [
        REPO_ROOT / "Docs/Database_Migrations.md",
        REPO_ROOT / "Docs/Code_Documentation/Database.md",
        REPO_ROOT / "Docs/Code_Documentation/index.md",
        REPO_ROOT / "Docs/Code_Documentation/Code_Map.md",
        REPO_ROOT / "Docs/Code_Documentation/Email_Search_Architecture.md",
        REPO_ROOT / "Docs/Code_Documentation/Pieces.md",
        REPO_ROOT / "Docs/Code_Documentation/Claims_Extraction.md",
        REPO_ROOT / "Docs/Code_Documentation/Ingestion_Media_Processing.md",
        REPO_ROOT / "Docs/Code_Documentation/RAG-Developer-Guide.md",
        REPO_ROOT / "Docs/Code_Documentation/Chunking_Templates_Developer_Guide.md",
        REPO_ROOT / "Docs/Code_Documentation/Databases/Media_DB_v2.md",
    ]

    offenders: set[str] = set()
    for path in doc_paths:
        text = path.read_text(encoding="utf-8")
        if any(fragment in text for fragment in _MEDIA_DB_V2_DOC_BLOCKER_FRAGMENTS):
            offenders.add(str(path.relative_to(REPO_ROOT)).replace("\\", "/"))

    return offenders


def test_active_code_docs_no_longer_point_to_media_db_v2_module() -> None:
    assert _media_db_active_doc_offenders() == set()


def _media_db_delete_blockers() -> set[str]:
    """Return the remaining delete blocker inventory."""

    blockers = _media_db_active_doc_offenders()
    legacy_module_path = "tldw_Server_API/app/core/DB_Management/Media_DB_v2.py"
    if (REPO_ROOT / legacy_module_path).exists():
        blockers.add(legacy_module_path)
    return blockers


def test_media_db_delete_blockers_match_known_inventory() -> None:
    expected = set()
    assert _media_db_active_doc_offenders() == set()
    assert _media_db_delete_blockers() == expected


def test_legacy_media_db_module_is_deleted() -> None:
    module_name = "tldw_Server_API.app.core.DB_Management.Media_DB_v2"
    sys.modules.pop(module_name, None)
    legacy_module_path = (
        Path(__file__).resolve().parents[2]
        / "app/core/DB_Management/Media_DB_v2.py"
    )
    assert not legacy_module_path.exists()
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module_name)


def test_media_database_impl_source_no_longer_imports_media_db_v2() -> None:
    source = (
        Path(__file__).resolve().parents[2]
        / "app/core/DB_Management/media_db/media_database_impl.py"
    ).read_text(encoding="utf-8")

    assert "Media_DB_v2 as _legacy_media_db" not in source


def test_media_database_impl_source_no_longer_clones_legacy_media_database() -> None:
    source = (
        Path(__file__).resolve().parents[2]
        / "app/core/DB_Management/media_db/media_database_impl.py"
    ).read_text(encoding="utf-8")

    assert "_clone_legacy_media_database" not in source


def test_media_db_package_has_no_internal_media_db_v2_imports() -> None:
    import re

    root = Path(__file__).resolve().parents[2] / "app/core/DB_Management/media_db"
    pattern = re.compile(
        r"from\s+tldw_Server_API\.app\.core\.DB_Management\s+import\s+Media_DB_v2\b"
    )

    offenders = []
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if pattern.search(text):
            offenders.append(str(path))

    assert offenders == []


def test_active_tests_no_longer_import_media_db_v2() -> None:
    tests_root = Path(__file__).resolve().parents[1]
    compatibility_tests = {
        tests_root / "DB_Management/test_media_db_api_imports.py",
        tests_root / "DB_Management/test_media_db_v2_regressions.py",
    }

    offenders = []
    for path in tests_root.rglob("*.py"):
        if path in compatibility_tests:
            continue
        import_targets = _import_targets_from_source(path.read_text(encoding="utf-8"))
        if _has_import_target_prefix(
            import_targets,
            "tldw_Server_API.app.core.DB_Management.Media_DB_v2",
        ):
            offenders.append(str(path))

    assert offenders == []


def test_media_db_legacy_identifiers_owns_media_db_v2_reference() -> None:
    assert "Media_DB_v2" in inspect.getsource(media_db_legacy_identifiers)


def test_meetings_webhook_dlq_service_no_longer_mentions_media_db_v2_in_source() -> None:
    assert "Media_DB_v2" not in inspect.getsource(meetings_webhook_dlq_service)


def test_db_deps_imports_errors_from_media_db_errors_and_not_media_db_v2(monkeypatch):
    monkeypatch.setattr(legacy_media_db, "DatabaseError", object(), raising=False)
    monkeypatch.setattr(legacy_media_db, "SchemaError", object(), raising=False)
    monkeypatch.setattr(legacy_media_db, "MediaDatabase", object(), raising=False)

    module = importlib.reload(media_db_deps)

    assert module.DatabaseError is media_db_errors.DatabaseError
    assert module.SchemaError is media_db_errors.SchemaError
    assert module.MediaDatabase is not legacy_media_db.MediaDatabase


def test_http_errors_imports_db_errors_from_media_db_errors(monkeypatch):
    monkeypatch.setattr(legacy_media_db, "ConflictError", object(), raising=False)
    monkeypatch.setattr(legacy_media_db, "DatabaseError", object(), raising=False)
    monkeypatch.setattr(legacy_media_db, "InputError", object(), raising=False)
    monkeypatch.setattr(legacy_media_db, "SchemaError", object(), raising=False)

    module = importlib.reload(http_errors)

    assert module.ConflictError is media_db_errors.ConflictError
    assert module.DatabaseError is media_db_errors.DatabaseError
    assert module.InputError is media_db_errors.InputError
    assert module.SchemaError is media_db_errors.SchemaError


def test_document_references_does_not_bind_media_database_from_media_db_v2(monkeypatch):
    monkeypatch.setattr(legacy_media_db, "MediaDatabase", object(), raising=False)

    module = importlib.reload(document_references)

    assert "MediaDatabase" not in module.__dict__
    assert module.get_latest_transcription is media_db_api.get_latest_transcription


def test_document_insights_does_not_bind_media_database_from_media_db_v2(monkeypatch):
    monkeypatch.setattr(legacy_media_db, "MediaDatabase", object(), raising=False)

    module = importlib.reload(document_insights)

    assert "MediaDatabase" not in module.__dict__
    assert module.get_latest_transcription is media_db_api.get_latest_transcription


def test_navigation_endpoint_does_not_bind_media_database_from_media_db_v2(monkeypatch):
    monkeypatch.setattr(legacy_media_db, "MediaDatabase", object(), raising=False)

    module = importlib.reload(media_navigation)

    assert "MediaDatabase" not in module.__dict__


def test_document_outline_does_not_bind_media_database_from_media_db_v2(monkeypatch):
    monkeypatch.setattr(legacy_media_db, "MediaDatabase", object(), raising=False)

    module = importlib.reload(document_outline)

    assert "MediaDatabase" not in module.__dict__


def test_process_documents_does_not_bind_media_database_from_media_db_v2(monkeypatch):
    monkeypatch.setattr(legacy_media_db, "MediaDatabase", object(), raising=False)

    module = importlib.reload(process_documents)

    assert "MediaDatabase" not in module.__dict__


def test_process_pdfs_does_not_bind_media_database_from_media_db_v2(monkeypatch):
    monkeypatch.setattr(legacy_media_db, "MediaDatabase", object(), raising=False)

    module = importlib.reload(process_pdfs)

    assert "MediaDatabase" not in module.__dict__


def test_reading_progress_does_not_bind_media_database_from_media_db_v2(monkeypatch):
    monkeypatch.setattr(legacy_media_db, "MediaDatabase", object(), raising=False)
    module_name = "tldw_Server_API.app.api.v1.endpoints.media.reading_progress"
    sys.modules.pop(module_name, None)

    module = importlib.import_module(module_name)

    assert "MediaDatabase" not in module.__dict__


def test_media_item_imports_db_errors_from_media_db_errors_and_not_media_db_v2(monkeypatch):
    monkeypatch.setattr(legacy_media_db, "ConflictError", object(), raising=False)
    monkeypatch.setattr(legacy_media_db, "DatabaseError", object(), raising=False)
    monkeypatch.setattr(legacy_media_db, "InputError", object(), raising=False)
    monkeypatch.setattr(legacy_media_db, "MediaDatabase", object(), raising=False)

    module = importlib.reload(media_item)

    assert module.ConflictError is media_db_errors.ConflictError
    assert module.DatabaseError is media_db_errors.DatabaseError
    assert module.InputError is media_db_errors.InputError
    assert "MediaDatabase" not in module.__dict__


def test_media_versions_imports_errors_and_state_helpers_outside_media_db_v2(monkeypatch):
    monkeypatch.setattr(legacy_media_db, "ConflictError", object(), raising=False)
    monkeypatch.setattr(legacy_media_db, "DatabaseError", object(), raising=False)
    monkeypatch.setattr(legacy_media_db, "InputError", object(), raising=False)
    monkeypatch.setattr(legacy_media_db, "MediaDatabase", object(), raising=False)
    monkeypatch.setattr(legacy_media_db, "check_media_exists", object(), raising=False)
    monkeypatch.setattr(legacy_media_db, "get_document_version", object(), raising=False)

    module = importlib.reload(media_versions)

    assert module.ConflictError is media_db_errors.ConflictError
    assert module.DatabaseError is media_db_errors.DatabaseError
    assert module.InputError is media_db_errors.InputError
    assert module.check_media_exists is media_db_api.check_media_exists
    assert module.get_document_version is media_db_api.get_document_version
    assert "MediaDatabase" not in module.__dict__


def test_items_imports_db_errors_from_media_db_errors(monkeypatch):
    monkeypatch.setattr(legacy_media_db, "DatabaseError", object(), raising=False)
    monkeypatch.setattr(legacy_media_db, "InputError", object(), raising=False)

    module = importlib.reload(items)

    assert module.DatabaseError is media_db_errors.DatabaseError
    assert module.InputError is media_db_errors.InputError


def test_data_tables_imports_input_error_from_media_db_errors_and_not_media_db_v2(monkeypatch):
    monkeypatch.setattr(legacy_media_db, "InputError", object(), raising=False)
    monkeypatch.setattr(legacy_media_db, "MediaDatabase", object(), raising=False)

    module = importlib.reload(data_tables)

    assert module.InputError is media_db_errors.InputError
    assert "MediaDatabase" not in module.__dict__


def test_sync_imports_db_errors_from_media_db_errors_and_not_media_db_v2(monkeypatch):
    monkeypatch.setattr(legacy_media_db, "ConflictError", object(), raising=False)
    monkeypatch.setattr(legacy_media_db, "DatabaseError", object(), raising=False)
    monkeypatch.setattr(legacy_media_db, "InputError", object(), raising=False)
    monkeypatch.setattr(legacy_media_db, "MediaDatabase", object(), raising=False)

    module = importlib.reload(sync)

    assert module.ConflictError is media_db_errors.ConflictError
    assert module.DatabaseError is media_db_errors.DatabaseError
    assert module.InputError is media_db_errors.InputError
    assert "MediaDatabase" not in module.__dict__


def test_db_manager_does_not_bind_detail_helpers_from_media_db_v2(monkeypatch):
    sentinel_details = object()
    sentinel_rich = object()
    sentinel_backup = object()

    monkeypatch.setattr(legacy_media_db, "get_full_media_details", sentinel_details, raising=False)
    monkeypatch.setattr(legacy_media_db, "get_full_media_details_rich", sentinel_rich, raising=False)
    monkeypatch.setattr(legacy_media_db, "create_automated_backup", sentinel_backup, raising=False)

    module = importlib.reload(DB_Manager)

    assert module.media_db_api.get_full_media_details is media_db_api.get_full_media_details
    assert module.media_db_api.get_full_media_details_rich is media_db_api.get_full_media_details_rich
    assert module.sqlite_create_automated_backup is media_db_legacy_backup.create_automated_backup


def test_db_manager_does_not_bind_media_database_from_media_db_v2(monkeypatch):
    monkeypatch.setattr(legacy_media_db, "MediaDatabase", object(), raising=False)

    module = importlib.reload(DB_Manager)

    assert "MediaDatabase" not in module.__dict__


def test_db_manager_no_longer_mentions_media_db_v2_in_source() -> None:
    assert "Media_DB_v2" not in inspect.getsource(DB_Manager)


def test_claim_review_assignment_does_not_bind_media_database_from_media_db_v2(monkeypatch):
    monkeypatch.setattr(legacy_media_db, "MediaDatabase", object(), raising=False)

    module = importlib.reload(review_assignment)

    assert "MediaDatabase" not in module.__dict__


def test_sync_coordinator_does_not_bind_media_database_from_media_db_v2(monkeypatch):
    monkeypatch.setattr(legacy_media_db, "MediaDatabase", object(), raising=False)

    module = importlib.reload(sync_coordinator)

    assert "MediaDatabase" not in module.__dict__


def test_quiz_generator_does_not_bind_media_database_from_media_db_v2(monkeypatch):
    monkeypatch.setattr(legacy_media_db, "MediaDatabase", object(), raising=False)

    module = importlib.reload(quiz_generator)

    assert "MediaDatabase" not in module.__dict__


def test_quiz_source_resolver_does_not_bind_media_database_from_media_db_v2(monkeypatch):
    monkeypatch.setattr(legacy_media_db, "MediaDatabase", object(), raising=False)

    module = importlib.reload(quiz_source_resolver)

    assert "MediaDatabase" not in module.__dict__


def test_document_processing_service_imports_managed_media_database_from_media_db_api():
    module = importlib.reload(document_processing_service)
    assert module.managed_media_database is media_db_api.managed_media_database


def test_media_ingest_jobs_worker_imports_create_media_database_from_media_db_api():
    module = importlib.reload(media_ingest_jobs_worker)
    assert module.create_media_database is media_db_api.create_media_database


def test_enhanced_web_scraping_service_imports_managed_media_database_from_media_db_api():
    module = importlib.reload(enhanced_web_scraping_service)
    assert module.managed_media_database is media_db_api.managed_media_database


def test_research_endpoint_imports_managed_media_database_from_media_db_api():
    module = importlib.reload(research)
    assert module.managed_media_database is media_db_api.managed_media_database


def test_research_endpoint_does_not_bind_media_database_from_media_db_v2(monkeypatch):
    monkeypatch.setattr(legacy_media_db, "MediaDatabase", object(), raising=False)

    module = importlib.reload(research)

    assert "MediaDatabase" not in module.__dict__


def test_rag_unified_endpoint_does_not_bind_media_database_from_media_db_v2(monkeypatch):
    monkeypatch.setattr(legacy_media_db, "MediaDatabase", object(), raising=False)

    module = importlib.reload(rag_unified_endpoint)

    assert "MediaDatabase" not in module.__dict__


def test_slides_endpoint_does_not_bind_media_database_from_media_db_v2(monkeypatch):
    monkeypatch.setattr(legacy_media_db, "MediaDatabase", object(), raising=False)

    module = importlib.reload(slides_endpoint)

    assert "MediaDatabase" not in module.__dict__


def test_quizzes_endpoint_does_not_bind_media_database_from_media_db_v2(monkeypatch):
    monkeypatch.setattr(legacy_media_db, "MediaDatabase", object(), raising=False)

    module = importlib.reload(quizzes_endpoint)

    assert "MediaDatabase" not in module.__dict__


def test_claims_endpoint_does_not_bind_media_database_from_media_db_v2(monkeypatch):
    monkeypatch.setattr(legacy_media_db, "MediaDatabase", object(), raising=False)

    module = importlib.reload(claims_endpoint)

    assert "MediaDatabase" not in module.__dict__


def test_chunking_endpoint_does_not_bind_media_database_from_media_db_v2(monkeypatch):
    monkeypatch.setattr(legacy_media_db, "MediaDatabase", object(), raising=False)

    module = importlib.reload(chunking_endpoint)

    assert "MediaDatabase" not in module.__dict__


def test_vector_stores_endpoint_does_not_bind_media_database_from_media_db_v2(monkeypatch):
    monkeypatch.setattr(legacy_media_db, "MediaDatabase", object(), raising=False)

    module = importlib.reload(vector_stores_endpoint)

    assert "MediaDatabase" not in module.__dict__


def test_text2sql_endpoint_does_not_bind_media_database_from_media_db_v2(monkeypatch):
    monkeypatch.setattr(legacy_media_db, "MediaDatabase", object(), raising=False)

    module = importlib.reload(text2sql_endpoint)

    assert "MediaDatabase" not in module.__dict__


def test_audio_history_endpoint_does_not_bind_media_database_from_media_db_v2(monkeypatch):
    monkeypatch.setattr(legacy_media_db, "MediaDatabase", object(), raising=False)

    module = importlib.reload(audio_history_endpoint)

    assert "MediaDatabase" not in module.__dict__


def test_audio_tts_endpoint_does_not_bind_media_database_from_media_db_v2(monkeypatch):
    monkeypatch.setattr(legacy_media_db, "MediaDatabase", object(), raising=False)

    module = importlib.reload(audio_tts_endpoint)

    assert "MediaDatabase" not in module.__dict__


def test_media_embeddings_endpoint_does_not_bind_media_database_from_media_db_v2(monkeypatch):
    monkeypatch.setattr(legacy_media_db, "MediaDatabase", object(), raising=False)

    module = importlib.reload(media_embeddings_endpoint)

    assert "MediaDatabase" not in module.__dict__


def test_media_debug_endpoint_does_not_bind_media_database_from_media_db_v2(monkeypatch):
    monkeypatch.setattr(legacy_media_db, "MediaDatabase", object(), raising=False)

    module = importlib.reload(media_debug_endpoint)

    assert "MediaDatabase" not in module.__dict__


def test_media_add_endpoint_does_not_bind_media_database_from_media_db_v2(monkeypatch):
    monkeypatch.setattr(legacy_media_db, "MediaDatabase", object(), raising=False)

    module = importlib.reload(media_add_endpoint)

    assert "MediaDatabase" not in module.__dict__


def test_media_file_endpoint_does_not_bind_media_database_from_media_db_v2(monkeypatch):
    monkeypatch.setattr(legacy_media_db, "MediaDatabase", object(), raising=False)

    module = importlib.reload(media_file_endpoint)

    assert "MediaDatabase" not in module.__dict__


def test_media_ingest_web_content_endpoint_does_not_bind_media_database_from_media_db_v2(monkeypatch):
    monkeypatch.setattr(legacy_media_db, "MediaDatabase", object(), raising=False)

    module = importlib.reload(media_ingest_web_content_endpoint)

    assert "MediaDatabase" not in module.__dict__


def test_media_process_ebooks_endpoint_does_not_bind_media_database_from_media_db_v2(monkeypatch):
    monkeypatch.setattr(legacy_media_db, "MediaDatabase", object(), raising=False)

    module = importlib.reload(media_process_ebooks_endpoint)

    assert "MediaDatabase" not in module.__dict__


def test_media_process_web_scraping_endpoint_does_not_bind_media_database_from_media_db_v2(monkeypatch):
    monkeypatch.setattr(legacy_media_db, "MediaDatabase", object(), raising=False)

    module = importlib.reload(media_process_web_scraping_endpoint)

    assert "MediaDatabase" not in module.__dict__


def test_media_process_videos_endpoint_does_not_bind_media_database_from_media_db_v2(monkeypatch):
    monkeypatch.setattr(legacy_media_db, "MediaDatabase", object(), raising=False)

    module = importlib.reload(media_process_videos_endpoint)

    assert "MediaDatabase" not in module.__dict__


def test_media_process_audios_endpoint_does_not_bind_media_database_from_media_db_v2(monkeypatch):
    monkeypatch.setattr(legacy_media_db, "MediaDatabase", object(), raising=False)

    module = importlib.reload(media_process_audios_endpoint)

    assert "MediaDatabase" not in module.__dict__


def test_media_process_emails_endpoint_does_not_bind_media_database_from_media_db_v2(monkeypatch):
    monkeypatch.setattr(legacy_media_db, "MediaDatabase", object(), raising=False)

    module = importlib.reload(media_process_emails_endpoint)

    assert "MediaDatabase" not in module.__dict__


def test_media_process_code_endpoint_does_not_bind_media_database_from_media_db_v2(monkeypatch):
    monkeypatch.setattr(legacy_media_db, "MediaDatabase", object(), raising=False)

    module = importlib.reload(media_process_code_endpoint)

    assert "MediaDatabase" not in module.__dict__


def test_media_reprocess_endpoint_does_not_bind_media_database_from_media_db_v2(monkeypatch):
    monkeypatch.setattr(legacy_media_db, "MediaDatabase", object(), raising=False)

    module = importlib.reload(media_reprocess_endpoint)

    assert "MediaDatabase" not in module.__dict__


def test_media_reprocess_endpoint_imports_db_errors_from_media_db_errors(monkeypatch):
    monkeypatch.setattr(legacy_media_db, "ConflictError", object(), raising=False)
    monkeypatch.setattr(legacy_media_db, "DatabaseError", object(), raising=False)
    monkeypatch.setattr(legacy_media_db, "InputError", object(), raising=False)

    module = importlib.reload(media_reprocess_endpoint)

    assert module.ConflictError is media_db_errors.ConflictError
    assert module.DatabaseError is media_db_errors.DatabaseError
    assert module.InputError is media_db_errors.InputError


def test_media_document_figures_endpoint_does_not_bind_media_database_from_media_db_v2(monkeypatch):
    monkeypatch.setattr(legacy_media_db, "MediaDatabase", object(), raising=False)

    module = importlib.reload(media_document_figures_endpoint)

    assert "MediaDatabase" not in module.__dict__


def test_audiobooks_endpoint_does_not_bind_media_database_from_media_db_v2(monkeypatch):
    monkeypatch.setattr(legacy_media_db, "MediaDatabase", object(), raising=False)

    module = importlib.reload(audiobooks_endpoint)

    assert "MediaDatabase" not in module.__dict__


def test_email_endpoint_does_not_bind_media_database_from_media_db_v2(monkeypatch):
    monkeypatch.setattr(legacy_media_db, "MediaDatabase", object(), raising=False)

    module = importlib.reload(email_endpoint)

    assert "MediaDatabase" not in module.__dict__


def test_email_endpoint_imports_db_errors_from_media_db_errors(monkeypatch):
    monkeypatch.setattr(legacy_media_db, "DatabaseError", object(), raising=False)
    monkeypatch.setattr(legacy_media_db, "InputError", object(), raising=False)

    module = importlib.reload(email_endpoint)

    assert module.DatabaseError is media_db_errors.DatabaseError
    assert module.InputError is media_db_errors.InputError


def test_media_listing_endpoint_does_not_bind_media_database_from_media_db_v2(monkeypatch):
    monkeypatch.setattr(legacy_media_db, "MediaDatabase", object(), raising=False)

    module = importlib.reload(media_listing_endpoint)

    assert "MediaDatabase" not in module.__dict__


def test_media_listing_endpoint_imports_db_errors_from_media_db_errors(monkeypatch):
    monkeypatch.setattr(legacy_media_db, "DatabaseError", object(), raising=False)
    monkeypatch.setattr(legacy_media_db, "InputError", object(), raising=False)

    module = importlib.reload(media_listing_endpoint)

    assert module.DatabaseError is media_db_errors.DatabaseError
    assert module.InputError is media_db_errors.InputError


def test_chunking_templates_endpoint_does_not_bind_media_database_from_media_db_v2(monkeypatch):
    monkeypatch.setattr(legacy_media_db, "MediaDatabase", object(), raising=False)

    module = importlib.reload(chunking_templates_endpoint)

    assert "MediaDatabase" not in module.__dict__


def test_document_annotations_endpoint_does_not_bind_media_database_from_media_db_v2(monkeypatch):
    monkeypatch.setattr(legacy_media_db, "MediaDatabase", object(), raising=False)

    module = importlib.reload(media_document_annotations_endpoint)

    assert "MediaDatabase" not in module.__dict__


def test_paper_search_endpoint_does_not_bind_media_database_from_media_db_v2(monkeypatch):
    monkeypatch.setattr(legacy_media_db, "MediaDatabase", object(), raising=False)

    module = importlib.reload(paper_search_endpoint)

    assert "MediaDatabase" not in module.__dict__


def test_connectors_worker_imports_create_media_database_from_media_db_api():
    module = importlib.reload(connectors_worker)
    assert module.create_media_database is media_db_api.create_media_database


def test_xml_ingestion_imports_managed_media_database_from_media_db_api():
    module = importlib.reload(XML_Ingestion_Lib)
    assert module.managed_media_database is media_db_api.managed_media_database


def test_xml_ingestion_lib_no_longer_imports_add_media_with_keywords_from_db_manager() -> None:
    source = _repo_path(
        "tldw_Server_API/app/core/Ingestion_Media_Processing/XML_Ingestion_Lib.py"
    ).read_text(encoding="utf-8")
    assert (
        "from tldw_Server_API.app.core.DB_Management.DB_Manager import add_media_with_keywords"
        not in source
    )


def test_ingestion_sources_worker_imports_create_media_database_from_media_db_api():
    module = importlib.reload(ingestion_sources_worker)
    assert module.create_media_database is media_db_api.create_media_database


def test_ingestion_sources_worker_imports_media_db_error_from_media_db_errors(monkeypatch):
    monkeypatch.setattr(legacy_media_db, "DatabaseError", object(), raising=False)

    module = importlib.reload(ingestion_sources_worker)

    assert module.MediaDatabaseError is media_db_errors.DatabaseError


def test_mediawiki_imports_managed_media_database_from_media_db_api():
    module = importlib.reload(Media_Wiki)
    assert module.managed_media_database is media_db_api.managed_media_database


def test_article_extractor_imports_managed_media_database_from_media_db_api():
    module = importlib.reload(Article_Extractor_Lib)
    assert module.managed_media_database is media_db_api.managed_media_database


def test_book_processing_imports_managed_media_database_from_media_db_api():
    module = importlib.reload(Book_Processing_Lib)
    assert module.managed_media_database is media_db_api.managed_media_database


def test_book_processing_lib_no_longer_imports_add_media_with_keywords_from_db_manager() -> None:
    source = _repo_path(
        "tldw_Server_API/app/core/Ingestion_Media_Processing/Books/Book_Processing_Lib.py"
    ).read_text(encoding="utf-8")
    assert (
        "from tldw_Server_API.app.core.DB_Management.DB_Manager import add_media_with_keywords"
        not in source
    )


def test_web_scraping_service_imports_managed_media_database_from_media_db_api():
    module = importlib.reload(web_scraping_service)
    assert module.managed_media_database is media_db_api.managed_media_database


def test_media_files_cleanup_service_imports_managed_media_database_from_media_db_api():
    module = importlib.reload(media_files_cleanup_service)
    assert module.managed_media_database is media_db_api.managed_media_database


def test_storage_cleanup_service_imports_managed_media_database_from_media_db_api():
    module = importlib.reload(storage_cleanup_service)
    assert module.managed_media_database is media_db_api.managed_media_database


def test_outputs_purge_scheduler_imports_managed_media_database_from_media_db_api():
    module = importlib.reload(outputs_purge_scheduler)
    assert module.managed_media_database is media_db_api.managed_media_database


def test_tts_history_cleanup_service_imports_create_media_database_from_media_db_api():
    module = importlib.reload(tts_history_cleanup_service)
    assert module.create_media_database is media_db_api.create_media_database
    assert "Media_DB_v2" not in inspect.getsource(module)


def test_audiobook_jobs_worker_imports_managed_media_database_from_media_db_api():
    module = importlib.reload(audiobook_jobs_worker)
    assert module.managed_media_database is media_db_api.managed_media_database


def test_slides_module_imports_managed_media_database_from_media_db_api():
    module = importlib.reload(slides_module)
    assert module.managed_media_database is media_db_api.managed_media_database


def test_quizzes_module_imports_managed_media_database_from_media_db_api():
    module = importlib.reload(quizzes_module)
    assert module.managed_media_database is media_db_api.managed_media_database


def test_data_tables_jobs_worker_imports_create_media_database_from_media_db_api():
    module = importlib.reload(data_tables_jobs_worker)
    assert module.create_media_database is media_db_api.create_media_database


def test_data_tables_jobs_worker_does_not_bind_media_database_from_media_db_v2(
    monkeypatch,
):
    monkeypatch.setattr(legacy_media_db, "MediaDatabase", object(), raising=False)

    module = importlib.reload(data_tables_jobs_worker)

    assert "MediaDatabase" not in module.__dict__


def test_claims_notifications_imports_managed_media_database_from_media_db_api(monkeypatch):
    monkeypatch.setattr(legacy_media_db, "MediaDatabase", object(), raising=False)
    module = importlib.reload(claims_notifications)
    assert module.managed_media_database is media_db_api.managed_media_database
    assert "MediaDatabase" not in module.__dict__


def test_claims_service_imports_managed_media_database_from_media_db_api():
    module = importlib.reload(claims_service)
    assert module.managed_media_database is media_db_api.managed_media_database


def test_ingestion_claims_does_not_bind_media_database_from_media_db_v2(monkeypatch):
    monkeypatch.setattr(legacy_media_db, "MediaDatabase", object(), raising=False)

    module = importlib.reload(ingestion_claims)

    assert "MediaDatabase" not in module.__dict__


def test_claims_rebuild_service_imports_managed_media_database_from_media_db_api():
    module = importlib.reload(claims_rebuild_service)
    assert module.managed_media_database is media_db_api.managed_media_database


def test_embeddings_jobs_worker_imports_managed_media_database_from_media_db_api():
    module = importlib.reload(embeddings_jobs_worker)
    assert module.managed_media_database is media_db_api.managed_media_database


def test_vector_compactor_imports_managed_media_database_from_media_db_api():
    module = importlib.reload(vector_compactor)
    assert module.managed_media_database is media_db_api.managed_media_database


def test_chromadb_library_imports_managed_media_database_from_media_db_api():
    module = importlib.reload(ChromaDB_Library)
    assert module.managed_media_database is media_db_api.managed_media_database


def test_template_initialization_imports_managed_media_database_from_media_db_api(monkeypatch):
    monkeypatch.setattr(legacy_media_db, "MediaDatabase", object(), raising=False)
    module = importlib.reload(template_initialization)
    assert module.managed_media_database is media_db_api.managed_media_database
    assert "MediaDatabase" not in module.__dict__


def test_watchlists_pipeline_imports_managed_media_database_from_media_db_api():
    module = importlib.reload(watchlists_pipeline)
    assert module.managed_media_database is media_db_api.managed_media_database


def test_workflow_knowledge_crud_imports_managed_media_database_from_media_db_api():
    module = importlib.reload(knowledge_crud)
    assert module.managed_media_database is media_db_api.managed_media_database


def test_unified_pipeline_imports_managed_media_database_from_media_db_api():
    module = importlib.reload(unified_pipeline)
    assert module.managed_media_database is media_db_api.managed_media_database


def test_agentic_chunker_imports_create_media_database_from_media_db_api(monkeypatch):
    monkeypatch.setattr(legacy_media_db, "MediaDatabase", object(), raising=False)
    module = importlib.reload(agentic_chunker)
    assert module.create_media_database is media_db_api.create_media_database


def test_database_retrievers_import_media_db_factory_and_errors_outside_shim(monkeypatch):
    monkeypatch.setattr(legacy_media_db, "MediaDatabase", object(), raising=False)
    monkeypatch.setattr(legacy_media_db, "DatabaseError", object(), raising=False)

    module = importlib.reload(database_retrievers)

    assert module.create_media_database is media_db_api.create_media_database
    assert module.MediaDatabaseError is media_db_errors.DatabaseError


def test_metadata_utils_imports_database_error_from_media_db_errors():
    source = inspect.getsource(metadata_utils)

    assert "core.DB_Management.Media_DB_v2" not in source
    assert "core.DB_Management.media_db.errors" in source


def test_workflow_media_ingest_imports_managed_media_database_from_media_db_api():
    module = importlib.reload(workflow_media_ingest)
    assert module.managed_media_database is media_db_api.managed_media_database


def test_visual_ingestion_imports_managed_media_database_from_media_db_api():
    module = importlib.reload(visual_ingestion)
    assert module.managed_media_database is media_db_api.managed_media_database


def test_claims_utils_imports_managed_media_database_from_media_db_api():
    module = importlib.reload(claims_utils)
    assert module.managed_media_database is media_db_api.managed_media_database


def test_claims_alerts_scheduler_imports_managed_media_database_from_media_db_api():
    module = importlib.reload(claims_alerts_scheduler)
    assert module.managed_media_database is media_db_api.managed_media_database


def test_claims_review_metrics_scheduler_imports_managed_media_database_from_media_db_api(monkeypatch):
    monkeypatch.setattr(legacy_media_db, "MediaDatabase", object(), raising=False)
    module = importlib.reload(claims_review_metrics_scheduler)
    assert module.managed_media_database is media_db_api.managed_media_database
    assert "MediaDatabase" not in module.__dict__


def test_media_endpoint_package_does_not_bind_media_database_from_shim(monkeypatch):
    monkeypatch.setattr(legacy_media_db, "MediaDatabase", object(), raising=False)
    module = importlib.reload(
        importlib.import_module("tldw_Server_API.app.api.v1.endpoints.media")
    )
    assert "MediaDatabase" not in module.__dict__


def test_chatbook_service_imports_media_factory_and_legacy_reads(monkeypatch):
    monkeypatch.setattr(legacy_media_db, "MediaDatabase", object(), raising=False)
    module = importlib.reload(chatbook_service)
    assert module.create_media_database is media_db_api.create_media_database
    assert module.get_media_prompts is media_db_api.get_media_prompts
    assert module.get_media_transcripts is media_db_api.get_media_transcripts
    assert "MediaDatabase" not in module.__dict__


def test_sync_client_imports_factory_and_db_errors_outside_shim(monkeypatch):
    monkeypatch.setattr(legacy_media_db, "ConflictError", object(), raising=False)
    monkeypatch.setattr(legacy_media_db, "DatabaseError", object(), raising=False)
    monkeypatch.setattr(legacy_media_db, "InputError", object(), raising=False)
    monkeypatch.setattr(legacy_media_db, "MediaDatabase", object(), raising=False)
    module = importlib.reload(sync_client_module)
    assert module.create_media_database is media_db_api.create_media_database
    assert module.ConflictError is media_db_errors.ConflictError
    assert module.DatabaseError is media_db_errors.DatabaseError
    assert module.InputError is media_db_errors.InputError
    assert "MediaDatabase" not in module.__dict__
    assert "Media_DB_v2" not in inspect.getsource(module)


def test_claims_service_does_not_bind_media_database_from_shim(monkeypatch):
    monkeypatch.setattr(legacy_media_db, "MediaDatabase", object(), raising=False)
    module = importlib.reload(claims_service)
    assert module.managed_media_database is media_db_api.managed_media_database
    assert "MediaDatabase" not in module.__dict__


def test_media_module_imports_create_media_database_from_media_db_api(monkeypatch):
    monkeypatch.setattr(legacy_media_db, "MediaDatabase", object(), raising=False)
    module = importlib.reload(media_module_impl)
    assert module.create_media_database is media_db_api.create_media_database
    assert "MediaDatabase" not in module.__dict__
    assert "Media_DB_v2" not in inspect.getsource(module)


def test_persistence_imports_factory_and_db_errors_outside_shim(monkeypatch):
    monkeypatch.setattr(legacy_media_db, "MediaDatabase", object(), raising=False)
    monkeypatch.setattr(legacy_media_db, "ConflictError", object(), raising=False)
    monkeypatch.setattr(legacy_media_db, "DatabaseError", object(), raising=False)
    monkeypatch.setattr(legacy_media_db, "InputError", object(), raising=False)

    module = importlib.reload(ingestion_persistence)

    assert module.create_media_database is media_db_api.create_media_database
    assert module.ConflictError is media_db_errors.ConflictError
    assert module.DatabaseError is media_db_errors.DatabaseError
    assert module.InputError is media_db_errors.InputError
    assert "MediaDatabase" not in module.__dict__
    assert "Media_DB_v2" not in inspect.getsource(module)


def test_legacy_state_uses_runtime_validation_without_shim_source() -> None:
    assert "Media_DB_v2" not in inspect.getsource(media_db_state)


def test_legacy_reads_uses_runtime_validation_without_shim_source() -> None:
    assert "Media_DB_v2" not in inspect.getsource(media_db_legacy_reads)


def test_legacy_wrappers_uses_runtime_validation_without_shim_source() -> None:
    assert "Media_DB_v2" not in inspect.getsource(legacy_wrappers)


def test_legacy_document_artifacts_uses_runtime_validation_without_shim_source() -> None:
    assert "Media_DB_v2" not in inspect.getsource(legacy_document_artifacts)


def test_legacy_content_queries_uses_runtime_validation_without_shim_source() -> None:
    assert "Media_DB_v2" not in inspect.getsource(legacy_content_queries)


def test_legacy_media_details_has_no_remaining_noncompat_imports() -> None:
    offenders: list[str] = []
    for path in _repo_path("tldw_Server_API/app").rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        if "legacy_media_details" in source:
            offenders.append(str(path))

    assert offenders == []


def test_media_db_v2_no_longer_mentions_legacy_media_details_in_source() -> None:
    assert "legacy_media_details" not in inspect.getsource(media_db_native_class)


def test_legacy_transcripts_uses_runtime_validation_without_shim_source() -> None:
    assert "Media_DB_v2" not in inspect.getsource(legacy_transcripts)


def test_legacy_maintenance_uses_runtime_validation_without_shim_source() -> None:
    assert "Media_DB_v2" not in inspect.getsource(legacy_maintenance)


def test_legacy_backup_no_longer_mentions_media_db_v2_in_source() -> None:
    assert "Media_DB_v2" not in inspect.getsource(media_db_legacy_backup)


def test_chunks_repository_no_longer_mentions_media_db_v2_in_source() -> None:
    assert "Media_DB_v2" not in inspect.getsource(chunks_repository)


def test_document_versions_repository_no_longer_mentions_media_db_v2_in_source() -> None:
    assert "Media_DB_v2" not in inspect.getsource(document_versions_repository)


def test_keywords_repository_no_longer_mentions_media_db_v2_in_source() -> None:
    assert "Media_DB_v2" not in inspect.getsource(keywords_repository)


def test_media_files_repository_no_longer_mentions_media_db_v2_in_source() -> None:
    assert "Media_DB_v2" not in inspect.getsource(media_files_repository)


def test_admin_bundle_service_imports_media_schema_helper_from_runtime_factory():
    module = importlib.reload(admin_bundle_service)
    assert module.get_current_media_schema_version is media_db_runtime_factory.get_current_media_schema_version


def test_embeddings_abtest_jobs_worker_imports_create_media_database_from_media_db_api():
    module = importlib.reload(embeddings_abtest_jobs_worker)
    assert module.create_media_database is media_db_api.create_media_database


def test_embeddings_abtest_service_does_not_bind_media_database_from_shim(monkeypatch):
    monkeypatch.setattr(legacy_media_db, "MediaDatabase", object(), raising=False)
    module = importlib.reload(embeddings_abtest_service)
    assert "MediaDatabase" not in module.__dict__


def test_claims_clustering_does_not_bind_media_database_from_shim(monkeypatch):
    monkeypatch.setattr(legacy_media_db, "MediaDatabase", object(), raising=False)
    module = importlib.reload(claims_clustering)
    assert "MediaDatabase" not in module.__dict__


def test_tts_jobs_worker_imports_create_media_database_from_media_db_api(monkeypatch):
    monkeypatch.setattr(legacy_media_db, "MediaDatabase", object(), raising=False)
    module = importlib.reload(tts_jobs_worker)
    assert module.create_media_database is media_db_api.create_media_database
    assert "MediaDatabase" not in module.__dict__


def test_media_db_api_managed_media_database_initializes_and_closes(monkeypatch, tmp_path):
    events = []
    db_path = str(tmp_path / "managed.db")

    class _FakeDb:
        def initialize_db(self):
            events.append("initialize")

        def close_connection(self):
            events.append("close")

    fake_db = _FakeDb()

    def _fake_create_media_database(client_id, **kwargs):
        events.append(("create", client_id, kwargs.get("db_path")))
        return fake_db

    monkeypatch.setattr(media_db_api, "create_media_database", _fake_create_media_database)

    with media_db_api.managed_media_database("managed-client", db_path=db_path) as db:
        assert db is fake_db
        events.append("body")

    assert events == [
        ("create", "managed-client", db_path),
        "initialize",
        "body",
        "close",
    ]


def test_media_db_api_managed_media_database_suppresses_selected_lifecycle_errors(monkeypatch):
    events = []

    class _FakeDb:
        def initialize_db(self):
            events.append("initialize")
            raise RuntimeError("init failed")

        def close_connection(self):
            events.append("close")
            raise ValueError("close failed")

    fake_db = _FakeDb()

    monkeypatch.setattr(media_db_api, "create_media_database", lambda *_args, **_kwargs: fake_db)

    with media_db_api.managed_media_database(
        "managed-client",
        suppress_init_exceptions=(RuntimeError,),
        suppress_close_exceptions=(ValueError,),
    ) as db:
        assert db is fake_db
        events.append("body")

    assert events == ["initialize", "body", "close"]


@pytest.mark.asyncio
async def test_users_db_get_user_media_db_uses_media_db_api_factory(monkeypatch):
    captured = {}
    sentinel = object()
    db_path = "users-media.db"

    def _fake_create_media_database(client_id, **kwargs):
        captured["client_id"] = client_id
        captured.update(kwargs)
        return sentinel

    def _raise_legacy_factory(*args, **kwargs):  # noqa: ARG001
        raise AssertionError("legacy DB_Manager factory should not be used")

    monkeypatch.setattr(Users_DB, "get_user_db_path", lambda *_args, **_kwargs: db_path)
    monkeypatch.setattr(media_db_api, "create_media_database", _fake_create_media_database)
    monkeypatch.setattr(DB_Manager, "create_media_database", _raise_legacy_factory)

    result = await Users_DB.get_user_media_db(42)

    assert result is sentinel
    assert captured["client_id"] == "42"
    assert captured["db_path"] == db_path
