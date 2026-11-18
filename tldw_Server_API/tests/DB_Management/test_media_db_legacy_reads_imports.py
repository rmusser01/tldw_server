import importlib

from tldw_Server_API.app.core.DB_Management.media_db import api as media_db_api
from tldw_Server_API.tests.DB_Management._media_db_legacy_stub import (
    install_legacy_media_db_stub,
)


def test_document_references_imports_latest_transcription_from_media_db_api(
    monkeypatch,
) -> None:
    media_db_v2 = install_legacy_media_db_stub(monkeypatch)
    document_references = importlib.import_module(
        "tldw_Server_API.app.api.v1.endpoints.media.document_references"
    )

    def _shim_should_not_be_bound(*args, **kwargs):
        raise AssertionError(
            "document_references should not bind get_latest_transcription from Media_DB_v2"
        )

    monkeypatch.setattr(
        media_db_v2,
        "get_latest_transcription",
        _shim_should_not_be_bound,
    )

    reloaded = importlib.reload(document_references)
    assert reloaded.get_latest_transcription is media_db_api.get_latest_transcription


def test_document_insights_imports_latest_transcription_from_media_db_api(
    monkeypatch,
) -> None:
    media_db_v2 = install_legacy_media_db_stub(monkeypatch)
    document_insights = importlib.import_module(
        "tldw_Server_API.app.api.v1.endpoints.media.document_insights"
    )

    def _shim_should_not_be_bound(*args, **kwargs):
        raise AssertionError(
            "document_insights should not bind get_latest_transcription from Media_DB_v2"
        )

    monkeypatch.setattr(
        media_db_v2,
        "get_latest_transcription",
        _shim_should_not_be_bound,
    )

    reloaded = importlib.reload(document_insights)
    assert reloaded.get_latest_transcription is media_db_api.get_latest_transcription


def test_quiz_source_resolver_imports_latest_transcription_from_media_db_api(
    monkeypatch,
) -> None:
    media_db_v2 = install_legacy_media_db_stub(monkeypatch)
    quiz_source_resolver = importlib.import_module(
        "tldw_Server_API.app.services.quiz_source_resolver"
    )

    def _shim_should_not_be_bound(*args, **kwargs):
        raise AssertionError(
            "quiz_source_resolver should not bind get_latest_transcription from Media_DB_v2"
        )

    monkeypatch.setattr(
        media_db_v2,
        "get_latest_transcription",
        _shim_should_not_be_bound,
    )

    reloaded = importlib.reload(quiz_source_resolver)
    assert reloaded.get_latest_transcription is media_db_api.get_latest_transcription


def test_slides_endpoint_imports_latest_transcription_from_media_db_api(
    monkeypatch,
) -> None:
    media_db_v2 = install_legacy_media_db_stub(monkeypatch)
    slides_endpoint = importlib.import_module(
        "tldw_Server_API.app.api.v1.endpoints.slides"
    )

    def _shim_should_not_be_bound(*args, **kwargs):
        raise AssertionError(
            "slides endpoint should not bind get_latest_transcription from Media_DB_v2"
        )

    monkeypatch.setattr(
        media_db_v2,
        "get_latest_transcription",
        _shim_should_not_be_bound,
    )

    reloaded = importlib.reload(slides_endpoint)
    assert reloaded.get_latest_transcription is media_db_api.get_latest_transcription


def test_data_tables_jobs_worker_imports_latest_transcription_from_media_db_api(
    monkeypatch,
) -> None:
    media_db_v2 = install_legacy_media_db_stub(monkeypatch)
    data_tables_jobs_worker = importlib.import_module(
        "tldw_Server_API.app.core.Data_Tables.jobs_worker"
    )

    def _shim_should_not_be_bound(*args, **kwargs):
        raise AssertionError(
            "data_tables.jobs_worker should not bind get_latest_transcription from Media_DB_v2"
        )

    monkeypatch.setattr(
        media_db_v2,
        "get_latest_transcription",
        _shim_should_not_be_bound,
    )

    reloaded = importlib.reload(data_tables_jobs_worker)
    assert reloaded.get_latest_transcription is media_db_api.get_latest_transcription


def test_navigation_core_imports_read_helpers_from_media_db_api(
    monkeypatch,
) -> None:
    media_db_v2 = install_legacy_media_db_stub(monkeypatch)
    navigation_endpoint = importlib.import_module(
        "tldw_Server_API.app.api.v1.endpoints.media.navigation"
    )
    navigation_core = importlib.import_module("tldw_Server_API.app.core.Media.navigation")

    def _shim_should_not_be_bound(*args, **kwargs):
        raise AssertionError(
            "navigation should not bind read helpers from Media_DB_v2"
        )

    monkeypatch.setattr(
        media_db_v2,
        "get_latest_transcription",
        _shim_should_not_be_bound,
    )
    monkeypatch.setattr(
        media_db_v2,
        "get_media_transcripts",
        _shim_should_not_be_bound,
    )

    reloaded = importlib.reload(navigation_core)
    assert reloaded.get_latest_transcription is media_db_api.get_latest_transcription
    assert reloaded.get_media_transcripts is media_db_api.get_media_transcripts
    endpoint = importlib.reload(navigation_endpoint)
    assert endpoint.load_navigation_document is reloaded.load_navigation_document
    assert endpoint._select_source_nodes is reloaded._select_source_nodes


def test_media_module_imports_read_helpers_from_media_db_api(
    monkeypatch,
) -> None:
    media_db_v2 = install_legacy_media_db_stub(monkeypatch)
    media_module_impl = importlib.import_module(
        "tldw_Server_API.app.core.MCP_unified.modules.implementations.media_module"
    )

    def _shim_should_not_be_bound(*args, **kwargs):
        raise AssertionError(
            "media_module should not bind read helpers from Media_DB_v2"
        )

    monkeypatch.setattr(
        media_db_v2,
        "get_latest_transcription",
        _shim_should_not_be_bound,
    )
    monkeypatch.setattr(
        media_db_v2,
        "get_media_transcripts",
        _shim_should_not_be_bound,
    )

    reloaded = importlib.reload(media_module_impl)
    assert reloaded.get_latest_transcription is media_db_api.get_latest_transcription
    assert reloaded.get_media_transcripts is media_db_api.get_media_transcripts


def test_chatbook_service_imports_read_helpers_from_media_db_api(
    monkeypatch,
) -> None:
    media_db_v2 = install_legacy_media_db_stub(monkeypatch)
    chatbook_service = importlib.import_module(
        "tldw_Server_API.app.core.Chatbooks.chatbook_service"
    )

    def _shim_should_not_be_bound(*args, **kwargs):
        raise AssertionError(
            "chatbook_service should not bind read helpers from Media_DB_v2"
        )

    monkeypatch.setattr(
        media_db_v2,
        "get_media_prompts",
        _shim_should_not_be_bound,
        raising=False,
    )
    monkeypatch.setattr(
        media_db_v2,
        "get_media_transcripts",
        _shim_should_not_be_bound,
        raising=False,
    )

    reloaded = importlib.reload(chatbook_service)
    assert reloaded.get_media_prompts is media_db_api.get_media_prompts
    assert reloaded.get_media_transcripts is media_db_api.get_media_transcripts
