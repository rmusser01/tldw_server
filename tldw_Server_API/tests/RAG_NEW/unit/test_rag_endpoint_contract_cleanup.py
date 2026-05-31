import inspect
import json
from pathlib import Path

import pytest
from fastapi import BackgroundTasks

from tldw_Server_API.app.api.v1.endpoints import rag_unified
import tldw_Server_API.app.api.v1.endpoints.rag_unified as rag_ep
import tldw_Server_API.app.core.RAG.rag_service as rag_service
from tldw_Server_API.app.core.RAG.rag_service.request_bundle import ResolvedRequestBundle
from tldw_Server_API.app.core.RAG.rag_service.request_resolution import ResolvedRAGRequest
import tldw_Server_API.app.core.RAG.rag_service.retrieval_plan as retrieval_plan_module
from tldw_Server_API.app.core.RAG.rag_service.retrieval_plan import RetrievalPlan
from tldw_Server_API.app.core.RAG.rag_service.unified_pipeline import UnifiedSearchResult


pytestmark = pytest.mark.unit


def test_standard_endpoint_does_not_import_core_evidence_coordinator():
    source = inspect.getsource(rag_unified)

    assert "coordinate_standard_result_evidence" not in source  # nosec B101
    assert "post_retrieval_coordinator" not in source  # nosec B101


def test_streaming_endpoint_delegates_event_generation_to_core():
    source = inspect.getsource(rag_unified)
    stream_source = source[source.find("async def unified_search_stream_endpoint") :]

    assert "stream_rag_events" in source  # nosec B101
    assert "yield json.dumps" in source  # nosec B101
    assert "unified_rag_pipeline(" not in stream_source  # nosec B101
    assert "agentic_rag_pipeline(" not in stream_source  # nosec B101


def test_all_unified_search_variants_accept_prompts_db_dependency() -> None:
    for endpoint_name in (
        "unified_search_endpoint",
        "unified_search_stream_endpoint",
        "unified_batch_endpoint",
        "resume_batch_endpoint",
    ):
        signature = inspect.signature(getattr(rag_ep, endpoint_name))
        assert "prompts_db" in signature.parameters, endpoint_name  # nosec B101


@pytest.mark.asyncio
async def test_streaming_endpoint_frames_core_events_as_ndjson(monkeypatch: pytest.MonkeyPatch) -> None:
    resolved_request = ResolvedRAGRequest(
        query="stream framing",
        strategy="standard",
        payload={"query": "stream framing", "strategy": "standard", "enable_generation": True},
        index_namespace=None,
        rag_profile=None,
        user_id="1",
        feedback_user_id="1",
    )
    retrieval_plan = RetrievalPlan(
        query="stream framing",
        sources=("media_db",),
        search_mode="hybrid",
        top_k=3,
        min_score=0.0,
        index_namespace=None,
    )
    bundle = ResolvedRequestBundle(
        resolved_request=resolved_request,
        retrieval_plan=retrieval_plan,
        pipeline_kwargs={"query": "stream framing"},
    )

    def fake_build_standard_request_bundle(*args, **kwargs):  # noqa: ANN002, ANN003, ARG001
        return bundle

    async def fake_log_rag_queries_for_org(*args, **kwargs):  # noqa: ANN002, ANN003, ARG001
        return None

    async def fake_stream_rag_events(**kwargs):  # noqa: ANN003, ARG001
        yield {"type": "first", "value": 1}
        yield {"type": "second", "value": 2}

    monkeypatch.setattr(rag_ep, "_build_standard_request_bundle", fake_build_standard_request_bundle)
    monkeypatch.setattr(rag_ep, "_log_rag_queries_for_org", fake_log_rag_queries_for_org)
    monkeypatch.setattr(rag_ep, "stream_rag_events", fake_stream_rag_events)

    response = await rag_ep.unified_search_stream_endpoint(
        request_raw=object(),
        request=rag_ep.UnifiedRAGRequest(query="stream framing", enable_generation=True),
        current_user=type("UserStub", (), {"id": 1, "username": "tester"})(),
        media_db=type("DBStub", (), {"db_path": "media.db"})(),
        chacha_db=type("DBStub", (), {"db_path": "notes.db"})(),
    )

    raw_chunks = []
    async for chunk in response.body_iterator:
        raw_chunks.append(chunk.decode() if isinstance(chunk, bytes) else chunk)
    events = [json.loads(line) for line in "".join(raw_chunks).splitlines()]

    assert response.media_type == "application/x-ndjson"  # nosec B101
    assert events == [  # nosec B101
        {"type": "first", "value": 1},
        {"type": "second", "value": 2},
    ]


def test_rag_endpoint_no_longer_exports_transitional_shim_helpers() -> None:
    shim_names = (
        "_apply_search_agent_defaults",
        "_build_agentic_request_context",
        "_resolve_standard_request",
        "_build_agentic_execution_payload",
        "_build_agentic_config",
        "_coordinate_standard_result_evidence",
        "convert_result_to_response",
    )
    for shim_name in shim_names:
        assert not hasattr(rag_ep, shim_name), f"transitional shim still exported: {shim_name}"  # nosec B101


def test_rag_service_readme_documents_current_standard_path_flow() -> None:
    readme_path = (
        Path(__file__).resolve().parents[3]
        / "app"
        / "core"
        / "RAG"
        / "rag_service"
        / "README.md"
    )
    readme_text = readme_path.read_text(encoding="utf-8")
    expected_flow = (
        "HTTP request -> request_bundle.build_request_bundle -> unified_rag_pipeline "
        "(retrieval + generation) -> post_retrieval_coordinator.coordinate_standard_result_evidence "
        "-> response_mapping.rag_result_to_response"
    )
    assert expected_flow in readme_text  # nosec B101


def test_checkpoint_config_sanitizer_drops_non_primitive_pipeline_objects() -> None:
    plan = RetrievalPlan(
        query="checkpoint",
        sources=("media_db",),
        search_mode="hybrid",
        top_k=3,
        min_score=0.0,
        index_namespace="tenant-a",
    )

    sanitized = rag_ep._sanitize_checkpoint_config_for_persistence(
        {
            "query": "checkpoint",
            "top_k": 3,
            "retrieval_plan": plan,
            "nested": {"keep": True, "drop": plan},
            "items": ["ok", plan, 2],
        }
    )

    assert sanitized == {
        "query": "checkpoint",
        "top_k": 3,
        "nested": {"keep": True},
        "items": ["ok", 2],
    }
    json.dumps(sanitized)


def test_rag_endpoint_private_helpers_keep_docstrings() -> None:
    helper_names = (
        "_build_unified_pipeline_kwargs",
        "_build_batch_pipeline_kwargs",
        "_build_standard_request_bundle",
        "_build_batch_request_bundle",
        "_build_resume_batch_request",
        "_checkpoint_safe_value",
        "_sanitize_checkpoint_config_for_persistence",
    )

    for helper_name in helper_names:
        assert inspect.getdoc(getattr(rag_ep, helper_name)), f"{helper_name} lacks a docstring"  # nosec B101


def test_retrieval_plan_module_has_docstring() -> None:
    assert inspect.getdoc(retrieval_plan_module)  # nosec B101


def test_rag_service_lazy_getattr_is_type_annotated() -> None:
    signature = inspect.signature(rag_service.__getattr__)

    assert signature.parameters["name"].annotation is str  # nosec B101
    assert signature.return_annotation is not inspect.Signature.empty  # nosec B101


@pytest.mark.asyncio
async def test_unified_search_standard_path_maps_core_result_directly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolved_request = ResolvedRAGRequest(
        query="standard delegation",
        strategy="standard",
        payload={"query": "standard delegation", "strategy": "standard"},
        index_namespace=None,
        rag_profile=None,
        user_id="1",
        feedback_user_id="1",
    )
    retrieval_plan = RetrievalPlan(
        query="standard delegation",
        sources=("media_db",),
        search_mode="hybrid",
        top_k=5,
        min_score=0.0,
        index_namespace=None,
    )
    standard_result = UnifiedSearchResult(
        documents=[{"id": "doc-standard", "content": "raw"}],
        query="standard delegation",
        metadata={"phase": "standard"},
    )
    bundle = ResolvedRequestBundle(
        resolved_request=resolved_request,
        retrieval_plan=retrieval_plan,
        pipeline_kwargs={"query": "standard delegation"},
    )

    captured: dict[str, object] = {}

    def fake_build_standard_request_bundle(*args, **kwargs):  # noqa: ANN002, ANN003, ARG001
        return bundle

    async def fake_unified_rag_pipeline(**kwargs):  # noqa: ANN003, ARG001
        return standard_result

    def fake_rag_result_from_unified_search_result(result):  # noqa: ANN001
        captured["mapping_input"] = result
        return {"mapped": True}

    def fake_rag_result_to_response(result):  # noqa: ANN001
        captured["mapped_result"] = result
        return {"ok": True, "payload": result}

    async def fake_log_rag_queries_for_org(*args, **kwargs):  # noqa: ANN002, ANN003, ARG001
        return None

    monkeypatch.setattr(rag_ep, "_build_standard_request_bundle", fake_build_standard_request_bundle)
    monkeypatch.setattr(rag_ep, "unified_rag_pipeline", fake_unified_rag_pipeline)
    monkeypatch.setattr(rag_ep, "rag_result_from_unified_search_result", fake_rag_result_from_unified_search_result)
    monkeypatch.setattr(rag_ep, "rag_result_to_response", fake_rag_result_to_response)
    monkeypatch.setattr(rag_ep, "_log_rag_queries_for_org", fake_log_rag_queries_for_org)

    response = await rag_ep.unified_search_endpoint(
        request_raw=object(),
        request=rag_ep.UnifiedRAGRequest(query="standard delegation"),
        background_tasks=BackgroundTasks(),
        current_user=type("UserStub", (), {"username": "tester"})(),
        media_db=type("DBStub", (), {"db_path": "media.db"})(),
        chacha_db=type("DBStub", (), {"db_path": "notes.db"})(),
    )

    assert captured["mapping_input"] is standard_result  # nosec B101
    assert response == {"ok": True, "payload": {"mapped": True}}  # nosec B101
