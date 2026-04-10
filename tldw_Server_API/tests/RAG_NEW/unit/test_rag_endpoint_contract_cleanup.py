import pytest
from fastapi import BackgroundTasks
from pathlib import Path

import tldw_Server_API.app.api.v1.endpoints.rag_unified as rag_ep
from tldw_Server_API.app.core.RAG.rag_service.request_bundle import ResolvedRequestBundle
from tldw_Server_API.app.core.RAG.rag_service.request_resolution import ResolvedRAGRequest
from tldw_Server_API.app.core.RAG.rag_service.retrieval_plan import RetrievalPlan
from tldw_Server_API.app.core.RAG.rag_service.unified_pipeline import UnifiedSearchResult


pytestmark = pytest.mark.unit


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
        assert not hasattr(rag_ep, shim_name), f"transitional shim still exported: {shim_name}"


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
    assert expected_flow in readme_text


@pytest.mark.asyncio
async def test_unified_search_standard_path_delegates_to_core_coordinate_helper(
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
    coordinated_result = UnifiedSearchResult(
        documents=[{"id": "doc-coordinated", "content": "coordinated"}],
        query="standard delegation",
        metadata={"phase": "coordinated"},
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

    def fake_coordinate_standard_result_evidence(result, request_contract):  # noqa: ANN001
        captured["called"] = True
        captured["result"] = result
        captured["request_contract"] = request_contract
        return coordinated_result

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
    monkeypatch.setattr(rag_ep, "coordinate_standard_result_evidence", fake_coordinate_standard_result_evidence)
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

    assert captured["called"] is True
    assert captured["result"] is standard_result
    assert captured["request_contract"] is resolved_request
    assert captured["mapping_input"] is coordinated_result
    assert response == {"ok": True, "payload": {"mapped": True}}
