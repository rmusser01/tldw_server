"""RAG generation, its auxiliary calls and research name their surfaces (TASK-13950.27 PR G)."""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import check_rate_limit
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.exceptions import LLMPolicyRefusal
from tldw_Server_API.app.core.LLM_Calls import gateway
from tldw_Server_API.app.main import app as fastapi_app
from tldw_Server_API.tests.Audio.issued_credentials import issued_credentials
from tldw_Server_API.tests.Moderation.conftest import PROVIDER, finish_frame, openai_frame, reply, DONE

pytestmark = pytest.mark.integration


class _Runtime:
    """Stands in for ProviderCredentialRuntime: issues a handle the gateway authenticates."""

    def __init__(self, **_kwargs: Any) -> None:
        pass

    async def resolve(self, provider: str, *, model: str | None = None) -> Any:
        return issued_credentials(provider, api_key="k", app_config={})

    async def mark_used(self, _handle: object) -> None:
        return None

    async def mark_failed(self, *_a: Any, **_k: Any) -> None:
        return None

    async def close(self) -> None:
        return None


@pytest.fixture
def spy(stub_llm_registry):
    """A test pre-hook recording ``call.surface`` for every gateway call."""
    surfaces: list[str | None] = []
    remove = gateway.add_policy_hooks(pre_call=lambda call: surfaces.append(call.surface))
    yield surfaces
    remove()


@pytest.fixture
def rag_module():
    import tldw_Server_API.app.api.v1.endpoints.rag_unified as rag_ep

    return rag_ep


@pytest.fixture
def client(monkeypatch, stub_llm_registry, rag_module, auth_headers):
    """The real RAG routes and generator, one retrieved document, the key runtime stubbed."""
    from tldw_Server_API.app.core.RAG.rag_service.types import DataSource, Document

    class OneDocumentRetriever:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.retrievers: dict[str, Any] = {}

        async def retrieve(self, query: str, **kwargs: Any) -> list[Document]:
            return [Document(id="d", content="Rowan directs the observatory.", metadata={"title": "Obs"},
                             source=DataSource.MEDIA_DB, score=0.9)]

    async def override_user():
        return User(id=1, username="tester", email=None, is_active=True)

    async def _noop():
        return None

    fastapi_app.dependency_overrides[get_request_user] = override_user
    fastapi_app.dependency_overrides[check_rate_limit] = _noop
    monkeypatch.setattr(rag_module, "MultiDatabaseRetriever", OneDocumentRetriever)
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Provider_Credentials.provider_credential_runtime.ProviderCredentialRuntime", _Runtime
    )
    with TestClient(fastapi_app, headers=auth_headers) as c:
        yield c
    fastapi_app.dependency_overrides.clear()


def _payload(**extra: Any) -> dict[str, Any]:
    return {
        "query": "Who directs the observatory?", "sources": ["media_db"], "enable_generation": True, "enable_cache": False,
        "enable_query_classification": False, "generation_provider": PROVIDER, "generation_model": "m", **extra,
    }


def _events(client: TestClient, payload: dict[str, Any]):
    with client.stream("POST", "/api/v1/rag/search/stream", json=payload) as resp:
        return resp.status_code, [json.loads(raw) for raw in resp.iter_lines() if raw]


def test_rag_generation_input_block_400(spy, moderation_hooks, client, stub_llm_registry):
    response = client.post("/api/v1/rag/search", json=_payload(query="forbidden question"))

    assert response.status_code == 400, response.text
    assert "Input violates moderation policy" in response.text
    assert stub_llm_registry.requests == []
    assert spy == ["rag.generation"]


def test_rag_generation_answer_redacted(moderation_hooks, client, stub_llm_registry):
    stub_llm_registry.reply = reply("the secret answer")

    response = client.post("/api/v1/rag/search", json=_payload())

    assert response.status_code == 200, response.text
    assert "the [X] answer" in response.text
    assert "secret" not in response.text


def test_rag_stream_answer_redacted(moderation_hooks, client, stub_llm_registry):
    stub_llm_registry.items = [openai_frame("the secret answer"), finish_frame(), DONE]

    status, events = _events(client, _payload())

    assert status == 200
    text = "".join(e["text"] for e in events if e.get("type") == "delta")
    assert text == "the [X] answer"
    assert "secret" not in json.dumps(events)


def test_rag_stream_input_block_is_an_error_event(moderation_hooks, client, stub_llm_registry):
    status, events = _events(client, _payload(query="forbidden question"))

    assert status == 200  # the stream's own error event, not an HTTP status
    assert events[-1]["type"] == "error"
    assert events[-1]["code"] == "llm_policy_refused"
    assert events[-1]["status_code"] == 400
    assert events[-1]["message"] == "Input violates moderation policy"
    assert stub_llm_registry.requests == []


# ---- the auxiliary RAG calls and research --------------------------------------------------------------------

Q = "Explain how the observatory schedules its telescopes"
AUX = {"llm_provider": PROVIDER, "llm_model": "m"}


async def _classify(**extra):
    from tldw_Server_API.app.core.RAG.rag_service.query_classifier import classify_query

    return await classify_query(Q, credential_runtime=_Runtime(), **AUX, **extra)


async def _reformulate():
    from tldw_Server_API.app.core.RAG.rag_service.query_classifier import reformulate_query

    return await reformulate_query(Q, [{"role": "user", "content": "hi"}], credential_runtime=_Runtime(), **AUX)


async def _media_search():
    from tldw_Server_API.app.core.RAG.rag_service.media_search import _reformulate_query

    return await _reformulate_query(Q, "rewrite", PROVIDER, "m", credential_runtime=_Runtime())


async def _suggestions():
    from tldw_Server_API.app.core.RAG.rag_service.suggestion_generator import generate_suggestions

    return await generate_suggestions(Q, "an answer", credential_runtime=_Runtime(), **AUX)


async def _research_loop():
    from tldw_Server_API.app.core.RAG.rag_service.query_classifier import QueryClassification
    from tldw_Server_API.app.core.RAG.rag_service.research_agent import research_loop

    return await research_loop(
        Q, QueryClassification(standalone_query=Q), max_iterations=1, credential_runtime=_Runtime(), **AUX
    )


async def _synthesis():
    from types import SimpleNamespace

    from tldw_Server_API.app.core.Research.providers.synthesis import SynthesisProvider

    return await SynthesisProvider().summarize(
        plan=SimpleNamespace(query=Q, focus_areas=["a"]), source_registry=[], evidence_notes=[], collection_summary={},
        config={"provider": PROVIDER, "model": "m"},
    )


async def _artifact():
    from tldw_Server_API.app.core.RAG.rag_service.types import DataSource, Document
    from tldw_Server_API.app.core.Research_Workspace.artifact_generation import generate_research_workspace_artifact

    docs = [Document(id="d", content="Rowan directs the observatory.", metadata={}, source=DataSource.MEDIA_DB, score=1.0)]
    async def verify(**_kwargs):
        raise RuntimeError("claims verification is its own surface (Task 9)")

    return await generate_research_workspace_artifact(verify_fn=verify,
        artifact_type="mindmap", source_documents=docs, generation_provider=PROVIDER, generation_model="m",
        verification_provider=PROVIDER, verification_model="m", temperature=0.1, top_p=1.0, max_tokens=50,
    )


SURFACES = {
    "classifier": ("rag.auxiliary", lambda: _classify()),
    "reformulation": ("rag.auxiliary", _reformulate),
    "media_search": ("rag.auxiliary", _media_search),
    "suggestions": ("rag.auxiliary", _suggestions),
    "research_agent": ("rag.auxiliary", _research_loop),
    "synthesis": ("research", _synthesis),
    "artifact_generation": ("research", _artifact),
}


@pytest.mark.parametrize("site", sorted(SURFACES))
async def test_surface_reaches_gateway(spy, site):
    surface, run = SURFACES[site]
    try:
        await run()
    except Exception:  # noqa: BLE001 - only the surface the gateway saw matters here
        pass
    assert spy == [surface]


@pytest.mark.parametrize("site", ["classifier", "reformulation", "media_search"])
async def test_rag_auxiliary_input_block_propagates(moderation_hooks, stub_llm_registry, site):
    """The user's own query is refused: the step raises the refusal instead of routing on a heuristic."""
    from tldw_Server_API.app.core.RAG.rag_service import media_search as ms
    from tldw_Server_API.app.core.RAG.rag_service import query_classifier as qc

    runs = {
        "classifier": lambda: qc.classify_query("forbidden stuff about the observatory", credential_runtime=_Runtime(), **AUX),
        "reformulation": lambda: qc.reformulate_query("forbidden", [{"role": "user", "content": "hi"}], credential_runtime=_Runtime(), **AUX),
        "media_search": lambda: ms._reformulate_query("forbidden stuff", "rewrite", PROVIDER, "m", credential_runtime=_Runtime()),
    }
    with pytest.raises(LLMPolicyRefusal):
        await runs[site]()
    assert stub_llm_registry.requests == []


@pytest.mark.parametrize("site", ["suggestions", "research_agent"])
async def test_optional_auxiliary_input_block_is_no_result(moderation_hooks, stub_llm_registry, site):
    """Suggestions and the research loop are optional and the answer stays moderated: a refusal ends them quietly."""
    stub_llm_registry.reply = reply("unused")
    from tldw_Server_API.app.core.RAG.rag_service.query_classifier import QueryClassification
    from tldw_Server_API.app.core.RAG.rag_service.research_agent import research_loop
    from tldw_Server_API.app.core.RAG.rag_service.suggestion_generator import generate_suggestions

    if site == "suggestions":
        out = await generate_suggestions("forbidden thing", "an answer", credential_runtime=_Runtime(), **AUX)
        assert isinstance(out, list)
    else:
        out = await research_loop(
            "forbidden thing", QueryClassification(standalone_query="forbidden thing"), max_iterations=1,
            credential_runtime=_Runtime(), **AUX,
        )
        assert out is not None
    assert stub_llm_registry.requests == []


async def test_rag_auxiliary_content_filter_not_retried(moderation_hooks, stub_llm_registry):
    """The classifier's reply is redacted to the refusal text: one provider call, the text is not parsed or reused."""
    stub_llm_registry.reply = reply("the secret verdict")  # redact -> "the [X] verdict" with finish_reason stop
    stub_llm_registry.reply = {**reply("x"), "choices": [{"index": 0, "finish_reason": "content_filter",
                                                          "message": {"role": "assistant", "content": "Output violates moderation policy"}}]}

    result = await _classify()

    assert len(stub_llm_registry.requests) == 1
    assert "Output violates" not in result.reasoning and result.standalone_query == Q


async def test_reformulation_content_filter_keeps_the_original_query(moderation_hooks, stub_llm_registry):
    stub_llm_registry.reply = {**reply("x"), "choices": [{"index": 0, "finish_reason": "content_filter",
                                                          "message": {"role": "assistant", "content": "Output violates moderation policy"}}]}

    assert await _reformulate() == Q
    assert await _media_search() == Q
    assert len(stub_llm_registry.requests) == 2


@pytest.mark.parametrize("site", ["synthesis", "artifact_generation"])
async def test_research_output_refusal_is_a_refusal(stub_llm_registry, site):
    stub_llm_registry.reply = {**reply("x"), "choices": [{"index": 0, "finish_reason": "content_filter",
                                                          "message": {"role": "assistant", "content": "Output violates moderation policy"}}]}
    with pytest.raises(LLMPolicyRefusal):
        await SURFACES[site][1]()
