from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import BackgroundTasks, HTTPException
from starlette.responses import JSONResponse

from tldw_Server_API.app.api.v1.schemas.media_request_models import AddMediaForm
from tldw_Server_API.app.core.Research.discovery.selection import ResolvedDiscoverySelection
from tldw_Server_API.app.core.exceptions import ResearchDiscoveryValidationError

pytestmark = pytest.mark.integration


def _form(selections=None, **overrides) -> AddMediaForm:
    selectors = selections or [("result-1", "candidate-1")]
    values = {
        "media_type": "pdf",
        "research_discovery_id": "rd_example",
        "research_discovery_selections": json.dumps(
            [{"result_id": result_id, "candidate_id": candidate_id} for result_id, candidate_id in selectors]
        ),
        "perform_analysis": False,
    }
    values.update(overrides)
    return AddMediaForm(**values)


def _resolved(index: int, url: str | None = None) -> ResolvedDiscoverySelection:
    return ResolvedDiscoverySelection(
        result_id=f"result-{index}",
        candidate_id=f"candidate-{index}",
        fingerprint=f"doi:10.1000/{index}",
        candidate_type="pdf",
        url=url or f"https://repo.example/paper-{index}.pdf",
        canonical_url=f"https://doi.org/10.1000/{index}",
        title=f"Paper {index}",
        authors=("Ada Lovelace",),
        identifiers={"doi": f"10.1000/{index}", "openalex_id": f"W{index}"},
        source_id="openalex",
        provider="unpaywall",
        access_status="open",
        license_hint="cc-by",
        safe_metadata={"journal": "Examples"},
    )


class _NoDuplicatesDB:
    def get_media_by_url(self, _url: str):
        return None

    def search_by_safe_metadata(self, **_kwargs):
        return [], 0


@pytest.mark.asyncio
async def test_handoff_resolves_server_metadata_and_calls_existing_persistence(monkeypatch):
    from tldw_Server_API.app.core.Ingestion_Media_Processing import research_discovery_handoff as handoff

    resolved = (_resolved(2), _resolved(1))
    captured: dict[str, Any] = {}

    def fake_resolve(**kwargs):
        captured["resolve"] = kwargs
        return resolved

    async def fake_persist(**kwargs):
        captured["persist"] = kwargs
        return JSONResponse(
            content={
                "results": [
                    {"status": "Success", "input_ref": item.url, "db_id": index}
                    for index, item in enumerate(resolved, start=1)
                ]
            }
        )

    monkeypatch.setattr(handoff, "resolve_discovery_selections", fake_resolve)
    monkeypatch.setattr(handoff, "add_media_persist", fake_persist)
    form = _form(
        selections=[("result-2", "candidate-2"), ("result-1", "candidate-1")],
        title="Client title",
        author="Client author",
    )

    response = await handoff.add_research_discovery_pdfs(
        background_tasks=BackgroundTasks(),
        form_data=form,
        files=None,
        db=_NoDuplicatesDB(),
        current_user=SimpleNamespace(id=42),
        usage_log=SimpleNamespace(),
        request=None,
    )

    assert captured["resolve"]["owner_user_id"] == "42"
    assert captured["resolve"]["discovery_id"] == "rd_example"
    assert captured["resolve"]["selections"] == (
        ("result-2", "candidate-2"),
        ("result-1", "candidate-1"),
    )
    persistence_form = captured["persist"]["form_data"]
    assert persistence_form.urls == [item.url for item in resolved]
    assert persistence_form.title is None
    assert persistence_form.author is None
    assert persistence_form.overwrite_existing is False
    assert captured["persist"]["max_download_bytes"] == 50 * 1024 * 1024
    assert captured["persist"]["allowed_download_content_types"] == {"application/pdf"}
    trusted = captured["persist"]["trusted_source_metadata_by_url"][resolved[0].url]
    assert trusted["title"] == "Paper 2"
    assert trusted["author"] == "Ada Lovelace"
    assert trusted["doi"] == "10.1000/2"
    assert trusted["provider_ids"] == {"openalex_id": "W2"}
    body = json.loads(response.body)
    assert [item["result_id"] for item in body["results"]] == ["result-2", "result-1"]
    assert [item["outcome"] for item in body["results"]] == ["created", "created"]


@pytest.mark.asyncio
async def test_handoff_skips_existing_identifier_before_download(monkeypatch):
    from tldw_Server_API.app.core.Ingestion_Media_Processing import research_discovery_handoff as handoff

    item = _resolved(1)

    class ExistingDB(_NoDuplicatesDB):
        def search_by_safe_metadata(self, **kwargs):
            if kwargs["filters"][0]["value"] == "10.1000/1":
                return [{"media_id": 91, "title": "Existing"}], 1
            return [], 0

        def get_media_by_id(self, media_id):
            assert media_id == 91
            return {"id": 91, "uuid": "media-91", "title": "Existing"}

    monkeypatch.setattr(handoff, "resolve_discovery_selections", lambda **_kwargs: (item,))

    async def should_not_persist(**_kwargs):
        raise AssertionError("existing selection must not be downloaded")

    monkeypatch.setattr(handoff, "add_media_persist", should_not_persist)

    response = await handoff.add_research_discovery_pdfs(
        background_tasks=BackgroundTasks(),
        form_data=_form(),
        files=None,
        db=ExistingDB(),
        current_user=SimpleNamespace(id=42),
        usage_log=SimpleNamespace(),
        request=None,
    )

    result = json.loads(response.body)["results"][0]
    assert result["outcome"] == "duplicate_existing"
    assert result["db_id"] == 91
    assert result["media_uuid"] == "media-91"


@pytest.mark.asyncio
async def test_handoff_normalizes_pmcid_for_existing_media_lookup(monkeypatch):
    from tldw_Server_API.app.core.Ingestion_Media_Processing import research_discovery_handoff as handoff

    item = replace(_resolved(1), identifiers={"pmcid": "PMC456"})

    class ExistingDB(_NoDuplicatesDB):
        def search_by_safe_metadata(self, **kwargs):
            if kwargs["filters"][0]["value"] == "456":
                return [{"id": 92, "uuid": "media-92"}], 1
            return [], 0

    monkeypatch.setattr(handoff, "resolve_discovery_selections", lambda **_kwargs: (item,))

    async def should_not_persist(**_kwargs):
        raise AssertionError("normalized PMCID duplicate must not be downloaded")

    monkeypatch.setattr(handoff, "add_media_persist", should_not_persist)
    response = await handoff.add_research_discovery_pdfs(
        background_tasks=BackgroundTasks(),
        form_data=_form(),
        files=None,
        db=ExistingDB(),
        current_user=SimpleNamespace(id=42),
        usage_log=SimpleNamespace(),
        request=None,
    )

    assert json.loads(response.body)["results"][0]["outcome"] == "duplicate_existing"


@pytest.mark.asyncio
async def test_handoff_blocks_known_restricted_access_before_download(monkeypatch):
    from tldw_Server_API.app.core.Ingestion_Media_Processing import research_discovery_handoff as handoff

    item = replace(_resolved(1), access_status="paywalled")
    monkeypatch.setattr(handoff, "resolve_discovery_selections", lambda **_kwargs: (item,))

    async def should_not_persist(**_kwargs):
        raise AssertionError("restricted selection must not be downloaded")

    monkeypatch.setattr(handoff, "add_media_persist", should_not_persist)
    response = await handoff.add_research_discovery_pdfs(
        background_tasks=BackgroundTasks(),
        form_data=_form(),
        files=None,
        db=_NoDuplicatesDB(),
        current_user=SimpleNamespace(id=42),
        usage_log=SimpleNamespace(),
        request=None,
    )

    result = json.loads(response.body)["results"][0]
    assert result["outcome"] == "policy_blocked"
    assert result["status"] == "Error"


@pytest.mark.asyncio
async def test_handoff_preserves_order_when_existing_and_new_results_mix(monkeypatch):
    from tldw_Server_API.app.core.Ingestion_Media_Processing import research_discovery_handoff as handoff

    existing, fresh = _resolved(1), _resolved(2)

    class MixedDB(_NoDuplicatesDB):
        def get_media_by_url(self, url):
            return {"id": 11, "uuid": "media-11"} if url == existing.url else None

    monkeypatch.setattr(handoff, "resolve_discovery_selections", lambda **_kwargs: (existing, fresh))

    async def fake_persist(**kwargs):
        assert kwargs["form_data"].urls == [fresh.url]
        return JSONResponse(content={"results": [{"status": "Success", "input_ref": fresh.url, "db_id": 22}]})

    monkeypatch.setattr(handoff, "add_media_persist", fake_persist)
    response = await handoff.add_research_discovery_pdfs(
        background_tasks=BackgroundTasks(),
        form_data=_form(selections=[("result-1", "candidate-1"), ("result-2", "candidate-2")]),
        files=None,
        db=MixedDB(),
        current_user=SimpleNamespace(id=42),
        usage_log=SimpleNamespace(),
        request=None,
    )

    results = json.loads(response.body)["results"]
    assert [result["result_id"] for result in results] == ["result-1", "result-2"]
    assert [result["outcome"] for result in results] == ["duplicate_existing", "created"]


@pytest.mark.asyncio
async def test_handoff_rejects_duplicate_normalized_candidate_urls(monkeypatch):
    from tldw_Server_API.app.core.Ingestion_Media_Processing import research_discovery_handoff as handoff

    first = _resolved(1, "https://repo.example/paper.pdf?utm_source=discovery")
    second = _resolved(2, "https://repo.example/paper.pdf")
    monkeypatch.setattr(handoff, "resolve_discovery_selections", lambda **_kwargs: (first, second))

    with pytest.raises(ResearchDiscoveryValidationError) as exc_info:
        await handoff.add_research_discovery_pdfs(
            background_tasks=BackgroundTasks(),
            form_data=_form(selections=[("result-1", "candidate-1"), ("result-2", "candidate-2")]),
            files=None,
            db=_NoDuplicatesDB(),
            current_user=SimpleNamespace(id=42),
            usage_log=SimpleNamespace(),
            request=None,
        )

    assert exc_info.value.public_detail == "research_discovery_duplicate_candidate_url"


@pytest.mark.asyncio
async def test_handoff_returns_stable_error_without_downstream_details(monkeypatch):
    from tldw_Server_API.app.core.Ingestion_Media_Processing import research_discovery_handoff as handoff

    item = _resolved(1)
    monkeypatch.setattr(handoff, "resolve_discovery_selections", lambda **_kwargs: (item,))

    async def fake_persist(**_kwargs):
        return JSONResponse(
            content={
                "results": [
                    {
                        "status": "Error",
                        "input_ref": item.url,
                        "error": "upstream failed with token=SECRET at /private/provider.key",
                    }
                ]
            }
        )

    monkeypatch.setattr(handoff, "add_media_persist", fake_persist)
    response = await handoff.add_research_discovery_pdfs(
        background_tasks=BackgroundTasks(),
        form_data=_form(),
        files=None,
        db=_NoDuplicatesDB(),
        current_user=SimpleNamespace(id=42),
        usage_log=SimpleNamespace(),
        request=None,
    )

    result = json.loads(response.body)["results"][0]
    assert result["outcome"] == "failed"
    assert result["error"] == "PDF ingestion failed."
    assert "SECRET" not in str(result)


@pytest.mark.asyncio
async def test_handoff_returns_200_when_persistence_warning_is_created(monkeypatch):
    from tldw_Server_API.app.core.Ingestion_Media_Processing import research_discovery_handoff as handoff

    item = _resolved(1)
    monkeypatch.setattr(handoff, "resolve_discovery_selections", lambda **_kwargs: (item,))

    async def fake_persist(**_kwargs):
        return JSONResponse(
            content={
                "results": [
                    {
                        "status": "Warning",
                        "input_ref": item.url,
                        "message": "Created with a non-fatal warning.",
                        "db_id": 12,
                    }
                ]
            },
            status_code=207,
        )

    monkeypatch.setattr(handoff, "add_media_persist", fake_persist)
    response = await handoff.add_research_discovery_pdfs(
        background_tasks=BackgroundTasks(),
        form_data=_form(),
        files=None,
        db=_NoDuplicatesDB(),
        current_user=SimpleNamespace(id=42),
        usage_log=SimpleNamespace(),
        request=None,
    )

    assert response.status_code == 200
    assert json.loads(response.body)["results"][0]["outcome"] == "created"


@pytest.mark.asyncio
async def test_media_add_route_branches_before_normal_persistence(monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints.media import add as add_endpoint

    expected = JSONResponse(content={"results": [{"outcome": "created"}]})

    async def fake_handoff(**_kwargs):
        return expected

    async def should_not_run(**_kwargs):
        raise AssertionError("normal persistence must not run in discovery mode")

    monkeypatch.setattr(add_endpoint, "add_research_discovery_pdfs", fake_handoff)
    monkeypatch.setattr(add_endpoint, "add_media_persist", should_not_run)

    response = await add_endpoint.add_media(
        request=SimpleNamespace(),
        background_tasks=BackgroundTasks(),
        form_data=_form(),
        files=None,
        db=object(),
        current_user=SimpleNamespace(id=42),
        usage_log=SimpleNamespace(),
    )

    assert response is expected


@pytest.mark.asyncio
async def test_media_add_route_leaves_normal_url_persistence_unchanged(monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints.media import add as add_endpoint

    expected = JSONResponse(content={"results": [{"status": "Success"}]})
    captured = {}

    async def fake_persist(**kwargs):
        captured.update(kwargs)
        return expected

    monkeypatch.setattr(add_endpoint, "add_media_persist", fake_persist)
    form = AddMediaForm(media_type="pdf", urls=["https://repo.example/direct.pdf"])

    response = await add_endpoint.add_media(
        request=SimpleNamespace(),
        background_tasks=BackgroundTasks(),
        form_data=form,
        files=None,
        db=object(),
        current_user=SimpleNamespace(id=42),
        usage_log=SimpleNamespace(),
    )

    assert response is expected
    assert captured["form_data"] is form


def test_media_add_route_preserves_existing_dependency_guards():
    from tldw_Server_API.app.api.v1.endpoints.media import add as add_endpoint

    route = next(route for route in add_endpoint.router.routes if route.path == "/add")

    assert len(route.dependencies) == 5


@pytest.mark.asyncio
async def test_media_add_route_maps_discovery_validation_to_422(monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints.media import add as add_endpoint

    async def fake_handoff(**_kwargs):
        raise ResearchDiscoveryValidationError("research_discovery_snapshot_unavailable")

    monkeypatch.setattr(add_endpoint, "add_research_discovery_pdfs", fake_handoff)

    with pytest.raises(HTTPException) as exc_info:
        await add_endpoint.add_media(
            request=SimpleNamespace(),
            background_tasks=BackgroundTasks(),
            form_data=_form(),
            files=None,
            db=object(),
            current_user=SimpleNamespace(id=42),
            usage_log=SimpleNamespace(),
        )

    assert exc_info.value.status_code == 422
    assert exc_info.value.detail == "research_discovery_snapshot_unavailable"


def test_no_research_discovery_ingest_route_exists():
    from tldw_Server_API.app.main import app

    assert "/api/v1/research/discovery/ingest" not in {route.path for route in app.routes}
