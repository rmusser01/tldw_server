"""Golden characterization for legacy research-discovery execution."""

from __future__ import annotations

import asyncio
import copy
import json
import re
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from itertools import count
from pathlib import Path
from typing import Any

import pytest

from tldw_Server_API.app.api.v1.schemas.research_discovery_schemas import (
    ResearchDiscoverySearchResponse,
)
from tldw_Server_API.app.core.exceptions import ResearchDiscoveryUpstreamError
from tldw_Server_API.app.core.Research.discovery.catalog import default_source_catalog
from tldw_Server_API.app.core.Research.discovery.router import (
    DiscoveryProviderError,
    ResearchSourceRouter,
)
from tldw_Server_API.app.core.Research.discovery.service import ResearchDiscoveryService

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
CONTRACT_PATH = ROOT / "Docs" / "Design" / "research_source_inventory" / "research-discovery-legacy-execution-v1.json"
SOURCE_IDS = (
    "openalex",
    "semantic_scholar",
    "crossref",
    "arxiv",
    "pubmed",
    "zenodo",
    "figshare",
    "osf",
)
GOLDEN_REQUEST = {
    "owner_user_id": "legacy-owner",
    "query": "legacy execution contract",
    "source_ids": list(reversed(SOURCE_IDS)),
    "categories": [],
    "per_source_limit": 2,
    "total_limit": 16,
    "filters": {"language": "en", "year_from": 2020},
}
_DB_COUNTER = count()
ASYNC_GUARD_SECONDS = 10.0


class _DefaultResolverTripwire:
    """Fail if the service constructs or invokes its network-capable default."""

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("default OA resolver must not be constructed")

    def resolve_for_result(self, **_kwargs: Any) -> list[Any]:
        raise AssertionError("default OA resolver must not be called")


class _NoIOOAResolver:
    """Deterministic resolver that records inputs and performs no I/O."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def resolve_for_result(self, **kwargs: Any) -> list[Any]:
        assert kwargs["raw_urls"] == ()
        self.calls.append(dict(kwargs))
        return []


class _CompletionControl:
    """Release adapters in a test-selected order without timing races."""

    def __init__(self, source_ids: tuple[str, ...]) -> None:
        self.source_ids = source_ids
        self.started: set[str] = set()
        self.all_started = asyncio.Event()
        self.release = {source_id: asyncio.Event() for source_id in source_ids}
        self.completed = {source_id: asyncio.Event() for source_id in source_ids}
        self.completion_order: list[str] = []

    async def wait(self, source_id: str) -> None:
        self.started.add(source_id)
        if self.started == set(self.source_ids):
            self.all_started.set()
        await self.release[source_id].wait()
        self.completion_order.append(source_id)
        self.completed[source_id].set()


class _RecordingAdapter:
    """Return one scripted payload while recording the real router call."""

    def __init__(
        self,
        *,
        source_id: str,
        outcome: object,
        completion_control: _CompletionControl | None,
    ) -> None:
        self.source_id = source_id
        self.outcome = outcome
        self.completion_control = completion_control
        self.calls: list[dict[str, Any]] = []
        self.source_objects: list[Any] = []

    async def search(
        self,
        *,
        query: str,
        source: Any,
        limit: int,
        filters: dict[str, Any],
    ) -> object:
        self.source_objects.append(source)
        self.calls.append(
            {
                "filters": dict(filters),
                "limit": limit,
                "query": query,
                "source_id": source.source_id,
                "source_priority": source.priority,
            }
        )
        if self.completion_control is not None:
            await self.completion_control.wait(source.source_id)
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return copy.deepcopy(self.outcome)


def _provider_record(source_id: str) -> dict[str, Any]:
    ordinal = SOURCE_IDS.index(source_id) + 1
    display_name = source_id.replace("_", " ").title()
    return {
        "abstract": f"Deterministic abstract from {display_name}.",
        "authors": [f"{display_name} Author"],
        "doi": f"10.4242/{source_id}",
        "fixture_label": f"{source_id}-legacy-record",
        "provider_ids": {"fixture_id": f"{source_id}-001"},
        "published_at": f"2026-01-{ordinal:02d}",
        "title": f"{display_name} Legacy Result",
        "url": f"https://records.example/{source_id}",
    }


def _build_service(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    outcomes: dict[str, object] | None = None,
    completion_control: _CompletionControl | None = None,
) -> tuple[Any, dict[str, _RecordingAdapter], _NoIOOAResolver, Any, list[str], ResearchDiscoveryService]:
    from tldw_Server_API.app.core.DB_Management.ResearchSessionsDB import ResearchSessionsDB
    from tldw_Server_API.app.core.Research.discovery import oa as oa_module

    catalog = default_source_catalog()
    scripted_outcomes: dict[str, object] = {source_id: [_provider_record(source_id)] for source_id in SOURCE_IDS}
    scripted_outcomes.update(outcomes or {})
    adapters = {
        source_id: _RecordingAdapter(
            source_id=source_id,
            outcome=scripted_outcomes[source_id],
            completion_control=completion_control,
        )
        for source_id in SOURCE_IDS
    }
    router = ResearchSourceRouter(
        catalog=catalog,
        adapters=adapters,
        per_source_timeout_seconds=ASYNC_GUARD_SECONDS,
        max_concurrency=4,
    )
    resolver = _NoIOOAResolver()
    db = ResearchSessionsDB(tmp_path / f"research-{next(_DB_COUNTER)}.db")
    db_factory_calls: list[str] = []

    def db_factory(owner_user_id: str) -> Any:
        db_factory_calls.append(owner_user_id)
        return db

    monkeypatch.setattr(oa_module, "ResearchOAResolver", _DefaultResolverTripwire)
    service = ResearchDiscoveryService(
        catalog=catalog,
        router=router,
        oa_resolver=resolver,
        db_factory=db_factory,
    )
    return catalog, adapters, resolver, db, db_factory_calls, service


def _stable_response_projection(
    response: object,
    *,
    discovery_id: str | None = None,
) -> dict[str, Any]:
    """Serialize every public field except explicitly volatile values."""
    if isinstance(response, dict) and "discovery_id" not in response:
        assert discovery_id is not None
        response = {"discovery_id": discovery_id, **response}
    payload = ResearchDiscoverySearchResponse.model_validate(response).model_dump(mode="json")
    payload.pop("discovery_id")
    payload["metrics"].pop("elapsed_ms")
    for status in payload["source_statuses"]:
        status.pop("elapsed_ms")
    return payload


def _parse_aware_datetime(value: str) -> datetime:
    """Parse a generated timestamp and require an explicit UTC offset."""
    parsed = datetime.fromisoformat(value)
    assert parsed.tzinfo is not None
    assert parsed.utcoffset() is not None
    return parsed


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _contract() -> dict[str, Any]:
    assert CONTRACT_PATH.is_file(), f"missing legacy execution contract: {CONTRACT_PATH}"
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def _provider_calls(adapters: dict[str, _RecordingAdapter]) -> list[dict[str, Any]]:
    return [adapters[source_id].calls[0] for source_id in SOURCE_IDS]


def _assert_selected_calls(
    *,
    catalog: Any,
    adapters: dict[str, _RecordingAdapter],
    selected_source_ids: list[str],
) -> None:
    for source_id, adapter in adapters.items():
        expected_count = 1 if source_id in selected_source_ids else 0
        assert len(adapter.calls) == expected_count
        if expected_count:
            assert adapter.source_objects == [catalog.get_source(source_id)]


def test_legacy_execution_contract_fixture_is_canonical_json() -> None:
    """The reviewed golden uses sorted keys and exactly one trailing newline."""
    raw_contract = CONTRACT_PATH.read_text(encoding="utf-8")
    assert raw_contract == _canonical_json(json.loads(raw_contract))


@pytest.mark.asyncio
async def test_all_eight_sources_match_frozen_execution_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real V1 service stack retains its stable all-source projection."""
    catalog, adapters, resolver, db, db_factory_calls, service = _build_service(tmp_path, monkeypatch)

    search_started_at = datetime.now(UTC)
    response = await service.search(**GOLDEN_REQUEST)
    search_finished_at = datetime.now(UTC)
    snapshot = db.get_discovery_snapshot(response.discovery_id, owner_user_id="legacy-owner")
    assert snapshot is not None

    actual_contract = {
        "all_sources_success": {
            "persisted_effective_config": snapshot.effective_config_json,
            "persisted_request": snapshot.request_json,
            "provider_calls": _provider_calls(adapters),
            "request": GOLDEN_REQUEST,
            "stable_response": _stable_response_projection(response),
        },
        "catalog_version": catalog.catalog_version,
        "contract_version": "research-discovery-legacy-execution-v1",
        "source_priority": [
            {"priority": source.priority, "source_id": source.source_id} for source in catalog.list_sources()
        ],
    }
    snapshot_projection = _stable_response_projection(
        snapshot.response_json,
        discovery_id=response.discovery_id,
    )
    snapshot_created_at = _parse_aware_datetime(snapshot.created_at)
    snapshot_expires_at = _parse_aware_datetime(snapshot.expires_at)

    assert actual_contract == _contract()
    assert re.fullmatch(r"rd_[0-9a-f]{12}", response.discovery_id)
    assert snapshot.id == response.discovery_id
    assert isinstance(response.metrics.elapsed_ms, float) and response.metrics.elapsed_ms >= 0
    assert all(isinstance(status.elapsed_ms, float) and status.elapsed_ms >= 0 for status in response.source_statuses)
    assert db_factory_calls == ["legacy-owner"]
    assert search_started_at <= snapshot_created_at <= search_finished_at
    assert snapshot_created_at < snapshot_expires_at
    assert snapshot_expires_at - snapshot_created_at == timedelta(hours=24)
    assert snapshot_projection == actual_contract["all_sources_success"]["stable_response"]
    assert snapshot.request_json == actual_contract["all_sources_success"]["persisted_request"]
    assert snapshot.effective_config_json == actual_contract["all_sources_success"]["persisted_effective_config"]
    assert snapshot.effective_config_json == response.effective_config
    assert len(resolver.calls) == len(SOURCE_IDS)
    assert {call["source_id"] for call in resolver.calls} == set(SOURCE_IDS)
    _assert_selected_calls(catalog=catalog, adapters=adapters, selected_source_ids=list(SOURCE_IDS))


@pytest.mark.parametrize(
    ("selection", "expected_source_ids", "expected_defaulted_categories"),
    [
        ({}, ["openalex", "semantic_scholar", "crossref"], ["open_research_graph"]),
        (
            {"source_ids": [], "categories": []},
            ["openalex", "semantic_scholar", "crossref"],
            ["open_research_graph"],
        ),
        ({"source_ids": ["osf", "arxiv"]}, ["arxiv", "osf"], []),
        ({"categories": ["repositories"]}, ["zenodo", "figshare", "osf"], []),
        (
            {"source_ids": ["osf", "pubmed"], "categories": ["open_research_graph"]},
            ["openalex", "semantic_scholar", "crossref", "pubmed", "osf"],
            [],
        ),
    ],
    ids=("omitted", "empty", "explicit", "category-only", "source-category-union"),
)
@pytest.mark.asyncio
async def test_selection_executes_only_resolved_sources_through_real_service(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    selection: dict[str, list[str]],
    expected_source_ids: list[str],
    expected_defaulted_categories: list[str],
) -> None:
    """Omitted, empty, explicit, category, and union selections stay stable."""
    catalog, adapters, resolver, db, db_factory_calls, service = _build_service(tmp_path, monkeypatch)

    response = await service.search(
        owner_user_id="selection-owner",
        query="selection contract",
        per_source_limit=3,
        filters={"year": 2026},
        **selection,
    )

    assert response.effective_config["source_ids"] == expected_source_ids
    assert response.effective_config["defaulted_categories"] == expected_defaulted_categories
    assert [result.primary_source_id for result in response.results] == expected_source_ids
    assert [status.source_id for status in response.source_statuses] == expected_source_ids
    assert db_factory_calls == ["selection-owner"]
    assert db.get_discovery_snapshot(response.discovery_id, owner_user_id="selection-owner") is not None
    assert len(resolver.calls) == len(expected_source_ids)
    _assert_selected_calls(catalog=catalog, adapters=adapters, selected_source_ids=expected_source_ids)


@pytest.mark.asyncio
async def test_result_and_status_order_follow_priority_not_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Concurrent provider completion cannot reorder legacy aggregation."""
    selected_source_ids = ("openalex", "semantic_scholar", "crossref")
    completion_control = _CompletionControl(selected_source_ids)
    _catalog, _adapters, _resolver, _db, _db_calls, service = _build_service(
        tmp_path,
        monkeypatch,
        completion_control=completion_control,
    )
    search_task = asyncio.create_task(service.search(owner_user_id="ordering-owner", query="ordering contract"))
    try:
        await asyncio.wait_for(completion_control.all_started.wait(), timeout=ASYNC_GUARD_SECONDS)
        for source_id in reversed(selected_source_ids):
            completion_control.release[source_id].set()
            await asyncio.wait_for(
                completion_control.completed[source_id].wait(),
                timeout=ASYNC_GUARD_SECONDS,
            )
        response = await asyncio.wait_for(
            asyncio.shield(search_task),
            timeout=ASYNC_GUARD_SECONDS,
        )
    except Exception:
        if not search_task.done():
            search_task.cancel()
        with suppress(asyncio.CancelledError, Exception):
            await search_task
        raise

    assert completion_control.completion_order == list(reversed(selected_source_ids))
    assert [result.primary_source_id for result in response.results] == list(selected_source_ids)
    assert [status.source_id for status in response.source_statuses] == list(selected_source_ids)


@pytest.mark.asyncio
async def test_partial_failure_freezes_status_warning_and_malformed_payload_semantics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One malformed item fails its whole source while peers still succeed."""
    openalex_record = _provider_record("openalex")
    openalex_record["warnings"] = ["openalex-result-warning"]
    crossref_valid_record = _provider_record("crossref")
    outcomes = {
        "openalex": [openalex_record],
        "semantic_scholar": DiscoveryProviderError(safe_message="Semantic Scholar fixture failure."),
        "crossref": [crossref_valid_record, "malformed-record"],
        "arxiv": TimeoutError("fixture timeout"),
    }
    catalog, adapters, resolver, _db, _db_calls, service = _build_service(
        tmp_path,
        monkeypatch,
        outcomes=outcomes,
    )

    response = await service.search(
        owner_user_id="partial-owner",
        query="partial failure contract",
        source_ids=["arxiv", "crossref", "semantic_scholar", "openalex"],
    )

    assert [result.primary_source_id for result in response.results] == ["openalex"]
    assert [
        (status.source_id, status.status, status.message, status.result_count, status.warnings)
        for status in response.source_statuses
    ] == [
        ("openalex", "ok", None, 1, ()),
        (
            "semantic_scholar",
            "provider_error",
            "Semantic Scholar fixture failure.",
            0,
            (),
        ),
        (
            "crossref",
            "internal_error",
            "Discovery adapter failed unexpectedly.",
            0,
            (),
        ),
        (
            "arxiv",
            "timeout",
            "Provider request timed out.",
            0,
            ("provider_call_may_continue_after_timeout",),
        ),
    ]
    assert response.warnings == (
        "semantic_scholar:provider_error:Semantic Scholar fixture failure.",
        "crossref:internal_error:Discovery adapter failed unexpectedly.",
        "arxiv:timeout:Provider request timed out.",
        "arxiv:provider_call_may_continue_after_timeout",
        "openalex-result-warning",
    )
    assert len(resolver.calls) == 1
    _assert_selected_calls(
        catalog=catalog,
        adapters=adapters,
        selected_source_ids=["openalex", "semantic_scholar", "crossref", "arxiv"],
    )


@pytest.mark.asyncio
async def test_all_provider_failures_raise_stable_terminal_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The service preserves its typed all-sources-failed outcome."""
    outcomes = {
        "semantic_scholar": DiscoveryProviderError(),
        "crossref": [_provider_record("crossref"), "malformed-record"],
        "arxiv": TimeoutError("fixture timeout"),
    }
    catalog, adapters, resolver, _db, db_factory_calls, service = _build_service(
        tmp_path,
        monkeypatch,
        outcomes=outcomes,
    )

    with pytest.raises(
        ResearchDiscoveryUpstreamError,
        match="^research_discovery_all_sources_failed$",
    ):
        await service.search(
            owner_user_id="failure-owner",
            query="all failure contract",
            source_ids=["arxiv", "crossref", "semantic_scholar"],
        )

    assert resolver.calls == []
    assert db_factory_calls == []
    _assert_selected_calls(
        catalog=catalog,
        adapters=adapters,
        selected_source_ids=["semantic_scholar", "crossref", "arxiv"],
    )


@pytest.mark.asyncio
async def test_valid_empty_provider_results_return_and_persist_empty_response(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Successful empty providers are not reclassified as upstream failure."""
    catalog, adapters, resolver, db, db_factory_calls, service = _build_service(
        tmp_path,
        monkeypatch,
        outcomes={"figshare": [], "osf": []},
    )

    response = await service.search(
        owner_user_id="empty-owner",
        query="valid empty contract",
        source_ids=["osf", "figshare"],
    )

    assert response.results == ()
    assert [(status.source_id, status.status, status.result_count) for status in response.source_statuses] == [
        ("figshare", "ok", 0),
        ("osf", "ok", 0),
    ]
    assert response.warnings == ()
    assert response.metrics.result_count == 0
    assert response.metrics.deduped_result_count == 0
    assert resolver.calls == []
    assert db_factory_calls == ["empty-owner"]
    assert db.get_discovery_snapshot(response.discovery_id, owner_user_id="empty-owner") is not None
    _assert_selected_calls(
        catalog=catalog,
        adapters=adapters,
        selected_source_ids=["figshare", "osf"],
    )
