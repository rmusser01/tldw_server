from datetime import datetime, timezone

import pytest

from tldw_Server_API.app.core.Research.models import ResearchPlan

pytestmark = pytest.mark.unit


def _plan(source_policy: str) -> ResearchPlan:
    return ResearchPlan(
        query="Compare internal notes with public sources",
        focus_areas=["evidence alignment"],
        source_policy=source_policy,
        autonomy_mode="checkpointed",
        stop_criteria={"min_cited_sections": 1},
    )


@pytest.mark.asyncio
async def test_broker_uses_only_local_lane_for_local_only_policy():
    from tldw_Server_API.app.core.Research.broker import ResearchBroker

    calls: list[str] = []

    async def local_lane(**kwargs):
        calls.append("local")
        return [
            {
                "id": "doc-1",
                "title": "Internal note",
                "content": "Internal note summary",
                "url": None,
            }
        ]

    async def academic_lane(**kwargs):
        calls.append("academic")
        return []

    async def web_lane(**kwargs):
        calls.append("web")
        return []

    broker = ResearchBroker(
        local_search_fn=local_lane,
        academic_search_fn=academic_lane,
        web_search_fn=web_lane,
    )

    result = await broker.collect_focus_area(
        session_id="rs_1",
        owner_user_id="1",
        focus_area="evidence alignment",
        plan=_plan("local_only"),
        context={},
    )

    assert calls == ["local"]
    assert len(result.sources) == 1
    assert result.sources[0].source_type == "local_document"
    assert result.evidence_notes[0].focus_area == "evidence alignment"


@pytest.mark.asyncio
async def test_broker_falls_back_to_external_lanes_when_local_first_is_sparse():
    from tldw_Server_API.app.core.Research.broker import ResearchBroker

    calls: list[str] = []

    async def local_lane(**kwargs):
        calls.append("local")
        return [{"id": "doc-1", "title": "Only local hit", "content": "one"}]

    async def academic_lane(**kwargs):
        calls.append("academic")
        return [{"doi": "10.1000/example", "title": "Academic source", "summary": "paper"}]

    async def web_lane(**kwargs):
        calls.append("web")
        return [{"url": "https://example.com/report", "title": "Web source", "snippet": "report"}]

    broker = ResearchBroker(
        local_search_fn=local_lane,
        academic_search_fn=academic_lane,
        web_search_fn=web_lane,
    )

    result = await broker.collect_focus_area(
        session_id="rs_1",
        owner_user_id="1",
        focus_area="evidence alignment",
        plan=_plan("local_first"),
        context={},
    )

    assert calls == ["local", "academic", "web"]
    assert {source.source_type for source in result.sources} == {
        "local_document",
        "academic_paper",
        "web_page",
    }
    assert result.collection_metrics["lane_counts"]["local"] == 1


@pytest.mark.asyncio
async def test_broker_dedupes_sources_across_lanes():
    from tldw_Server_API.app.core.Research.broker import ResearchBroker

    async def local_lane(**kwargs):
        return [{"url": "https://example.com/shared", "title": "Shared source", "content": "local"}]

    async def academic_lane(**kwargs):
        return []

    async def web_lane(**kwargs):
        return [{"url": "https://example.com/shared", "title": "Shared source", "snippet": "web"}]

    broker = ResearchBroker(
        local_search_fn=local_lane,
        academic_search_fn=academic_lane,
        web_search_fn=web_lane,
    )

    result = await broker.collect_focus_area(
        session_id="rs_1",
        owner_user_id="1",
        focus_area="evidence alignment",
        plan=_plan("balanced"),
        context={},
    )

    assert len(result.sources) == 1
    assert len(result.evidence_notes) == 1
    assert result.collection_metrics["deduped_sources"] == 1


class _ProviderStub:
    def __init__(self, records=None, *, error: Exception | None = None):
        self._records = list(records or [])
        self._error = error
        self.calls: list[dict[str, object]] = []

    async def search(self, *, focus_area: str, query: str, owner_user_id: str, config: dict[str, object]):
        self.calls.append(
            {
                "focus_area": focus_area,
                "query": query,
                "owner_user_id": owner_user_id,
                "config": dict(config),
            }
        )
        if self._error is not None:
            raise self._error
        return list(self._records)


@pytest.mark.asyncio
async def test_broker_passes_resolved_provider_config_and_preserves_provider_identity():
    from tldw_Server_API.app.core.Research.broker import ResearchBroker

    local_provider = _ProviderStub(
        [
            {
                "id": "doc-1",
                "title": "Internal note",
                "content": "Internal note summary",
                "provider": "local_corpus",
            }
        ]
    )
    academic_provider = _ProviderStub(
        [
            {
                "doi": "10.1000/example",
                "title": "Academic source",
                "summary": "paper",
                "provider": "arxiv",
            }
        ]
    )
    web_provider = _ProviderStub(
        [
            {
                "url": "https://example.com/report",
                "title": "Web source",
                "snippet": "report",
                "provider": "kagi",
            }
        ]
    )

    broker = ResearchBroker(
        local_provider=local_provider,
        academic_provider=academic_provider,
        web_provider=web_provider,
    )

    collection_started_at = datetime.now(timezone.utc)
    result = await broker.collect_focus_area(
        session_id="rs_1",
        owner_user_id="1",
        focus_area="evidence alignment",
        plan=_plan("balanced"),
        provider_config={
            "local": {"top_k": 4, "sources": ["media_db"]},
            "academic": {"providers": ["arxiv"], "max_results": 2},
            "web": {"engine": "kagi", "result_count": 3},
        },
        context={},
    )
    collection_finished_at = datetime.now(timezone.utc)
    retrieved_at = [datetime.fromisoformat(source.retrieved_at) for source in result.sources]

    assert local_provider.calls[0]["config"] == {"top_k": 4, "sources": ["media_db"]}
    assert academic_provider.calls[0]["config"] == {"providers": ["arxiv"], "max_results": 2}
    assert web_provider.calls[0]["config"] == {"engine": "kagi", "result_count": 3}
    assert {source.provider for source in result.sources} == {"local_corpus", "arxiv", "kagi"}
    assert all(value.tzinfo is not None and value.utcoffset() is not None for value in retrieved_at)
    assert all(collection_started_at <= value <= collection_finished_at for value in retrieved_at)
    assert [
        (source.source_id, note.note_id) for source, note in zip(result.sources, result.evidence_notes, strict=True)
    ] == [
        ("src_7aef3f6cf7e9", "note_9c1e03fe5b64"),
        ("src_72b0a1007cc7", "note_3fd24e3f8693"),
        ("src_47fc41c2bc20", "note_817ae2d4c1e1"),
    ]


@pytest.mark.asyncio
async def test_broker_records_structured_lane_errors_without_failing_collection():
    from tldw_Server_API.app.core.Research.broker import ResearchBroker

    local_provider = _ProviderStub(
        [
            {
                "id": "doc-1",
                "title": "Internal note",
                "content": "Internal note summary",
                "provider": "local_corpus",
            }
        ]
    )
    academic_provider = _ProviderStub([])
    web_provider = _ProviderStub(error=RuntimeError("web search failed"))

    broker = ResearchBroker(
        local_provider=local_provider,
        academic_provider=academic_provider,
        web_provider=web_provider,
    )

    result = await broker.collect_focus_area(
        session_id="rs_1",
        owner_user_id="1",
        focus_area="evidence alignment",
        plan=_plan("balanced"),
        provider_config={
            "local": {"top_k": 5, "sources": ["media_db"]},
            "academic": {"providers": ["arxiv"], "max_results": 2},
            "web": {"engine": "duckduckgo", "result_count": 3},
        },
        context={},
    )

    assert len(result.sources) == 1
    assert result.collection_metrics["lane_errors"] == [
        {
            "focus_area": "evidence alignment",
            "lane": "web",
            "message": "web search failed",
        }
    ]
