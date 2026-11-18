"""Public verification binds client locators to owned, unchanged canonical source bytes."""

import asyncio
import hashlib
import json
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_request_user
from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
from tldw_Server_API.app.api.v1.endpoints import research_workspace
from tldw_Server_API.app.core.Claims_Extraction import workspace_citations
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.media_db.api import create_media_database, get_media_repository


@pytest.fixture
def client(tmp_path, monkeypatch):
    workspace_db = CharactersRAGDB(tmp_path / "workspace.db", "verification-test")
    media_db = create_media_database("verification-test", db_path=str(tmp_path / "media.db"))
    workspace_db.upsert_workspace("workspace", "Owned workspace")
    workspace_db.add_workspace_source("workspace", {"id": "source", "media_id": 1, "title": "Source", "source_type": "pdf"})
    content = "  Exact <br> canonical &amp; text. Repeated. Repeated."
    get_media_repository(media_db).add_media_with_keywords(url="research://citation-test", title="Source",
        media_type="document", content=content, keywords=[], overwrite=False)
    real_load = workspace_citations.load_navigation_sections
    snapshot = asyncio.run(real_load(1, media_db))
    loads, calls = [], []

    async def load(media_id, db):
        loads.append((media_id, db))
        return await real_load(media_id, db)

    async def analyze(*args, **kwargs):
        calls.append(args[2])
        return {"choices": [{"finish_reason": "stop", "message": {"content":
            '{"rationale":"All propositions checked against cited bytes.","label":"supported","confidence":0.9}'}}]}

    monkeypatch.setattr(workspace_citations, "load_navigation_sections", load)
    app = FastAPI()
    app.include_router(research_workspace.router, prefix="/api/v1")
    app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id=7)
    app.dependency_overrides[get_chacha_db_for_user] = lambda: workspace_db
    app.dependency_overrides[get_media_db_for_user] = lambda: media_db
    app.dependency_overrides[research_workspace.get_citation_analyzer] = lambda: analyze
    route = next(route for route in research_workspace.router.routes if route.path.endswith("/citations/verify"))
    for dependency in route.dependencies:
        app.dependency_overrides[dependency.dependency] = lambda: None
    with TestClient(app) as http:
        yield http, snapshot, calls, loads, workspace_db
    workspace_db.close_connection()
    media_db.close_connection()


def payload(**locator):
    return {"workspace_id": "workspace", "text": "Canonical claim. [P1]", "api_provider": "openai", "model": "judge",
        "evidence": [{"id": "P1", "source_id": "source", "quote": "Exact <br> canonical &amp; text.", **locator}]}


def test_verification_resolves_owned_exact_quote_and_keeps_stable_hash(client):
    http, snapshot, calls, loads, _ = client
    result = http.post("/api/v1/research-workspace/citations/verify", json=payload()).json()
    assert result["verdict"] == "supported"
    assert result["evidence"]["P1"] == {"source_id": "source", "media_id": 1, "start_char": 2, "end_char": 34,
        "navigation_version": snapshot["navigation_version"], "page_spans": [], "text_sha256": hashlib.sha256(snapshot["content"][2:34].encode()).hexdigest()}
    assert len(calls) == len(loads) == 1 and snapshot["content"][2:34] in calls[0]


@pytest.mark.parametrize("change", [
    {"quote": "canonical plain text"}, {"quote": "Repeated."}, {"source_id": "foreign-source"},
    {"navigation_version": "old"}, {"start_char": 0, "end_char": 31},
])
def test_invalid_or_foreign_locator_cannot_reach_judge(client, change):
    http, _, calls, _, _ = client
    result = http.post("/api/v1/research-workspace/citations/verify", json=payload(**change))
    assert result.status_code == 200
    assert result.json()["findings"][0]["status"] == "not_checked"
    assert result.json()["coverage"]["evaluated"] == 0 and calls == []


def test_foreign_workspace_is_not_found_before_any_source_or_provider_lookup(client):
    http, _, calls, loads, _ = client
    body = {**payload(), "workspace_id": "foreign"}
    assert http.post("/api/v1/research-workspace/citations/verify", json=body).status_code == 404
    assert calls == loads == []


def test_request_rejects_duplicate_ids_and_incomplete_offsets(client):
    http, _, calls, _, _ = client
    duplicate = payload()
    duplicate["evidence"] *= 2
    assert http.post("/api/v1/research-workspace/citations/verify", json=duplicate).status_code == 422
    assert http.post("/api/v1/research-workspace/citations/verify", json=payload(start_char=2)).status_code == 422
    assert calls == []


def test_verification_ingress_has_claim_auth_and_stable_rate_limit():
    from tldw_Server_API.app.core.AuthNZ.permissions import MEDIA_READ

    route = next(route for route in research_workspace.router.routes if route.path.endswith("/citations/verify"))
    calls = [dependency.call for dependency in route.dependant.dependencies]
    assert any(getattr(call, "_tldw_rate_limit_resource", None) == "research_workspace.citations.verify" for call in calls)
    values = [cell.cell_contents for call in calls for cell in (getattr(call, "__closure__", None) or [])]
    assert any(isinstance(value, list) and MEDIA_READ in value for value in values)


@pytest.mark.asyncio
@pytest.mark.parametrize("database_store", [False, True])
async def test_verification_exact_governor_mapping_survives_file_or_database_policy(database_store):
    from tldw_Server_API.app.core.Resource_Governance.policy_loader import PolicyLoader, PolicyReloadConfig

    class Store:
        async def get_latest_policy(self):
            return 1, {"rag.default": {"requests": {"rpm": 300}, "scopes": ["user", "api_key"]}}, {}, 0

    loader = PolicyLoader("tldw_Server_API/Config_Files/resource_governor_policies.yaml",
        PolicyReloadConfig(enabled=False), store=Store() if database_store else None)
    snapshot = await loader.load_once()
    assert snapshot.route_map["by_path"]["/api/v1/research-workspace/citations/verify"] == "rag.default"
    assert "rag.default" in snapshot.policies
    from tldw_Server_API.app.core.Resource_Governance.policy_resolver import PolicyResolver

    assert PolicyResolver(snapshot.route_map, []).resolve("/api/v1/research-workspace/citations/verify", "POST") == "rag.default"


@pytest.mark.asyncio
async def test_analyzer_passes_owned_credential_and_accounts_then_closes(monkeypatch):
    from starlette.requests import Request

    from tldw_Server_API.app.core.Chat import chat_service
    from tldw_Server_API.app.core.Provider_Credentials import provider_credential_runtime

    events = []
    owned_credentials = object()

    class Runtime:
        def __init__(self, **kwargs):
            events.append(("scope", kwargs["user_id"], kwargs["team_ids"], kwargs["org_ids"]))

        async def resolve(self, provider, *, model):
            events.append(("resolve", provider, model))
            return owned_credentials

        async def mark_used(self, credentials):
            assert credentials is owned_credentials
            events.append("used")

        async def close(self):
            events.append("closed")

    async def chat(**kwargs):
        assert kwargs[provider_credential_runtime.PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY] is owned_credentials
        assert kwargs["api_endpoint"] == "openai" and kwargs["model"] == "judge"
        assert kwargs["messages_payload"] == [{"role": "user", "content": "Exact prompt"}]
        assert kwargs["response_format"] == {"type": "json_object"} and not kwargs["streaming"]
        events.append("chat")
        return {"choices": [{"finish_reason": "stop", "message": {"content": "{}"}}]}

    monkeypatch.setattr(provider_credential_runtime, "ProviderCredentialRuntime", Runtime)
    monkeypatch.setattr(chat_service, "perform_chat_api_call_async", chat)
    request = Request({"type": "http", "headers": [], "method": "POST", "path": "/", "state": {}})
    dependency = research_workspace.get_citation_analyzer(request, SimpleNamespace(id=7))
    analyze = await dependency.__anext__()
    raw = await analyze("openai", "Claim", "Exact prompt", None, "Judge", 0,
        model_override="judge", response_format={"type": "json_object"})
    await dependency.aclose()
    assert raw["choices"][0]["finish_reason"] == "stop"
    assert events == [("scope", 7, [], []), ("resolve", "openai", "judge"), "chat", "used", "closed"]


def test_server_attribution_context_is_scoped_to_the_cited_contribution():
    first = "# First\n**Ada**\nPolicy first.\nAUTHOR'S NOTE: No individual view is asserted.\n"
    second = "# Second\n**Bea**\nPolicy second.\nAUTHOR'S NOTE: The author alone is responsible.\n"
    text = first + second
    nodes = [{"id": "first", "title": "First", "level": 1, "target_type": "char_range", "target_start": 0,
        "target_end": len(first)}, {"id": "second", "title": "Second", "level": 1, "target_type": "char_range",
        "target_start": len(first), "target_end": len(text)}]
    snapshot = {"content": text, "nodes": nodes, "navigation_version": "version", "page_map": []}
    start = text.index("Policy second.")
    rendered = workspace_citations._attribution_context(snapshot, {"id": "source", "media_id": 1}, start, start + 14)
    assert "Bea" in rendered and "The author alone is responsible." in rendered
    assert "Ada" not in rendered and "No individual view" not in rendered
    assert workspace_citations._attribution_context({**snapshot, "nodes": []}, {"id": "source", "media_id": 1}, start, start + 14) == ""


def _nested_source_context_fixture():
    text = "# Part\n## Chapter\n### First\n**Ada Example**\nFirst policy.\nAUTHOR'S NOTE: Ada alone.\n"
    second_start = len(text)
    text += "### Second\n**Bea Example**\nSecond policy.\nAUTHOR'S NOTE: Collective views.\nENDNOTES\nA reference.\n"
    nodes = [
        {"id": "part", "title": "Part", "level": 1, "parent_id": None, "target_type": "char_range", "target_start": 0, "target_end": len(text)},
        {"id": "chapter", "title": "Chapter", "level": 2, "parent_id": "part", "target_type": "char_range", "target_start": text.index("## Chapter"), "target_end": len(text)},
        {"id": "first", "title": "First", "level": 3, "parent_id": "chapter", "target_type": "char_range", "target_start": text.index("### First"), "target_end": second_start},
        {"id": "second", "title": "Second", "level": 3, "parent_id": "chapter", "target_type": "char_range", "target_start": second_start, "target_end": len(text)},
    ]
    return {"content": text, "nodes": nodes, "navigation_version": "version", "page_map": []}


@pytest.mark.parametrize("name,policy,note,other", [
    ("Ada Example", "First policy.", "Ada alone.", "Bea Example"),
    ("Bea Example", "Second policy.", "Collective views.", "Ada Example"),
])
def test_cited_nested_contribution_retains_its_byline_without_whole_part_omission(name, policy, note, other):
    snapshot = _nested_source_context_fixture()
    # A large Part has many unrelated chapters; their combined heading context
    # exceeds the existing cap even though this cited contribution is bounded.
    for index in range(12):
        left = len(snapshot["content"])
        snapshot["content"] += f"## Other chapter {index}\n" + "Unrelated body. " * 80
        snapshot["nodes"].append({"id": f"other-{index}", "title": f"Other {index}", "level": 2,
            "parent_id": "part", "target_type": "char_range", "target_start": left, "target_end": len(snapshot["content"])})
    snapshot["nodes"][0]["target_end"] = len(snapshot["content"])
    start = snapshot["content"].index(policy)
    rendered = workspace_citations._attribution_context(snapshot, {"id": "source", "media_id": 1}, start, start + len(policy))
    assert name in rendered and note in rendered and other not in rendered
    contexts = json.loads(rendered.rsplit("\n", 1)[-1])
    assert {context["title"] for context in contexts} == {"First" if name == "Ada Example" else "Second"}


@pytest.mark.parametrize("problem", ["missing_parent", "cycle", "overlapping_sibling"])
def test_nested_attribution_rejects_ambiguous_or_broken_ancestry(problem):
    snapshot = _nested_source_context_fixture()
    if problem == "missing_parent":
        snapshot["nodes"][3]["parent_id"] = "missing"
    elif problem == "cycle":
        snapshot["nodes"][1]["parent_id"] = "second"
    else:
        second = snapshot["nodes"][3]
        snapshot["nodes"].append({**second, "id": "overlap", "target_start": second["target_start"] + 1})
    start = snapshot["content"].index("Second policy.")
    assert workspace_citations._attribution_context(snapshot, {"id": "source", "media_id": 1}, start, start + 14) == ""


def test_nested_reference_does_not_inherit_body_author():
    snapshot = _nested_source_context_fixture()
    start = snapshot["content"].index("A reference.")
    rendered = workspace_citations._attribution_context(snapshot, {"id": "source", "media_id": 1}, start, start + 12)
    contexts = json.loads(rendered.rsplit("\n", 1)[-1])
    assert {context["type"] for context in contexts} == {"reference"}
    assert "Bea Example" not in rendered and "Ada Example" not in rendered


def test_quote_crossing_contributions_retains_both_bounded_contexts():
    snapshot = _nested_source_context_fixture()
    start = snapshot["content"].index("First policy.")
    end = snapshot["content"].index("Second policy.") + len("Second policy.")
    rendered = workspace_citations._attribution_context(snapshot, {"id": "source", "media_id": 1}, start, end)
    assert "Ada Example" in rendered and "Bea Example" in rendered
    contexts = json.loads(rendered.rsplit("\n", 1)[-1])
    assert {context["title"] for context in contexts} == {"First", "Second"}


def _shared_reference_snapshot():
    text = "# Part\n## 27. Financial Regulatory Agencies\n"
    first = len(text)
    text += ("### Consumer Financial Protection Bureau\n**Robert Bowes**\n"
        "The bureau should change.[53]\nAUTHOR'S NOTE: These are collective views.\n")
    second = len(text)
    text += ("### Securities and Exchange Commission\n**David Burton**\n"
        "A different policy applies.[54]\nENDNOTES\n"
        "53. Transfer details cite Bibliography Person.\n"
        "54. A separate reference.\n55. Closing reference.\n")
    chapter_end = len(text)
    text += "## 28. Other Agency\n**Unrelated Author**\nUNRELATED CHAPTER POLICY.[53]\nENDNOTES\n53. Unrelated note.\n"
    nodes = [
        {"id": "part", "title": "Part", "level": 1, "parent_id": None, "target_start": 0, "target_end": len(text)},
        {"id": "chapter", "title": "27. Financial Regulatory Agencies", "level": 2, "parent_id": "part",
            "target_start": text.index("## 27."), "target_end": chapter_end},
        {"id": "cfpb", "title": "Consumer Financial Protection Bureau", "level": 3, "parent_id": "chapter",
            "target_start": first, "target_end": second},
        {"id": "sec", "title": "Securities and Exchange Commission", "level": 3, "parent_id": "chapter",
            "target_start": second, "target_end": chapter_end},
        {"id": "other", "title": "28. Other Agency", "level": 2, "parent_id": "part",
            "target_start": chapter_end, "target_end": len(text)},
    ]
    return {"content": text, "nodes": [{**node, "target_type": "char_range"} for node in nodes],
        "navigation_version": "shared-notes-version", "page_map": []}


def test_shared_note_after_last_contributor_resolves_earlier_verified_contribution():
    snapshot = _shared_reference_snapshot()
    source = snapshot["content"]
    start, end = source.index("53. Transfer") + 4, source.index("54. A separate")
    rendered = workspace_citations._attribution_context(snapshot, {"id": "source", "media_id": 1}, start, end)
    assert "Robert Bowes" in rendered and "These are collective views." in rendered
    assert "David Burton" not in rendered and "Unrelated Author" not in rendered
    contexts = json.loads(rendered.rsplit("\n", 1)[-1])
    notes = [note for context in contexts for note in context.get("footnotes", [])]
    assert len(notes) == 1
    assert (notes[0]["start_char"], notes[0]["end_char"]) == (start - 4, end - 1)
    assert notes[0]["body_marker"] == {"start_char": source.index("[53]"), "end_char": source.index("[53]") + 4, "quote": "[53]"}


def test_shared_reference_passage_keeps_each_note_contribution_bounded():
    snapshot = _shared_reference_snapshot()
    source = snapshot["content"]
    start, end = source.index("53. Transfer") + 4, source.index("55. Closing")
    rendered = workspace_citations._attribution_context(snapshot, {"id": "source", "media_id": 1}, start, end)
    contexts = json.loads(rendered.rsplit("\n", 1)[-1])
    notes = [note for context in contexts for note in context.get("footnotes", [])]
    assert [(note["start_char"], note["end_char"]) for note in notes] == [
        (start - 4, source.index("54. A separate") - 1), (source.index("54. A separate"), end - 1)]
    assert [note["contribution_start_char"] for note in notes] == [source.index("### Consumer"), source.index("### Securities")]
    assert "Robert Bowes" in rendered and "David Burton" in rendered
    assert "Unrelated Author" not in rendered


@pytest.mark.parametrize("problem", ["missing_marker", "duplicate_marker", "broken_ancestry"])
def test_unresolved_shared_note_cannot_borrow_another_contribution_or_chapter(problem):
    snapshot = _shared_reference_snapshot()
    if problem == "missing_marker":
        snapshot["content"] = snapshot["content"].replace("[53]", "(53)", 1)
    elif problem == "duplicate_marker":
        snapshot["content"] = snapshot["content"].replace("[54]", "[53]", 1)
    else:
        snapshot["nodes"][2]["parent_id"] = "missing"
    source = snapshot["content"]
    start, end = source.index("53. Transfer") + 4, source.index("54. A separate")
    rendered = workspace_citations._attribution_context(snapshot, {"id": "source", "media_id": 1}, start, end)
    assert "Robert Bowes" not in rendered and "David Burton" not in rendered and "Unrelated Author" not in rendered


def test_shared_reference_api_keeps_owned_quote_and_invalidates_changed_link_context(client, monkeypatch):
    from tldw_Server_API.app.core.Research_Workspace.section_verification import evidence_digest

    http, _, calls, loads, workspace_db = client
    snapshot = _shared_reference_snapshot()

    async def load(media_id, db):
        loads.append((media_id, db))
        return snapshot

    monkeypatch.setattr(workspace_citations, "load_navigation_sections", load)
    source = snapshot["content"]
    start, end = source.index("53. Transfer") + 4, source.index("54. A separate")
    locator = {"id": "P1", "source_id": "source", "quote": source[start:end], "start_char": start,
        "end_char": end, "navigation_version": snapshot["navigation_version"]}
    request = {**payload(), "text": "The cited note is linked to the CFPB contribution. [P1]", "evidence": [locator]}
    first = http.post("/api/v1/research-workspace/citations/verify", json=request).json()
    resolved = asyncio.run(workspace_citations.resolve_workspace_citations("workspace", [locator],
        workspace_db=workspace_db, media_db=None))
    assert resolved["P1"]["text"] == source[start:end]
    assert "Robert Bowes" in resolved["P1"]["attribution_context"]
    evidence_prompt = calls[0].split("EVIDENCE:\n", 1)[1].split("\n\nVerify EVERY", 1)[0]
    assert source[start:end] in evidence_prompt
    assert "bureau should change" not in evidence_prompt and "different policy" not in evidence_prompt
    assert "UNRELATED CHAPTER POLICY" not in calls[0]

    snapshot["content"] = source.replace("[53]", "(53)", 1)
    second = http.post("/api/v1/research-workspace/citations/verify", json=request).json()
    changed = asyncio.run(workspace_citations.resolve_workspace_citations("workspace", [locator],
        workspace_db=workspace_db, media_db=None))
    assert first["evidence"]["P1"]["text_sha256"] == second["evidence"]["P1"]["text_sha256"]
    assert first["evidence"]["P1"]["attribution_context_sha256"] != second["evidence"]["P1"]["attribution_context_sha256"]
    assert evidence_digest(resolved) != evidence_digest(changed)

    before = len(loads), len(calls)
    for change in ({"source_id": "foreign-source"}, {"navigation_version": "stale"}):
        response = http.post("/api/v1/research-workspace/citations/verify",
            json={**request, "evidence": [{**locator, **change}]}).json()
        assert response["coverage"]["evaluated"] == 0
    assert len(loads) == before[0] + 1 and len(calls) == before[1]
