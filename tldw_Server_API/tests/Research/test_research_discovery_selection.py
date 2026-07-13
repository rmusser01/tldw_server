from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest


def _candidate(result_id: str, label: str, **overrides):
    from tldw_Server_API.app.core.Research.discovery.oa import build_candidate_id

    candidate = {
        "candidate_type": "pdf",
        "safe_url": f"https://repo.example/{label}.pdf",
        "resolver_reference": None,
        "url_redacted": False,
        "requires_reresolution": False,
        "provider": "unpaywall",
        "access_status": "open",
        "license_hint": "cc-by",
        "content_type_hint": "application/pdf",
        "rank": 1,
        "confidence": 0.9,
        "warnings": [],
    }
    candidate.update(overrides)
    candidate["candidate_id"] = build_candidate_id(
        result_fingerprint=f"doi:10.1000/{result_id}",
        candidate_type=candidate["candidate_type"],
        provider=candidate["provider"],
        safe_url=candidate["safe_url"],
        resolver_reference=candidate["resolver_reference"],
    )
    return candidate


def _result(result_id: str, candidate_id: str, **overrides):
    result = {
        "result_id": result_id,
        "fingerprint": f"doi:10.1000/{result_id}",
        "primary_source_id": "openalex",
        "primary_provider": "openalex",
        "title": f"Title {result_id}",
        "authors": ["Ada Lovelace", "Grace Hopper"],
        "doi": f"10.1000/{result_id}",
        "pmid": "123",
        "pmcid": "PMC456",
        "arxiv_id": "2401.00001",
        "provider_ids": {
            "openalex_id": f"W-{result_id}",
            "api_key": "SECRET",
            "unsafe": "https://provider.example/item?token=SECRET",
        },
        "canonical_url": f"https://doi.org/10.1000/{result_id}",
        "safe_metadata": {"journal": "Examples", "api_key": "SECRET"},
    }
    result.update(overrides)
    result.setdefault("oa_candidates", [_candidate(result_id, candidate_id)])
    return result


def _pair(result_id: str, candidate_label: str, **candidate_overrides):
    return (result_id, _candidate(result_id, candidate_label, **candidate_overrides)["candidate_id"])


def _create_snapshot(db, *, owner_user_id="owner-1", results=None, retention_hours=24):
    return db.create_discovery_snapshot(
        owner_user_id=owner_user_id,
        query="open access examples",
        request_json={},
        response_json={"results": results if results is not None else [_result("r1", "c1")]},
        effective_config_json={},
        catalog_version="research-discovery-v1",
        retention_hours=retention_hours,
    )


def test_resolves_server_owned_pdf_selections_in_request_order(tmp_path):
    from tldw_Server_API.app.core.DB_Management.ResearchSessionsDB import ResearchSessionsDB
    from tldw_Server_API.app.core.Research.discovery.selection import resolve_discovery_selections

    db = ResearchSessionsDB(tmp_path / "research.db")
    snapshot = _create_snapshot(
        db,
        results=[_result("r1", "c1"), _result("r2", "c2")],
    )

    resolved = resolve_discovery_selections(
        owner_user_id="owner-1",
        discovery_id=snapshot.id,
        selections=(_pair("r2", "c2"), _pair("r1", "c1")),
        snapshot_db=db,
    )

    assert [item.result_id for item in resolved] == ["r2", "r1"]
    assert resolved[0].candidate_type == "pdf"
    assert resolved[0].url == "https://repo.example/c2.pdf"
    assert resolved[0].canonical_url == "https://doi.org/10.1000/r2"
    assert resolved[0].authors == ("Ada Lovelace", "Grace Hopper")
    assert resolved[0].identifiers == {
        "arxiv_id": "2401.00001",
        "doi": "10.1000/r2",
        "openalex_id": "W-r2",
        "pmcid": "PMC456",
        "pmid": "123",
    }
    assert resolved[0].source_id == "openalex"
    assert resolved[0].provider == "unpaywall"
    assert resolved[0].safe_metadata == {"journal": "Examples"}


def test_resolves_legacy_nullable_and_blank_optional_fields(tmp_path):
    from tldw_Server_API.app.core.DB_Management.ResearchSessionsDB import ResearchSessionsDB
    from tldw_Server_API.app.core.Research.discovery.selection import resolve_discovery_selections

    db = ResearchSessionsDB(tmp_path / "research.db")
    candidate = _candidate(
        "r1",
        "c1",
        access_status="  ",
        license_hint="",
        content_type_hint=" ",
    )
    candidate["resolver_reference"] = " "
    result = _result(
        "r1",
        "c1",
        canonical_url=" ",
        provider_ids=None,
        safe_metadata=None,
        oa_candidates=[candidate],
    )
    snapshot = _create_snapshot(db, results=[result])

    resolved = resolve_discovery_selections(
        owner_user_id="owner-1",
        discovery_id=snapshot.id,
        selections=(("r1", candidate["candidate_id"]),),
        snapshot_db=db,
    )[0]

    assert resolved.canonical_url is None
    assert resolved.access_status is None
    assert resolved.license_hint is None
    assert resolved.safe_metadata == {}


def test_coerces_numeric_identifier_values_without_weakening_structural_fields(tmp_path):
    from tldw_Server_API.app.core.DB_Management.ResearchSessionsDB import ResearchSessionsDB
    from tldw_Server_API.app.core.Research.discovery.selection import resolve_discovery_selections

    db = ResearchSessionsDB(tmp_path / "research.db")
    result = _result(
        "r1",
        "c1",
        pmid=123,
        provider_ids={"openalex_id": 456.5, "nested_metadata": {"ignored": True}},
    )
    snapshot = _create_snapshot(db, results=[result])

    resolved = resolve_discovery_selections(
        owner_user_id="owner-1",
        discovery_id=snapshot.id,
        selections=(_pair("r1", "c1"),),
        snapshot_db=db,
    )[0]

    assert resolved.identifiers["pmid"] == "123"
    assert resolved.identifiers["openalex_id"] == "456.5"
    assert "nested_metadata" not in resolved.identifiers


def test_resolved_selection_is_frozen(tmp_path):
    from tldw_Server_API.app.core.DB_Management.ResearchSessionsDB import ResearchSessionsDB
    from tldw_Server_API.app.core.Research.discovery.selection import resolve_discovery_selections

    db = ResearchSessionsDB(tmp_path / "research.db")
    snapshot = _create_snapshot(db)
    resolved = resolve_discovery_selections(
        owner_user_id="owner-1",
        discovery_id=snapshot.id,
        selections=(_pair("r1", "c1"),),
        snapshot_db=db,
    )[0]

    with pytest.raises(FrozenInstanceError):
        resolved.title = "client replacement"


def test_resolves_provider_id_fingerprint(tmp_path):
    from tldw_Server_API.app.core.DB_Management.ResearchSessionsDB import ResearchSessionsDB
    from tldw_Server_API.app.core.Research.discovery.identity import build_fingerprint
    from tldw_Server_API.app.core.Research.discovery.oa import build_candidate_id
    from tldw_Server_API.app.core.Research.discovery.selection import resolve_discovery_selections

    db = ResearchSessionsDB(tmp_path / "research.db")
    result = _result(
        "r1",
        "c1",
        doi=None,
        pmid=None,
        pmcid=None,
        arxiv_id=None,
        provider_ids={"openalex_id": "W-r1"},
    )
    result["fingerprint"] = build_fingerprint(dict(result, source_id="openalex", provider="openalex"))
    result["oa_candidates"] = [_candidate("r1", "c1")]
    result["oa_candidates"][0]["candidate_id"] = build_candidate_id(
        result_fingerprint=result["fingerprint"],
        candidate_type="pdf",
        provider="unpaywall",
        safe_url="https://repo.example/c1.pdf",
        resolver_reference=None,
    )
    snapshot = _create_snapshot(db, results=[result])

    resolved = resolve_discovery_selections(
        owner_user_id="owner-1",
        discovery_id=snapshot.id,
        selections=(("r1", result["oa_candidates"][0]["candidate_id"]),),
        snapshot_db=db,
    )

    assert resolved[0].fingerprint == result["fingerprint"]


def test_default_db_uses_owner_scoped_research_path(monkeypatch, tmp_path):
    from tldw_Server_API.app.core.Research.discovery import selection as selection_module

    real_db = selection_module.ResearchSessionsDB(tmp_path / "research.db")
    snapshot = _create_snapshot(real_db)
    captured = {}

    class FakeResearchSessionsDB:
        def __init__(self, db_path):
            captured["db_path"] = db_path

        def get_discovery_snapshot(self, discovery_id, *, owner_user_id):
            return real_db.get_discovery_snapshot(discovery_id, owner_user_id=owner_user_id)

    expected_path = tmp_path / "owner-1-research.db"
    monkeypatch.setattr(
        selection_module.DatabasePaths,
        "get_research_sessions_db_path",
        lambda owner_user_id: expected_path,
    )
    monkeypatch.setattr(selection_module, "ResearchSessionsDB", FakeResearchSessionsDB)

    selection_module.resolve_discovery_selections(
        owner_user_id="owner-1",
        discovery_id=snapshot.id,
        selections=(_pair("r1", "c1"),),
    )

    assert captured["db_path"] == expected_path


@pytest.mark.parametrize("mode", ["missing", "expired", "foreign"])
def test_unavailable_snapshots_share_one_public_error(tmp_path, mode):
    from tldw_Server_API.app.core.DB_Management.ResearchSessionsDB import ResearchSessionsDB
    from tldw_Server_API.app.core.exceptions import ResearchDiscoveryValidationError
    from tldw_Server_API.app.core.Research.discovery.selection import resolve_discovery_selections

    db = ResearchSessionsDB(tmp_path / "research.db")
    if mode == "missing":
        discovery_id = "rd_missing"
        owner_user_id = "owner-1"
    else:
        snapshot = _create_snapshot(
            db,
            owner_user_id="owner-2" if mode == "foreign" else "owner-1",
            retention_hours=-1 if mode == "expired" else 24,
        )
        discovery_id = snapshot.id
        owner_user_id = "owner-1"

    with pytest.raises(ResearchDiscoveryValidationError) as exc_info:
        resolve_discovery_selections(
            owner_user_id=owner_user_id,
            discovery_id=discovery_id,
            selections=(_pair("r1", "c1"),),
            snapshot_db=db,
        )

    assert exc_info.value.public_detail == "research_discovery_snapshot_unavailable"


@pytest.mark.parametrize(
    ("response_json", "error"),
    [
        ({}, "research_discovery_snapshot_malformed"),
        ({"results": "not-a-list"}, "research_discovery_snapshot_malformed"),
        ({"results": [_result("r1", "c1"), _result("r1", "c2")]}, "research_discovery_snapshot_malformed"),
        (
            {"results": [_result("r1", "c1", oa_candidates=[_candidate("r1", "c1"), _candidate("r1", "c1")])]},
            "research_discovery_snapshot_malformed",
        ),
        (
            {"results": [_result("r1", "c1", fingerprint="doi:10.1000/tampered")]},
            "research_discovery_snapshot_malformed",
        ),
        (
            {"results": [_result("r1", "c1", title=123)]},
            "research_discovery_snapshot_malformed",
        ),
    ],
)
def test_rejects_malformed_snapshot_payloads(tmp_path, response_json, error):
    from tldw_Server_API.app.core.DB_Management.ResearchSessionsDB import ResearchSessionsDB
    from tldw_Server_API.app.core.exceptions import ResearchDiscoveryValidationError
    from tldw_Server_API.app.core.Research.discovery.selection import resolve_discovery_selections

    db = ResearchSessionsDB(tmp_path / "research.db")
    snapshot = db.create_discovery_snapshot(
        owner_user_id="owner-1",
        query="malformed",
        request_json={},
        response_json=response_json,
        effective_config_json={},
        catalog_version="research-discovery-v1",
    )

    with pytest.raises(ResearchDiscoveryValidationError) as exc_info:
        resolve_discovery_selections(
            owner_user_id="owner-1",
            discovery_id=snapshot.id,
            selections=(_pair("r1", "c1"),),
            snapshot_db=db,
        )

    assert exc_info.value.public_detail == error


def test_rejects_tampered_candidate_identity(tmp_path):
    from tldw_Server_API.app.core.DB_Management.ResearchSessionsDB import ResearchSessionsDB
    from tldw_Server_API.app.core.exceptions import ResearchDiscoveryValidationError
    from tldw_Server_API.app.core.Research.discovery.selection import resolve_discovery_selections

    db = ResearchSessionsDB(tmp_path / "research.db")
    tampered_id = "oa_candidate:tampered"
    snapshot = _create_snapshot(
        db,
        results=[
            _result(
                "r1",
                "c1",
                oa_candidates=[dict(_candidate("r1", "c1"), candidate_id=tampered_id)],
            )
        ],
    )

    with pytest.raises(ResearchDiscoveryValidationError) as exc_info:
        resolve_discovery_selections(
            owner_user_id="owner-1",
            discovery_id=snapshot.id,
            selections=(("r1", tampered_id),),
            snapshot_db=db,
        )

    assert exc_info.value.public_detail == "research_discovery_snapshot_malformed"


@pytest.mark.parametrize(
    ("selections", "error"),
    [
        ((), "research_discovery_selections_required"),
        ((_pair("r1", "c1"), _pair("r1", "c1")), "research_discovery_duplicate_selection"),
        ((("r1",),), "research_discovery_selection_malformed"),
        ((("r1", ""),), "research_discovery_selection_malformed"),
        (tuple((f"r{i}", f"c{i}") for i in range(6)), "research_discovery_selection_limit_exceeded"),
    ],
)
def test_rejects_invalid_selection_pairs(tmp_path, selections, error):
    from tldw_Server_API.app.core.DB_Management.ResearchSessionsDB import ResearchSessionsDB
    from tldw_Server_API.app.core.exceptions import ResearchDiscoveryValidationError
    from tldw_Server_API.app.core.Research.discovery.selection import resolve_discovery_selections

    db = ResearchSessionsDB(tmp_path / "research.db")
    snapshot = _create_snapshot(db)

    with pytest.raises(ResearchDiscoveryValidationError) as exc_info:
        resolve_discovery_selections(
            owner_user_id="owner-1",
            discovery_id=snapshot.id,
            selections=selections,
            snapshot_db=db,
        )

    assert exc_info.value.public_detail == error


@pytest.mark.parametrize(
    ("selection", "error"),
    [
        (("missing", _pair("r1", "c1")[1]), "research_discovery_selection_not_found"),
        (("r1", "missing"), "research_discovery_selection_not_found"),
        (("r1", _pair("r2", "c2")[1]), "research_discovery_selection_not_found"),
    ],
)
def test_rejects_missing_or_mismatched_selection_pairs(tmp_path, selection, error):
    from tldw_Server_API.app.core.DB_Management.ResearchSessionsDB import ResearchSessionsDB
    from tldw_Server_API.app.core.exceptions import ResearchDiscoveryValidationError
    from tldw_Server_API.app.core.Research.discovery.selection import resolve_discovery_selections

    db = ResearchSessionsDB(tmp_path / "research.db")
    snapshot = _create_snapshot(db, results=[_result("r1", "c1"), _result("r2", "c2")])

    with pytest.raises(ResearchDiscoveryValidationError) as exc_info:
        resolve_discovery_selections(
            owner_user_id="owner-1",
            discovery_id=snapshot.id,
            selections=(selection,),
            snapshot_db=db,
        )

    assert exc_info.value.public_detail == error


@pytest.mark.parametrize(
    "candidate_overrides",
    [
        {"candidate_type": "html_full_text"},
        {"safe_url": None},
        {"url_redacted": True},
        {"requires_reresolution": True},
    ],
)
def test_rejects_candidates_outside_phase2a_pdf_contract(tmp_path, candidate_overrides):
    from tldw_Server_API.app.core.DB_Management.ResearchSessionsDB import ResearchSessionsDB
    from tldw_Server_API.app.core.exceptions import ResearchDiscoveryValidationError
    from tldw_Server_API.app.core.Research.discovery.selection import resolve_discovery_selections

    db = ResearchSessionsDB(tmp_path / "research.db")
    snapshot = _create_snapshot(
        db,
        results=[
            _result(
                "r1",
                "c1",
                oa_candidates=[_candidate("r1", "c1", **candidate_overrides)],
            )
        ],
    )

    with pytest.raises(ResearchDiscoveryValidationError) as exc_info:
        resolve_discovery_selections(
            owner_user_id="owner-1",
            discovery_id=snapshot.id,
            selections=(_pair("r1", "c1", **candidate_overrides),),
            snapshot_db=db,
        )

    assert exc_info.value.public_detail == "research_discovery_candidate_not_ingestable"
