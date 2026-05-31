import hashlib
import json
import os
import tempfile
from typing import AsyncGenerator, Tuple

from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal
from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user
from tldw_Server_API.app.core.AuthNZ.permissions import CLAIMS_REVIEW
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal, AuthContext
from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase


def _seed_cluster_search_db() -> Tuple[str, int, int]:


    tmpdir = tempfile.mkdtemp(prefix="claims_search_")
    db_path = os.path.join(tmpdir, "media.db")
    db = MediaDatabase(db_path=db_path, client_id="1")
    db.initialize_db()
    content = "Alpha claim. Beta claim. Gamma claim. Delta claim."
    media_id, _, _ = db.add_media_with_keywords(title="Doc", media_type="text", content=content, keywords=None)
    chunk_hash = hashlib.sha256(content.encode()).hexdigest()
    db.upsert_claims([
        {
            "media_id": media_id,
            "chunk_index": 0,
            "span_start": None,
            "span_end": None,
            "claim_text": "Alpha claim",
            "confidence": 0.8,
            "extractor": "heuristic",
            "extractor_version": "v1",
            "chunk_hash": chunk_hash,
        },
        {
            "media_id": media_id,
            "chunk_index": 0,
            "span_start": None,
            "span_end": None,
            "claim_text": "Delta claim",
            "confidence": 0.8,
            "extractor": "heuristic",
            "extractor_version": "v1",
            "chunk_hash": chunk_hash,
        },
        {
            "media_id": media_id,
            "chunk_index": 0,
            "span_start": None,
            "span_end": None,
            "claim_text": "Beta claim",
            "confidence": 0.8,
            "extractor": "heuristic",
            "extractor_version": "v1",
            "chunk_hash": chunk_hash,
        },
        {
            "media_id": media_id,
            "chunk_index": 0,
            "span_start": None,
            "span_end": None,
            "claim_text": "Gamma claim",
            "confidence": 0.8,
            "extractor": "heuristic",
            "extractor_version": "v1",
            "chunk_hash": chunk_hash,
        },
    ])
    rows = db.execute_query(
        "SELECT id, claim_text FROM Claims WHERE media_id = ? AND deleted = 0",
        (media_id,),
    ).fetchall()
    claim_ids = {row["claim_text"]: int(row["id"]) for row in rows}
    cluster_a = db.create_claim_cluster(
        user_id="1",
        canonical_claim_text="Alpha claim",
        representative_claim_id=claim_ids["Alpha claim"],
    )
    cluster_b = db.create_claim_cluster(
        user_id="1",
        canonical_claim_text="Beta claim",
        representative_claim_id=claim_ids["Beta claim"],
    )
    cluster_a_id = int(cluster_a.get("id"))
    cluster_b_id = int(cluster_b.get("id"))
    db.add_claim_to_cluster(
        cluster_id=cluster_a_id,
        claim_id=claim_ids["Alpha claim"],
        similarity_score=1.0,
    )
    db.add_claim_to_cluster(
        cluster_id=cluster_a_id,
        claim_id=claim_ids["Gamma claim"],
        similarity_score=0.9,
    )
    db.add_claim_to_cluster(
        cluster_id=cluster_b_id,
        claim_id=claim_ids["Beta claim"],
        similarity_score=1.0,
    )
    db.execute_query(
        "UPDATE claim_cluster_membership SET cluster_joined_at = ? "
        "WHERE cluster_id = ? AND claim_id = ?",
        ("2026-01-01 00:00:00", cluster_a_id, claim_ids["Alpha claim"]),
        commit=True,
    )
    db.execute_query(
        "UPDATE claim_cluster_membership SET cluster_joined_at = ? "
        "WHERE cluster_id = ? AND claim_id = ?",
        ("2026-01-02 00:00:00", cluster_a_id, claim_ids["Gamma claim"]),
        commit=True,
    )
    db.execute_query(
        "UPDATE Claims SET review_status = ? WHERE id = ?",
        ("approved", claim_ids["Alpha claim"]),
        commit=True,
    )
    db.execute_query(
        "UPDATE Claims SET review_status = ? WHERE id = ?",
        ("rejected", claim_ids["Gamma claim"]),
        commit=True,
    )
    for index in range(3):
        db.insert_claim_notification(
            user_id="1",
            kind="review_update",
            target_user_id="1",
            target_review_group=None,
            resource_type="claim",
            resource_id=str(index + 1),
            payload_json=json.dumps({"index": index}),
        )
    db.close_connection()
    return db_path, cluster_a_id, cluster_b_id


def _principal_override():


    async def _override(request=None):
        principal = AuthPrincipal(
            kind="user",
            user_id=1,
            api_key_id=None,
            subject="reviewer",
            token_type="access",
            jti=None,
            roles=["reviewer"],
            permissions=[CLAIMS_REVIEW],
            is_admin=False,
            org_ids=[],
            team_ids=[],
        )
        if request is not None:
            try:
                request.state.auth = AuthContext(
                    principal=principal,
                    ip=None,
                    user_agent=None,
                    request_id=None,
                )
            except Exception:
                _ = None
        return principal

    return _override


def test_claims_cluster_links_and_search():


    from tldw_Server_API.app.main import app as fastapi_app

    class _User:
        def __init__(self) -> None:
            self.id = 1
            self.username = "reviewer"
            self.is_admin = False

    async def _override_user():
        return _User()

    db_path, cluster_a_id, cluster_b_id = _seed_cluster_search_db()

    async def _override_db() -> AsyncGenerator[MediaDatabase, None]:
        override_db = MediaDatabase(db_path=db_path, client_id="1")
        try:
            yield override_db
        finally:
            try:
                override_db.close_connection()
            except Exception:
                _ = None

    fastapi_app.dependency_overrides[get_auth_principal] = _principal_override()
    fastapi_app.dependency_overrides[get_request_user] = _override_user
    fastapi_app.dependency_overrides[get_media_db_for_user] = _override_db

    try:
        with TestClient(fastapi_app) as client:
            r_page = client.get("/api/v1/claims/search?q=claim&limit=2&offset=0")
            assert r_page.status_code == 200, r_page.text
            page_data = r_page.json()
            assert len(page_data["results"]) == 2
            assert page_data["total"] == 4
            assert page_data["pagination"] == {
                "mode": "offset",
                "limit": 2,
                "offset": 0,
                "total": 4,
                "has_more": True,
                "next_offset": 2,
            }
            assert page_data["has_more"] is True
            assert page_data["next_offset"] == 2

            r_digest = client.get("/api/v1/claims/notifications/digest?limit=2&offset=0")
            assert r_digest.status_code == 200, r_digest.text
            digest = r_digest.json()
            assert digest["total"] == 2
            assert digest["pagination"] == {
                "mode": "offset",
                "limit": 2,
                "offset": 0,
                "total": None,
                "has_more": True,
                "next_offset": 2,
            }
            assert digest["has_more"] is True
            assert digest["next_offset"] == 2

            r_claims_legacy = client.get("/api/v1/claims?limit=1&offset=0")
            assert r_claims_legacy.status_code == 200, r_claims_legacy.text
            assert isinstance(r_claims_legacy.json(), list)

            r_claims = client.get("/api/v1/claims?limit=2&offset=0&envelope=true")
            assert r_claims.status_code == 200, r_claims.text
            claims_page = r_claims.json()
            assert len(claims_page["items"]) == 2
            assert claims_page["pagination"] == {
                "mode": "offset",
                "limit": 2,
                "offset": 0,
                "total": None,
                "has_more": True,
                "next_offset": 2,
            }
            assert claims_page["has_more"] is True
            assert claims_page["next_offset"] == 2

            r_clusters_legacy = client.get("/api/v1/claims/clusters?limit=1&offset=0")
            assert r_clusters_legacy.status_code == 200, r_clusters_legacy.text
            assert isinstance(r_clusters_legacy.json(), list)

            r_clusters = client.get("/api/v1/claims/clusters?limit=1&offset=0&envelope=true")
            assert r_clusters.status_code == 200, r_clusters.text
            clusters_page = r_clusters.json()
            assert len(clusters_page["items"]) == 1
            assert clusters_page["pagination"] == {
                "mode": "offset",
                "limit": 1,
                "offset": 0,
                "total": None,
                "has_more": True,
                "next_offset": 1,
            }
            assert clusters_page["has_more"] is True
            assert clusters_page["next_offset"] == 1

            r_members_legacy = client.get(
                f"/api/v1/claims/clusters/{cluster_a_id}/members?limit=1&offset=0"
            )
            assert r_members_legacy.status_code == 200, r_members_legacy.text
            assert isinstance(r_members_legacy.json(), list)

            r_members = client.get(
                f"/api/v1/claims/clusters/{cluster_a_id}/members?limit=1&offset=0&envelope=true"
            )
            assert r_members.status_code == 200, r_members.text
            members_page = r_members.json()
            assert len(members_page["items"]) == 1
            assert members_page["pagination"] == {
                "mode": "offset",
                "limit": 1,
                "offset": 0,
                "total": None,
                "has_more": True,
                "next_offset": 1,
            }
            assert members_page["has_more"] is True
            assert members_page["next_offset"] == 1

            r_timeline_legacy = client.get(
                f"/api/v1/claims/clusters/{cluster_a_id}/timeline?limit=1&offset=0"
            )
            assert r_timeline_legacy.status_code == 200, r_timeline_legacy.text
            assert "pagination" not in r_timeline_legacy.json()

            r_timeline = client.get(
                f"/api/v1/claims/clusters/{cluster_a_id}/timeline?limit=1&offset=0&envelope=true"
            )
            assert r_timeline.status_code == 200, r_timeline.text
            timeline_page = r_timeline.json()
            assert len(timeline_page["timeline"]) == 1
            assert timeline_page["pagination"] == {
                "mode": "offset",
                "limit": 1,
                "offset": 0,
                "total": None,
                "has_more": True,
                "next_offset": 1,
            }
            assert timeline_page["has_more"] is True
            assert timeline_page["next_offset"] == 1

            r_evidence_legacy = client.get(
                f"/api/v1/claims/clusters/{cluster_a_id}/evidence?limit=1&offset=0"
            )
            assert r_evidence_legacy.status_code == 200, r_evidence_legacy.text
            assert "pagination" not in r_evidence_legacy.json()

            r_evidence = client.get(
                f"/api/v1/claims/clusters/{cluster_a_id}/evidence?limit=1&offset=0&envelope=true"
            )
            assert r_evidence.status_code == 200, r_evidence.text
            evidence_page = r_evidence.json()
            assert sum(evidence_page["counts"].values()) == 1
            assert evidence_page["pagination"] == {
                "mode": "offset",
                "limit": 1,
                "offset": 0,
                "total": None,
                "has_more": True,
                "next_offset": 1,
            }
            assert evidence_page["has_more"] is True
            assert evidence_page["next_offset"] == 1

            r_search = client.get("/api/v1/claims/search?q=claim&group_by_cluster=true&limit=10")
            assert r_search.status_code == 200, r_search.text
            data = r_search.json()
            assert data["group_by_cluster"] is True
            clusters = data.get("clusters") or []
            orphaned = data.get("orphaned") or []
            assert len(clusters) == 2
            assert len(orphaned) == 1

            r_link = client.post(
                f"/api/v1/claims/clusters/{cluster_a_id}/links",
                json={"child_cluster_id": cluster_b_id, "relation_type": "related"},
            )
            assert r_link.status_code == 200, r_link.text
            payload = r_link.json()
            assert payload["child_cluster_id"] == cluster_b_id

            r_links = client.get(f"/api/v1/claims/clusters/{cluster_a_id}/links")
            assert r_links.status_code == 200, r_links.text
            assert any(
                item["child_cluster_id"] == cluster_b_id and item["direction"] == "outbound"
                for item in r_links.json()
            )

            r_inbound = client.get(f"/api/v1/claims/clusters/{cluster_b_id}/links?direction=inbound")
            assert r_inbound.status_code == 200, r_inbound.text
            assert any(
                item["parent_cluster_id"] == cluster_a_id and item["direction"] == "inbound"
                for item in r_inbound.json()
            )

            r_del = client.delete(f"/api/v1/claims/clusters/{cluster_a_id}/links/{cluster_b_id}")
            assert r_del.status_code == 200, r_del.text
            assert r_del.json()["status"] == "deleted"
    finally:
        fastapi_app.dependency_overrides.pop(get_auth_principal, None)
        fastapi_app.dependency_overrides.pop(get_request_user, None)
        fastapi_app.dependency_overrides.pop(get_media_db_for_user, None)
