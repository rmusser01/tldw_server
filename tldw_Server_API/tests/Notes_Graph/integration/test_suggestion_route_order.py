from __future__ import annotations

from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.endpoints import notes_graph_suggestions

from .test_suggestion_endpoints import FINGERPRINT, NOTE_ID, FakeAPI, _app, _base_permissions


def test_static_rejection_reset_route_precedes_dynamic_suggestion_routes() -> None:
    paths = [route.path for route in notes_graph_suggestions.router.routes]
    reset = "/{note_id}/graph/suggestions/rejections/reset"
    accept = "/{note_id}/graph/suggestions/{suggestion_id}/accept"
    reject = "/{note_id}/graph/suggestions/{suggestion_id}/reject"

    assert reset in paths
    assert paths.index(reset) < paths.index(accept)
    assert paths.index(reset) < paths.index(reject)


def test_every_suggestion_route_declares_notes_token_scope_and_rbac_rate_limit() -> None:
    for route in notes_graph_suggestions.router.routes:
        calls = [dependency.call for dependency in route.dependant.dependencies]
        token_scopes = [
            call for call in calls if getattr(call, "_tldw_token_scope", False)
        ]
        rate_limits = [
            call
            for call in calls
            if getattr(call, "_tldw_rate_limit_resource", None) is not None
        ]
        assert len(token_scopes) == 1, route.path
        assert token_scopes[0]._tldw_token_scope_required == "notes"
        assert len(rate_limits) == 1, route.path


def test_static_rejection_reset_route_invokes_reset_not_dynamic_accept_or_reject() -> None:
    fake = FakeAPI()
    with TestClient(_app(fake, _base_permissions())) as client:
        response = client.post(
            f"/api/v1/notes/{NOTE_ID}/graph/suggestions/rejections/reset",
            json={
                "expected_rejection_revision": 1,
                "source_fingerprint": FINGERPRINT,
                "confirm": True,
            },
            headers={"Idempotency-Key": "reset-key"},
        )

    assert response.status_code == 200
    assert [name for name, _kwargs in fake.calls] == ["reset"]
