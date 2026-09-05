"""Snapshot authorization and path-free API contracts."""

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps import auth_deps
from tldw_Server_API.tests.LLM_Local.test_llamacpp_admin_config_api import (
    _admin_principal,
    _make_app_with_manager,
    _ManagerWithoutHandler,
)

pytestmark = pytest.mark.integration

ROUTES = [
    ("GET", "/slots"),
    ("GET", "/snapshots"),
    ("POST", "/snapshots"),
    ("POST", "/snapshots/s1/restore"),
    ("DELETE", "/snapshots/s1"),
    ("GET", "/snapshot-operations/o1"),
]


@pytest.mark.parametrize("method,suffix", ROUTES)
def test_non_admin_denied_before_missing_owner_lookup(method, suffix):
    app = _make_app_with_manager(_ManagerWithoutHandler())

    async def ordinary_user():
        return _admin_principal().model_copy(update={"roles": [], "is_admin": False})

    app.dependency_overrides[auth_deps.get_auth_principal] = ordinary_user
    with TestClient(app) as client:
        response = client.request(method, "/api/v1/llamacpp/profiles/missing" + suffix)
    assert response.status_code == 403


@pytest.mark.parametrize("suffix", ["/snapshots", "/snapshots/s1/restore"])
def test_snapshot_mutation_rejects_paths(suffix):
    app = _make_app_with_manager(_ManagerWithoutHandler())
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/llamacpp/profiles/p1" + suffix,
            json={
                "slot_id": 0,
                "expected_launch_generation": "g1",
                "request_id": "token",
                "path": "/outside/cache.bin",
                "replace_confirmed": True,
            },
        )
    assert response.status_code == 422


@pytest.mark.parametrize("query", ["offset=-1", "limit=0", "limit=101"])
def test_catalog_rejects_invalid_pagination(query):
    with TestClient(_make_app_with_manager(_ManagerWithoutHandler())) as client:
        response = client.get("/api/v1/llamacpp/profiles/p1/snapshots?" + query)
    assert response.status_code == 422


@pytest.mark.parametrize("method,suffix", ROUTES)
def test_all_routes_enforce_rate_limit(method, suffix):
    from fastapi import HTTPException

    app = _make_app_with_manager(_ManagerWithoutHandler())

    async def limited():
        raise HTTPException(429, "rate limited")

    app.dependency_overrides[auth_deps.check_rate_limit] = limited
    with TestClient(app) as client:
        response = client.request(method, "/api/v1/llamacpp/profiles/p1" + suffix)
    assert response.status_code == 429


@pytest.mark.parametrize(
    "state,recovery_action",
    [
        ("complete", "none"),
        ("failed", "retry_manually"),
        ("outcome_unknown", "stop_runtime"),
    ],
)
def test_real_supervisor_stopped_catalog_token_and_cross_profile_receipts(tmp_path, state, recovery_action):
    from tldw_Server_API.app.core.Local_LLM.llamacpp_profile_store import JsonLlamaCppProfileStore
    from tldw_Server_API.app.core.Local_LLM.llamacpp_runtime_models import LlamaCppProfile
    from tldw_Server_API.app.core.Local_LLM.llamacpp_snapshot_models import OperationReceipt
    from tldw_Server_API.app.core.Local_LLM.llamacpp_snapshot_operations import SnapshotOperations
    from tldw_Server_API.app.core.Local_LLM.llamacpp_snapshot_store import SnapshotStore
    from tldw_Server_API.app.core.Local_LLM.llamacpp_supervisor_service import LlamaCppSupervisor
    from tldw_Server_API.tests.LLM_Local.test_llamacpp_supervisor_service import make_config

    config = make_config(tmp_path)
    profiles = JsonLlamaCppProfileStore(tmp_path / "profiles.json")
    profiles.upsert(LlamaCppProfile(profile_id="p1", name="one"))
    profiles.upsert(LlamaCppProfile(profile_id="p2", name="two", port=8081))
    supervisor = LlamaCppSupervisor(config=config, store=profiles)
    with SnapshotStore(tmp_path / "snapshots") as storage:
        supervisor._snapshots = SnapshotOperations(storage)
        operation = OperationReceipt(
            profile_id="p1",
            operation_id="operation1",
            launch_generation="generation",
            request_digest="a" * 64,
            kind="save",
            state=state,
        )
        assert operation.recovery_action == recovery_action
        assert "recovery_action" not in operation.model_dump(mode="json")
        assert OperationReceipt.model_validate_json(operation.model_dump_json()) == operation
        storage.write_receipt(operation)
        app = _make_app_with_manager(
            SimpleNamespace(llamacpp_supervisor=supervisor, logger=_ManagerWithoutHandler.logger)
        )
        with TestClient(app) as client:
            slots = client.get("/api/v1/llamacpp/profiles/p1/slots")
            assert slots.status_code == 200
            assert slots.json()["latest_operation_id"] == "operation1"
            assert slots.json()["request_id"]
            assert slots.json()["slots"] == []
            catalog = client.get("/api/v1/llamacpp/profiles/p1/snapshots")
            assert catalog.status_code == 200 and catalog.json()["total"] == 0
            receipt = client.get("/api/v1/llamacpp/profiles/p1/snapshot-operations/operation1")
            assert receipt.status_code == 200
            assert receipt.json()["recovery_action"] == recovery_action
            assert "request_digest" not in receipt.json()
            assert "dispatched" not in receipt.json()
            cross = client.get("/api/v1/llamacpp/profiles/p2/snapshot-operations/operation1")
            assert cross.status_code == 404
            for method, suffix in [
                ("DELETE", "/snapshots/other-profile-snapshot"),
                ("GET", "/snapshot-operations/other"),
            ]:
                assert client.request(method, "/api/v1/llamacpp/profiles/p2" + suffix).status_code == 404


def test_authenticated_save_restore_delete_with_real_supervisor_factory(tmp_path):
    import asyncio

    from tldw_Server_API.app.core.Local_LLM.llamacpp_profile_store import JsonLlamaCppProfileStore
    from tldw_Server_API.app.core.Local_LLM.llamacpp_runtime_models import LlamaCppProfile
    from tldw_Server_API.app.core.Local_LLM.llamacpp_supervisor_service import LlamaCppSupervisor
    from tldw_Server_API.tests.LLM_Local.test_llamacpp_snapshot_operations import Runner, Transport
    from tldw_Server_API.tests.LLM_Local.test_llamacpp_supervisor_service import make_config, make_model

    config = make_config(tmp_path)
    model = make_model(config)
    profiles = JsonLlamaCppProfileStore(tmp_path / "profiles.json")
    profiles.upsert(LlamaCppProfile(profile_id="p1", name="one", model_path=str(model), snapshots_enabled=True))
    supervisor = LlamaCppSupervisor(config=config, store=profiles)
    app = _make_app_with_manager(SimpleNamespace(llamacpp_supervisor=supervisor, logger=_ManagerWithoutHandler.logger))
    base = "/api/v1/llamacpp/profiles/p1"
    with TestClient(app) as client:
        # Exercise the production lazy owner/key factory before adding an owned fake child.
        stopped = client.get(base + "/slots")
        assert stopped.status_code == 200 and stopped.json()["capability"] == "stopped"
        service = supervisor._snapshots
        runner = Runner(model, config.executable_path)
        runner.snapshot_working = service.store.launch_directory("p1", "generation1")
        supervisor._runners["p1"] = runner
        service.supported_builds = {runner.snapshot_fingerprint.executable_sha256}
        service.transport = Transport(runner)

        async def settle():
            await asyncio.gather(*list(service.tasks.values()))

        slots = client.get(base + "/slots").json()
        assert slots["capability"] == "ready"
        assert slots["slots"] == [{"slot_id": 0, "busy": False, "token_count": 4}]
        body = {
            "slot_id": 0,
            "expected_launch_generation": slots["launch_generation"],
            "request_id": slots["request_id"],
        }
        saved = client.post(base + "/snapshots", json=body)
        assert saved.status_code == 202
        client.portal.call(settle)
        catalog = client.get(base + "/snapshots")
        assert catalog.status_code == 200
        item = catalog.json()["snapshots"][0]
        assert item["compatibility"] == "compatible" and item["byte_count"] == 5
        assert "fingerprint" not in item and "actor_id" not in item and "sha256" not in item
        assert str(tmp_path) not in catalog.text
        snapshot_id = item["snapshot_id"]
        assert client.delete(base).status_code == 409
        body.update(request_id=client.get(base + "/slots").json()["request_id"], replace_confirmed=True)
        restored = client.post(base + f"/snapshots/{snapshot_id}/restore", json=body)
        assert restored.status_code == 202
        client.portal.call(settle)
        receipt = client.get(base + "/snapshot-operations/" + restored.json()["operation_id"])
        assert receipt.json()["state"] == "complete"
        assert receipt.json()["token_count"] == 4
        assert client.delete(base + f"/snapshots/{snapshot_id}").status_code == 200
        assert client.get(base + "/snapshots").json()["total"] == 0
        service.store.close()
