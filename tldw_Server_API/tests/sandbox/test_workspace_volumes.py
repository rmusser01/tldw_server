from __future__ import annotations

from tldw_Server_API.app.core.Sandbox.models import WorkspaceVolume, WorkspaceVolumeState
from tldw_Server_API.app.core.Sandbox.store import IdempotencyConflict, InMemoryStore, SQLiteStore
from tldw_Server_API.app.core.Sandbox.workspace_volumes import SandboxWorkspaceVolumeService


def test_sqlite_store_persists_workspace_volume_idempotency_and_lists_by_workspace(tmp_path):
    store = SQLiteStore(db_path=str(tmp_path / "sandbox.db"))
    service = SandboxWorkspaceVolumeService(store=store)

    first = service.provision_workspace_volume(
        workspace_id="workspace-1",
        user_id="user-1",
        display_name="Project root",
        idempotency_key="provision-root",
        requested_runtime="docker",
        diagnostics={
            "message": "cannot allocate host path /Users/alice/private/project",
            "api_key": "sk-secret-value",
            "nested": {"token": "secret-token", "path": "/private/tmp/raw-host"},
        },
    )
    retry = service.provision_workspace_volume(
        workspace_id="workspace-1",
        user_id="user-1",
        display_name="Project root",
        idempotency_key="provision-root",
        requested_runtime="docker",
    )

    assert retry.id == first.id
    assert retry.state is WorkspaceVolumeState.not_configured
    assert retry.runtime == "docker"
    assert retry.mount_path is None
    assert store.find_workspace_volume_by_idempotency(
        user_id="user-1",
        workspace_id="workspace-1",
        idempotency_key="provision-root",
    ).id == first.id
    assert [volume.id for volume in store.list_workspace_volumes_for_workspace("workspace-1")] == [first.id]

    persisted = store.get_workspace_volume(first.id)
    assert persisted is not None
    diagnostic_text = repr(persisted.diagnostics)
    assert "/Users/alice" not in diagnostic_text
    assert "/private/tmp" not in diagnostic_text
    assert "sk-secret-value" not in diagnostic_text
    assert "secret-token" not in diagnostic_text
    assert diagnostic_text.count("x") < 1000


def test_service_rejects_idempotency_conflict_for_different_workspace_or_runtime(tmp_path):
    store = SQLiteStore(db_path=str(tmp_path / "sandbox.db"))
    service = SandboxWorkspaceVolumeService(store=store)
    service.provision_workspace_volume(
        workspace_id="workspace-1",
        user_id="user-1",
        display_name="Project root",
        idempotency_key="provision-root",
        requested_runtime="docker",
    )

    other_workspace = service.provision_workspace_volume(
        workspace_id="workspace-2",
        user_id="user-1",
        display_name="Project root",
        idempotency_key="provision-root",
        requested_runtime="docker",
    )
    assert other_workspace.workspace_id == "workspace-2"
    assert other_workspace.id != store.find_workspace_volume_by_idempotency(
        user_id="user-1",
        workspace_id="workspace-1",
        idempotency_key="provision-root",
    ).id

    try:
        service.provision_workspace_volume(
            workspace_id="workspace-1",
            user_id="user-1",
            display_name="Project root",
            idempotency_key="provision-root",
            requested_runtime="vz_linux",
        )
    except IdempotencyConflict as exc:
        assert exc.original_id
    else:
        raise AssertionError("Expected idempotency conflict for changed runtime")


def test_service_validate_and_resolve_fail_closed_for_wrong_owner_and_missing_mount(tmp_path):
    store = SQLiteStore(db_path=str(tmp_path / "sandbox.db"))
    service = SandboxWorkspaceVolumeService(store=store)
    volume = service.provision_workspace_volume(
        workspace_id="workspace-1",
        user_id="user-1",
        display_name="Project root",
        idempotency_key=None,
        requested_runtime="docker",
    )

    wrong_user = service.validate_workspace_volume(
        workspace_id="workspace-1",
        user_id="user-2",
        sandbox_volume_id=volume.id,
    )
    assert wrong_user.state == "unavailable"
    assert wrong_user.reason_code == "workspace_sandbox_volume_owner_mismatch"

    binding = service.validate_workspace_volume(
        workspace_id="workspace-1",
        user_id="user-1",
        sandbox_volume_id=volume.id,
    )
    mount = service.resolve_workspace_volume_mount(
        workspace_id="workspace-1",
        root_id="primary",
        sandbox_volume_id=volume.id,
    )

    assert binding.state == "not_configured"
    assert binding.display_name == "Project root"
    assert mount.state == "not_configured"
    assert mount.local_path is None
    assert mount.reason_code == "workspace_sandbox_volume_runtime_not_configured"


def test_store_updates_volume_state_and_bound_ready_mount_hint_round_trips(tmp_path):
    store = SQLiteStore(db_path=str(tmp_path / "sandbox.db"))
    service = SandboxWorkspaceVolumeService(store=store)
    volume = service.provision_workspace_volume(
        workspace_id="workspace-1",
        user_id="user-1",
        display_name="Project root",
        idempotency_key=None,
        requested_runtime="docker",
    )

    store.update_workspace_volume_state(
        volume.id,
        state=WorkspaceVolumeState.ready,
        mount_path="/workspace/project",
        diagnostics={"message": "mounted"},
    )
    service.bind_workspace_volume_root(
        sandbox_volume_id=volume.id,
        workspace_id="workspace-1",
        root_id="primary",
    )

    reloaded = service.validate_workspace_volume(
        workspace_id="workspace-1",
        user_id="user-1",
        sandbox_volume_id=volume.id,
    )
    mount = service.resolve_workspace_volume_mount(
        workspace_id="workspace-1",
        root_id="primary",
        sandbox_volume_id=volume.id,
    )

    assert reloaded.state == "ready"
    assert mount.state == "ready"
    assert mount.local_path == "/workspace/project"

    mismatch = service.resolve_workspace_volume_mount(
        workspace_id="workspace-1",
        root_id="other-root",
        sandbox_volume_id=volume.id,
    )
    assert mismatch.state == "unavailable"
    assert mismatch.reason_code == "workspace_sandbox_volume_root_mismatch"

    store.update_workspace_volume_state(volume.id, state=WorkspaceVolumeState.failed)
    failed = store.get_workspace_volume(volume.id)
    assert failed is not None
    assert failed.mount_path == "/workspace/project"


def test_store_direct_writes_bound_and_redact_workspace_volume_diagnostics(tmp_path):
    stores = [
        InMemoryStore(),
        SQLiteStore(db_path=str(tmp_path / "sandbox.db")),
    ]
    for store in stores:
        store.put_workspace_volume(
            WorkspaceVolume(
                id=f"volume-{type(store).__name__}",
                workspace_id="workspace-1",
                user_id="user-1",
                state=WorkspaceVolumeState.ready,
                runtime="docker",
                mount_path="/Users/alice/private/project",
                diagnostics={
                    "api_key": "sk-secret-value",
                    "message": "cannot mount /Users/alice/private/project " + ("x" * 500),
                    "nested": {"token": "secret-token", "path": "/private/tmp/raw-host"},
                },
            )
        )
        volume = store.get_workspace_volume(f"volume-{type(store).__name__}")

        assert volume is not None
        assert volume.mount_path is None
        diagnostic_text = repr(volume.diagnostics)
        assert "/Users/alice" not in diagnostic_text
        assert "/private/tmp" not in diagnostic_text
        assert "sk-secret-value" not in diagnostic_text
        assert "secret-token" not in diagnostic_text
        assert len(str(volume.diagnostics.get("message", ""))) <= 240

        updated = store.update_workspace_volume_state(
            volume.id,
            state=WorkspaceVolumeState.failed,
            mount_path="/var/folders/raw-host",
            diagnostics={"password": "super-secret", "message": "/var/folders/raw-host"},
        )

        assert updated is not None
        assert updated.mount_path is None
        updated_text = repr(updated.diagnostics)
        assert "/var/folders" not in updated_text
        assert "super-secret" not in updated_text
