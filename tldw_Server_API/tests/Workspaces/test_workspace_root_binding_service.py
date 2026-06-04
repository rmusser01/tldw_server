from __future__ import annotations

from dataclasses import dataclass

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    ConflictError,
    InputError,
)
from tldw_Server_API.app.core.Workspaces import root_binding_service
from tldw_Server_API.app.core.Workspaces.root_binding_service import (
    SandboxInventoryMount,
    SandboxVolumeBinding,
    WorkspaceRootAttachRequest,
    WorkspaceRootConfigurationError,
    WorkspaceRootConflictError,
    WorkspaceRootInputError,
    WorkspaceRootValidationError,
    attach_primary_workspace_root,
)


@pytest.fixture
def db(tmp_path):
    return CharactersRAGDB(db_path=str(tmp_path / "chacha.db"), client_id="user-1")


def test_host_local_attach_validates_allowlist_and_persists_absolute_path(db, tmp_path):
    allowed = tmp_path / "allowed"
    project = allowed / "project"
    project.mkdir(parents=True)
    db.upsert_workspace("ws-1", "Workspace")

    root = attach_primary_workspace_root(
        db=db,
        workspace_id="ws-1",
        user_id="user-1",
        request=WorkspaceRootAttachRequest(
            backend="host_local",
            absolute_root=str(project),
        ),
        allowed_roots=(allowed,),
    )

    assert root["root_id"] == "primary"
    assert root["backend"] == "host_local"
    assert root["absolute_root"] == str(project.resolve())
    assert root["root_state"] == "attached"
    assert root["display_name"] == "project"


def test_host_local_attach_uses_configured_allowed_roots_when_not_injected(db, tmp_path, monkeypatch):
    allowed = tmp_path / "configured"
    project = allowed / "project"
    project.mkdir(parents=True)
    db.upsert_workspace("ws-1", "Workspace")
    monkeypatch.setattr(
        root_binding_service.config,
        "get_workspace_project_root_allowed_roots",
        lambda: (allowed,),
    )

    root = attach_primary_workspace_root(
        db=db,
        workspace_id="ws-1",
        user_id="user-1",
        request=WorkspaceRootAttachRequest(
            backend="host_local",
            absolute_root=str(project),
        ),
    )

    assert root["absolute_root"] == str(project.resolve())


def test_host_local_attach_without_expected_version_succeeds_and_increments_workspace_version(db, tmp_path):
    allowed = tmp_path / "allowed"
    project = allowed / "project"
    project.mkdir(parents=True)
    workspace = db.upsert_workspace("ws-1", "Workspace")

    root = attach_primary_workspace_root(
        db=db,
        workspace_id="ws-1",
        user_id="user-1",
        request=WorkspaceRootAttachRequest(
            backend="host_local",
            absolute_root=str(project),
        ),
        allowed_roots=(allowed,),
    )

    assert root["root_id"] == "primary"
    assert db.get_workspace("ws-1")["version"] == workspace["version"] + 1


def test_omitted_root_id_is_idempotent_and_uses_current_root_id(db, tmp_path):
    allowed = tmp_path / "allowed"
    project = allowed / "project"
    project.mkdir(parents=True)
    db.upsert_workspace("ws-1", "Workspace")

    first = attach_primary_workspace_root(
        db=db,
        workspace_id="ws-1",
        user_id="user-1",
        request=WorkspaceRootAttachRequest(
            backend="host_local",
            root_id="custom-root",
            absolute_root=str(project),
        ),
        allowed_roots=(allowed,),
    )
    retry = attach_primary_workspace_root(
        db=db,
        workspace_id="ws-1",
        user_id="user-1",
        request=WorkspaceRootAttachRequest(
            backend="host_local",
            absolute_root=str(project),
        ),
        allowed_roots=(allowed,),
    )

    assert first["root_id"] == "custom-root"
    assert retry["root_id"] == "custom-root"
    assert retry["absolute_root"] == str(project.resolve())


def test_invalid_root_id_raises_input_error(db, tmp_path):
    allowed = tmp_path / "allowed"
    project = allowed / "project"
    project.mkdir(parents=True)
    db.upsert_workspace("ws-1", "Workspace")

    with pytest.raises(WorkspaceRootInputError):
        attach_primary_workspace_root(
            db=db,
            workspace_id="ws-1",
            user_id="user-1",
            request=WorkspaceRootAttachRequest(
                backend="host_local",
                root_id="not allowed",
                absolute_root=str(project),
            ),
            allowed_roots=(allowed,),
        )


def test_display_name_longer_than_120_raises_input_error(db, tmp_path):
    allowed = tmp_path / "allowed"
    project = allowed / "project"
    project.mkdir(parents=True)
    db.upsert_workspace("ws-1", "Workspace")

    with pytest.raises(WorkspaceRootInputError):
        attach_primary_workspace_root(
            db=db,
            workspace_id="ws-1",
            user_id="user-1",
            request=WorkspaceRootAttachRequest(
                backend="host_local",
                absolute_root=str(project),
                display_name="x" * 121,
            ),
            allowed_roots=(allowed,),
        )


def test_no_host_local_allowed_roots_raises_configuration_error(db, tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    db.upsert_workspace("ws-1", "Workspace")

    with pytest.raises(WorkspaceRootConfigurationError) as exc_info:
        attach_primary_workspace_root(
            db=db,
            workspace_id="ws-1",
            user_id="user-1",
            request=WorkspaceRootAttachRequest(
                backend="host_local",
                absolute_root=str(project),
            ),
            allowed_roots=(),
        )

    assert exc_info.value.code == "workspace_project_roots_not_configured"


def test_host_local_outside_allowlist_raises_validation_error(db, tmp_path):
    allowed = tmp_path / "allowed"
    outside = tmp_path / "outside"
    allowed.mkdir()
    outside.mkdir()
    db.upsert_workspace("ws-1", "Workspace")

    with pytest.raises(WorkspaceRootValidationError) as exc_info:
        attach_primary_workspace_root(
            db=db,
            workspace_id="ws-1",
            user_id="user-1",
            request=WorkspaceRootAttachRequest(
                backend="host_local",
                absolute_root=str(outside),
            ),
            allowed_roots=(allowed,),
        )

    assert exc_info.value.code == "workspace_project_root_outside_allowed_roots"


def test_host_local_outside_missing_path_is_rejected_as_outside_allowlist(db, tmp_path):
    allowed = tmp_path / "allowed"
    outside_missing = tmp_path / "outside" / "missing"
    allowed.mkdir()
    db.upsert_workspace("ws-1", "Workspace")

    with pytest.raises(WorkspaceRootValidationError) as exc_info:
        attach_primary_workspace_root(
            db=db,
            workspace_id="ws-1",
            user_id="user-1",
            request=WorkspaceRootAttachRequest(
                backend="host_local",
                absolute_root=str(outside_missing),
            ),
            allowed_roots=(allowed,),
        )

    assert exc_info.value.code == "workspace_project_root_outside_allowed_roots"


def test_host_local_symlink_root_raises_input_error(db, tmp_path):
    allowed = tmp_path / "allowed"
    real_project = allowed / "project"
    symlink_project = allowed / "linked-project"
    real_project.mkdir(parents=True)
    symlink_project.symlink_to(real_project, target_is_directory=True)
    db.upsert_workspace("ws-1", "Workspace")

    with pytest.raises(WorkspaceRootInputError) as exc_info:
        attach_primary_workspace_root(
            db=db,
            workspace_id="ws-1",
            user_id="user-1",
            request=WorkspaceRootAttachRequest(
                backend="host_local",
                absolute_root=str(symlink_project),
            ),
            allowed_roots=(allowed,),
        )

    assert exc_info.value.code == "workspace_project_root_symlink"


def test_host_local_allowed_symlink_to_outside_target_raises_input_error(db, tmp_path):
    allowed = tmp_path / "allowed"
    outside = tmp_path / "outside"
    outside_project = outside / "project"
    symlink_project = allowed / "linked-project"
    outside_project.mkdir(parents=True)
    symlink_project.parent.mkdir(parents=True)
    symlink_project.symlink_to(outside_project, target_is_directory=True)
    db.upsert_workspace("ws-1", "Workspace")

    with pytest.raises(WorkspaceRootInputError) as exc_info:
        attach_primary_workspace_root(
            db=db,
            workspace_id="ws-1",
            user_id="user-1",
            request=WorkspaceRootAttachRequest(
                backend="host_local",
                absolute_root=str(symlink_project),
            ),
            allowed_roots=(allowed,),
        )

    assert exc_info.value.code == "workspace_project_root_symlink"


def test_host_local_outside_symlink_path_is_rejected_as_outside_allowlist(db, tmp_path):
    allowed = tmp_path / "allowed"
    outside = tmp_path / "outside"
    target = outside / "target"
    symlink = outside / "linked"
    allowed.mkdir()
    target.mkdir(parents=True)
    symlink.symlink_to(target, target_is_directory=True)
    db.upsert_workspace("ws-1", "Workspace")

    with pytest.raises(WorkspaceRootValidationError) as exc_info:
        attach_primary_workspace_root(
            db=db,
            workspace_id="ws-1",
            user_id="user-1",
            request=WorkspaceRootAttachRequest(
                backend="host_local",
                absolute_root=str(symlink),
            ),
            allowed_roots=(allowed,),
        )

    assert exc_info.value.code == "workspace_project_root_outside_allowed_roots"


def test_different_primary_root_without_replace_existing_raises_conflict(db, tmp_path):
    allowed = tmp_path / "allowed"
    project_one = allowed / "one"
    project_two = allowed / "two"
    project_one.mkdir(parents=True)
    project_two.mkdir()
    db.upsert_workspace("ws-1", "Workspace")
    attach_primary_workspace_root(
        db=db,
        workspace_id="ws-1",
        user_id="user-1",
        request=WorkspaceRootAttachRequest(
            backend="host_local",
            absolute_root=str(project_one),
        ),
        allowed_roots=(allowed,),
    )

    with pytest.raises(WorkspaceRootConflictError) as exc_info:
        attach_primary_workspace_root(
            db=db,
            workspace_id="ws-1",
            user_id="user-1",
            request=WorkspaceRootAttachRequest(
                backend="host_local",
                absolute_root=str(project_two),
            ),
            allowed_roots=(allowed,),
        )

    assert exc_info.value.code == "workspace_primary_root_exists"


def test_different_primary_root_with_replace_existing_replaces(db, tmp_path):
    allowed = tmp_path / "allowed"
    project_one = allowed / "one"
    project_two = allowed / "two"
    project_one.mkdir(parents=True)
    project_two.mkdir()
    db.upsert_workspace("ws-1", "Workspace")
    attach_primary_workspace_root(
        db=db,
        workspace_id="ws-1",
        user_id="user-1",
        request=WorkspaceRootAttachRequest(
            backend="host_local",
            absolute_root=str(project_one),
        ),
        allowed_roots=(allowed,),
    )

    root = attach_primary_workspace_root(
        db=db,
        workspace_id="ws-1",
        user_id="user-1",
        request=WorkspaceRootAttachRequest(
            backend="host_local",
            root_id="replacement",
            absolute_root=str(project_two),
            replace_existing=True,
        ),
        allowed_roots=(allowed,),
    )

    assert root["root_id"] == "replacement"
    assert root["absolute_root"] == str(project_two.resolve())
    assert db.get_workspace_primary_root("ws-1")["root_id"] == "replacement"


def test_default_sandbox_resolver_persists_fail_closed_mount_state(db):
    db.upsert_workspace("ws-1", "Workspace")

    root = attach_primary_workspace_root(
        db=db,
        workspace_id="ws-1",
        user_id="user-1",
        request=WorkspaceRootAttachRequest(
            backend="sandbox_volume",
            sandbox_volume_id="volume-1",
        ),
    )

    assert root["backend"] == "sandbox_volume"
    assert root["root_state"] == "attached"
    assert root["sandbox_mount_state"] == "not_configured"
    assert root["absolute_root"] is None
    assert root["sandbox_volume_id"] == "volume-1"


@dataclass(frozen=True)
class _StaticSandboxResolver:
    state: str

    def validate_workspace_volume(self, *, workspace_id, user_id, sandbox_volume_id):
        return SandboxVolumeBinding(
            sandbox_volume_id=sandbox_volume_id,
            state=self.state,
            display_name="Resolved volume",
        )


@dataclass(frozen=True)
class _InvalidStateSandboxResolver:
    def validate_workspace_volume(self, *, workspace_id, user_id, sandbox_volume_id):
        return SandboxVolumeBinding(
            sandbox_volume_id=sandbox_volume_id,
            state="mounted",  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class _MismatchedSandboxResolver:
    def validate_workspace_volume(self, *, workspace_id, user_id, sandbox_volume_id):
        return SandboxVolumeBinding(sandbox_volume_id="different", state="ready")


@dataclass(frozen=True)
class _FailingSandboxResolver:
    def validate_workspace_volume(self, *, workspace_id, user_id, sandbox_volume_id):
        raise RuntimeError("backend leaked /Users/alice/private/project")


@dataclass(frozen=True)
class _FailingSandboxMountResolver:
    def resolve_workspace_volume_mount(self, *, workspace_id, root_id, sandbox_volume_id):
        raise RuntimeError("mount leaked /Users/alice/private/project")


def test_sandbox_resolver_invalid_state_is_rejected(db):
    db.upsert_workspace("ws-1", "Workspace")

    with pytest.raises(WorkspaceRootConfigurationError) as exc_info:
        attach_primary_workspace_root(
            db=db,
            workspace_id="ws-1",
            user_id="user-1",
            request=WorkspaceRootAttachRequest(
                backend="sandbox_volume",
                sandbox_volume_id="volume-1",
            ),
            sandbox_resolver=_InvalidStateSandboxResolver(),
        )

    assert exc_info.value.code == "workspace_sandbox_volume_state_invalid"
    assert db.get_workspace_primary_root("ws-1") is None


def test_sandbox_resolver_volume_id_mismatch_is_rejected(db):
    db.upsert_workspace("ws-1", "Workspace")

    with pytest.raises(WorkspaceRootConfigurationError) as exc_info:
        attach_primary_workspace_root(
            db=db,
            workspace_id="ws-1",
            user_id="user-1",
            request=WorkspaceRootAttachRequest(
                backend="sandbox_volume",
                sandbox_volume_id="volume-1",
            ),
            sandbox_resolver=_MismatchedSandboxResolver(),
        )

    assert exc_info.value.code == "workspace_sandbox_volume_id_mismatch"
    assert db.get_workspace_primary_root("ws-1") is None


def test_sandbox_resolver_exception_is_wrapped_as_configuration_error(db):
    db.upsert_workspace("ws-1", "Workspace")

    with pytest.raises(WorkspaceRootConfigurationError) as exc_info:
        attach_primary_workspace_root(
            db=db,
            workspace_id="ws-1",
            user_id="user-1",
            request=WorkspaceRootAttachRequest(
                backend="sandbox_volume",
                sandbox_volume_id="volume-1",
            ),
            sandbox_resolver=_FailingSandboxResolver(),
        )

    assert exc_info.value.code == "workspace_sandbox_volume_resolver_failed"
    assert "/Users/alice" not in str(exc_info.value)
    assert db.get_workspace_primary_root("ws-1") is None


def test_ready_sandbox_resolver_repairs_prior_unavailable_retry(db):
    db.upsert_workspace("ws-1", "Workspace")
    first = attach_primary_workspace_root(
        db=db,
        workspace_id="ws-1",
        user_id="user-1",
        request=WorkspaceRootAttachRequest(
            backend="sandbox_volume",
            sandbox_volume_id="volume-1",
        ),
        sandbox_resolver=_StaticSandboxResolver("unavailable"),
    )

    repaired = attach_primary_workspace_root(
        db=db,
        workspace_id="ws-1",
        user_id="user-1",
        request=WorkspaceRootAttachRequest(
            backend="sandbox_volume",
            sandbox_volume_id="volume-1",
        ),
        sandbox_resolver=_StaticSandboxResolver("ready"),
    )

    assert first["sandbox_mount_state"] == "unavailable"
    assert repaired["root_id"] == first["root_id"]
    assert repaired["sandbox_mount_state"] == "ready"
    assert repaired["display_name"] == "Resolved volume"


@pytest.mark.parametrize("state", ["provisioning", "cleanup_pending"])
def test_sandbox_resolver_accepts_new_non_strict_volume_states(db, state):
    db.upsert_workspace("ws-1", "Workspace")

    root = attach_primary_workspace_root(
        db=db,
        workspace_id="ws-1",
        user_id="user-1",
        request=WorkspaceRootAttachRequest(
            backend="sandbox_volume",
            sandbox_volume_id="volume-1",
        ),
        sandbox_resolver=_StaticSandboxResolver(state),
    )

    assert root["backend"] == "sandbox_volume"
    assert root["sandbox_mount_state"] == state


@pytest.mark.parametrize(
    "state",
    ["not_configured", "unavailable", "failed", "cleanup_pending"],
)
def test_strict_sandbox_validation_fails_closed_for_unusable_volume_states(db, state):
    db.upsert_workspace("ws-1", "Workspace")

    with pytest.raises(WorkspaceRootConfigurationError) as exc_info:
        attach_primary_workspace_root(
            db=db,
            workspace_id="ws-1",
            user_id="user-1",
            request=WorkspaceRootAttachRequest(
                backend="sandbox_volume",
                sandbox_volume_id="volume-1",
                strict_sandbox_validation=True,
            ),
            sandbox_resolver=_StaticSandboxResolver(state),
        )

    assert exc_info.value.code == "workspace_sandbox_volume_resolver_unavailable"
    assert db.get_workspace_primary_root("ws-1") is None


class _VersionConflictDB:
    def get_workspace(self, workspace_id):
        return {"version": 1}

    def get_workspace_primary_root(self, workspace_id):
        return None

    def upsert_workspace_primary_root(self, workspace_id, payload):
        raise ConflictError("Workspace 'ws-1' version mismatch.")


class _LoadConflictDB:
    def get_workspace(self, workspace_id):
        return {"version": 1}

    def get_workspace_primary_root(self, workspace_id):
        raise ConflictError("Workspace 'ws-1' version mismatch.")

    def upsert_workspace_primary_root(self, workspace_id, payload):
        pytest.fail("upsert should not run when current-root load fails")


class _LoadInputErrorDB:
    def get_workspace(self, workspace_id):
        return {"version": 1}

    def get_workspace_primary_root(self, workspace_id):
        raise InputError("workspace_id is invalid.")

    def upsert_workspace_primary_root(self, workspace_id, payload):
        pytest.fail("upsert should not run when current-root load fails")


class _WorkspaceLoadConflictDB:
    def get_workspace(self, workspace_id):
        raise ConflictError("Workspace 'ws-1' version mismatch.")

    def get_workspace_primary_root(self, workspace_id):
        pytest.fail("current root should not load when workspace load fails")

    def upsert_workspace_primary_root(self, workspace_id, payload):
        pytest.fail("upsert should not run when workspace load fails")


class _MissingWorkspaceDB:
    def get_workspace(self, workspace_id):
        return None

    def get_workspace_primary_root(self, workspace_id):
        pytest.fail("current root should not load when workspace is missing")

    def upsert_workspace_primary_root(self, workspace_id, payload):
        pytest.fail("upsert should not run when workspace is missing")


class _RacingAttachDB:
    def __init__(self, real_db, replacement_root):
        self.real_db = real_db
        self.replacement_root = replacement_root
        self.loaded_workspace_version = None

    def get_workspace(self, workspace_id):
        workspace = self.real_db.get_workspace(workspace_id)
        self.loaded_workspace_version = workspace["version"]
        return workspace

    def get_workspace_primary_root(self, workspace_id):
        assert self.loaded_workspace_version is not None
        self.real_db.upsert_workspace_primary_root(workspace_id, self.replacement_root)
        return None

    def upsert_workspace_primary_root(self, workspace_id, payload):
        return self.real_db.upsert_workspace_primary_root(workspace_id, payload)


def test_db_conflict_with_version_text_is_wrapped_as_version_mismatch():
    with pytest.raises(WorkspaceRootConflictError) as exc_info:
        attach_primary_workspace_root(
            db=_VersionConflictDB(),
            workspace_id="ws-1",
            user_id="user-1",
            request=WorkspaceRootAttachRequest(
                backend="sandbox_volume",
                sandbox_volume_id="volume-1",
            ),
        )

    assert exc_info.value.code == "workspace_version_mismatch"


def test_db_conflict_from_workspace_load_is_wrapped_as_version_mismatch():
    with pytest.raises(WorkspaceRootConflictError) as exc_info:
        attach_primary_workspace_root(
            db=_WorkspaceLoadConflictDB(),
            workspace_id="ws-1",
            user_id="user-1",
            request=WorkspaceRootAttachRequest(
                backend="sandbox_volume",
                sandbox_volume_id="volume-1",
            ),
        )

    assert exc_info.value.code == "workspace_version_mismatch"


def test_missing_workspace_load_is_wrapped_as_workspace_not_found():
    with pytest.raises(root_binding_service.WorkspaceRootNotFoundError) as exc_info:
        attach_primary_workspace_root(
            db=_MissingWorkspaceDB(),
            workspace_id="ws-missing",
            user_id="user-1",
            request=WorkspaceRootAttachRequest(
                backend="sandbox_volume",
                sandbox_volume_id="volume-1",
            ),
        )

    assert exc_info.value.code == "workspace_not_found"
    assert exc_info.value.status_code == 404


def test_db_conflict_from_current_root_load_is_wrapped_as_version_mismatch():
    with pytest.raises(WorkspaceRootConflictError) as exc_info:
        attach_primary_workspace_root(
            db=_LoadConflictDB(),
            workspace_id="ws-1",
            user_id="user-1",
            request=WorkspaceRootAttachRequest(
                backend="sandbox_volume",
                sandbox_volume_id="volume-1",
            ),
        )

    assert exc_info.value.code == "workspace_version_mismatch"


def test_db_input_error_from_current_root_load_is_wrapped_as_invalid_request():
    with pytest.raises(WorkspaceRootInputError) as exc_info:
        attach_primary_workspace_root(
            db=_LoadInputErrorDB(),
            workspace_id="ws-1",
            user_id="user-1",
            request=WorkspaceRootAttachRequest(
                backend="sandbox_volume",
                sandbox_volume_id="volume-1",
            ),
        )

    assert exc_info.value.code == "workspace_root_invalid_request"


def test_service_auto_expected_version_prevents_read_before_write_replacement_race(db, tmp_path):
    allowed = tmp_path / "allowed"
    requested = allowed / "requested"
    requested.mkdir(parents=True)
    db.upsert_workspace("ws-1", "Workspace")
    racing_db = _RacingAttachDB(
        db,
        {"root_id": "other", "backend": "sandbox_volume", "sandbox_volume_id": "volume-other"},
    )

    with pytest.raises(WorkspaceRootConflictError) as exc_info:
        attach_primary_workspace_root(
            db=racing_db,
            workspace_id="ws-1",
            user_id="user-1",
            request=WorkspaceRootAttachRequest(
                backend="host_local",
                absolute_root=str(requested),
            ),
            allowed_roots=(allowed,),
        )

    assert exc_info.value.code == "workspace_version_mismatch"
    assert db.get_workspace_primary_root("ws-1")["root_id"] == "other"


def test_inventory_sandbox_mount_resolver_exception_returns_failure_result():
    result = root_binding_service.resolve_workspace_root_for_inventory_scan(
        root={
            "workspace_id": "ws-1",
            "root_id": "primary",
            "backend": "sandbox_volume",
            "sandbox_volume_id": "volume-1",
        },
        sandbox_mount_resolver=_FailingSandboxMountResolver(),
    )

    assert result.ok is False
    assert result.backend == "sandbox_volume"
    assert result.failure_code == "sandbox_mount_resolution_failed"
    assert result.message == "Workspace sandbox volume mount could not be resolved."


def test_inventory_sandbox_mount_ready_resolver_still_returns_local_path(tmp_path):
    class _ReadySandboxMountResolver:
        def resolve_workspace_volume_mount(self, *, workspace_id, root_id, sandbox_volume_id):
            return SandboxInventoryMount(
                sandbox_volume_id=sandbox_volume_id,
                state="ready",
                local_path=str(tmp_path),
            )

    result = root_binding_service.resolve_workspace_root_for_inventory_scan(
        root={
            "workspace_id": "ws-1",
            "root_id": "primary",
            "backend": "sandbox_volume",
            "sandbox_volume_id": "volume-1",
            "version": 7,
        },
        sandbox_mount_resolver=_ReadySandboxMountResolver(),
    )

    assert result.ok is True
    assert result.local_path == tmp_path.resolve()
    assert result.root_snapshot_token is not None
    assert result.root_snapshot_token.startswith("primary:7:")
