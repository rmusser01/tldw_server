from __future__ import annotations

from dataclasses import dataclass

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    ConflictError,
    InputError,
)
from tldw_Server_API.app.core.Workspaces.root_binding_service import (
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


class _VersionConflictDB:
    def get_workspace_primary_root(self, workspace_id):
        return None

    def upsert_workspace_primary_root(self, workspace_id, payload):
        raise ConflictError("Workspace 'ws-1' version mismatch.")


class _LoadConflictDB:
    def get_workspace_primary_root(self, workspace_id):
        raise ConflictError("Workspace 'ws-1' version mismatch.")

    def upsert_workspace_primary_root(self, workspace_id, payload):
        pytest.fail("upsert should not run when current-root load fails")


class _LoadInputErrorDB:
    def get_workspace_primary_root(self, workspace_id):
        raise InputError("workspace_id is invalid.")

    def upsert_workspace_primary_root(self, workspace_id, payload):
        pytest.fail("upsert should not run when current-root load fails")


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
