import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    ConflictError,
    InputError,
)


@pytest.fixture
def db(tmp_path):
    return CharactersRAGDB(db_path=str(tmp_path / "chacha.db"), client_id="user-1")


def test_workspace_defaults_to_research_profile(db):
    ws = db.upsert_workspace("ws-1", "Workspace")
    assert ws["workspace_profile"] == "research"


def test_workspace_can_be_created_as_project_profile(db):
    ws = db.upsert_workspace("ws-1", "Workspace", workspace_profile="project")
    assert ws["workspace_profile"] == "project"


def test_invalid_workspace_profile_raises_input_error(db):
    with pytest.raises(InputError):
        db.upsert_workspace("ws-1", "Workspace", workspace_profile="invalid")


def test_upsert_primary_host_local_root_upgrades_workspace_profile(db):
    workspace = db.upsert_workspace("ws-1", "Workspace")
    root = db.upsert_workspace_primary_root(
        "ws-1",
        {
            "root_id": "root-host",
            "backend": "host_local",
            "display_name": "Local project",
            "absolute_root": "/Users/example/project",
            "root_state": "attached",
        },
    )
    assert root["workspace_id"] == "ws-1"
    assert root["backend"] == "host_local"
    assert root["root_state"] == "attached"
    assert root["is_primary"] in (True, 1)
    updated_workspace = db.get_workspace("ws-1")
    assert updated_workspace["workspace_profile"] == "project"
    assert updated_workspace["version"] == workspace["version"] + 1


def test_upsert_without_profile_does_not_downgrade_project_workspace(db):
    db.upsert_workspace("ws-1", "Workspace")
    db.upsert_workspace_primary_root("ws-1", {"root_id": "root-1", "backend": "host_local"})

    ws = db.upsert_workspace("ws-1", "Renamed Workspace")

    assert ws["name"] == "Renamed Workspace"
    assert ws["workspace_profile"] == "project"


def test_upsert_primary_sandbox_root_is_first_class(db):
    db.upsert_workspace("ws-1", "Workspace")
    root = db.upsert_workspace_primary_root(
        "ws-1",
        {
            "root_id": "root-sandbox",
            "backend": "sandbox_volume",
            "display_name": "Sandbox project",
            "sandbox_volume_id": "volume-123",
            "root_state": "not_configured",
        },
    )
    assert root["backend"] == "sandbox_volume"
    assert root["sandbox_volume_id"] == "volume-123"


def test_primary_root_upsert_replaces_existing_primary_root(db):
    db.upsert_workspace("ws-1", "Workspace")
    db.upsert_workspace_primary_root("ws-1", {"root_id": "root-1", "backend": "host_local"})
    root = db.upsert_workspace_primary_root("ws-1", {"root_id": "root-2", "backend": "sandbox_volume"})
    roots = db.list_workspace_project_roots("ws-1")
    assert [item["root_id"] for item in roots] == ["root-2"]
    assert root["root_id"] == "root-2"


def test_upsert_primary_root_enforces_expected_workspace_version(db):
    db.upsert_workspace("ws-1", "Workspace")

    with pytest.raises(ConflictError):
        db.upsert_workspace_primary_root(
            "ws-1",
            {
                "root_id": "primary",
                "backend": "host_local",
                "absolute_root": "/Users/example/project",
                "expected_workspace_version": 999,
            },
        )


def test_upsert_primary_root_rejects_malformed_expected_workspace_version(db):
    db.upsert_workspace("ws-1", "Workspace")

    with pytest.raises(InputError):
        db.upsert_workspace_primary_root(
            "ws-1",
            {
                "root_id": "primary",
                "backend": "host_local",
                "expected_workspace_version": "not-an-int",
            },
        )


def test_upsert_primary_root_replaces_same_root_id_when_replace_existing_true(db):
    db.upsert_workspace("ws-1", "Workspace")
    db.upsert_workspace_primary_root(
        "ws-1",
        {"root_id": "primary", "backend": "host_local", "absolute_root": "/old"},
    )

    root = db.upsert_workspace_primary_root(
        "ws-1",
        {
            "root_id": "primary",
            "backend": "sandbox_volume",
            "sandbox_volume_id": "volume-1",
            "replace_existing": True,
        },
    )

    assert root["backend"] == "sandbox_volume"
    assert root["sandbox_volume_id"] == "volume-1"
    assert root["absolute_root"] is None


def test_upsert_primary_root_expected_workspace_version_gates_replacement(db):
    workspace = db.upsert_workspace("ws-1", "Workspace")
    root = db.upsert_workspace_primary_root(
        "ws-1",
        {
            "root_id": "primary",
            "backend": "host_local",
            "absolute_root": "/old",
            "expected_workspace_version": workspace["version"],
        },
    )
    updated_workspace = db.get_workspace("ws-1")
    assert updated_workspace["version"] == workspace["version"] + 1

    with pytest.raises(ConflictError):
        db.upsert_workspace_primary_root(
            "ws-1",
            {
                "root_id": "replacement",
                "backend": "sandbox_volume",
                "sandbox_volume_id": "volume-1",
                "expected_workspace_version": workspace["version"],
                "replace_existing": True,
            },
        )

    assert db.get_workspace_primary_root("ws-1")["root_id"] == root["root_id"]
    assert db.get_workspace_primary_root("ws-1")["backend"] == "host_local"


def test_upsert_primary_root_expected_workspace_version_rolls_back_mid_transaction_change(db):
    db.upsert_workspace("ws-1", "Workspace")
    db.upsert_workspace_primary_root(
        "ws-1",
        {"root_id": "primary", "backend": "host_local", "absolute_root": "/old"},
    )
    workspace = db.get_workspace("ws-1")
    db.execute_query(
        """
        CREATE TRIGGER simulate_concurrent_workspace_root_attach
        BEFORE DELETE ON workspace_project_roots
        BEGIN
            UPDATE workspaces
               SET version = version + 1
             WHERE id = OLD.workspace_id;
        END
        """,
        commit=True,
    )

    with pytest.raises(ConflictError):
        db.upsert_workspace_primary_root(
            "ws-1",
            {
                "root_id": "replacement",
                "backend": "sandbox_volume",
                "sandbox_volume_id": "volume-1",
                "expected_workspace_version": workspace["version"],
                "replace_existing": True,
            },
        )

    root = db.get_workspace_primary_root("ws-1")
    assert root["root_id"] == "primary"
    assert root["backend"] == "host_local"
    assert db.get_workspace("ws-1")["version"] == workspace["version"]


def test_invalid_root_backend_raises_input_error(db):
    db.upsert_workspace("ws-1", "Workspace")
    with pytest.raises(InputError):
        db.upsert_workspace_primary_root("ws-1", {"root_id": "root-1", "backend": "git_clone"})


def test_soft_deleted_workspace_roots_are_not_listed(db):
    db.upsert_workspace("ws-1", "Workspace")
    db.upsert_workspace_primary_root("ws-1", {"root_id": "root-1", "backend": "host_local"})
    ws = db.get_workspace("ws-1")
    db.delete_workspace("ws-1", expected_version=ws["version"])
    assert db.list_workspace_project_roots("ws-1") == []


def test_update_workspace_project_root_state_uses_optimistic_locking(db):
    db.upsert_workspace("ws-1", "Workspace")
    root = db.upsert_workspace_primary_root("ws-1", {"root_id": "root-1", "backend": "host_local"})

    updated = db.update_workspace_project_root_state(
        "ws-1",
        "root-1",
        {"root_state": "attached", "git_state": "clean"},
        expected_version=root["version"],
    )

    assert updated["root_state"] == "attached"
    assert updated["git_state"] == "clean"
    assert updated["version"] == root["version"] + 1
    with pytest.raises(ConflictError):
        db.update_workspace_project_root_state(
            "ws-1",
            "root-1",
            {"root_state": "missing"},
            expected_version=root["version"],
        )


def test_update_workspace_project_root_state_rejects_binding_fields(db):
    db.upsert_workspace("ws-1", "Workspace")
    root = db.upsert_workspace_primary_root(
        "ws-1",
        {"root_id": "root-1", "backend": "host_local", "absolute_root": "/old"},
    )

    with pytest.raises(InputError):
        db.update_workspace_project_root_state(
            "ws-1",
            "root-1",
            {"absolute_root": "/new"},
            expected_version=root["version"],
        )


def test_retrying_same_primary_root_upsert_preserves_operational_state(db):
    db.upsert_workspace("ws-1", "Workspace")
    payload = {"root_id": "root-1", "backend": "host_local", "root_state": "not_configured"}
    root = db.upsert_workspace_primary_root("ws-1", payload)
    updated = db.update_workspace_project_root_state(
        "ws-1",
        "root-1",
        {"root_state": "attached", "git_state": "clean"},
        expected_version=root["version"],
    )

    retried = db.upsert_workspace_primary_root(
        "ws-1",
        {"root_id": "root-1", "backend": "host_local"},
    )

    assert retried["root_state"] == "attached"
    assert retried["git_state"] == "clean"
    assert retried["version"] == updated["version"]


def test_retrying_same_primary_root_upsert_repairs_operational_state_when_provided(db):
    db.upsert_workspace("ws-1", "Workspace")
    root = db.upsert_workspace_primary_root(
        "ws-1",
        {
            "root_id": "primary",
            "backend": "sandbox_volume",
            "sandbox_volume_id": "volume-1",
            "root_state": "attached",
            "sandbox_mount_state": "unavailable",
        },
    )

    repaired = db.upsert_workspace_primary_root(
        "ws-1",
        {
            "root_id": "primary",
            "backend": "sandbox_volume",
            "sandbox_volume_id": "volume-1",
            "root_state": "attached",
            "sandbox_mount_state": "ready",
        },
    )

    assert repaired["sandbox_mount_state"] == "ready"
    assert repaired["version"] == root["version"] + 1
