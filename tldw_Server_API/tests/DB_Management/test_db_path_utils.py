import pytest
from pathlib import Path

from tldw_Server_API.app.core.DB_Management import db_path_utils
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.config import settings
from tldw_Server_API.app.core.exceptions import InvalidStoragePathError


def _expect_equal(actual, expected, message: str) -> None:
    if actual != expected:
        pytest.fail(f"{message}: expected {expected!r}, got {actual!r}")


def _expect_true(condition: bool, message: str) -> None:
    if not condition:
        pytest.fail(message)


def test_get_user_base_directory_expands_user(monkeypatch, tmp_path):

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setitem(settings, "USER_DB_BASE_DIR", "~/custom_db_root")

    user_dir = DatabasePaths.get_user_base_directory(42)

    expected = tmp_path / "custom_db_root" / "42"
    _expect_equal(user_dir, expected, "expanded user dir should match HOME-relative override")
    _expect_true(expected.exists(), "expanded user dir should be created")


def test_get_user_base_directory_resolves_relative(monkeypatch, tmp_path):

    monkeypatch.setitem(settings, "USER_DB_BASE_DIR", "relative_root")
    monkeypatch.setattr(db_path_utils, "get_project_root", lambda: str(tmp_path))

    user_dir = DatabasePaths.get_user_base_directory(7)

    expected = tmp_path / "relative_root" / "7"
    _expect_equal(user_dir, expected, "relative user dir should resolve from project root")
    _expect_true(expected.exists(), "relative user dir should be created")


def test_get_user_base_directory_single_user_none_resolves_fixed_id(monkeypatch, tmp_path):
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setitem(settings, "SINGLE_USER_FIXED_ID", "42")
    monkeypatch.setitem(settings, "USER_DB_BASE_DIR", str(tmp_path / "user_dbs"))

    user_dir = DatabasePaths.get_user_base_directory(None)

    expected = tmp_path / "user_dbs" / "42"
    _expect_equal(user_dir, expected, "single-user None user_id should map to fixed id")
    _expect_true(expected.exists(), "single-user base dir should be created")


def test_get_user_base_directory_multi_user_none_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setitem(settings, "USER_DB_BASE_DIR", str(tmp_path / "user_dbs"))

    with pytest.raises(ValueError, match="user_id is required in multi-user mode"):
        DatabasePaths.get_user_base_directory(None)


def test_user_db_base_dir_settings_override_env_outside_tests(monkeypatch, tmp_path):
    env_base = tmp_path / "env_base"
    settings_base = tmp_path / "settings_base"
    monkeypatch.setenv("USER_DB_BASE_DIR", str(env_base))
    monkeypatch.setitem(settings, "USER_DB_BASE_DIR", str(settings_base))
    monkeypatch.setattr(db_path_utils, "_is_test_context", lambda: False)

    base_dir = DatabasePaths.get_user_db_base_dir()

    _expect_equal(base_dir, settings_base, "settings USER_DB_BASE_DIR should override env outside tests")
    _expect_true(base_dir.exists(), "resolved user DB base dir should exist")


def test_invalid_user_id_rejected_outside_tests(monkeypatch, tmp_path):
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setitem(settings, "USER_DB_BASE_DIR", str(tmp_path / "user_dbs"))
    monkeypatch.setattr(db_path_utils, "_is_test_context", lambda: False)

    with pytest.raises(ValueError, match="Invalid user_id"):
        DatabasePaths.get_user_base_directory("..")


def test_non_numeric_user_id_rejected_outside_tests(monkeypatch, tmp_path):
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setitem(settings, "USER_DB_BASE_DIR", str(tmp_path / "user_dbs"))
    monkeypatch.setattr(db_path_utils, "_is_test_context", lambda: False)

    with pytest.raises(ValueError, match="Invalid user_id"):
        DatabasePaths.get_user_base_directory("user_abc")


def test_non_numeric_user_id_allowed_in_tests(monkeypatch, tmp_path):
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setitem(settings, "USER_DB_BASE_DIR", str(tmp_path / "user_dbs"))
    monkeypatch.setattr(db_path_utils, "_is_test_context", lambda: True)

    user_dir = DatabasePaths.get_user_base_directory("user_abc")
    _expect_equal(user_dir.name, "user_abc", "test mode should preserve safe non-numeric user ids")


def test_user_db_base_dir_relative_escape_rejected(monkeypatch, tmp_path):
    project_root = tmp_path / "project"
    project_root.mkdir()
    monkeypatch.setattr(db_path_utils, "get_project_root", lambda: str(project_root))
    monkeypatch.setitem(settings, "USER_DB_BASE_DIR", "../outside")
    monkeypatch.delenv("USER_DB_BASE_DIR", raising=False)

    with pytest.raises(InvalidStoragePathError):
        DatabasePaths.get_user_base_directory(1)


def test_prompts_db_path_salt_rejects_path_separators(monkeypatch, tmp_path):
    monkeypatch.setitem(settings, "USER_DB_BASE_DIR", str(tmp_path / "user_dbs"))

    with pytest.raises(InvalidStoragePathError):
        DatabasePaths.get_prompts_db_path(1, salt="../bad")


def test_prompts_db_path_salt_accepts_safe_value(monkeypatch, tmp_path):
    monkeypatch.setitem(settings, "USER_DB_BASE_DIR", str(tmp_path / "user_dbs"))

    path = DatabasePaths.get_prompts_db_path(1, salt="safe_123")

    _expect_equal(path.name, "user_prompts_v2_safe_123.sqlite", "safe prompts salt should affect filename")


def test_user_db_base_dir_test_fallback_is_not_repo_local(monkeypatch):
    """Test mode fallback should be isolated and must not write into repo-local Databases/user_databases."""
    monkeypatch.delenv("USER_DB_BASE_DIR", raising=False)
    monkeypatch.delenv("USER_DB_BASE", raising=False)
    monkeypatch.setitem(settings, "USER_DB_BASE_DIR", None)
    monkeypatch.setitem(settings, "USER_DB_BASE", None)
    monkeypatch.setattr(db_path_utils, "_is_test_context", lambda: True)

    repo_local_default = (Path.cwd() / "Databases" / "user_databases").resolve()
    resolved = DatabasePaths.get_user_db_base_dir()

    _expect_true(
        resolved != repo_local_default,
        "test fallback must not use repo-local Databases/user_databases",
    )
    _expect_true(
        repo_local_default not in resolved.parents,
        "test fallback must remain outside repo-local user DB tree",
    )


def test_user_db_base_dir_test_fallback_uses_unique_run_tag(monkeypatch):
    monkeypatch.delenv("USER_DB_BASE_DIR", raising=False)
    monkeypatch.delenv("USER_DB_BASE", raising=False)
    monkeypatch.delenv("TLDW_TEST_RUN_ID", raising=False)
    monkeypatch.delenv("PYTEST_XDIST_WORKER", raising=False)
    monkeypatch.setitem(settings, "USER_DB_BASE_DIR", None)
    monkeypatch.setitem(settings, "USER_DB_BASE", None)
    monkeypatch.setattr(db_path_utils, "_is_test_context", lambda: True)

    resolved = DatabasePaths.get_user_db_base_dir()

    _expect_true(resolved.name != "default", "test fallback path should include a unique run tag")


def test_resolve_trusted_database_path_anchors_relative_paths_to_project_root(monkeypatch, tmp_path):
    project_root = tmp_path / "project"
    outside_cwd = tmp_path / "outside"
    project_root.mkdir()
    outside_cwd.mkdir()

    monkeypatch.setattr(db_path_utils, "get_project_root", lambda: str(project_root))
    monkeypatch.chdir(outside_cwd)

    resolved = db_path_utils.resolve_trusted_database_path("Databases/app.db")

    _expect_equal(
        resolved,
        project_root / "Databases" / "app.db",
        "relative trusted DB paths should anchor to the project root",
    )


def test_resolve_trusted_database_path_accepts_absolute_path_inside_project_root(monkeypatch, tmp_path):
    project_root = tmp_path / "project"
    db_dir = project_root / "Databases"
    db_dir.mkdir(parents=True)

    monkeypatch.setattr(db_path_utils, "get_project_root", lambda: str(project_root))
    monkeypatch.setattr(db_path_utils, "_is_test_context", lambda: False)

    resolved = db_path_utils.resolve_trusted_database_path(db_dir / "absolute.db")

    _expect_equal(
        resolved,
        db_dir / "absolute.db",
        "absolute trusted DB paths should be accepted when they stay inside trusted roots",
    )


def test_resolve_trusted_database_path_rejects_symlink_escape(monkeypatch, tmp_path):
    project_root = tmp_path / "project"
    trusted_dir = project_root / "Databases"
    outside_dir = tmp_path / "outside"
    project_root.mkdir()
    trusted_dir.mkdir()
    outside_dir.mkdir()
    (trusted_dir / "escape").symlink_to(outside_dir, target_is_directory=True)

    monkeypatch.setattr(db_path_utils, "get_project_root", lambda: str(project_root))
    monkeypatch.setattr(db_path_utils, "_is_test_context", lambda: False)

    with pytest.raises(InvalidStoragePathError):
        db_path_utils.resolve_trusted_database_path(trusted_dir / "escape" / "users.db")


def test_resolve_trusted_database_path_rejects_symlinked_database_file(monkeypatch, tmp_path):
    project_root = tmp_path / "project"
    trusted_dir = project_root / "Databases"
    outside_dir = tmp_path / "outside"
    project_root.mkdir()
    trusted_dir.mkdir()
    outside_dir.mkdir()

    outside_db = outside_dir / "users.db"
    outside_db.write_text("outside", encoding="utf-8")
    (trusted_dir / "users.db").symlink_to(outside_db)

    monkeypatch.setattr(db_path_utils, "get_project_root", lambda: str(project_root))
    monkeypatch.setattr(db_path_utils, "_is_test_context", lambda: False)

    with pytest.raises(InvalidStoragePathError):
        db_path_utils.resolve_trusted_database_path(trusted_dir / "users.db")


def test_resolve_trusted_database_path_accepts_symlink_alias_to_temp_root(monkeypatch, tmp_path):
    project_root = tmp_path / "project"
    real_temp_root = tmp_path / "private_tmp"
    temp_alias = tmp_path / "var_alias"
    project_root.mkdir()
    real_temp_root.mkdir()
    temp_alias.symlink_to(real_temp_root, target_is_directory=True)

    monkeypatch.setattr(db_path_utils, "get_project_root", lambda: str(project_root))
    monkeypatch.setattr(db_path_utils, "_is_test_context", lambda: True)
    monkeypatch.setattr(db_path_utils.tempfile, "gettempdir", lambda: str(temp_alias))

    resolved = db_path_utils.resolve_trusted_database_path(temp_alias / "dbs" / "app.db")

    _expect_equal(
        resolved,
        real_temp_root / "dbs" / "app.db",
        "trusted temp paths should be compared using canonicalized paths",
    )


def test_resolve_trusted_database_path_rejects_unresolved_home_prefix(monkeypatch, tmp_path):
    project_root = tmp_path / "project"
    project_root.mkdir()

    monkeypatch.setattr(db_path_utils, "get_project_root", lambda: str(project_root))
    monkeypatch.setattr(db_path_utils, "_is_test_context", lambda: False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))

    with pytest.raises(InvalidStoragePathError):
        db_path_utils.resolve_trusted_database_path("~missing-user/app.db")


def test_ensure_trusted_database_parent_dir_creates_missing_parent(monkeypatch, tmp_path):
    project_root = tmp_path / "project"
    project_root.mkdir()
    target_path = project_root / "Databases" / "nested" / "app.db"

    monkeypatch.setattr(db_path_utils, "get_project_root", lambda: str(project_root))
    monkeypatch.setattr(db_path_utils, "_is_test_context", lambda: False)

    resolved = db_path_utils.ensure_trusted_database_parent_dir(
        target_path,
        label="test database",
    )

    _expect_equal(resolved, target_path, "trusted parent dir helper should preserve the resolved DB path")
    _expect_true(target_path.parent.exists(), "trusted parent dir helper should create the parent directory")


def test_require_trusted_database_parent_exists_rejects_missing_parent(monkeypatch, tmp_path):
    project_root = tmp_path / "project"
    project_root.mkdir()
    target_path = project_root / "Databases" / "missing" / "app.db"

    monkeypatch.setattr(db_path_utils, "get_project_root", lambda: str(project_root))
    monkeypatch.setattr(db_path_utils, "_is_test_context", lambda: False)

    with pytest.raises(ValueError, match="parent directory must already exist"):
        db_path_utils.require_trusted_database_parent_exists(
            target_path,
            label="test database",
        )


def test_require_trusted_database_parent_exists_accepts_symlink_alias_to_temp_root(monkeypatch, tmp_path):
    project_root = tmp_path / "project"
    real_temp_root = tmp_path / "private_tmp"
    temp_alias = tmp_path / "var_alias"
    project_root.mkdir()
    real_temp_root.mkdir()
    temp_alias.symlink_to(real_temp_root, target_is_directory=True)
    (real_temp_root / "dbs").mkdir()

    monkeypatch.setattr(db_path_utils, "get_project_root", lambda: str(project_root))
    monkeypatch.setattr(db_path_utils, "_is_test_context", lambda: True)
    monkeypatch.setattr(db_path_utils.tempfile, "gettempdir", lambda: str(temp_alias))

    resolved = db_path_utils.require_trusted_database_parent_exists(
        temp_alias / "dbs" / "app.db",
        label="test database",
    )

    _expect_equal(
        resolved,
        real_temp_root / "dbs" / "app.db",
        "trusted temp parent check should accept canonicalized alias roots",
    )
