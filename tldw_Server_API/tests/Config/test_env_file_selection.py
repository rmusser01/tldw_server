import importlib
from pathlib import Path

import dotenv
import pytest

from tldw_Server_API.app.core import config

pytestmark = pytest.mark.unit


def test_explicit_tldw_env_file_is_first_dotenv_candidate(
    tmp_path: Path,
    monkeypatch,
) -> None:
    explicit_env = tmp_path / "runtime.env"
    project_root = tmp_path / "tldw_Server_API"
    repo_root = tmp_path

    monkeypatch.setenv("TLDW_ENV_FILE", str(explicit_env))

    candidates = config._candidate_env_paths(project_root, repo_root)

    assert candidates[0] == explicit_env.resolve()  # nosec B101
    assert project_root / "Config_Files" / ".env" in candidates  # nosec B101


def test_exclusive_tldw_env_file_does_not_load_canonical_env(
    tmp_path: Path,
    monkeypatch,
) -> None:
    sentinel = "TLDW_EXCLUSIVE_ENV_SENTINEL"
    project_root = tmp_path / "tldw_Server_API"
    canonical_env = project_root / "Config_Files" / ".env"
    explicit_env = tmp_path / "runtime.env"
    canonical_env.parent.mkdir(parents=True)
    canonical_env.write_text(f"{sentinel}=host-secret\n", encoding="utf-8")
    explicit_env.write_text("AUTH_MODE=single_user\n", encoding="utf-8")

    monkeypatch.setattr(config, "__file__", str(project_root / "app/core/config.py"))
    monkeypatch.setenv("TLDW_ENV_FILE", str(explicit_env))
    monkeypatch.setenv("TLDW_ENV_FILE_EXCLUSIVE", "1")
    monkeypatch.delenv(sentinel, raising=False)

    config._load_env_files_early()

    assert config._candidate_env_paths(project_root, tmp_path) == [explicit_env.resolve()]  # nosec B101
    assert sentinel not in config.os.environ  # nosec B101


def test_exclusive_tldw_env_file_must_reference_an_existing_file(
    tmp_path: Path,
    monkeypatch,
) -> None:
    missing_env = tmp_path / "missing.env"
    monkeypatch.setenv("TLDW_ENV_FILE", str(missing_env))
    monkeypatch.setenv("TLDW_ENV_FILE_EXCLUSIVE", "1")

    with pytest.raises(FileNotFoundError, match="TLDW_ENV_FILE_EXCLUSIVE"):
        config._candidate_env_paths(tmp_path / "tldw_Server_API", tmp_path)

    assert not missing_env.exists()  # nosec B101


def test_chat_request_schema_does_not_bypass_exclusive_env_file(
    tmp_path: Path,
    monkeypatch,
) -> None:
    explicit_env = tmp_path / "runtime.env"
    working_directory_env = tmp_path / ".env"
    explicit_env.write_text("AUTH_MODE=single_user\n", encoding="utf-8")
    working_directory_env.write_text("HOST_PROVIDER_SECRET=must-not-load\n", encoding="utf-8")
    loaded_paths: list[Path] = []

    def record_dotenv_load(*, dotenv_path=None, **_kwargs) -> bool:
        loaded_paths.append(Path(dotenv_path).resolve())
        return True

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TLDW_ENV_FILE", str(explicit_env))
    monkeypatch.setenv("TLDW_ENV_FILE_EXCLUSIVE", "1")
    monkeypatch.setattr(dotenv, "load_dotenv", record_dotenv_load)

    schema = importlib.import_module(
        "tldw_Server_API.app.api.v1.schemas.chat_request_schemas"
    )
    importlib.reload(schema)

    assert working_directory_env.resolve() not in loaded_paths  # nosec B101
