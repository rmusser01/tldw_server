from pathlib import Path

from tldw_Server_API.app.core import config


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
