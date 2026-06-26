from unittest.mock import MagicMock


def test_load_prompt_prefers_env_prompt_file(tmp_path, monkeypatch):
    cfg_dir = tmp_path / "cfg"
    prompts = cfg_dir / "Prompts"
    prompts.mkdir(parents=True)
    (prompts / "demo.prompts.md").write_text(
        "# Existing Key\n```\nfrom-md\n```\n",
        encoding="utf-8",
    )

    override_file = tmp_path / "override.txt"
    override_file.write_text("from-env-file", encoding="utf-8")

    monkeypatch.setenv("TLDW_CONFIG_DIR", str(cfg_dir))
    monkeypatch.setenv("TLDW_PROMPT_FILE_DEMO__EXISTING_KEY", str(override_file))

    from tldw_Server_API.app.core.Utils import prompt_loader as pl

    assert pl.load_prompt("demo", "Existing Key") == "from-env-file"


def test_load_env_prompt_file_logs_warning_on_oserror(monkeypatch):
    env_name = "TLDW_PROMPT_FILE_DEMO__EXISTING_KEY"
    monkeypatch.setenv(env_name, "/definitely/missing/prompt-override.txt")

    from tldw_Server_API.app.core.Utils import prompt_loader as pl

    mock_warning = MagicMock()
    monkeypatch.setattr(pl.logger, "warning", mock_warning, raising=True)

    assert pl._load_env_prompt_file("demo", "existing key") is None
    mock_warning.assert_called_once()
    warning_call = mock_warning.call_args[0]
    assert "Prompt override file read failed for env" in warning_call[0]
    assert warning_call[1] == env_name
    assert warning_call[2] == "demo"
    assert warning_call[3] == "existing key"


def test_env_prompt_override_uses_integrity_source_label(tmp_path, monkeypatch):
    env_name = "TLDW_PROMPT_FILE_DEMO__EXISTING_KEY"
    override_file = tmp_path / "override.txt"
    override_file.write_text("approved-env-file", encoding="utf-8")
    monkeypatch.setenv(env_name, str(override_file))

    from tldw_Server_API.app.core.Context_Integrity.inventory import (
        inventory_env_prompt_overrides,
    )
    from tldw_Server_API.app.core.Context_Integrity.models import ContextIntegrityBootState
    from tldw_Server_API.app.core.Context_Integrity.resolver import (
        ContextIntegrityResolver,
        clear_global_context_integrity_resolver,
        set_global_context_integrity_resolver,
    )
    from tldw_Server_API.app.core.Utils import prompt_loader as pl

    asset = inventory_env_prompt_overrides(environ={env_name: str(override_file)})[0]
    resolver = ContextIntegrityResolver(
        ContextIntegrityBootState(
            mode="enforce",
            degraded=False,
            manifest_sequence=1,
            manifest_digest="sha256:manifest",
            approved_digests_by_asset_id={asset.asset_id: asset.digest},
        )
    )
    set_global_context_integrity_resolver(resolver)
    try:
        assert pl.load_prompt("demo", "Existing Key") == "approved-env-file"
        override_file.write_text("modified-env-file", encoding="utf-8")
        assert pl.load_prompt("demo", "Existing Key") is None
    finally:
        clear_global_context_integrity_resolver()
