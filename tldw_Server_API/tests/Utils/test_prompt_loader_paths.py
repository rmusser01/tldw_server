import os
from pathlib import Path


def test_prompts_dir_default_points_to_api_config_prompts():
    from tldw_Server_API.app.core.Utils import prompt_loader as pl

    p = Path(pl._prompts_dir()).resolve()
    # Exists and has expected structure: .../tldw_Server_API/Config_Files/Prompts
    assert p.exists(), f"Prompts dir does not exist: {p}"
    assert p.name == "Prompts", f"Unexpected leaf dir: {p.name}"
    assert p.parent.name == "Config_Files", f"Unexpected parent dir: {p.parent}"
    assert p.parent.parent.name == "tldw_Server_API", f"Unexpected api root: {p.parent.parent}"


def test_prompts_dir_respects_env_override(tmp_path, monkeypatch):
    # Create an override config dir with Prompts subfolder
    cfg_dir = tmp_path / "my_config"
    prompts = cfg_dir / "Prompts"
    prompts.mkdir(parents=True)
    # Set env to override
    monkeypatch.setenv("TLDW_CONFIG_DIR", str(cfg_dir))
    try:
        from tldw_Server_API.app.core.Utils import prompt_loader as pl

        p = Path(pl._prompts_dir()).resolve()
        assert p == prompts.resolve(), f"Env override not respected: {p} vs {prompts}"
    finally:
        monkeypatch.delenv("TLDW_CONFIG_DIR", raising=False)


def test_load_prompt_markdown_key_found(tmp_path, monkeypatch):
    cfg_dir = tmp_path / "my_config"
    prompts = cfg_dir / "Prompts"
    prompts.mkdir(parents=True)
    (prompts / "demo.prompts.md").write_text(
        "# Existing Key\n```\nhello from prompt\n```\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("TLDW_CONFIG_DIR", str(cfg_dir))
    try:
        from tldw_Server_API.app.core.Utils import prompt_loader as pl

        value = pl.load_prompt("demo", "Existing Key")
        assert value == "hello from prompt"
    finally:
        monkeypatch.delenv("TLDW_CONFIG_DIR", raising=False)


def test_load_prompt_markdown_missing_key_returns_none(tmp_path, monkeypatch):
    cfg_dir = tmp_path / "my_config"
    prompts = cfg_dir / "Prompts"
    prompts.mkdir(parents=True)
    (prompts / "demo.prompts.md").write_text(
        "# Existing Key\n```\nhello from prompt\n```\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("TLDW_CONFIG_DIR", str(cfg_dir))
    try:
        from tldw_Server_API.app.core.Utils import prompt_loader as pl

        value = pl.load_prompt("demo", "Missing Key")
        assert value is None
    finally:
        monkeypatch.delenv("TLDW_CONFIG_DIR", raising=False)


def test_load_prompt_blocks_quarantined_prompt_file(tmp_path, monkeypatch):
    from tldw_Server_API.app.core.Context_Integrity.models import (
        ContextIntegrityBootState,
        ContextIntegrityFinding,
    )
    from tldw_Server_API.app.core.Context_Integrity.resolver import (
        ContextIntegrityResolver,
        clear_global_context_integrity_resolver,
        set_global_context_integrity_resolver,
    )
    from tldw_Server_API.app.core.Utils import prompt_loader as pl

    cfg_dir = tmp_path / "cfg"
    prompts = cfg_dir / "Prompts"
    prompts.mkdir(parents=True)
    (prompts / "demo.prompts.md").write_text(
        "# Existing Key\n```\nfrom-md\n```\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("TLDW_CONFIG_DIR", str(cfg_dir))
    resolver = ContextIntegrityResolver(
        ContextIntegrityBootState(
            mode="enforce",
            degraded=False,
            manifest_sequence=1,
            manifest_digest="sha256:manifest",
            findings=(
                ContextIntegrityFinding(
                    asset_id="prompt_file:demo.prompts.md",
                    state="changed_approved_non_executable",
                    severity="warning",
                    summary="changed",
                    remediation="review",
                    source_type="prompt_file",
                ),
            ),
        )
    )
    set_global_context_integrity_resolver(resolver)
    try:
        assert pl.load_prompt("demo", "Existing Key") is None
    finally:
        clear_global_context_integrity_resolver()


def test_load_prompt_uses_verified_bytes_without_second_read(tmp_path, monkeypatch):
    from tldw_Server_API.app.core.Utils import prompt_loader as pl

    cfg_dir = tmp_path / "cfg"
    prompts = cfg_dir / "Prompts"
    prompts.mkdir(parents=True)
    prompt_file = prompts / "demo.prompts.md"
    prompt_file.write_text("# Existing Key\n```\nfrom-md\n```\n", encoding="utf-8")
    monkeypatch.setenv("TLDW_CONFIG_DIR", str(cfg_dir))

    read_count = {"count": 0}
    original_read_text = pl._read_prompt_file_text

    def _counting_read(path, *args, **kwargs):
        read_count["count"] += 1
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(pl, "_read_prompt_file_text", _counting_read)

    assert pl.load_prompt("demo", "Existing Key") == "from-md"
    assert read_count["count"] == 1


def test_load_prompt_blocks_live_edit_after_boot(tmp_path, monkeypatch):
    from tldw_Server_API.app.core.Context_Integrity.inventory import (
        inventory_prompt_files,
    )
    from tldw_Server_API.app.core.Context_Integrity.models import ContextIntegrityBootState
    from tldw_Server_API.app.core.Context_Integrity.resolver import (
        ContextIntegrityResolver,
        clear_global_context_integrity_resolver,
        set_global_context_integrity_resolver,
    )
    from tldw_Server_API.app.core.Utils import prompt_loader as pl

    cfg_dir = tmp_path / "cfg"
    prompts = cfg_dir / "Prompts"
    prompts.mkdir(parents=True)
    prompt_file = prompts / "demo.prompts.md"
    prompt_file.write_text("# Existing Key\n```\nfrom-md\n```\n", encoding="utf-8")
    monkeypatch.setenv("TLDW_CONFIG_DIR", str(cfg_dir))
    asset = inventory_prompt_files(prompts_dir=prompts)[0]
    resolver = ContextIntegrityResolver(
        ContextIntegrityBootState(
            mode="enforce",
            degraded=False,
            manifest_sequence=1,
            manifest_digest="sha256:manifest",
            approved_digests_by_asset_id={asset.asset_id: asset.digest},
        )
    )
    prompt_file.write_text("# Existing Key\n```\nmodified\n```\n", encoding="utf-8")
    set_global_context_integrity_resolver(resolver)
    try:
        assert pl.load_prompt("demo", "Existing Key") is None
    finally:
        clear_global_context_integrity_resolver()
