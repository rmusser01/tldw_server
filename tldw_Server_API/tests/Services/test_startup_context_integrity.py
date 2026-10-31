from __future__ import annotations

from collections.abc import Iterator
import json
import os
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _clear_context_integrity_environment(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    for env_name in (
        "CONTEXT_INTEGRITY_MANIFEST_PATH",
        "CONTEXT_INTEGRITY_HMAC_SECRET",
        "CONTEXT_INTEGRITY_HMAC_KEY_ID",
        "CONTEXT_INTEGRITY_MODE",
    ):
        monkeypatch.delenv(env_name, raising=False)
    for env_name in list(os.environ):
        if env_name.startswith("TLDW_PROMPT_FILE_"):
            monkeypatch.delenv(env_name, raising=False)

    from tldw_Server_API.app.core.Context_Integrity.resolver import (
        clear_global_context_integrity_resolver,
    )

    clear_global_context_integrity_resolver()
    yield
    clear_global_context_integrity_resolver()


def _state() -> object:
    return type("State", (), {})()


def _entry_from_asset(asset: Any) -> dict[str, object]:
    return {
        "asset_id": asset.asset_id,
        "source_type": asset.source_type,
        "digest": asset.digest,
        "display_name": asset.display_name,
        "executable": asset.executable,
        "required": asset.required,
        "owner_scope": asset.owner_scope,
        "path": asset.path,
        "metadata": dict(asset.metadata),
    }


def test_startup_context_integrity_sets_resolver_and_warning(tmp_path: Path) -> None:
    from tldw_Server_API.app.services.startup_context_integrity import (
        produce_context_integrity_startup_warnings,
    )
    from tldw_Server_API.app.services.startup_warning_registry import (
        StartupWarningRegistry,
    )

    prompts = tmp_path / "Prompts"
    prompts.mkdir()
    (prompts / "rag.prompts.yaml").write_text("answer: changed", encoding="utf-8")

    registry = StartupWarningRegistry(startup_id="boot-1")
    app_state = _state()

    findings = produce_context_integrity_startup_warnings(
        app_state=app_state,
        registry=registry,
        prompts_dir=prompts,
        user_skill_roots=[],
        approved_entries=[],
        mode="enforce",
    )

    assert len(findings) == 2
    assert registry.summary(component_prefix="context_integrity")["total"] == 2
    assert app_state.context_integrity_boot_state.degraded is True
    assert any(finding.state == "degraded_integrity" for finding in findings)
    assert app_state.context_integrity_resolver.finding_for("prompt_file:rag.prompts.yaml") is not None


def test_valid_empty_manifest_does_not_degrade(tmp_path: Path) -> None:
    from tldw_Server_API.app.services.startup_context_integrity import (
        produce_context_integrity_startup_warnings,
    )
    from tldw_Server_API.app.services.startup_warning_registry import (
        StartupWarningRegistry,
    )

    prompts = tmp_path / "Prompts"
    prompts.mkdir()
    registry = StartupWarningRegistry(startup_id="boot-2")
    app_state = _state()

    findings = produce_context_integrity_startup_warnings(
        app_state=app_state,
        registry=registry,
        prompts_dir=prompts,
        user_skill_roots=[],
        approved_entries=[],
        manifest_loaded=True,
        manifest_sequence=3,
        manifest_digest="sha256:empty",
        mode="enforce",
    )

    assert findings == ()
    assert registry.summary(component_prefix="context_integrity")["total"] == 0
    assert app_state.context_integrity_boot_state.degraded is False
    assert app_state.context_integrity_boot_state.manifest_sequence == 3
    assert app_state.context_integrity_boot_state.manifest_digest == "sha256:empty"
    assert not any(finding.state == "degraded_integrity" for finding in findings)


def test_missing_env_manifest_degrades_when_approved_entries_not_injected(
    tmp_path: Path,
) -> None:
    from tldw_Server_API.app.services.startup_context_integrity import (
        produce_context_integrity_startup_warnings,
    )
    from tldw_Server_API.app.services.startup_warning_registry import (
        StartupWarningRegistry,
    )

    prompts = tmp_path / "Prompts"
    prompts.mkdir()
    registry = StartupWarningRegistry(startup_id="boot-3")
    app_state = _state()

    findings = produce_context_integrity_startup_warnings(
        app_state=app_state,
        registry=registry,
        prompts_dir=prompts,
        user_skill_roots=[],
        approved_entries=None,
        mode="enforce",
    )

    assert [finding.state for finding in findings] == ["degraded_integrity"]
    assert app_state.context_integrity_boot_state.degraded is True
    assert registry.summary(component_prefix="context_integrity")["total"] == 1


def test_invalid_context_integrity_mode_is_rejected(tmp_path: Path) -> None:
    from tldw_Server_API.app.services.startup_context_integrity import (
        produce_context_integrity_startup_warnings,
    )
    from tldw_Server_API.app.services.startup_warning_registry import (
        StartupWarningRegistry,
    )

    prompts = tmp_path / "Prompts"
    prompts.mkdir()

    with pytest.raises(ValueError, match="Unsupported Context Integrity mode"):
        produce_context_integrity_startup_warnings(
            app_state=_state(),
            registry=StartupWarningRegistry(startup_id="boot-invalid-mode"),
            prompts_dir=prompts,
            user_skill_roots=[],
            approved_entries=[],
            mode="permissive",
        )


@pytest.mark.parametrize("mode", ["audit_only", "enforce", "hardened"])
def test_context_integrity_mode_uses_supported_deployment_setting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    from tldw_Server_API.app.services.startup_context_integrity import (
        produce_context_integrity_startup_warnings,
    )
    from tldw_Server_API.app.services.startup_warning_registry import (
        StartupWarningRegistry,
    )

    prompts = tmp_path / "Prompts"
    prompts.mkdir()
    app_state = _state()
    monkeypatch.setenv("CONTEXT_INTEGRITY_MODE", mode)

    produce_context_integrity_startup_warnings(
        app_state=app_state,
        registry=StartupWarningRegistry(startup_id=f"boot-{mode}"),
        prompts_dir=prompts,
        user_skill_roots=[],
        approved_entries=[],
    )

    assert app_state.context_integrity_boot_state.mode == mode


@pytest.mark.parametrize("deployment_mode", ["audit_only", "permissive"])
def test_explicit_context_integrity_mode_overrides_deployment_setting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    deployment_mode: str,
) -> None:
    from tldw_Server_API.app.services.startup_context_integrity import (
        produce_context_integrity_startup_warnings,
    )
    from tldw_Server_API.app.services.startup_warning_registry import (
        StartupWarningRegistry,
    )

    prompts = tmp_path / "Prompts"
    prompts.mkdir()
    app_state = _state()
    monkeypatch.setenv("CONTEXT_INTEGRITY_MODE", deployment_mode)

    produce_context_integrity_startup_warnings(
        app_state=app_state,
        registry=StartupWarningRegistry(startup_id="boot-explicit-mode"),
        prompts_dir=prompts,
        user_skill_roots=[],
        approved_entries=[],
        manifest_loaded=True,
        mode="enforce",
    )

    assert app_state.context_integrity_boot_state.mode == "enforce"


def test_context_integrity_mode_defaults_to_enforce_when_setting_is_unset(
    tmp_path: Path,
) -> None:
    from tldw_Server_API.app.services.startup_context_integrity import (
        produce_context_integrity_startup_warnings,
    )
    from tldw_Server_API.app.services.startup_warning_registry import (
        StartupWarningRegistry,
    )

    prompts = tmp_path / "Prompts"
    prompts.mkdir()
    app_state = _state()

    produce_context_integrity_startup_warnings(
        app_state=app_state,
        registry=StartupWarningRegistry(startup_id="boot-default-mode"),
        prompts_dir=prompts,
        user_skill_roots=[],
        approved_entries=[],
    )

    assert app_state.context_integrity_boot_state.mode == "enforce"


@pytest.mark.parametrize("mode", ["", "   ", "permissive"])
def test_invalid_context_integrity_mode_setting_blocks_startup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    from tldw_Server_API.app.core.Context_Integrity.inventory import InventoryResult
    from tldw_Server_API.app.services import startup_context_integrity
    from tldw_Server_API.app.services.startup_warning_registry import (
        StartupWarningRegistry,
    )

    empty_inventory = InventoryResult(assets=(), findings=())
    prompt_inventory = Mock(return_value=empty_inventory)
    env_override_inventory = Mock(return_value=empty_inventory)
    user_skill_inventory = Mock(return_value=empty_inventory)
    monkeypatch.setattr(
        startup_context_integrity,
        "inventory_prompt_files_with_findings",
        prompt_inventory,
    )
    monkeypatch.setattr(
        startup_context_integrity,
        "inventory_env_prompt_overrides_with_findings",
        env_override_inventory,
    )
    monkeypatch.setattr(
        startup_context_integrity,
        "inventory_user_skills_with_findings",
        user_skill_inventory,
    )

    prompts = tmp_path / "Prompts"
    prompts.mkdir()
    monkeypatch.setenv("CONTEXT_INTEGRITY_MODE", mode)

    with pytest.raises(
        ValueError,
        match=r"Invalid CONTEXT_INTEGRITY_MODE.*Expected one of",
    ):
        startup_context_integrity.produce_context_integrity_startup_warnings(
            app_state=_state(),
            registry=StartupWarningRegistry(startup_id="boot-invalid-setting"),
            prompts_dir=prompts,
            user_skill_roots=[(42, tmp_path / "skills")],
            approved_entries=[],
        )

    prompt_inventory.assert_not_called()
    env_override_inventory.assert_not_called()
    user_skill_inventory.assert_not_called()


def test_env_manifest_happy_path_populates_boot_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.Context_Integrity.inventory import (
        inventory_prompt_files_with_findings,
    )
    from tldw_Server_API.app.core.Context_Integrity.manifest import (
        HmacManifestSigner,
        create_signed_manifest,
    )
    from tldw_Server_API.app.core.Context_Integrity.resolver import (
        get_global_context_integrity_resolver,
    )
    from tldw_Server_API.app.services.startup_context_integrity import (
        produce_context_integrity_startup_warnings,
    )
    from tldw_Server_API.app.services.startup_warning_registry import (
        StartupWarningRegistry,
    )

    prompts = tmp_path / "Prompts"
    prompts.mkdir()
    (prompts / "rag.prompts.yaml").write_text("answer: approved", encoding="utf-8")
    asset = inventory_prompt_files_with_findings(prompts_dir=prompts).assets[0]
    signer = HmacManifestSigner(key_id="test-key", secret=b"secret")
    signed_manifest = create_signed_manifest(
        sequence=9,
        entries=[_entry_from_asset(asset)],
        signer=signer,
    )
    manifest_path = tmp_path / "context-manifest.json"
    manifest_path.write_text(json.dumps(signed_manifest), encoding="utf-8")
    monkeypatch.setenv("CONTEXT_INTEGRITY_MANIFEST_PATH", str(manifest_path))
    monkeypatch.setenv("CONTEXT_INTEGRITY_HMAC_SECRET", "secret")
    monkeypatch.setenv("CONTEXT_INTEGRITY_HMAC_KEY_ID", "test-key")
    registry = StartupWarningRegistry(startup_id="boot-4")
    app_state = _state()

    findings = produce_context_integrity_startup_warnings(
        app_state=app_state,
        registry=registry,
        prompts_dir=prompts,
        user_skill_roots=[],
        approved_entries=None,
        mode="enforce",
    )

    assert findings == ()
    assert app_state.context_integrity_boot_state.degraded is False
    assert app_state.context_integrity_boot_state.manifest_sequence == 9
    assert app_state.context_integrity_boot_state.manifest_digest == signed_manifest["manifest_digest"]
    assert app_state.context_integrity_boot_state.approved_digests_by_asset_id == {
        asset.asset_id: asset.digest,
    }
    assert get_global_context_integrity_resolver() is app_state.context_integrity_resolver


def test_env_manifest_invalid_signature_degrades(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.Context_Integrity.manifest import (
        HmacManifestSigner,
        create_signed_manifest,
    )
    from tldw_Server_API.app.services.startup_context_integrity import (
        produce_context_integrity_startup_warnings,
    )
    from tldw_Server_API.app.services.startup_warning_registry import (
        StartupWarningRegistry,
    )

    prompts = tmp_path / "Prompts"
    prompts.mkdir()
    signer = HmacManifestSigner(key_id="test-key", secret=b"secret")
    signed_manifest = create_signed_manifest(sequence=4, entries=[], signer=signer)
    signed_manifest["manifest"]["sequence"] = 5
    manifest_path = tmp_path / "context-manifest.json"
    manifest_path.write_text(json.dumps(signed_manifest), encoding="utf-8")
    monkeypatch.setenv("CONTEXT_INTEGRITY_MANIFEST_PATH", str(manifest_path))
    monkeypatch.setenv("CONTEXT_INTEGRITY_HMAC_SECRET", "secret")
    monkeypatch.setenv("CONTEXT_INTEGRITY_HMAC_KEY_ID", "test-key")
    registry = StartupWarningRegistry(startup_id="boot-5")
    app_state = _state()

    findings = produce_context_integrity_startup_warnings(
        app_state=app_state,
        registry=registry,
        prompts_dir=prompts,
        user_skill_roots=[],
        approved_entries=None,
        mode="enforce",
    )

    assert [finding.state for finding in findings] == [
        "signature_invalid",
        "degraded_integrity",
    ]
    assert app_state.context_integrity_boot_state.degraded is True


def test_rich_inventory_findings_are_preserved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.Context_Integrity.inventory import InventoryResult
    from tldw_Server_API.app.core.Context_Integrity.models import (
        ContextAssetDescriptor,
        ContextIntegrityFinding,
    )
    from tldw_Server_API.app.services import startup_context_integrity
    from tldw_Server_API.app.services.startup_context_integrity import (
        produce_context_integrity_startup_warnings,
    )
    from tldw_Server_API.app.services.startup_warning_registry import (
        StartupWarningRegistry,
    )

    prompt_asset = ContextAssetDescriptor(
        asset_id="prompt_file:approved.prompts.yaml",
        source_type="prompt_file",
        digest="sha256:approved",
        display_name="approved.prompts.yaml",
    )
    prompt_finding = ContextIntegrityFinding(
        asset_id="prompt_file:broken.prompts.yaml",
        state="verification_error",
        severity="error",
        summary="prompt inventory failed",
        remediation="fix prompt path",
        source_type="prompt_file",
    )
    env_finding = ContextIntegrityFinding(
        asset_id="prompt_file:env:TLDW_PROMPT_FILE_BAD:bad.yaml",
        state="verification_error",
        severity="error",
        summary="env prompt inventory failed",
        remediation="fix env prompt path",
        source_type="prompt_file",
    )
    skill_finding = ContextIntegrityFinding(
        asset_id="skill:user:42/broken",
        state="verification_error",
        severity="error",
        summary="skill inventory failed",
        remediation="fix skill path",
        source_type="skill_file",
    )

    monkeypatch.setattr(
        startup_context_integrity,
        "inventory_prompt_files_with_findings",
        lambda *, prompts_dir: InventoryResult(
            assets=(prompt_asset,),
            findings=(prompt_finding,),
        ),
    )
    monkeypatch.setattr(
        startup_context_integrity,
        "inventory_env_prompt_overrides_with_findings",
        lambda **_kwargs: InventoryResult(assets=(), findings=(env_finding,)),
    )
    monkeypatch.setattr(
        startup_context_integrity,
        "inventory_user_skills_with_findings",
        lambda *, user_id, skills_root: InventoryResult(
            assets=(),
            findings=(skill_finding,),
        ),
    )
    registry = StartupWarningRegistry(startup_id="boot-6")
    app_state = _state()

    findings = produce_context_integrity_startup_warnings(
        app_state=app_state,
        registry=registry,
        prompts_dir=tmp_path,
        user_skill_roots=[(42, tmp_path / "skills")],
        approved_entries=[_entry_from_asset(prompt_asset)],
        manifest_loaded=True,
        mode="enforce",
    )

    assert tuple(finding.asset_id for finding in findings) == (
        prompt_finding.asset_id,
        env_finding.asset_id,
        skill_finding.asset_id,
    )
    assert registry.summary(component_prefix="context_integrity")["total"] == 3


def test_discover_user_skill_roots_uses_existing_user_database_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.DB_Management import db_path_utils
    from tldw_Server_API.app.services import startup_context_integrity

    base_dir = tmp_path / "user_databases"
    (base_dir / "1" / "skills").mkdir(parents=True)
    (base_dir / "2" / "skills").mkdir(parents=True)
    (base_dir / "not-a-user" / "skills").mkdir(parents=True)
    (base_dir / "3").mkdir()

    monkeypatch.setattr(
        db_path_utils,
        "settings",
        {
            "USER_DB_BASE_DIR": None,
            "USER_DB_BASE": None,
        },
    )
    monkeypatch.delenv("USER_DB_BASE_DIR", raising=False)
    monkeypatch.setenv("USER_DB_BASE", str(base_dir))

    assert startup_context_integrity._discover_user_skill_roots() == [
        (1, base_dir / "1" / "skills"),
        (2, base_dir / "2" / "skills"),
    ]


def test_discover_user_skill_roots_does_not_create_missing_user_database_base(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.DB_Management import db_path_utils
    from tldw_Server_API.app.services import startup_context_integrity

    missing_base = tmp_path / "missing_user_databases"
    monkeypatch.setattr(
        db_path_utils,
        "settings",
        {
            "USER_DB_BASE_DIR": None,
            "USER_DB_BASE": None,
        },
    )
    monkeypatch.setenv("USER_DB_BASE_DIR", str(missing_base))
    monkeypatch.delenv("USER_DB_BASE", raising=False)

    assert startup_context_integrity._discover_user_skill_roots() == []
    assert not missing_base.exists()
