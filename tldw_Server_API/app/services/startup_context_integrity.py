"""Startup producer for Context Integrity warnings and resolver state."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.config_paths import resolve_prompts_dir
from tldw_Server_API.app.core.Context_Integrity.inventory import (
    InventoryResult,
    inventory_env_prompt_overrides_with_findings,
    inventory_prompt_files_with_findings,
    inventory_user_skills_with_findings,
)
from tldw_Server_API.app.core.Context_Integrity.manifest import (
    HmacManifestSigner,
    verify_signed_manifest,
)
from tldw_Server_API.app.core.Context_Integrity.models import (
    ContextAssetDescriptor,
    ContextAssetSource,
    ContextIntegrityBootState,
    ContextIntegrityFinding,
)
from tldw_Server_API.app.core.Context_Integrity.resolver import (
    ContextIntegrityResolver,
    set_global_context_integrity_resolver,
)
from tldw_Server_API.app.core.Context_Integrity.verifier import verify_inventory
from tldw_Server_API.app.core.DB_Management import db_path_utils
from tldw_Server_API.app.services.startup_warning_models import StartupWarningRecord
from tldw_Server_API.app.services.startup_warning_registry import (
    StartupWarningRegistry,
)


def _signature_invalid_finding(
    *,
    summary: str,
    details: Mapping[str, Any] | None = None,
) -> ContextIntegrityFinding:
    return ContextIntegrityFinding(
        asset_id="manifest:env",
        state="signature_invalid",
        severity="error",
        summary=summary,
        remediation="Fix the configured manifest path/key or import a valid signed manifest.",
        source_type="manifest",
        details=details or {},
    )


def _verification_error_finding(
    *,
    asset_id: str,
    source_type: ContextAssetSource,
    summary: str,
    details: Mapping[str, Any] | None = None,
) -> ContextIntegrityFinding:
    return ContextIntegrityFinding(
        asset_id=asset_id,
        state="verification_error",
        severity="error",
        summary=summary,
        remediation="Fix file permissions or remove unsafe context paths, then restart.",
        source_type=source_type,
        details=details or {},
    )


def _load_startup_manifest_from_env() -> tuple[
    tuple[Mapping[str, Any], ...],
    int | None,
    str | None,
    bool,
    ContextIntegrityFinding | None,
]:
    """Load and verify an operator-provided signed manifest from environment."""
    manifest_path = os.getenv("CONTEXT_INTEGRITY_MANIFEST_PATH")
    secret = os.getenv("CONTEXT_INTEGRITY_HMAC_SECRET")
    key_id = os.getenv("CONTEXT_INTEGRITY_HMAC_KEY_ID") or "local-hmac"

    if not manifest_path and not secret:
        return (), None, None, False, None
    if not manifest_path or not secret:
        return (
            (),
            None,
            None,
            False,
            _signature_invalid_finding(
                summary="Context Integrity manifest environment configuration is incomplete.",
                details={
                    "has_manifest_path": bool(manifest_path),
                    "has_hmac_secret": bool(secret),
                },
            ),
        )

    try:
        signed_manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        verified = verify_signed_manifest(
            signed_manifest,
            signer=HmacManifestSigner(
                key_id=key_id,
                secret=secret.encode("utf-8"),
            ),
        )
    except (OSError, ValueError, TypeError) as exc:
        return (
            (),
            None,
            None,
            False,
            _signature_invalid_finding(
                summary="Configured Context Integrity manifest could not be verified.",
                details={
                    "error_type": exc.__class__.__name__,
                    "manifest_path": manifest_path,
                },
            ),
        )

    return (
        verified.entries,
        verified.sequence,
        verified.manifest_digest,
        True,
        None,
    )


def _settings_value(name: str) -> Any:
    try:
        return db_path_utils.settings.get(name)
    except Exception:
        return None


def _resolve_candidate_like_database_paths(
    raw_path: str | Path | None,
    *,
    project_root: Path,
) -> Path | None:
    if not raw_path:
        return None
    try:
        candidate = Path(raw_path).expanduser()
        if not candidate.is_absolute():
            return (project_root / candidate).resolve()
        return candidate.resolve()
    except Exception:
        return None


def _resolve_user_db_base_dir_for_discovery() -> Path:
    """Resolve the user DB base path for read-only discovery without creating it."""
    env_user_db_base = os.getenv("USER_DB_BASE_DIR")
    settings_user_db_base = _settings_value("USER_DB_BASE_DIR")
    project_root = Path(db_path_utils.get_project_root())
    default_base = (project_root / "Databases" / "user_databases").resolve()
    user_db_base = settings_user_db_base or env_user_db_base

    if db_path_utils._is_test_context() and env_user_db_base:
        settings_candidate = _resolve_candidate_like_database_paths(
            settings_user_db_base,
            project_root=project_root,
        )
        if settings_candidate is None or settings_candidate == default_base:
            user_db_base = env_user_db_base

    if db_path_utils._is_test_context() and not env_user_db_base:
        settings_candidate = _resolve_candidate_like_database_paths(
            settings_user_db_base,
            project_root=project_root,
        )
        if settings_candidate is None or settings_candidate == default_base:
            user_db_base = None

    if not user_db_base:
        legacy_base = os.getenv("USER_DB_BASE") or _settings_value("USER_DB_BASE")
        if legacy_base:
            logger.warning(
                "USER_DB_BASE is deprecated; use USER_DB_BASE_DIR instead. "
                "Context Integrity discovery will stop honoring USER_DB_BASE "
                "in a future release."
            )
            user_db_base = legacy_base

    if not user_db_base:
        if db_path_utils._is_test_context():
            run_tag = db_path_utils._get_test_fallback_run_tag()
            safe_run_tag = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(run_tag))
            return (Path(tempfile.gettempdir()) / "tldw_user_databases_test" / safe_run_tag).resolve()
        logger.warning(
            "USER_DB_BASE_DIR not configured, using fallback for Context Integrity " "discovery: {}",
            default_base,
        )
        return default_base

    return db_path_utils._normalize_user_db_base_dir(Path(user_db_base))


def _discover_user_skill_roots() -> list[tuple[int, Path]]:
    """Discover existing per-user skill roots without creating skill folders."""
    try:
        base_dir = _resolve_user_db_base_dir_for_discovery()
    except Exception as exc:  # pragma: no cover - defensive startup guard
        logger.warning("Context Integrity could not discover user skill roots: {}", exc)
        return []

    if not base_dir.exists():
        return []

    roots: list[tuple[int, Path]] = []
    try:
        user_dirs = sorted(path for path in base_dir.iterdir() if path.is_dir())
    except OSError as exc:
        logger.warning("Context Integrity could not scan user skill roots: {}", exc)
        return []

    for user_dir in user_dirs:
        if not user_dir.name.isdigit():
            continue
        skills_root = user_dir / "skills"
        if skills_root.is_dir():
            roots.append((int(user_dir.name), skills_root))
    return roots


def _finding_to_warning(finding: ContextIntegrityFinding) -> StartupWarningRecord:
    details = {
        "asset_id": finding.asset_id,
        "current_digest": finding.current_digest,
        "approved_digest": finding.approved_digest,
    }
    details.update(dict(finding.details))
    return StartupWarningRecord(
        component=f"context_integrity.{finding.source_type}",
        severity=finding.severity,
        startup_action="warn",
        code=finding.state,
        summary=finding.summary,
        remediation=finding.remediation,
        details=details,
        detected_at=finding.detected_at,
    )


def _collect_inventory_result(
    result: InventoryResult,
    *,
    current_assets: list[ContextAssetDescriptor],
    findings: list[ContextIntegrityFinding],
) -> None:
    current_assets.extend(result.assets)
    findings.extend(result.findings)


def _approved_digests_by_asset_id(
    approved_entries: Iterable[Mapping[str, Any]],
) -> dict[str, str]:
    approved_digests: dict[str, str] = {}
    for entry in approved_entries:
        if "asset_id" not in entry or "digest" not in entry:
            continue
        approved_digests[str(entry["asset_id"])] = str(entry["digest"])
    return approved_digests


def produce_context_integrity_startup_warnings(
    *,
    app_state: object,
    registry: StartupWarningRegistry,
    prompts_dir: Path | None = None,
    user_skill_roots: Iterable[tuple[int, Path]] | None = None,
    approved_entries: Iterable[Mapping[str, Any]] | None = None,
    manifest_loaded: bool | None = None,
    manifest_sequence: int | None = None,
    manifest_digest: str | None = None,
    mode: str = "enforce",
) -> tuple[ContextIntegrityFinding, ...]:
    """Build inventory, verify approved state, attach resolver, and emit warnings."""
    current_assets: list[ContextAssetDescriptor] = []
    findings_list: list[ContextIntegrityFinding] = []
    resolved_prompts_dir = prompts_dir or resolve_prompts_dir()

    try:
        _collect_inventory_result(
            inventory_prompt_files_with_findings(prompts_dir=resolved_prompts_dir),
            current_assets=current_assets,
            findings=findings_list,
        )
    except OSError as exc:
        findings_list.append(
            _verification_error_finding(
                asset_id="prompt_file:*",
                source_type="prompt_file",
                summary="Prompt file inventory failed.",
                details={
                    "error_type": exc.__class__.__name__,
                    "path": str(resolved_prompts_dir),
                },
            )
        )

    try:
        _collect_inventory_result(
            inventory_env_prompt_overrides_with_findings(),
            current_assets=current_assets,
            findings=findings_list,
        )
    except OSError as exc:
        findings_list.append(
            _verification_error_finding(
                asset_id="prompt_file:env:*",
                source_type="prompt_file",
                summary="Environment prompt override inventory failed.",
                details={"error_type": exc.__class__.__name__},
            )
        )

    roots = list(user_skill_roots) if user_skill_roots is not None else _discover_user_skill_roots()
    for user_id, skills_root in roots:
        try:
            _collect_inventory_result(
                inventory_user_skills_with_findings(
                    user_id=user_id,
                    skills_root=skills_root,
                ),
                current_assets=current_assets,
                findings=findings_list,
            )
        except OSError as exc:
            findings_list.append(
                _verification_error_finding(
                    asset_id=f"skill:user:{user_id}:*",
                    source_type="skill_file",
                    summary=f"Skill inventory failed for user {user_id}.",
                    details={
                        "error_type": exc.__class__.__name__,
                        "path": str(skills_root),
                    },
                )
            )

    if approved_entries is None:
        (
            approved_entries_tuple,
            manifest_sequence,
            manifest_digest,
            loaded_from_env,
            manifest_finding,
        ) = _load_startup_manifest_from_env()
        manifest_loaded = loaded_from_env
        if manifest_finding is not None:
            findings_list.append(manifest_finding)
    else:
        approved_entries_tuple = tuple(approved_entries)
        manifest_loaded = bool(manifest_loaded)

    findings_list.extend(
        verify_inventory(
            current_assets=current_assets,
            approved_entries=approved_entries_tuple,
        )
    )

    degraded = not bool(manifest_loaded)
    if degraded:
        findings_list.append(
            ContextIntegrityFinding(
                asset_id="manifest:none",
                state="degraded_integrity",
                severity="error",
                summary="No approved Context Integrity manifest was loaded.",
                remediation="Import or approve a signed manifest before protected assets are used.",
                source_type="manifest",
            )
        )

    findings = tuple(findings_list)
    boot_state = ContextIntegrityBootState(
        mode=mode,  # type: ignore[arg-type]
        degraded=degraded,
        manifest_sequence=manifest_sequence,
        manifest_digest=manifest_digest,
        approved_digests_by_asset_id=_approved_digests_by_asset_id(
            approved_entries_tuple,
        ),
        findings=findings,
    )
    resolver = ContextIntegrityResolver(boot_state)
    setattr(app_state, "context_integrity_resolver", resolver)
    setattr(app_state, "context_integrity_boot_state", boot_state)
    set_global_context_integrity_resolver(resolver)

    for finding in findings:
        registry.add_warning(_finding_to_warning(finding))
        logger.warning(
            "Context integrity startup finding: {} {}",
            finding.state,
            finding.asset_id,
        )

    return findings
