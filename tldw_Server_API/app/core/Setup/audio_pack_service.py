"""Offline-pack helpers for setup audio bundle manifests."""

from __future__ import annotations

import hashlib
import json
import platform
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.Setup import setup_manager
from tldw_Server_API.app.core.Setup.audio_readiness_store import AudioReadinessStore
from tldw_Server_API.app.core.Setup.audio_bundle_catalog import (
    AUDIO_BUNDLE_CATALOG_VERSION,
    DEFAULT_AUDIO_RESOURCE_PROFILE,
    build_audio_selection_key,
    get_audio_bundle_catalog,
)

CONFIG_ROOT = setup_manager.CONFIG_RELATIVE_PATH.parent
AUDIO_PACK_FORMAT = "audio_bundle_pack_manifest_v1"
AUDIO_PACKS_DIRNAME = "audio_packs"
_AUDIO_PACK_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*\.json$")

PACK_ISSUE_UNKNOWN_BUNDLE = "unknown_bundle"
PACK_ISSUE_MANIFEST_CHECKSUM = "manifest_checksum_mismatch"
PACK_ISSUE_PLATFORM_MISMATCH = "platform_mismatch"
PACK_ISSUE_ARCH_MISMATCH = "arch_mismatch"
PACK_ISSUE_PYTHON_MISMATCH = "python_version_mismatch"
PACK_ISSUE_INVALID_TTS_CHOICE = "invalid_tts_choice"
PACK_ISSUE_SELECTION_KEY_MISMATCH = "selection_key_mismatch"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalise_python_version(version: str | None = None) -> str:
    if version:
        parts = str(version).split(".")
        if len(parts) >= 2:
            return f"{parts[0]}.{parts[1]}"
        return str(version)
    return f"{sys.version_info.major}.{sys.version_info.minor}"


def _default_compatibility() -> dict[str, str]:
    return {
        "platform": platform.system().lower(),
        "arch": platform.machine().lower(),
        "python_version": _normalise_python_version(),
    }


def get_audio_pack_root() -> Path:
    """Return the setup-managed directory used for offline audio pack manifests."""
    root = CONFIG_ROOT / AUDIO_PACKS_DIRNAME
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def normalize_audio_pack_name(pack_name: str) -> str:
    """Validate that a caller provided only a bare JSON filename for a managed pack."""
    normalized = str(pack_name or "").strip()
    if not normalized or "/" in normalized or "\\" in normalized or not _AUDIO_PACK_NAME_PATTERN.fullmatch(normalized):
        raise ValueError(
            "Audio pack names must be plain JSON filenames inside the managed audio_packs directory."
        )
    return normalized


def resolve_audio_pack_path(pack_name: str) -> Path:
    """Resolve a managed pack filename into the setup-controlled audio pack directory."""
    root = get_audio_pack_root()
    candidate = (root / normalize_audio_pack_name(pack_name)).resolve()
    if candidate.parent != root:
        raise ValueError(
            "Audio pack names must be plain JSON filenames inside the managed audio_packs directory."
        )
    return candidate


def _resolve_audio_pack_read_path(pack_name: str | Path) -> Path:
    if isinstance(pack_name, Path):
        return pack_name.expanduser().resolve()
    return resolve_audio_pack_path(pack_name)


def _display_audio_pack_path(pack_name: str | Path) -> str:
    if isinstance(pack_name, Path):
        return str(pack_name.expanduser().resolve())
    return str(Path(AUDIO_PACKS_DIRNAME) / normalize_audio_pack_name(pack_name))


def _canonical_manifest_payload(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256_hexdigest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file_hexdigest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_asset_manifest(installed_assets: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    return [dict(entry) for entry in (installed_assets or []) if isinstance(entry, dict)]


def _append_issue(issues: list[str], issue_codes: list[str], code: str, message: str) -> None:
    issues.append(message)
    issue_codes.append(code)


def _calculate_asset_checksums(assets: list[dict[str, Any]]) -> dict[str, str]:
    checksums: dict[str, str] = {}
    for asset in assets:
        path_value = asset.get("path") or asset.get("asset_path")
        if not path_value:
            continue
        asset_path = Path(path_value)
        if not asset_path.is_file():
            continue
        checksums[str(asset_path)] = _sha256_file_hexdigest(asset_path)
    return checksums


def build_audio_pack_manifest(
    *,
    bundle_id: str,
    resource_profile: str = DEFAULT_AUDIO_RESOURCE_PROFILE,
    tts_choice: str | None = None,
    catalog_version: str = AUDIO_BUNDLE_CATALOG_VERSION,
    compatibility: dict[str, str] | None = None,
    installed_assets: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a portable v1 audio pack manifest."""

    bundle = get_audio_bundle_catalog().bundle_by_id(bundle_id)
    profile = bundle.profile_by_id(resource_profile)
    try:
        canonical_tts_choice = profile.canonical_tts_choice(tts_choice)
    except KeyError as exc:
        raise ValueError(str(exc)) from exc
    assets = _copy_asset_manifest(installed_assets)

    manifest = {
        "format": AUDIO_PACK_FORMAT,
        "bundle_id": bundle.bundle_id,
        "bundle_label": bundle.label,
        "resource_profile": profile.profile_id,
        "tts_choice": canonical_tts_choice,
        "profile_label": profile.label,
        "catalog_version": catalog_version,
        "selection_key": build_audio_selection_key(
            bundle.bundle_id,
            profile.profile_id,
            catalog_version,
            tts_choice=canonical_tts_choice,
        ),
        "compatibility": compatibility or _default_compatibility(),
        "assets": assets,
        "created_at": _utc_now(),
        "checksums": {},
    }

    manifest_without_checksums = dict(manifest)
    manifest_without_checksums["checksums"] = {}
    manifest["checksums"] = {
        "manifest_sha256": _sha256_hexdigest(_canonical_manifest_payload(manifest_without_checksums)),
        "assets": _calculate_asset_checksums(assets),
    }
    return manifest


def write_audio_pack_manifest(
    *,
    pack_name: str,
    bundle_id: str,
    resource_profile: str = DEFAULT_AUDIO_RESOURCE_PROFILE,
    tts_choice: str | None = None,
    catalog_version: str = AUDIO_BUNDLE_CATALOG_VERSION,
    compatibility: dict[str, str] | None = None,
    installed_assets: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Write an audio pack manifest to disk and return the manifest."""

    manifest = build_audio_pack_manifest(
        bundle_id=bundle_id,
        resource_profile=resource_profile,
        tts_choice=tts_choice,
        catalog_version=catalog_version,
        compatibility=compatibility,
        installed_assets=installed_assets,
    )
    destination = resolve_audio_pack_path(pack_name)
    destination.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_audio_pack_manifest(pack_name: str | Path) -> dict[str, Any]:
    """Load an audio pack manifest from disk."""

    return json.loads(_resolve_audio_pack_read_path(pack_name).read_text(encoding="utf-8"))


def validate_audio_pack_manifest(
    pack_name: str | Path,
    *,
    machine_profile: dict[str, Any] | None = None,
    python_version: str | None = None,
) -> dict[str, Any]:
    """Validate manifest checksum and local compatibility for an audio pack."""

    raw_manifest = load_audio_pack_manifest(pack_name)
    issues: list[str] = []
    issue_codes: list[str] = []
    warnings: list[str] = []

    if not isinstance(raw_manifest, dict):
        return {
            "compatible": False,
            "issues": ["Audio pack manifest must be a JSON object."],
            "warnings": warnings,
            "manifest": {},
            "selection_key": None,
            "bundle_label": None,
        }

    manifest = dict(raw_manifest)

    if manifest.get("format") != AUDIO_PACK_FORMAT:
        _append_issue(issues, issue_codes, "unsupported_format", "Unsupported audio pack format.")

    bundle_id = manifest.get("bundle_id")
    resource_profile = manifest.get("resource_profile")
    catalog_version = manifest.get("catalog_version") or AUDIO_BUNDLE_CATALOG_VERSION
    canonical_tts_choice = None
    try:
        bundle = get_audio_bundle_catalog().bundle_by_id(bundle_id)
        profile = bundle.profile_by_id(resource_profile)
    except KeyError:
        bundle = None
        profile = None
        _append_issue(
            issues,
            issue_codes,
            PACK_ISSUE_UNKNOWN_BUNDLE,
            "Referenced audio bundle or resource profile is not available in this catalog.",
        )

    checksum_payload = dict(manifest)
    checksum_payload["checksums"] = {}
    expected_manifest_checksum = _sha256_hexdigest(_canonical_manifest_payload(checksum_payload))
    checksums = manifest.get("checksums")
    if checksums is None:
        checksums = {}
    elif not isinstance(checksums, dict):
        issues.append("Manifest checksums entry must be an object.")
        checksums = {}
    actual_manifest_checksum = checksums.get("manifest_sha256")
    if actual_manifest_checksum != expected_manifest_checksum:
        _append_issue(
            issues,
            issue_codes,
            PACK_ISSUE_MANIFEST_CHECKSUM,
            "Manifest checksum mismatch.",
        )

    local_profile = machine_profile or _default_compatibility()
    compatibility = manifest.get("compatibility")
    if compatibility is None:
        compatibility = {}
    elif not isinstance(compatibility, dict):
        issues.append("Manifest compatibility entry must be an object.")
        compatibility = {}
    if compatibility.get("platform") and compatibility["platform"] != local_profile.get("platform"):
        _append_issue(
            issues,
            issue_codes,
            PACK_ISSUE_PLATFORM_MISMATCH,
            f"Pack platform {compatibility['platform']} does not match local platform {local_profile.get('platform')}."
        )
    if compatibility.get("arch") and compatibility["arch"] != local_profile.get("arch"):
        _append_issue(
            issues,
            issue_codes,
            PACK_ISSUE_ARCH_MISMATCH,
            f"Pack arch {compatibility['arch']} does not match local arch {local_profile.get('arch')}.",
        )

    expected_python = _normalise_python_version(compatibility.get("python_version"))
    local_python = _normalise_python_version(python_version)
    if expected_python and expected_python != local_python:
        _append_issue(
            issues,
            issue_codes,
            PACK_ISSUE_PYTHON_MISMATCH,
            f"Pack Python {expected_python} does not match local Python {local_python}.",
        )

    if profile is not None:
        try:
            expected_tts_choice = profile.canonical_tts_choice(manifest.get("tts_choice"))
        except KeyError as exc:
            _append_issue(
                issues,
                issue_codes,
                PACK_ISSUE_INVALID_TTS_CHOICE,
                str(exc),
            )
            expected_tts_choice = None
        canonical_tts_choice = expected_tts_choice

    canonical_selection_key = build_audio_selection_key(
        bundle_id,
        resource_profile,
        catalog_version,
        tts_choice=canonical_tts_choice,
    )
    manifest_selection_key = manifest.get("selection_key")
    if manifest_selection_key != canonical_selection_key:
        _append_issue(
            issues,
            issue_codes,
            PACK_ISSUE_SELECTION_KEY_MISMATCH,
            "Pack selection key does not match the canonical bundle/profile/TTS choice identity.",
        )

    manifest_assets = manifest.get("assets")
    if manifest_assets is None:
        manifest_assets = []
    elif not isinstance(manifest_assets, list):
        issues.append("Manifest assets entry must be a list.")
        manifest_assets = []

    for asset in manifest_assets:
        if not isinstance(asset, dict):
            issues.append("Manifest assets entries must be objects.")
            continue
        path_value = asset.get("path") or asset.get("asset_path")
        if not path_value:
            continue
        asset_path = Path(path_value)
        if not asset_path.exists():
            warnings.append(f"Referenced asset is not present locally: {asset_path}")

    selection_key = manifest.get("selection_key") if isinstance(manifest.get("selection_key"), str) else None
    if not selection_key and isinstance(bundle_id, str) and isinstance(resource_profile, str):
        selection_key = build_audio_selection_key(bundle_id, resource_profile, catalog_version)

    return {
        "compatible": not issues,
        "issues": issues,
        "issue_codes": issue_codes,
        "warnings": warnings,
        "manifest": manifest,
        "tts_choice": canonical_tts_choice,
        "selection_key": canonical_selection_key,
        "bundle_label": bundle.label if bundle else None,
    }


def register_imported_audio_pack(
    pack_name: str | Path,
    *,
    readiness_store: AudioReadinessStore,
    machine_profile: dict[str, Any] | None = None,
    python_version: str | None = None,
) -> dict[str, Any]:
    """Validate an imported pack and persist its metadata into readiness."""

    validation = validate_audio_pack_manifest(
        pack_name,
        machine_profile=machine_profile,
        python_version=python_version,
    )
    readiness = readiness_store.load()
    manifest = validation["manifest"]
    blocking_codes = {
        PACK_ISSUE_INVALID_TTS_CHOICE,
        PACK_ISSUE_SELECTION_KEY_MISMATCH,
        PACK_ISSUE_MANIFEST_CHECKSUM,
        PACK_ISSUE_UNKNOWN_BUNDLE,
    }
    blocking_issue = next(
        (
            issue
            for code, issue in zip(validation["issue_codes"], validation["issues"], strict=False)
            if code in blocking_codes
        ),
        None,
    )
    if blocking_issue is not None:
        raise ValueError(blocking_issue)
    imported_packs = list(readiness.get("imported_packs") or [])
    imported_packs.append(
        {
            "pack_path": _display_audio_pack_path(pack_name),
            "bundle_id": manifest.get("bundle_id"),
            "resource_profile": manifest.get("resource_profile"),
            "tts_choice": validation["tts_choice"],
            "catalog_version": manifest.get("catalog_version"),
            "selection_key": validation["selection_key"],
            "compatible": validation["compatible"],
            "issues": list(validation["issues"]),
            "warnings": list(validation["warnings"]),
            "imported_at": _utc_now(),
            "manifest_sha256": (manifest.get("checksums") if isinstance(manifest.get("checksums"), dict) else {}).get(
                "manifest_sha256"
            ),
        }
    )

    updated = readiness_store.update(
        selected_bundle_id=manifest.get("bundle_id"),
        selected_resource_profile=manifest.get("resource_profile") or DEFAULT_AUDIO_RESOURCE_PROFILE,
        tts_choice=validation["tts_choice"],
        catalog_version=manifest.get("catalog_version") or AUDIO_BUNDLE_CATALOG_VERSION,
        selection_key=validation["selection_key"],
        machine_profile=machine_profile or readiness.get("machine_profile"),
        installed_asset_manifests=manifest.get("assets", []),
        imported_packs=imported_packs,
    )
    return {
        **validation,
        "audio_readiness": updated,
    }


__all__ = [
    "AUDIO_PACK_FORMAT",
    "AUDIO_PACKS_DIRNAME",
    "build_audio_pack_manifest",
    "get_audio_pack_root",
    "load_audio_pack_manifest",
    "normalize_audio_pack_name",
    "register_imported_audio_pack",
    "resolve_audio_pack_path",
    "validate_audio_pack_manifest",
    "write_audio_pack_manifest",
]
