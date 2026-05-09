"""Persona visual pack archive assembler."""

from __future__ import annotations

import uuid
import zipfile
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.Persona.visual_service import VISUAL_STORAGE_PREFIX

from .archive import validate_archive_members
from .constants import (
    ASSET_BYTES_STATUS_MISSING,
    ASSET_BYTES_STATUS_PRESENT,
    CHECKSUMS_PATH,
    MANIFEST_PATH,
    PERSONA_VISUAL_PACK_EXTENSION,
    PERSONA_VISUAL_PACK_SCHEMA_VERSION,
)
from .fingerprints import (
    canonical_json_bytes,
    canonical_payload_fingerprint,
    sha256_bytes,
    sha256_file,
)
from .models import PersonaVisualPackExportOptions, PersonaVisualPackExportResult

if TYPE_CHECKING:
    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB


class PersonaVisualPackExporter:
    """Assemble a persona visual pack into a portable backup archive."""

    def __init__(
        self,
        *,
        db: CharactersRAGDB,
        user_id: str,
        staging_root: Path,
    ) -> None:
        self.db = db
        self.user_id = str(user_id)
        self.staging_root = Path(staging_root)

    def export_pack(
        self,
        *,
        persona_id: str,
        pack_id: str,
        options: PersonaVisualPackExportOptions,
        progress: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> PersonaVisualPackExportResult:
        pack = self.db.get_persona_visual_pack(
            pack_id=pack_id,
            persona_id=persona_id,
            user_id=self.user_id,
        )
        if not pack:
            raise ValueError("pack_not_found")

        self._progress(progress, "collecting_metadata", {"pack_id": pack_id})
        assets = self.db.list_persona_visual_assets(
            pack_id=pack_id,
            persona_id=persona_id,
            user_id=self.user_id,
        )
        warnings: list[str] = []
        archive_path = self._archive_path(pack)
        payload_fingerprint = ""
        try:
            with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                exported_assets, asset_checksums, asset_fingerprints = self._export_assets(
                    assets,
                    options=options,
                    warnings=warnings,
                    archive=archive,
                )
                sections: dict[str, Any] = {
                    "metadata/pack.json": {"pack": self._export_pack_row(pack)},
                    "metadata/assets.json": {"assets": exported_assets},
                }
                fingerprint_payload = {
                    "pack": _canonical_pack_for_fingerprint(sections["metadata/pack.json"]["pack"]),
                    "assets": asset_fingerprints,
                }
                payload_fingerprint = canonical_payload_fingerprint(fingerprint_payload)
                metadata_payloads = {
                    path: canonical_json_bytes(payload)
                    for path, payload in sorted(sections.items())
                }
                checksums = {
                    path: sha256_bytes(content)
                    for path, content in metadata_payloads.items()
                }
                checksums.update(asset_checksums)
                manifest = self._archive_manifest(
                    pack=pack,
                    assets=exported_assets,
                    options=options,
                    warnings=warnings,
                    checksums=checksums,
                    canonical_fingerprint=payload_fingerprint,
                )
                manifest_payload = canonical_json_bytes(manifest)
                checksums[MANIFEST_PATH] = sha256_bytes(manifest_payload)
                checksums_payload = canonical_json_bytes(dict(sorted(checksums.items())))

                self._progress(progress, "writing_archive", {"asset_count": len(asset_checksums)})
                archive_payloads = {
                    MANIFEST_PATH: manifest_payload,
                    **metadata_payloads,
                    CHECKSUMS_PATH: checksums_payload,
                    "README.md": self._readme_payload(pack),
                    "signatures/README.md": b"Signatures are reserved for future persona visual pack versions.\n",
                }
                _write_payloads_to_archive(archive, archive_payloads)
        except Exception:
            archive_path.unlink(missing_ok=True)
            raise

        validate_archive_members(archive_path)
        archive_hash = sha256_file(archive_path)
        file_size_bytes = archive_path.stat().st_size
        self._progress(
            progress,
            "completed",
            {"archive_sha256": archive_hash, "file_size_bytes": file_size_bytes},
        )
        return PersonaVisualPackExportResult(
            archive_path=archive_path,
            archive_sha256=archive_hash,
            canonical_payload_fingerprint=payload_fingerprint,
            file_size_bytes=file_size_bytes,
            warnings=warnings,
        )

    def _export_assets(
        self,
        assets: list[dict[str, Any]],
        *,
        options: PersonaVisualPackExportOptions,
        warnings: list[str],
        archive: zipfile.ZipFile,
    ) -> tuple[list[dict[str, Any]], dict[str, str], list[dict[str, Any]]]:
        exported_assets: list[dict[str, Any]] = []
        asset_checksums: dict[str, str] = {}
        asset_fingerprints: list[dict[str, Any]] = []

        for asset in assets:
            exported = self._export_asset_row(asset)
            asset_id = str(asset["id"])
            asset_path = self._asset_storage_path(asset)
            if asset_path.is_file():
                asset_bytes = asset_path.read_bytes()
                asset_sha256 = sha256_bytes(asset_bytes)
                expected_sha256 = str(asset.get("checksum_sha256") or "")
                if expected_sha256 and asset_sha256 != expected_sha256:
                    raise ValueError(f"asset_checksum_mismatch:asset:{asset_id}")
                archive_path = self._asset_archive_path(asset)
                exported["asset_bytes_status"] = ASSET_BYTES_STATUS_PRESENT
                exported["asset_path"] = archive_path
                exported["asset_sha256"] = asset_sha256
                exported["asset_size_bytes"] = len(asset_bytes)
                archive.writestr(archive_path, asset_bytes)
                asset_checksums[archive_path] = asset_sha256
                asset_fingerprints.append(
                    {
                        "source_asset_id": asset_id,
                        "asset_role": asset.get("asset_role"),
                        "checksum": asset_sha256,
                    }
                )
            else:
                exported["asset_bytes_status"] = ASSET_BYTES_STATUS_MISSING
                warnings.append(f"missing_asset_bytes:asset:{asset_id}")
                if options.strict:
                    raise ValueError(f"missing_asset_bytes:asset:{asset_id}")
                asset_fingerprints.append(
                    {
                        "source_asset_id": asset_id,
                        "asset_role": asset.get("asset_role"),
                        "checksum": asset.get("checksum_sha256"),
                        "missing": True,
                    }
                )
            exported_assets.append(exported)

        return exported_assets, asset_checksums, asset_fingerprints

    def _archive_manifest(
        self,
        *,
        pack: Mapping[str, Any],
        assets: list[dict[str, Any]],
        options: PersonaVisualPackExportOptions,
        warnings: list[str],
        checksums: dict[str, str],
        canonical_fingerprint: str,
    ) -> dict[str, Any]:
        sections = [
            {"path": path, "sha256": checksum}
            for path, checksum in sorted(checksums.items())
            if path.startswith("metadata/") or path.startswith("assets/")
        ]
        assets_with_bytes = sum(
            1 for asset in assets if asset.get("asset_bytes_status") == ASSET_BYTES_STATUS_PRESENT
        )
        return {
            "schema_version": PERSONA_VISUAL_PACK_SCHEMA_VERSION,
            "exported_by": {"app": "tldw_server"},
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "archive_profile": "backup",
            "pack_title": pack["title"],
            "renderer_type": pack["renderer_type"],
            "source_pack_fingerprint": canonical_payload_fingerprint(
                {"pack": _canonical_pack_for_fingerprint(self._export_pack_row(pack))}
            ),
            "canonical_payload_fingerprint": canonical_fingerprint,
            "counts": {
                "assets": len(assets),
                "assets_with_bytes": assets_with_bytes,
                "missing_assets": len(assets) - assets_with_bytes,
            },
            "include_images": True,
            "provenance_mode": (
                "full" if options.include_full_provenance else "redacted"
            ),
            "trust_hints": {
                "source_owner_user_id": self.user_id,
                "source_persona_id": pack["persona_id"],
                "source_pack_id": pack["id"],
            },
            "encryption": {"encrypted": False, "scheme": None},
            "sections": sections,
            "warnings": list(warnings),
        }

    def _asset_storage_path(self, asset: Mapping[str, Any]) -> Path:
        storage_key = str(asset.get("storage_key") or "")
        prefix = f"{VISUAL_STORAGE_PREFIX}/"
        relative_key = storage_key[len(prefix):] if storage_key.startswith(prefix) else storage_key
        relative_path = _safe_relative_storage_path(relative_key)
        base = DatabasePaths.get_user_persona_visuals_dir(self.user_id).resolve(strict=False)
        target_path = (base / Path(*relative_path.parts)).resolve(strict=False)
        if not target_path.is_relative_to(base):
            raise ValueError("invalid_storage_path")
        return target_path

    def _asset_archive_path(self, asset: Mapping[str, Any]) -> str:
        source_asset_id = _safe_archive_component(str(asset.get("id") or "asset"))
        extension = _asset_extension(asset)
        return f"assets/persona_visuals/{source_asset_id}{extension}"

    def _archive_path(self, pack: Mapping[str, Any]) -> Path:
        self.staging_root.mkdir(parents=True, exist_ok=True)
        safe_title = "".join(
            char.lower() if char.isalnum() else "-"
            for char in str(pack["title"]).strip()
        ).strip("-")
        safe_title = safe_title[:64] or "persona-visual-pack"
        return self.staging_root / (
            f"{safe_title}-{uuid.uuid4().hex[:12]}{PERSONA_VISUAL_PACK_EXTENSION}"
        )

    def _export_pack_row(self, row: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "source_pack_id": row["id"],
            "source_persona_id": row["persona_id"],
            "title": row["title"],
            "renderer_type": row["renderer_type"],
            "status": row["status"],
            "manifest_version": row["manifest_version"],
            "visual_manifest": row.get("manifest") if isinstance(row.get("manifest"), dict) else {},
            "parent_pack_id": row.get("parent_pack_id"),
            "revision_number": row.get("revision_number"),
            "provenance": row.get("provenance"),
            "active_at": row.get("active_at"),
            "created_at": row.get("created_at"),
            "last_modified": row.get("last_modified"),
            "version": row.get("version"),
        }

    def _export_asset_row(self, row: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "source_asset_id": row["id"],
            "source_pack_id": row["pack_id"],
            "source_persona_id": row["persona_id"],
            "asset_role": row["asset_role"],
            "storage_key": row.get("storage_key"),
            "original_filename": row.get("original_filename"),
            "mime_type": row.get("mime_type"),
            "byte_size": row.get("byte_size"),
            "checksum_sha256": row.get("checksum_sha256"),
            "width": row.get("width"),
            "height": row.get("height"),
            "duration_ms": row.get("duration_ms"),
            "provenance": row.get("provenance"),
            "created_at": row.get("created_at"),
            "last_modified": row.get("last_modified"),
            "version": row.get("version"),
        }

    def _readme_payload(self, pack: Mapping[str, Any]) -> bytes:
        title = str(pack.get("title") or "Persona Visual Pack")
        return (
            f"# {title}\n\n"
            "This archive contains a tldw persona visual pack export.\n"
        ).encode("utf-8")

    def _progress(
        self,
        progress: Callable[[str, dict[str, Any]], None] | None,
        stage: str,
        payload: dict[str, Any],
    ) -> None:
        if progress is not None:
            progress(stage, payload)


def _write_payloads_to_archive(
    archive: zipfile.ZipFile,
    payloads: Mapping[str, bytes],
) -> None:
    for path, payload in sorted(payloads.items()):
        archive.writestr(path, payload)


def _safe_relative_storage_path(relative_key: str) -> PurePosixPath:
    if not relative_key or "\\" in relative_key:
        raise ValueError("invalid_storage_path")
    if relative_key.startswith("/"):
        raise ValueError("invalid_storage_path")
    path = PurePosixPath(relative_key)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("invalid_storage_path")
    return path


def _safe_archive_component(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"-", "_"} else "-" for char in value)
    return safe.strip("-_") or "asset"


def _asset_extension(asset: Mapping[str, Any]) -> str:
    filename = str(asset.get("original_filename") or asset.get("storage_key") or "")
    suffix = Path(filename).suffix.lower()
    if suffix in {".png", ".jpg", ".jpeg", ".webp", ".gif"}:
        return ".jpg" if suffix == ".jpeg" else suffix
    mime_type = str(asset.get("mime_type") or "").lower()
    return {
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/webp": ".webp",
        "image/gif": ".gif",
    }.get(mime_type, ".bin")


def _canonical_pack_for_fingerprint(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "title": row.get("title"),
        "renderer_type": row.get("renderer_type"),
        "manifest_version": row.get("manifest_version"),
        "visual_manifest": row.get("visual_manifest"),
        "revision_number": row.get("revision_number"),
        "provenance": row.get("provenance"),
    }
