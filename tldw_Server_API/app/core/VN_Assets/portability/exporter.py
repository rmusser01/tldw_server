"""Backup-grade VN asset pack archive assembler."""

from __future__ import annotations

import base64
import json
import uuid
import zipfile
from collections.abc import Awaitable, Callable, Mapping
from datetime import datetime, timezone
from inspect import isawaitable
from pathlib import Path
from typing import Any

from tldw_Server_API.app.core.DB_Management.VNAssetPacks_DB import VNAssetPacksRepository
from tldw_Server_API.app.core.VN_Assets.manifest import build_manifest
from tldw_Server_API.app.core.VN_Assets.models import VNAssetItem, VNAssetPack, VNAssetSlot
from tldw_Server_API.app.core.VN_Assets.portability.archive import validate_archive_members
from tldw_Server_API.app.core.VN_Assets.portability.constants import (
    ASSET_BYTES_STATUS_MISSING,
    ASSET_BYTES_STATUS_PRESENT,
    CHECKSUMS_PATH,
    MANIFEST_PATH,
    VNPACK_EXTENSION,
    VNPACK_SCHEMA_VERSION,
)
from tldw_Server_API.app.core.VN_Assets.portability.fingerprints import (
    canonical_json_bytes,
    canonical_payload_fingerprint,
    sha256_bytes,
    sha256_file,
)
from tldw_Server_API.app.core.VN_Assets.portability.models import (
    VNPackExportOptions,
    VNPackExportResult,
)
from tldw_Server_API.app.core.VN_Assets.storage import generated_file_matches_vn_asset


class VNPackExporter:
    """Assemble a single VN asset pack into a portable backup archive."""

    def __init__(
        self,
        *,
        repo: VNAssetPacksRepository,
        owner_user_id: int,
        generated_files_repo: Any,
        read_generated_file_bytes: Callable[[dict[str, Any]], bytes | Awaitable[bytes]],
        staging_root: Path,
    ) -> None:
        self.repo = repo
        self.owner_user_id = int(owner_user_id)
        self.generated_files_repo = generated_files_repo
        self.read_generated_file_bytes = read_generated_file_bytes
        self.staging_root = Path(staging_root)

    async def export_pack(
        self,
        *,
        pack_id: int,
        options: VNPackExportOptions,
        progress: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> VNPackExportResult:
        pack = self.repo.get_pack(pack_id)
        if pack is None or int(pack["owner_user_id"]) != self.owner_user_id or bool(pack.get("deleted")):
            raise ValueError("pack_not_found")
        source_world_book_ids = _loads_json(pack.get("source_world_book_ids_json"), [])
        if options.include_world_book_payloads and source_world_book_ids:
            raise ValueError("world_book_payloads_unavailable")

        self._progress(progress, "collecting_metadata", {"pack_id": int(pack_id)})
        slots = sorted(self.repo.list_slots(pack_id), key=lambda row: int(row["id"]))
        slot_by_id = {int(slot["id"]): slot for slot in slots}
        items = sorted(self.repo.list_items(pack_id), key=lambda row: int(row["id"]))
        batches = sorted(self.repo.list_batches(pack_id), key=lambda row: int(row["id"]))

        warnings: list[str] = []
        self._progress(progress, "collecting_assets", {"item_count": len(items)})
        archive_path = self._archive_path(pack)
        try:
            with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                exported_items, asset_checksums, asset_fingerprints = await self._export_items(
                    items,
                    slot_by_id=slot_by_id,
                    options=options,
                    warnings=warnings,
                    archive=archive,
                )

                sections: dict[str, Any] = {
                    "metadata/pack.json": {"pack": self._export_pack_row(pack)},
                    "metadata/slots.json": {
                        "slots": [self._export_slot_row(slot, slot_by_id=slot_by_id) for slot in slots]
                    },
                    "metadata/items.json": {"items": exported_items},
                    "metadata/batches.json": {"batches": [self._export_batch_row(batch) for batch in batches]},
                    "metadata/provenance.json": {
                        "mode": "full" if options.include_full_provenance else "redacted",
                        "items": [
                            self._export_item_provenance(item, include_full=options.include_full_provenance)
                            for item in items
                        ],
                    },
                    "metadata/runtime_manifest.json": self._runtime_manifest(pack, slots, items),
                }
                if options.include_character_payload:
                    sections["metadata/character.json"] = {
                        "character": self._export_character_row(
                            self._require_character(int(pack["primary_character_id"]))
                        )
                    }
                if options.include_world_book_payloads:
                    sections["metadata/world_books.json"] = {
                        "world_books": [],
                        "source_world_book_ids": source_world_book_ids,
                    }

                fingerprint_payload = {
                    "pack": _canonical_pack_for_fingerprint(sections["metadata/pack.json"]["pack"]),
                    "slots": [
                        _canonical_slot_for_fingerprint(slot)
                        for slot in sections["metadata/slots.json"]["slots"]
                    ],
                    "items": [
                        _canonical_item_for_fingerprint(item)
                        for item in sections["metadata/items.json"]["items"]
                    ],
                    "batches": [
                        _canonical_batch_for_fingerprint(batch)
                        for batch in sections["metadata/batches.json"]["batches"]
                    ],
                    "provenance": [
                        _canonical_provenance_for_fingerprint(item)
                        for item in sections["metadata/provenance.json"]["items"]
                    ],
                    "assets": asset_fingerprints,
                }
                if "metadata/character.json" in sections:
                    fingerprint_payload["character"] = _canonical_character_for_fingerprint(
                        sections["metadata/character.json"]["character"]
                    )
                if "metadata/world_books.json" in sections:
                    fingerprint_payload["world_books"] = sections["metadata/world_books.json"]
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
                manifest = self._manifest(
                    pack=pack,
                    slots=slots,
                    items=exported_items,
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
                    "signatures/README.md": b"Signatures are reserved for future VN pack versions.\n",
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
            {
                "archive_sha256": archive_hash,
                "file_size_bytes": file_size_bytes,
            },
        )
        return VNPackExportResult(
            archive_path=archive_path,
            archive_sha256=archive_hash,
            canonical_payload_fingerprint=payload_fingerprint,
            file_size_bytes=file_size_bytes,
            warnings=warnings,
        )

    async def _export_items(
        self,
        items: list[dict[str, Any]],
        *,
        slot_by_id: Mapping[int, Mapping[str, Any]],
        options: VNPackExportOptions,
        warnings: list[str],
        archive: zipfile.ZipFile,
    ) -> tuple[list[dict[str, Any]], dict[str, str], list[dict[str, Any]]]:
        exported_items: list[dict[str, Any]] = []
        asset_checksums: dict[str, str] = {}
        asset_fingerprints: list[dict[str, Any]] = []

        for item in items:
            slot = _require_slot(slot_by_id, int(item["slot_id"]))
            exported = self._export_item_row(item, slot=slot)
            item_id = int(item["id"])
            asset_bytes: bytes | None = None
            file_record = await self._get_file_record(item)
            if file_record is not None and generated_file_matches_vn_asset(
                file_record,
                user_id=self.owner_user_id,
                item_id=item_id,
            ):
                try:
                    asset_bytes = await _maybe_await(self.read_generated_file_bytes(file_record))
                except (OSError, ValueError, RuntimeError) as exc:
                    warnings.append(f"missing_asset_bytes:item:{item_id}:{exc}")
            else:
                warnings.append(f"missing_asset_bytes:item:{item_id}")

            if asset_bytes:
                asset_path = self._asset_path(item, file_record, slot)
                asset_sha256 = sha256_bytes(asset_bytes)
                exported["asset_bytes_status"] = ASSET_BYTES_STATUS_PRESENT
                exported["asset_path"] = asset_path
                exported["asset_sha256"] = asset_sha256
                exported["asset_size_bytes"] = len(asset_bytes)
                archive.writestr(asset_path, asset_bytes)
                asset_checksums[asset_path] = asset_sha256
                asset_fingerprints.append(
                    {
                        "asset_type": slot["asset_type"],
                        "slot_key": slot["slot_key"],
                        "variant_index": item["variant_index"],
                        "depth_kind": item.get("depth_kind"),
                        "checksum": asset_sha256,
                    }
                )
            else:
                exported["asset_bytes_status"] = ASSET_BYTES_STATUS_MISSING
                if options.strict:
                    raise ValueError(f"missing_asset_bytes:item:{item_id}")
            exported_items.append(exported)

        return exported_items, asset_checksums, asset_fingerprints

    async def _get_file_record(self, item: Mapping[str, Any]) -> dict[str, Any] | None:
        raw_file_id = item.get("generated_file_id")
        if raw_file_id in (None, ""):
            return None
        getter = getattr(self.generated_files_repo, "get_file_by_id", None)
        if not callable(getter):
            raise ValueError("generated_files_repo_unavailable")
        record = await _maybe_await(getter(int(raw_file_id)))
        return dict(record) if record else None

    def _manifest(
        self,
        *,
        pack: Mapping[str, Any],
        slots: list[dict[str, Any]],
        items: list[dict[str, Any]],
        options: VNPackExportOptions,
        warnings: list[str],
        checksums: dict[str, str],
        canonical_fingerprint: str,
    ) -> dict[str, Any]:
        review_counts = _review_counts(items)
        section_entries = [
            {"path": path, "sha256": checksum}
            for path, checksum in sorted(checksums.items())
            if path.startswith("metadata/") or path.startswith("assets/")
        ]
        return {
            "schema_version": VNPACK_SCHEMA_VERSION,
            "exported_by": {"app": "tldw_server"},
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "archive_profile": "backup",
            "pack_title": pack["title"],
            "content_rating": pack["content_rating"],
            "source_pack_fingerprint": canonical_payload_fingerprint({"pack": self._export_pack_row(pack)}),
            "canonical_payload_fingerprint": canonical_fingerprint,
            "counts": {
                "slots": len(slots),
                "items": len(items),
                "approved": review_counts.get("approved", 0),
                "draft": review_counts.get("draft", 0),
                "rejected": review_counts.get("rejected", 0),
                "hidden": review_counts.get("hidden", 0),
                "assets_with_bytes": sum(
                    1 for item in items if item.get("asset_bytes_status") == ASSET_BYTES_STATUS_PRESENT
                ),
            },
            "include_images": True,
            "include_character": bool(options.include_character_payload),
            "include_world_books": bool(options.include_world_book_payloads),
            "provenance_mode": "full" if options.include_full_provenance else "redacted",
            "trust_hints": {
                "source_owner_user_id": self.owner_user_id,
                "source_pack_id": int(pack["id"]),
            },
            "encryption": {"encrypted": False, "scheme": None},
            "sections": section_entries,
            "warnings": list(warnings),
        }

    def _archive_path(self, pack: Mapping[str, Any]) -> Path:
        self.staging_root.mkdir(parents=True, exist_ok=True)
        safe_title = "".join(
            char.lower() if char.isalnum() else "-"
            for char in str(pack["title"]).strip()
        ).strip("-")
        safe_title = safe_title[:64] or "vn-pack"
        return self.staging_root / f"{safe_title}-{uuid.uuid4().hex[:12]}{VNPACK_EXTENSION}"

    def _export_pack_row(self, row: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "source_pack_id": int(row["id"]),
            "title": row["title"],
            "description": row.get("description"),
            "status": row["status"],
            "content_rating": row["content_rating"],
            "primary_character_id": row["primary_character_id"],
            "source_world_book_ids": _loads_json(row.get("source_world_book_ids_json"), []),
            "scenario_notes": row.get("scenario_notes"),
            "style_prompt": row.get("style_prompt"),
            "negative_prompt": row.get("negative_prompt"),
            "default_backend": row.get("default_backend"),
            "default_model": row.get("default_model"),
            "default_dimensions": _loads_json(row.get("default_dimensions_json"), None),
            "style_lock": _loads_json(row.get("style_lock_json"), None),
            "generation_budget": _loads_json(row.get("generation_budget_json"), None),
            "created_at": row.get("created_at"),
            "updated_at": row.get("updated_at"),
            "version": row.get("version"),
        }

    def _export_slot_row(
        self,
        row: Mapping[str, Any],
        *,
        slot_by_id: Mapping[int, Mapping[str, Any]],
    ) -> dict[str, Any]:
        depends_on_slot_id = row.get("depends_on_slot_id")
        depends_on_slot_key = None
        if depends_on_slot_id is not None:
            depends_on_slot_key = _require_slot(slot_by_id, int(depends_on_slot_id))["slot_key"]
        return {
            "source_slot_id": int(row["id"]),
            "source_pack_id": int(row["pack_id"]),
            "asset_type": row["asset_type"],
            "slot_key": row["slot_key"],
            "labels": _loads_json(row.get("labels_json"), {}),
            "prompt_template": row.get("prompt_template"),
            "negative_prompt_template": row.get("negative_prompt_template"),
            "variant_count": row["variant_count"],
            "width": row.get("width"),
            "height": row.get("height"),
            "backend_override": row.get("backend_override"),
            "model_override": row.get("model_override"),
            "seed_policy": _loads_json(row.get("seed_policy_json"), None),
            "requires_review": bool(row["requires_review"]),
            "required_for_runtime": bool(row["required_for_runtime"]),
            "depends_on_slot_id": depends_on_slot_id,
            "depends_on_slot_key": depends_on_slot_key,
            "status": row["status"],
            "last_error": row.get("last_error"),
            "created_at": row.get("created_at"),
            "updated_at": row.get("updated_at"),
        }

    def _export_item_row(
        self,
        row: Mapping[str, Any],
        *,
        slot: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            "source_item_id": int(row["id"]),
            "source_pack_id": int(row["pack_id"]),
            "source_slot_id": int(row["slot_id"]),
            "asset_type": slot["asset_type"],
            "slot_key": slot["slot_key"],
            "variant_index": row["variant_index"],
            "file_artifact_id": row.get("file_artifact_id"),
            "source_generated_file_id": row.get("generated_file_id"),
            "mime_type": row.get("mime_type"),
            "width": row.get("width"),
            "height": row.get("height"),
            "bytes": row.get("bytes"),
            "review_status": row["review_status"],
            "preferred": bool(row["preferred"]),
            "source": row["source"],
            "generation_job_id": row.get("generation_job_id"),
            "depth_kind": row.get("depth_kind"),
            "parent_item_id": row.get("parent_item_id"),
            "has_alpha": None if row.get("has_alpha") is None else bool(row["has_alpha"]),
            "crop_box": _loads_json(row.get("crop_box_json"), None),
            "anchor": _loads_json(row.get("anchor_json"), None),
            "scale_hint": row.get("scale_hint"),
            "trim_status": row.get("trim_status"),
            "quality_flags": _loads_json(row.get("quality_flags_json"), []),
            "created_at": row.get("created_at"),
            "updated_at": row.get("updated_at"),
        }

    def _export_batch_row(self, row: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "source_batch_id": int(row["id"]),
            "source_pack_id": int(row["pack_id"]),
            "job_batch_id": row.get("job_batch_id"),
            "requested_by_user_id": row.get("requested_by_user_id"),
            "status": row.get("status"),
            "total_slots": row.get("total_slots"),
            "total_variants": row.get("total_variants"),
            "planned_count": row.get("planned_count"),
            "enqueued_count": row.get("enqueued_count"),
            "completed_count": row.get("completed_count"),
            "failed_count": row.get("failed_count"),
            "cancelled_count": row.get("cancelled_count"),
            "started_at": row.get("started_at"),
            "completed_at": row.get("completed_at"),
            "options": _loads_json(row.get("options_json"), {}),
            "created_at": row.get("created_at"),
            "updated_at": row.get("updated_at"),
        }

    def _require_character(self, character_id: int) -> dict[str, Any]:
        character = self.repo.get_character(character_id)
        if character is None:
            raise ValueError("primary_character_not_found")
        return character

    def _export_character_row(self, row: Mapping[str, Any]) -> dict[str, Any]:
        exported: dict[str, Any] = {
            "source_character_id": int(row["id"]),
            "name": row["name"],
            "description": row.get("description"),
            "personality": row.get("personality"),
            "scenario": row.get("scenario"),
            "system_prompt": row.get("system_prompt"),
            "post_history_instructions": row.get("post_history_instructions"),
            "first_message": row.get("first_message"),
            "message_example": row.get("message_example"),
            "creator_notes": row.get("creator_notes"),
            "alternate_greetings": _loads_json(row.get("alternate_greetings"), row.get("alternate_greetings")),
            "tags": _loads_json(row.get("tags"), row.get("tags")),
            "creator": row.get("creator"),
            "character_version": row.get("character_version"),
            "extensions": _loads_json(row.get("extensions"), row.get("extensions")),
            "created_at": row.get("created_at"),
            "last_modified": row.get("last_modified"),
        }
        image = row.get("image")
        if isinstance(image, bytes) and image:
            exported["image_base64"] = base64.b64encode(image).decode("ascii")
            exported["image_sha256"] = sha256_bytes(image)
        return exported

    def _export_item_provenance(self, row: Mapping[str, Any], *, include_full: bool) -> dict[str, Any]:
        prompt_snapshot = _loads_json(row.get("source_prompt_snapshot_json"), {})
        context_snapshot = _loads_json(row.get("source_context_snapshot_json"), {})
        backend_metadata = _loads_json(row.get("backend_metadata_json"), {})
        prompt_text = _first_text(prompt_snapshot.get("prompt"))
        negative_prompt_text = _first_text(prompt_snapshot.get("negative_prompt"))
        exported = {
            "source_item_id": int(row["id"]),
            "generation_job_id": row.get("generation_job_id"),
            "backend": backend_metadata.get("backend"),
            "model": backend_metadata.get("model"),
            "seed": backend_metadata.get("seed"),
            "dimensions": {
                "width": row.get("width"),
                "height": row.get("height"),
            },
            "prompt": {
                "prompt_present": bool(prompt_text),
                "prompt_sha256": sha256_bytes(prompt_text.encode("utf-8")) if prompt_text else None,
                "negative_prompt_present": bool(negative_prompt_text),
                "negative_prompt_sha256": (
                    sha256_bytes(negative_prompt_text.encode("utf-8")) if negative_prompt_text else None
                ),
            },
            "source_context_sha256": (
                sha256_bytes(canonical_json_bytes(context_snapshot)) if context_snapshot else None
            ),
        }
        if include_full:
            exported["prompt_snapshot"] = _redact_secrets(prompt_snapshot)
            exported["source_context_snapshot"] = _redact_secrets(context_snapshot)
            exported["backend_metadata"] = _redact_secrets(backend_metadata)
        return exported

    def _runtime_manifest(
        self,
        pack: Mapping[str, Any],
        slots: list[dict[str, Any]],
        items: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return build_manifest(
            pack=VNAssetPack(
                id=int(pack["id"]),
                owner_user_id=int(pack["owner_user_id"]),
                title=str(pack["title"]),
                primary_character_id=int(pack["primary_character_id"]),
                description=pack.get("description"),
                status=str(pack["status"]),
                content_rating=str(pack["content_rating"]),
            ),
            slots=[_slot_model(slot) for slot in slots],
            items=[_item_model(item) for item in items],
        )

    def _asset_path(
        self,
        item: Mapping[str, Any],
        file_record: Mapping[str, Any] | None,
        slot: Mapping[str, Any],
    ) -> str:
        extension = _extension_from_record(item, file_record)
        raw_key = f"{slot['asset_type']}-{slot['slot_key']}-variant-{int(item['variant_index'])}"
        source_key = f"{_safe_asset_key(raw_key)}-{sha256_bytes(raw_key.encode('utf-8'))[:12]}"
        return f"assets/items/{source_key}.{extension}"

    def _readme_payload(self, pack: Mapping[str, Any]) -> bytes:
        return (
            f"# {pack['title']}\n\n"
            "This archive is a tldw_server VN asset pack backup bundle.\n"
        ).encode("utf-8")

    def _progress(
        self,
        progress: Callable[[str, dict[str, Any]], None] | None,
        stage: str,
        payload: dict[str, Any],
    ) -> None:
        if progress is not None:
            progress(stage, payload)


def _slot_model(row: Mapping[str, Any]) -> VNAssetSlot:
    return VNAssetSlot(
        id=int(row["id"]),
        pack_id=int(row["pack_id"]),
        asset_type=str(row["asset_type"]),
        slot_key=str(row["slot_key"]),
        labels=_loads_json(row.get("labels_json"), {}),
        prompt_template=row.get("prompt_template"),
        negative_prompt_template=row.get("negative_prompt_template"),
        variant_count=int(row["variant_count"]),
        width=row.get("width"),
        height=row.get("height"),
        requires_review=bool(row["requires_review"]),
        required_for_runtime=bool(row["required_for_runtime"]),
        depends_on_slot_id=row.get("depends_on_slot_id"),
        status=str(row["status"]),
        last_error=row.get("last_error"),
    )


def _item_model(row: Mapping[str, Any]) -> VNAssetItem:
    return VNAssetItem(
        id=int(row["id"]),
        pack_id=int(row["pack_id"]),
        slot_id=int(row["slot_id"]),
        variant_index=int(row["variant_index"]),
        review_status=str(row["review_status"]),
        generated_file_id=row.get("generated_file_id"),
        file_artifact_id=row.get("file_artifact_id"),
        storage_ref=row.get("storage_ref"),
        mime_type=row.get("mime_type"),
        width=row.get("width"),
        height=row.get("height"),
        preferred=bool(row["preferred"]),
        source=str(row["source"]),
        depth_kind=row.get("depth_kind"),
        parent_item_id=row.get("parent_item_id"),
        has_alpha=None if row.get("has_alpha") is None else bool(row["has_alpha"]),
        crop_box=_loads_json(row.get("crop_box_json"), None),
        anchor=_loads_json(row.get("anchor_json"), None),
        scale_hint=row.get("scale_hint"),
        trim_status=str(row.get("trim_status") or "unknown"),
        quality_flags=_loads_json(row.get("quality_flags_json"), []),
    )


def _write_payloads_to_archive(
    archive: zipfile.ZipFile,
    payloads: Mapping[str, bytes],
) -> None:
    for path in sorted(payloads):
        archive.writestr(path, payloads[path])


async def _maybe_await(value: Any) -> Any:
    if isawaitable(value):
        return await value
    return value


def _loads_json(value: Any, default: Any) -> Any:
    if value in (None, ""):
        return default
    if isinstance(value, Mapping):
        return dict(value)
    try:
        loaded = json.loads(str(value))
    except json.JSONDecodeError:
        return default
    return loaded


def _first_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _extension_from_record(
    item: Mapping[str, Any],
    file_record: Mapping[str, Any] | None,
) -> str:
    mime_type = str(
        (file_record or {}).get("mime_type")
        or item.get("mime_type")
        or "application/octet-stream"
    ).lower()
    if mime_type == "image/jpeg":
        return "jpg"
    if mime_type.startswith("image/"):
        extension = mime_type.split("/", 1)[1].split(";", 1)[0]
        if extension:
            return _safe_extension(extension)
    filename = str((file_record or {}).get("filename") or "")
    suffix = Path(filename).suffix.lstrip(".")
    return _safe_extension(suffix or "bin")


def _safe_extension(value: str) -> str:
    cleaned = "".join(char.lower() for char in str(value) if char.isalnum())
    return cleaned or "bin"


def _safe_asset_key(value: str) -> str:
    cleaned = "".join(
        char.lower() if char.isalnum() else "-"
        for char in str(value).strip()
    ).strip("-")
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    return cleaned[:160] or "asset"


def _require_slot(
    slot_by_id: Mapping[int, Mapping[str, Any]],
    slot_id: int,
) -> Mapping[str, Any]:
    slot = slot_by_id.get(int(slot_id))
    if slot is None:
        raise ValueError(f"slot_not_found:{slot_id}")
    return slot


def _review_counts(items: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        status = str(item.get("review_status") or "")
        counts[status] = counts.get(status, 0) + 1
    return counts


def _canonical_pack_for_fingerprint(pack: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: pack.get(key)
        for key in (
            "title",
            "description",
            "status",
            "content_rating",
            "scenario_notes",
            "style_prompt",
            "negative_prompt",
            "default_backend",
            "default_model",
            "default_dimensions",
            "style_lock",
            "generation_budget",
        )
    }


def _canonical_slot_for_fingerprint(slot: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: slot.get(key)
        for key in (
            "asset_type",
            "slot_key",
            "labels",
            "prompt_template",
            "negative_prompt_template",
            "variant_count",
            "width",
            "height",
            "backend_override",
            "model_override",
            "seed_policy",
            "requires_review",
            "required_for_runtime",
            "depends_on_slot_key",
            "status",
            "last_error",
        )
    }


def _canonical_item_for_fingerprint(item: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: item.get(key)
        for key in (
            "variant_index",
            "asset_type",
            "slot_key",
            "mime_type",
            "width",
            "height",
            "bytes",
            "review_status",
            "preferred",
            "source",
            "depth_kind",
            "has_alpha",
            "crop_box",
            "anchor",
            "scale_hint",
            "trim_status",
            "quality_flags",
            "asset_bytes_status",
            "asset_sha256",
            "asset_size_bytes",
        )
    }


def _canonical_batch_for_fingerprint(batch: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: batch.get(key)
        for key in (
            "status",
            "total_slots",
            "total_variants",
            "planned_count",
            "enqueued_count",
            "completed_count",
            "failed_count",
            "cancelled_count",
            "options",
        )
    }


def _canonical_provenance_for_fingerprint(item: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: item.get(key)
        for key in (
            "backend",
            "model",
            "seed",
            "dimensions",
            "prompt",
        )
    }


def _canonical_character_for_fingerprint(character: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: character.get(key)
        for key in (
            "name",
            "description",
            "personality",
            "scenario",
            "system_prompt",
            "post_history_instructions",
            "first_message",
            "message_example",
            "creator_notes",
            "alternate_greetings",
            "tags",
            "creator",
            "character_version",
            "extensions",
            "image_sha256",
        )
    }


def _redact_secrets(value: Any) -> Any:
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            lowered = str(key).lower()
            if any(token in lowered for token in ("api_key", "token", "secret", "password")):
                result[str(key)] = "[redacted]"
            else:
                result[str(key)] = _redact_secrets(item)
        return result
    if isinstance(value, list):
        return [_redact_secrets(item) for item in value]
    return value
