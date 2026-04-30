"""Durable local image-store metadata for sandbox VM templates."""

from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MANIFEST_SCHEMA_VERSION = 1


class ImageStoreError(RuntimeError):
    """Base exception for sandbox image-store failures."""


class ImageStoreValidationError(ImageStoreError):
    """Raised when a template or bundle cannot be registered safely."""


@dataclass(slots=True)
class CloneItem:
    """A planned per-run clone or copy from a template artifact."""

    source_path: str
    target_path: str
    mode: str


@dataclass(slots=True)
class TemplateArtifact:
    """Hash-addressed metadata for a file that belongs to a template."""

    name: str
    path: str
    size_bytes: int
    sha256: str


@dataclass(slots=True)
class RunCloneManifest:
    """Deterministic clone plan for a sandbox run based on a registered template."""

    template_id: str
    run_id: str
    clone_items: list[CloneItem] = field(default_factory=list)


@dataclass(slots=True)
class TemplateRecord:
    """Persisted record for a VM template or canonical bundle."""

    template_id: str
    runtime: str
    template_name: str
    disk_paths: list[str] = field(default_factory=list)
    source_path: str | None = None
    labels: dict[str, str] = field(default_factory=dict)
    artifacts: list[TemplateArtifact] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)
    registered_at: str | None = None
    manifest_path: str | None = None


@dataclass(slots=True)
class GarbageCollectionCandidate:
    """Dry-run candidate for image-store cleanup."""

    run_id: str
    path: str
    size_bytes: int
    reason: str
    template_id: str | None = None


@dataclass(slots=True)
class GarbageCollectionPlan:
    """Dry-run image-store cleanup plan."""

    run_candidates: list[GarbageCollectionCandidate] = field(default_factory=list)


class SandboxImageStore:
    """Filesystem-backed VM template manifest store.

    The store owns local inventory, provenance, and deterministic clone-manifest
    planning. It does not prove that a template can boot; the macOS helper stays
    the runtime authority for bootability.
    """

    def __init__(self, root_path: str | Path) -> None:
        self.root_path = Path(root_path)
        self._templates: dict[str, TemplateRecord] = {}
        self._run_clone_manifests: dict[str, RunCloneManifest] = {}
        self.root_path.mkdir(parents=True, exist_ok=True)
        self._load_templates()
        self._load_run_clone_manifests()

    def register_template(
        self,
        *,
        runtime: str,
        template_name: str,
        disk_paths: list[str],
        source_path: str | Path | None = None,
        labels: dict[str, str] | None = None,
        provenance: dict[str, Any] | None = None,
        allow_existing: bool = False,
    ) -> str:
        """Register artifact paths as a durable template manifest.

        The method validates artifact existence, computes artifact size and
        SHA-256 metadata, writes `<root>/templates/<runtime>/<name>/manifest.json`,
        and returns the stable `runtime:name` template id.
        """

        runtime_name = self._normalize_manifest_segment(runtime, "runtime")
        normalized_name = self._normalize_manifest_segment(template_name, "template_name")
        template_id = f"{runtime_name}:{normalized_name}"
        manifest_path = self._manifest_path(runtime=runtime_name, template_name=normalized_name)

        if not allow_existing and (template_id in self._templates or manifest_path.exists()):
            raise ImageStoreValidationError(f"template_duplicate: {template_id}")

        artifacts = [self._artifact_from_path(Path(path).expanduser()) for path in disk_paths]
        if not artifacts:
            raise ImageStoreValidationError("template_artifacts_required")

        registered_at = datetime.now(timezone.utc).isoformat()
        record = TemplateRecord(
            template_id=template_id,
            runtime=runtime_name,
            template_name=normalized_name,
            disk_paths=[artifact.path for artifact in artifacts],
            source_path=str(Path(source_path).expanduser()) if source_path is not None else None,
            labels=dict(labels or {}),
            artifacts=artifacts,
            provenance=dict(provenance or {}),
            registered_at=registered_at,
            manifest_path=str(manifest_path),
        )
        self._write_manifest(record)
        self._templates[template_id] = record
        return template_id

    def register_bundle(
        self,
        *,
        runtime: str,
        template_name: str,
        bundle_path: str | Path,
        labels: dict[str, str] | None = None,
        allow_existing: bool = False,
    ) -> str:
        """Register a canonical bundle directory as a template.

        Bundle registration reads canonical artifact names from `manifest.json`
        when present, captures optional `build-info.json` provenance, and then
        delegates to `register_template()`.
        """

        bundle = Path(bundle_path).expanduser()
        if not bundle.is_dir():
            raise ImageStoreValidationError(f"bundle_missing: {bundle}")

        bundle_manifest = self._read_optional_json(bundle / "manifest.json")
        provenance = self._read_optional_json(bundle / "build-info.json")
        artifact_names = self._bundle_artifact_names(bundle_manifest, bundle=bundle)
        artifact_paths = [str(bundle / artifact_name) for artifact_name in artifact_names]

        return self.register_template(
            runtime=runtime,
            template_name=template_name,
            disk_paths=artifact_paths,
            source_path=bundle,
            labels=labels,
            provenance=provenance,
            allow_existing=allow_existing,
        )

    def get_template(self, template_id: str) -> TemplateRecord | None:
        """Return a registered template record by id, or `None` when absent."""

        return self._templates.get(str(template_id))

    def list_templates(self, *, runtime: str | None = None) -> list[TemplateRecord]:
        """List registered templates, optionally filtered by runtime."""

        runtime_name = str(runtime).strip() if runtime is not None else None
        records = self._templates.values()
        if runtime_name:
            records = [record for record in records if record.runtime == runtime_name]
        return sorted(records, key=lambda record: record.template_id)

    def prepare_run_clone(self, *, template_id: str, run_id: str) -> RunCloneManifest:
        """Build a deterministic per-run clone manifest for a template."""

        normalized_run_id = self._normalize_manifest_segment(run_id, "run_id")
        template = self._templates[template_id]
        run_root = self.root_path / "runs" / normalized_run_id
        manifest = RunCloneManifest(
            template_id=template.template_id,
            run_id=normalized_run_id,
            clone_items=[
                CloneItem(
                    source_path=str(source_path),
                    target_path=str(run_root / Path(source_path).name),
                    mode="clone",
                )
                for source_path in template.disk_paths
            ],
        )
        self._write_run_clone_manifest(manifest)
        self._run_clone_manifests[normalized_run_id] = manifest
        return manifest

    def get_run_clone_manifest(self, run_id: str) -> RunCloneManifest | None:
        """Return a persisted run clone manifest by run id, or `None` when absent."""

        return self._run_clone_manifests.get(str(run_id))

    def list_run_clone_manifests(self) -> list[RunCloneManifest]:
        """List persisted run clone manifests in deterministic run-id order."""

        return sorted(
            self._run_clone_manifests.values(),
            key=lambda manifest: manifest.run_id,
        )

    def plan_garbage_collection(self, *, active_run_ids: set[str] | None = None) -> GarbageCollectionPlan:
        """Return inactive run directories that could be deleted by a later GC step.

        This method is intentionally dry-run only. It never removes files.
        """

        active_ids = set(active_run_ids or set())
        runs_root = self.root_path / "runs"
        if not runs_root.exists():
            return GarbageCollectionPlan()

        candidates = []
        for run_path in sorted(path for path in runs_root.iterdir() if path.is_dir()):
            if run_path.name in active_ids:
                continue
            manifest = self._run_clone_manifests.get(run_path.name)
            reason = "inactive_run"
            if manifest is None:
                reason = "legacy_run_directory"
            else:
                non_manifest_children = [
                    child.name for child in run_path.iterdir() if child.name != "manifest.json"
                ]
                if not non_manifest_children:
                    reason = "planning_only_run_manifest"
            candidates.append(
                GarbageCollectionCandidate(
                    run_id=run_path.name,
                    path=str(run_path),
                    size_bytes=self._tree_size(run_path),
                    reason=reason,
                    template_id=(manifest.template_id if manifest is not None else None),
                )
            )
        return GarbageCollectionPlan(run_candidates=candidates)

    def _load_templates(self) -> None:
        templates_root = self.root_path / "templates"
        if not templates_root.exists():
            return

        for manifest_path in sorted(templates_root.glob("*/*/manifest.json")):
            record = self._read_manifest(manifest_path)
            self._validate_manifest_location(record, manifest_path)
            if record.template_id in self._templates:
                existing = self._templates[record.template_id].manifest_path
                raise ImageStoreValidationError(
                    f"template_duplicate_on_reload: {record.template_id}: {existing}: {manifest_path}"
                )
            self._templates[record.template_id] = record

    def _load_run_clone_manifests(self) -> None:
        runs_root = self.root_path / "runs"
        if not runs_root.exists():
            return

        for manifest_path in sorted(runs_root.glob("*/manifest.json")):
            manifest = self._read_run_clone_manifest(manifest_path)
            if manifest.run_id in self._run_clone_manifests:
                existing = self._run_manifest_path(manifest.run_id)
                raise ImageStoreValidationError(
                    f"run_manifest_duplicate_on_reload: {manifest.run_id}: {existing}: {manifest_path}"
                )
            self._run_clone_manifests[manifest.run_id] = manifest

    def _write_manifest(self, record: TemplateRecord) -> None:
        manifest_path = Path(record.manifest_path or self._manifest_path(runtime=record.runtime, template_name=record.template_name))
        payload = self._record_to_manifest(record)
        self._write_json_document(payload, manifest_path=manifest_path)

    def _write_run_clone_manifest(self, manifest: RunCloneManifest) -> None:
        self._write_json_document(
            self._run_clone_manifest_to_payload(manifest),
            manifest_path=self._run_manifest_path(manifest.run_id),
        )

    def _write_json_document(
        self,
        payload: dict[str, Any],
        *,
        manifest_path: Path,
    ) -> None:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                dir=manifest_path.parent,
                prefix=".manifest.json.",
                suffix=".tmp",
                delete=False,
                encoding="utf-8",
            ) as tmp_file:
                json.dump(payload, tmp_file, indent=2, sort_keys=True)
                tmp_file.write("\n")
                tmp_path = Path(tmp_file.name)
            tmp_path.replace(manifest_path)
        finally:
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)

    def _validate_manifest_location(self, record: TemplateRecord, manifest_path: Path) -> None:
        runtime_name = manifest_path.parent.parent.name
        template_name = manifest_path.parent.name
        expected_template_id = f"{record.runtime}:{record.template_name}"
        if (
            record.runtime != runtime_name
            or record.template_name != template_name
            or record.template_id != expected_template_id
        ):
            raise ImageStoreValidationError(
                f"manifest_path_mismatch: {manifest_path}: {record.template_id}"
            )

    def _read_manifest(self, manifest_path: Path) -> TemplateRecord:
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ImageStoreValidationError(f"manifest_invalid_json: {manifest_path}") from exc
        if not isinstance(payload, dict):
            raise ImageStoreValidationError(f"manifest_expected_object: {manifest_path}")

        if payload.get("schema_version") != MANIFEST_SCHEMA_VERSION:
            raise ImageStoreValidationError(f"manifest_unsupported_schema: {manifest_path}")

        required_fields = {"template_id", "runtime", "template_name", "disk_paths", "artifacts"}
        missing_fields = sorted(required_fields.difference(payload))
        if missing_fields:
            missing_text = ",".join(missing_fields)
            raise ImageStoreValidationError(f"manifest_missing_fields: {manifest_path}: {missing_text}")

        raw_artifacts = payload["artifacts"]
        if not isinstance(raw_artifacts, list):
            raise ImageStoreValidationError(f"manifest_artifacts_invalid: {manifest_path}")

        artifacts = [
            self._artifact_from_manifest_payload(artifact, manifest_path=manifest_path)
            for artifact in raw_artifacts
        ]
        artifact_paths = self._validated_artifact_paths(artifacts, manifest_path=manifest_path)
        raw_disk_paths = payload.get("disk_paths", [])
        if not isinstance(raw_disk_paths, list):
            raise ImageStoreValidationError(f"manifest_disk_paths_invalid: {manifest_path}")
        disk_paths = [str(Path(str(path)).expanduser()) for path in raw_disk_paths]
        if disk_paths != artifact_paths:
            raise ImageStoreValidationError(f"manifest_disk_paths_mismatch: {manifest_path}")
        return TemplateRecord(
            template_id=str(payload["template_id"]),
            runtime=str(payload["runtime"]),
            template_name=str(payload["template_name"]),
            disk_paths=artifact_paths,
            source_path=payload.get("source_path"),
            labels=dict(payload.get("labels", {})),
            artifacts=artifacts,
            provenance=dict(payload.get("provenance", {})),
            registered_at=payload.get("registered_at"),
            manifest_path=str(manifest_path),
        )

    def _validated_artifact_paths(
        self,
        artifacts: list[TemplateArtifact],
        *,
        manifest_path: Path,
    ) -> list[str]:
        artifact_paths = []
        for artifact in artifacts:
            artifact_path = Path(artifact.path).expanduser()
            if not artifact_path.is_file():
                raise ImageStoreValidationError(f"manifest_artifact_missing: {manifest_path}: {artifact_path}")
            artifact_paths.append(str(artifact_path))
        return artifact_paths

    def _artifact_from_manifest_payload(self, artifact: Any, *, manifest_path: Path) -> TemplateArtifact:
        if not isinstance(artifact, dict):
            raise ImageStoreValidationError(f"manifest_artifact_invalid: {manifest_path}")
        required_fields = {"name", "path", "size_bytes", "sha256"}
        missing_fields = sorted(required_fields.difference(artifact))
        if missing_fields:
            missing_text = ",".join(missing_fields)
            raise ImageStoreValidationError(f"manifest_artifact_missing_fields: {manifest_path}: {missing_text}")
        return TemplateArtifact(
            name=str(artifact["name"]),
            path=str(artifact["path"]),
            size_bytes=int(artifact["size_bytes"]),
            sha256=str(artifact["sha256"]),
        )

    def _record_to_manifest(self, record: TemplateRecord) -> dict[str, Any]:
        return {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "template_id": record.template_id,
            "runtime": record.runtime,
            "template_name": record.template_name,
            "source_path": record.source_path,
            "registered_at": record.registered_at,
            "disk_paths": list(record.disk_paths),
            "artifacts": [
                {
                    "name": artifact.name,
                    "path": artifact.path,
                    "size_bytes": artifact.size_bytes,
                    "sha256": artifact.sha256,
                }
                for artifact in record.artifacts
            ],
            "labels": dict(record.labels),
            "provenance": dict(record.provenance),
        }

    def _run_clone_manifest_to_payload(self, manifest: RunCloneManifest) -> dict[str, Any]:
        return {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "template_id": manifest.template_id,
            "run_id": manifest.run_id,
            "clone_items": [
                {
                    "source_path": item.source_path,
                    "target_path": item.target_path,
                    "mode": item.mode,
                }
                for item in manifest.clone_items
            ],
        }

    def _read_run_clone_manifest(self, manifest_path: Path) -> RunCloneManifest:
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ImageStoreValidationError(f"run_manifest_invalid_json: {manifest_path}") from exc
        if not isinstance(payload, dict):
            raise ImageStoreValidationError(f"run_manifest_expected_object: {manifest_path}")
        if payload.get("schema_version") != MANIFEST_SCHEMA_VERSION:
            raise ImageStoreValidationError(f"run_manifest_unsupported_schema: {manifest_path}")

        required_fields = {"template_id", "run_id", "clone_items"}
        missing_fields = sorted(required_fields.difference(payload))
        if missing_fields:
            missing_text = ",".join(missing_fields)
            raise ImageStoreValidationError(
                f"run_manifest_missing_fields: {manifest_path}: {missing_text}"
            )

        run_id = str(payload["run_id"])
        expected_path = self._run_manifest_path(run_id)
        if manifest_path != expected_path:
            raise ImageStoreValidationError(
                f"run_manifest_path_mismatch: {manifest_path}: {run_id}"
            )

        raw_clone_items = payload["clone_items"]
        if not isinstance(raw_clone_items, list):
            raise ImageStoreValidationError(f"run_manifest_clone_items_invalid: {manifest_path}")

        return RunCloneManifest(
            template_id=str(payload["template_id"]),
            run_id=run_id,
            clone_items=[
                self._clone_item_from_payload(item, manifest_path=manifest_path)
                for item in raw_clone_items
            ],
        )

    def _clone_item_from_payload(self, item: Any, *, manifest_path: Path) -> CloneItem:
        if not isinstance(item, dict):
            raise ImageStoreValidationError(f"run_manifest_clone_item_invalid: {manifest_path}")
        required_fields = {"source_path", "target_path", "mode"}
        missing_fields = sorted(required_fields.difference(item))
        if missing_fields:
            missing_text = ",".join(missing_fields)
            raise ImageStoreValidationError(
                f"run_manifest_clone_item_missing_fields: {manifest_path}: {missing_text}"
            )
        return CloneItem(
            source_path=str(item["source_path"]),
            target_path=str(item["target_path"]),
            mode=str(item["mode"]),
        )

    def _manifest_path(self, *, runtime: str, template_name: str) -> Path:
        return self.root_path / "templates" / runtime / template_name / "manifest.json"

    def _run_manifest_path(self, run_id: str) -> Path:
        return self.root_path / "runs" / run_id / "manifest.json"

    def _artifact_from_path(self, path: Path) -> TemplateArtifact:
        if not path.is_file():
            raise ImageStoreValidationError(f"template_artifact_missing: {path}")
        return TemplateArtifact(
            name=path.name,
            path=str(path),
            size_bytes=path.stat().st_size,
            sha256=self._sha256_file(path),
        )

    def _read_optional_json(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ImageStoreValidationError(f"json_invalid: {path}") from exc
        if not isinstance(payload, dict):
            raise ImageStoreValidationError(f"json_expected_object: {path}")
        return payload

    def _bundle_artifact_names(self, bundle_manifest: dict[str, Any], *, bundle: Path) -> list[str]:
        kernel_name = str(bundle_manifest.get("kernel", "kernel"))
        rootfs_name = str(bundle_manifest.get("rootfs", "rootfs.img"))
        initrd_name = bundle_manifest.get("initrd")
        if initrd_name is None and (bundle / "initrd").is_file():
            initrd_name = "initrd"
        artifact_names = [
            self._safe_bundle_artifact_name(kernel_name, field_name="kernel"),
            self._safe_bundle_artifact_name(rootfs_name, field_name="rootfs"),
        ]
        if initrd_name:
            artifact_names.append(self._safe_bundle_artifact_name(str(initrd_name), field_name="initrd"))
        return artifact_names

    def _safe_bundle_artifact_name(self, value: str, *, field_name: str) -> str:
        name = str(value).strip()
        if not name or name in {".", ".."} or Path(name).name != name:
            raise ImageStoreValidationError(f"bundle_artifact_name_invalid: {field_name}")
        return name

    def _normalize_manifest_segment(self, value: str, field_name: str) -> str:
        normalized = str(value).strip()
        if not normalized:
            raise ImageStoreValidationError(f"{field_name}_required")
        if normalized in {".", ".."} or "/" in normalized or "\\" in normalized:
            raise ImageStoreValidationError(f"{field_name}_invalid")
        return normalized

    def _sha256_file(self, path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _tree_size(self, path: Path) -> int:
        total = 0
        for item in path.rglob("*"):
            try:
                if item.is_file():
                    total += item.stat().st_size
            except FileNotFoundError:
                continue
        return total
