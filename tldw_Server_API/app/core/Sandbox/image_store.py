from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class CloneItem:
    source_path: str
    target_path: str
    mode: str


@dataclass(slots=True)
class TemplateRecord:
    template_id: str
    runtime: str
    template_name: str
    disk_paths: list[str] = field(default_factory=list)


@dataclass(slots=True)
class RunCloneManifest:
    template_id: str
    run_id: str
    clone_items: list[CloneItem] = field(default_factory=list)


class SandboxImageStore:
    """Manifest-oriented VM template store.

    The initial implementation is intentionally lightweight: it records template
    metadata and returns deterministic clone manifests without performing APFS
    cloning yet. It is not the authoritative source of runnable-template truth;
    the macOS helper owns validation for templates that can actually boot.
    """

    def __init__(self, root_path: str | Path) -> None:
        self.root_path = Path(root_path)
        self._templates: dict[str, TemplateRecord] = {}

    def register_template(
        self,
        *,
        runtime: str,
        template_name: str,
        disk_paths: list[str],
    ) -> str:
        runtime_name = str(runtime).strip()
        normalized_name = str(template_name).strip()
        template_id = f"{runtime_name}:{normalized_name}"
        self._templates[template_id] = TemplateRecord(
            template_id=template_id,
            runtime=runtime_name,
            template_name=normalized_name,
            disk_paths=[str(path) for path in disk_paths],
        )
        return template_id

    def prepare_run_clone(self, *, template_id: str, run_id: str) -> RunCloneManifest:
        template = self._templates[template_id]
        run_root = self.root_path / "runs" / str(run_id)
        clone_items = [
            CloneItem(
                source_path=str(source_path),
                target_path=str(run_root / Path(source_path).name),
                mode="clone",
            )
            for source_path in template.disk_paths
        ]
        return RunCloneManifest(
            template_id=template.template_id,
            run_id=str(run_id),
            clone_items=clone_items,
        )
