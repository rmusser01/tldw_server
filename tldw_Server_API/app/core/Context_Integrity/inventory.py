"""Filesystem inventory adapters for Context Integrity."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import errno
import os
from pathlib import Path
import stat

from tldw_Server_API.app.core.Context_Integrity.canonicalization import (
    canonical_filesystem_digest,
)
from tldw_Server_API.app.core.Context_Integrity.models import (
    ContextAssetDescriptor,
    ContextAssetSource,
    ContextIntegrityFinding,
)

_PROMPT_SUFFIXES = {".md", ".yaml", ".yml", ".json", ".txt"}
_SKILL_TEXT_SUFFIXES = {".md", ".txt", ".json", ".yaml", ".yml", ".py", ".sh"}


@dataclass(frozen=True, slots=True)
class InventoryResult:
    """Inventory assets plus non-fatal discovery findings."""

    assets: tuple[ContextAssetDescriptor, ...]
    findings: tuple[ContextIntegrityFinding, ...] = ()


def _verification_error(
    *,
    asset_id: str,
    source_type: ContextAssetSource,
    summary: str,
    path: Path,
    details: Mapping[str, str] | None = None,
) -> ContextIntegrityFinding:
    finding_details = {"path": str(path)}
    if details:
        finding_details.update(details)
    return ContextIntegrityFinding(
        asset_id=asset_id,
        state="verification_error",
        severity="error",
        summary=summary,
        remediation="Review the filesystem path and quarantine or restore the affected asset.",
        source_type=source_type,
        details=finding_details,
    )


def _path_resolves_within(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
    except (OSError, RuntimeError, ValueError):
        return False
    return True


def _symlink_finding(
    *,
    asset_id: str,
    source_type: ContextAssetSource,
    path: Path,
    root: Path | None,
) -> ContextIntegrityFinding:
    escaped = root is not None and not _path_resolves_within(path, root)
    summary = (
        "Symlinked context path escapes the inventory root and was skipped."
        if escaped
        else "Symlinked context path was skipped."
    )
    details = {"root": str(root)} if root is not None else None
    return _verification_error(
        asset_id=asset_id,
        source_type=source_type,
        summary=summary,
        path=path,
        details=details,
    )


def _validate_inventory_root(
    *,
    root: Path,
    asset_id: str,
    source_type: ContextAssetSource,
) -> tuple[bool, ContextIntegrityFinding | None]:
    try:
        root_stat = root.lstat()
    except FileNotFoundError:
        return False, None
    except OSError as exc:
        return (
            False,
            _verification_error(
                asset_id=asset_id,
                source_type=source_type,
                summary=f"Unable to inspect context inventory root: {exc.__class__.__name__}.",
                path=root,
                details={"error": str(exc)},
            ),
        )

    if stat.S_ISLNK(root_stat.st_mode):
        return (
            False,
            _verification_error(
                asset_id=asset_id,
                source_type=source_type,
                summary="Context inventory root is a symlink and was skipped.",
                path=root,
            ),
        )
    if not stat.S_ISDIR(root_stat.st_mode):
        return (
            False,
            _verification_error(
                asset_id=asset_id,
                source_type=source_type,
                summary="Context inventory root is not a directory and was skipped.",
                path=root,
            ),
        )
    return True, None


def _not_regular_error(path: Path) -> OSError:
    return OSError(errno.EINVAL, "Path is not a regular file", str(path))


def _path_changed_error(path: Path) -> OSError:
    return OSError(errno.ELOOP, "Path changed while opening", str(path))


def _open_no_follow(path: Path) -> int:
    flags = os.O_RDONLY
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is not None:
        return os.open(path, flags | nofollow)

    initial_stat = path.lstat()
    if not stat.S_ISREG(initial_stat.st_mode):
        raise _not_regular_error(path)

    fd = os.open(path, flags)
    opened_successfully = False
    try:
        opened_stat = os.fstat(fd)
        if (opened_stat.st_dev, opened_stat.st_ino) != (
            initial_stat.st_dev,
            initial_stat.st_ino,
        ):
            raise _path_changed_error(path)
        opened_successfully = True
        return fd
    finally:
        if not opened_successfully:
            os.close(fd)


def _read_no_follow_bytes(path: Path) -> bytes:
    fd = _open_no_follow(path)
    try:
        opened_stat = os.fstat(fd)
        if not stat.S_ISREG(opened_stat.st_mode):
            raise _not_regular_error(path)

        chunks: list[bytes] = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                return b"".join(chunks)
            chunks.append(chunk)
    finally:
        os.close(fd)


def _read_regular_file(
    *,
    path: Path,
    asset_id: str,
    source_type: ContextAssetSource,
    findings: list[ContextIntegrityFinding],
) -> bytes | None:
    try:
        return _read_no_follow_bytes(path)
    except OSError as exc:
        findings.append(
            _verification_error(
                asset_id=asset_id,
                source_type=source_type,
                summary=f"Unable to read context file: {exc.__class__.__name__}.",
                path=path,
                details={"error": str(exc)},
            )
        )
        return None


def _read_skill_file_map(
    *,
    skill_dir: Path,
    asset_id: str,
    findings: list[ContextIntegrityFinding],
) -> dict[str, bytes]:
    files: dict[str, bytes] = {}

    def _on_walk_error(exc: OSError) -> None:
        error_path = Path(exc.filename) if exc.filename else skill_dir
        findings.append(
            _verification_error(
                asset_id=asset_id,
                source_type="skill_file",
                summary=f"Unable to scan skill directory: {exc.__class__.__name__}.",
                path=error_path,
                details={"error": str(exc)},
            )
        )

    for current_root, dirnames, filenames in os.walk(
        skill_dir,
        topdown=True,
        onerror=_on_walk_error,
        followlinks=False,
    ):
        current_path = Path(current_root)
        for dirname in list(dirnames):
            child = current_path / dirname
            if child.is_symlink():
                findings.append(
                    _symlink_finding(
                        asset_id=asset_id,
                        source_type="skill_file",
                        path=child,
                        root=skill_dir,
                    )
                )
                dirnames.remove(dirname)

        for filename in filenames:
            path = current_path / filename
            if path.is_symlink():
                findings.append(
                    _symlink_finding(
                        asset_id=asset_id,
                        source_type="skill_file",
                        path=path,
                        root=skill_dir,
                    )
                )
                continue
            if path.suffix.lower() not in _SKILL_TEXT_SUFFIXES:
                continue
            try:
                relative = path.relative_to(skill_dir).as_posix()
            except ValueError:
                findings.append(
                    _verification_error(
                        asset_id=asset_id,
                        source_type="skill_file",
                        summary="Skill file path escaped the skill directory and was skipped.",
                        path=path,
                        details={"root": str(skill_dir)},
                    )
                )
                continue
            content = _read_regular_file(
                path=path,
                asset_id=asset_id,
                source_type="skill_file",
                findings=findings,
            )
            if content is not None:
                files[relative] = content

    return files


def inventory_user_skills_with_findings(*, user_id: int, skills_root: Path) -> InventoryResult:
    """Inventory per-user skill directories with non-fatal discovery findings."""
    root_ok, root_finding = _validate_inventory_root(
        root=skills_root,
        asset_id=f"skill:user:{user_id}",
        source_type="skill_file",
    )
    if not root_ok:
        return InventoryResult(
            assets=(),
            findings=(root_finding,) if root_finding is not None else (),
        )

    assets: list[ContextAssetDescriptor] = []
    findings: list[ContextIntegrityFinding] = []
    try:
        with os.scandir(skills_root) as iterator:
            entries = sorted(iterator, key=lambda entry: entry.name)
    except OSError as exc:
        return InventoryResult(
            assets=(),
            findings=(
                _verification_error(
                    asset_id=f"skill:user:{user_id}",
                    source_type="skill_file",
                    summary=f"Unable to scan user skills root: {exc.__class__.__name__}.",
                    path=skills_root,
                    details={"error": str(exc)},
                ),
            ),
        )

    for entry in entries:
        skill_dir = Path(entry.path)
        asset_id = f"skill:user:{user_id}/{entry.name}"
        try:
            if entry.is_symlink():
                findings.append(
                    _symlink_finding(
                        asset_id=asset_id,
                        source_type="skill_file",
                        path=skill_dir,
                        root=skills_root,
                    )
                )
                continue
            if not entry.is_dir(follow_symlinks=False):
                continue
        except OSError as exc:
            findings.append(
                _verification_error(
                    asset_id=asset_id,
                    source_type="skill_file",
                    summary=f"Unable to inspect skill directory: {exc.__class__.__name__}.",
                    path=skill_dir,
                    details={"error": str(exc)},
                )
            )
            continue

        skill_file = skill_dir / "SKILL.md"
        if skill_file.is_symlink():
            findings.append(
                _symlink_finding(
                    asset_id=asset_id,
                    source_type="skill_file",
                    path=skill_file,
                    root=skill_dir,
                )
            )
            continue
        if not skill_file.exists():
            continue

        files = _read_skill_file_map(
            skill_dir=skill_dir,
            asset_id=asset_id,
            findings=findings,
        )
        if "SKILL.md" not in files:
            continue

        metadata = {"skill_name": entry.name}
        digest = canonical_filesystem_digest(
            source_type="skill_file",
            asset_id=asset_id,
            files=files,
            metadata=metadata,
        )
        assets.append(
            ContextAssetDescriptor(
                asset_id=asset_id,
                source_type="skill_file",
                digest=digest,
                display_name=entry.name,
                executable=True,
                owner_scope=f"user:{user_id}",
                path=str(skill_dir),
                metadata=metadata,
            )
        )

    return InventoryResult(assets=tuple(assets), findings=tuple(findings))


def inventory_user_skills(user_id: int, skills_root: Path) -> list[ContextAssetDescriptor]:
    """Inventory per-user skill directories."""
    return list(inventory_user_skills_with_findings(user_id=user_id, skills_root=skills_root).assets)


def inventory_prompt_files_with_findings(*, prompts_dir: Path) -> InventoryResult:
    """Inventory config prompt files under a Prompts directory with findings."""
    root_ok, root_finding = _validate_inventory_root(
        root=prompts_dir,
        asset_id=f"prompt_file:{prompts_dir.name}",
        source_type="prompt_file",
    )
    if not root_ok:
        return InventoryResult(
            assets=(),
            findings=(root_finding,) if root_finding is not None else (),
        )

    assets: list[ContextAssetDescriptor] = []
    findings: list[ContextIntegrityFinding] = []
    try:
        entries = sorted(prompts_dir.iterdir(), key=lambda path: path.name)
    except OSError as exc:
        return InventoryResult(
            assets=(),
            findings=(
                _verification_error(
                    asset_id=f"prompt_file:{prompts_dir.name}",
                    source_type="prompt_file",
                    summary=f"Unable to scan prompt directory: {exc.__class__.__name__}.",
                    path=prompts_dir,
                    details={"error": str(exc)},
                ),
            ),
        )

    for path in entries:
        asset_id = f"prompt_file:{path.name}"
        if path.is_symlink():
            findings.append(
                _symlink_finding(
                    asset_id=asset_id,
                    source_type="prompt_file",
                    path=path,
                    root=prompts_dir,
                )
            )
            continue
        if path.is_dir():
            continue
        if path.suffix.lower() not in _PROMPT_SUFFIXES:
            continue

        prompt_bytes = _read_regular_file(
            path=path,
            asset_id=asset_id,
            source_type="prompt_file",
            findings=findings,
        )
        if prompt_bytes is None:
            continue

        metadata = {"path": path.name}
        digest = canonical_filesystem_digest(
            source_type="prompt_file",
            asset_id=asset_id,
            files={path.name: prompt_bytes},
            metadata=metadata,
        )
        assets.append(
            ContextAssetDescriptor(
                asset_id=asset_id,
                source_type="prompt_file",
                digest=digest,
                display_name=path.name,
                executable=False,
                owner_scope="system",
                path=str(path),
                metadata=metadata,
            )
        )

    return InventoryResult(assets=tuple(assets), findings=tuple(findings))


def inventory_prompt_files(prompts_dir: Path) -> list[ContextAssetDescriptor]:
    """Inventory config prompt files under a Prompts directory."""
    return list(inventory_prompt_files_with_findings(prompts_dir=prompts_dir).assets)


def inventory_env_prompt_overrides_with_findings(
    *,
    environ: Mapping[str, str] | None = None,
) -> InventoryResult:
    """Inventory configured prompt override files from TLDW_PROMPT_FILE_* vars."""
    source = environ if environ is not None else os.environ
    assets: list[ContextAssetDescriptor] = []
    findings: list[ContextIntegrityFinding] = []

    for env_name, raw_path in sorted(source.items()):
        if not env_name.startswith("TLDW_PROMPT_FILE_") or not raw_path.strip():
            continue

        try:
            path = Path(raw_path.strip()).expanduser()
        except RuntimeError as exc:
            path = Path(raw_path.strip())
            findings.append(
                _verification_error(
                    asset_id=f"prompt_file:env:{env_name}:{path.name}",
                    source_type="prompt_file",
                    summary=f"Unable to expand prompt override path: {exc.__class__.__name__}.",
                    path=path,
                    details={"error": str(exc)},
                )
            )
            continue

        source_label = f"env:{env_name}"
        asset_id = f"prompt_file:{source_label}:{path.name}"
        metadata = {"path": str(path), "source_label": source_label}
        if path.is_symlink():
            findings.append(
                _symlink_finding(
                    asset_id=asset_id,
                    source_type="prompt_file",
                    path=path,
                    root=None,
                )
            )
            continue

        prompt_bytes = _read_regular_file(
            path=path,
            asset_id=asset_id,
            source_type="prompt_file",
            findings=findings,
        )
        if prompt_bytes is None:
            continue

        digest = canonical_filesystem_digest(
            source_type="prompt_file",
            asset_id=asset_id,
            files={path.name: prompt_bytes},
            metadata=metadata,
        )
        assets.append(
            ContextAssetDescriptor(
                asset_id=asset_id,
                source_type="prompt_file",
                digest=digest,
                display_name=env_name,
                executable=False,
                owner_scope="system",
                path=str(path),
                metadata=metadata,
            )
        )

    return InventoryResult(assets=tuple(assets), findings=tuple(findings))


def inventory_env_prompt_overrides(
    environ: Mapping[str, str] | None = None,
) -> list[ContextAssetDescriptor]:
    """Inventory configured prompt override files from TLDW_PROMPT_FILE_* vars."""
    return list(inventory_env_prompt_overrides_with_findings(environ=environ).assets)
