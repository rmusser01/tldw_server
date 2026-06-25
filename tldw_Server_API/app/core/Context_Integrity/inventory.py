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


def _symlink_finding(
    *,
    asset_id: str,
    source_type: ContextAssetSource,
    path: Path,
    root: Path | None,
) -> ContextIntegrityFinding:
    details = {"root": str(root)} if root is not None else None
    return _verification_error(
        asset_id=asset_id,
        source_type=source_type,
        summary="Symlinked context path was skipped.",
        path=path,
        details=details,
    )


def _validate_inventory_root(
    *,
    root: Path,
    asset_id: str,
    source_type: ContextAssetSource,
) -> tuple[bool, ContextIntegrityFinding | None]:
    if root.is_absolute():
        current = Path(root.anchor)
        parts = root.parts[1:]
    else:
        current = Path(".")
        parts = root.parts

    if any(part == ".." for part in parts):
        return (
            False,
            _verification_error(
                asset_id=asset_id,
                source_type=source_type,
                summary="Context inventory root contains parent traversal and was skipped.",
                path=root,
            ),
        )

    for part in parts:
        if part in ("", "."):
            continue
        current = current / part
        try:
            component_stat = current.lstat()
        except FileNotFoundError:
            return False, None
        except OSError as exc:
            return (
                False,
                _verification_error(
                    asset_id=asset_id,
                    source_type=source_type,
                    summary=f"Unable to inspect context inventory root component: {exc.__class__.__name__}.",
                    path=current,
                    details={"root": str(root), "error": str(exc)},
                ),
            )
        if stat.S_ISLNK(component_stat.st_mode):
            return (
                False,
                _verification_error(
                    asset_id=asset_id,
                    source_type=source_type,
                    summary="Context inventory root contains a symlink component and was skipped.",
                    path=current,
                    details={"root": str(root)},
                ),
            )
    return True, None


def _not_regular_error(path: Path) -> OSError:
    return OSError(errno.EINVAL, "Path is not a regular file", str(path))


def _path_changed_error(path: Path) -> OSError:
    return OSError(errno.ELOOP, "Path changed while opening", str(path))


def _not_directory_error(path: Path) -> OSError:
    return OSError(errno.ENOTDIR, "Path is not a directory", str(path))


def _symlink_directory_component_error(path: Path) -> OSError:
    return OSError(errno.ELOOP, "Directory path component is a symlink", str(path))


def _unsupported_fd_error(feature: str) -> OSError:
    return OSError(errno.ENOTSUP, f"Required fd-relative filesystem support is unavailable: {feature}")


def _require_fd_traversal_support() -> None:
    if os.listdir not in getattr(os, "supports_fd", set()):
        raise _unsupported_fd_error("os.listdir(fd)")
    if os.open not in os.supports_dir_fd:
        raise _unsupported_fd_error("os.open(dir_fd=...)")
    if os.stat not in os.supports_dir_fd:
        raise _unsupported_fd_error("os.stat(dir_fd=...)")
    if os.stat not in os.supports_follow_symlinks:
        raise _unsupported_fd_error("os.stat(follow_symlinks=False)")
    if getattr(os, "O_NOFOLLOW", None) is None:
        raise _unsupported_fd_error("O_NOFOLLOW")
    if getattr(os, "O_DIRECTORY", None) is None:
        raise _unsupported_fd_error("O_DIRECTORY")


def _dir_open_flags() -> int:
    _require_fd_traversal_support()
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    cloexec = getattr(os, "O_CLOEXEC", None)
    if cloexec is not None:
        flags |= cloexec
    return flags


def _file_open_flags() -> int:
    _require_fd_traversal_support()
    flags = os.O_RDONLY | os.O_NOFOLLOW
    nonblock = getattr(os, "O_NONBLOCK", None)
    if nonblock is not None:
        flags |= nonblock
    cloexec = getattr(os, "O_CLOEXEC", None)
    if cloexec is not None:
        flags |= cloexec
    return flags


def _open_dir_no_follow_fd(path: Path) -> int:
    initial_stat = path.lstat()
    if not stat.S_ISDIR(initial_stat.st_mode):
        raise _not_directory_error(path)

    fd = os.open(path, _dir_open_flags())
    opened_successfully = False
    try:
        opened_stat = os.fstat(fd)
        if not stat.S_ISDIR(opened_stat.st_mode):
            raise _not_directory_error(path)
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


def _open_cwd_no_follow_fd() -> int:
    fd = os.open(".", _dir_open_flags())
    opened_successfully = False
    try:
        opened_stat = os.fstat(fd)
        if not stat.S_ISDIR(opened_stat.st_mode):
            raise _not_directory_error(Path("."))
        opened_successfully = True
        return fd
    finally:
        if not opened_successfully:
            os.close(fd)


def _stat_child_no_follow(*, dir_fd: int, name: str, path: Path) -> os.stat_result:
    _require_fd_traversal_support()
    try:
        return os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
    except TypeError as exc:
        raise _unsupported_fd_error("os.stat(dir_fd=..., follow_symlinks=False)") from exc


def _open_child_dir_no_follow_fd(dir_fd: int, name: str, path: Path) -> int:
    initial_stat = _stat_child_no_follow(dir_fd=dir_fd, name=name, path=path)
    if not stat.S_ISDIR(initial_stat.st_mode):
        raise _not_directory_error(path)

    fd = os.open(name, _dir_open_flags(), dir_fd=dir_fd)
    opened_successfully = False
    try:
        opened_stat = os.fstat(fd)
        if not stat.S_ISDIR(opened_stat.st_mode):
            raise _not_directory_error(path)
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


def _open_directory_path_no_follow_fd(path: Path) -> int:
    _require_fd_traversal_support()
    if path.is_absolute():
        fd = _open_dir_no_follow_fd(Path(path.anchor))
        parts = path.parts[1:]
        current_path = Path(path.anchor)
    else:
        fd = _open_cwd_no_follow_fd()
        parts = path.parts
        current_path = Path(".")

    opened_successfully = False
    try:
        for part in parts:
            if part in ("", "."):
                continue
            if part == "..":
                raise OSError(errno.EINVAL, "Parent path traversal is not supported", str(path))

            child_path = current_path / part
            child_stat = _stat_child_no_follow(dir_fd=fd, name=part, path=child_path)
            if stat.S_ISLNK(child_stat.st_mode):
                raise _symlink_directory_component_error(child_path)
            if not stat.S_ISDIR(child_stat.st_mode):
                raise _not_directory_error(child_path)

            child_fd = _open_child_dir_no_follow_fd(fd, part, child_path)
            old_fd = fd
            fd = child_fd
            os.close(old_fd)
            current_path = child_path

        opened_successfully = True
        return fd
    finally:
        if not opened_successfully:
            os.close(fd)


def _open_file_no_follow_fd(*, dir_fd: int, name: str, path: Path) -> int:
    initial_stat = _stat_child_no_follow(dir_fd=dir_fd, name=name, path=path)
    if not stat.S_ISREG(initial_stat.st_mode):
        raise _not_regular_error(path)

    fd = os.open(name, _file_open_flags(), dir_fd=dir_fd)
    opened_successfully = False
    try:
        opened_stat = os.fstat(fd)
        if not stat.S_ISREG(opened_stat.st_mode):
            raise _not_regular_error(path)
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


def _read_no_follow_bytes_fd(*, dir_fd: int, name: str, path: Path) -> bytes:
    fd = _open_file_no_follow_fd(dir_fd=dir_fd, name=name, path=path)
    try:
        chunks: list[bytes] = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                return b"".join(chunks)
            chunks.append(chunk)
    finally:
        os.close(fd)


def _read_regular_file_from_dir_fd(
    *,
    dir_fd: int,
    name: str,
    path: Path,
    asset_id: str,
    source_type: ContextAssetSource,
    findings: list[ContextIntegrityFinding],
) -> bytes | None:
    try:
        return _read_no_follow_bytes_fd(dir_fd=dir_fd, name=name, path=path)
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


def _read_skill_file_map_fd(
    *,
    dir_fd: int,
    skill_dir: Path,
    asset_id: str,
    findings: list[ContextIntegrityFinding],
    relative_prefix: str = "",
) -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    try:
        names = sorted(os.listdir(dir_fd))
    except (OSError, TypeError) as exc:
        findings.append(
            _verification_error(
                asset_id=asset_id,
                source_type="skill_file",
                summary=f"Unable to scan skill directory: {exc.__class__.__name__}.",
                path=skill_dir / relative_prefix if relative_prefix else skill_dir,
                details={"error": str(exc)},
            )
        )
        return files

    for name in names:
        relative = f"{relative_prefix}{name}"
        path = skill_dir / relative
        try:
            child_stat = _stat_child_no_follow(dir_fd=dir_fd, name=name, path=path)
        except OSError as exc:
            findings.append(
                _verification_error(
                    asset_id=asset_id,
                    source_type="skill_file",
                    summary=f"Unable to inspect skill path: {exc.__class__.__name__}.",
                    path=path,
                    details={"error": str(exc)},
                )
            )
            continue

        if stat.S_ISLNK(child_stat.st_mode):
            findings.append(
                _symlink_finding(
                    asset_id=asset_id,
                    source_type="skill_file",
                    path=path,
                    root=skill_dir,
                )
            )
            continue

        if stat.S_ISDIR(child_stat.st_mode):
            try:
                child_fd = _open_child_dir_no_follow_fd(dir_fd, name, path)
            except OSError as exc:
                findings.append(
                    _verification_error(
                        asset_id=asset_id,
                        source_type="skill_file",
                        summary=f"Unable to open skill directory: {exc.__class__.__name__}.",
                        path=path,
                        details={"error": str(exc)},
                    )
                )
                continue
            try:
                files.update(
                    _read_skill_file_map_fd(
                        dir_fd=child_fd,
                        skill_dir=skill_dir,
                        asset_id=asset_id,
                        findings=findings,
                        relative_prefix=f"{relative}/",
                    )
                )
            finally:
                os.close(child_fd)
            continue

        if path.suffix.lower() not in _SKILL_TEXT_SUFFIXES:
            continue

        content = _read_regular_file_from_dir_fd(
            dir_fd=dir_fd,
            name=name,
            path=path,
            asset_id=asset_id,
            source_type="skill_file",
            findings=findings,
        )
        if content is not None:
            files[relative] = content

    return files


def _open_root_fd_or_result(
    *,
    root: Path,
    asset_id: str,
    source_type: ContextAssetSource,
    scan_label: str,
) -> tuple[int | None, InventoryResult | None]:
    root_ok, root_finding = _validate_inventory_root(
        root=root,
        asset_id=asset_id,
        source_type=source_type,
    )
    if not root_ok:
        return (
            None,
            InventoryResult(
                assets=(),
                findings=(root_finding,) if root_finding is not None else (),
            ),
        )

    try:
        return _open_directory_path_no_follow_fd(root), None
    except OSError as exc:
        return (
            None,
            InventoryResult(
                assets=(),
                findings=(
                    _verification_error(
                        asset_id=asset_id,
                        source_type=source_type,
                        summary=f"Unable to open {scan_label}: {exc.__class__.__name__}.",
                        path=root,
                        details={"error": str(exc)},
                    ),
                ),
            ),
        )


def inventory_user_skills_with_findings(*, user_id: int, skills_root: Path) -> InventoryResult:
    """Inventory per-user skill directories with non-fatal discovery findings."""
    root_asset_id = f"skill:user:{user_id}"
    root_fd, root_result = _open_root_fd_or_result(
        root=skills_root,
        asset_id=root_asset_id,
        source_type="skill_file",
        scan_label="user skills root",
    )
    if root_result is not None:
        return root_result
    if root_fd is None:
        return InventoryResult(assets=(), findings=())

    assets: list[ContextAssetDescriptor] = []
    findings: list[ContextIntegrityFinding] = []
    try:
        try:
            names = sorted(os.listdir(root_fd))
        except (OSError, TypeError) as exc:
            return InventoryResult(
                assets=(),
                findings=(
                    _verification_error(
                        asset_id=root_asset_id,
                        source_type="skill_file",
                        summary=f"Unable to scan user skills root: {exc.__class__.__name__}.",
                        path=skills_root,
                        details={"error": str(exc)},
                    ),
                ),
            )

        for name in names:
            skill_dir = skills_root / name
            asset_id = f"skill:user:{user_id}/{name}"
            try:
                entry_stat = _stat_child_no_follow(dir_fd=root_fd, name=name, path=skill_dir)
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

            if stat.S_ISLNK(entry_stat.st_mode):
                findings.append(
                    _symlink_finding(
                        asset_id=asset_id,
                        source_type="skill_file",
                        path=skill_dir,
                        root=skills_root,
                    )
                )
                continue
            if not stat.S_ISDIR(entry_stat.st_mode):
                continue

            try:
                skill_fd = _open_child_dir_no_follow_fd(root_fd, name, skill_dir)
            except OSError as exc:
                findings.append(
                    _verification_error(
                        asset_id=asset_id,
                        source_type="skill_file",
                        summary=f"Unable to open skill directory: {exc.__class__.__name__}.",
                        path=skill_dir,
                        details={"error": str(exc)},
                    )
                )
                continue

            try:
                files = _read_skill_file_map_fd(
                    dir_fd=skill_fd,
                    skill_dir=skill_dir,
                    asset_id=asset_id,
                    findings=findings,
                )
            finally:
                os.close(skill_fd)
            if "SKILL.md" not in files:
                continue

            metadata = {"skill_name": name}
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
                    display_name=name,
                    executable=True,
                    owner_scope=f"user:{user_id}",
                    path=str(skill_dir),
                    metadata=metadata,
                )
            )
    finally:
        os.close(root_fd)

    return InventoryResult(assets=tuple(assets), findings=tuple(findings))


def inventory_user_skills(user_id: int, skills_root: Path) -> list[ContextAssetDescriptor]:
    """Inventory per-user skill directories."""
    return list(inventory_user_skills_with_findings(user_id=user_id, skills_root=skills_root).assets)


def inventory_prompt_files_with_findings(*, prompts_dir: Path) -> InventoryResult:
    """Inventory config prompt files under a Prompts directory with findings."""
    root_asset_id = f"prompt_file:{prompts_dir.name}"
    root_fd, root_result = _open_root_fd_or_result(
        root=prompts_dir,
        asset_id=root_asset_id,
        source_type="prompt_file",
        scan_label="prompt directory",
    )
    if root_result is not None:
        return root_result
    if root_fd is None:
        return InventoryResult(assets=(), findings=())

    assets: list[ContextAssetDescriptor] = []
    findings: list[ContextIntegrityFinding] = []
    try:
        try:
            names = sorted(os.listdir(root_fd))
        except (OSError, TypeError) as exc:
            return InventoryResult(
                assets=(),
                findings=(
                    _verification_error(
                        asset_id=root_asset_id,
                        source_type="prompt_file",
                        summary=f"Unable to scan prompt directory: {exc.__class__.__name__}.",
                        path=prompts_dir,
                        details={"error": str(exc)},
                    ),
                ),
            )
        for name in names:
            path = prompts_dir / name
            asset_id = f"prompt_file:{name}"
            try:
                entry_stat = _stat_child_no_follow(dir_fd=root_fd, name=name, path=path)
            except OSError as exc:
                findings.append(
                    _verification_error(
                        asset_id=asset_id,
                        source_type="prompt_file",
                        summary=f"Unable to inspect prompt path: {exc.__class__.__name__}.",
                        path=path,
                        details={"error": str(exc)},
                    )
                )
                continue

            if stat.S_ISLNK(entry_stat.st_mode):
                findings.append(
                    _symlink_finding(
                        asset_id=asset_id,
                        source_type="prompt_file",
                        path=path,
                        root=prompts_dir,
                    )
                )
                continue
            if stat.S_ISDIR(entry_stat.st_mode):
                continue
            if path.suffix.lower() not in _PROMPT_SUFFIXES:
                continue

            prompt_bytes = _read_regular_file_from_dir_fd(
                dir_fd=root_fd,
                name=name,
                path=path,
                asset_id=asset_id,
                source_type="prompt_file",
                findings=findings,
            )
            if prompt_bytes is None:
                continue

            metadata = {"path": name}
            digest = canonical_filesystem_digest(
                source_type="prompt_file",
                asset_id=asset_id,
                files={name: prompt_bytes},
                metadata=metadata,
            )
            assets.append(
                ContextAssetDescriptor(
                    asset_id=asset_id,
                    source_type="prompt_file",
                    digest=digest,
                    display_name=name,
                    executable=False,
                    owner_scope="system",
                    path=str(path),
                    metadata=metadata,
                )
            )
    finally:
        os.close(root_fd)

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
        try:
            parent_fd = _open_directory_path_no_follow_fd(path.parent)
        except OSError as exc:
            findings.append(
                _verification_error(
                    asset_id=asset_id,
                    source_type="prompt_file",
                    summary=f"Unable to open prompt override parent: {exc.__class__.__name__}.",
                    path=path.parent,
                    details={"error": str(exc)},
                )
            )
            continue

        try:
            try:
                entry_stat = _stat_child_no_follow(dir_fd=parent_fd, name=path.name, path=path)
            except OSError as exc:
                findings.append(
                    _verification_error(
                        asset_id=asset_id,
                        source_type="prompt_file",
                        summary=f"Unable to inspect prompt override path: {exc.__class__.__name__}.",
                        path=path,
                        details={"error": str(exc)},
                    )
                )
                continue

            if stat.S_ISLNK(entry_stat.st_mode):
                findings.append(
                    _symlink_finding(
                        asset_id=asset_id,
                        source_type="prompt_file",
                        path=path,
                        root=None,
                    )
                )
                continue

            prompt_bytes = _read_regular_file_from_dir_fd(
                dir_fd=parent_fd,
                name=path.name,
                path=path,
                asset_id=asset_id,
                source_type="prompt_file",
                findings=findings,
            )
            if prompt_bytes is None:
                continue
        finally:
            os.close(parent_fd)

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
