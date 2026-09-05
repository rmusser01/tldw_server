"""Private, manifest-last storage for managed llama.cpp slot snapshots."""

from __future__ import annotations

import errno
import fcntl
import hashlib
import json
import os
import re
import stat
import uuid
from pathlib import Path

from pydantic import ValidationError

from .llamacpp_snapshot_models import OperationReceipt, SnapshotMetadata

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
_CHUNK_SIZE = 1024 * 1024
_MAX_MANIFEST_BYTES = 1024 * 1024


class SnapshotStoreError(RuntimeError):
    """Base error for private snapshot storage."""


class SnapshotCorruptError(SnapshotStoreError):
    """Raised when committed bytes no longer match their manifest."""


class SnapshotNotFoundError(SnapshotStoreError):
    """Raised when no valid committed snapshot or receipt exists."""


class SnapshotStorageUnavailableError(SnapshotStoreError):
    """Raised when storage exists but cannot be read reliably."""


class _MalformedManifestError(SnapshotStoreError):
    """Internal signal for an incomplete or invalid catalog entry."""


class SnapshotStore:
    """Filesystem store held by one process for its entire lifetime."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self._lock_fd: int | None = None
        self._ensure_private_dir(self.root)
        self._acquire_owner_lock()

    def __enter__(self) -> SnapshotStore:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def close(self) -> None:
        """Release the process ownership fence."""
        if self._lock_fd is not None:
            fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
            os.close(self._lock_fd)
            self._lock_fd = None

    def list(self, profile_id: str) -> list[SnapshotMetadata]:
        """List only snapshots with a valid manifest and regular binary."""
        self._require_open()
        paths = self._profile_paths(profile_id)
        results: list[SnapshotMetadata] = []
        try:
            manifests = list(paths["manifests"].iterdir())
        except OSError as exc:
            raise SnapshotStorageUnavailableError("snapshot catalog is unavailable") from exc
        for manifest in manifests:
            if manifest.suffix != ".json" or manifest.is_symlink():
                continue
            try:
                item = self._read_model(manifest, SnapshotMetadata)
                if item.profile_id != profile_id or manifest.stem != item.snapshot_id:
                    continue
                binary = paths["snapshots"] / f"{item.snapshot_id}.bin"
                self._require_regular_private_file(binary)
            except (SnapshotNotFoundError, _MalformedManifestError, ValidationError, ValueError):
                continue
            results.append(item)
        return sorted(results, key=lambda item: item.commit_sequence, reverse=True)

    def commit(self, profile_id: str, staged: Path, metadata: SnapshotMetadata) -> SnapshotMetadata:
        """Verify and atomically publish bytes before publishing metadata."""
        self._require_open()
        self._validate_id(profile_id)
        if metadata.profile_id != profile_id:
            raise SnapshotStoreError("metadata profile does not match storage profile")
        self._validate_id(metadata.snapshot_id)
        paths = self._profile_paths(profile_id)
        source_fd = self._open_regular_readonly(Path(staged))
        source_before = os.fstat(source_fd)
        binary = paths["snapshots"] / f"{metadata.snapshot_id}.bin"
        manifest = paths["manifests"] / f"{metadata.snapshot_id}.json"
        if binary.exists() or manifest.exists():
            os.close(source_fd)
            raise SnapshotStoreError("snapshot ID is already committed")
        temp = paths["snapshots"] / f".{uuid.uuid4().hex}.tmp"
        temp_fd: int | None = None
        try:
            temp_fd = self._open_exclusive_write(temp)
            digest = hashlib.sha256()
            size = 0
            while chunk := os.read(source_fd, _CHUNK_SIZE):
                self._write_chunk(temp_fd, chunk)
                digest.update(chunk)
                size += len(chunk)
            source_after = os.fstat(source_fd)
            before_identity = (
                source_before.st_dev,
                source_before.st_ino,
                source_before.st_size,
                source_before.st_mtime_ns,
            )
            after_identity = (
                source_after.st_dev,
                source_after.st_ino,
                source_after.st_size,
                source_after.st_mtime_ns,
            )
            if before_identity != after_identity:
                raise SnapshotStoreError("staged snapshot changed while copying")
            if size != metadata.byte_count or digest.hexdigest() != metadata.sha256:
                raise SnapshotCorruptError("staged snapshot does not match metadata")
            self._checkpoint("copy")
            self._checkpoint("file_fsync")
            os.fsync(temp_fd)
            os.close(temp_fd)
            temp_fd = None
            self._checkpoint("binary_rename")
            os.replace(temp, binary)
            self._checkpoint("directory_fsync")
            self._fsync_dir(paths["snapshots"])
            self._publish_json(manifest, metadata.model_dump(mode="json"), checkpoint_prefix="manifest")
            return metadata
        finally:
            os.close(source_fd)
            if temp_fd is not None:
                os.close(temp_fd)
            try:
                temp.unlink()
            except FileNotFoundError:
                pass

    def stage_restore(self, profile_id: str, snapshot_id: str, working: Path) -> Path:
        """Copy a verified committed snapshot into a private launch directory."""
        self._require_open()
        paths = self._profile_paths(profile_id)
        self._validate_id(snapshot_id)
        metadata = self._read_snapshot(paths, profile_id, snapshot_id)
        self._ensure_private_dir(Path(working))
        source = paths["snapshots"] / f"{snapshot_id}.bin"
        source_fd = self._open_regular_readonly(source)
        destination = Path(working) / f"restore-{uuid.uuid4().hex}.bin"
        destination_fd: int | None = None
        try:
            destination_fd = self._open_exclusive_write(destination)
            digest = hashlib.sha256()
            size = 0
            while chunk := os.read(source_fd, _CHUNK_SIZE):
                self._write_chunk(destination_fd, chunk)
                digest.update(chunk)
                size += len(chunk)
            if size != metadata.byte_count or digest.hexdigest() != metadata.sha256:
                raise SnapshotCorruptError("committed snapshot hash mismatch")
            os.fsync(destination_fd)
            return destination
        except BaseException:
            try:
                destination.unlink()
            except FileNotFoundError:
                pass
            raise
        finally:
            os.close(source_fd)
            if destination_fd is not None:
                os.close(destination_fd)

    def prune(self, profile_id: str, keep: int) -> list[str]:
        """Delete oldest commit sequences and return IDs that could not be removed."""
        self._require_open()
        if keep < 1:
            raise ValueError("keep must be at least one")
        committed = self.list(profile_id)
        failed: list[str] = []
        for item in sorted(committed[keep:], key=lambda value: value.commit_sequence):
            try:
                self.delete(profile_id, item.snapshot_id)
            except OSError:
                failed.append(item.snapshot_id)
        return failed

    def delete(self, profile_id: str, snapshot_id: str) -> None:
        """Delete one committed snapshot without accepting filesystem paths."""
        self._require_open()
        paths = self._profile_paths(profile_id)
        self._validate_id(snapshot_id)
        self._read_snapshot(paths, profile_id, snapshot_id)
        binary = paths["snapshots"] / f"{snapshot_id}.bin"
        manifest = paths["manifests"] / f"{snapshot_id}.json"
        binary.unlink()
        self._fsync_dir(paths["snapshots"])
        manifest.unlink()
        self._fsync_dir(paths["manifests"])

    def write_receipt(self, receipt: OperationReceipt) -> None:
        """Atomically persist a path-free operation receipt."""
        self._require_open()
        paths = self._profile_paths(receipt.profile_id)
        target = paths["receipts"] / f"{receipt.operation_id}.json"
        self._publish_json(target, receipt.model_dump(mode="json"), replace=True)

    def read_receipt(self, profile_id: str, operation_id: str) -> OperationReceipt:
        """Read and validate one receipt within its profile boundary."""
        self._require_open()
        paths = self._profile_paths(profile_id)
        self._validate_id(operation_id)
        target = paths["receipts"] / f"{operation_id}.json"
        try:
            receipt = self._read_model(target, OperationReceipt)
        except (SnapshotNotFoundError, _MalformedManifestError, ValidationError, ValueError) as exc:
            raise SnapshotNotFoundError("operation receipt not found") from exc
        if receipt.profile_id != profile_id or receipt.operation_id != operation_id:
            raise SnapshotNotFoundError("operation receipt not found")
        return receipt

    def _profile_paths(self, profile_id: str) -> dict[str, Path]:
        self._validate_id(profile_id)
        profile = self.root / profile_id
        self._ensure_private_dir(profile)
        result = {"profile": profile}
        for name in ("snapshots", "manifests", "receipts"):
            path = profile / name
            self._ensure_private_dir(path)
            result[name] = path
        return result

    def _read_snapshot(self, paths: dict[str, Path], profile_id: str, snapshot_id: str) -> SnapshotMetadata:
        manifest = paths["manifests"] / f"{snapshot_id}.json"
        try:
            item = self._read_model(manifest, SnapshotMetadata)
            binary = paths["snapshots"] / f"{snapshot_id}.bin"
            self._require_regular_private_file(binary)
        except (SnapshotNotFoundError, _MalformedManifestError, ValidationError, ValueError) as exc:
            raise SnapshotNotFoundError("snapshot not found") from exc
        if item.profile_id != profile_id or item.snapshot_id != snapshot_id:
            raise SnapshotNotFoundError("snapshot not found")
        return item

    def _publish_json(
        self,
        target: Path,
        value: object,
        *,
        replace: bool = False,
        checkpoint_prefix: str | None = None,
    ) -> None:
        if not replace and target.exists():
            raise SnapshotStoreError("published file already exists")
        payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
        if len(payload) > _MAX_MANIFEST_BYTES:
            raise SnapshotStoreError("manifest exceeds size limit")
        temp = target.parent / f".{uuid.uuid4().hex}.tmp"
        fd: int | None = None
        try:
            fd = self._open_exclusive_write(temp)
            if checkpoint_prefix:
                self._checkpoint(f"{checkpoint_prefix}_write")
            self._write_chunk(fd, payload)
            if checkpoint_prefix:
                self._checkpoint(f"{checkpoint_prefix}_fsync")
            os.fsync(fd)
            os.close(fd)
            fd = None
            if checkpoint_prefix:
                self._checkpoint(f"{checkpoint_prefix}_rename")
            os.replace(temp, target)
            if checkpoint_prefix:
                self._checkpoint(f"{checkpoint_prefix}_directory_fsync")
            self._fsync_dir(target.parent)
        finally:
            if fd is not None:
                os.close(fd)
            try:
                temp.unlink()
            except FileNotFoundError:
                pass

    def _read_model(self, path: Path, model: type[SnapshotMetadata] | type[OperationReceipt]):
        fd = self._open_regular_readonly(path)
        try:
            info = os.fstat(fd)
            if info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) & 0o077:
                raise SnapshotStoreError("manifest confinement is not private")
            chunks: list[bytes] = []
            total = 0
            try:
                while chunk := os.read(fd, min(_CHUNK_SIZE, _MAX_MANIFEST_BYTES + 1 - total)):
                    chunks.append(chunk)
                    total += len(chunk)
                    if total > _MAX_MANIFEST_BYTES:
                        raise _MalformedManifestError("manifest exceeds size limit")
            except OSError as exc:
                raise SnapshotStorageUnavailableError("snapshot metadata read failed") from exc
            return model.model_validate_json(b"".join(chunks))
        finally:
            os.close(fd)

    def _acquire_owner_lock(self) -> None:
        lock_path = self.root / ".owner.lock"
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0) | self._no_follow_flag()
        fd = os.open(lock_path, flags, 0o600)
        os.fchmod(fd, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            os.close(fd)
            if exc.errno in (errno.EACCES, errno.EAGAIN):
                raise SnapshotStoreError("snapshot root is owned by another process") from exc
            raise
        self._lock_fd = fd

    @staticmethod
    def _validate_id(value: str) -> None:
        if not _ID_RE.fullmatch(value):
            raise SnapshotStoreError("invalid opaque identifier")

    @staticmethod
    def _ensure_private_dir(path: Path) -> None:
        SnapshotStore._reject_symlink_components(path)
        try:
            path.mkdir(mode=0o700)
        except FileExistsError:
            pass
        try:
            info = path.lstat()
        except OSError as exc:
            raise SnapshotStoreError("private storage directory unavailable") from exc
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid():
            raise SnapshotStoreError("unsupported private storage confinement")
        if stat.S_IMODE(info.st_mode) != 0o700:
            try:
                path.chmod(0o700)
            except OSError as exc:
                raise SnapshotStoreError("unsupported private storage confinement") from exc
            if stat.S_IMODE(path.lstat().st_mode) != 0o700:
                raise SnapshotStoreError("unsupported private storage confinement")

    @staticmethod
    def _open_regular_readonly(path: Path) -> int:
        SnapshotStore._reject_symlink_components(path)
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | SnapshotStore._no_follow_flag()
        try:
            fd = os.open(path, flags)
        except FileNotFoundError as exc:
            raise SnapshotNotFoundError("file not found") from exc
        except OSError as exc:
            if exc.errno == errno.ELOOP:
                raise SnapshotStoreError("file is unsafe") from exc
            raise SnapshotStorageUnavailableError("file is unavailable") from exc
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            os.close(fd)
            raise SnapshotStoreError("file must be regular")
        return fd

    @staticmethod
    def _require_regular_private_file(path: Path) -> None:
        fd = SnapshotStore._open_regular_readonly(path)
        try:
            info = os.fstat(fd)
            if info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) & 0o077:
                raise SnapshotStoreError("snapshot file confinement is not private")
        finally:
            os.close(fd)

    @staticmethod
    def _open_exclusive_write(path: Path) -> int:
        SnapshotStore._reject_symlink_components(path)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0) | SnapshotStore._no_follow_flag()
        return os.open(path, flags, 0o600)

    @staticmethod
    def _write_chunk(fd: int, payload: bytes) -> None:
        view = memoryview(payload)
        while view:
            written = os.write(fd, view)
            if written == 0:
                raise OSError(errno.ENOSPC, "short filesystem write")
            view = view[written:]

    @staticmethod
    def _fsync_dir(path: Path) -> None:
        SnapshotStore._reject_symlink_components(path)
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | SnapshotStore._no_follow_flag()
        if hasattr(os, "O_DIRECTORY"):
            flags |= os.O_DIRECTORY
        fd = os.open(path, flags)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def _require_open(self) -> None:
        if self._lock_fd is None:
            raise SnapshotStoreError("snapshot store is closed")

    def _checkpoint(self, _boundary: str) -> None:
        """No-op fault-injection seam used to verify publication recovery."""

    @staticmethod
    def _no_follow_flag() -> int:
        flag = getattr(os, "O_NOFOLLOW", 0)
        if not flag:
            raise SnapshotStoreError("no-follow filesystem confinement is unsupported")
        return flag

    @staticmethod
    def _reject_symlink_components(path: Path) -> None:
        """Reject every existing symlink component without resolving the path."""
        SnapshotStore._no_follow_flag()
        absolute = path if path.is_absolute() else Path.cwd() / path
        current = Path(absolute.anchor)
        for part in absolute.parts[1:]:
            current = current / part
            try:
                info = current.lstat()
            except FileNotFoundError:
                continue
            except OSError as exc:
                raise SnapshotStorageUnavailableError("path confinement check failed") from exc
            if stat.S_ISLNK(info.st_mode):
                raise SnapshotStoreError("symlink path components are unsupported")
