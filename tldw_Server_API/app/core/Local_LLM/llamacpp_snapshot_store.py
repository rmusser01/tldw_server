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

from pydantic import BaseModel, Field, ValidationError

from .llamacpp_snapshot_models import OperationReceipt, SnapshotMetadata

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
_CHUNK_SIZE = 1024 * 1024
_MAX_MANIFEST_BYTES = 1024 * 1024


class _Sequence(BaseModel):
    value: int = Field(ge=0)


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
        self._owner_pid = os.getpid()
        self._ensure_private_dir(self.root)
        self._acquire_owner_lock()

    def __enter__(self) -> SnapshotStore:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def close(self) -> None:
        """Release the process ownership fence."""
        if self._lock_fd is not None:
            if os.getpid() == self._owner_pid:
                fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
            os.close(self._lock_fd)
            self._lock_fd = None

    def list(self, profile_id: str) -> list[SnapshotMetadata]:
        """List only snapshots with a valid manifest and regular binary."""
        self._require_open()
        paths = self._profile_paths(profile_id)
        results: list[SnapshotMetadata] = []
        for name in self._list_names(paths["manifests"]):
            manifest = paths["manifests"] / name
            if manifest.suffix != ".json":
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
        if self._exists(binary) or self._exists(manifest):
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
            self._replace(temp, binary)
            self._checkpoint("directory_fsync")
            self._fsync_dir(paths["snapshots"])
            self._publish_json(manifest, metadata.model_dump(mode="json"), checkpoint_prefix="manifest")
            return metadata
        finally:
            os.close(source_fd)
            if temp_fd is not None:
                os.close(temp_fd)
            try:
                self._unlink(temp)
            except SnapshotNotFoundError:
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
                self._unlink(destination)
            except SnapshotNotFoundError:
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
        self._unlink(binary)
        self._fsync_dir(paths["snapshots"])
        self._unlink(manifest)
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

    def list_receipts(self, profile_id: str) -> list[OperationReceipt]:
        """Read retained receipts; receipt history is never automatically pruned."""
        self._require_open()
        paths = self._profile_paths(profile_id)
        result = []
        for name in self._list_names(paths["receipts"]):
            if name.endswith(".json"):
                result.append(self.read_receipt(profile_id, name[:-5]))
        return sorted(result, key=lambda item: (item.created_at, item.operation_id))

    def token_key(self) -> bytes:
        """Load or atomically create a private signing key under the owner fence."""
        self._require_open()
        path = self.root / ".request-key.json"
        if not self._exists(path):
            self._publish_json(path, {"key": os.urandom(32).hex()})

        class Key(BaseModel):
            key: str = Field(pattern=r"^[0-9a-f]{64}$")

        return bytes.fromhex(self._read_model(path, Key).key)

    def allocate_sequence(self, profile_id: str) -> int:
        """Durably allocate an increasing sequence even after all snapshots are deleted."""
        self._require_open()
        path = self._profile_paths(profile_id)["profile"] / "sequence.json"
        try:
            current = self._read_model(path, _Sequence).value
        except SnapshotNotFoundError:
            current = max((item.commit_sequence for item in self.list(profile_id)), default=0)
        self._publish_json(path, {"value": current + 1}, replace=True)
        return current + 1

    def launch_directory(self, profile_id: str, generation: str) -> Path:
        """Create a private generated directory, never exposing the committed catalog."""
        self._require_open()
        self._validate_id(generation)
        base = self._profile_paths(profile_id)["profile"] / "working"
        self._ensure_private_dir(base)
        path = base / generation
        self._ensure_private_dir(path)
        return path

    def cleanup_launch(self, profile_id: str, generation: str) -> None:
        """Remove working files only after the caller proves the owned child exited."""
        path = self.launch_directory(profile_id, generation)
        for name in self._list_names(path):
            self._unlink(path / name)
        parent_fd = self._open_directory_fd(path.parent)
        try:
            os.rmdir(path.name, dir_fd=parent_fd)
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)

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
        if not replace and self._exists(target):
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
            self._replace(temp, target)
            if checkpoint_prefix:
                self._checkpoint(f"{checkpoint_prefix}_directory_fsync")
            self._fsync_dir(target.parent)
        finally:
            if fd is not None:
                os.close(fd)
            try:
                self._unlink(temp)
            except SnapshotNotFoundError:
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
        parent_fd = self._open_directory_fd(lock_path.parent)
        try:
            fd = os.open(lock_path.name, flags, 0o600, dir_fd=parent_fd)
        finally:
            os.close(parent_fd)
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
        fd = SnapshotStore._open_directory_fd(path, create=True)
        try:
            info = os.fstat(fd)
            if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid():
                raise SnapshotStoreError("unsupported private storage confinement")
            if stat.S_IMODE(info.st_mode) != 0o700:
                try:
                    os.fchmod(fd, 0o700)
                except OSError as exc:
                    raise SnapshotStoreError("unsupported private storage confinement") from exc
                if stat.S_IMODE(os.fstat(fd).st_mode) != 0o700:
                    raise SnapshotStoreError("unsupported private storage confinement")
        finally:
            os.close(fd)

    @staticmethod
    def _open_regular_readonly(path: Path) -> int:
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | SnapshotStore._no_follow_flag()
        parent_fd = SnapshotStore._open_directory_fd(path.parent)
        try:
            fd = os.open(path.name, flags, dir_fd=parent_fd)
        except FileNotFoundError as exc:
            raise SnapshotNotFoundError("file not found") from exc
        except OSError as exc:
            if exc.errno == errno.ELOOP:
                raise SnapshotStoreError("file is unsafe") from exc
            raise SnapshotStorageUnavailableError("file is unavailable") from exc
        finally:
            os.close(parent_fd)
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
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0) | SnapshotStore._no_follow_flag()
        parent_fd = SnapshotStore._open_directory_fd(path.parent)
        try:
            return os.open(path.name, flags, 0o600, dir_fd=parent_fd)
        finally:
            os.close(parent_fd)

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
        fd = SnapshotStore._open_directory_fd(path)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def _require_open(self) -> None:
        if self._lock_fd is None or os.getpid() != self._owner_pid:
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
    def _open_directory_fd(path: Path, *, create: bool = False) -> int:
        """Walk a directory using held descriptors so ancestors cannot be swapped."""
        no_follow = SnapshotStore._no_follow_flag()
        absolute = path if path.is_absolute() else Path.cwd() / path
        if ".." in absolute.parts:
            raise SnapshotStoreError("parent traversal is unsupported")
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | no_follow
        if hasattr(os, "O_DIRECTORY"):
            flags |= os.O_DIRECTORY
        fd = os.open(absolute.anchor, flags)
        parts = absolute.parts[1:]
        try:
            for index, part in enumerate(parts):
                is_final = index == len(parts) - 1
                try:
                    next_fd = os.open(part, flags, dir_fd=fd)
                except FileNotFoundError:
                    if not create or not is_final:
                        raise SnapshotNotFoundError("directory not found") from None
                    os.mkdir(part, 0o700, dir_fd=fd)
                    next_fd = os.open(part, flags, dir_fd=fd)
                except OSError as exc:
                    if exc.errno in (errno.ELOOP, errno.ENOTDIR):
                        raise SnapshotStoreError("symlink directory components are unsupported") from exc
                    raise SnapshotStorageUnavailableError("directory traversal failed") from exc
                os.close(fd)
                fd = next_fd
            return fd
        except BaseException:
            os.close(fd)
            raise

    @staticmethod
    def _replace(source: Path, target: Path) -> None:
        source_fd = SnapshotStore._open_directory_fd(source.parent)
        try:
            target_fd = SnapshotStore._open_directory_fd(target.parent)
            try:
                os.replace(source.name, target.name, src_dir_fd=source_fd, dst_dir_fd=target_fd)
            finally:
                os.close(target_fd)
        finally:
            os.close(source_fd)

    @staticmethod
    def _unlink(path: Path) -> None:
        parent_fd = SnapshotStore._open_directory_fd(path.parent)
        try:
            try:
                os.unlink(path.name, dir_fd=parent_fd)
            except FileNotFoundError as exc:
                raise SnapshotNotFoundError("file not found") from exc
        finally:
            os.close(parent_fd)

    @staticmethod
    def _exists(path: Path) -> bool:
        parent_fd = SnapshotStore._open_directory_fd(path.parent)
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | SnapshotStore._no_follow_flag()
        try:
            try:
                fd = os.open(path.name, flags, dir_fd=parent_fd)
            except FileNotFoundError:
                return False
            except OSError as exc:
                if exc.errno == errno.ELOOP:
                    raise SnapshotStoreError("unsafe file target") from exc
                raise SnapshotStorageUnavailableError("file existence check failed") from exc
            os.close(fd)
            return True
        finally:
            os.close(parent_fd)

    @staticmethod
    def _list_names(path: Path) -> list[str]:
        fd = SnapshotStore._open_directory_fd(path)
        try:
            return os.listdir(fd)
        except OSError as exc:
            raise SnapshotStorageUnavailableError("snapshot catalog is unavailable") from exc
        finally:
            os.close(fd)
