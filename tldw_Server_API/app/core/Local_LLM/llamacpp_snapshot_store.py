"""Private, manifest-last storage for managed llama.cpp slot snapshots."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import re
import stat
import uuid
from collections.abc import Callable
from pathlib import Path

try:
    import fcntl
except ImportError:
    fcntl = None

from pydantic import BaseModel, Field, ValidationError

from ..exceptions import (
    SnapshotCorruptError,
    SnapshotNotFoundError,
    SnapshotStorageUnavailableError,
    SnapshotStoreError,
)
from .llamacpp_snapshot_models import OperationReceipt, SnapshotMetadata

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
_CHUNK_SIZE = 1024 * 1024
_MAX_MANIFEST_BYTES = 1024 * 1024


class _Sequence(BaseModel):
    value: int = Field(ge=0)


class _PendingBinary(BaseModel):
    """Durable proof that a copied inode has not attempted manifest publication."""

    temporary_name: str = Field(pattern=r"^\.[0-9a-f]{32}\.tmp$")
    device: int
    inode: int


class _MalformedManifestError(SnapshotStoreError):
    """Internal signal for an incomplete or invalid catalog entry."""


class SnapshotStore:
    """Filesystem store held by one process for its entire lifetime."""

    def __init__(self, root: Path):
        if fcntl is None:
            raise SnapshotStorageUnavailableError("Snapshot storage requires POSIX ownership locking.")
        self.root = Path(root).absolute()
        self._lock_fd: int | None = None
        self._root_fd: int | None = None
        self._root_identity: tuple[int, int] | None = None
        self._owner_pid = os.getpid()
        try:
            self._ensure_private_dir(self.root)
            self._root_fd = self._open_directory_fd(self.root)
            info = os.fstat(self._root_fd)
            self._root_identity = (info.st_dev, info.st_ino)
            self._acquire_owner_lock()
            self._recover_pending_binaries()
        except BaseException:
            self.close()
            raise

    def __enter__(self) -> SnapshotStore:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    @classmethod
    def profile_state_proven_absent(cls, root: Path, profile_id: str) -> bool:
        """Prove an unsupported platform has no storage for one profile.

        A platform with ownership locking must acquire that lock and inspect the
        catalog normally.  Without it, only an absent profile directory under an
        unchanged, confined root is sufficient evidence to skip initialization.
        """
        cls._validate_id(profile_id)
        if fcntl is not None:
            return False
        root = Path(root)
        try:
            initial = root.lstat()
        except FileNotFoundError:
            return True
        except OSError as exc:
            raise SnapshotStorageUnavailableError("snapshot root cannot be inspected") from exc
        if not stat.S_ISDIR(initial.st_mode) or stat.S_ISLNK(initial.st_mode):
            raise SnapshotStoreError("snapshot root confinement is unsafe")
        if os.name == "posix" and (initial.st_uid != os.geteuid() or stat.S_IMODE(initial.st_mode) != 0o700):
            raise SnapshotStoreError("snapshot root confinement is not private")
        try:
            (root / profile_id).lstat()
        except FileNotFoundError:
            try:
                final = root.lstat()
            except OSError as exc:
                raise SnapshotStorageUnavailableError("snapshot root changed during inspection") from exc
            if (initial.st_dev, initial.st_ino, initial.st_mode, initial.st_mtime_ns) != (
                final.st_dev,
                final.st_ino,
                final.st_mode,
                final.st_mtime_ns,
            ):
                raise SnapshotStorageUnavailableError("snapshot root changed during inspection") from None
            return True
        except OSError as exc:
            raise SnapshotStorageUnavailableError("snapshot profile state cannot be inspected") from exc
        return False

    def close(self) -> None:
        """Release the process ownership fence."""
        try:
            if self._lock_fd is not None:
                try:
                    if os.getpid() == self._owner_pid:
                        fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
                finally:
                    os.close(self._lock_fd)
                    self._lock_fd = None
        finally:
            if self._root_fd is not None:
                os.close(self._root_fd)
                self._root_fd = None

    def has_retained_state(self, profile_id: str) -> bool:
        """Return false only for proven-empty snapshot and manifest directories.

        Corrupt, incomplete and unknown entries retain profile ownership. Storage
        inspection errors propagate so a caller cannot mistake them for absence.
        Receipts and the allocation sequence alone do not retain snapshot bytes.
        """
        self._require_open()
        paths = self._profile_paths(profile_id)
        return bool(self._list_names(paths["snapshots"]) or self._list_names(paths["manifests"]))

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
        binary = paths["snapshots"] / f"{metadata.snapshot_id}.bin"
        manifest = paths["manifests"] / f"{metadata.snapshot_id}.json"
        pending = paths["snapshots"] / f".pending-{metadata.snapshot_id}.json"
        if self._exists(binary) or self._exists(manifest) or self._exists(pending):
            raise SnapshotStoreError("snapshot ID is already committed")
        temp = paths["snapshots"] / f".{uuid.uuid4().hex}.tmp"
        temp_fd: int | None = None
        copied_identity: tuple[int, int] | None = None
        pending_published = False
        source_fd = self._open_regular_readonly(Path(staged))
        try:
            source_before = os.fstat(source_fd)
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
            copied = os.fstat(temp_fd)
            copied_identity = (copied.st_dev, copied.st_ino)
            os.close(temp_fd)
            temp_fd = None
            self._publish_json(
                pending,
                _PendingBinary(
                    temporary_name=temp.name,
                    device=copied.st_dev,
                    inode=copied.st_ino,
                ).model_dump(),
            )
            pending_published = True
            self._checkpoint("binary_rename")
            self._replace(temp, binary)
            self._checkpoint("directory_fsync")
            self._fsync_dir(paths["snapshots"])
            self._publish_json(
                manifest,
                metadata.model_dump(mode="json"),
                checkpoint_prefix="manifest",
                before_publish=lambda: self._remove_pending_marker(pending),
            )
            return metadata
        except BaseException:
            if copied_identity is not None and not self._exists(manifest):
                self._remove_matching_file(binary, copied_identity)
                if pending_published and self._exists(pending):
                    self._remove_pending_marker(pending)
            raise
        finally:
            os.close(source_fd)
            if temp_fd is not None:
                os.close(temp_fd)
            try:
                self._unlink(temp)
                self._fsync_dir(temp.parent)
            except SnapshotNotFoundError:
                pass

    def stage_restore(self, profile_id: str, snapshot_id: str, working: Path, *, filename: str | None = None) -> Path:
        """Copy a verified committed snapshot into a private launch directory."""
        self._require_open()
        if filename is not None and not re.fullmatch(r"restore-[0-9a-f]{32}\.bin", filename):
            raise SnapshotStoreError("invalid generated restore filename")
        paths = self._profile_paths(profile_id)
        self._validate_id(snapshot_id)
        metadata = self._read_snapshot(paths, profile_id, snapshot_id)
        self._ensure_private_dir(Path(working))
        source = paths["snapshots"] / f"{snapshot_id}.bin"
        source_fd = self._open_regular_readonly(source)
        destination = Path(working) / (filename or f"restore-{uuid.uuid4().hex}.bin")
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
            if destination_fd is not None:
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
        parent_fd = self._directory_fd(path.parent)
        try:
            os.rmdir(path.name, dir_fd=parent_fd)
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)

    def remove_working_file(self, profile_id: str, generation: str, filename: str) -> None:
        """Remove one generated operation file after its verified completion."""
        self._require_open()
        self._validate_id(profile_id)
        self._validate_id(generation)
        if not re.fullmatch(r"(?:save|restore)-[0-9a-f]{32}\.bin", filename):
            raise SnapshotStoreError("invalid generated working filename")
        directory = self.root / profile_id / "working" / generation
        self._unlink(directory / filename)
        self._fsync_dir(directory)

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
        before_publish: Callable[[], None] | None = None,
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
            if before_publish is not None:
                before_publish()
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
                self._fsync_dir(temp.parent)
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
        parent_fd = self._directory_fd(lock_path.parent)
        try:
            fd = os.open(lock_path.name, flags, 0o600, dir_fd=parent_fd)
        finally:
            os.close(parent_fd)
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise SnapshotStoreError("ownership lock must be regular")
            os.fchmod(fd, 0o600)
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException as exc:
            os.close(fd)
            if isinstance(exc, OSError) and exc.errno in (errno.EACCES, errno.EAGAIN):
                raise SnapshotStoreError("snapshot root is owned by another process") from exc
            raise
        self._lock_fd = fd

    @staticmethod
    def _validate_id(value: str) -> None:
        if not _ID_RE.fullmatch(value):
            raise SnapshotStoreError("invalid opaque identifier")

    def _ensure_private_dir(self, path: Path) -> None:
        fd = self._directory_fd(path, create=True)
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

    def _open_regular_readonly(self, path: Path) -> int:
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | SnapshotStore._no_follow_flag()
        parent_fd = self._directory_fd(path.parent)
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

    def _require_regular_private_file(self, path: Path) -> None:
        fd = self._open_regular_readonly(path)
        try:
            info = os.fstat(fd)
            if info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) & 0o077:
                raise SnapshotStoreError("snapshot file confinement is not private")
        finally:
            os.close(fd)

    def _open_exclusive_write(self, path: Path) -> int:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0) | SnapshotStore._no_follow_flag()
        parent_fd = self._directory_fd(path.parent)
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

    def _fsync_dir(self, path: Path) -> None:
        fd = self._directory_fd(path)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def _require_open(self) -> None:
        if self._lock_fd is None or os.getpid() != self._owner_pid:
            raise SnapshotStoreError("snapshot store is closed")
        fd = self._directory_fd(self.root)
        os.close(fd)

    def _directory_fd(self, path: Path, *, create: bool = False) -> int:
        return self._open_directory_fd(
            path,
            create=create,
            expected_root=(self.root, self._root_identity) if self._root_identity is not None else None,
        )

    def _checkpoint(self, _boundary: str) -> None:
        """No-op fault-injection seam used to verify publication recovery."""

    @staticmethod
    def _no_follow_flag() -> int:
        flag = getattr(os, "O_NOFOLLOW", 0)
        if not flag:
            raise SnapshotStoreError("no-follow filesystem confinement is unsupported")
        return flag

    @staticmethod
    def _open_directory_fd(
        path: Path,
        *,
        create: bool = False,
        expected_root: tuple[Path, tuple[int, int]] | None = None,
    ) -> int:
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
                if expected_root is not None and absolute.parts[: index + 2] == expected_root[0].parts:
                    info = os.fstat(fd)
                    if (info.st_dev, info.st_ino) != expected_root[1]:
                        raise SnapshotStorageUnavailableError("snapshot root identity changed")
            return fd
        except BaseException:
            os.close(fd)
            raise

    def _replace(self, source: Path, target: Path) -> None:
        source_fd = self._directory_fd(source.parent)
        try:
            target_fd = self._directory_fd(target.parent)
            try:
                os.replace(source.name, target.name, src_dir_fd=source_fd, dst_dir_fd=target_fd)
            finally:
                os.close(target_fd)
        finally:
            os.close(source_fd)

    def _unlink(self, path: Path) -> None:
        parent_fd = self._directory_fd(path.parent)
        try:
            try:
                os.unlink(path.name, dir_fd=parent_fd)
            except FileNotFoundError as exc:
                raise SnapshotNotFoundError("file not found") from exc
        finally:
            os.close(parent_fd)

    def _exists(self, path: Path) -> bool:
        parent_fd = self._directory_fd(path.parent)
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

    def _list_names(self, path: Path) -> list[str]:
        fd = self._directory_fd(path)
        try:
            return os.listdir(fd)
        except OSError as exc:
            raise SnapshotStorageUnavailableError("snapshot catalog is unavailable") from exc
        finally:
            os.close(fd)

    def _remove_matching_file(self, path: Path, identity: tuple[int, int]) -> None:
        """Remove only this publication's inode, preserving replaced evidence."""
        parent_fd = self._directory_fd(path.parent)
        try:
            try:
                info = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
            except FileNotFoundError:
                return
            if stat.S_ISREG(info.st_mode) and (info.st_dev, info.st_ino) == identity:
                os.unlink(path.name, dir_fd=parent_fd)
                os.fsync(parent_fd)
        finally:
            os.close(parent_fd)

    def _remove_pending_marker(self, path: Path) -> None:
        # Retire proof durably BEFORE attempting a manifest rename. After this
        # point a crash has unknown publication outcome and must preserve bytes.
        self._unlink(path)
        self._fsync_dir(path.parent)

    def _recover_pending_binaries(self) -> None:
        """Reclaim only journaled pre-publication copies under the owner lock.

        A binary without a manifest alone is not proof: its committed manifest
        could have been lost or corrupted. Missing/invalid journals and any
        mismatched inode remain operator recovery evidence.
        """
        for profile_id in self._list_names(self.root):
            if not _ID_RE.fullmatch(profile_id):
                continue
            profile = self.root / profile_id
            try:
                names = self._list_names(profile / "snapshots")
                self._list_names(profile / "manifests")
            except SnapshotNotFoundError:
                continue
            for name in names:
                match = re.fullmatch(r"\.pending-([A-Za-z0-9][A-Za-z0-9_-]{0,127})\.json", name)
                if match is None:
                    continue
                marker = profile / "snapshots" / name
                try:
                    pending = self._read_model(marker, _PendingBinary)
                except (_MalformedManifestError, ValidationError, SnapshotNotFoundError):
                    continue
                if self._exists(profile / "manifests" / f"{match[1]}.json"):
                    continue
                identity = (pending.device, pending.inode)
                candidates = [profile / "snapshots" / f"{match[1]}.bin", profile / "snapshots" / pending.temporary_name]
                safe = True
                for candidate in candidates:
                    try:
                        fd = self._open_regular_readonly(candidate)
                    except SnapshotNotFoundError:
                        continue
                    try:
                        info = os.fstat(fd)
                        if (info.st_dev, info.st_ino) != identity:
                            safe = False
                    finally:
                        os.close(fd)
                if safe:
                    for candidate in candidates:
                        self._remove_matching_file(candidate, identity)
                    self._remove_pending_marker(marker)
