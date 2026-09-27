"""Content fingerprint helpers for managed llama.cpp snapshots."""

from __future__ import annotations

import hashlib
import json
import os
from collections import OrderedDict
from pathlib import Path
from threading import Lock

from tldw_Server_API.app.core.exceptions import UnstableFingerprintError

from .llamacpp_snapshot_models import Fingerprint

_CHUNK_SIZE = 1024 * 1024
_FileIdentity = tuple[int, int, int, int, int]


class FingerprintHashCache:
    """Bounded, service-owned LRU of hashes verified against stable file metadata.

    The lock covers reads as well as lookups so concurrent polling cannot stream
    the same large file repeatedly. Each hit still verifies the opened file and
    pathname before and after lookup; failed or unstable reads are never added.
    """

    def __init__(self, max_entries: int = 32) -> None:
        """Create a cache retaining at most ``max_entries`` file identities."""
        if max_entries < 1:
            raise ValueError("max_entries must be positive")
        self._max_entries = max_entries
        self._entries: OrderedDict[_FileIdentity, str] = OrderedDict()
        self._lock = Lock()

    def hash_file(self, path: Path) -> str:
        """Return a verified digest, streaming only an uncached file identity."""
        with self._lock:
            digest, identity = _hash_file_stable(path, self._entries)
            self._entries[identity] = digest
            self._entries.move_to_end(identity)
            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)
            return digest


def compare_fingerprints(saved: Fingerprint, current: Fingerprint | None) -> list[str]:
    """Return every mismatched identity field, failing closed when unknown."""
    if current is None:
        return ["compatibility_unknown"]
    return [name for name in type(saved).model_fields if getattr(saved, name) != getattr(current, name)]


def hash_file_stable(path: Path) -> str:
    """Hash a regular file and reject identity changes during the read."""
    return _hash_file_stable(path)[0]


def _hash_file_stable(
    path: Path,
    entries: OrderedDict[_FileIdentity, str] | None = None,
) -> tuple[str, _FileIdentity]:
    """Validate file identity around streaming or reuse of a known digest."""
    no_follow = getattr(os, "O_NOFOLLOW", 0)
    if not no_follow:
        raise UnstableFingerprintError("no-follow file identity checks are unsupported")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | no_follow
    fd = os.open(path, flags)
    try:
        before = os.fstat(fd)
        path_before = os.stat(path, follow_symlinks=False)
        if not _is_regular(before.st_mode):
            raise ValueError("fingerprint source must be a regular file")
        identity_before = _identity(before)
        if identity_before != _identity(path_before):
            raise UnstableFingerprintError("fingerprint source changed before hashing")
        result = entries.get(identity_before) if entries is not None else None
        if result is None:
            digest = hashlib.sha256()
            while chunk := os.read(fd, _CHUNK_SIZE):
                digest.update(chunk)
            result = digest.hexdigest()
        after = os.fstat(fd)
        path_after = os.stat(path, follow_symlinks=False)
        if (
            identity_before != _identity(path_before)
            or identity_before != _identity(after)
            or identity_before != _identity(path_after)
        ):
            raise UnstableFingerprintError("fingerprint source changed while hashing")
        return result, identity_before
    finally:
        os.close(fd)


def canonical_sha256(value: object) -> str:
    """Hash canonical JSON for effective options and adapter descriptors."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_fingerprint(
    *,
    model: Path,
    executable: Path,
    effective_options: object,
    adapters: object,
    projector: Path | None = None,
    cache: FingerprintHashCache | None = None,
) -> Fingerprint:
    """Build content identity, optionally reusing a caller-owned stable hash cache."""
    hash_file = cache.hash_file if cache is not None else hash_file_stable
    return Fingerprint(
        model_sha256=hash_file(model),
        executable_sha256=hash_file(executable),
        projector_sha256=hash_file(projector) if projector is not None else None,
        effective_options_sha256=canonical_sha256(effective_options),
        adapters_sha256=canonical_sha256(adapters),
    )


def _is_regular(mode: int) -> bool:
    import stat

    return stat.S_ISREG(mode)


def _identity(info: os.stat_result) -> _FileIdentity:
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
