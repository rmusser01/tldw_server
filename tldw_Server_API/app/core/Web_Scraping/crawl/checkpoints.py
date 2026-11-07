"""Private, bounded continuation files with a persistent native writer lock.

Files are not authorization grants. The caller supplies the authorized directory
and the live plan; legacy local files explicitly have incomplete accounting.
"""

from __future__ import annotations

import json
import math
import os
import stat
import threading
import weakref
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path, PureWindowsPath
from typing import Any
from urllib.parse import parse_qsl, urlsplit
from uuid import uuid4

from ...Infrastructure.distributed_lock import _acquire_platform_file_lock, _release_platform_file_lock
from ..contracts.execution import COMPLETION_REASONS, INTERRUPTION_REASONS, _private_article_key
from ..jobs.admission import bounded_snapshot
from ..runtime.controls import COUNTERS, BudgetSnapshot
from .frontier import Frontier
from .models import CheckpointLimits, CheckpointScope, CheckpointState, CrawlPlan, FrontierEntry, FrontierSnapshot

_DESCRIPTOR_PATHS = os.open in os.supports_dir_fd and hasattr(os, "O_NOFOLLOW")
# Wire version 1 is explicit; future dataclass defaults must not replenish usage.
_STATE_FIELDS = frozenset(
    {"schema_version", "scope", "binding", "frontier", "budget", "stop_reason", "accounting_complete", "pages_scraped"}
)
_BUDGET_FIELDS = frozenset(
    {
        "page_attempts",
        "network_requests",
        "browser_attempts",
        "active_probes",
        "transfer_bytes",
        "discovered_candidates",
        "discovered_url_bytes",
        "remaining_seconds",
        "accounting_complete",
    }
)
_FRONTIER_FIELDS = frozenset({"queued", "inflight", "completed", "results", "visited"})
_ENTRY_FIELDS = frozenset({"url", "parent_url", "depth", "sequence", "score"})


class CheckpointError(ValueError):
    """Public-safe machine code only; underlying paths/data never cross boundary."""

    def __init__(self, code: str = "checkpoint_failed") -> None:
        self.code = code
        super().__init__(code)


class _CheckpointRegistry:
    """Own the local writer index and its fork-sensitive synchronization."""

    def __init__(self) -> None:
        self.owners: dict[tuple[int, int, str], CheckpointStore] = {}
        self.lock = threading.Lock()
        self.stores: weakref.WeakSet = weakref.WeakSet()

    def after_fork(self) -> None:
        # Never unlock an inherited open-file description: that unlocks the
        # parent's writer too. Child copies may only close their descriptors.
        for store in tuple(self.stores):
            store._mutex = threading.RLock()
            for name in ("_lock_fd", "_directory_fd"):
                fd = getattr(store, name, None)
                if fd is not None:
                    os.close(fd)
                    setattr(store, name, None)
        self.owners.clear()
        self.lock = threading.Lock()


_registry = _CheckpointRegistry()
_after_fork = _registry.after_fork


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork)


def _integer(value: Any, maximum: int) -> int:
    if type(value) is not int or not 0 <= value <= maximum:
        raise CheckpointError()
    return value


def _wire_object(value: Any, required: frozenset[str]) -> dict[str, Any]:
    """Reject missing/unknown wire fields before internal model construction."""
    if type(value) is not dict or value.keys() != required:
        raise CheckpointError()
    return value


def _url(value: Any, plan: CrawlPlan) -> str:
    if (
        type(value) is not str
        or not value
        or len(value) > plan.limits.max_url_bytes
        or len(value.encode("utf-8")) > plan.limits.max_url_bytes
    ):
        raise CheckpointError()
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise CheckpointError()
    sensitive = {"key", "sig", "signature", "credential", "auth"}
    if (
        parsed.username is not None
        or parsed.password is not None
        or any(
            _private_article_key(key)
            or key.lower() in sensitive
            or key.lower().endswith(("signature", "credential", "security-token"))
            for key, _ in parse_qsl(parsed.query, keep_blank_values=True)
            + parse_qsl(parsed.fragment, keep_blank_values=True)
        )
    ):
        # Do not resume a different resource by silently rewriting execution URL
        # identity. Credentials must be reacquired through live composition.
        raise CheckpointError()
    return value


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    return value


def _binding(plan: CrawlPlan) -> dict:
    _url(plan.request.base_url, plan)
    # Local storage location is not execution behavior and must never be JSON.
    options = {key: _plain(value) for key, value in plan.options.items() if key != "resume_file"}
    binding = {
        "seed": plan.request.base_url,
        "profile": plan.profile,
        "options": options,
        "provenance": {key: value for key, value in plan.provenance.items() if key != "resume_file"},
    }
    bounded_snapshot(binding, 134_217_728)
    return binding


def _article(value: Any, limit: int) -> dict:
    if type(value) is not dict:
        raise CheckpointError()

    def without_paths(item, *, prose=False):
        if type(item) is dict:
            return {
                key: without_paths(child, prose=item is value and key in {"content", "raw_content", "raw_html"})
                for key, child in item.items()
                if type(key) is str
                and not "".join(c for c in key.lower() if c.isalnum()).endswith(
                    ("path", "paths", "directory", "resume_file")
                )
            }
        if type(item) in (list, tuple):
            return [without_paths(child) for child in item]
        if type(item) is str and not prose and (Path(item).is_absolute() or PureWindowsPath(item).is_absolute()):
            return "[private-path]"
        return item

    # Existing structured projection retains direct intentional article prose
    # but strips execution keys, URL credentials and arbitrary diagnostics.
    result = bounded_snapshot(without_paths(value), limit, redact=True, article_path=())[0]
    if type(value.get("error")) is str and value["error"] in INTERRUPTION_REASONS | {"browser_transport_unavailable"}:
        result["error"] = value["error"]
    return result


def _pairs(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise CheckpointError()
        result[key] = value
    return result


def _nonfinite(_value: str) -> None:
    raise CheckpointError()


class CheckpointStore:
    """Synchronous structural port; close serializes behind actual file operations.

    POSIX service operations are relative to a pinned, no-follow directory fd.
    Platforms lacking descriptor-relative no-follow operations fail closed for
    service paths; explicitly trusted local mode still uses native file locks.
    """

    scope: CheckpointScope
    limits: CheckpointLimits

    def __init__(self) -> None:
        self._mutex = threading.RLock()
        self._pid = os.getpid()
        self._directory_fd: int | None = None
        self._lock_fd: int | None = None
        self._locked = False
        self._key = None
        self._closed = False

    @classmethod
    def open(cls, path: Path, *, scope: CheckpointScope, limits: CheckpointLimits) -> CheckpointStore:
        """Acquire immediate same-process and native exclusive writer ownership."""
        store = cls()
        store.scope, store.limits = scope, limits
        try:
            if any(type(value) is not str or not value for value in (scope.owner, scope.namespace)):
                raise CheckpointError()
            if type(scope.trusted_local_path) is not bool:
                raise CheckpointError()
            path = Path(path)
            if scope.trusted_local_path:
                path = path.absolute()
                parent = path.parent.resolve(strict=True)
                parts = ()
            else:
                if scope.root is None or not _DESCRIPTOR_PATHS:
                    raise CheckpointError()
                # Only the composition-owned root may resolve platform aliases.
                root = Path(scope.root).absolute()
                parent = root.resolve(strict=True)
                if path.is_absolute():
                    try:
                        relative = path.relative_to(root)
                    except ValueError:
                        relative = path.relative_to(parent)
                else:
                    relative = path
                if not relative.parts or any(part in {"..", "."} for part in relative.parts):
                    raise CheckpointError()
                parts = relative.parts[:-1]
                path = parent / relative
            store._name = path.name
            if not store._name or store._name in {".", ".."}:
                raise CheckpointError()
            if _DESCRIPTOR_PATHS:
                fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
                store._directory_fd = fd
                store._root, store._parts = parent, parts
                store._root_identity = os.fstat(fd)
                for part in parts:
                    next_fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
                    os.close(fd)
                    fd = store._directory_fd = next_fd
                parent_stat = os.fstat(fd)
            else:
                parent_stat = parent.stat()
            store._parent = path.parent
            store._key = (parent_stat.st_dev, parent_stat.st_ino, store._name)
            with _registry.lock:
                if store._key in _registry.owners:
                    raise CheckpointError("checkpoint_in_use")
                _registry.owners[store._key] = store
                _registry.stores.add(store)
            store._check_target(store._name)
            store._lock_fd = store._open_file(store._name + ".lock", os.O_CREAT | os.O_RDWR)
            try:
                _acquire_platform_file_lock(store._lock_fd)
            except OSError:
                raise CheckpointError("checkpoint_in_use") from None
            store._locked = True
            store._check_file(store._lock_fd, store._name + ".lock")
            return store
        except Exception as exc:  # noqa: BLE001 - no path-bearing OS exceptions
            store.close()
            raise CheckpointError(exc.code if isinstance(exc, CheckpointError) else "checkpoint_failed") from None

    def _active(self) -> None:
        if self._closed or os.getpid() != self._pid:
            raise CheckpointError()
        if self._directory_fd is not None:
            fd = os.open(self._root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                root = os.fstat(fd)
                if (root.st_dev, root.st_ino) != (self._root_identity.st_dev, self._root_identity.st_ino):
                    raise CheckpointError()
                for part in self._parts:
                    next_fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
                    os.close(fd)
                    fd = next_fd
                current, pinned = os.fstat(fd), os.fstat(self._directory_fd)
                if (current.st_dev, current.st_ino) != (pinned.st_dev, pinned.st_ino):
                    raise CheckpointError()
            finally:
                os.close(fd)
        if self._lock_fd is not None:
            self._check_file(self._lock_fd, self._name + ".lock")

    def _stat(self, name: str):
        if self._directory_fd is not None:
            return os.stat(name, dir_fd=self._directory_fd, follow_symlinks=False)
        return (self._parent / name).lstat()

    def _check_target(self, name: str) -> None:
        try:
            info = self._stat(name)
        except FileNotFoundError:
            return
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise CheckpointError()

    def _check_file(self, fd: int, name: str) -> None:
        info, current = os.fstat(fd), self._stat(name)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
            or (info.st_dev, info.st_ino) != (current.st_dev, current.st_ino)
        ):
            raise CheckpointError()

    def _open_file(self, name: str, flags: int) -> int:
        self._check_target(name)
        flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        if self._directory_fd is not None:
            fd = os.open(name, flags, 0o600, dir_fd=self._directory_fd)
        else:
            fd = os.open(self._parent / name, flags, 0o600)
        try:
            self._check_file(fd, name)
        except BaseException:
            os.close(fd)
            raise
        return fd

    def _unlink(self, name: str) -> None:
        if self._directory_fd is not None:
            os.unlink(name, dir_fd=self._directory_fd)
        else:
            os.unlink(self._parent / name)

    def _validate(self, state: CheckpointState) -> None:
        if state.scope != self.scope or type(state.schema_version) is not int or state.schema_version != 1:
            raise CheckpointError()
        if type(state.accounting_complete) is not bool or type(state.budget.accounting_complete) is not bool:
            raise CheckpointError()
        if state.stop_reason is not None and state.stop_reason not in COMPLETION_REASONS | INTERRUPTION_REASONS:
            raise CheckpointError()
        if state.pages_scraped is not None:
            _integer(state.pages_scraped, min(self.limits.max_results, state.plan.limits.page_attempts))
        for name in COUNTERS:
            _integer(getattr(state.budget, name), getattr(state.plan.limits, name))
        remaining = state.budget.remaining_seconds
        if (
            type(remaining) not in (int, float)
            or not math.isfinite(remaining)
            or not 0 <= remaining <= state.plan.limits.execution_seconds
        ):
            raise CheckpointError()
        snapshot, limits = state.frontier, state.plan.limits
        if (
            len(snapshot.queued) + len(snapshot.inflight) > min(self.limits.max_frontier, limits.frontier_entries)
            or len(snapshot.completed) > min(self.limits.max_candidates, limits.discovered_candidates)
            or len(snapshot.results) > min(self.limits.max_results, limits.page_attempts)
            or len(snapshot.visited) > min(self.limits.max_candidates, limits.discovered_candidates)
        ):
            raise CheckpointError()
        total, sequences = 0, set()
        for item in snapshot.queued + snapshot.inflight + snapshot.completed:
            for url in (item.url, item.parent_url):
                if url is not None:
                    total += len(_url(url, state.plan).encode("utf-8"))
            _integer(item.depth, self.limits.max_candidates)
            _integer(item.sequence, self.limits.max_candidates + self.limits.max_frontier)
            if item.sequence in sequences or type(item.score) not in (int, float) or not math.isfinite(item.score):
                raise CheckpointError()
            sequences.add(item.sequence)
        for url in snapshot.visited:
            total += len(_url(url, state.plan).encode("utf-8"))
        if total > limits.discovered_url_bytes or len(set(snapshot.visited)) != len(snapshot.visited):
            raise CheckpointError()
        Frontier.from_snapshot(state.plan, snapshot)
        _binding(state.plan)

    def _encode(self, state: CheckpointState) -> dict:
        self._validate(state)
        snapshot = state.frontier
        return {
            "schema_version": 1,
            "scope": {"owner": self.scope.owner, "namespace": self.scope.namespace},
            "binding": _binding(state.plan),
            "frontier": {
                "queued": [asdict(item) for item in snapshot.queued],
                "inflight": [asdict(item) for item in snapshot.inflight],
                "completed": [asdict(item) for item in snapshot.completed],
                "visited": list(snapshot.visited),
                "results": [_article(item, self.limits.max_file_bytes) for item in snapshot.results],
            },
            "budget": asdict(state.budget),
            "stop_reason": state.stop_reason,
            "accounting_complete": state.accounting_complete,
            "pages_scraped": state.pages_scraped,
        }

    def _decode(self, data: dict, plan: CrawlPlan) -> CheckpointState:
        if type(data) is not dict:
            raise CheckpointError()
        if "schema_version" not in data:
            return self._legacy(data, plan)
        _wire_object(data, _STATE_FIELDS)
        if type(data["schema_version"]) is not int or data["schema_version"] != 1:
            raise CheckpointError()
        if data["scope"] != {"owner": self.scope.owner, "namespace": self.scope.namespace} or json.dumps(
            data["binding"], sort_keys=True, allow_nan=False
        ) != json.dumps(_binding(plan), sort_keys=True, allow_nan=False):
            raise CheckpointError()
        budget = _wire_object(data["budget"], _BUDGET_FIELDS)
        raw = _wire_object(data["frontier"], _FRONTIER_FIELDS)
        collections = {}
        for name, maximum in (
            ("queued", self.limits.max_frontier),
            ("inflight", self.limits.max_frontier),
            ("completed", self.limits.max_candidates),
            ("results", self.limits.max_results),
            ("visited", self.limits.max_candidates),
        ):
            values = raw[name]
            if type(values) is not list or len(values) > maximum:
                raise CheckpointError()
            collections[name] = (
                tuple(FrontierEntry(**_wire_object(item, _ENTRY_FIELDS)) for item in values)
                if name in {"queued", "inflight", "completed"}
                else tuple(values)
            )
        collections["results"] = tuple(_article(item, self.limits.max_file_bytes) for item in collections["results"])
        state = CheckpointState(
            self.scope,
            plan,
            FrontierSnapshot(**collections),
            BudgetSnapshot(**budget),
            data["stop_reason"],
            data["accounting_complete"],
            pages_scraped=data["pages_scraped"],
        )
        self._validate(state)
        return state

    def _legacy(self, data: dict, plan: CrawlPlan) -> CheckpointState:
        if plan.profile != "legacy_recursive" or set(data) != {
            "visited",
            "to_visit",
            "scraped_articles",
            "pages_scraped",
        }:
            raise CheckpointError()
        pages = _integer(data["pages_scraped"], min(self.limits.max_results, plan.limits.page_attempts))
        visited, queued, articles = (data[name] for name in ("visited", "to_visit", "scraped_articles"))
        for values, maximum in (
            (visited, self.limits.max_candidates),
            (queued, self.limits.max_frontier),
            (articles, self.limits.max_results),
        ):
            if type(values) is not list or len(values) > maximum:
                raise CheckpointError()
        completed = tuple(
            FrontierEntry(_url(item["url"], plan), None, 0, index, 0.0) for index, item in enumerate(articles)
        )
        pending, seen = [], set(visited)
        for url, depth in queued:
            _url(url, plan)
            _integer(depth, self.limits.max_candidates)
            if url not in seen:
                pending.append(FrontierEntry(url, None, depth, len(completed) + len(pending), 0.0))
                seen.add(url)
        pending = tuple(pending)
        state = CheckpointState(
            self.scope,
            plan,
            FrontierSnapshot(pending, (), completed, tuple(articles), tuple(visited)),
            BudgetSnapshot(0, 0, 0, 0, 0, 0, plan.limits.execution_seconds, False),
            accounting_complete=False,
            pages_scraped=pages,
        )
        self._validate(state)
        return CheckpointState(
            self.scope,
            plan,
            FrontierSnapshot(
                pending,
                (),
                completed,
                tuple(_article(item, self.limits.max_file_bytes) for item in articles),
                tuple(visited),
            ),
            state.budget,
            accounting_complete=False,
            pages_scraped=state.pages_scraped,
        )

    def load(self, *, plan: CrawlPlan, scope: CheckpointScope) -> CheckpointState | None:
        """Read limit+1 bytes at most before strict JSON and binding validation."""
        with self._mutex:
            try:
                self._active()
                if scope != self.scope:
                    raise CheckpointError()
                try:
                    fd = self._open_file(self._name, os.O_RDONLY)
                except FileNotFoundError:
                    return None
                with os.fdopen(fd, "rb") as stream:
                    raw = stream.read(self.limits.max_file_bytes + 1)
                if len(raw) > self.limits.max_file_bytes:
                    raise CheckpointError()
                return self._decode(json.loads(raw, object_pairs_hook=_pairs, parse_constant=_nonfinite), plan)
            except Exception:  # noqa: BLE001 - contain untrusted parser/OS details
                raise CheckpointError() from None

    def save(self, state: CheckpointState) -> None:
        """Stream strict JSON to a unique exclusive temporary file, then replace."""
        with self._mutex:
            temporary = None
            try:
                self._active()
                payload = self._encode(state)
                temporary = ".checkpoint-" + uuid4().hex + ".tmp"
                fd = self._open_file(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
                with os.fdopen(fd, "wb") as stream:
                    total = 0
                    for chunk in json.JSONEncoder(ensure_ascii=True, allow_nan=False, separators=(",", ":")).iterencode(
                        payload
                    ):
                        # Avoid retaining a second large encoded article string.
                        for start in range(0, len(chunk), 65536):
                            block = chunk[start : start + 65536].encode("ascii")
                            total += len(block)
                            if total > self.limits.max_file_bytes:
                                raise CheckpointError()
                            stream.write(block)
                    stream.flush()
                    os.fsync(stream.fileno())
                    self._check_file(stream.fileno(), temporary)
                self._check_target(self._name)
                self._check_file(self._lock_fd, self._name + ".lock")
                if self._directory_fd is not None:
                    os.replace(temporary, self._name, src_dir_fd=self._directory_fd, dst_dir_fd=self._directory_fd)
                else:
                    os.replace(self._parent / temporary, self._parent / self._name)
                temporary = None
            except Exception:  # noqa: BLE001 - preserve prior file and public-safe error
                raise CheckpointError() from None
            finally:
                if temporary is not None:
                    try:
                        self._unlink(temporary)
                    except FileNotFoundError:
                        pass
                    except OSError:
                        raise CheckpointError() from None

    def clear(self) -> None:
        """Remove only the continuation, never the stable lock inode."""
        with self._mutex:
            try:
                self._active()
                self._check_target(self._name)
                self._check_file(self._lock_fd, self._name + ".lock")
                try:
                    self._unlink(self._name)
                except FileNotFoundError:
                    pass
            except Exception:  # noqa: BLE001 - no filesystem details in diagnostics
                raise CheckpointError() from None

    def close(self) -> None:
        """Release writer ownership after all synchronous operations really exit."""
        if os.getpid() != self._pid:
            return
        with self._mutex:
            if self._closed:
                return
            self._closed = True
            failed = False
            try:
                if self._lock_fd is not None:
                    try:
                        if self._locked:
                            _release_platform_file_lock(self._lock_fd)
                    except OSError:
                        failed = True
                    finally:
                        fd, self._lock_fd = self._lock_fd, None
                        try:
                            os.close(fd)
                        except OSError:
                            failed = True
            finally:
                if self._directory_fd is not None:
                    fd, self._directory_fd = self._directory_fd, None
                    try:
                        os.close(fd)
                    except OSError:
                        failed = True
                with _registry.lock:
                    if _registry.owners.get(self._key) is self:
                        del _registry.owners[self._key]
            if failed:
                raise CheckpointError() from None
