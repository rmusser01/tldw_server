"""Per-workspace session cache for LSP backend processes."""

from __future__ import annotations

import inspect
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, fields
from pathlib import Path

from .config import DEFAULT_LSP_CONFIG, LspRuntimeConfig

BackendFactory = Callable[[str, Path, LspRuntimeConfig], object | Awaitable[object]]
SessionKey = tuple[str, str, tuple[tuple[str, object], ...]]


@dataclass(slots=True)
class LspManagedSession:
    """Cached backend object plus session identity and idle metadata."""

    backend_id: str
    workspace_root: Path
    backend: object
    config_fingerprint: tuple[tuple[str, object], ...]
    created_monotonic: float
    last_used_monotonic: float

    def touch(self, *, now: float | None = None) -> None:
        self.last_used_monotonic = time.monotonic() if now is None else now

    async def close(self) -> None:
        close = getattr(self.backend, "close", None)
        if close is None:
            return
        result = close()
        if inspect.isawaitable(result):
            await result


class LspSessionManager:
    """Cache LSP backend sessions by backend id and canonical workspace root."""

    def __init__(
        self,
        *,
        config: LspRuntimeConfig = DEFAULT_LSP_CONFIG,
        backend_factory: BackendFactory,
    ):
        self.config = config
        self._backend_factory = backend_factory
        self._sessions: dict[SessionKey, LspManagedSession] = {}
        self._config_fingerprint = _config_fingerprint(config)

    @property
    def active_session_count(self) -> int:
        return len(self._sessions)

    async def get_session(self, backend_id: str, *, workspace_root: Path) -> LspManagedSession:
        canonical_root = workspace_root.resolve()
        key = self._session_key(backend_id, canonical_root)
        existing = self._sessions.get(key)
        if existing is not None:
            existing.touch()
            return existing

        backend = self._backend_factory(backend_id, canonical_root, self.config)
        if inspect.isawaitable(backend):
            backend = await backend
        now = time.monotonic()
        session = LspManagedSession(
            backend_id=backend_id,
            workspace_root=canonical_root,
            backend=backend,
            config_fingerprint=self._config_fingerprint,
            created_monotonic=now,
            last_used_monotonic=now,
        )
        self._sessions[key] = session
        return session

    async def stop_all(self) -> dict[str, str]:
        """Close all sessions, continuing after individual close failures."""

        errors: dict[str, str] = {}
        for key, session in list(self._sessions.items()):
            try:
                await session.close()
            except Exception as exc:  # noqa: BLE001
                # Backend close implementations are plugin code; stop_all must sweep every session.
                errors[session.backend_id] = exc.__class__.__name__
            finally:
                self._sessions.pop(key, None)
        return errors

    async def evict_idle_sessions(self, *, now: float | None = None) -> dict[str, str]:
        """Close sessions idle for longer than the configured idle TTL."""

        cutoff_now = time.monotonic() if now is None else now
        errors: dict[str, str] = {}
        for key, session in list(self._sessions.items()):
            if cutoff_now - session.last_used_monotonic <= self.config.idle_ttl_seconds:
                continue
            try:
                await session.close()
            except Exception as exc:  # noqa: BLE001
                # Idle eviction should remove every expired session even if one backend close fails.
                errors[session.backend_id] = exc.__class__.__name__
            finally:
                self._sessions.pop(key, None)
        return errors

    def _session_key(self, backend_id: str, workspace_root: Path) -> SessionKey:
        return (backend_id, str(workspace_root), self._config_fingerprint)


def _config_fingerprint(config: LspRuntimeConfig) -> tuple[tuple[str, object], ...]:
    return tuple((field.name, getattr(config, field.name)) for field in fields(config))
