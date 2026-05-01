"""Process-local backend concurrency gates for VN asset generation."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from threading import Lock

LOCAL_IMAGE_BACKENDS = {
    "automatic1111",
    "comfyui",
    "invokeai",
    "stable_diffusion_cpp",
    "stable-diffusion-cpp",
    "swarmui",
}


@dataclass
class BackendGenerationLease:
    acquired: bool
    backend: str
    model: str | None = None
    _release_callback: object | None = None
    _released: bool = False

    def release(self) -> None:
        if not self.acquired or self._released or not callable(self._release_callback):
            return
        self._released = True
        self._release_callback()

    def __enter__(self) -> BackendGenerationLease:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.release()


class BackendGenerationGate:
    """Small in-process limit map keyed by backend name."""

    def __init__(
        self,
        *,
        default_local_limit: int | None = None,
        default_remote_limit: int | None = None,
    ) -> None:
        self.default_local_limit = default_local_limit if default_local_limit is not None else _env_int(
            "VN_ASSETS_LOCAL_BACKEND_CONCURRENCY",
            1,
        )
        self.default_remote_limit = default_remote_limit if default_remote_limit is not None else _env_int(
            "VN_ASSETS_REMOTE_BACKEND_CONCURRENCY",
            4,
        )
        self._active_by_backend: dict[str, int] = {}
        self._lock = Lock()

    def try_acquire(self, backend: str, *, model: str | None = None) -> BackendGenerationLease:
        normalized_backend = _normalize_backend(backend)
        with self._lock:
            active = self._active_by_backend.get(normalized_backend, 0)
            limit = self._limit_for_backend(normalized_backend)
            if active >= limit:
                return BackendGenerationLease(
                    acquired=False,
                    backend=normalized_backend,
                    model=model,
                )
            self._active_by_backend[normalized_backend] = active + 1

        return BackendGenerationLease(
            acquired=True,
            backend=normalized_backend,
            model=model,
            _release_callback=lambda: self._release(normalized_backend),
        )

    def _release(self, backend: str) -> None:
        with self._lock:
            active = self._active_by_backend.get(backend, 0)
            if active <= 1:
                self._active_by_backend.pop(backend, None)
            else:
                self._active_by_backend[backend] = active - 1

    def _limit_for_backend(self, backend: str) -> int:
        override = _env_int(_backend_override_env_name(backend), 0)
        if override > 0:
            return override
        if backend in LOCAL_IMAGE_BACKENDS:
            return max(1, self.default_local_limit)
        return max(1, self.default_remote_limit)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _normalize_backend(backend: str) -> str:
    return str(backend or "").strip().lower().replace("-", "_") or "default"


def _backend_override_env_name(backend: str) -> str:
    key = re.sub(r"[^A-Z0-9]+", "_", backend.upper()).strip("_")
    return f"VN_ASSETS_BACKEND_CONCURRENCY_{key or 'DEFAULT'}"


_DEFAULT_BACKEND_GENERATION_GATE: BackendGenerationGate | None = None
_DEFAULT_BACKEND_GENERATION_GATE_LOCK = Lock()


def get_default_backend_generation_gate() -> BackendGenerationGate:
    """Return the process-local default generation gate."""
    global _DEFAULT_BACKEND_GENERATION_GATE
    with _DEFAULT_BACKEND_GENERATION_GATE_LOCK:
        if _DEFAULT_BACKEND_GENERATION_GATE is None:
            _DEFAULT_BACKEND_GENERATION_GATE = BackendGenerationGate()
    return _DEFAULT_BACKEND_GENERATION_GATE
