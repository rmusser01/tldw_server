"""Singleton / lifecycle isolation guard.

Detects process-global state that one test module leaves mutated for the
*next* module to inherit — the failure class behind the audit's top-severity
defects (audits/2026-07-04-test-suite-audit-round2.md, RA5):

* #2580 — service-layer singleton caches registered against the wrong DB
* #2581 — Embeddings/TTS drain state corrupting subsequent suites
* #2585 — ``reload_app_main()`` swapping the cached app-module identity

Mechanism (deliberately different from ``http_client_patch_guard``, which
intercepts at patch time): a curated watchlist of known-dangerous globals is
snapshotted at every test-module boundary. If a module ends with a watched
global changed from how that module started it (and the change persists into
the next module), the guard reports a leak, attributing it to the module that
caused it.

Only modules already present in ``sys.modules`` are inspected — the guard
never force-imports anything, so it is cheap and free of import side effects.
A module that a lane never loads simply has nothing to leak.

Activation (opt-in) via the ``TLDW_SINGLETON_GUARD`` env var:

* unset / ``"off"`` — disabled (default)
* ``"warn"``        — emit a warning per leak (does not fail the run)
* ``"error"``       — additionally force a non-zero session exit

Register via ``pytest_plugins`` in ``tldw_Server_API/tests/conftest.py`` (the
``http_client_patch_guard`` precedent) — NOT via the ``plugins`` key in
``pyproject.toml``, which is not a real pytest option and is silently ignored.
"""
from __future__ import annotations

import os
import sys
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pytest


@dataclass(frozen=True)
class WatchedGlobal:
    """One process-global whose state the guard tracks across module boundaries."""

    label: str
    module: str
    #: Reads a cheap, comparable snapshot from the imported module (int count,
    #: bool presence, frozenset of keys, ...). Must never mutate.
    reader: Callable[[Any], object]
    #: Reason a change here is dangerous (shown in the leak report).
    why: str


def _len_attr(attr: str) -> Callable[[Any], object]:
    """Reader: ``len(getattr(module, attr))`` when present, else None."""

    def read(module: Any) -> object:
        target = getattr(module, attr, None)
        if target is None:
            return None
        try:
            return len(target)
        except TypeError:
            return repr(target)

    return read


def _is_set_attr(attr: str) -> Callable[[Any], object]:
    """Reader: whether ``module.attr`` is currently non-None (singleton set)."""

    def read(module: Any) -> object:
        return getattr(module, attr, None) is not None

    return read


def _call_len(func_name: str, cache_attr: str = "cache_info") -> Callable[[Any], object]:
    """Reader: current size of an ``@lru_cache``-wrapped function."""

    def read(module: Any) -> object:
        func = getattr(module, func_name, None)
        info = getattr(func, cache_attr, None)
        if not callable(info):
            return None
        try:
            return info().currsize
        except Exception:
            return None

    return read


def _identity_attr(attr: str) -> Callable[[Any], object]:
    """Reader: ``id(module.attr)`` when set, else None.

    Detects a same-named object being *rebound* to a different instance (a DB
    backend swap) across a boundary, which ``_is_set_attr`` would miss.
    """

    def read(module: Any) -> object:
        target = getattr(module, attr, None)
        return id(target) if target is not None else None

    return read


def _module_self_id(module: Any) -> object:
    """Reader: ``id`` of the watched module object itself.

    Detects ``reload_app_main()`` swapping the app-main module in
    ``sys.modules`` (#2585) — a changed id means other references are stale.
    """
    return id(module)


# --------------------------------------------------------------------------- #
# Curated watchlist — the known-dangerous globals from the audit inventory.
# Ordered by defect adjacency. Add entries as new leak sources are found; each
# should read a comparable snapshot cheaply and never mutate.
# --------------------------------------------------------------------------- #
WATCHLIST: list[WatchedGlobal] = [
    # 1. Content DB backend — #2580 epicenter. A rebind (identity change)
    #    across a boundary means a DB swap leaked into the next suite.
    WatchedGlobal(
        label="content_db_backend",
        module="tldw_Server_API.app.core.DB_Management.media_db.runtime.defaults",
        reader=_identity_attr("content_db_backend"),
        why="content DB backend rebound without reset_media_runtime_defaults()",
    ),
    # 2. Per-user ChaCha DB cache — #2580. Live handles bound to USER_DB_BASE_DIR
    #    carried across a boundary return wrong-DB handles.
    WatchedGlobal(
        label="_chacha_db_instances",
        module="tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps",
        reader=_len_attr("_chacha_db_instances"),
        why="per-user ChaChaNotesDB handles left cached across suites",
    ),
    # 3. App-main module identity — #2585. reload_app_main() swaps this module;
    #    a changed id means references held elsewhere are now stale.
    WatchedGlobal(
        label="app.main module identity",
        module="tldw_Server_API.app.main",
        reader=_module_self_id,
        why="reload_app_main() swapped sys.modules['...app.main']",
    ),
    # 4. RAG shared semantic caches — #2580. A cache bound to a prior test's
    #    DB/namespace staying visible to the next.
    WatchedGlobal(
        label="_SHARED_CACHES",
        module="tldw_Server_API.app.core.RAG.rag_service.semantic_cache",
        reader=_len_attr("_SHARED_CACHES"),
        why="shared semantic caches left populated across suites",
    ),
    # 5-7. Embeddings drain singletons — #2581. None->set (or lingering after a
    #      suite drained them) bleeds drained lifecycle state into the next.
    WatchedGlobal(
        label="_pool_manager",
        module="tldw_Server_API.app.core.Embeddings.connection_pool",
        reader=_is_set_attr("_pool_manager"),
        why="embeddings connection-pool manager left live/drained across suites",
    ),
    WatchedGlobal(
        label="_async_service_fallback",
        module="tldw_Server_API.app.core.Embeddings.async_embeddings",
        reader=_is_set_attr("_async_service_fallback"),
        why="async embedding service singleton left live across suites",
    ),
    WatchedGlobal(
        label="_batcher_fallback",
        module="tldw_Server_API.app.core.Embeddings.request_batching",
        reader=_is_set_attr("_batcher_fallback"),
        why="request batcher singleton left live/drained across suites",
    ),
]


def _is_regression(start: object, end: object) -> bool:
    """True when *end* holds MORE/DIFFERENT live state than *start*.

    Only the dangerous direction is a leak — a module *clearing* its state
    (``True -> False``, a shrinking count, ``set -> None``) is good hygiene,
    not pollution:

    * bool (is-set readers): ``False -> True`` is the leak (became live)
    * int (len readers): growth is the leak
    * identity ids / mixed None: becoming set or being rebound to a different
      object is the leak; becoming ``None`` (cleared) is safe
    """
    if start == end:
        return False
    # bool is a subclass of int — check it first.
    if isinstance(start, bool) or isinstance(end, bool):
        return bool(end) and not bool(start)
    if isinstance(start, int) and isinstance(end, int):
        return end > start
    # identity ids (or a None<->value transition): cleared is safe, set/rebind leaks
    if end is None:
        return False
    return start != end


@dataclass
class _ModuleState:
    name: str
    start: dict[str, object] = field(default_factory=dict)


class SingletonGuard:
    """Snapshots the watchlist at module boundaries and reports leaks."""

    def __init__(self, mode: str) -> None:
        self.mode = mode  # "warn" | "error"
        self._current: _ModuleState | None = None
        self.leaks: list[str] = []

    def _snapshot(self) -> dict[str, object]:
        """Read every watched global that is currently importable/loaded."""
        snap: dict[str, object] = {}
        for w in WATCHLIST:
            module = sys.modules.get(w.module)
            if module is None:
                continue  # not loaded on this lane -> nothing to leak yet
            try:
                snap[w.label] = w.reader(module)
            except Exception:
                # a flaky reader must never break the run
                continue
        return snap

    def _finalize(self, state: _ModuleState) -> None:
        """Compare a finishing module's end snapshot against its start.

        Only labels present at *module start* are compared — a watched module
        that first loaded mid-run is a lazy import, not inherited state. A key
        maps to ``None`` only when the module was loaded but the global was
        empty/unset; that is a real value, so a ``None -> set`` transition
        (a singleton becoming live, a DB backend getting bound) IS a leak.
        """
        end = self._snapshot()
        by_label = {w.label: w for w in WATCHLIST}
        for label, start_val in state.start.items():
            if label not in end:
                continue  # module unloaded mid-run — nothing to inherit
            end_val = end[label]
            if _is_regression(start_val, end_val):
                w = by_label.get(label)
                why = f" — {w.why}" if w else ""
                self.leaks.append(
                    f"{state.name}: left '{label}' changed {start_val!r} -> {end_val!r}{why}"
                )

    def enter_module(self, name: str) -> None:
        if self._current is not None and self._current.name == name:
            return
        if self._current is not None:
            self._finalize(self._current)
        self._current = _ModuleState(name=name, start=self._snapshot())

    def finish(self) -> None:
        if self._current is not None:
            self._finalize(self._current)
            self._current = None
        if not self.leaks:
            return
        header = (
            f"[singleton-guard] {len(self.leaks)} cross-module state leak(s) detected "
            f"(mode={self.mode}):"
        )
        body = "\n".join(f"  - {leak}" for leak in self.leaks)
        message = f"{header}\n{body}"
        # Printed to stderr (not warnings.warn): sessionfinish runs AFTER pytest's
        # warning summary is rendered, so a warning here would be invisible.
        print(message, file=sys.stderr)  # noqa: T201 - guard diagnostic
        if self.mode == "warn":
            warnings.warn(
                f"[singleton-guard] {len(self.leaks)} cross-module state leak(s); "
                "see stderr for detail",
                stacklevel=1,
            )


def _resolve_mode() -> str | None:
    raw = (os.getenv("TLDW_SINGLETON_GUARD") or "").strip().lower()
    if raw in ("warn", "error"):
        return raw
    return None


def pytest_configure(config: pytest.Config) -> None:
    mode = _resolve_mode()
    if mode is None:
        return
    guard = SingletonGuard(mode)
    config._singleton_guard = guard  # type: ignore[attr-defined]


def pytest_runtest_setup(item: pytest.Item) -> None:
    guard = getattr(item.config, "_singleton_guard", None)
    if guard is None:
        return
    module = getattr(item, "module", None)
    name = getattr(module, "__name__", None) or str(getattr(item, "fspath", "?"))
    guard.enter_module(name)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    guard = getattr(session.config, "_singleton_guard", None)
    if guard is None:
        return
    guard.finish()
    if guard.mode == "error" and guard.leaks and exitstatus == 0:
        session.exitstatus = 1
