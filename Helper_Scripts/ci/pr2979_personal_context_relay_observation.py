"""Content-free Windows relay observation; timings include diagnostic overhead."""

from __future__ import annotations

import functools
import hashlib
import importlib
import json
import os
import re
import sys
import threading
import time
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

TARGET = "test_abandoned_preparation_expires_without_covering_source"
RELAY_MODULE = "tldw_Server_API.app.core.Sync.v2.personal_context_relay"
STORE_MODULE = "tldw_Server_API.app.core.Personalization.personal_context_publication"
SOURCE_PATHS = (
    "tldw_Server_API/tests/Personalization/test_personal_context_activation.py",
    "tldw_Server_API/app/core/Sync/v2/personal_context_relay.py",
    "tldw_Server_API/app/core/Personalization/personal_context_publication.py",
)


def unavailable() -> None:
    """Disclose unavailable evidence without replacing the original outcome."""
    try:
        print("UAT558_DIAGNOSTIC_UNAVAILABLE")
    except (OSError, MemoryError):
        pass


def bindings(item: Any, relay_module: Any, store_module: Any) -> dict[str, Any]:
    """Bind known loaded modules to fixed source paths; emit no machine paths."""
    root = Path.cwd().resolve()
    modules = (getattr(item, "module", None), relay_module, store_module)
    origins = {}
    hashes = {}
    for label, module, relative in zip(("test", "relay", "store"), modules, SOURCE_PATHS, strict=True):
        filename = getattr(module, "__file__", None)
        origins[label] = bool(filename) and Path(filename).resolve() == root / relative
        hashes[label] = hashlib.sha256((root / relative).read_bytes()).hexdigest()
    versions = {}
    for name in ("fastapi", "pydantic", "pydantic_core", "starlette", "pytest"):
        value = getattr(sys.modules.get(name), "__version__", None)
        versions[name] = value if type(value) is str and re.fullmatch(r"[0-9]+(?:\.[0-9]+){1,3}", value) else None
    return {
        "helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_sha256": hashes,
        "loaded_origins_match": origins,
        "runtime_versions": versions,
        "python_version": list(sys.version_info[:3]),
    }


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item: Any) -> Generator[None, None, None]:
    """Observe only the two original abandoned-preparation cases, then restore."""
    cases = {TARGET + "[False]": "legacy_false", TARGET + "[True]": "legacy_true"}
    case = cases.get(item.name)
    if case is None:
        yield
        return
    prefix = os.environ.get("UAT558_PROBE_OUTPUT")
    try:
        if not prefix:
            raise AttributeError
        relay_module = importlib.import_module(RELAY_MODULE)
        store_module = importlib.import_module(STORE_MODULE)
        relay_class = relay_module.PersonalContextRelay
        budget_class = relay_module.PersonalContextRecoveryBudget
        store_class = store_module.PersonalContextPublicationRelayStore
        targets = [
            (store_class, name, "source." + name)
            for name in ("unfinished_stage_identities", "earliest_nonterminal_batch", "renew_lease")
        ] + [
            (budget_class, name, "budget." + name)
            for name in ("deadline_open", "can_inspect", "consume", "consume_returned")
        ]
        originals = [getattr(owner, name) for owner, name, _ in targets]
        original_relay, original_owned = relay_class.relay_profile, relay_class._relay_owned
        original_lease = store_class.profile_lease
        if not all(callable(fn) for fn in (*originals, original_relay, original_owned, original_lease)):
            raise AttributeError
    except (ImportError, AttributeError, MemoryError):
        try:
            yield
        finally:
            unavailable()
        return

    records: list[dict[str, Any]] = []
    active: dict[str, Any] | None = None
    active_store = active_budget = None
    active_thread: int | None = None
    complete = True

    def now() -> int | None:
        nonlocal complete
        try:
            return time.perf_counter_ns()
        except (OSError, MemoryError):
            complete = False
            return None

    def event(label: str, started: int | None, **fields: Any) -> None:
        nonlocal complete
        if active is None or threading.get_ident() != active_thread:
            return
        try:
            ended = now()
            origin = active["started"]
            active["events"].append(
                {
                    "phase": label,
                    "at_ns": ended - origin if ended is not None and origin is not None else None,
                    "wall_ns": ended - started if ended is not None and started is not None else None,
                    **fields,
                }
            )
        except (MemoryError, TypeError):
            complete = False

    def timed(label: str, original: Any, is_budget: bool) -> Any:
        @functools.wraps(original)
        def call(instance: Any, *args: Any, **kwargs: Any) -> Any:
            if instance is not (active_budget if is_budget else active_store) or threading.get_ident() != active_thread:
                return original(instance, *args, **kwargs)
            started, returned, value = now(), False, None
            try:
                value = original(instance, *args, **kwargs)
                returned = True
                return value
            finally:
                # Inspect only original fixed boolean/counter fields, never source rows.
                nonlocal complete
                try:
                    fields: dict[str, Any] = {"returned": returned}
                    if type(value) is bool:
                        fields["decision"] = value
                    if is_budget:
                        fields["remaining_rows"] = instance.remaining_rows
                    event(label, started, **fields)
                except (MemoryError, AttributeError, TypeError):
                    complete = False

        return call

    @functools.wraps(original_owned)
    def owned(instance: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal active_budget
        if active is not None and instance.publications is active_store and threading.get_ident() == active_thread:
            active_budget = kwargs.get("budget")
        return original_owned(instance, *args, **kwargs)

    @contextmanager
    @functools.wraps(original_lease)
    def lease(instance: Any, *args: Any, **kwargs: Any) -> Generator[Any, None, None]:
        if instance is not active_store or threading.get_ident() != active_thread:
            with original_lease(instance, *args, **kwargs) as token:
                yield token
            return
        started, exit_started, entered, normal = now(), None, False, False
        try:
            with original_lease(instance, *args, **kwargs) as token:
                entered = True
                event("lease.enter", started, granted=token is not None)
                try:
                    yield token
                finally:
                    exit_started = now()
            normal = True
        finally:
            if not entered:
                event("lease.enter", started, returned=False)
            event("lease.exit", exit_started, returned=normal)

    @functools.wraps(original_relay)
    def relay(instance: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal active, active_store, active_budget, active_thread, complete
        if active is not None:
            return original_relay(instance, *args, **kwargs)
        try:
            active = {"started": now(), "events": []}
            active_store, active_thread = instance.publications, threading.get_ident()
        except MemoryError:
            complete = False
            active = None
            return original_relay(instance, *args, **kwargs)
        result, returned = None, False
        try:
            result = original_relay(instance, *args, **kwargs)
            returned = True
            return result
        finally:
            try:
                event("relay.exit", active["started"], returned=returned)
                progress = {}
                for name in ("staged_rows", "inspected_rows", "source_exhausted", "visible_lookahead"):
                    value = getattr(result, name, None)
                    if type(value) in (int, bool):
                        progress[name] = value
                continuation = getattr(result, "continuation", None)
                if type(continuation) is str and continuation in (
                    "complete",
                    "personal_context_relay_pending",
                    "relay_poisoned",
                ):
                    progress["continuation"] = continuation
                active.pop("started")
                active["result"] = progress
                records.append(active)
            except (MemoryError, AttributeError, TypeError):
                complete = False
            finally:
                active = active_store = active_budget = active_thread = None

    try:
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(relay_class, "relay_profile", relay)
            patch.setattr(relay_class, "_relay_owned", owned)
            patch.setattr(store_class, "profile_lease", lease)
            for (owner, name, label), original in zip(targets, originals, strict=True):
                patch.setattr(owner, name, timed(label, original, owner is budget_class))
            yield
    finally:
        try:
            document = {
                "schema_version": 1,
                "case": case,
                "complete": complete,
                "intervals_are_inclusive": True,
                "bindings": bindings(item, relay_module, store_module),
                "records": records,
            }
            Path(prefix + "-" + case + ".json").write_text(json.dumps(document, indent=2), encoding="utf-8")
        except (OSError, MemoryError, AttributeError, TypeError):
            unavailable()
