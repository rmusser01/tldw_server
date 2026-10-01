"""UAT568: observe an owned pytest child; retain its cleanup and natural exit.

Copied into RUNNER_TEMP before checkout of the immutable tested source. No
termination watchdog: the original Actions job maximum remains the outer bound.
Only symbols/phase numbers are recorded; native sample headers are discarded.
"""

import ast
import atexit
import builtins
import faulthandler
import gc
import importlib.metadata
import json
import os
import re
import runpy
import signal
import subprocess
import sys
import threading
import time
import types
from pathlib import Path

ROOT = Path(os.environ["PROMPT_SHUTDOWN_PROBE_DIR"])
ROOT.mkdir(parents=True, exist_ok=True)
EVENT_PATH = ROOT / "events.jsonl"
TRACE = (ROOT / "faulthandler.log").open("a", buffering=1)
GRAPH_ENABLED = False


def record(event: str, **fields) -> None:
    """Persist and stream observations without changing delegated operations."""
    try:
        payload = {"event": event, "monotonic_ns": time.monotonic_ns(), "pid": os.getpid(), **fields}
        line = json.dumps(payload, sort_keys=True) + "\n"
        with EVENT_PATH.open("a") as handle:
            handle.write(line)
        os.write(1, line.encode())
    except (OSError, ValueError, TypeError, RuntimeError):
        # An unavailable observation never replaces the original pytest result.
        try:
            os.write(2, b"UAT568 observation unavailable\n")
        except OSError:
            return  # Closed diagnostic descriptors cannot replace cleanup.


def snapshot(event: str) -> None:
    """Record ownership and symbol locations, excluding names/locals/values."""
    try:
        frames = sys._current_frames()
        rows = []
        for thread in threading.enumerate():
            frame = frames.get(thread.ident)
            stack = []
            while frame is not None:
                stack.append(
                    {
                        "file": Path(frame.f_code.co_filename).name,
                        "function": frame.f_code.co_name,
                        "line": frame.f_lineno,
                    }
                )
                frame = frame.f_back
            rows.append({"ident": thread.ident, "daemon": thread.daemon, "alive": thread.is_alive(), "stack": stack})
        record(event, threads=rows)
        TRACE.write("\n=== " + event + " ===\n")
        faulthandler.dump_traceback(file=TRACE, all_threads=True)
    except (OSError, ValueError, TypeError, RuntimeError):
        record("snapshot_unavailable")


def _graph_symbols(indices: dict[int, int]) -> tuple[dict[int, str], dict[int, str], bool]:
    """Label fixed native types and source-attested exported class/module symbols."""
    labels = {}
    for name in ("dict", "list", "tuple", "set", "frozenset", "type", "object", "str", "int", "bytes"):
        labels[id(builtins.__dict__[name])] = "builtins." + name
    for name in ("SimpleNamespace", "FunctionType", "ModuleType", "CodeType", "FrameType", "TracebackType",
                 "CellType", "MethodType", "GeneratorType", "CoroutineType", "AsyncGeneratorType"):
        labels[id(types.__dict__[name])] = "types." + name
    roots = {}
    complete = True
    paths = [Path(entry or os.getcwd()).resolve() for entry in sys.path if type(entry) is str]
    exports = {}
    for name in tuple(sys.modules):
        module = sys.modules.get(name)
        if type(module) is not types.ModuleType or type(name) is not str:
            continue
        if not all(part.isidentifier() for part in name.split(".")) or id(module) not in indices:
            continue
        namespace = types.ModuleType.__getattribute__(module, "__dict__")
        filename = namespace.get("__file__")
        if type(filename) is not str or not filename.endswith(".py"):
            continue
        try:
            path = Path(filename).resolve()
            attested = False
            for base in paths:
                if path.is_relative_to(base):
                    parts = path.relative_to(base).with_suffix("").parts
                    if parts[-1] == "__init__":
                        parts = parts[:-1]
                    if ".".join(parts) == name:
                        attested = True
                        break
            if not attested:
                continue
            if str(path) not in exports:
                with open(path, encoding="utf-8") as source:
                    tree = ast.parse(source.read())
                exports[str(path)] = [node.name for node in tree.body
                                      if isinstance(node, ast.ClassDef) and not node.name.startswith("_")]
                del tree
            roots[id(module)] = name
            for symbol in exports[str(path)]:
                candidate = namespace.get(symbol)
                if issubclass(type(candidate), type):
                    owner = type.__dict__["__module__"].__get__(candidate)
                    qualified = type.__dict__["__qualname__"].__get__(candidate)
                    if type(owner) is str and type(qualified) is str and owner == name and qualified == symbol:
                        labels[id(candidate)] = name + "." + symbol
        except (OSError, ValueError, TypeError, RuntimeError, MemoryError, SyntaxError):
            # Source faults only reduce symbol coverage; never expose dynamic names.
            complete = False
    return labels, roots, complete


def graph_snapshot(phase: str) -> None:
    """Stream tracked C tp_traverse edges; this is neither all roots nor causation.

    The census temporarily holds every tracked object. Maps store primitive IDs
    only; adjacency is streamed one object at a time. Census/index shallow bytes
    exclude primitive entries, source parsing and JSON buffers, so are not peak RSS.
    Edges are observed sequentially, not atomically. Allocations can perturb
    automatic GC; its configuration is never changed.
    """
    if type(phase) is not str or phase not in ("pre-unconfigure", "pre-module-clear"):
        return
    started = time.monotonic_ns()
    objects = references = obj = handle = None
    indices, type_indices, labels, roots = {}, {}, {}, {}
    type_labels = []
    count = written = census_bytes = indices_bytes = max_referents = 0
    complete = symbols_complete = False
    try:
        handle = (ROOT / ("graph-" + phase + ".jsonl")).open("w", encoding="utf-8")
        handle.write(json.dumps({"phase": phase, "scope": "tracked-tp_traverse"}) + "\n")
        objects = gc.get_objects()
        count = len(objects)
        indices = {id(obj): index for index, obj in enumerate(objects)}
        census_bytes, indices_bytes = sys.getsizeof(objects), sys.getsizeof(indices)
        labels, roots, symbols_complete = _graph_symbols(indices)
        for node, obj in enumerate(objects):
            kind = id(type(obj))
            if kind not in type_indices:
                type_indices[kind] = len(type_labels)
                type_labels.append(labels.get(kind))
            references = gc.get_referents(obj)
            max_referents = max(max_referents, len(references))
            edges = [indices[id(target)] for target in references if id(target) in indices]
            handle.write(json.dumps([node, type_indices[kind], edges], separators=(",", ":")) + "\n")
            references = None
            written += 1
        complete = True
    except (OSError, ValueError, TypeError, RuntimeError, MemoryError):
        # No observation failure, including MemoryError, may replace cleanup.
        complete = False
    finally:
        elapsed = time.monotonic_ns() - started
        try:
            if handle is not None:
                handle.write(json.dumps({"complete": complete, "tracked_objects": count, "nodes_written": written,
                                         "types": type_labels, "module_roots": [[indices[key], label]
                                                                                for key, label in roots.items()],
                                         "source_symbols_complete": symbols_complete, "elapsed_ns": elapsed,
                                         "census_shallow_bytes": census_bytes, "indices_shallow_bytes": indices_bytes,
                                         "max_referents": max_referents}) + "\n")
                handle.close()
        except (OSError, ValueError, TypeError, RuntimeError, MemoryError):
            complete = False
        finally:
            # Drop census and current tp_traverse results before original cleanup,
            # including failure paths. Nothing retaining instances is cached.
            objects = references = obj = handle = None
            indices = type_indices = labels = roots = None
            type_labels = None
        try:
            record("graph_capture", phase=phase, complete=complete, source_symbols_complete=symbols_complete,
                   tracked_objects=count, nodes_written=written, elapsed_ns=time.monotonic_ns() - started)
        except (OSError, ValueError, TypeError, RuntimeError, MemoryError):
            return  # Even event allocation failure cannot replace delegated cleanup.


def _pre_module_clear() -> None:
    """Observe after later ordinary atexit callbacks, before module clearing."""
    graph_snapshot("pre-module-clear")
    snapshot("atexit_early_registration")


def initialize() -> None:
    """Register the same private probe's normal shutdown and stack observers."""
    global GRAPH_ENABLED
    GRAPH_ENABLED = True
    faulthandler.enable(file=TRACE, all_threads=True)
    faulthandler.register(signal.SIGUSR1, file=TRACE, all_threads=True, chain=False)
    record("launcher_start", python_version=sys.version.split()[0], implementation=sys.implementation.name)
    atexit.register(_pre_module_clear)
    threading._register_atexit(snapshot, "threading_atexit_early_registration")


def pytest_configure(config) -> None:
    """Delegate every original call once, including its exceptions and return."""
    original = config._ensure_unconfigure
    captured = False

    def observed_unconfigure():
        nonlocal captured
        if GRAPH_ENABLED and not captured:
            captured = True
            graph_snapshot("pre-unconfigure")
        snapshot("ensure_unconfigure_enter")
        try:
            return original()
        finally:
            snapshot("ensure_unconfigure_exit")

    config._ensure_unconfigure = observed_unconfigure


def pytest_collection_finish(session) -> None:
    """Verify managed source origins and report installed dependency versions."""
    checkout = Path.cwd().resolve()
    managed = [
        module
        for name, module in sys.modules.items()
        if name == "tldw_Server_API" or name.startswith("tldw_Server_API.")
    ]
    valid = all(
        not getattr(module, "__file__", None) or Path(module.__file__).resolve().is_relative_to(checkout)
        for module in managed
    )
    versions = {}
    for name in ("fastapi", "pydantic", "pydantic-core", "starlette", "pytest", "torch"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    record(
        "collection_finish",
        selected=len(session.items),
        timeout=session.config.getini("timeout"),
        timeout_method=session.config.getini("timeout_method"),
        asyncio_plugin_loaded=session.config.pluginmanager.hasplugin("pytest_asyncio.plugin"),
        timeout_plugin_loaded=session.config.pluginmanager.hasplugin("pytest_timeout"),
        managed_modules=len(managed),
        managed_origins_valid=valid,
        versions=versions,
    )
    if not valid:
        raise RuntimeError("UAT568 managed source origin mismatch")


def pytest_sessionfinish(session, exitstatus) -> None:
    """Observe session completion separately from the printed pytest summary."""
    snapshot("pytest_sessionfinish")
    record(
        "pytest_exitstatus", exitstatus=int(exitstatus), collected=session.testscollected, failures=session.testsfailed
    )
    atexit.register(snapshot, "atexit_late_registration")
    threading._register_atexit(snapshot, "threading_atexit_late_registration")


def pytest_unconfigure(config) -> None:
    """Observe the existing unconfigure hook without changing it."""
    snapshot("pytest_unconfigure")


def sample_child(child: subprocess.Popen, checkpoint: int | str) -> bool:
    """Only sample this parent's still-live, unreaped child; never terminate it."""
    if child.poll() is not None:
        return False
    # Native sampling cannot deliver a Python signal after its handler has
    # been removed during interpreter shutdown, including marker-read races.
    record("native_sample_requested", child_pid=child.pid, checkpoint=checkpoint)
    if sys.platform != "darwin":
        record("native_sample_unavailable", child_pid=child.pid, checkpoint=checkpoint)
        return True
    # Raw headers stay outside the always-uploaded results tree, even if the
    # original Actions maximum interrupts this parent during native sampling.
    raw = ROOT.parent / "native-private.sample"
    try:
        result = subprocess.run(
            ["/usr/bin/sample", str(child.pid), "5", "-file", str(raw)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
            check=False,
        )
        frames = []
        for line in raw.read_text().splitlines() if raw.exists() else []:
            match = re.match(r"^([ +!|:]*)([0-9]+) (.+?)  \(in ([^()]+)\)", line)
            if match:
                frames.append(
                    {"depth": len(match[1]), "samples": int(match[2]), "symbol": match[3], "image": Path(match[4]).name}
                )
        (ROOT / f"native-{checkpoint}.json").write_text(
            json.dumps(
                {"pid": child.pid, "checkpoint": checkpoint, "sample_exit": result.returncode, "frames": frames},
                indent=2,
            )
        )
        record(
            "native_sample_finished",
            child_pid=child.pid,
            checkpoint=checkpoint,
            sample_exit=result.returncode,
            frame_count=len(frames),
        )
    except (OSError, subprocess.TimeoutExpired):
        record("native_sample_unavailable", child_pid=child.pid, checkpoint=checkpoint)
    finally:
        try:
            raw.unlink(missing_ok=True)
        except OSError:
            record("native_sample_cleanup_unavailable", child_pid=child.pid, checkpoint=checkpoint)
    return True


def supervise(arguments: list[str]) -> int:
    """Inherit original output and return the child's actual natural status."""
    started = time.monotonic()
    session_finished = None
    post_atexit = None
    sent = []
    late_sent = []
    command = [sys.executable, str(Path(__file__).resolve()), "--child", *arguments]
    # This parent is the sole reaper: a child exiting during sample stays an
    # owned zombie until the next poll, so its PID cannot identify another task.
    child = subprocess.Popen(command)
    record("child_started", child_pid=child.pid)
    try:
        with EVENT_PATH.open() as events:
            while child.poll() is None:
                line = events.readline()
                while line:
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        events.seek(events.tell() - len(line))
                        break  # A writer can still be finishing its last line.
                    if (
                        event["event"] == "pytest_sessionfinish"
                        and event["pid"] == child.pid
                        and session_finished is None
                    ):
                        session_finished = time.monotonic()
                    if (
                        event["event"] == "atexit_early_registration"
                        and event["pid"] == child.pid
                        and post_atexit is None
                    ):
                        post_atexit = time.monotonic()
                    line = events.readline()
                if session_finished is not None and post_atexit is None:
                    for checkpoint in (15, 30, 60, 90):
                        if time.monotonic() - session_finished >= checkpoint and checkpoint not in sent:
                            if child.poll() is None:
                                if sample_child(child, checkpoint):
                                    sent.append(checkpoint)
                if post_atexit is not None:
                    for checkpoint in (15, 30, 60, 90, 300, 600, 900):
                        if time.monotonic() - post_atexit >= checkpoint and checkpoint not in late_sent:
                            if child.poll() is None:
                                if sample_child(child, f"atexit-{checkpoint}"):
                                    late_sent.append(checkpoint)
                time.sleep(0.5)
    except (OSError, ValueError):
        record("parent_observations_unavailable", child_pid=child.pid)
    finally:
        child.wait()  # Artifact failures cannot orphan or terminate pytest.
    outcome = {
        "pid": child.pid,
        "child_process_exit": child.returncode,
        "forced_termination": False,
        "elapsed_seconds": time.monotonic() - started,
        "sessionfinish_to_parent_observed_exit_seconds": None
        if session_finished is None
        else time.monotonic() - session_finished,
        "sessionfinish_native_checkpoints": sent,
        "post_atexit_native_checkpoints": late_sent,
        "post_atexit_to_parent_observed_exit_seconds": None
        if post_atexit is None
        else time.monotonic() - post_atexit,
        "acceptance": False,
    }
    try:
        (ROOT / "result.json").write_text(json.dumps(outcome, indent=2))
    except OSError:
        record("parent_artifacts_unavailable", child_pid=child.pid)
    record("child_natural_exit", **outcome)
    return child.returncode if child.returncode >= 0 else 128 - child.returncode


if __name__ == "__main__":
    if sys.argv[1:2] == ["--child"]:
        # Match python -m pytest's checkout-first imports and load one probe.
        sys.path.insert(0, os.getcwd())
        sys.modules["pr2979_native_shutdown"] = sys.modules[__name__]
        initialize()
        sys.argv = ["pytest", "-p", "pr2979_native_shutdown", *sys.argv[2:]]
        try:
            runpy.run_module("pytest", run_name="__main__", alter_sys=True)
        except SystemExit as error:
            snapshot("pytest_system_exit")
            record(
                "system_exit_code",
                code=error.code if isinstance(error.code, int) else None,
                non_integer_code=error.code is not None and not isinstance(error.code, int),
            )
            raise
        finally:
            record("launcher_finally")
    else:
        sys.exit(supervise(sys.argv[1:]))
