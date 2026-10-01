"""Synthetic checks for diagnostic transparency; never runs Prompt Studio."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

OBSERVER = Path(os.environ.get("OBSERVER_SOURCE", str(Path(__file__).with_name("pr2979_native_shutdown.py"))))
PRIVACY_SENTINEL = "synthetic-private-value-DO-NOT-RECORD"


def test_streamed_phases_preserve_delegate_return_and_exception(tmp_path):
    script = """
import importlib.util, os, sys, threading
from types import SimpleNamespace
spec = importlib.util.spec_from_file_location('probe', sys.argv[1])
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)
secret = sys.argv[2]
threading.current_thread().name = secret
calls = []
def cleanup():
    calls.append(1)
    if len(calls) == 2:
        raise RuntimeError(secret)
    if len(calls) == 3:
        raise SystemExit(11)
    return 7
config = SimpleNamespace(_ensure_unconfigure=cleanup)
probe.pytest_configure(config)
assert config._ensure_unconfigure() == 7
try:
    config._ensure_unconfigure()
except RuntimeError as error:
    assert str(error) == secret
else:
    raise AssertionError('original exception lost')
assert calls == [1, 1]
probe.record('synthetic_finished')
probe.EVENT_PATH = probe.ROOT
os.close(1)
os.close(2)
try:
    config._ensure_unconfigure()
except SystemExit as error:
    assert error.code == 11
else:
    raise AssertionError('original SystemExit lost')
assert calls == [1, 1, 1]
"""
    env = {**os.environ, "PROMPT_SHUTDOWN_PROBE_DIR": str(tmp_path)}
    result = subprocess.run(
        [sys.executable, "-c", script, str(OBSERVER), PRIVACY_SENTINEL],
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert [row["event"] for row in events] == [
        "ensure_unconfigure_enter",
        "ensure_unconfigure_exit",
        "ensure_unconfigure_enter",
        "ensure_unconfigure_exit",
        "synthetic_finished",
    ]
    assert '"event": "synthetic_finished"' in result.stdout
    assert (
        PRIVACY_SENTINEL
        not in result.stdout
        + result.stderr
        + (tmp_path / "events.jsonl").read_text()
        + (tmp_path / "faulthandler.log").read_text()
    )


@pytest.mark.parametrize("body, expected", [("assert True", 0), ("assert False", 1)])
@pytest.mark.parametrize("result_writable", [True, False])
def test_parent_preserves_pytest_natural_exit(tmp_path, body, expected, result_writable):
    case = tmp_path / "test_synthetic.py"
    case.write_text("def test_synthetic():\n    " + body + "\n")
    records = tmp_path / "observations"
    if not result_writable:
        (records / "result.json").mkdir(parents=True)
    env = {**os.environ, "PROMPT_SHUTDOWN_PROBE_DIR": str(records), "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"}
    result = subprocess.run(
        [
            sys.executable,
            str(OBSERVER),
            "-c",
            "/dev/null",
            "--confcutdir",
            str(tmp_path),
            "-p",
            "pytest_timeout",
            "-o",
            "cache_dir=" + str(tmp_path / "cache"),
            "-q",
            str(case),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == expected, result.stdout + result.stderr
    events = [json.loads(line) for line in (records / "events.jsonl").read_text().splitlines()]
    names = [row["event"] for row in events]
    assert names.count("ensure_unconfigure_enter") == names.count("ensure_unconfigure_exit") >= 1
    assert {
        "pytest_sessionfinish",
        "pytest_system_exit",
        "atexit_early_registration",
        "atexit_late_registration",
    } <= set(names)
    assert next(row for row in events if row["event"] == "system_exit_code")["code"] == expected
    if not result_writable:
        assert "parent_artifacts_unavailable" in result.stdout
        return
    outcome = json.loads((records / "result.json").read_text())
    assert outcome["child_process_exit"] == expected
    assert outcome["forced_termination"] is False
    assert outcome["sessionfinish_native_checkpoints"] == []


def test_parent_samples_its_live_child_and_allows_delayed_natural_exit(tmp_path):
    case = tmp_path / "test_synthetic.py"
    case.write_text("def test_synthetic():\n    assert True\n")
    # The real child remains alive after sessionfinish. Only observer artifacts
    # are checked for privacy; ordinary pytest output remains ordinary pytest.
    (tmp_path / "conftest.py").write_text(
        "import time, threading\ndef pytest_unconfigure(config):\n    secret = "
        + repr(PRIVACY_SENTINEL)
        + "\n    threading.current_thread().name = secret\n    time.sleep(17)\n"
    )
    records = tmp_path / "observations"
    env = {**os.environ, "PROMPT_SHUTDOWN_PROBE_DIR": str(records), "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"}
    result = subprocess.run(
        [
            sys.executable,
            str(OBSERVER),
            "-c",
            "/dev/null",
            "--confcutdir",
            str(tmp_path),
            "-p",
            "pytest_timeout",
            "-o",
            "cache_dir=" + str(tmp_path / "cache"),
            "-q",
            str(case),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=35,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    outcome = json.loads((records / "result.json").read_text())
    assert outcome["child_process_exit"] == 0 and outcome["forced_termination"] is False
    assert outcome["sessionfinish_native_checkpoints"] == [15]
    assert "pytest_unconfigure" in (records / "faulthandler.log").read_text()
    assert PRIVACY_SENTINEL not in "".join(
        path.read_text() for path in records.iterdir() if path.suffix in {".json", ".jsonl", ".log"}
    )
    if sys.platform == "darwin":
        native = json.loads((records / "native-15.json").read_text())
        assert native["pid"] == outcome["pid"]
        assert native["frames"]


def test_parent_observes_post_atexit_child_without_python_signals(tmp_path):
    case = tmp_path / "test_synthetic.py"
    case.write_text("def test_synthetic():\n    assert True\n")
    # This object survives until interpreter module clearing after all atexit
    # callbacks. Default SIGUSR1 makes any late Python signal visibly fatal.
    (tmp_path / "conftest.py").write_text(
        "import signal, time\n"
        "class FinalizationHold:\n"
        "    def __init__(self):\n"
        "        self.private = " + repr(PRIVACY_SENTINEL) + "\n"
        "    def __del__(self, sleep=time.sleep, set_signal=signal.signal, "
        "sig=signal.SIGUSR1, default=signal.SIG_DFL):\n"
        "        set_signal(sig, default)\n"
        "        sleep(23)\n"
        "hold = FinalizationHold()\n"
    )
    records = tmp_path / "observations"
    env = {**os.environ, "PROMPT_SHUTDOWN_PROBE_DIR": str(records), "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"}
    result = subprocess.run(
        [sys.executable, str(OBSERVER), "-c", "/dev/null", "--confcutdir", str(tmp_path),
         "-p", "pytest_timeout", "-o", "cache_dir=" + str(tmp_path / "cache"), "-q", str(case)],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=40,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    outcome = json.loads((records / "result.json").read_text())
    assert outcome["child_process_exit"] == 0 and outcome["forced_termination"] is False
    assert outcome["post_atexit_native_checkpoints"] == [15]
    events = [json.loads(line) for line in (records / "events.jsonl").read_text().splitlines()]
    marker = next(row for row in events if row["event"] == "atexit_early_registration")
    assert not [row for row in events if row["event"] == "owned_child_stack_requested"
                and row["monotonic_ns"] >= marker["monotonic_ns"]]
    assert outcome["post_atexit_to_parent_observed_exit_seconds"] >= 15
    assert PRIVACY_SENTINEL not in "".join(
        path.read_text() for path in records.iterdir() if path.suffix in {".json", ".jsonl", ".log"}
    )
    if sys.platform == "darwin":
        native = json.loads((records / "native-atexit-15.json").read_text())
        assert native["pid"] == outcome["pid"] and native["frames"]
        finished = next(row for row in events if row["event"] == "native_sample_finished"
                        and row["checkpoint"] == "atexit-15")
        assert finished["monotonic_ns"] > marker["monotonic_ns"]


def test_native_graph_records_private_cycle_without_values(tmp_path):
    case = tmp_path / "test_synthetic.py"
    case.write_text("def test_synthetic():\n    assert True\n")
    (tmp_path / "conftest.py").write_text(
        "from types import SimpleNamespace\n"
        "cycle = SimpleNamespace(private=" + repr(PRIVACY_SENTINEL) + ")\n"
        "cycle.peer = cycle\n"
    )
    records = tmp_path / "observations"
    result = subprocess.run(
        [sys.executable, str(OBSERVER), "-c", "/dev/null", "--confcutdir", str(tmp_path),
         "-p", "pytest_timeout", "-o", "cache_dir=" + str(tmp_path / "cache"), "-q", str(case)],
        cwd=tmp_path, env={**os.environ, "PROMPT_SHUTDOWN_PROBE_DIR": str(records),
                          "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"},
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    graphs = [records / ("graph-" + phase + ".jsonl")
              for phase in ("pre-unconfigure", "pre-module-clear")]
    assert all(path.is_file() for path in graphs), "actual retained graph missing"
    for path in graphs:
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        header, footer = rows[0], rows[-1]
        nodes = rows[1:-1]
        assert header["phase"] in {"pre-unconfigure", "pre-module-clear"}
        assert footer["complete"] is True and footer["tracked_objects"] == len(nodes)
        assert [row[0] for row in nodes] == list(range(len(nodes)))
        assert all(0 <= edge < len(nodes) for row in nodes for edge in row[2])
        symbols = footer["types"]
        namespace_ids = {row[0] for row in nodes if symbols[row[1]] == "types.SimpleNamespace"}
        edges = {row[0]: row[2] for row in nodes}
        assert any(symbols[nodes[child][1]] == "builtins.dict" and n in edges[child]
                   for n in namespace_ids for child in edges[n])
        assert footer["elapsed_ns"] >= 0 and header["scope"] == "tracked-tp_traverse"
        assert PRIVACY_SENTINEL not in path.read_text()
    outcome = json.loads((records / "result.json").read_text())
    assert outcome["child_process_exit"] == 0 and outcome["forced_termination"] is False


def test_graph_releases_instances_and_hides_dynamic_symbols(tmp_path):
    (tmp_path / "guarded_symbols.py").write_text(
        "class Meta(type):\n"
        "    def __getattribute__(self, name):\n"
        "        raise AssertionError('type attribute hook invoked')\n"
        "    @property\n"
        "    def __dict__(self):\n"
        "        raise AssertionError('type descriptor invoked')\n"
        "class Declared(metaclass=Meta):\n"
        "    pass\n"
    )
    script = """
import gc, importlib.util, sys, types, weakref
spec = importlib.util.spec_from_file_location('probe', sys.argv[1])
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)
secret = sys.argv[2]
sys.path.insert(0, sys.argv[3])
from guarded_symbols import Declared
guarded = Declared()
guarded_ref = weakref.ref(guarded)
def poison(self):
    raise AssertionError('repr invoked')
kind = type(secret, (), {'__repr__': poison, '__module__': secret})
item = kind()
item.private = secret
ref = weakref.ref(item)
module = types.ModuleType(secret)
module.private = item
sys.modules[secret] = module
before = (gc.isenabled(), gc.get_debug(), gc.get_threshold(), tuple(gc.callbacks))
probe.graph_snapshot(secret)
probe.graph_snapshot(item)
assert not (probe.ROOT / ('graph-' + secret + '.jsonl')).exists()
probe.graph_snapshot('pre-unconfigure')
del module.private, item, guarded
assert guarded_ref() is None
assert ref() is None, 'observer retained an ordinary instance'
assert before == (gc.isenabled(), gc.get_debug(), gc.get_threshold(), tuple(gc.callbacks))
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(OBSERVER), PRIVACY_SENTINEL, str(tmp_path)],
        env={**os.environ, "PROMPT_SHUTDOWN_PROBE_DIR": str(tmp_path)},
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    graph = (tmp_path / "graph-pre-unconfigure.jsonl").read_text()
    footer = json.loads(graph.splitlines()[-1])
    assert footer["complete"] is True
    assert "guarded_symbols.Declared" in footer["types"]
    assert "guarded_symbols" in [row[1] for row in footer["module_roots"]]
    assert PRIVACY_SENTINEL not in graph + result.stdout + result.stderr


@pytest.mark.parametrize("fault", ["output", "objects", "referents", "memory", "source", "source-memory"])
def test_graph_failures_preserve_cleanup_and_gc_configuration(tmp_path, fault):
    script = """
import gc, importlib.util, sys, weakref
from types import SimpleNamespace
spec = importlib.util.spec_from_file_location('probe', sys.argv[1])
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)
secret, fault = sys.argv[2:]
probe.initialize()
class Ordinary:
    pass
item = Ordinary()
ref = weakref.ref(item)
before = (gc.isenabled(), gc.get_debug(), gc.get_threshold(), tuple(gc.callbacks))
def unavailable(*args, **kwargs):
    if 'memory' in fault:
        raise MemoryError(secret)
    raise OSError(secret)
if fault == 'output':
    (probe.ROOT / 'graph-pre-unconfigure.jsonl').mkdir()
elif fault.startswith('source'):
    probe.open = unavailable
else:
    probe.gc = SimpleNamespace(get_objects=gc.get_objects, get_referents=gc.get_referents)
    if fault in {'objects', 'memory'}:
        probe.gc.get_objects = unavailable
    else:
        probe.gc.get_referents = unavailable
calls = []
def cleanup():
    calls.append(1)
    if len(calls) == 2:
        raise RuntimeError(secret)
    if len(calls) == 3:
        raise SystemExit(11)
    return 7
for outcome in range(3):
    config = SimpleNamespace(_ensure_unconfigure=cleanup)
    probe.pytest_configure(config)
    if outcome == 0:
        assert config._ensure_unconfigure() == 7
    else:
        try:
            config._ensure_unconfigure()
        except RuntimeError as error:
            assert outcome == 1 and str(error) == secret
        except SystemExit as error:
            assert outcome == 2 and error.code == 11
        else:
            raise AssertionError('original exception lost')
assert calls == [1, 1, 1]
del item
assert ref() is None, 'failure path retained an instance'
assert before == (gc.isenabled(), gc.get_debug(), gc.get_threshold(), tuple(gc.callbacks))
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(OBSERVER), PRIVACY_SENTINEL, fault],
        env={**os.environ, "PROMPT_SHUTDOWN_PROBE_DIR": str(tmp_path)},
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    captures = [row for row in events if row["event"] == "graph_capture" and row["phase"] == "pre-unconfigure"]
    assert len(captures) == 3
    assert all(row["complete"] is fault.startswith("source") for row in captures)
    if fault.startswith("source"):
        footer = json.loads((tmp_path / "graph-pre-unconfigure.jsonl").read_text().splitlines()[-1])
        assert footer["source_symbols_complete"] is False
        assert footer["module_roots"] == []
    assert PRIVACY_SENTINEL not in result.stdout + result.stderr + (tmp_path / "events.jsonl").read_text()
