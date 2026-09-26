"""Exercise origin and retained-alias checks against actual imported application code."""

import importlib.util
import json
import os
import subprocess  # nosec B404
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "Dockerfiles/candidates/compatibility/application-provenance.py"
XML = "tldw_Server_API.app.core.Ingestion_Media_Processing.XML_Ingestion_Lib"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def context(monkeypatch, tmp_path):
    assert SCRIPT.is_file(), "same-process provenance is not implemented"
    gate = load(SCRIPT, "candidate_provenance")
    launcher = load(ROOT / "Dockerfiles/candidates/combined-expat/run-tests.py", "candidate_launcher")
    modules = {}
    for nodeid in launcher.TESTS:
        path, function = nodeid.split("::")
        if path not in modules:
            modules[path] = load(ROOT / path, Path(path).stem)
    items = [
        SimpleNamespace(
            nodeid=nodeid,
            module=modules[nodeid.split("::")[0]],
            obj=getattr(modules[nodeid.split("::")[0]], nodeid.split("::")[1]),
        )
        for nodeid in launcher.TESTS
    ]
    # This local verification venv shares read-only dependencies with the repo
    # venv. Model that known site boundary without weakening the native policy.
    config = gate.sysconfig
    site = Path(sys.modules["defusedxml.ElementTree"].__file__).resolve().parents[1]
    gate.sysconfig = SimpleNamespace(
        get_path=lambda name: str(site) if name == "purelib" else config.get_path(name),
        get_config_var=lambda name: str(tmp_path) if name == "DESTSHARED" else config.get_config_var(name),
    )
    # uv's local macOS Python embeds these parsers; supply Linux-style file
    # metadata around the actual parser objects to exercise the origin boundary.
    # This fixture is never native binary qualification.
    for name in ("pyexpat", "_elementtree"):
        path = tmp_path / (name + config.get_config_var("EXT_SUFFIX"))
        path.write_bytes(b"Linux extension metadata fixture")
        monkeypatch.setattr(sys.modules[name], "__file__", str(path), raising=False)
        monkeypatch.setattr(sys.modules[name], "__spec__", SimpleNamespace(origin=str(path)))
    return gate, items


def test_actual_application_and_defusedxml_shim_origins_are_accepted(monkeypatch, tmp_path):
    gate, items = context(monkeypatch, tmp_path)
    result = gate.snapshot(ROOT, items)
    assert result["modules"][XML]["path"] == str(
        ROOT / "tldw_Server_API/app/core/Ingestion_Media_Processing/XML_Ingestion_Lib.py"
    )
    assert result["modules"]["pyexpat"]["sha256"]
    assert "defusedxml.ElementTree._XMLParser.__init__" in result["callables"]


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "file",
        "spec",
        "stale-parser",
        "test-alias",
        "callable",
        "extension-alias",
        "processing-alias",
        "collected-callable",
        "package-parser-alias",
    ],
)
def test_rejects_missing_substituted_or_stale_runtime_bindings(monkeypatch, tmp_path, mutation):
    gate, items = context(monkeypatch, tmp_path)
    module = sys.modules[XML]
    if mutation == "missing":
        monkeypatch.delitem(sys.modules, XML)
    elif mutation == "file":
        foreign = tmp_path / "XML_Ingestion_Lib.py"
        foreign.write_text("# outside candidate application\n")
        monkeypatch.setattr(module, "__file__", str(foreign))
    elif mutation == "spec":
        monkeypatch.setattr(module.__spec__, "origin", str(tmp_path / "elsewhere.py"))
    elif mutation == "stale-parser":
        monkeypatch.setattr(module, "DET", SimpleNamespace(fromstring=lambda _: None))
    elif mutation == "test-alias":
        monkeypatch.setattr(items[0].module, "xml_lib", SimpleNamespace(import_xml_handler=lambda: None))
    elif mutation == "callable":
        monkeypatch.setattr(module, "import_xml_handler", lambda: None)
    elif mutation == "extension-alias":
        monkeypatch.setattr(sys.modules["xml.parsers.expat"], "ParserCreate", lambda: None)
    elif mutation == "processing-alias":
        monkeypatch.setattr(module, "improved_chunking_process", lambda *_: None)
    elif mutation == "collected-callable":
        items[0].obj = lambda: None
    elif mutation == "package-parser-alias":
        monkeypatch.setattr(sys.modules["defusedxml"], "ElementTree", SimpleNamespace(fromstring=lambda _: None))
    with pytest.raises(ValueError):
        gate.snapshot(ROOT, items)


def test_intentional_processing_and_database_mocks_remain_allowed(monkeypatch, tmp_path):
    gate, items = context(monkeypatch, tmp_path)
    module = sys.modules[XML]
    monkeypatch.setattr(module, "improved_chunking_process", lambda *_: None)
    monkeypatch.setattr(module, "managed_media_database", lambda *_: None)
    assert gate.snapshot(ROOT, items, allow_processing_mock=True)["modules"][XML]["sha256"]


def test_candidate_mode_rejects_unqualified_local_interpreter(monkeypatch, tmp_path):
    gate, items = context(monkeypatch, tmp_path)
    with pytest.raises(ValueError):
        gate.snapshot(ROOT, items, candidate=True)


@pytest.mark.parametrize("name", ["fromstring", "parse", "iterparse"])
def test_rejects_generated_defusedxml_functions_capturing_foreign_parser(monkeypatch, tmp_path, name):
    gate, items = context(monkeypatch, tmp_path)
    det = sys.modules["defusedxml.ElementTree"]
    called = []

    class ForeignParser:
        def __init__(self, **kwargs):
            called.append(True)

        def feed(self, text):
            pass

        def close(self):
            return None

    functions = det._generate_etree_functions(ForeignParser, det._TreeBuilder, det._parse, det._iterparse)
    functions[2]("<root/>")
    assert called == [True]
    monkeypatch.setattr(det, name, dict(zip(("parse", "iterparse", "fromstring"), functions))[name])
    with pytest.raises(ValueError):
        gate.snapshot(ROOT, items)


def test_rejects_shared_foreign_factory_behind_both_expat_aliases(monkeypatch, tmp_path):
    gate, items = context(monkeypatch, tmp_path)

    def foreign_factory(*args, **kwargs):
        return None

    monkeypatch.setattr(sys.modules["pyexpat"], "ParserCreate", foreign_factory)
    monkeypatch.setattr(sys.modules["xml.parsers.expat"], "ParserCreate", foreign_factory)
    with pytest.raises(ValueError):
        gate.snapshot(ROOT, items)


@pytest.mark.parametrize("mutation", [None, "missing-observation", "checker-error", "failed-test", "changed-bytes"])
def test_observer_cannot_turn_missing_or_failed_evidence_into_a_pass(monkeypatch, tmp_path, mutation):
    gate, items = context(monkeypatch, tmp_path)
    assert hasattr(gate, "evidence_class"), "same-process observer is not implemented"
    launcher = load(ROOT / "Dockerfiles/candidates/combined-expat/run-tests.py", "observed_launcher")
    evidence = gate.evidence_class(launcher.Evidence, ROOT, candidate=False)()
    evidence.pytest_collection_finish(SimpleNamespace(items=items))
    for item in items:
        hook = evidence.pytest_runtest_call(item)
        next(hook)
        if mutation == "checker-error" and item is items[-1]:
            monkeypatch.delitem(sys.modules, XML)
        with pytest.raises(StopIteration):
            next(hook)
        for phase in ("setup", "call", "teardown"):
            outcome = "failed" if mutation == "failed-test" and phase == "call" else "passed"
            evidence.pytest_runtest_logreport(SimpleNamespace(nodeid=item.nodeid, when=phase, outcome=outcome))
    if mutation == "missing-observation":
        evidence.observations.pop()
    elif mutation == "changed-bytes":
        evidence.observations[-1]["snapshot"]["modules"][XML]["sha256"] = "0" * 64
    result = evidence.result(0)
    assert result["passed"] is (mutation is None)
    assert len(result["reports"]) == 24


def test_wrapper_rejects_changed_launcher_before_execution(tmp_path):
    gate = load(SCRIPT, "candidate_provenance")
    launcher = tmp_path / "launcher.py"
    marker = tmp_path / "executed"
    launcher.write_text(f"from pathlib import Path\nPath({str(marker)!r}).touch()\n")
    with pytest.raises(ValueError, match="unreviewed"):
        gate.run(launcher)
    assert not marker.exists()


def test_real_eight_test_launcher_preserves_outcomes_with_same_process_observer(tmp_path):
    tools = tmp_path / "tools"
    tools.mkdir()
    report = tmp_path / "report.json"
    # Adapt only this local runtime's metadata fixture, not the native CLI.
    driver = f"""
import importlib.util, sys
from pathlib import Path
import pytest
spec=importlib.util.spec_from_file_location('provenance_tests', {str(Path(__file__))!r})
fixture=importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)
with pytest.MonkeyPatch.context() as patches:
    gate, _ = fixture.context(patches, Path({str(tmp_path)!r}))
    factory=gate.evidence_class
    gate.evidence_class=lambda base, root: factory(base, root, candidate=False)
    sys.argv=['launcher', '--root', {str(ROOT)!r}, '--tools', {str(tools)!r}, '--report', {str(report)!r}]
    raise SystemExit(gate.run(Path({str(ROOT / 'Dockerfiles/candidates/combined-expat/run-tests.py')!r})))
"""
    result = subprocess.run(  # nosec B603
        [sys.executable, "-c", driver],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=90,
        env={**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"},
    )
    assert result.returncode == 0, result.stdout[-5000:] + result.stderr[-5000:]
    evidence = json.loads(report.read_text())
    assert evidence["passed"] is True
    assert len(evidence["reports"]) == 24
    assert len(evidence["provenance"]["observations"]) == 17
    assert evidence["provenance"]["errors"] == []
