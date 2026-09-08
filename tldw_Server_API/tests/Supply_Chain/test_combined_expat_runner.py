"""Execute the candidate launcher against small real pytest suites."""

import importlib.util
import json
import os
import subprocess  # nosec B404
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "Dockerfiles/candidates/combined-expat/run-tests.py"
TESTS = (
    "MediaIngestion_NEW/unit/test_xml_ingestion.py::test_import_xml_handler_reports_malformed_input",
    "MediaIngestion_NEW/unit/test_xml_ingestion.py::test_import_xml_handler_sanitizes_unexpected_processing_failure",
    "MediaIngestion_NEW/unit/test_xml_ingestion.py::test_import_xml_handler_uses_managed_media_database",
    "Chunking/test_xml_allows_url_text.py::test_xml_allows_urls_in_text_nodes",
    "Chunking/test_xml_allows_url_text.py::test_xml_allows_system_in_text_nodes",
    "Chunking/test_json_xml_offsets.py::test_chunk_with_metadata_json_offsets_match_source",
    "Chunking/test_json_xml_offsets.py::test_chunk_with_metadata_xml_offsets_match_source",
    "Chunking/test_xml_tail_preservation.py::test_xml_chunk_preserves_tail_text",
)


def launcher():
    assert SCRIPT.is_file(), "isolated application launcher is not implemented"
    spec = importlib.util.spec_from_file_location("combined_runner", SCRIPT)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def test_required_identities_are_the_eight_reviewed_application_tests():
    assert tuple(launcher().TESTS) == tuple("tldw_Server_API/tests/" + test for test in TESTS)


@pytest.mark.parametrize("alias_kind", ["canonical", "dot", "symlink", "relative"])
def test_rejects_existing_tools_path_alias_before_importing_pytest(tmp_path, alias_kind):
    launcher()
    tools = tmp_path / "tools"
    tools.mkdir()
    marker = tmp_path / "pytest-imported"
    (tools / "pytest.py").write_text(f"from pathlib import Path\nPath({str(marker)!r}).touch()\nraise SystemExit(93)\n")
    alias = str(tools)
    if alias_kind == "dot":
        alias += "/."
    elif alias_kind == "symlink":
        link = tmp_path / "tools-alias"
        link.symlink_to(tools, target_is_directory=True)
        alias = str(link)
    elif alias_kind == "relative":
        alias = "tools"
    report = tmp_path / "report.json"
    bootstrap = (
        "import runpy, sys; "
        f"sys.path.insert(0, {alias!r}); "
        f"sys.argv = [{str(SCRIPT)!r}, '--root', {str(tmp_path)!r}, '--tools', {str(tools)!r}, '--report', {str(report)!r}]; "
        f"runpy.run_path({str(SCRIPT)!r}, run_name='__main__')"
    )
    result = subprocess.run(  # nosec B603
        [sys.executable, "-I", "-c", bootstrap],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 2, result.stdout + result.stderr
    assert not marker.exists()
    assert not report.exists()


@pytest.mark.parametrize(
    "outcome", ["pass", "skip", "xfail", "xpass", "fail", "setup-error", "teardown-error", "missing"]
)
def test_launcher_requires_every_test_phase_to_pass_and_preserves_import_order(tmp_path, outcome):
    launcher()
    project = tmp_path / "project"
    project.mkdir()
    # Normal conftests must load; test-tool .pth files must not execute.
    (project / "conftest.py").write_text("import pytest\n@pytest.fixture\ndef normal_fixture():\n    return 'normal'\n")
    files = {}
    for index, identity in enumerate(TESTS):
        filename, name = identity.split("::")
        body = "    assert normal_fixture == 'normal'\n    import contract_dependency\n    assert contract_dependency.ORIGIN == 'runtime'\n"
        decorators = ""
        fixture = "normal_fixture"
        if index == 0:
            if outcome == "skip":
                body = "    pytest.skip('control')\n"
            elif outcome == "xfail":
                body = "    pytest.xfail('control')\n"
            elif outcome == "xpass":
                decorators = "@pytest.mark.xfail(reason='control', strict=False)\n"
            elif outcome == "fail":
                body = "    assert False, 'control'\n"
            elif outcome in {"setup-error", "teardown-error"}:
                fixture = "broken_fixture"
                setup = (
                    "    raise RuntimeError('control')\n"
                    if outcome == "setup-error"
                    else "    yield\n    raise RuntimeError('control')\n"
                )
                files.setdefault(filename, []).append("@pytest.fixture\ndef broken_fixture():\n" + setup)
                body = "    pass\n"
            elif outcome == "missing":
                continue
        files.setdefault(filename, []).append(f"{decorators}def {name}({fixture}):\n{body}")
    for filename, contents in files.items():
        path = project / "tldw_Server_API/tests" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("import pytest\n" + "\n".join(contents))
    runtime = tmp_path / "runtime"
    tools = tmp_path / "tools"
    runtime.mkdir()
    tools.mkdir()
    (runtime / "contract_dependency.py").write_text("ORIGIN = 'runtime'\n")
    (tools / "contract_dependency.py").write_text("ORIGIN = 'tools'\n")
    marker = tmp_path / "pth-executed"
    (tools / "danger.pth").write_text(f"import pathlib; pathlib.Path({str(marker)!r}).touch()\n")
    report = tmp_path / "report.json"
    bootstrap = (
        "import runpy, sys; "
        f"sys.path.insert(0, {str(runtime)!r}); "
        f"sys.argv = [{str(SCRIPT)!r}, '--root', {str(project)!r}, '--tools', {str(tools)!r}, '--report', {str(report)!r}]; "
        f"runpy.run_path({str(SCRIPT)!r}, run_name='__main__')"
    )
    result = subprocess.run(  # nosec B603
        [sys.executable, "-I", "-c", bootstrap],
        cwd=project,
        env={**os.environ, "PYTEST_ADDOPTS": "-k nothing_should_match"},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert (result.returncode == 0) == (outcome == "pass"), result.stdout + result.stderr
    evidence = json.loads(report.read_text())
    assert evidence["passed"] == (outcome == "pass")
    assert evidence["scope"] == "application-tests-only"
    assert evidence["expected"] == ["tldw_Server_API/tests/" + test for test in TESTS]
    assert not marker.exists()
    if outcome == "pass":
        assert len(evidence["reports"]) == 24
