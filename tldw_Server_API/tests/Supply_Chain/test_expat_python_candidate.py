"""Fail-closed boundaries for the bundled Expat candidate, without native builds."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import ModuleType
import subprocess  # nosec B404

import pytest

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[3]
DIRECTORY = ROOT / "Dockerfiles/candidates/expat"
URL = "https://github.com/libexpat/libexpat/releases/download/R_2_8_4/expat-2.8.4.tar.gz"


def load_tool(name: str) -> ModuleType:
    path = DIRECTORY / name
    assert path.is_file(), f"missing bundled candidate tool: {name}"
    spec = importlib.util.spec_from_file_location(name.replace("-", "_"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_fixture(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "python"
    baseline = tmp_path / "baseline"
    (root / "Modules/expat").mkdir(parents=True)
    (root / "Misc").mkdir()
    baseline.mkdir()
    for name in ("expat_config.h", "pyexpatns.h"):
        (root / "Modules/expat" / name).write_text("preserved " + name)
        (baseline / name).write_text("preserved " + name)
    before = (
        'expected_libexpat_tag="R_2_8_3"\nexpected_libexpat_version="2.8.3"\n'
        'expected_libexpat_sha256="22920a86c83f32300b11463635b71f11137a917975af297725e55525027d4e50"\n'
        "# exact upstream procedure\n"
    )
    (baseline / "refresh.sh").write_text(before)
    (root / "Modules/expat/refresh.sh").write_text(
        before.replace("R_2_8_3", "R_2_8_4")
        .replace("2.8.3", "2.8.4")
        .replace(
            "22920a86c83f32300b11463635b71f11137a917975af297725e55525027d4e50",
            "b8ece2437692dad44d851c4532723390a5a330990007706be9c8d2b90d294f36",
        )
    )
    (root / "Modules/expat/expat_external.h").write_text('#include "pyexpatns.h"\n')
    (root / "Modules/expat/xmlparse.c").write_bytes(b"parser\r\n")
    old = {
        "packages": [
            {
                "name": "expat",
                "SPDXID": "SPDXRef-PACKAGE-expat",
                "versionInfo": "2.8.3",
                "downloadLocation": URL.replace("R_2_8_4", "R_2_8_3").replace("2.8.4", "2.8.3"),
                "checksums": [
                    {
                        "algorithm": "SHA256",
                        "checksumValue": "22920a86c83f32300b11463635b71f11137a917975af297725e55525027d4e50",
                    }
                ],
                "externalRefs": [
                    {
                        "referenceType": "cpe23Type",
                        "referenceLocator": "cpe:2.3:a:libexpat_project:libexpat:2.8.3:*:*:*:*:*:*:*",
                    }
                ],
            },
            {"name": "unrelated", "versionInfo": "1"},
        ],
        "files": [],
        "relationships": [],
    }
    (baseline / "sbom.spdx.json").write_text(json.dumps(old))
    new = json.loads(json.dumps(old))
    new["packages"][0].update(
        versionInfo="2.8.4",
        downloadLocation=URL,
        checksums=[
            {"algorithm": "SHA256", "checksumValue": "b8ece2437692dad44d851c4532723390a5a330990007706be9c8d2b90d294f36"}
        ],
    )
    new["packages"][0]["externalRefs"][0][
        "referenceLocator"
    ] = "cpe:2.3:a:libexpat_project:libexpat:2.8.4:*:*:*:*:*:*:*"
    for name, content in (("expat_external.h", b'#include "pyexpatns.h"\n'), ("xmlparse.c", b"parser\n")):
        identifier = "SPDXRef-FILE-Modules-expat-" + name
        new["files"].append(
            {
                "SPDXID": identifier,
                "fileName": "Modules/expat/" + name,
                "checksums": [
                    {"algorithm": "SHA1", "checksumValue": hashlib.sha1(content, usedforsecurity=False).hexdigest()},
                    {"algorithm": "SHA256", "checksumValue": hashlib.sha256(content).hexdigest()},
                ],
            }
        )
        new["relationships"].append(
            {"spdxElementId": "SPDXRef-PACKAGE-expat", "relatedSpdxElement": identifier, "relationshipType": "CONTAINS"}
        )
    (root / "Misc/sbom.spdx.json").write_text(json.dumps(new))
    for path in (root / "Misc/externals.spdx.json", baseline / "externals.spdx.json"):
        path.write_text('{"packages":[{"name":"windows","versionInfo":"1"}]}')
    return root, baseline


def test_source_verification_accepts_refreshed_files_and_normalized_hashes(tmp_path: Path) -> None:
    root, baseline = source_fixture(tmp_path)
    assert load_tool("python-source.py").verify(root, baseline) == {"version": "2.8.4", "files": 2}


@pytest.mark.parametrize(
    "mutation",
    [
        "header",
        "procedure",
        "source",
        "missing-file",
        "duplicate-file",
        "relationship",
        "version",
        "unrelated",
        "externals",
        "namespace",
    ],
)
def test_source_verification_rejects_incomplete_or_altered_refresh(tmp_path: Path, mutation: str) -> None:
    root, baseline = source_fixture(tmp_path)
    sbom_path = root / "Misc/sbom.spdx.json"
    sbom = json.loads(sbom_path.read_text())
    if mutation == "header":
        (root / "Modules/expat/pyexpatns.h").write_text("changed")
    elif mutation == "procedure":
        (root / "Modules/expat/refresh.sh").write_text("skip actual refresh")
    elif mutation == "source":
        (root / "Modules/expat/xmlparse.c").write_text("stale SBOM")
    elif mutation == "missing-file":
        sbom["files"].pop()
    elif mutation == "duplicate-file":
        sbom["files"].append(sbom["files"][0])
    elif mutation == "relationship":
        sbom["relationships"].pop()
    elif mutation == "version":
        sbom["packages"][0]["versionInfo"] = "2.8.3"
    elif mutation == "unrelated":
        sbom["packages"][1]["versionInfo"] = "2"
    elif mutation == "externals":
        (root / "Misc/externals.spdx.json").write_text('{"packages":[]}')
    else:
        (root / "Modules/expat/expat_external.h").write_text("no namespace")
    sbom_path.write_text(json.dumps(sbom))
    with pytest.raises(ValueError):
        load_tool("python-source.py").verify(root, baseline)


@pytest.mark.parametrize(
    "arguments",
    [["--location", URL], ["--location", "https://example.invalid/"], ["--location", URL, "-o", "elsewhere"], []],
)
def test_offline_refresh_adapter_emits_only_the_approved_archive(tmp_path: Path, arguments: list[str]) -> None:
    script = DIRECTORY / "offline-bin/curl"
    assert script.exists(), "missing restricted offline refresh adapter"
    (tmp_path / "expat-2.8.4.tar.gz").write_bytes(b"authenticated immutable fixture")
    result = subprocess.run(  # nosec B603
        ["/bin/bash", "-c", script.read_text().replace("/work/downloads", str(tmp_path)), "curl", *arguments],
        capture_output=True,
        timeout=10,
    )
    if arguments == ["--location", URL]:
        assert result.returncode == 0 and result.stdout == b"authenticated immutable fixture"
    else:
        assert result.returncode != 0 and result.stdout == b""


def test_python_xml_controls_cover_non_null_child_and_legitimate_documents() -> None:
    assert load_tool("python-controls.py").controls() == {"default-precedence": 3, "child-dtd-copy": 3, "namespace": 3}


@pytest.mark.parametrize("step", ["baseline-tests", "xml-tests"])
def test_xml_runner_isolates_module_reloads_without_dropping_suites(step: str) -> None:
    # Removing worker isolation reproduces CPython's JUnit Element-type crash.
    script = (DIRECTORY / "python-qualify.sh").read_text().replace("\\\n", "")
    command = next(line.strip() for line in script.splitlines() if line.strip().startswith(f"run_step {step} "))
    probe = """
set -euo pipefail
PY_SOURCE=/candidate
EVIDENCE=/evidence
XML_TESTS=(test_pyexpat test_xml_etree test_xml_etree_c test_minidom test_sax)
run_step() { printf '%s\\0' "$@"; }
"""
    result = subprocess.run(  # nosec B603
        ["/bin/bash", "-c", probe + command], capture_output=True, check=True, timeout=10
    )
    arguments = result.stdout.decode().rstrip("\0").split("\0")
    assert "-j1" in arguments
    assert arguments[-5:] == ["test_pyexpat", "test_xml_etree", "test_xml_etree_c", "test_minidom", "test_sax"]
    assert arguments[arguments.index("--timeout") + 1] == "300"
    assert arguments[arguments.index("--junit-xml") + 1].startswith("/evidence/")


@pytest.mark.parametrize("mutation", [None, "failure", "new-skip", "missing-suite", "empty"])
def test_xml_suite_gate_rejects_failures_new_skips_and_missing_coverage(tmp_path: Path, mutation: str | None) -> None:
    cases = [
        f'<testcase name="test.{name}.Tests.test_parse"/>'
        for name in ("test_pyexpat", "test_xml_etree", "test_xml_etree_c", "test_minidom", "test_sax")
    ]
    baseline = tmp_path / "baseline.xml"
    candidate = tmp_path / "candidate.xml"
    baseline.write_text("<testsuites><testsuite>" + "".join(cases) + "</testsuite></testsuites>")
    if mutation == "failure":
        cases[0] = cases[0].replace("/>", "><failure>failed</failure></testcase>")
    elif mutation == "new-skip":
        cases[0] = cases[0].replace("/>", "><skipped>no parser</skipped></testcase>")
    elif mutation == "missing-suite":
        cases.pop()
    elif mutation == "empty":
        cases.clear()
    candidate.write_text("<testsuites><testsuite>" + "".join(cases) + "</testsuite></testsuites>")
    module = load_tool("python-suite.py")
    if mutation is None:
        assert module.compare(baseline, candidate) == {"tests": 5, "skipped": []}
    else:
        with pytest.raises(ValueError):
            module.compare(baseline, candidate)


@pytest.mark.parametrize(
    "data",
    [
        b'<!DOCTYPE testsuites [<!ENTITY x "expand">]><testsuites/>',
        "<!DOCTYPE testsuites><testsuites/>".encode("utf-16"),
        b" " * (10 * 1024 * 1024 + 1),
    ],
    ids=["dtd", "utf16", "oversized"],
)
def test_suite_reader_rejects_entity_definitions_encodings_and_oversized_reports(tmp_path: Path, data: bytes) -> None:
    report = tmp_path / "report.xml"
    report.write_bytes(data)
    with pytest.raises(ValueError, match="DTD|UTF-8|limit"):
        load_tool("python-suite.py").read_suite(report)


@pytest.mark.parametrize("failed_phase", ["prepare", "build", "install", None])
def test_python_controller_requires_all_phases_and_retains_failed_evidence(
    tmp_path: Path, failed_phase: str | None
) -> None:
    from tldw_Server_API.tests.Supply_Chain.test_expat_native_candidate import controller_environment

    script = DIRECTORY / "python-qualify.sh"
    assert script.exists(), "missing Python native controller"
    env = controller_environment(tmp_path, FAIL_PHASE=failed_phase or "")
    for phase, files in {
        "prepare": ("source-verification.log", "python-source.tar.xz", "baseline.xml"),
        "build": ("xml-tests.log", "xml-results.xml", "suite-comparison.log", "python-controls.log"),
        "install": ("python-controls.log", "binary-checksums.log"),
    }.items():
        for filename in files:
            (tmp_path / "fixtures" / phase / filename).write_text("recorded worker evidence")
    result = subprocess.run(  # nosec B603
        ["/bin/bash", str(script), "controller", str(tmp_path / "evidence")],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if failed_phase:
        assert result.returncode == 7, result.stdout + result.stderr
        assert (tmp_path / f"evidence/{failed_phase}/container.exit").read_text() == "7\n"
        assert not (tmp_path / "evidence/python-qualified.txt").exists()
    else:
        assert result.returncode == 0, result.stdout + result.stderr
        assert (tmp_path / "evidence/python-qualified.txt").is_file()
        assert not (tmp_path / "evidence/system-qualified.txt").exists()
        commands = (tmp_path / "commands").read_text().splitlines()
        creates = [line for line in commands if line.startswith("create ")]
        assert len(creates) == 3 and all("--network none" in line for line in creates)


def test_suite_comparison_detects_a_missing_test_even_when_counts_match(tmp_path: Path) -> None:
    names = [
        f"test.{name}.Tests.test_parse"
        for name in ("test_pyexpat", "test_xml_etree", "test_xml_etree_c", "test_minidom", "test_sax")
    ]
    names.append("test.test_pyexpat.Tests.test_extra")
    baseline = tmp_path / "baseline.xml"
    candidate = tmp_path / "candidate.xml"
    baseline.write_text(
        "<testsuites><testsuite>"
        + "".join(f'<testcase name="{name}"/>' for name in names)
        + "</testsuite></testsuites>"
    )
    names[-1] = names[0]
    candidate.write_text(
        "<testsuites><testsuite>"
        + "".join(f'<testcase name="{name}"/>' for name in names)
        + "</testsuite></testsuites>"
    )
    with pytest.raises(ValueError):
        load_tool("python-suite.py").compare(baseline, candidate)


@pytest.mark.parametrize("mutation", [None, "missing", "system-link", "unprefixed-symbol"])
def test_elf_gate_stops_on_each_bad_library_instead_of_overwriting_failure(
    tmp_path: Path, mutation: str | None
) -> None:
    script = (DIRECTORY / "python-qualify.sh").read_text()
    body = script.split("check_elf() {\n", 1)[1].split("\n}\n", 1)[0]
    prefix = tmp_path / "runtime"
    prefix.mkdir()
    for name in ("python", "libpython", "pyexpat", "elementtree"):
        (prefix / name).write_bytes(b"ELF fixture")
    probe = r"""
set -euo pipefail
die() { printf '%s\n' "$*" >&2; exit 1; }
python() { printf '%s\n' "$PREFIX/python" "$PREFIX/libpython" "$PREFIX/pyexpat" "$PREFIX/elementtree"; }
readelf() {
    if [[ "$*" == *pyexpat ]]; then
        if [[ "$*" == *--dynamic* && "$MUTATION" == system-link ]]; then echo 'NEEDED Shared library: [libexpat.so.1]'; fi
        if [[ "$*" == *--dyn-syms* && "$MUTATION" == unprefixed-symbol ]]; then echo '1: 1234 12 FUNC GLOBAL DEFAULT 10 XML_Parse'; fi
    fi
    return 0
}
ldd() { if [[ "$1" == *pyexpat && "$MUTATION" == missing ]]; then echo 'libmissing.so => not found'; fi; return 0; }
"""
    result = subprocess.run(  # nosec B603
        ["/bin/bash", "-c", probe + "check_elf() {\n" + body + '\n}\ncheck_elf ignored "$PREFIX"'],
        env={**os.environ, "PREFIX": str(prefix), "EVIDENCE": str(tmp_path), "MUTATION": mutation or ""},
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert (result.returncode == 0) == (mutation is None), result.stdout + result.stderr


@pytest.mark.parametrize("phase", ["prepare", "build", "install"])
def test_python_phase_gate_rejects_status_only_without_test_evidence(tmp_path: Path, phase: str) -> None:
    script = DIRECTORY / "python-qualify.sh"
    assert script.exists(), "missing Python native controller"
    (tmp_path / "phase.exit").write_text("0\n")
    (tmp_path / "complete.txt").write_text(phase + "\n")
    result = subprocess.run(  # nosec B603
        ["/bin/bash", str(script), "verify-evidence", str(tmp_path), phase],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0


@pytest.mark.parametrize(
    "mutation", [None, "version", "elementtree", "extension-path", "interpreter-path", "shared-library"]
)
def test_python_identity_requires_the_owning_candidate_runtime(mutation: str | None) -> None:
    identity = {
        "python_version": "3.12.14",
        "expat_version": "expat_2.8.4",
        "elementtree_version": "Expat 2.8.4",
        "executable": "/work/Python-3.12.14/python",
        "pyexpat": "/work/Python-3.12.14/build/pyexpat.so",
        "elementtree": "/work/Python-3.12.14/build/_elementtree.so",
        "libpython": "/work/Python-3.12.14/libpython3.12.so.1.0",
    }
    if mutation == "version":
        identity["expat_version"] = "expat_2.8.3"
    elif mutation == "elementtree":
        identity["elementtree_version"] = "Expat 2.8.3"
    elif mutation == "extension-path":
        identity["pyexpat"] = "/usr/local/lib/pyexpat.so"
    elif mutation == "interpreter-path":
        identity["executable"] = "/usr/local/bin/python"
    elif mutation == "shared-library":
        identity["libpython"] = "/usr/local/lib/libpython3.12.so.1.0"
    module = load_tool("python-controls.py")
    if mutation is None:
        module.validate_identity(identity, "/work/Python-3.12.14")
    else:
        with pytest.raises(ValueError):
            module.validate_identity(identity, "/work/Python-3.12.14")
