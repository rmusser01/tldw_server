"""Assert opt-in dispatch evidence in the isolated backend test container."""

import json
import os
import subprocess  # nosec B404  # launches only the test fixture child
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
pytestmark = pytest.mark.skipif(
    os.getenv("TLDW_OUTBOUND_TEST_ISOLATED") != "1",
    reason="run with the outbound check's prepared, network-isolated backend image",
)


@pytest.mark.parametrize(
    ("case", "operation", "events", "rejection", "raw_attempts"),
    [
        ("download-default", "completed", ["hub_metadata", "fixture_download"], None, 0),
        ("download", "completed", ["hub_metadata", "fixture_download"], None, 0),
        ("download-origin", "failed", [], "undeclared_origin", 0),
        ("download-redirect", "failed", ["hub_metadata", "fixture_redirect"], "undeclared_origin", 0),
        ("download-file", "failed", [], "asset_identity_mismatch", 0),
        ("download-repository", "failed", [], "asset_identity_mismatch", 0),
        ("download-caught-dns", "completed", ["hub_metadata", "fixture_download"], None, 1),
        ("provider", "completed", ["openai_completion"], None, 0),
        ("provider-default", "completed", ["openai_default_completion"], None, 0),
        ("provider-origin", "failed", [], "undeclared_origin", 0),
        ("provider-caught-dns", "completed", ["openai_completion"], None, 1),
    ],
    ids=[
        "download-default", "download", "download-origin", "download-redirect", "download-file",
        "download-repository", "download-caught-dns", "provider", "provider-default", "provider-origin",
        "provider-caught-dns",
    ],
)
def test_real_download_and_provider_outbound_evidence(
    case: str,
    operation: str,
    events: list[str],
    rejection: str | None,
    raw_attempts: int,
    tmp_path: Path,
) -> None:
    result = subprocess.run(  # nosec B603  # fixed test-only Python module in the isolated container
        [sys.executable, "-m", "Helper_Scripts.ci.outbound_opt_in", case, str(tmp_path)],
        cwd=ROOT,
        env={
            "HOME": str(tmp_path),
            "PATH": os.defpath,
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
        },
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=60,
        check=False,
    )

    report = tmp_path / "evidence.json"
    assert report.is_file(), f"{case} did not produce terminal evidence (exit {result.returncode})"
    assert report.stat().st_size <= 4096
    evidence = json.loads(report.read_text())
    violations = ["raw_network_attempt"] if raw_attempts else []
    if rejection is not None:
        violations.append(rejection)
    assert evidence["status"] == ("fail" if violations else "pass")
    assert result.returncode == int(bool(violations))
    assert evidence["operation"] == operation
    assert evidence["events"] == events
    assert evidence["raw_attempts"] == raw_attempts
    assert evidence["violations"] == violations
    if case.startswith("download"):
        assert evidence["sdk_version"]
    assert set(evidence) == {"status", "operation", "events", "raw_attempts", "violations", "sdk_version"}
