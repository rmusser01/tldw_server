"""Exercise native build guards and retained OCI identity from the real workflow."""

import hashlib
import io
import json
import os
import subprocess  # nosec B404
import tarfile
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[3]
WORKFLOW = ROOT / ".github/workflows/ffmpeg-candidate.yml"


def step(name):
    assert WORKFLOW.is_file(), "native FFmpeg artifact workflow is missing"
    workflow = yaml.safe_load(WORKFLOW.read_text())
    return next(item for item in workflow["jobs"]["build"]["steps"] if item.get("name") == name)["run"]


@pytest.mark.parametrize(
    "kernel,arch,daemon,checkout,success",
    [
        ("Linux", "x86_64", "amd64", "a" * 40, True),
        ("Linux", "x86_64", "x86_64", "a" * 40, True),
        ("Darwin", "x86_64", "amd64", "a" * 40, False),
        ("Linux", "aarch64", "amd64", "a" * 40, False),
        ("Linux", "x86_64", "aarch64", "a" * 40, False),
        ("Linux", "x86_64", "amd64", "b" * 40, False),
    ],
)
def test_native_guard_rejects_emulation_or_wrong_checkout(tmp_path, kernel, arch, daemon, checkout, success):
    script = step("Require native checkout")
    commands = tmp_path / "commands"
    commands.mkdir()
    for name, body in {
        "git": 'printf "%s\\n" "$CHECKOUT"',
        "uname": 'if [ "$1" = -s ]; then printf "%s\\n" "$KERNEL"; else printf "%s\\n" "$ARCH"; fi',
        "docker": 'printf "%s\\n" "$DAEMON"',
    }.items():
        target = commands / name
        target.write_text("#!/bin/sh\n" + body + "\n")
        target.chmod(0o755)
    result = subprocess.run(  # nosec B603
        ["/bin/bash", "-c", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=10,
        env={
            **os.environ,
            "PATH": f"{commands}:/usr/bin:/bin",
            "GITHUB_SHA": "a" * 40,
            "RUNNER_TEMP": str(tmp_path),
            "KERNEL": kernel,
            "ARCH": arch,
            "DAEMON": daemon,
            "CHECKOUT": checkout,
        },
    )
    assert (result.returncode == 0) is success, result.stdout + result.stderr


@pytest.mark.parametrize("mutation", [None, "config-bytes", "wrong-platform", "wrong-subject", "detached-subject"])
def test_retained_oci_identity_rejects_unbound_or_tampered_inputs(tmp_path, mutation):
    script = step("Bind retained OCI artifact")
    files = {}

    def blob(value):
        data = json.dumps(value).encode()
        digest = hashlib.sha256(data).hexdigest()
        files[f"blobs/sha256/{digest}"] = data
        return "sha256:" + digest

    config = blob({"os": "linux", "architecture": "arm64" if mutation == "wrong-platform" else "amd64"})
    manifest = blob({"config": {"digest": config}, "layers": []})
    index = {"manifests": [{"digest": manifest, "platform": {"os": "linux", "architecture": "amd64"}}]}
    files["index.json"] = json.dumps(index).encode()
    subject = "sha256:" + hashlib.sha256(files["index.json"]).hexdigest()
    if mutation == "config-bytes":
        files[f"blobs/sha256/{config[7:]}"] += b" "
    if mutation == "wrong-subject":
        subject = "sha256:" + "0" * 64
    if mutation == "detached-subject":
        subject = blob({"config": {"digest": config}, "layers": [], "annotations": {"unbound": "true"}})
    with tarfile.open(tmp_path / "ffmpeg.oci.tar", "w") as bundle:
        for name, data in files.items():
            member = tarfile.TarInfo(name)
            member.size = len(data)
            bundle.addfile(member, io.BytesIO(data))
    archive_hash = hashlib.sha256((tmp_path / "ffmpeg.oci.tar").read_bytes()).hexdigest()
    result = subprocess.run(  # nosec B603
        ["/bin/bash", "-c", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=15,
        env={**os.environ, "RUNNER_TEMP": str(tmp_path), "CANDIDATE_SUBJECT": subject, "GITHUB_SHA": "a" * 40},
    )
    report = tmp_path / "ffmpeg-native-evidence/oci-subject.json"
    if mutation:
        assert result.returncode != 0
        assert not report.exists()
    else:
        assert result.returncode == 0, result.stdout + result.stderr
        identity = json.loads(report.read_text())
        assert identity["config_digest"] == config
        assert identity["subject_digest"] == subject
        assert identity["commit"] == "a" * 40
        assert f"{archive_hash}  ./ffmpeg.oci.tar\n" in (report.parent / "SHA256SUMS").read_text()
