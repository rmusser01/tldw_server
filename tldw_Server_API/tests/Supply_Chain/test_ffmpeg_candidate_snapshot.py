"""Configuration-contract tests for the FFmpeg candidate's Debian snapshot."""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DOCKERFILE = REPO_ROOT / "Dockerfiles/candidates/ffmpeg/Dockerfile"
DEBIAN_SOURCES = REPO_ROOT / "Dockerfiles/candidates/ffmpeg/debian.sources"
PINNED_BASE = "python:3.12.14-slim-trixie@" "sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea"


def _parse_deb822(path: Path) -> list[dict[str, str]]:
    assert path.is_file(), f"missing snapshot source definition: {path}"
    records = []
    for paragraph in path.read_text(encoding="utf-8").strip().split("\n\n"):
        fields = {}
        for line in paragraph.splitlines():
            key, value = line.split(":", 1)
            fields[key] = value.strip()
        records.append(fields)
    return records


def _docker_stages() -> list[tuple[str, str, str]]:
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    matches = list(re.finditer(r"^FROM\s+(\S+)\s+AS\s+(\S+)\s*$", dockerfile, re.MULTILINE))
    return [
        (
            match.group(1),
            match.group(2),
            dockerfile[match.end() : matches[index + 1].start() if index + 1 < len(matches) else None],
        )
        for index, match in enumerate(matches)
    ]


def _assert_order(text: str, *needles: str) -> None:
    positions = [text.index(needle) for needle in needles]
    assert positions == sorted(positions)


def test_debian_sources_pin_main_and_security_snapshots() -> None:
    assert _parse_deb822(DEBIAN_SOURCES) == [
        {
            "Types": "deb deb-src",
            "URIs": "https://snapshot.debian.org/archive/debian/20260906T000000Z/",
            "Suites": "trixie trixie-updates",
            "Components": "main",
            "Signed-By": "/usr/share/keyrings/debian-archive-keyring.gpg",
            "Check-Valid-Until": "no",
        },
        {
            "Types": "deb deb-src",
            "URIs": "https://snapshot.debian.org/archive/debian-security/20260906T000000Z/",
            "Suites": "trixie-security",
            "Components": "main",
            "Signed-By": "/usr/share/keyrings/debian-archive-keyring.gpg",
            "Check-Valid-Until": "no",
        },
    ]


def test_snapshot_configuration_keeps_apt_authentication_enabled() -> None:
    assert DEBIAN_SOURCES.is_file()
    combined = (DEBIAN_SOURCES.read_text(encoding="utf-8") + DOCKERFILE.read_text(encoding="utf-8")).lower()
    for forbidden in (
        "deb.debian.org",
        "security.debian.org",
        "trusted=yes",
        "trusted: yes",
        "allow-insecure",
        "allow-unauthenticated",
        "acquire::allowinsecurerepositories",
        "apt::get::allowunauthenticated",
        "acquire::check-valid-until",
    ):
        assert forbidden not in combined
    assert re.search(r"^ARG\s+.*snapshot", combined, re.MULTILINE) is None


def test_every_apt_stage_inherits_the_shared_snapshot_base() -> None:
    stages = _docker_stages()
    assert [(base, name) for base, name, _ in stages] == [
        (PINNED_BASE, "snapshot-base"),
        ("snapshot-base", "build-deps"),
        ("build-deps", "build"),
        ("snapshot-base", "candidate"),
    ]
    stage_bodies = {name: body for _, name, body in stages}
    assert [name for _, name, body in stages if "apt-get update" in body] == ["build-deps", "candidate"]
    snapshot_base = stage_bodies["snapshot-base"]
    _assert_order(
        snapshot_base,
        "rm -f /etc/apt/sources.list /etc/apt/sources.list.d/*",
        "COPY Dockerfiles/candidates/ffmpeg/debian.sources",
    )
    assert "apt-get update" not in stage_bodies["snapshot-base"]
    assert "apt-get update" not in stage_bodies["build"]


def test_build_deps_target_stops_after_snapshot_package_resolution() -> None:
    stage_bodies = {name: body for _, name, body in _docker_stages()}
    assert "build-deps" in stage_bodies
    build_deps = stage_bodies["build-deps"]
    assert "apt-get build-dep -y --no-install-recommends ffmpeg" in build_deps
    assert "./configure" not in build_deps
    assert "/usr/bin/meson setup" not in build_deps
    assert "./configure" in stage_bodies["build"]


def test_build_and_runtime_keep_separate_signed_snapshot_evidence() -> None:
    stage_bodies = {name: body for _, name, body in _docker_stages()}
    assert {"build-deps", "candidate"} <= stage_bodies.keys()
    for stage, directory, install in (
        ("build-deps", "apt-build", "apt-get install -y --no-install-recommends"),
        ("candidate", "apt-runtime", "xargs -r apt-get install -y --no-install-recommends"),
    ):
        body = stage_bodies[stage]
        evidence = f"/opt/tldw-ffmpeg9/share/candidate/{directory}"
        _assert_order(
            body,
            "apt-get update",
            install,
            f"mkdir -p {evidence}",
            "cp /etc/apt/sources.list.d/debian.sources",
            "rm -rf /var/lib/apt/lists/*",
        )
        assert "for suite in trixie trixie-updates trixie-security" in body
        assert 'test "${#metadata[@]}" -eq 1' in body
        assert 'test -f "${metadata[0]}"' in body
        assert 'cp -- "${metadata[0]}" "$evidence_dir/"' in body
