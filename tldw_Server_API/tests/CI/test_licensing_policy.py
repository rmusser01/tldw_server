from __future__ import annotations

import json
import os
import subprocess
from hashlib import sha256
from pathlib import Path

LICENSE_DIGESTS = {
    "LICENSES/PolyForm-Perimeter-1.0.1.txt": "5c7a5ccd847fcc285dda039e511ba013693fe979dfc5faee47f6fb59c7add337",
    "LICENSES/PolyForm-Countdown-1.0.0-template.txt": "eebf8d02412aa89d3d82fabdf6c67dfef04067e79f3f42d102a770c73590f2bf",
    "LICENSES/GPL-3.0-only.txt": "3972dc9744f6499f0f9b2dbf76696f2ae7ad8af9b23dde66d6af86c9dfb36986",
    "LICENSES/AGPL-3.0-only.txt": "0d96a4ff68ad6d4b6f1f30f713b18d5184912ba8dd389f86aa7710db079abcb0",
    "LICENSES/Apache-2.0.txt": "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30",
}
PROTECTED_PATHS = [
    "admin-ui/**",
    "apps/tldw-frontend/**",
    "apps/extension/**",
    "apps/packages/ui/**",
]
PROTECTED_PACKAGES = [
    "admin-ui",
    "apps/tldw-frontend",
    "apps/extension",
    "apps/packages/ui",
]


def _read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def test_verbatim_license_corpus_matches_reviewed_upstream_bytes() -> None:
    for path, expected_digest in LICENSE_DIGESTS.items():
        actual_digest = sha256(Path(path).read_bytes()).hexdigest()
        assert actual_digest == expected_digest, path


def test_root_license_maps_only_the_approved_protected_paths() -> None:
    text = _read("LICENSE")
    protected_section = text.split("## Protected Frontend Material", 1)[1].split("\n## ", 1)[0]
    protected_paths = [
        line.removeprefix("- `").removesuffix("`") for line in protected_section.splitlines() if line.startswith("- ")
    ]
    assert protected_paths == PROTECTED_PATHS
    assert "PolyForm Perimeter License 1.0.1" in text
    assert "GPL-3.0-only" in text
    assert "Apache-2.0" in text
    assert "previously public" in text
    assert "No trademark rights are granted" in text


def test_historical_record_preserves_public_refs_and_prior_grants() -> None:
    record = json.loads(_read("LICENSES/history/pre-source-available.json"))
    assert record["schema_version"] == 1
    assert record["recorded_on"] == "2026-07-21"
    assert record["repository"] == "https://github.com/rmusser01/tldw_server"
    assert record["public_refs"]["refs/heads/main"] == "7a23be3202e360f2d8e7cfe208e13ba406cf0507"
    assert record["public_refs"]["refs/heads/dev"] == "29acaca8c781213e27b12066372df13855e2e7a6"
    assert record["public_refs"]["refs/pull/2727/head"] == "e8bcc4c8b705df50a5f7e6299335ba8001ff4811"
    assert record["prior_grants_preserved"] is True
    assert record["ref_snapshot_is_exhaustive"] is False


def test_countdown_template_is_not_misrepresented_as_an_active_grant() -> None:
    readme = _read("LICENSES/releases/README.md")
    assert "No protected frontend release may be published" in readme
    assert "completed release-specific Countdown grant" in readme
    template = _read("LICENSES/PolyForm-Countdown-1.0.0-template.txt")
    assert "{start date}" in template
    assert "{Copy the scheduled license terms here.}" in template


def test_release_0_1_42_has_completed_source_record() -> None:
    release_dir = Path("LICENSES/releases/0.1.42")
    record = json.loads((release_dir / "release.json").read_text(encoding="utf-8"))

    assert record["schema_version"] == 1
    assert record["release_id"] == "0.1.42"
    assert record["product_version"] == "0.1.42"
    assert record["repository"] == "https://github.com/rmusser01/tldw_server"
    assert record["protected_source_revision"] == "0f3983788c413e0d17ffe7eabe8cff4a9f6ae723"
    assert record["release_date"] == "2026-07-26"
    assert record["countdown_start"] == "2028-07-26T12:00:00Z"
    assert record["protected_paths"] == PROTECTED_PATHS
    assert record["initial_license"] == {
        "name": "PolyForm Perimeter License 1.0.1",
        "path": "LICENSES/PolyForm-Perimeter-1.0.1.txt",
    }
    assert record["countdown_grant"] == {
        "additional_license": "AGPL-3.0-only",
        "path": "LICENSES/releases/0.1.42/PolyForm-Countdown-1.0.0.txt",
    }
    assert record["artifact_verification"]["protected_source_snapshot"] == {
        "manifest": "LICENSES/releases/0.1.42/protected-files.sha256",
        "result": "verified",
        "source_revision": "0f3983788c413e0d17ffe7eabe8cff4a9f6ae723",
    }
    assert record["artifact_verification"]["protected_binaries"] == {
        "published": False,
        "artifacts": [],
    }
    assert record["human_review_required"] is True

    grant = (release_dir / "PolyForm-Countdown-1.0.0.txt").read_text(encoding="utf-8")
    expected_grant = _read("LICENSES/PolyForm-Countdown-1.0.0-template.txt").replace(
        "{start date}",
        "2028-07-26",
    ).replace(
        "{Copy the scheduled license terms here.}",
        _read("LICENSES/AGPL-3.0-only.txt").rstrip(),
    )
    assert grant == expected_grant

    manifest_path = release_dir / "protected-files.sha256"
    manifest_bytes = manifest_path.read_bytes()
    assert sha256(manifest_bytes).hexdigest() == record["protected_file_manifest"]["sha256"]
    assert record["protected_file_manifest"]["path"] == str(manifest_path)

    entries = {}
    for line in manifest_bytes.decode("utf-8").splitlines():
        digest, path = line.split("  ", 1)
        assert len(digest) == 64
        assert path not in entries
        entries[path] = digest

    tracked_output = subprocess.run(
        ["git", "ls-files", "-s", "-z", "--", *PROTECTED_PACKAGES],
        check=True,
        capture_output=True,
    ).stdout
    tracked = {}
    for entry in tracked_output.split(b"\0"):
        if not entry:
            continue
        metadata, raw_path = entry.split(b"\t", 1)
        mode = metadata.split(b" ", 1)[0].decode("ascii")
        tracked[os.fsdecode(raw_path)] = mode

    assert set(entries) == set(tracked)
    for path, mode in tracked.items():
        data = os.fsencode(os.readlink(path)) if mode == "120000" else Path(path).read_bytes()
        assert sha256(data).hexdigest() == entries[path], path


def test_protected_packages_use_local_license_notices() -> None:
    for package in PROTECTED_PACKAGES:
        manifest = json.loads(_read(f"{package}/package.json"))
        assert manifest["license"] == "SEE LICENSE IN LICENSE"
        notice = _read(f"{package}/LICENSE")
        normalized_notice = " ".join(notice.split())
        assert "PolyForm Perimeter License 1.0.1" in notice
        assert (
            "Repository-authored code, tests, build definitions, and original assets in this package"
            in normalized_notice
        )
        assert "Markdown documentation in this package remains GPL-3.0-only" in normalized_notice
        assert "Repository-authored material in this package" not in normalized_notice
        assert "Required Notice:" in notice
        assert "LICENSES/releases" in notice
        assert "No trademark rights are granted" in notice


def test_public_copy_distinguishes_server_and_frontend_terms() -> None:
    root_readme = _read("README.md")
    extension_readme = _read("apps/extension/README.md")
    landing = _read("apps/tldw-frontend/components/landing/LandingLayout.tsx")
    contributing = _read("CONTRIBUTING.md")

    assert "Frontend: source-available" in root_readme
    assert "Server: GPL-3.0-only" in root_readme
    assert "source-available under PolyForm Perimeter 1.0.1" in extension_readme
    assert "Frontend source-available" in landing
    assert "Temporary licensing contribution gate" in contributing
    assert "apps/packages/ui/**" in contributing
    assert "Open source under GPL v2.0" not in landing


def test_extension_contribution_copy_obeys_temporary_gate() -> None:
    extension_readme = _read("apps/extension/README.md")
    assert "extension code contributions are temporarily paused" in extension_readme
    assert "Contributions are welcome! Please open an issue or PR." not in extension_readme


def test_landing_links_to_current_repository() -> None:
    landing = _read("apps/tldw-frontend/components/landing/LandingLayout.tsx")
    assert 'href="https://github.com/rmusser01/tldw_server"' in landing
    assert 'href="https://github.com/rmusser01/tldw"' not in landing


def test_contributing_links_to_current_issue_tracker() -> None:
    contributing = _read("CONTRIBUTING.md")
    assert "https://github.com/rmusser01/tldw_server/issues" in contributing
    assert "https://github.com/rmusser01/tldw/issues" not in contributing


def test_third_party_notices_preserve_frontend_upstream_terms() -> None:
    notices = _read("THIRD_PARTY_NOTICES.txt")
    assert "Host project: multi-license; see LICENSE" in notices
    assert "apps/packages/ui/src/Licenses/Page-Assist-LICENCE" in notices
    assert "apps/extension/public/pdf.worker.min.mjs" in notices
