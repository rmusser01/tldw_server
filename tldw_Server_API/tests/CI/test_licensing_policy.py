from __future__ import annotations

from hashlib import sha256
import json
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


def _read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def test_verbatim_license_corpus_matches_reviewed_upstream_bytes() -> None:
    for path, expected_digest in LICENSE_DIGESTS.items():
        actual_digest = sha256(Path(path).read_bytes()).hexdigest()
        assert actual_digest == expected_digest, path


def test_root_license_maps_only_the_approved_protected_paths() -> None:
    text = _read("LICENSE")
    for path in PROTECTED_PATHS:
        assert f"`{path}`" in text
    assert "`apps/**`" not in text
    assert "PolyForm Perimeter License 1.0.1" in text
    assert "GPL-3.0-only" in text
    assert "Apache-2.0" in text
    assert "previously public" in text
    assert "No trademark rights are granted" in text


def test_historical_record_preserves_public_refs_and_prior_grants() -> None:
    record = json.loads(_read("LICENSES/history/pre-source-available.json"))
    assert record["schema_version"] == 1
    assert record["repository"] == "https://github.com/rmusser01/tldw_server"
    assert record["public_refs"]["refs/heads/main"] == "7a23be3202e360f2d8e7cfe208e13ba406cf0507"
    assert record["public_refs"]["refs/heads/dev"] == "29acaca8c781213e27b12066372df13855e2e7a6"
    assert record["public_refs"]["refs/pull/2727/head"] == "60ce244fb6a65a79489b3f77299340afa501be24"
    assert record["prior_grants_preserved"] is True
    assert record["ref_snapshot_is_exhaustive"] is False


def test_countdown_template_is_not_misrepresented_as_an_active_grant() -> None:
    readme = _read("LICENSES/releases/README.md")
    assert "No protected frontend release may be published" in readme
    assert "completed release-specific Countdown grant" in readme
    assert not any(
        child.is_dir()
        for child in Path("LICENSES/releases").iterdir()
    )
