from pathlib import Path

import yaml


def test_published_has_same_profile_pages_as_manifest() -> None:
    manifest = yaml.safe_load(Path("Docs/Getting_Started/onboarding_manifest.yaml").read_text())
    for _, meta in manifest["profiles"].items():
        published = Path("Docs/Published") / meta["published_path"]
        assert published.exists(), f"Missing published mirror: {published}"


def test_published_getting_started_has_no_local_filesystem_links() -> None:
    forbidden = ("/Users/", "macbook-dev", "appledev")
    for path in Path("Docs/Published/Getting_Started").glob("*.md"):
        text = path.read_text(encoding="utf-8")
        for snippet in forbidden:
            assert snippet not in text, f"{path} contains local filesystem reference {snippet}"
