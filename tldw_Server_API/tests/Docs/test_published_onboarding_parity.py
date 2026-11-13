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


def test_adr_number_identifies_one_document_in_each_catalog() -> None:
    """A merged ADR must not silently share an identifier with an unrelated proposal."""
    for catalog in (Path("Docs/ADR"), Path("Docs/Published/ADR")):
        identities: dict[str, list[str]] = {}
        for path in catalog.glob("[0-9][0-9][0-9]-*.md"):
            identities.setdefault(path.name.split("-", 1)[0], []).append(path.name)
        duplicates = {number: names for number, names in identities.items() if len(names) > 1}
        assert not duplicates, f"{catalog} has ambiguous ADR identities: {duplicates}"  # nosec B101
