from pathlib import Path

ADR_ROOT = Path(__file__).resolve().parents[3] / "Docs" / "ADR"


def test_adr_numbers_are_unique() -> None:
    numbers = [path.name.split("-", 1)[0] for path in ADR_ROOT.glob("[0-9][0-9][0-9]-*.md")]
    # Pytest assertion verifies the repository's unique ADR identity contract.
    assert len(numbers) == len(set(numbers))  # nosec B101


def test_adr_index_lists_each_numbered_record() -> None:
    index = (ADR_ROOT / "README.md").read_text(encoding="utf-8")
    for path in sorted(ADR_ROOT.glob("[0-9][0-9][0-9]-*.md")):
        if path.name.startswith("000-"):
            continue
        # Pytest assertions preserve ADR discoverability and filename/title identity.
        assert f"]({path.name})" in index  # nosec B101
        assert path.read_text(encoding="utf-8").startswith(f"# ADR-{path.name[:3]}:")  # nosec B101
