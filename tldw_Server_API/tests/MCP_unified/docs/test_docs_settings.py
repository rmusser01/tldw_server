from __future__ import annotations

from pathlib import Path

import pytest

from mcp_unified.docs.settings import DocsSettings


@pytest.mark.parametrize("value", ["false", "0", "no", "off", False])
def test_from_mapping_coerces_false_values_for_web_acquisition(value: str | bool) -> None:
    settings = DocsSettings.from_mapping({"enable_web_acquisition": value})

    assert settings.enable_web_acquisition is False  # nosec B101


@pytest.mark.parametrize("value", ["true", "1", "yes", "on", True])
def test_from_mapping_coerces_true_values_for_web_acquisition(value: str | bool) -> None:
    settings = DocsSettings.from_mapping({"enable_web_acquisition": value})

    assert settings.enable_web_acquisition is True  # nosec B101


def test_from_mapping_rejects_unknown_web_acquisition_string() -> None:
    with pytest.raises(ValueError, match="enable_web_acquisition"):
        DocsSettings.from_mapping({"enable_web_acquisition": "sometimes"})


def test_from_mapping_accepts_single_trusted_root_path_value(tmp_path: Path) -> None:
    root = tmp_path / "docs"
    root.mkdir()

    settings = DocsSettings.from_mapping({"trusted_roots": str(root)})

    assert settings.trusted_roots == (root.resolve(),)  # nosec B101


def test_from_mapping_accepts_iterable_trusted_root_values(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()

    settings = DocsSettings.from_mapping({"trusted_roots": [first, str(second)]})

    assert settings.trusted_roots == (first.resolve(), second.resolve())  # nosec B101


@pytest.mark.parametrize("value", [0, -1, "-20"])
def test_from_mapping_rejects_non_positive_max_import_file_bytes(value: int | str) -> None:
    with pytest.raises(ValueError, match="max_import_file_bytes"):
        DocsSettings.from_mapping({"max_import_file_bytes": value})
