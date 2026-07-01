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


def test_from_mapping_uses_safe_url_acquisition_defaults() -> None:
    settings = DocsSettings.from_mapping({})

    assert settings.enable_web_acquisition is False  # nosec B101
    assert settings.web_source_profile == "locked_down"  # nosec B101
    assert settings.preapproved_domains == ()  # nosec B101
    assert settings.allowed_url_prefixes == ()  # nosec B101
    assert settings.denied_domains == ()  # nosec B101
    assert settings.max_url_redirects == 3  # nosec B101
    assert settings.max_url_body_bytes == 2_000_000  # nosec B101
    assert settings.url_request_timeout_seconds == 10.0  # nosec B101
    assert "text/html" in settings.allowed_content_types  # nosec B101
    assert settings.respect_robots is False  # nosec B101
    assert settings.allow_arbitrary_public_domains is False  # nosec B101


def test_from_mapping_parses_url_acquisition_values() -> None:
    settings = DocsSettings.from_mapping(
        {
            "enable_web_acquisition": "true",
            "web_source_profile": "local_first",
            "preapproved_domains": "docs.python.org",
            "allowed_url_prefixes": ["https://docs.python.org/3/"],
            "denied_domains": ["blocked.example"],
            "max_url_redirects": "5",
            "max_url_body_bytes": "4096",
            "url_request_timeout_seconds": "2.5",
            "allowed_content_types": "text/plain",
            "url_user_agent": "tldw-docs-test/1",
            "respect_robots": "true",
            "allow_arbitrary_public_domains": "false",
        }
    )

    assert settings.web_source_profile == "local_first"  # nosec B101
    assert settings.preapproved_domains == ("docs.python.org",)  # nosec B101
    assert settings.allowed_url_prefixes == ("https://docs.python.org/3/",)  # nosec B101
    assert settings.denied_domains == ("blocked.example",)  # nosec B101
    assert settings.max_url_redirects == 5  # nosec B101
    assert settings.max_url_body_bytes == 4096  # nosec B101
    assert settings.url_request_timeout_seconds == 2.5  # nosec B101
    assert settings.allowed_content_types == ("text/plain",)  # nosec B101
    assert settings.url_user_agent == "tldw-docs-test/1"  # nosec B101
    assert settings.respect_robots is True  # nosec B101


@pytest.mark.parametrize("profile", ["", "open", "offline", "LOCAL_FIRST"])
def test_from_mapping_rejects_unknown_web_source_profile(profile: str) -> None:
    with pytest.raises(ValueError, match="web_source_profile"):
        DocsSettings.from_mapping({"web_source_profile": profile})


@pytest.mark.parametrize("field", ["max_url_redirects", "max_url_body_bytes"])
def test_from_mapping_rejects_non_positive_url_limits(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        DocsSettings.from_mapping({field: 0})


def test_from_mapping_rejects_non_positive_url_timeout() -> None:
    with pytest.raises(ValueError, match="url_request_timeout_seconds"):
        DocsSettings.from_mapping({"url_request_timeout_seconds": 0})


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
