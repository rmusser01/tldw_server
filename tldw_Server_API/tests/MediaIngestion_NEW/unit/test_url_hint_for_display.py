"""User-facing URL hints: enough to tell which source failed, without its secrets."""

import pytest

from tldw_Server_API.app.core.Ingestion_Media_Processing.logging_safety import url_hint_for_display

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("url", "hint"),
    [
        ("https://example.com/news/article-42", "example.com/news/article-42"),
        ("https://example.com/", "example.com"),
        ("https://example.com", "example.com"),
        # Credentials, port, query and fragment are where tokens live: all dropped.
        ("https://user:pass@example.com:8443/a?token=secret#frag", "example.com/a"),
        # Long paths keep only their tail, so a path-embedded token is truncated.
        (
            "https://blog.example.org/2026/09/28/a-very-long-article-slug-that-keeps-going",
            "blog.example.org/…ng-article-slug-that-keeps-going",
        ),
        # Readable slugs survive, digits and all.
        ("https://example.com/news/my-article-2026-09-28", "example.com/news/my-article-2026-09-28"),
        # Token-looking segments are masked wherever they sit in the path.
        ("https://example.com/share/Ab3dE9fGh1JkLmN0pQrS", "example.com/share/…"),
        ("https://example.com/d/d41d8cd98f00b204e9800998ecf8427e/file.pdf", "example.com/d/…/file.pdf"),
        ("https://example.com/i/550e8400-e29b-41d4-a716-446655440000", "example.com/i/…"),
        (
            "https://cdn.example.com/files/report/eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.sig",
            "cdn.example.com/files/report/…",
        ),
    ],
)
def test_hint_is_host_plus_path_tail(url: str, hint: str) -> None:
    """Host plus at most the last 32 path characters; never query, fragment or userinfo."""
    assert url_hint_for_display(url) == hint


@pytest.mark.parametrize("value", [None, "", "not a url", "https://[bad", "/relative/path"])
def test_no_hint_without_a_host(value: object) -> None:
    """No host means nothing safe and useful to show."""
    assert url_hint_for_display(value) is None
