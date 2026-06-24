"""Regression tests for media-ingestion URL logging and egress boundaries."""

import pytest
from _pytest.monkeypatch import MonkeyPatch


@pytest.mark.unit
def test_redact_url_for_log_strips_credentials_query_and_fragment() -> None:
    """Verify full URLs lose credentials, query strings, and fragments in logs."""
    from tldw_Server_API.app.core.Ingestion_Media_Processing.logging_safety import (
        redact_url_for_log,
    )

    redacted = redact_url_for_log(
        "https://user:pass@example.com:8443/video.mp4?token=secret#frag"
    )

    assert redacted == "https://example.com:8443/video.mp4"
    assert "user" not in redacted
    assert "pass" not in redacted
    assert "token" not in redacted
    assert "frag" not in redacted


@pytest.mark.unit
def test_redact_url_for_log_strips_schemeless_query_and_userinfo() -> None:
    """Verify scheme-less URLs are still redacted before logging."""
    from tldw_Server_API.app.core.Ingestion_Media_Processing.logging_safety import (
        redact_url_for_log,
    )

    redacted = redact_url_for_log("user:pass@vimeo.com/123?token=secret#frag")

    assert redacted == "vimeo.com/123"
    assert "user" not in redacted
    assert "pass" not in redacted
    assert "token" not in redacted
    assert "frag" not in redacted


@pytest.mark.unit
@pytest.mark.parametrize(
    "helper_name,args",
    [
        ("get_video_info", ("https://example.com/video",)),
        ("get_youtube", ("https://example.com/video",)),
        ("get_playlist_videos", ("https://example.com/playlist",)),
        ("extract_video_info", ("https://example.com/video",)),
        ("extract_metadata", ("https://example.com/video",)),
        ("get_youtube_playlist_urls", ("PL123",)),
    ],
)
def test_yt_dlp_metadata_helpers_block_disallowed_urls_before_network(
    monkeypatch: MonkeyPatch,
    helper_name: str,
    args: tuple[str, ...],
) -> None:
    """Verify blocked URLs fail before constructing yt-dlp clients."""
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Video import (
        Video_DL_Ingestion_Lib as video_lib,
    )

    class _Denied:
        """Egress policy result that denies the URL."""

        allowed = False
        reason = "blocked"

    class _UnexpectedYoutubeDL:
        """yt-dlp sentinel that fails if constructed."""

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            """Raise if egress checks allow construction."""
            raise AssertionError("yt-dlp should not be constructed for blocked URLs")

    monkeypatch.setattr(video_lib, "evaluate_url_policy", lambda *_args, **_kwargs: _Denied())
    monkeypatch.setattr(video_lib.yt_dlp, "YoutubeDL", _UnexpectedYoutubeDL)

    with pytest.raises(ValueError, match="blocked"):
        getattr(video_lib, helper_name)(*args)
