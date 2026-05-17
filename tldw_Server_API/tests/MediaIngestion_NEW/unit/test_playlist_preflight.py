import pytest

from tldw_Server_API.app.core.Ingestion_Media_Processing.Video.playlist_preflight import (
    classify_playlist_url,
    extract_playlist_preflight,
    normalize_preflight_items,
    resolve_duplicate_policy_action,
)


pytestmark = pytest.mark.unit


def test_youtube_watch_list_url_detects_playlist_context():
    parsed = classify_playlist_url(
        "https://www.youtube.com/watch?v=PrNmmN6qBiw&list=PL0065D9B288E6804B"
    )

    assert parsed.source_kind == "youtube_watch_playlist"
    assert parsed.playlist_id == "PL0065D9B288E6804B"
    assert parsed.video_id == "PrNmmN6qBiw"
    assert parsed.is_playlist is True


def test_preflight_duplicate_in_batch_uses_normalized_source_id():
    items = normalize_preflight_items(
        [
            {"source_url": "https://youtu.be/abc123", "title": "A"},
            {"source_url": "https://www.youtube.com/watch?v=abc123", "title": "A duplicate"},
        ]
    )

    assert [item.duplicate_status for item in items] == [
        "new",
        "duplicate_in_batch",
    ]
    assert items[1].duplicate_of_ordinal == 1


def test_generic_entry_id_is_not_assumed_to_be_youtube_video():
    items = normalize_preflight_items(
        [
            {
                "id": "generic-entry-123",
                "url": "generic-entry-123",
                "title": "Conference archive page",
            }
        ]
    )

    assert items[0].source_url == "generic-entry-123"
    assert items[0].normalized_source_id == "url:generic-entry-123"


def test_extract_playlist_preflight_rejects_single_video_url():
    class FakeYoutubeDL:
        def __init__(self, _opts):
            pass

        def __enter__(self):
            raise AssertionError("single-video URLs should fail before yt-dlp extraction")

        def __exit__(self, *_args):
            return False

    with pytest.raises(ValueError, match="not_playlist_url"):
        extract_playlist_preflight(
            "https://www.youtube.com/watch?v=abc123",
            youtube_dl_cls=FakeYoutubeDL,
        )


def test_extract_playlist_preflight_uses_metadata_only_ytdlp_options():
    calls = []

    class FakeYoutubeDL:
        def __init__(self, opts):
            self.opts = opts

        def __enter__(self):
            calls.append(("enter", self.opts))
            return self

        def __exit__(self, *_args):
            return False

        def extract_info(self, url, *, download):
            calls.append(("extract", url, download))
            return {
                "id": "PLtest",
                "title": "Conference 2010",
                "webpage_url": "https://www.youtube.com/playlist?list=PLtest",
                "entries": [
                    {
                        "id": "abc123",
                        "url": "abc123",
                        "title": "Opening Keynote",
                        "duration": 120,
                        "channel": "Conference Org",
                    },
                    {
                        "id": "def456",
                        "webpage_url": "https://www.youtube.com/watch?v=def456",
                        "title": "Systems Talk",
                        "duration": 240,
                    },
                ],
            }

    result = extract_playlist_preflight(
        "https://www.youtube.com/playlist?list=PLtest",
        max_items=10,
        youtube_dl_cls=FakeYoutubeDL,
    )

    assert result.playlist_id == "PLtest"
    assert result.playlist_title == "Conference 2010"
    assert result.item_count == 2
    assert [item.source_url for item in result.items] == [
        "https://www.youtube.com/watch?v=abc123",
        "https://www.youtube.com/watch?v=def456",
    ]
    assert calls[0][1]["skip_download"] is True
    assert calls[0][1]["extract_flat"] is True
    assert calls[1] == (
        "extract",
        "https://www.youtube.com/playlist?list=PLtest",
        False,
    )


def test_extract_playlist_preflight_truncates_large_playlist():
    class FakeYoutubeDL:
        def __init__(self, _opts):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def extract_info(self, _url, *, download):
            assert download is False
            return {
                "id": "PLtest",
                "title": "Conference 2010",
                "entries": [
                    {"id": "abc123", "url": "abc123", "title": "Opening"},
                    {"id": "def456", "url": "def456", "title": "Systems"},
                    {"id": "ghi789", "url": "ghi789", "title": "Closing"},
                ],
            }

    result = extract_playlist_preflight(
        "https://www.youtube.com/playlist?list=PLtest",
        max_items=2,
        youtube_dl_cls=FakeYoutubeDL,
    )

    assert result.item_count == 2
    assert result.selected_count == 2
    assert result.warnings == ["Playlist truncated to 2 items."]


@pytest.mark.parametrize(
    ("policy", "expected_status", "expected_submit"),
    [
        ("skip", "skipped_existing", False),
        ("overwrite", "planned", True),
        ("update_metadata_only", "skipped_existing", False),
        ("include_existing", "skipped_existing", False),
    ],
)
def test_duplicate_policy_action_for_existing_duplicate(policy, expected_status, expected_submit):
    action = resolve_duplicate_policy_action(
        duplicate_status="duplicate_existing",
        policy=policy,
    )

    assert action.planned_status == expected_status
    assert action.should_submit_job is expected_submit


def test_duplicate_policy_action_does_not_skip_new_items():
    action = resolve_duplicate_policy_action(
        duplicate_status="new",
        policy="skip",
    )

    assert action.planned_status == "planned"
    assert action.should_submit_job is True


def test_duplicate_policy_action_does_not_skip_unknown_items():
    action = resolve_duplicate_policy_action(
        duplicate_status="unknown",
        policy="skip",
    )

    assert action.planned_status == "planned"
    assert action.should_submit_job is True
