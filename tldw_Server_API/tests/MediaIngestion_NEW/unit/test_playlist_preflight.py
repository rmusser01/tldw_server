from collections import deque

import pytest

import tldw_Server_API.app.core.Ingestion_Media_Processing.Video.playlist_preflight as playlist_preflight
import tldw_Server_API.app.core.Ingestion_Media_Processing.Video.playlist_preflight_runner as playlist_preflight_runner
from tldw_Server_API.app.core.Ingestion_Media_Processing.Video.playlist_preflight import (
    PlaylistPreflightData,
    classify_playlist_url,
    extract_playlist_preflight,
    normalize_preflight_items,
    resolve_duplicate_policy_action,
)

pytestmark = pytest.mark.unit


def test_youtube_watch_list_url_detects_playlist_context():
    parsed = classify_playlist_url("https://www.youtube.com/watch?v=PrNmmN6qBiw&list=PL0065D9B288E6804B")

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


def test_preflight_item_without_source_is_not_selected():
    items = normalize_preflight_items(
        [
            {
                "ordinal": 1,
                "title": "Private unavailable talk",
            }
        ]
    )

    assert items[0].source_url == ""
    assert items[0].duplicate_status == "unknown"
    assert items[0].selected is False


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


def test_extract_playlist_preflight_configured_limit_plus_one_is_not_partial_success():
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

    with pytest.raises(playlist_preflight.PlaylistPreflightProcessError, match="playlist_too_large") as exc_info:
        extract_playlist_preflight(
            "https://www.youtube.com/playlist?list=PLtest",
            max_items=2,
            youtube_dl_cls=FakeYoutubeDL,
        )

    assert exc_info.value.code == "playlist_too_large"


def test_extract_playlist_preflight_warns_and_deselects_entry_without_source():
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
                    {"title": "Private unavailable talk"},
                ],
            }

    result = extract_playlist_preflight(
        "https://www.youtube.com/playlist?list=PLtest",
        max_items=10,
        youtube_dl_cls=FakeYoutubeDL,
    )

    assert result.item_count == 1
    assert result.selected_count == 0
    assert result.duplicate_count == 1
    assert result.warnings == ["Playlist entry 1 has no source URL."]
    assert result.items[0].duplicate_status == "unknown"
    assert result.items[0].selected is False


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


class _FakeRecvConnection:
    def __init__(self, messages=()):
        self.messages = deque(messages)
        self.closed = False

    def poll(self, _timeout=0.0):
        return bool(self.messages)

    def recv(self):
        return self.messages.popleft()

    def close(self):
        self.closed = True


class _FakeSendConnection:
    def __init__(self, recv):
        self.recv = recv
        self.closed = False

    def send(self, payload):
        self.recv.messages.append(payload)

    def close(self):
        self.closed = True


class _FakeProcess:
    def __init__(self, *, target, args, run_target=True, stubborn=False, exit_without_target=False):
        self.target = target
        self.args = args
        self.run_target = run_target
        self.stubborn = stubborn
        self.exit_without_target = exit_without_target
        self.alive = False
        self.started = False
        self.terminate_calls = 0
        self.kill_calls = 0
        self.join_calls = []
        self.closed = False

    def start(self):
        self.started = True
        self.alive = True
        if self.exit_without_target:
            self.alive = False
            return
        if self.run_target:
            self.target(*self.args)
            self.alive = False

    def is_alive(self):
        return self.alive

    def terminate(self):
        self.terminate_calls += 1
        if not self.stubborn:
            self.alive = False

    def kill(self):
        self.kill_calls += 1
        self.alive = False

    def join(self, timeout=None):
        self.join_calls.append(timeout)

    def close(self):
        self.closed = True


class _FakeSpawnContext:
    def __init__(
        self,
        *,
        messages=(),
        run_target=True,
        stubborn=False,
        exit_without_target=False,
        process_error=False,
    ):
        self.recv = _FakeRecvConnection(messages)
        self.send = _FakeSendConnection(self.recv)
        self.run_target = run_target
        self.stubborn = stubborn
        self.exit_without_target = exit_without_target
        self.process_error = process_error
        self.process = None

    def Pipe(self, *, duplex):
        assert duplex is False
        return self.recv, self.send

    def Process(self, *, target, args):
        if self.process_error:
            raise RuntimeError("local process capacity detail")
        self.process = _FakeProcess(
            target=target,
            args=args,
            run_target=self.run_target,
            stubborn=self.stubborn,
            exit_without_target=self.exit_without_target,
        )
        return self.process


@pytest.mark.asyncio
async def test_process_runner_returns_successful_normalized_extraction(monkeypatch):
    extracted = PlaylistPreflightData(
        source_url="https://www.youtube.com/playlist?list=PLtest",
        source_kind="youtube_playlist",
        playlist_id="PLtest",
        playlist_title="Conference",
        video_id=None,
        item_count=1,
        selected_count=1,
        duplicate_count=0,
        warnings=[],
        items=normalize_preflight_items(
            [{"id": "abc123", "source_url": "https://youtu.be/abc123", "title": "Opening"}]
        ),
    )
    calls = []

    def fake_extract(url, *, max_items):
        calls.append((url, max_items))
        return extracted

    monkeypatch.setattr(playlist_preflight, "extract_playlist_preflight", fake_extract, raising=True)
    context = _FakeSpawnContext()

    result = await playlist_preflight_runner.run_playlist_preflight_process(
        extracted.source_url,
        max_items=10,
        timeout_seconds=1,
        mp_context=context,
    )

    assert calls == [(extracted.source_url, 10)]
    assert result.items[0].source_url == "https://www.youtube.com/watch?v=abc123"
    assert context.recv.closed is True
    assert context.send.closed is True
    assert context.process.join_calls
    assert context.process.closed is True


@pytest.mark.asyncio
async def test_process_runner_timeout_terminates_then_kills_stubborn_child():
    context = _FakeSpawnContext(run_target=False, stubborn=True)

    with pytest.raises(playlist_preflight_runner.PlaylistPreflightProcessError, match="playlist_preflight_timeout"):
        await playlist_preflight_runner.run_playlist_preflight_process(
            "https://www.youtube.com/playlist?list=PLtest",
            max_items=10,
            timeout_seconds=0.001,
            poll_interval_seconds=0.001,
            join_timeout_seconds=0.001,
            mp_context=context,
        )

    assert context.process.terminate_calls == 1
    assert context.process.kill_calls == 1
    assert len(context.process.join_calls) >= 2
    assert context.recv.closed is True
    assert context.send.closed is True
    assert context.process.closed is True


@pytest.mark.asyncio
async def test_process_runner_process_creation_failure_closes_pipe_with_safe_error():
    context = _FakeSpawnContext(process_error=True)

    with pytest.raises(
        playlist_preflight_runner.PlaylistPreflightProcessError,
        match="playlist_preflight_capacity_unavailable",
    ):
        await playlist_preflight_runner.run_playlist_preflight_process(
            "https://www.youtube.com/playlist?list=PLtest",
            max_items=10,
            timeout_seconds=1,
            mp_context=context,
        )

    assert context.recv.closed is True
    assert context.send.closed is True


@pytest.mark.asyncio
async def test_process_runner_cancellation_terminates_child_without_kill():
    context = _FakeSpawnContext(run_target=False)

    with pytest.raises(playlist_preflight_runner.PlaylistPreflightProcessError, match="playlist_preflight_cancelled"):
        await playlist_preflight_runner.run_playlist_preflight_process(
            "https://www.youtube.com/playlist?list=PLtest",
            max_items=10,
            timeout_seconds=1,
            cancel_check=lambda: True,
            poll_interval_seconds=0.001,
            join_timeout_seconds=0.001,
            mp_context=context,
        )

    assert context.process.terminate_calls == 1
    assert context.process.kill_calls == 0
    assert context.process.join_calls


@pytest.mark.asyncio
async def test_process_runner_cancel_check_failure_is_safe_and_terminates_child():
    context = _FakeSpawnContext(run_target=False)

    def failed_cancel_check():
        raise RuntimeError("token=do-not-expose")

    with pytest.raises(playlist_preflight_runner.PlaylistPreflightProcessError) as exc_info:
        await playlist_preflight_runner.run_playlist_preflight_process(
            "https://www.youtube.com/playlist?list=PLtest&token=secret",
            max_items=10,
            timeout_seconds=1,
            cancel_check=failed_cancel_check,
            mp_context=context,
        )

    assert exc_info.value.code == "playlist_preflight_cancelled"
    assert "do-not-expose" not in str(exc_info.value)
    assert context.process.terminate_calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "messages",
    [
        (["not-a-mapping"]),
        ([{"status": "ok", "result": []}]),
        ([{"status": "ok", "result": {}}, {"status": "error", "code": "second"}]),
    ],
)
async def test_process_runner_rejects_malformed_or_multiple_child_payloads(messages):
    context = _FakeSpawnContext(messages=messages, run_target=False, exit_without_target=True)
    context.process = None

    with pytest.raises(playlist_preflight_runner.PlaylistPreflightProcessError, match="playlist_preflight_invalid_result"):
        await playlist_preflight_runner.run_playlist_preflight_process(
            "https://www.youtube.com/playlist?list=PLtest",
            max_items=10,
            timeout_seconds=1,
            mp_context=context,
        )

    assert context.recv.closed is True
    assert context.send.closed is True
    assert context.process.closed is True


@pytest.mark.asyncio
async def test_process_runner_rejects_inconsistent_child_counts():
    extracted = PlaylistPreflightData(
        source_url="https://www.youtube.com/playlist?list=PLtest",
        source_kind="youtube_playlist",
        playlist_id="PLtest",
        playlist_title="Conference",
        video_id=None,
        item_count=1,
        selected_count=1,
        duplicate_count=0,
        warnings=[],
        items=normalize_preflight_items([{"source_url": "https://youtu.be/abc123"}]),
    )
    payload = extracted.to_dict()
    payload["selected_count"] = 0
    context = _FakeSpawnContext(
        messages=[{"status": "ok", "result": payload}],
        run_target=False,
        exit_without_target=True,
    )

    with pytest.raises(
        playlist_preflight_runner.PlaylistPreflightProcessError,
        match="playlist_preflight_invalid_result",
    ):
        await playlist_preflight_runner.run_playlist_preflight_process(
            extracted.source_url,
            max_items=10,
            timeout_seconds=1,
            mp_context=context,
        )
