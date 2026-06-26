import json
import types

import pytest


@pytest.fixture()
def recorder_and_stub(monkeypatch):
    """Provide a simple stub for yt_dlp.YoutubeDL that records init options."""
    monkeypatch.setenv(
        "WORKFLOWS_EGRESS_ALLOWLIST",
        "example.com,youtu.be,youtube.com,www.youtube.com",
    )

    class Recorder:
        last_opts = None
        last_calls = []

    class FakeYoutubeDL:
        def __init__(self, opts):
            Recorder.last_opts = opts

        def __enter__(self):

            return self

        def __exit__(self, exc_type, exc, tb):

            return False

        def extract_info(self, url, download=False):

            Recorder.last_calls.append((url, bool(download)))
            return {"ok": True, "url": url}

    # Patch the module's yt_dlp to our stub (module with YoutubeDL attr)
    import tldw_Server_API.app.core.Ingestion_Media_Processing.Video.Video_DL_Ingestion_Lib as vmod
    stub_mod = types.SimpleNamespace(YoutubeDL=FakeYoutubeDL)
    monkeypatch.setattr(vmod, "yt_dlp", stub_mod)
    # Return both the recorder and the target module
    return Recorder, vmod


def _cookie_header_from_opts(opts):


    headers = (opts or {}).get("http_headers") or {}
    return headers.get("Cookie")


def test_get_video_info_sets_cookie_header(recorder_and_stub):


    R, vmod = recorder_and_stub
    cookies = {"a": "1", "b": "two"}
    vmod.get_video_info("https://example.com/v", use_cookies=True, cookies=cookies)
    assert R.last_opts is not None
    cookie_val = _cookie_header_from_opts(R.last_opts)
    # Order-preserving for dict literal; accept either order defensively
    assert cookie_val in {"a=1; b=two", "b=two; a=1"}


def test_get_youtube_sets_cookie_header_json_string(recorder_and_stub):


    R, vmod = recorder_and_stub
    cookies_json = json.dumps({"sid": "abc", "session": "xyz"})
    vmod.get_youtube("https://youtu.be/abc", use_cookies=True, cookies=cookies_json)
    assert R.last_opts is not None
    cookie_val = _cookie_header_from_opts(R.last_opts)
    assert cookie_val in {"sid=abc; session=xyz", "session=xyz; sid=abc"}


def test_get_playlist_videos_no_cookie_if_disabled(recorder_and_stub):


    R, vmod = recorder_and_stub
    vmod.get_playlist_videos("https://youtube.com/playlist?list=PL123", use_cookies=False, cookies={"k": "v"})
    assert R.last_opts is not None
    cookie_val = _cookie_header_from_opts(R.last_opts)
    assert cookie_val is None
