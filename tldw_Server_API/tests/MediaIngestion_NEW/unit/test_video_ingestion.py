from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import wave

import pytest

from tldw_Server_API.app.core.Ingestion_Media_Processing.Video.Video_DL_Ingestion_Lib import (
    parse_and_expand_urls,
    _resolve_eval_api_key,
    process_videos,
    process_single_video,
    _cookies_to_header_value,
)
from tldw_Server_API.app.core.Ingestion_Media_Processing.Video import Video_DL_Ingestion_Lib as video_lib


@pytest.mark.unit
def test_parse_and_expand_urls_preserves_non_vimeo_entries():
    local_path = "/Users/example/video.mp4"
    generic_url = "https://example.com/video.mp4"

    expanded = parse_and_expand_urls([local_path, generic_url])

    assert local_path in expanded
    assert generic_url in expanded


@pytest.mark.unit
def test_parse_and_expand_urls_normalizes_vimeo():
    vimeo_url = "http://vimeo.com/12345"
    expanded = parse_and_expand_urls([vimeo_url])
    assert expanded == ["https://vimeo.com/12345"]


@pytest.mark.unit
def test_parse_and_expand_urls_ignores_non_youtube_list_queries():
    tricky_url = "https://notyoutube.com/watch?list=foo"
    expanded = parse_and_expand_urls([tricky_url])
    assert expanded == [tricky_url]


@pytest.mark.unit
def test_parse_and_expand_urls_respects_vimeo_netloc_and_query():
    vimeo_url = "https://player.vimeo.com/video/12345?h=abc123"
    expanded = parse_and_expand_urls([vimeo_url])
    assert expanded == [vimeo_url]


@pytest.mark.unit
def test_cookies_to_header_value_supports_netscape_export():
    netscape_blob = """# Netscape HTTP Cookie File\n#HttpOnly_example.com\tTRUE\t/\tFALSE\t0\tsessionid\tabc123\nexample.com\tTRUE\t/\tTRUE\t0\tcsrftoken\txyz456\n"""
    header = _cookies_to_header_value(netscape_blob)
    assert header == "sessionid=abc123; csrftoken=xyz456"


@pytest.mark.unit
@patch("tldw_Server_API.app.core.Ingestion_Media_Processing.Video.Video_DL_Ingestion_Lib._resolve_eval_api_key", return_value="resolved-key")
@patch("tldw_Server_API.app.core.Ingestion_Media_Processing.Video.Video_DL_Ingestion_Lib.run_geval")
@patch("tldw_Server_API.app.core.Ingestion_Media_Processing.Video.Video_DL_Ingestion_Lib.process_single_video")
def test_confabulation_check_invokes_geval(mock_single, mock_geval, _mock_resolve_key, tmp_path):
    transcript_text = "this is a transcript"
    summary_text = "summary content"

    mock_single.return_value = {
        "status": "Success",
        "input_ref": "https://example.com/video",
        "processing_source": "https://example.com/video",
        "media_type": "video",
        "metadata": {},
        "content": transcript_text,
        "segments": [],
        "chunks": [],
        "analysis": summary_text,
        "analysis_details": {},
        "error": None,
        "warnings": [],
    }
    mock_geval.return_value = {"metrics": {"coherence": 5}}

    result = process_videos(
        inputs=["https://example.com/video"],
        start_time=None,
        end_time=None,
        diarize=False,
        vad_use=False,
        transcription_model="whisper-small",
        transcription_language="en",
        perform_analysis=False,
        custom_prompt=None,
        system_prompt=None,
        perform_chunking=False,
        chunk_method=None,
        max_chunk_size=1000,
        chunk_overlap=0,
        use_adaptive_chunking=False,
        use_multi_level_chunking=False,
        chunk_language=None,
        summarize_recursively=False,
        api_name="openai",
        use_cookies=False,
        cookies=None,
        timestamp_option=False,
        perform_confabulation_check=True,
        temp_dir=str(tmp_path),
        keep_original=False,
        perform_diarization=False,
    )

    mock_geval.assert_called_once()
    args, kwargs = mock_geval.call_args
    assert args[:4] == (transcript_text, summary_text, "resolved-key", "openai")
    assert kwargs.get("user_identifier") is None
    assert result["processed_count"] == 1
    assert result["confabulation_results"].startswith("Confabulation checks completed")


@pytest.mark.unit
@patch("tldw_Server_API.app.core.Ingestion_Media_Processing.Video.Video_DL_Ingestion_Lib._resolve_eval_api_key", return_value=None)
@patch("tldw_Server_API.app.core.Ingestion_Media_Processing.Video.Video_DL_Ingestion_Lib.run_geval")
@patch("tldw_Server_API.app.core.Ingestion_Media_Processing.Video.Video_DL_Ingestion_Lib.process_single_video")
def test_confabulation_requires_keys_for_commercial_providers(mock_single, mock_geval, _mock_resolve_key, tmp_path):
    transcript_text = "doc transcript"
    summary_text = "summary text"

    mock_single.return_value = {
        "status": "Success",
        "input_ref": "https://example.com/video",
        "processing_source": "https://example.com/video",
        "media_type": "video",
        "metadata": {},
        "content": transcript_text,
        "segments": [],
        "chunks": [],
        "analysis": summary_text,
        "analysis_details": {},
        "error": None,
        "warnings": [],
    }

    result = process_videos(
        inputs=["https://example.com/video"],
        start_time=None,
        end_time=None,
        diarize=False,
        vad_use=False,
        transcription_model="whisper-small",
        transcription_language="en",
        perform_analysis=False,
        custom_prompt=None,
        system_prompt=None,
        perform_chunking=False,
        chunk_method=None,
        max_chunk_size=1000,
        chunk_overlap=0,
        use_adaptive_chunking=False,
        use_multi_level_chunking=False,
        chunk_language=None,
        summarize_recursively=False,
        api_name="google",
        use_cookies=False,
        cookies=None,
        timestamp_option=False,
        perform_confabulation_check=True,
        temp_dir=str(tmp_path),
        keep_original=False,
        perform_diarization=False,
    )

    mock_geval.assert_not_called()
    assert result["confabulation_results"] == "Confabulation check skipped: missing API key for provider 'google'."


@pytest.mark.unit
@patch("tldw_Server_API.app.core.Ingestion_Media_Processing.Video.Video_DL_Ingestion_Lib._resolve_eval_api_key", return_value=None)
@patch("tldw_Server_API.app.core.Ingestion_Media_Processing.Video.Video_DL_Ingestion_Lib.run_geval")
@patch("tldw_Server_API.app.core.Ingestion_Media_Processing.Video.Video_DL_Ingestion_Lib.process_single_video")
def test_confabulation_allows_keyless_provider(mock_single, mock_geval, _mock_resolve_key, tmp_path):
    transcript_text = "local transcript"
    summary_text = "summary via local model"

    mock_single.return_value = {
        "status": "Success",
        "input_ref": "https://local.test/video",
        "processing_source": "https://local.test/video",
        "media_type": "video",
        "metadata": {},
        "content": transcript_text,
        "segments": [],
        "chunks": [],
        "analysis": summary_text,
        "analysis_details": {},
        "error": None,
        "warnings": [],
    }

    result = process_videos(
        inputs=["https://local.test/video"],
        start_time=None,
        end_time=None,
        diarize=False,
        vad_use=False,
        transcription_model="whisper-small",
        transcription_language="en",
        perform_analysis=False,
        custom_prompt=None,
        system_prompt=None,
        perform_chunking=False,
        chunk_method=None,
        max_chunk_size=1000,
        chunk_overlap=0,
        use_adaptive_chunking=False,
        use_multi_level_chunking=False,
        chunk_language=None,
        summarize_recursively=False,
        api_name="llama.cpp",
        use_cookies=False,
        cookies=None,
        timestamp_option=False,
        perform_confabulation_check=True,
        temp_dir=str(tmp_path),
        keep_original=False,
        perform_diarization=False,
        user_id=42,
    )

    mock_geval.assert_called_once()
    args, kwargs = mock_geval.call_args
    assert args[:3] == (transcript_text, summary_text, None)
    assert args[3] == "llama.cpp"
    assert kwargs.get("user_identifier") == "42"
    assert result["processed_count"] == 1
    assert result["confabulation_results"].startswith("Confabulation checks completed")


@pytest.mark.unit
@patch("tldw_Server_API.app.core.Ingestion_Media_Processing.Video.Video_DL_Ingestion_Lib.process_single_video")
def test_process_videos_sanitizes_item_processing_failure(mock_single, tmp_path):
    mock_single.side_effect = RuntimeError("video parser exploded at /private/cache/video.mp4")

    result = process_videos(
        inputs=["https://example.com/video.mp4"],
        start_time=None,
        end_time=None,
        diarize=False,
        vad_use=False,
        transcription_model="whisper-small",
        transcription_language="en",
        perform_analysis=False,
        custom_prompt=None,
        system_prompt=None,
        perform_chunking=False,
        chunk_method=None,
        max_chunk_size=1000,
        chunk_overlap=0,
        use_adaptive_chunking=False,
        use_multi_level_chunking=False,
        chunk_language=None,
        summarize_recursively=False,
        api_name=None,
        use_cookies=False,
        cookies=None,
        timestamp_option=False,
        perform_confabulation_check=False,
        temp_dir=str(tmp_path),
        keep_original=False,
        perform_diarization=False,
    )

    item = result["results"][0]
    assert result["errors"] == ["Video processing failed"]
    assert item["status"] == "Error"
    assert item["error"] == "Video processing failed"
    assert "video parser exploded" not in result["errors"][0]
    assert "/private/cache/video.mp4" not in result["errors"][0]


@pytest.mark.unit
def test_process_single_video_sanitizes_unexpected_processing_failure(monkeypatch, tmp_path):
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"\x00" * 2048)

    def _fail_transcription(**_kwargs):
        raise RuntimeError("video transcription exploded at /private/cache/clip.wav")

    monkeypatch.setattr(video_lib, "perform_transcription", _fail_transcription)

    result = process_single_video(
        video_input=str(video_path),
        start_seconds=0,
        end_seconds=None,
        diarize=False,
        vad_use=False,
        transcription_model="base",
        transcription_language="en",
        perform_analysis=False,
        custom_prompt=None,
        system_prompt=None,
        perform_chunking=False,
        chunk_method=None,
        max_chunk_size=1000,
        chunk_overlap=0,
        use_adaptive_chunking=False,
        use_multi_level_chunking=False,
        chunk_language=None,
        summarize_recursively=False,
        api_name=None,
        use_cookies=False,
        cookies=None,
        timestamp_option=False,
        temp_dir=str(tmp_path),
    )

    assert result["status"] == "Error"
    assert result["error"] == "Video processing failed"
    assert "video transcription exploded" not in str(result)
    assert "/private/cache/clip.wav" not in str(result)


@pytest.mark.unit
@patch("tldw_Server_API.app.core.Ingestion_Media_Processing.Video.Video_DL_Ingestion_Lib.parse_and_expand_urls")
@patch("tldw_Server_API.app.core.Ingestion_Media_Processing.Video.Video_DL_Ingestion_Lib.process_single_video")
def test_process_videos_expands_playlist_inputs(mock_single, mock_expand, tmp_path):
    mock_expand.return_value = ["https://example.com/watch?v=a", "https://example.com/watch?v=b"]
    mock_single.side_effect = [
        {
            "status": "Success",
            "input_ref": "https://example.com/watch?v=a",
            "processing_source": "https://example.com/watch?v=a",
            "media_type": "video",
            "metadata": {},
            "content": "transcript a",
            "segments": [],
            "chunks": [],
            "analysis": None,
            "analysis_details": {},
            "error": None,
            "warnings": [],
        },
        {
            "status": "Success",
            "input_ref": "https://example.com/watch?v=b",
            "processing_source": "https://example.com/watch?v=b",
            "media_type": "video",
            "metadata": {},
            "content": "transcript b",
            "segments": [],
            "chunks": [],
            "analysis": None,
            "analysis_details": {},
            "error": None,
            "warnings": [],
        },
    ]

    result = process_videos(
        inputs=["https://youtube.com/playlist?list=abc123"],
        start_time=None,
        end_time=None,
        diarize=False,
        vad_use=False,
        transcription_model="whisper-small",
        transcription_language="en",
        perform_analysis=False,
        custom_prompt=None,
        system_prompt=None,
        perform_chunking=False,
        chunk_method=None,
        max_chunk_size=1000,
        chunk_overlap=0,
        use_adaptive_chunking=False,
        use_multi_level_chunking=False,
        chunk_language=None,
        summarize_recursively=False,
        api_name=None,
        use_cookies=False,
        cookies=None,
        timestamp_option=False,
        perform_confabulation_check=False,
        temp_dir=str(tmp_path),
        keep_original=False,
        perform_diarization=False,
    )

    mock_expand.assert_called_once_with(["https://youtube.com/playlist?list=abc123"])
    assert mock_single.call_count == 2
    called_inputs = [call.kwargs["video_input"] for call in mock_single.call_args_list]
    assert called_inputs == ["https://example.com/watch?v=a", "https://example.com/watch?v=b"]
    assert result["processed_count"] == 2


@pytest.mark.unit
def test_resolve_eval_api_key_supports_configured_providers(monkeypatch):
    fake_config = {
        "openai_api": {"api_key": "cfg-openai"},
        "custom_openai_api": {"api_key": "cfg-custom1"},
        "custom_openai_api_2": {"api_key": "cfg-custom2"},
        "llama_api": {"api_key": "cfg-llama"},
    }
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.Video.Video_DL_Ingestion_Lib.loaded_config_data",
        fake_config,
        raising=False,
    )

    assert _resolve_eval_api_key("openai") == "cfg-openai"
    assert _resolve_eval_api_key("custom-openai-api") == "cfg-custom1"
    assert _resolve_eval_api_key("custom-openai-api-2") == "cfg-custom2"
    assert _resolve_eval_api_key("llama.cpp") == "cfg-llama"


@pytest.mark.unit
def test_resolve_eval_api_key_normalizes_environment_lookup(monkeypatch):
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.Video.Video_DL_Ingestion_Lib.loaded_config_data",
        {},
        raising=False,
    )
    monkeypatch.setenv("LLAMA_CPP_API_KEY", "env-llama")
    monkeypatch.setenv("CUSTOM_OPENAI2_API_KEY", "env-custom2")

    assert _resolve_eval_api_key("llama.cpp") == "env-llama"
    assert _resolve_eval_api_key("custom-openai-api-2") == "env-custom2"


@pytest.mark.unit
def test_resolve_eval_api_key_supports_numbered_custom_openai(monkeypatch):
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.Video.Video_DL_Ingestion_Lib.loaded_config_data",
        {"custom_openai_api_37": {"api_key": "cfg-custom37"}},
        raising=False,
    )
    assert _resolve_eval_api_key("custom-openai-api-37") == "cfg-custom37"

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.Video.Video_DL_Ingestion_Lib.loaded_config_data",
        {},
        raising=False,
    )
    monkeypatch.setenv("CUSTOM_OPENAI_API_KEY_37", "env-custom37")
    assert _resolve_eval_api_key("custom-openai-api-37") == "env-custom37"


@pytest.mark.unit
@patch("tldw_Server_API.app.core.Ingestion_Media_Processing.Video.Video_DL_Ingestion_Lib.yt_dlp.YoutubeDL")
def test_download_video_returns_downloaded_file(mock_ydl_cls, tmp_path):
    download_dir = tmp_path / "downloads"
    download_dir.mkdir()
    downloaded_file = download_dir / "sample.m4a"
    downloaded_file.write_bytes(b"\x00\x01")

    mock_ydl = mock_ydl_cls.return_value.__enter__.return_value
    mock_ydl.extract_info.return_value = {
        "requested_downloads": [
            {"filepath": str(downloaded_file)},
        ]
    }

    result = video_lib.download_video(
        video_url="https://example.com/video",
        download_path=str(download_dir),
        info_dict={"filesize": 1024},
        download_video_flag=False,
        use_cookies=False,
        cookies=None,
    )

    assert result == str(downloaded_file)
    mock_ydl.extract_info.assert_called_once_with("https://example.com/video", download=True)


@pytest.mark.unit
@patch("tldw_Server_API.app.core.Ingestion_Media_Processing.Video.Video_DL_Ingestion_Lib.yt_dlp.YoutubeDL")
def test_download_video_respects_declared_size(mock_ydl_cls, tmp_path):
    info_dict = {"filesize": video_lib.DEFAULT_MAX_VIDEO_FILE_SIZE_BYTES + 1}

    with pytest.raises(ValueError):
        video_lib.download_video(
            video_url="https://example.com/video",
            download_path=str(tmp_path),
            info_dict=info_dict,
            download_video_flag=False,
            use_cookies=False,
            cookies=None,
        )

    mock_ydl_cls.assert_not_called()


@patch("tldw_Server_API.app.core.Ingestion_Media_Processing.Video.Video_DL_Ingestion_Lib.perform_transcription")
@patch("tldw_Server_API.app.core.Ingestion_Media_Processing.Video.Video_DL_Ingestion_Lib.download_video")
@patch("tldw_Server_API.app.core.Ingestion_Media_Processing.Video.Video_DL_Ingestion_Lib.extract_metadata")
def test_process_single_video_remote_download_success(
    mock_extract_metadata,
    mock_download,
    mock_transcribe,
    tmp_path,
):
    mock_extract_metadata.return_value = {"title": "Online Clip"}

    downloaded_file = tmp_path / "downloaded.wav"
    with wave.open(str(downloaded_file), "wb") as wave_file:
        wave_file.setnchannels(1)
        wave_file.setsampwidth(2)
        wave_file.setframerate(8000)
        wave_file.writeframes(b"\x00\x00" * 8)
    mock_download.return_value = str(downloaded_file)

    mock_transcribe.return_value = ("audio.wav", [{"Text": "hello", "Time_Start": 0, "Time_End": 1}])

    result = process_single_video(
        video_input="https://example.com/video",
        start_seconds=0,
        end_seconds=None,
        diarize=False,
        vad_use=False,
        transcription_model="whisper-small",
        transcription_language="en",
        perform_analysis=False,
        custom_prompt=None,
        system_prompt=None,
        perform_chunking=False,
        chunk_method=None,
        max_chunk_size=1000,
        chunk_overlap=0,
        use_adaptive_chunking=False,
        use_multi_level_chunking=False,
        chunk_language=None,
        summarize_recursively=False,
        api_name=None,
        use_cookies=False,
        cookies=None,
        timestamp_option=False,
        temp_dir=str(tmp_path),
        keep_intermediate_audio=False,
        perform_diarization=False,
    )

    assert result["status"] == "Success"
    assert result["processing_source"] == str(downloaded_file)
    mock_download.assert_called_once()


@patch("tldw_Server_API.app.core.Ingestion_Media_Processing.Video.Video_DL_Ingestion_Lib.perform_transcription")
@patch("tldw_Server_API.app.core.Ingestion_Media_Processing.Video.Video_DL_Ingestion_Lib.download_video")
@patch("tldw_Server_API.app.core.Ingestion_Media_Processing.Video.Video_DL_Ingestion_Lib.extract_metadata")
def test_process_single_video_remote_download_validation_rejected(
    mock_extract_metadata,
    mock_download,
    mock_transcribe,
    tmp_path,
):
    mock_extract_metadata.return_value = {"title": "Online Clip"}
    downloaded_file = tmp_path / "downloaded_payload.exe"
    downloaded_file.write_bytes(b"MZ")
    mock_download.return_value = str(downloaded_file)
    mock_transcribe.side_effect = AssertionError(
        "perform_transcription should not run when URL validation fails"
    )

    result = process_single_video(
        video_input="https://example.com/video",
        start_seconds=0,
        end_seconds=None,
        diarize=False,
        vad_use=False,
        transcription_model="whisper-small",
        transcription_language="en",
        perform_analysis=False,
        custom_prompt=None,
        system_prompt=None,
        perform_chunking=False,
        chunk_method=None,
        max_chunk_size=1000,
        chunk_overlap=0,
        use_adaptive_chunking=False,
        use_multi_level_chunking=False,
        chunk_language=None,
        summarize_recursively=False,
        api_name=None,
        use_cookies=False,
        cookies=None,
        timestamp_option=False,
        temp_dir=str(tmp_path),
        keep_intermediate_audio=False,
        perform_diarization=False,
    )

    assert result["status"] == "Error"
    assert "downloaded file failed validation" in str(result.get("error", "")).lower()


@patch("tldw_Server_API.app.core.Ingestion_Media_Processing.Video.Video_DL_Ingestion_Lib.perform_transcription")
@patch("tldw_Server_API.app.core.Ingestion_Media_Processing.Video.Video_DL_Ingestion_Lib.download_video")
@patch("tldw_Server_API.app.core.Ingestion_Media_Processing.Video.Video_DL_Ingestion_Lib.extract_metadata")
def test_process_single_video_remote_download_rejects_when_quota_exceeded(
    mock_extract_metadata,
    mock_download,
    mock_transcribe,
    monkeypatch,
    tmp_path,
):
    mock_extract_metadata.return_value = {"title": "Online Clip"}

    downloaded_file = tmp_path / "downloaded.wav"
    with wave.open(str(downloaded_file), "wb") as wave_file:
        wave_file.setnchannels(1)
        wave_file.setsampwidth(2)
        wave_file.setframerate(8000)
        wave_file.writeframes(b"\x00\x00" * 8)
    mock_download.return_value = str(downloaded_file)
    mock_transcribe.side_effect = AssertionError(
        "perform_transcription should not run when quota is exceeded"
    )

    class _RejectingQuotaService:
        async def check_quota(self, user_id: int, new_bytes: int, raise_on_exceed: bool = False):
            assert user_id == 42
            assert new_bytes == downloaded_file.stat().st_size
            return False, {
                "current_usage_mb": 10,
                "new_size_mb": 1,
                "quota_mb": 10,
                "available_mb": 0,
            }

    monkeypatch.setitem(
        __import__("sys").modules,
        "tldw_Server_API.app.services.storage_quota_service",
        SimpleNamespace(get_storage_quota_service=lambda: _RejectingQuotaService()),
    )

    result = process_single_video(
        video_input="https://example.com/video",
        start_seconds=0,
        end_seconds=None,
        diarize=False,
        vad_use=False,
        transcription_model="whisper-small",
        transcription_language="en",
        perform_analysis=False,
        custom_prompt=None,
        system_prompt=None,
        perform_chunking=False,
        chunk_method=None,
        max_chunk_size=1000,
        chunk_overlap=0,
        use_adaptive_chunking=False,
        use_multi_level_chunking=False,
        chunk_language=None,
        summarize_recursively=False,
        api_name=None,
        use_cookies=False,
        cookies=None,
        timestamp_option=False,
        temp_dir=str(tmp_path),
        keep_intermediate_audio=False,
        perform_diarization=False,
        user_id=42,
    )

    assert result["status"] == "Error"
    assert "storage quota exceeded" in str(result.get("error", "")).lower()


@patch("tldw_Server_API.app.core.Ingestion_Media_Processing.Video.Video_DL_Ingestion_Lib.process_single_video")
def test_process_videos_collects_warnings_separately(mock_single, tmp_path):
    warning_msg = "chunking skipped due to length"
    mock_single.return_value = {
        "status": "Warning",
        "input_ref": "https://example.com/video",
        "processing_source": "https://example.com/video",
        "media_type": "video",
        "metadata": {},
        "content": "partial transcript",
        "segments": [],
        "chunks": [],
        "analysis": None,
        "analysis_details": {},
        "error": None,
        "warnings": [warning_msg],
        "kept_video_path": None,
    }

    result = process_videos(
        inputs=["https://example.com/video"],
        start_time=None,
        end_time=None,
        diarize=False,
        vad_use=False,
        transcription_model="whisper-small",
        transcription_language="en",
        perform_analysis=False,
        custom_prompt=None,
        system_prompt=None,
        perform_chunking=False,
        chunk_method=None,
        max_chunk_size=1000,
        chunk_overlap=0,
        use_adaptive_chunking=False,
        use_multi_level_chunking=False,
        chunk_language=None,
        summarize_recursively=False,
        api_name=None,
        use_cookies=False,
        cookies=None,
        timestamp_option=False,
        perform_confabulation_check=False,
        temp_dir=str(tmp_path),
        keep_original=False,
        perform_diarization=False,
    )

    assert result["warnings_count"] == 1
    assert warning_msg in result["warnings"]
    assert warning_msg not in result["errors"]


@patch("tldw_Server_API.app.core.Ingestion_Media_Processing.Video.Video_DL_Ingestion_Lib.process_single_video")
def test_process_videos_counts_success_and_warning_as_processed(mock_single, tmp_path):
    success_result = {
        "status": "Success",
        "input_ref": "https://example.com/a",
        "processing_source": "https://example.com/a",
        "media_type": "video",
        "metadata": {},
        "content": "ok",
        "segments": [],
        "chunks": [],
        "analysis": None,
        "analysis_details": {},
        "error": None,
        "warnings": [],
        "kept_video_path": None,
    }
    warning_result = {
        "status": "Warning",
        "input_ref": "https://example.com/b",
        "processing_source": "https://example.com/b",
        "media_type": "video",
        "metadata": {},
        "content": "partial",
        "segments": [],
        "chunks": [],
        "analysis": None,
        "analysis_details": {},
        "error": None,
        "warnings": ["partial transcript"],
        "kept_video_path": None,
    }
    error_result = {
        "status": "Error",
        "input_ref": "https://example.com/c",
        "processing_source": "https://example.com/c",
        "media_type": "video",
        "metadata": {},
        "content": None,
        "segments": None,
        "chunks": None,
        "analysis": None,
        "analysis_details": {},
        "error": "failed download",
        "warnings": None,
        "kept_video_path": None,
    }
    mock_single.side_effect = [success_result, warning_result, error_result]

    result = process_videos(
        inputs=[
            "https://example.com/a",
            "https://example.com/b",
            "https://example.com/c",
        ],
        start_time=None,
        end_time=None,
        diarize=False,
        vad_use=False,
        transcription_model="whisper-small",
        transcription_language="en",
        perform_analysis=False,
        custom_prompt=None,
        system_prompt=None,
        perform_chunking=False,
        chunk_method=None,
        max_chunk_size=1000,
        chunk_overlap=0,
        use_adaptive_chunking=False,
        use_multi_level_chunking=False,
        chunk_language=None,
        summarize_recursively=False,
        api_name=None,
        use_cookies=False,
        cookies=None,
        timestamp_option=False,
        perform_confabulation_check=False,
        temp_dir=str(tmp_path),
        keep_original=False,
        perform_diarization=False,
    )

    assert result["processed_count"] == 2
    assert result["errors_count"] == 1


@patch("tldw_Server_API.app.core.Ingestion_Media_Processing.Video.Video_DL_Ingestion_Lib._store_video_file")
@patch("tldw_Server_API.app.core.Ingestion_Media_Processing.Video.Video_DL_Ingestion_Lib.extract_text_from_segments", return_value="hello world")
@patch("tldw_Server_API.app.core.Ingestion_Media_Processing.Video.Video_DL_Ingestion_Lib.perform_transcription")
def test_process_single_video_sets_kept_path_on_success(
    mock_transcribe,
    _mock_extract,
    mock_store,
    tmp_path,
):
    mock_transcribe.return_value = ("audio.wav", [{"Text": "hello world", "Time_Start": 0, "Time_End": 1}])
    kept_path = tmp_path / "stored" / "video.mp4"
    mock_store.return_value = kept_path

    local_video = tmp_path / "local.mp4"
    local_video.write_bytes(b"\x00\x01")

    result = process_single_video(
        video_input=str(local_video),
        start_seconds=0,
        end_seconds=None,
        diarize=False,
        vad_use=False,
        transcription_model="whisper-small",
        transcription_language="en",
        perform_analysis=False,
        custom_prompt=None,
        system_prompt=None,
        perform_chunking=False,
        chunk_method=None,
        max_chunk_size=1000,
        chunk_overlap=0,
        use_adaptive_chunking=False,
        use_multi_level_chunking=False,
        chunk_language=None,
        summarize_recursively=False,
        api_name=None,
        use_cookies=False,
        cookies=None,
        timestamp_option=False,
        temp_dir=str(tmp_path),
        keep_intermediate_audio=False,
        perform_diarization=False,
        keep_original=True,
        user_id=42,
    )

    mock_store.assert_called_once()
    assert result["status"] == "Success"
    assert result["kept_video_path"] == str(kept_path)


@pytest.mark.unit
def test_process_single_video_rejects_local_path_outside_temp_dir(tmp_path):
    allowed_dir = tmp_path / "allowed"
    allowed_dir.mkdir()
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    outside_path = outside_dir / "local.mp4"

    result = process_single_video(
        video_input=str(outside_path),
        start_seconds=0,
        end_seconds=None,
        diarize=False,
        vad_use=False,
        transcription_model="whisper-small",
        transcription_language="en",
        perform_analysis=False,
        custom_prompt=None,
        system_prompt=None,
        perform_chunking=False,
        chunk_method=None,
        max_chunk_size=1000,
        chunk_overlap=0,
        use_adaptive_chunking=False,
        use_multi_level_chunking=False,
        chunk_language=None,
        summarize_recursively=False,
        api_name=None,
        use_cookies=False,
        cookies=None,
        timestamp_option=False,
        temp_dir=str(allowed_dir),
        keep_intermediate_audio=False,
        perform_diarization=False,
        keep_original=False,
        user_id=42,
    )

    assert result["status"] == "Error"
    assert "rejected outside temp directory" in (result.get("error") or "")


@pytest.mark.unit
def test_process_single_video_blocks_disallowed_url(monkeypatch, tmp_path):
    def fake_evaluate_url_policy(*_args, **_kwargs):
        class _Res:
            allowed = False
            reason = "blocked"
        return _Res()

    monkeypatch.setattr(video_lib, "evaluate_url_policy", fake_evaluate_url_policy)

    result = process_single_video(
        video_input="https://example.com/video",
        start_seconds=0,
        end_seconds=None,
        diarize=False,
        vad_use=False,
        transcription_model="whisper-small",
        transcription_language="en",
        perform_analysis=False,
        custom_prompt=None,
        system_prompt=None,
        perform_chunking=False,
        chunk_method=None,
        max_chunk_size=1000,
        chunk_overlap=0,
        use_adaptive_chunking=False,
        use_multi_level_chunking=False,
        chunk_language=None,
        summarize_recursively=False,
        api_name=None,
        use_cookies=False,
        cookies=None,
        timestamp_option=False,
        temp_dir=str(tmp_path),
        keep_intermediate_audio=False,
        perform_diarization=False,
        keep_original=False,
        user_id=42,
    )

    assert result["status"] == "Error"
    assert "blocked" in (result.get("error") or "")


@pytest.mark.unit
def test_download_video_blocks_disallowed_url(monkeypatch, tmp_path):
    def fake_evaluate_url_policy(*_args, **_kwargs):
        class _Res:
            allowed = False
            reason = "blocked"
        return _Res()

    monkeypatch.setattr(video_lib, "evaluate_url_policy", fake_evaluate_url_policy)

    with pytest.raises(ValueError) as exc:
        video_lib.download_video(
            "https://example.com/video",
            str(tmp_path),
            info_dict=None,
            download_video_flag=True,
            use_cookies=False,
            cookies=None,
        )

    assert "blocked" in str(exc.value)


@patch("tldw_Server_API.app.core.Ingestion_Media_Processing.Video.Video_DL_Ingestion_Lib.extract_text_from_segments")
@patch("tldw_Server_API.app.core.Ingestion_Media_Processing.Video.Video_DL_Ingestion_Lib.perform_transcription")
def test_process_single_video_returns_error_for_sentinel_transcript(mock_transcribe, mock_extract, tmp_path):
    mock_transcribe.return_value = ("audio.wav", [{"Text": "text", "Time_Start": 0, "Time_End": 1}])
    mock_extract.return_value = video_lib._TRANSCRIPTION_EXTRACTION_ERROR_SENTINEL

    local_video = tmp_path / "local_sentinel.mp4"
    local_video.write_bytes(b"\x00\x01")

    result = process_single_video(
        video_input=str(local_video),
        start_seconds=0,
        end_seconds=None,
        diarize=False,
        vad_use=False,
        transcription_model="whisper-small",
        transcription_language="en",
        perform_analysis=False,
        custom_prompt=None,
        system_prompt=None,
        perform_chunking=False,
        chunk_method=None,
        max_chunk_size=1000,
        chunk_overlap=0,
        use_adaptive_chunking=False,
        use_multi_level_chunking=False,
        chunk_language=None,
        summarize_recursively=False,
        api_name=None,
        use_cookies=False,
        cookies=None,
        timestamp_option=False,
        temp_dir=str(tmp_path),
        keep_intermediate_audio=False,
        perform_diarization=False,
    )

    assert result["status"] == "Error"
    assert "unable to extract" in (result.get("error") or "").lower()


def test_store_video_file_respects_limits(monkeypatch, tmp_path):


    storage_root = tmp_path / "storage"
    monkeypatch.setattr(video_lib, "_VIDEO_STORAGE_ROOT", storage_root)
    monkeypatch.setattr(video_lib, "_KEEP_VIDEO_MAX_FILES", 2)
    monkeypatch.setattr(video_lib, "_KEEP_VIDEO_MAX_STORAGE_MB", 1)
    monkeypatch.setattr(video_lib, "_KEEP_VIDEO_MAX_BYTES", 1 * 1024 * 1024)
    monkeypatch.setattr(video_lib, "_KEEP_VIDEO_RETENTION_SECONDS", 3600)

    for idx in range(3):
        file_path = tmp_path / f"video_{idx}.mp4"
        file_path.write_bytes(b"x" * 1024)  # 1KB
        stored_path = video_lib._store_video_file(file_path, user_id=123)
        assert stored_path is not None

    stored_dir = storage_root / "123"
    stored_files = sorted(p.name for p in stored_dir.iterdir() if p.is_file())
    assert len(stored_files) == 2
    assert "video_0.mp4" not in stored_files

    big_file = tmp_path / "too_big.mp4"
    big_file.write_bytes(b"x" * (2 * 1024 * 1024))  # 2MB
    assert video_lib._store_video_file(big_file, user_id=123) is None
