"""The helpers LLM_Calls shared with Chat moved down to LLM_Calls and core/Utils (TASK-13950.13).

The old Chat paths re-export the same objects, so existing importers keep working.
"""

import pytest

pytestmark = pytest.mark.unit


def test_chat_paths_re_export_the_moved_objects() -> None:
    from tldw_Server_API.app.core.Chat import bounded_daemon as chat_bounded_daemon
    from tldw_Server_API.app.core.Chat import chat_helpers, streaming_utils
    from tldw_Server_API.app.core.Chat.prompt_cost_envelope import estimate_segment_tokens as chat_estimate
    from tldw_Server_API.app.core.LLM_Calls import bounded_daemon, provider_results
    from tldw_Server_API.app.core.Utils.tokenizer import estimate_segment_tokens

    assert chat_bounded_daemon is bounded_daemon
    assert chat_bounded_daemon.STREAM_DAEMON_POOL is bounded_daemon.STREAM_DAEMON_POOL
    assert chat_bounded_daemon.__all__ == bounded_daemon.__all__
    assert streaming_utils.normalize_provider_stream_error is provider_results.normalize_provider_stream_error
    assert streaming_utils.sanitized_provider_stream_exception is provider_results.sanitized_provider_stream_exception
    assert streaming_utils._extract_text_from_upstream_sse is provider_results._extract_text_from_upstream_sse
    assert chat_helpers.extract_response_content is provider_results.extract_response_content
    assert chat_estimate is estimate_segment_tokens



def test_a_patch_through_the_old_bounded_daemon_path_reaches_the_moved_module(monkeypatch) -> None:
    from tldw_Server_API.app.core.Chat import bounded_daemon as chat_bounded_daemon
    from tldw_Server_API.app.core.LLM_Calls import bounded_daemon

    pool = object()
    monkeypatch.setattr(chat_bounded_daemon, "STREAM_DAEMON_POOL", pool)

    assert bounded_daemon.STREAM_DAEMON_POOL is pool


def test_estimate_segment_tokens_redacts_data_uris_and_rounds_up() -> None:
    from tldw_Server_API.app.core.Utils.tokenizer import estimate_segment_tokens

    assert estimate_segment_tokens("") == 0
    assert estimate_segment_tokens("abcde") == 2
    assert estimate_segment_tokens("data:image/png;base64," + "A" * 4000) == 8
