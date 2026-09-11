"""Regression tests for proposition analyzer credential snapshots."""

import threading
from concurrent.futures import ThreadPoolExecutor

from loguru import logger

from tldw_Server_API.app.core.Chunking.strategies.propositions import (
    PropositionChunkingStrategy,
)


def _assert_snapshot_call(call, expected_config, expected_key, expected_handle=None):
    args, kwargs = call
    assert args[3] == expected_key
    assert kwargs["app_config"] == expected_config
    assert kwargs["credentials_resolved"] is True
    if expected_handle is not None:
        assert kwargs["provider_credentials"] is expected_handle


def test_proposition_analyzer_uses_latest_authoritative_snapshot_after_a_to_b_rotation():
    calls = []
    config_a = {"openai_api": {"api_base_url": "https://a.example/v1"}}
    config_b = {"openai_api": {"api_base_url": "https://b.example/v1"}}

    def analyzer(*args, **kwargs):
        calls.append((args, kwargs))
        return "snapshot-b"

    strategy = PropositionChunkingStrategy(
        llm_call_func=analyzer,
        llm_config={
            "api_name": "openai",
            "api_key": "key-a",
            "app_config": config_a,
            "credentials_resolved": True,
        },
    )
    strategy.llm_config = {
        "api_name": "openai",
        "api_key": "key-b",
        "app_config": config_b,
        "credentials_resolved": True,
    }

    assert strategy._call_llm("rotation prompt") == "snapshot-b"
    assert len(calls) == 1
    _assert_snapshot_call(calls[0], config_b, "key-b")


def test_proposition_analyzer_uses_authoritative_snapshot_after_absent_to_b_rotation():
    calls = []
    config_b = {"openai_api": {"api_base_url": "https://b.example/v1"}}

    def analyzer(*args, **kwargs):
        calls.append((args, kwargs))
        return "snapshot-b"

    strategy = PropositionChunkingStrategy(llm_call_func=analyzer, llm_config={})
    strategy.llm_config = {
        "api_name": "openai",
        "api_key": "key-b",
        "app_config": config_b,
        "credentials_resolved": True,
    }

    assert strategy._call_llm("rotation prompt") == "snapshot-b"
    assert len(calls) == 1
    _assert_snapshot_call(calls[0], config_b, "key-b")


def test_concurrent_proposition_analyzers_keep_snapshots_isolated():
    barrier = threading.Barrier(2)
    calls = {}
    calls_lock = threading.Lock()

    def analyzer(*args, **kwargs):
        label = args[1].rsplit("-", 1)[-1]
        barrier.wait(timeout=5)
        with calls_lock:
            calls[label] = (args, kwargs)
        return f"snapshot-{label}"

    handles = {label: object() for label in ("a", "b")}
    strategies = {
        label: PropositionChunkingStrategy(
            llm_call_func=analyzer,
            llm_config={
                "api_name": "openai",
                "api_key": f"key-{label}",
                "app_config": {
                    "openai_api": {
                        "api_base_url": f"https://{label}.example/v1",
                    }
                },
                "credentials_resolved": True,
                "provider_credentials": handles[label],
            },
        )
        for label in ("a", "b")
    }

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            label: executor.submit(strategy._call_llm, f"concurrent-{label}")
            for label, strategy in strategies.items()
        }
        assert {label: future.result(timeout=5) for label, future in futures.items()} == {
            "a": "snapshot-a",
            "b": "snapshot-b",
        }

    assert set(calls) == {"a", "b"}
    for label, call in calls.items():
        _assert_snapshot_call(
            call,
            {"openai_api": {"api_base_url": f"https://{label}.example/v1"}},
            f"key-{label}",
            handles[label],
        )


def test_proposition_analyzer_forwards_runtime_certified_bedrock_default_chain():
    calls = []
    app_config = {
        "bedrock_api": {
            "model": "bedrock-model",
            "region": "us-west-2",
            "_runtime_auth_source": "aws_default_chain",
        }
    }

    def analyzer(*args, **kwargs):
        calls.append((args, kwargs))
        return "bedrock snapshot"

    strategy = PropositionChunkingStrategy(
        llm_call_func=analyzer,
        llm_config={
            "api_name": "bedrock",
            "api_key": None,
            "app_config": app_config,
            "credentials_resolved": True,
        },
    )

    assert strategy._call_llm("bedrock prompt") == "bedrock snapshot"
    assert len(calls) == 1
    _assert_snapshot_call(calls[0], app_config, None)


def test_proposition_analyzer_failure_logs_type_without_exception_message():
    secret = "provider-secret-must-not-be-logged"
    messages = []

    def analyzer(*_args, **_kwargs):
        raise RuntimeError(secret)

    strategy = PropositionChunkingStrategy(llm_call_func=analyzer)
    sink_id = logger.add(messages.append, level="ERROR", format="{message}")
    try:
        assert strategy._call_llm("failure prompt") is None
    finally:
        logger.remove(sink_id)

    rendered = "".join(messages)
    assert "RuntimeError" in rendered
    assert secret not in rendered
