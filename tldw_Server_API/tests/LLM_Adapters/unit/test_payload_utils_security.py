"""Security regressions for provider payload and header helpers."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from tldw_Server_API.app.core.LLM_Calls.payload_utils import (
    encode_provider_model_path,
    merge_extra_headers,
    resolve_runtime_embedding_base_url,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("model", "encoded"),
    [
        ("org/model", "org/model"),
        ("org/model:provider", "org/model:provider"),
        ("org/mødel", "org/m%C3%B8del"),
    ],
)
def test_provider_model_path_encodes_valid_segments(model: str, encoded: str) -> None:
    assert encode_provider_model_path(model) == encoded


@pytest.mark.parametrize(
    "model",
    [
        "",
        "   ",
        "/org/model",
        "org/model/",
        "org//model",
        ".",
        "..",
        "org/../model",
        "org\\model",
        "org/%2e%2e/model",
        "org/model?alt=admin",
        "org/model#fragment",
        "org/model\nforwarded",
    ],
)
def test_provider_model_path_rejects_route_control_syntax(model: str) -> None:
    with pytest.raises(ValueError, match="model identifier"):
        encode_provider_model_path(model)


def test_extra_headers_drop_server_managed_names_and_duplicate_case_variants() -> None:
    headers = merge_extra_headers(
        {"Content-Type": "application/json", "Authorization": "Bearer trusted"},
        {
            "extra_headers": {
                "Host": "attacker.example",
                "Proxy-Authorization": "Bearer attacker",
                "Content-Length": "999",
                "Content-Type": "text/plain",
                "Transfer-Encoding": "chunked",
                "Forwarded": "host=attacker.example",
                "X-Original-URL": "/admin",
                "Cookie": "session=attacker",
                "X-API-Key": "attacker-key",
                "X-Amz-Date": "20990101T000000Z",
                "X-Amz-Content-Sha256": "attacker-signature-body",
                "ApiKey": "attacker-camel-key",
                "X-ApiKey-Value": "attacker-key-value",
                "providerApiKey": "attacker-provider-key",
                "baseUrl": "https://attacker.example",
                "X_API_KEY": "attacker-underscore-key",
                "X_GOOG_API_KEY": "attacker-google-key",
                "X_Provider_Extension": "proxy-normalized-unsafe",
                "X-Provider-Extension": "first",
                "x-provider-extension": "second",
                "X-Amzn-Bedrock-GuardrailIdentifier": "guardrail-123",
            }
        },
    )

    assert headers == {
        "Content-Type": "application/json",
        "Authorization": "Bearer trusted",
        "X-Provider-Extension": "first",
        "X-Amzn-Bedrock-GuardrailIdentifier": "guardrail-123",
    }


def test_generated_google_api_key_header_survives_underscore_alias_filtering() -> None:
    """The adapter's official header stays while public proxy aliases are denied."""
    headers = merge_extra_headers(
        {"Content-Type": "application/json", "x-goog-api-key": "trusted-google-key"},
        {
            "extra_headers": {
                "X_GOOG_API_KEY": "attacker-google-key",
                "X-Provider-Extension": "kept",
            }
        },
    )

    assert headers == {
        "Content-Type": "application/json",
        "x-goog-api-key": "trusted-google-key",
        "X-Provider-Extension": "kept",
    }


@pytest.mark.parametrize(
    "unsafe_value",
    [
        "allowed\r\nX-Injected: attacker",
        "allowed\nX-Injected: attacker",
        "allowed\rX-Injected: attacker",
        "allowed\x00attacker",
        "allowed\x1fattacker",
        "allowed\x7fattacker",
    ],
)
def test_extra_header_values_reject_controls_before_adapter_dispatch(
    unsafe_value: str,
) -> None:
    with pytest.raises(ValueError, match="header value") as exc_info:
        merge_extra_headers(
            {"Authorization": "Bearer trusted"},
            {"extra_headers": {"X-Provider-Extension": unsafe_value}},
        )

    assert "attacker" not in str(exc_info.value)


@pytest.mark.concurrent
def test_concurrent_extra_header_merges_remain_request_local() -> None:
    """Unsafe headers are dropped without sharing accepted extensions across calls."""

    def _merge(label: str) -> dict[str, str]:
        return merge_extra_headers(
            {"Authorization": f"Bearer trusted-{label}"},
            {
                "extra_headers": {
                    "Host": f"attacker-{label}.example",
                    "Content-Type": f"attacker/{label}",
                    "X-Amz-Date": f"attacker-date-{label}",
                    "X-Amz-Content-Sha256": f"attacker-hash-{label}",
                    "ApiKey": f"attacker-camel-key-{label}",
                    "X-ApiKey-Value": f"attacker-key-value-{label}",
                    "providerApiKey": f"attacker-provider-key-{label}",
                    "baseUrl": f"https://attacker-{label}.example",
                    "X_API_KEY": f"attacker-key-{label}",
                    "X_GOOG_API_KEY": f"attacker-google-{label}",
                    "X-Provider-Extension": label,
                }
            },
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        alpha, beta = executor.map(_merge, ("alpha", "beta"))

    assert alpha == {
        "Authorization": "Bearer trusted-alpha",
        "X-Provider-Extension": "alpha",
    }
    assert beta == {
        "Authorization": "Bearer trusted-beta",
        "X-Provider-Extension": "beta",
    }


def test_runtime_embedding_base_url_requires_server_provenance() -> None:
    from tldw_Server_API.app.core.AuthNZ.byok_config import (
        runtime_base_url_override_provenance,
    )

    endpoint = "https://tenant.example/v1/"
    assert resolve_runtime_embedding_base_url(
        {
            "base_url": endpoint,
            "credentials_resolved": True,
            "_runtime_base_url_override": runtime_base_url_override_provenance(),
        },
        provider="openai-embeddings",
    ) == endpoint.rstrip("/")


@pytest.mark.parametrize(
    "payload",
    [
        {"base_url": "https://attacker.example", "credentials_resolved": True},
        {"base_url": "https://attacker.example", "credentials_resolved": False},
        {"credentials_resolved": True},
        {
            "base_url": "https://user:secret@example.test/v1",
            "credentials_resolved": True,
        },
        {
            "base_url": "https://example.test/v1?key=secret",
            "credentials_resolved": True,
        },
    ],
)
def test_runtime_embedding_base_url_fails_closed_for_incomplete_or_unsafe_contracts(
    payload: dict[str, object],
) -> None:
    from tldw_Server_API.app.core.Chat.Chat_Deps import ChatConfigurationError

    with pytest.raises(ChatConfigurationError, match="endpoint configuration") as exc_info:
        resolve_runtime_embedding_base_url(payload, provider="openai-embeddings")

    assert "attacker" not in str(exc_info.value)
    assert "secret" not in str(exc_info.value)


@pytest.mark.parametrize(
    "unsafe_base_url",
    [
        "https://user:secret@example.test/v1",
        "https://example.test/v1?key=secret",
        "https://example.test/v1#fragment",
        "https://example.test/v1\\admin",
        " https://example.test/v1",
        "ftp://example.test/v1",
    ],
)
def test_runtime_embedding_base_url_rejects_unsafe_values_with_valid_provenance(
    unsafe_base_url: str,
) -> None:
    from tldw_Server_API.app.core.AuthNZ.byok_config import (
        runtime_base_url_override_provenance,
    )
    from tldw_Server_API.app.core.Chat.Chat_Deps import ChatConfigurationError

    with pytest.raises(ChatConfigurationError, match="endpoint configuration") as exc_info:
        resolve_runtime_embedding_base_url(
            {
                "base_url": unsafe_base_url,
                "credentials_resolved": True,
                "_runtime_base_url_override": runtime_base_url_override_provenance(),
            },
            provider="openai-embeddings",
        )

    assert "secret" not in str(exc_info.value)
    assert "example.test" not in str(exc_info.value)
