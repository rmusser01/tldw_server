from __future__ import annotations

import socket

import pytest

pytestmark = pytest.mark.unit


def test_redact_agent_output_masks_secret_shapes_and_truncates() -> None:
    from tldw_Server_API.app.core.Agent_Client_Protocol.hardening import redact_agent_output

    private_key_header = "-----BEGIN " + "OPENSSH PRIVATE " + "KEY-----"
    private_key_footer = "-----END " + "OPENSSH PRIVATE " + "KEY-----"
    raw = (
        "Authorization: Bearer sk-live-secret-token\n"
        "api_key=sk-another-secret-token\n"
        f"{private_key_header}\n"
        "private material\n"
        f"{private_key_footer}\n"
        + ("x" * 500)
    )

    redacted = redact_agent_output(raw, max_chars=120)

    assert "sk-live-secret-token" not in redacted
    assert "sk-another-secret-token" not in redacted
    assert "private material" not in redacted
    assert "[REDACTED]" in redacted
    assert "[truncated]" in redacted
    assert len(redacted) <= 140


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:8080/mcp",
        "http://localhost:8080/mcp",
        "http://169.254.169.254/latest/meta-data",
        "file:///tmp/mcp.sock",
    ],
)
def test_mcp_http_url_validation_rejects_local_and_non_http_urls(url: str) -> None:
    from tldw_Server_API.app.core.Agent_Client_Protocol.hardening import validate_mcp_http_url

    with pytest.raises(ValueError):
        validate_mcp_http_url(url)


def test_sse_post_url_validation_requires_same_origin() -> None:
    from tldw_Server_API.app.core.Agent_Client_Protocol.hardening import validate_sse_post_url

    with pytest.raises(ValueError, match="same origin"):
        validate_sse_post_url(
            "https://mcp.example.com/sse",
            "https://evil.example.com/messages",
            allow_private_network=True,
        )


def test_mcp_http_url_validation_rejects_hostname_resolving_to_loopback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.Agent_Client_Protocol import hardening

    def fake_getaddrinfo(*args: object, **kwargs: object) -> list[tuple[object, ...]]:
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 443))]

    hardening._DNS_RESOLUTION_CACHE.clear()
    monkeypatch.setattr(hardening.socket, "getaddrinfo", fake_getaddrinfo)

    with pytest.raises(ValueError, match="unsafe host"):
        hardening.validate_mcp_http_url("https://safe-looking.example/mcp")


def test_session_launch_validation_rejects_missing_cwd_roots() -> None:
    from tldw_Server_API.app.core.Agent_Client_Protocol.hardening import (
        validate_acp_session_launch_inputs,
    )

    with pytest.raises(ValueError, match="cwd roots"):
        validate_acp_session_launch_inputs(
            cwd="/repo",
            allowed_cwd_roots=[],
            runner_cwd=None,
        )
