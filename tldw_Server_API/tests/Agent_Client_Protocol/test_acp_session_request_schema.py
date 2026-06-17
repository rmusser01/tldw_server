from __future__ import annotations

import pytest
from pydantic import ValidationError

from tldw_Server_API.app.api.v1.schemas.agent_client_protocol import (
    ACPMCPServerConfig,
    ACPSessionNewRequest,
)


@pytest.mark.unit
def test_session_new_requires_absolute_cwd() -> None:
    with pytest.raises(ValidationError, match="absolute"):
        ACPSessionNewRequest.model_validate({"cwd": "relative/path"})


@pytest.mark.unit
def test_session_new_normalizes_absolute_cwd_whitespace() -> None:
    request = ACPSessionNewRequest.model_validate({"cwd": " /repo "})

    assert request.cwd == "/repo"


@pytest.mark.unit
def test_stdio_mcp_server_requires_absolute_command() -> None:
    with pytest.raises(ValidationError, match="absolute"):
        ACPMCPServerConfig.model_validate(
            {
                "name": "workspace",
                "type": "stdio",
                "command": "mcp-filesystem",
            }
        )


@pytest.mark.unit
def test_stdio_mcp_server_normalizes_absolute_command_whitespace() -> None:
    server = ACPMCPServerConfig.model_validate(
        {
            "name": "workspace",
            "type": "stdio",
            "command": " /usr/local/bin/mcp-filesystem ",
        }
    )

    assert server.command == "/usr/local/bin/mcp-filesystem"


@pytest.mark.unit
def test_http_mcp_server_requires_url() -> None:
    with pytest.raises(ValidationError, match="url"):
        ACPMCPServerConfig.model_validate(
            {
                "name": "remote",
                "type": "http",
            }
        )


@pytest.mark.unit
def test_http_mcp_server_normalizes_url_whitespace() -> None:
    server = ACPMCPServerConfig.model_validate(
        {
            "name": "remote",
            "type": "http",
            "url": " https://mcp.example.com ",
        }
    )

    assert server.url == "https://mcp.example.com"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("transport", "url"),
    [
        ("http", "ftp://mcp.example.com"),
        ("sse", "ws://mcp.example.com/events"),
        ("websocket", "https://mcp.example.com/ws"),
    ],
)
def test_remote_mcp_server_requires_transport_specific_url_scheme(
    transport: str,
    url: str,
) -> None:
    with pytest.raises(ValidationError, match="url"):
        ACPMCPServerConfig.model_validate(
            {
                "name": "remote",
                "type": transport,
                "url": url,
            }
        )


@pytest.mark.unit
def test_http_and_sse_mcp_server_configs_are_valid() -> None:
    request = ACPSessionNewRequest.model_validate(
        {
            "cwd": "/repo",
            "mcp_servers": [
                {
                    "name": "remote-http",
                    "type": "http",
                    "url": "https://mcp.example.com",
                    "headers": [{"name": "Authorization", "value": "Bearer token"}],
                },
                {
                    "name": "remote-sse",
                    "type": "sse",
                    "url": "https://mcp.example.com/sse",
                },
            ],
        }
    )

    dumped = request.model_dump(exclude_none=True)
    assert dumped["mcp_servers"] == [
        {
            "name": "remote-http",
            "type": "http",
            "url": "https://mcp.example.com",
            "headers": [{"name": "Authorization", "value": "Bearer token"}],
        },
        {
            "name": "remote-sse",
            "type": "sse",
            "url": "https://mcp.example.com/sse",
        },
    ]


@pytest.mark.unit
def test_mcp_server_env_dict_is_normalized_to_name_value_pairs() -> None:
    server = ACPMCPServerConfig.model_validate(
        {
            "name": "workspace",
            "type": "stdio",
            "command": "/usr/local/bin/mcp-filesystem",
            "env": {
                "WORKSPACE_TOKEN": "token-value",
                "TRACE": "1",
            },
        }
    )

    assert server.model_dump(exclude_none=True)["env"] == [
        {"name": "WORKSPACE_TOKEN", "value": "token-value"},
        {"name": "TRACE", "value": "1"},
    ]


@pytest.mark.unit
@pytest.mark.parametrize("field_name", ["env", "headers"])
def test_mcp_server_name_value_dict_rejects_null_values(field_name: str) -> None:
    with pytest.raises(ValidationError):
        ACPMCPServerConfig.model_validate(
            {
                "name": "workspace",
                "type": "stdio",
                "command": "/usr/local/bin/mcp-filesystem",
                field_name: {"TOKEN": None},
            }
        )
