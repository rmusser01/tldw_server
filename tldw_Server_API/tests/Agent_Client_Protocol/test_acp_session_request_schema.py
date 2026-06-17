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
def test_http_mcp_server_requires_url() -> None:
    with pytest.raises(ValidationError, match="url"):
        ACPMCPServerConfig.model_validate(
            {
                "name": "remote",
                "type": "http",
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
