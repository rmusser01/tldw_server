from __future__ import annotations

import pytest


def test_run_command_module_accepts_snake_case_idempotency_key() -> None:
    from tldw_Server_API.app.core.MCP_unified.modules.base import ModuleConfig
    from tldw_Server_API.app.core.MCP_unified.modules.implementations.run_command_module import (
        RunCommandModule,
    )

    module = RunCommandModule(ModuleConfig(name="run"))

    module.validate_tool_arguments(
        "run",
        {
            "command": "help",
            "idempotency_key": "parent-123",
        },
    )


def test_run_command_module_parent_idempotency_key_prefers_snake_case_argument() -> None:
    from tldw_Server_API.app.core.MCP_unified.modules.implementations.run_command_module import (
        RunCommandModule,
    )

    value = RunCommandModule._parent_idempotency_key(
        None,
        {
            "command": "help",
            "idempotency_key": "parent-123",
        },
    )

    assert value == "parent-123"


def test_run_command_module_rejects_non_string_snake_case_idempotency_key() -> None:
    from tldw_Server_API.app.core.MCP_unified.modules.base import ModuleConfig
    from tldw_Server_API.app.core.MCP_unified.modules.implementations.run_command_module import (
        RunCommandModule,
    )

    module = RunCommandModule(ModuleConfig(name="run"))

    with pytest.raises(ValueError, match="idempotency_key must be a string"):
        module.validate_tool_arguments(
            "run",
            {
                "command": "help",
                "idempotency_key": 7,
            },
        )
