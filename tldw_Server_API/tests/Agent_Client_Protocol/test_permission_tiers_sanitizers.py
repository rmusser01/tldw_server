from __future__ import annotations

import pytest


_LEAK = "permission policy backend exploded at /tmp/acp-policy-secret-token"


@pytest.mark.unit
def test_policy_permission_tier_failure_log_is_sanitized(monkeypatch: pytest.MonkeyPatch) -> None:
    from tldw_Server_API.app.core.Agent_Client_Protocol import permission_tiers
    import tldw_Server_API.app.services.admin_acp_sessions_service as store_src

    class _PolicyStore:
        def resolve_permission_tier(self, tool_name: str) -> str | None:
            raise RuntimeError(_LEAK)

    records: list[str] = []
    sink_id = permission_tiers.logger.add(lambda message: records.append(str(message)), format="{message} {extra}")
    monkeypatch.setattr(store_src, "_store", _PolicyStore(), raising=False)

    try:
        result = permission_tiers.resolve_policy_permission_tier("fs.read")
    finally:
        permission_tiers.logger.remove(sink_id)

    assert result is None
    rendered = "\n".join(records)
    assert "Failed to resolve ACP permission tier from admin policy store" in rendered
    assert "RuntimeError" in rendered
    assert "permission policy backend exploded" not in rendered
    assert "/tmp/acp-policy-secret-token" not in rendered
    assert "exc_info" not in rendered


@pytest.mark.unit
@pytest.mark.parametrize(
    "tool_name",
    [
        "thread.delete",
        "file_read_delete",
        "get_shell_history",
        "list_terminal_sessions",
        "Bash(git:push --force)",
        "read_and_write_file",
        "get_and_update_profile",
        "list_and_destroy_instances",
        "find_and_patch_record",
    ],
)
def test_permission_tier_checks_destructive_tokens_before_auto_tokens(
    tool_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.Agent_Client_Protocol.permission_tiers import determine_permission_tier
    import tldw_Server_API.app.services.admin_acp_sessions_service as store_src

    monkeypatch.setattr(store_src, "_store", None, raising=False)

    assert determine_permission_tier(tool_name) == "individual"
