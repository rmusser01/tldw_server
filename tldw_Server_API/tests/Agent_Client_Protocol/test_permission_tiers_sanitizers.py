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
