from __future__ import annotations

import pytest

from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio import stt_policy


def _principal(*, org_ids: list[int], active_org_id: int | None) -> AuthPrincipal:
    return AuthPrincipal(
        kind="user",
        user_id=123,
        api_key_id=None,
        username="policy-user",
        email=None,
        subject="user-123",
        token_type="access",
        jti=None,
        roles=[],
        permissions=[],
        is_admin=False,
        org_ids=org_ids,
        team_ids=[],
        active_org_id=active_org_id,
        active_team_id=None,
    )


@pytest.mark.asyncio
async def test_resolve_effective_stt_policy_prefers_active_org_id(monkeypatch: pytest.MonkeyPatch) -> None:
    requested_org_ids: list[int] = []

    class _Repo:
        def __init__(self, _db: object) -> None:
            return None

        async def get_settings(self, org_id: int) -> dict[str, object]:
            requested_org_ids.append(org_id)
            return {
                "delete_audio_after_success": False,
                "audio_retention_hours": 4.0,
                "redact_pii": True,
                "allow_unredacted_partials": False,
                "redact_categories": ["pii_email"],
            }

    monkeypatch.setattr(
        stt_policy,
        "_stt_policy_from_config",
        lambda: stt_policy.STTPolicy(
            org_id=None,
            delete_audio_after_success=True,
            audio_retention_hours=0.0,
            redact_pii=False,
            allow_unredacted_partials=False,
            redact_categories=[],
        ),
    )
    monkeypatch.setattr(stt_policy, "AuthnzOrgSttSettingsRepo", _Repo)

    policy = await stt_policy.resolve_effective_stt_policy(
        principal=_principal(org_ids=[10, 20], active_org_id=20),
        user_id=123,
        db=object(),
    )

    assert requested_org_ids == [20]
    assert policy.org_id == 20
    assert policy.redact_pii is True
    assert policy.redact_categories == ["pii_email"]
