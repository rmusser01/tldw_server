from __future__ import annotations

from typing import Any

import pytest
from loguru import logger

from tldw_Server_API.app.core.AuthNZ.repos.byok_oauth_state_repo import (
    AuthnzByokOAuthStateRepo,
)
from tldw_Server_API.app.core.AuthNZ.repos.federated_managed_grant_repo import (
    FederatedManagedGrantRepo,
)
from tldw_Server_API.app.core.AuthNZ.repos.identity_provider_repo import IdentityProviderRepo
from tldw_Server_API.app.core.AuthNZ.repos.managed_secret_refs_repo import (
    ManagedSecretRefsRepo,
)
from tldw_Server_API.app.core.AuthNZ.repos.org_provider_secrets_repo import (
    AuthnzOrgProviderSecretsRepo,
)

pytestmark = pytest.mark.unit

_LEAK = "authnz row backend exploded at /tmp/authnz-row-secret-token"


def _capture_logs() -> tuple[list[str], int]:
    records: list[str] = []
    sink_id = logger.add(lambda message: records.append(str(message)), format="{message} {extra}")
    return records, sink_id


def _assert_safe_log(rendered: str) -> None:
    assert "authnz row backend exploded" not in rendered
    assert "/tmp/authnz-row-secret-token" not in rendered
    assert "exc_info" not in rendered


class _FlakyMappingRow:
    def __init__(self, data: dict[str, Any]) -> None:
        self._data = dict(data)
        self._keys_calls = 0

    def keys(self):
        self._keys_calls += 1
        if self._keys_calls == 1:
            raise RuntimeError(_LEAK)
        return list(self._data.keys())

    def __getitem__(self, key: str) -> Any:
        return self._data[key]


@pytest.mark.parametrize(
    ("coerce_row", "payload"),
    [
        (IdentityProviderRepo._row_to_dict, {"provider": "oidc", "slug": "corp"}),
        (AuthnzByokOAuthStateRepo._row_to_dict, {"state_hash": "hash-1", "provider": "github"}),
        (FederatedManagedGrantRepo._row_to_dict, {"provider": "github", "provider_user_id": "u-1"}),
        (AuthnzOrgProviderSecretsRepo._row_to_dict, {"provider": "openai", "key_hint": "1234"}),
    ],
)
def test_authnz_repo_mapping_fallback_logs_omit_raw_exception(coerce_row, payload: dict[str, Any]) -> None:
    records, sink_id = _capture_logs()
    try:
        row = coerce_row(_FlakyMappingRow(payload))
    finally:
        logger.remove(sink_id)

    assert row == payload
    _assert_safe_log("\n".join(records))


def test_managed_secret_ref_mapping_fallback_log_omits_raw_exception() -> None:
    payload = {
        "secret_ref": "ref-1",
        "capabilities_json": '{"read": true}',
        "metadata_json": '{"owner": "admin"}',
    }

    records, sink_id = _capture_logs()
    try:
        row = ManagedSecretRefsRepo._row_to_dict(_FlakyMappingRow(payload))
    finally:
        logger.remove(sink_id)

    assert row["secret_ref"] == "ref-1"
    assert row["capabilities"] == {"read": True}
    assert row["metadata"] == {"owner": "admin"}
    _assert_safe_log("\n".join(records))
