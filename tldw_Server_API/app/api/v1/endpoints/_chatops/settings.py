"""Environment-driven settings and installation-record helpers shared by the ChatOps
support modules (ADR-050 stage 2c).

Each provider builds one ``ChatOpsSettings`` and binds its methods to the private
names the endpoints already import (``_oauth_client_id``, ``_replay_window_seconds``,
...), so callers are unchanged. What stays per provider: the env prefix, the OAuth
URL defaults, which tenant fields an installation record exposes, and payload
encryption (tests patch its helpers on each support module).
"""

from __future__ import annotations

import os
from typing import Any

from fastapi import HTTPException, status


def env_int(name: str, default: int) -> int:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def coerce_nonempty_string(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned if cleaned else None


class ChatOpsSettings:
    def __init__(
        self,
        *,
        provider: str,
        env_prefix: str,
        default_oauth_auth_url: str,
        default_oauth_token_url: str,
        installation_fields: tuple[str, ...],
    ) -> None:
        self.provider = provider
        self.env_prefix = env_prefix
        self.default_oauth_auth_url = default_oauth_auth_url
        self.default_oauth_token_url = default_oauth_token_url
        self.installation_fields = installation_fields

    def _env(self, suffix: str) -> str:
        return f"{self.env_prefix}_{suffix}"

    def env_int(self, suffix: str, default: int) -> int:
        return env_int(self._env(suffix), default)

    def env_str(self, suffix: str) -> str:
        return (os.getenv(self._env(suffix)) or "").strip()

    # --- ingress ---------------------------------------------------------------

    def replay_window_seconds(self) -> int:
        return self.env_int("REPLAY_WINDOW_SECONDS", 300)

    def dedupe_ttl_seconds(self) -> int:
        return self.env_int("DEDUPE_TTL_SECONDS", 3600)

    def ingress_rate_limit_per_minute(self) -> int:
        return self.env_int("INGRESS_RATE_LIMIT_PER_MINUTE", 120)

    def policy_user_quota_per_minute(self) -> int:
        return self.env_int("POLICY_USER_QUOTA_PER_MINUTE", 60)

    # --- OAuth -----------------------------------------------------------------

    def oauth_client_id(self) -> str:
        return self.env_str("CLIENT_ID")

    def oauth_client_secret(self) -> str:
        return self.env_str("CLIENT_SECRET")

    def oauth_redirect_uri(self) -> str:
        redirect_uri = coerce_nonempty_string(os.getenv(self._env("OAUTH_REDIRECT_URI")))
        if not redirect_uri:
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail=f"{self._env('OAUTH_REDIRECT_URI')} is not configured",
            )
        return redirect_uri

    def oauth_auth_url(self) -> str:
        return coerce_nonempty_string(os.getenv(self._env("OAUTH_AUTH_URL"))) or self.default_oauth_auth_url

    def oauth_token_url(self) -> str:
        return coerce_nonempty_string(os.getenv(self._env("OAUTH_TOKEN_URL"))) or self.default_oauth_token_url

    def oauth_state_ttl_seconds(self) -> int:
        return self.env_int("OAUTH_STATE_TTL_SECONDS", 600)

    # --- installation records ----------------------------------------------------

    def default_installations_payload(self) -> dict[str, Any]:
        return {"provider": self.provider, "credential_version": 1, "installations": {}}

    def normalize_installations_payload(self, payload: dict[str, Any] | None) -> dict[str, Any]:
        merged = self.default_installations_payload()
        if isinstance(payload, dict):
            merged.update(payload)
        if not isinstance(merged.get("installations"), dict):
            merged["installations"] = {}
        return merged

    def public_installation_record(self, installation: dict[str, Any]) -> dict[str, Any]:
        """What an installation exposes to API callers: tenant fields, then the common ones."""
        record = {field: installation.get(field) for field in self.installation_fields}
        record.update(
            {
                "scope": installation.get("scope"),
                "installed_at": installation.get("installed_at"),
                "installed_by": installation.get("installed_by"),
                "disabled": bool(installation.get("disabled")),
            }
        )
        return record
