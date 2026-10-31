from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from tldw_Server_API.app.api.v1.schemas.integrations_control_plane_schemas import (
    IntegrationConnection,
    IntegrationOverviewResponse,
)
from tldw_Server_API.app.core.AuthNZ.user_provider_secrets import normalize_provider_name
from tldw_Server_API.app.core.Utils.iso_datetime import parse_iso_utc as _coerce_datetime

_PERSONAL_PROVIDER_ORDER = ("slack", "discord")
_WORKSPACE_PROVIDER_ORDER = ("slack", "discord", "telegram")
_PROVIDER_LABELS = {
    "slack": "Slack",
    "discord": "Discord",
    "telegram": "Telegram",
}


def _coerce_nonnegative_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _coerce_metadata(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return {}
        if isinstance(parsed, dict):
            return dict(parsed)
    return {}


def _coerce_display_name(provider: str, fallback: str | None = None) -> str:
    cleaned_fallback = str(fallback or "").strip()
    return cleaned_fallback or _PROVIDER_LABELS[provider]


def _latest_datetime(values: list[Any]) -> datetime | None:
    parsed_values = [candidate for candidate in (_coerce_datetime(value) for value in values) if candidate is not None]
    return max(parsed_values) if parsed_values else None


def _earliest_datetime(values: list[Any]) -> datetime | None:
    parsed_values = [candidate for candidate in (_coerce_datetime(value) for value in values) if candidate is not None]
    return min(parsed_values) if parsed_values else None


@dataclass
class IntegrationsControlPlaneService:
    """Normalizes provider-specific integration state for admin and personal UIs."""

    user_provider_secrets_repo: Any
    org_provider_secrets_repo: Any
    workspace_installations_repo: Any

    async def build_personal_overview(self, *, user_id: int) -> IntegrationOverviewResponse:
        rows = await self.user_provider_secrets_repo.list_secrets_for_user(int(user_id), include_revoked=False)
        rows_by_provider = {
            normalize_provider_name(str(row.get("provider") or "")): row
            for row in rows
            if str(row.get("provider") or "").strip()
        }

        items = [
            self._build_personal_provider_item(provider=provider, row=rows_by_provider.get(provider))
            for provider in _PERSONAL_PROVIDER_ORDER
        ]
        return IntegrationOverviewResponse(scope="personal", items=items)

    async def build_workspace_overview(
        self,
        *,
        org_id: int,
        scope_type: str = "org",
        scope_id: int | None = None,
    ) -> IntegrationOverviewResponse:
        resolved_scope_type = str(scope_type or "org").strip().lower() or "org"
        resolved_scope_id = int(scope_id if scope_id is not None else org_id)

        items = [
            await self._build_workspace_registry_item(org_id=int(org_id), provider="slack"),
            await self._build_workspace_registry_item(org_id=int(org_id), provider="discord"),
            await self._build_workspace_telegram_item(scope_type=resolved_scope_type, scope_id=resolved_scope_id),
        ]
        return IntegrationOverviewResponse(scope="workspace", items=items)

    def _build_personal_provider_item(self, *, provider: str, row: dict[str, Any] | None) -> IntegrationConnection:
        if row is None:
            return IntegrationConnection(
                id=f"personal:{provider}",
                provider=provider,
                scope="personal",
                display_name=_PROVIDER_LABELS[provider],
                status="disconnected",
                enabled=False,
                actions=["connect"],
            )

        metadata = _coerce_metadata(row.get("metadata"))
        installation_count = _coerce_nonnegative_int(metadata.get("installation_count"))
        active_installation_count = _coerce_nonnegative_int(metadata.get("active_installation_count"))
        if installation_count is not None:
            metadata["installation_count"] = installation_count
        if active_installation_count is not None:
            metadata["active_installation_count"] = active_installation_count

        enabled = not (
            installation_count is not None
            and installation_count > 0
            and active_installation_count is not None
            and active_installation_count == 0
        )
        actions = ["reconnect", "disable", "remove"] if enabled else ["reconnect", "enable", "remove"]
        return IntegrationConnection(
            id=f"personal:{provider}",
            provider=provider,
            scope="personal",
            display_name=_PROVIDER_LABELS[provider],
            status="connected" if enabled else "disabled",
            enabled=enabled,
            connected_at=_coerce_datetime(row.get("created_at")),
            updated_at=_coerce_datetime(row.get("updated_at")),
            metadata=metadata,
            actions=actions,
        )

    async def _build_workspace_registry_item(self, *, org_id: int, provider: str) -> IntegrationConnection:
        rows = await self.workspace_installations_repo.list_installations(
            org_id=int(org_id),
            provider=provider,
            include_disabled=True,
        )
        if not rows:
            return IntegrationConnection(
                id=f"workspace:{provider}",
                provider=provider,
                scope="workspace",
                display_name=_PROVIDER_LABELS[provider],
                status="disconnected",
                enabled=False,
                actions=["connect"],
            )

        active_rows = [row for row in rows if not bool(row.get("disabled"))]
        enabled = bool(active_rows)
        display_row = active_rows[0] if active_rows else rows[0]
        health_statuses = [str(row.get("last_health_status") or "").strip() for row in rows if row.get("last_health_status")]
        metadata = {
            "installation_count": len(rows),
            "active_installation_count": len(active_rows),
            "external_ids": [str(row.get("external_id")) for row in rows if row.get("external_id") is not None],
        }
        health = {"status": health_statuses[-1]} if health_statuses else None
        return IntegrationConnection(
            id=f"workspace:{provider}",
            provider=provider,
            scope="workspace",
            display_name=_coerce_display_name(provider, display_row.get("display_name")),
            status="connected" if enabled else "disabled",
            enabled=enabled,
            connected_at=_earliest_datetime([row.get("created_at") for row in rows]),
            updated_at=_latest_datetime([row.get("updated_at") for row in rows]),
            health=health,
            metadata=metadata,
            actions=["manage"],
        )

    async def _build_workspace_telegram_item(self, *, scope_type: str, scope_id: int) -> IntegrationConnection:
        row = await self.org_provider_secrets_repo.fetch_secret(
            scope_type,
            int(scope_id),
            "telegram",
            include_revoked=False,
        )
        if row is None:
            return IntegrationConnection(
                id="workspace:telegram",
                provider="telegram",
                scope="workspace",
                display_name=_PROVIDER_LABELS["telegram"],
                status="needs_config",
                enabled=False,
                actions=["configure_bot", "generate_pairing_code"],
            )

        metadata = _coerce_metadata(row.get("metadata"))
        enabled = bool(metadata.get("enabled", True))
        bot_username = str(metadata.get("bot_username") or "").strip().lstrip("@")
        display_name = f"@{bot_username}" if bot_username else _PROVIDER_LABELS["telegram"]
        return IntegrationConnection(
            id="workspace:telegram",
            provider="telegram",
            scope="workspace",
            display_name=display_name,
            status="connected" if enabled else "disabled",
            enabled=enabled,
            connected_at=_coerce_datetime(row.get("created_at")),
            updated_at=_coerce_datetime(row.get("updated_at")),
            metadata=metadata,
            actions=["configure_bot", "generate_pairing_code", "manage_linked_actors"],
        )
