"""Versioned event catalog for canonical outgoing webhooks."""

from __future__ import annotations

from dataclasses import dataclass

from .domain import WebhookError, WebhookErrorCode

EVENT_API_VERSION = "2026-07-01"


@dataclass(frozen=True)
class WebhookCatalogItem:
    """One explicitly supported webhook event type."""

    event_type: str
    description: str


EVENT_CATALOG = (
    WebhookCatalogItem("user.created", "A user account was created."),
    WebhookCatalogItem("user.deleted", "A user account was deleted."),
    WebhookCatalogItem("incident.created", "An incident was created."),
    WebhookCatalogItem("incident.updated", "An incident was updated."),
    WebhookCatalogItem("incident.resolved", "An incident was resolved."),
    WebhookCatalogItem("incident.notify", "An incident notification was requested."),
)

_CATALOG_ORDER = {
    item.event_type: position for position, item in enumerate(EVENT_CATALOG)
}


def normalize_subscriptions(event_types: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    """Validate and order subscriptions using the immutable server catalog."""
    if not event_types:
        raise WebhookError(WebhookErrorCode.EVENT_UNSUPPORTED)
    if len(set(event_types)) != len(event_types):
        raise WebhookError(WebhookErrorCode.EVENT_UNSUPPORTED)
    if any(event_type not in _CATALOG_ORDER for event_type in event_types):
        raise WebhookError(WebhookErrorCode.EVENT_UNSUPPORTED)
    return tuple(sorted(event_types, key=_CATALOG_ORDER.__getitem__))


def validate_subscriptions(
    event_types: list[str] | tuple[str, ...],
) -> tuple[str, ...]:
    """Return the canonical event order or raise a bounded domain error."""
    return normalize_subscriptions(event_types)
