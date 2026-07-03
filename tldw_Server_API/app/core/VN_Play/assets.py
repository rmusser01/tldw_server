"""Approved-manifest asset resolution for VN Play visual directives."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from tldw_Server_API.app.core.VN_Play.models import VisualDirectiveResolution


class VisualIdentityDirectiveResolver(Protocol):
    """Minimal Visual Identity resolver contract used by VN Play casting."""

    def resolve_expression_asset(
        self,
        *,
        actor_kind: str,
        actor_id: int | str,
        requested_expression_key: str,
        manual_override_expression_key: str | None = None,
        mood_expression_key: str | None = None,
        role_id: str | None = None,
        role_label: str | None = None,
        override_pack_id: Any | None = None,
        override_pack_version_id: Any | None = None,
        allow_override_fallback: bool = False,
    ) -> Any:
        """Resolve a Visual Identity expression asset without persisting bindings."""


_ASSET_TYPE_COLLECTION_ALIASES: dict[str, tuple[str, ...]] = {
    "background": ("backgrounds", "background"),
    "backgrounds": ("backgrounds", "background"),
    "sprite": ("sprites", "sprite"),
    "sprites": ("sprites", "sprite"),
    "depth": ("depth_companions", "depth_companion", "depth"),
    "depth_companion": ("depth_companions", "depth_companion", "depth"),
    "depth_companions": ("depth_companions", "depth_companion", "depth"),
    "cg": ("cgs", "cg"),
    "cgs": ("cgs", "cg"),
}

_VISUAL_IDENTITY_ACTOR_KINDS = {"character", "persona"}


def resolve_visual_directive(
    manifest: Mapping[str, Any],
    directive: Mapping[str, Any],
    *,
    seed: str,
) -> VisualDirectiveResolution:
    """Resolve one visual directive against an approved VN asset manifest."""
    directive_dict = dict(directive)
    candidates = [
        item
        for item in _iter_manifest_items(manifest, directive_dict.get("asset_type"))
        if _is_approved_item(item)
        and _matches_slot_key(item, directive_dict)
        and _matches_labels(item, directive_dict)
    ]
    if not candidates:
        return VisualDirectiveResolution(
            applied=False,
            reason="asset_not_found",
            directive=directive_dict,
        )

    selected = sorted(candidates, key=lambda item: _candidate_sort_key(item, seed))[0]
    return VisualDirectiveResolution(
        applied=True,
        item=dict(selected),
        directive=directive_dict,
    )


def resolve_scene_directives(
    manifest: Mapping[str, Any],
    directives: Sequence[Mapping[str, Any]],
    *,
    seed: str,
) -> list[VisualDirectiveResolution]:
    """Resolve a set of visual directives with deterministic per-directive ordering."""
    return [
        resolve_visual_directive(manifest, directive, seed=f"{seed}:{index}")
        for index, directive in enumerate(directives)
    ]


def is_visual_identity_directive(directive: Mapping[str, Any]) -> bool:
    """Return whether a runtime sprite directive targets Visual Identity casting."""
    payload = _visual_identity_payload(directive)
    asset_type = _optional_string(payload.get("asset_type") or directive.get("asset_type"))
    if asset_type and asset_type.strip().lower() not in {"sprite", "sprites"}:
        return False
    actor_kind = _optional_string(payload.get("actor_kind"))
    if actor_kind not in _VISUAL_IDENTITY_ACTOR_KINDS:
        return False
    return _has_actor_id(payload.get("actor_id"))


def resolve_visual_identity_directive(
    resolver: VisualIdentityDirectiveResolver,
    directive: Mapping[str, Any],
) -> VisualDirectiveResolution | None:
    """Resolve one VN sprite directive through Visual Identity without saving cast state."""
    directive_dict = dict(directive)
    if not is_visual_identity_directive(directive_dict):
        return None

    payload = _visual_identity_payload(directive_dict)
    actor_kind = _optional_string(payload.get("actor_kind")) or ""
    actor_id = payload.get("actor_id")
    expression_key = _directive_expression_key(payload, directive_dict)
    try:
        resolved = resolver.resolve_expression_asset(
            actor_kind=actor_kind,
            actor_id=actor_id,
            requested_expression_key=expression_key,
            manual_override_expression_key=_optional_string(
                payload.get("manual_override_expression_key")
            ),
            mood_expression_key=_optional_string(payload.get("mood_expression_key")),
            role_id=_optional_string(payload.get("role_id")),
            role_label=_optional_string(payload.get("role_label")),
            override_pack_id=_optional_int(payload.get("override_pack_id")),
            override_pack_version_id=_optional_int(payload.get("override_pack_version_id")),
            allow_override_fallback=_optional_bool(payload.get("allow_override_fallback")),
        )
    except ValueError as exc:
        return VisualDirectiveResolution(
            applied=False,
            reason=str(exc),
            directive=directive_dict,
        )

    item = _visual_identity_item(directive_dict, resolved)
    if item is None:
        return VisualDirectiveResolution(
            applied=False,
            reason=str(getattr(resolved, "fallback_reason", None) or "asset_not_found"),
            directive=directive_dict,
        )
    return VisualDirectiveResolution(applied=True, item=item, directive=directive_dict)


def _iter_manifest_items(
    manifest: Mapping[str, Any],
    asset_type: Any,
) -> list[dict[str, Any]]:
    assets = manifest.get("assets", {})
    if not isinstance(assets, Mapping):
        return []

    collection_keys = _collection_keys_for_asset_type(asset_type)
    if collection_keys is not None:
        items: list[dict[str, Any]] = []
        for collection_key in collection_keys:
            items.extend(_list_of_dicts(assets.get(collection_key, [])))
        return items

    all_items: list[dict[str, Any]] = []
    for items in assets.values():
        all_items.extend(_list_of_dicts(items))
    return all_items


def _collection_keys_for_asset_type(asset_type: Any) -> list[str] | None:
    if not isinstance(asset_type, str) or not asset_type.strip():
        return None

    normalized = asset_type.strip().lower()
    aliases = _ASSET_TYPE_COLLECTION_ALIASES.get(normalized, (asset_type,))
    return list(dict.fromkeys((*aliases, asset_type)))


def _is_approved_item(item: Mapping[str, Any]) -> bool:
    if item.get("approved") is False:
        return False
    review_status = item.get("review_status")
    if isinstance(review_status, str) and review_status != "approved":
        return False
    status = item.get("status")
    if isinstance(status, str) and status not in {"approved", "ready"}:
        return False
    return True


def _matches_slot_key(item: Mapping[str, Any], directive: Mapping[str, Any]) -> bool:
    slot_key = directive.get("slot_key")
    if not isinstance(slot_key, str) or not slot_key:
        return True
    return item.get("slot_key") == slot_key


def _matches_labels(item: Mapping[str, Any], directive: Mapping[str, Any]) -> bool:
    requested = directive.get("labels", {})
    if not isinstance(requested, Mapping) or not requested:
        return True
    labels = item.get("labels", {})
    if not isinstance(labels, Mapping):
        return False
    return all(labels.get(key) == value for key, value in requested.items())


def _candidate_sort_key(item: Mapping[str, Any], seed: str) -> tuple[int, str]:
    preferred_rank = 0 if bool(item.get("preferred")) else 1
    identity = "|".join(
        [
            seed,
            str(item.get("slot_key", "")),
            str(item.get("item_id", "")),
            str(item.get("variant_index", "")),
        ]
    )
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    return (preferred_rank, digest)


def _list_of_dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _visual_identity_payload(directive: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(directive)
    nested = directive.get("visual_identity")
    if isinstance(nested, Mapping):
        payload.update(dict(nested))
    return payload


def _directive_expression_key(
    payload: Mapping[str, Any],
    directive: Mapping[str, Any],
) -> str:
    for key in ("expression_key", "expression", "mood_expression_key"):
        value = _optional_string(payload.get(key))
        if value:
            return value
    labels = directive.get("labels")
    if isinstance(labels, Mapping):
        for key in ("expression", "expression_key", "emotion", "mood"):
            value = _optional_string(labels.get(key))
            if value:
                return value
    return "neutral"


def _visual_identity_item(
    directive: Mapping[str, Any],
    resolved: Any,
) -> dict[str, Any] | None:
    content_url = _visual_identity_content_url(resolved)
    if content_url is None:
        return None

    labels = dict(directive.get("labels")) if isinstance(directive.get("labels"), Mapping) else {}
    metadata = (
        dict(directive.get("metadata")) if isinstance(directive.get("metadata"), Mapping) else {}
    )
    visual_identity_metadata = {
        "actor_kind": getattr(resolved, "actor_kind", None),
        "actor_id": getattr(resolved, "actor_id", None),
        "role_id": getattr(resolved, "role_id", None),
        "role_label": getattr(resolved, "role_label", None),
        "pack_id": getattr(resolved, "pack_id", None),
        "pack_version_id": getattr(resolved, "pack_version_id", None),
        "asset_id": getattr(resolved, "asset_id", None),
        "expression_key": getattr(resolved, "expression_key", None),
        "requested_expression_key": getattr(resolved, "requested_expression_key", None),
        "fallback_reason": getattr(resolved, "fallback_reason", None),
        "resolution_source": getattr(resolved, "resolution_source", None),
        "is_animated": bool(getattr(resolved, "is_animated", False)),
    }
    metadata["visual_identity"] = visual_identity_metadata

    expression_key = _optional_string(getattr(resolved, "expression_key", None))
    if expression_key:
        labels.setdefault("expression", expression_key)
    role_id = _optional_string(getattr(resolved, "role_id", None))
    role_label = _optional_string(getattr(resolved, "role_label", None))
    if role_id:
        labels.setdefault("role_id", role_id)
    if role_label:
        labels.setdefault("role_label", role_label)

    item = {
        "source": "visual_identity",
        "asset_type": "sprite",
        "slot_key": _visual_identity_slot_key(directive, resolved),
        "content_url": content_url,
        "labels": labels,
        "metadata": metadata,
        "visual_identity_asset_id": getattr(resolved, "asset_id", None),
        "visual_identity_pack_id": getattr(resolved, "pack_id", None),
        "visual_identity_pack_version_id": getattr(resolved, "pack_version_id", None),
        "content_type": getattr(resolved, "content_type", None),
        "storage_relpath": getattr(resolved, "storage_relpath", None),
    }
    return {key: value for key, value in item.items() if value is not None}


def _visual_identity_content_url(resolved: Any) -> str | None:
    asset_url = _optional_string(getattr(resolved, "asset_url", None))
    if asset_url:
        return asset_url
    pack_id = getattr(resolved, "pack_id", None)
    asset_id = getattr(resolved, "asset_id", None)
    if pack_id is None or asset_id is None:
        return None
    try:
        return f"/api/v1/visual-identities/packs/{int(pack_id)}/assets/{int(asset_id)}/content"
    except (TypeError, ValueError):
        return None


def _visual_identity_slot_key(directive: Mapping[str, Any], resolved: Any) -> str:
    slot_key = _optional_string(directive.get("slot_key"))
    if slot_key:
        return slot_key
    actor_kind = _optional_string(getattr(resolved, "actor_kind", None)) or "actor"
    actor_id = _optional_string(getattr(resolved, "actor_id", None)) or "unknown"
    expression_key = _optional_string(getattr(resolved, "expression_key", None)) or "neutral"
    return f"visual_identity.{actor_kind}.{actor_id}.{expression_key}"


def _optional_string(value: Any) -> str | None:
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    return None


def _optional_int(value: Any) -> Any | None:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


def _optional_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return False


def _has_actor_id(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return True
