from tldw_Server_API.app.core.Visual_Identities.service import VisualIdentityResolvedAsset
from tldw_Server_API.app.core.VN_Play.assets import (
    resolve_visual_directive,
    resolve_visual_identity_directive,
)


class _FakeVisualIdentityResolver:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def resolve_expression_asset(self, **kwargs):
        self.calls.append(dict(kwargs))
        return VisualIdentityResolvedAsset(
            actor_kind=str(kwargs["actor_kind"]),
            actor_id=kwargs["actor_id"],
            role_id=kwargs.get("role_id"),
            role_label=kwargs.get("role_label"),
            pack_id=5,
            pack_version_id=6,
            expression_key="happy",
            requested_expression_key=str(kwargs["requested_expression_key"]),
            asset_id=9,
            storage_relpath="visual_identities/asset.webp",
            fallback_reason="requested",
            is_animated=True,
            content_type="image/webp",
            resolution_source="override",
        )


class _StrictOverrideIdResolver:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def resolve_expression_asset(self, **kwargs):
        self.calls.append(dict(kwargs))
        if kwargs.get("override_pack_id") == "bad-pack":
            raise ValueError("pack_not_found")
        raise AssertionError("invalid override id was not preserved for strict validation")


def test_resolver_prefers_preferred_approved_item() -> None:
    manifest = {
        "assets": {
            "sprite": [
                {
                    "item_id": 1,
                    "slot_key": "sprite.happy",
                    "labels": {"emotion": "happy"},
                    "preferred": False,
                },
                {
                    "item_id": 2,
                    "slot_key": "sprite.happy.alt",
                    "labels": {"emotion": "happy"},
                    "preferred": True,
                },
            ]
        }
    }

    resolved = resolve_visual_directive(
        manifest,
        {"asset_type": "sprite", "labels": {"emotion": "happy"}},
        seed="s",
    )

    assert resolved.applied is True
    assert resolved.item["item_id"] == 2


def test_visual_identity_directive_resolves_role_override_sprite() -> None:
    resolver = _FakeVisualIdentityResolver()

    resolved = resolve_visual_identity_directive(
        resolver,
        {
            "asset_type": "sprite",
            "actor_kind": "character",
            "actor_id": 42,
            "role_id": "hero",
            "role_label": "Hero",
            "expression_key": "happy",
            "override_pack_id": 5,
            "override_pack_version_id": 6,
            "allow_override_fallback": True,
            "labels": {"emotion": "happy"},
        },
    )

    assert resolver.calls == [
        {
            "actor_kind": "character",
            "actor_id": 42,
            "requested_expression_key": "happy",
            "manual_override_expression_key": None,
            "mood_expression_key": None,
            "role_id": "hero",
            "role_label": "Hero",
            "override_pack_id": 5,
            "override_pack_version_id": 6,
            "allow_override_fallback": True,
        }
    ]
    assert resolved is not None
    assert resolved.applied is True
    assert resolved.item is not None
    assert resolved.item["source"] == "visual_identity"
    assert resolved.item["asset_type"] == "sprite"
    assert resolved.item["content_url"] == (
        "/api/v1/visual-identities/packs/5/assets/9/content"
    )
    assert resolved.item["labels"]["emotion"] == "happy"
    assert resolved.item["metadata"]["visual_identity"]["role_id"] == "hero"
    assert resolved.item["metadata"]["visual_identity"]["resolution_source"] == "override"


def test_visual_identity_directive_preserves_invalid_override_ids_for_strict_resolver() -> None:
    resolver = _StrictOverrideIdResolver()

    resolved = resolve_visual_identity_directive(
        resolver,
        {
            "asset_type": "sprite",
            "actor_kind": "character",
            "actor_id": 42,
            "expression_key": "happy",
            "override_pack_id": "bad-pack",
            "override_pack_version_id": 6,
        },
    )

    assert resolver.calls[0]["override_pack_id"] == "bad-pack"
    assert resolved is not None
    assert resolved.applied is False
    assert resolved.reason == "pack_not_found"


def test_resolver_supports_manifest_collection_aliases() -> None:
    manifest = {
        "assets": {
            "backgrounds": [
                {
                    "item_id": 10,
                    "slot_key": "background.library",
                    "asset_type": "background",
                    "labels": {"location": "library"},
                    "review_status": "approved",
                    "content_url": "/api/v1/vn-assets/packs/1/items/10/content",
                }
            ],
            "sprites": [
                {
                    "item_id": 20,
                    "slot_key": "sprite.happy",
                    "asset_type": "sprite",
                    "labels": {"emotion": "happy"},
                    "review_status": "approved",
                    "content_url": "/api/v1/vn-assets/packs/1/items/20/content",
                }
            ],
        }
    }

    background = resolve_visual_directive(
        manifest,
        {"asset_type": "background", "labels": {"location": "library"}},
        seed="seed",
    )
    sprite = resolve_visual_directive(
        manifest,
        {"asset_type": "sprite", "labels": {"emotion": "happy"}},
        seed="seed",
    )

    assert background.applied is True
    assert background.item["item_id"] == 10
    assert sprite.applied is True
    assert sprite.item["item_id"] == 20


def test_resolver_rejects_unmatched_directive() -> None:
    resolved = resolve_visual_directive(
        {"assets": {"sprite": []}},
        {"slot_key": "sprite.missing"},
        seed="s",
    )

    assert resolved.applied is False
    assert resolved.reason == "asset_not_found"


def test_resolver_ignores_explicitly_unapproved_items() -> None:
    manifest = {
        "assets": {
            "sprite": [
                {"item_id": 1, "slot_key": "sprite.happy", "review_status": "draft"},
                {"item_id": 2, "slot_key": "sprite.happy", "review_status": "approved"},
            ]
        }
    }

    resolved = resolve_visual_directive(manifest, {"slot_key": "sprite.happy"}, seed="s")

    assert resolved.applied is True
    assert resolved.item["item_id"] == 2


def test_resolver_variant_selection_is_seed_stable() -> None:
    manifest = {
        "assets": {
            "sprites": [
                {
                    "item_id": 1,
                    "slot_key": "sprite.happy",
                    "labels": {"emotion": "happy"},
                    "review_status": "approved",
                    "variant_index": 0,
                },
                {
                    "item_id": 2,
                    "slot_key": "sprite.happy",
                    "labels": {"emotion": "happy"},
                    "review_status": "approved",
                    "variant_index": 1,
                },
            ]
        }
    }

    first = resolve_visual_directive(
        manifest,
        {"asset_type": "sprite", "labels": {"emotion": "happy"}},
        seed="stable-seed",
    )
    second = resolve_visual_directive(
        manifest,
        {"asset_type": "sprite", "labels": {"emotion": "happy"}},
        seed="stable-seed",
    )

    assert first.applied is True
    assert second.applied is True
    assert first.item["item_id"] == second.item["item_id"]
