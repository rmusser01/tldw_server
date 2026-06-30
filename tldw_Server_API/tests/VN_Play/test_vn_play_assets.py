from tldw_Server_API.app.core.VN_Play.assets import resolve_visual_directive


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
