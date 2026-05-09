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
