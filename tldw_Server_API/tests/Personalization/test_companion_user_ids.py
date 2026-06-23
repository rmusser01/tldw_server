from tldw_Server_API.app.core.Personalization.companion_user_ids import (
    resolve_companion_storage_user_id,
)


def test_resolve_companion_storage_user_id_preserves_numeric_ids() -> None:
    assert resolve_companion_storage_user_id("42") == "42"


def test_resolve_companion_storage_user_id_derives_stable_numeric_key_for_text_ids() -> None:
    left = resolve_companion_storage_user_id("user@example.com")
    right = resolve_companion_storage_user_id("user@example.com")

    assert left == right
    assert left.isdigit()
    assert int(left) > 2**32
