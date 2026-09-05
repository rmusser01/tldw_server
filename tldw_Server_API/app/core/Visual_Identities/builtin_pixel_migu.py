"""One-time, user-owned pixel-migu character and expression-pack installation."""

from __future__ import annotations

import json
from dataclasses import asdict
from importlib import resources

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import BackendType, CharactersRAGDB
from tldw_Server_API.app.core.Visual_Identities.service import VisualIdentityService
from tldw_Server_API.app.core.Visual_Identities.storage import validate_and_store_visual_identity_asset


def ensure_pixel_migu_character(db: CharactersRAGDB, *, owner_user_id: int) -> int | None:
    """Seed once per user, preserving later edits, bindings and deletion choices.

    Args:
        db: The owner's per-user ChaChaNotes database.
        owner_user_id: Authenticated owner of the database and copied assets.

    Returns:
        The original seeded character ID, or None for an existing name collision
        or a backend without Shared Visual Identity support.
    """
    if db.backend_type != BackendType.SQLITE:
        return None
    service = VisualIdentityService(db, owner_user_id=owner_user_id)
    repo = service.repository
    repo.initialize_schema()
    receipt_key = {
        "owner_user_id": owner_user_id,
        "scope": "builtin_character",
        "resource_id": "pixel-migu",
        "idempotency_key": "initial-install",
    }
    # Serialize concurrent first-open attempts; every DB side effect and the
    # permanent receipt commit together. Never repair user content on replay.
    with db.transaction():
        receipt = repo.get_idempotency_record(**receipt_key)
        if receipt is not None:
            return json.loads(receipt["response_json"]).get("character_id")
        character_id = None
        if db.get_character_card_by_name("pixel-migu", include_deleted=True) is None:
            character_id = _install_character(service)
        repo.create_idempotency_record(
            **receipt_key,
            payload_hash="pixel-migu-initial-install",
            response={"character_id": character_id},
        )
        return character_id


def _install_character(service: VisualIdentityService) -> int:
    """Copy the immutable bundled card and expression assets into owned storage."""
    root = resources.files("tldw_Server_API.app.core.Visual_Identities").joinpath("assets", "pixel-migu")
    card = json.loads(root.joinpath("pixel-migu.character.json").read_text(encoding="utf-8"))["data"]
    card["first_message"] = card.pop("first_mes")
    card["message_example"] = card.pop("mes_example")
    card["image"] = root.joinpath("pixel-migu.character.png").read_bytes()
    card["creator_notes"] = "Bundled pixel-migu with 18 expression slots. Buddy visuals are configured separately."
    character_id = service.db.add_character_card(card)
    if character_id is None:
        raise RuntimeError("pixel_migu_character_creation_failed")
    manifest = json.loads(root.joinpath("visual_identity_pack.json").read_text(encoding="utf-8"))
    draft = service.repository.create_draft(
        owner_user_id=service.owner_user_id,
        title=manifest["title"],
        source_kind="imported",
        source_filename="pixel-migu bundled expressions",
        status="ready_for_review",
        default_expression_key=manifest["default_expression_key"],
    )
    for asset in manifest["assets"]:
        filename = asset["storage_relpath"].rsplit("/", 1)[-1]
        with resources.as_file(root.joinpath("expressions", filename)) as source_path:
            stored = validate_and_store_visual_identity_asset(
                source_path=source_path,
                owner_user_id=service.owner_user_id,
                expression_key=asset["expression_key"],
                content_type="image/png",
            )
        if stored.sha256 != asset["sha256"]:
            raise ValueError("pixel_migu_asset_hash_mismatch")
        metadata = asdict(stored)
        metadata["storage_relpath"] = metadata.pop("relpath")
        service.repository.create_asset(
            owner_user_id=service.owner_user_id,
            draft_id=int(draft["id"]),
            expression_key=asset["expression_key"],
            original_expression_key=asset["original_label"],
            display_label=asset["display_label"],
            source_filename=filename,
            **metadata,
        )
    service.activate_draft(draft_id=int(draft["id"]), actor_kind="character", actor_id=character_id)
    return character_id
