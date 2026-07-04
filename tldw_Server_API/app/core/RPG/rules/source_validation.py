from __future__ import annotations

from typing import Any

from tldw_Server_API.app.core.DB_Management.Collections_DB import CollectionsDatabase
from tldw_Server_API.app.core.RPG.rules.refs import RulesPackSourceValidation

READY_COLLECTION_ITEM_STATUSES = {"completed", "skipped_existing"}


class RPGRulesSourceValidator:
    def __init__(self, media_db: Any, collections_db: CollectionsDatabase) -> None:
        self.media_db = media_db
        self.collections_db = collections_db

    async def validate_media_item(self, owner_user_id: int, media_id: int) -> RulesPackSourceValidation:
        media = self._readable_media_by_id(owner_user_id=owner_user_id, media_id=media_id)
        if not media:
            return RulesPackSourceValidation(
                ref_id=f"media_item:{media_id}",
                readable=False,
                display_name=None,
            )
        return RulesPackSourceValidation(
            ref_id=f"media_item:{media_id}",
            readable=True,
            display_name=media_display_name(media),
            ready_media_ids=[int(media_id)],
        )

    async def validate_media_collection(
        self,
        owner_user_id: int,
        collection_id: int,
    ) -> RulesPackSourceValidation:
        try:
            collection = self.collections_db.get_media_collection(collection_id)
        except KeyError:
            return RulesPackSourceValidation(
                ref_id=f"media_collection:{collection_id}",
                readable=False,
                display_name=None,
            )

        ready_media_ids = [
            int(item.media_id)
            for item in collection.items
            if item.media_id is not None
            and item.status in READY_COLLECTION_ITEM_STATUSES
            and self._readable_media_by_id(owner_user_id=owner_user_id, media_id=int(item.media_id)) is not None
        ]
        return RulesPackSourceValidation(
            ref_id=f"media_collection:{collection_id}",
            readable=True,
            display_name=collection.name,
            ready_media_ids=ready_media_ids,
        )

    def _readable_media_by_id(self, *, owner_user_id: int, media_id: int) -> dict[str, Any] | None:
        media = self.media_db.get_media_by_id(media_id, include_deleted=False, include_trash=False)
        if not media:
            return None
        if not media_belongs_to_owner(media, owner_user_id=owner_user_id, media_db=self.media_db):
            return None
        return media


def media_display_name(media: dict[str, Any]) -> str | None:
    for key in ("title", "name", "filename", "url"):
        value = str(media.get(key) or "").strip()
        if value:
            return value
    return None


def media_belongs_to_owner(media: dict[str, Any], *, owner_user_id: int, media_db: Any) -> bool:
    allowed_owner_ids = {str(owner_user_id)}
    client_id = getattr(media_db, "client_id", None)
    if client_id is not None:
        allowed_owner_ids.add(str(client_id))

    owner_value = media.get("owner_user_id")
    if owner_value is not None:
        owner_text = str(owner_value).strip()
        return not owner_text or owner_text in allowed_owner_ids

    client_value = media.get("client_id")
    if client_value is not None:
        client_text = str(client_value).strip()
        return not client_text or client_text in allowed_owner_ids

    return True
