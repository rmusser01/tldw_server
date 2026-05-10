"""Service layer for user-scoped Persona Visual pack library operations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    ConflictError,
    InputError,
)
from tldw_Server_API.app.core.Persona.visual_service import (
    PersonaVisualService,
    PersonaVisualServiceError,
)

if TYPE_CHECKING:
    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB


_UNSET = object()


class PersonaVisualLibraryServiceError(Exception):
    """Service-level Persona Visual library failure with stable API-facing codes."""

    def __init__(self, code: str, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.details = details or {}


class PersonaVisualLibraryService:
    """Reference-backed personal library for reusable Persona Visual packs."""

    def __init__(
        self,
        db: CharactersRAGDB,
        *,
        visual_service: PersonaVisualService | None = None,
    ) -> None:
        self._db = db
        self._visual_service = visual_service or PersonaVisualService(db)

    def save_pack(
        self,
        *,
        user_id: str,
        source_persona_id: str,
        source_pack_id: str,
        title: str | None = None,
        notes: str | None = None,
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        """Save a same-user source pack as idempotent personal-library metadata."""
        try:
            return self._db.upsert_persona_visual_library_item(
                user_id=str(user_id or "").strip(),
                source_persona_id=str(source_persona_id or "").strip(),
                source_pack_id=str(source_pack_id or "").strip(),
                title=self._normalize_optional_title(title),
                notes=self._normalize_optional_notes(notes),
                tags=self._normalize_tags(tags),
            )
        except InputError as exc:
            raise PersonaVisualLibraryServiceError(
                "invalid_library_metadata",
                str(exc),
            ) from exc
        except ConflictError as exc:
            raise PersonaVisualLibraryServiceError(
                "source_pack_not_found",
                "Source Persona Visual pack not found for user.",
                details={"source_pack_id": source_pack_id},
            ) from exc

    def list_items(
        self,
        *,
        user_id: str,
        include_deleted: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        return self._db.list_persona_visual_library_items(
            user_id=str(user_id or "").strip(),
            include_deleted=include_deleted,
            limit=limit,
            offset=offset,
        )

    def get_item_for_source(
        self,
        *,
        user_id: str,
        source_persona_id: str,
        source_pack_id: str,
    ) -> dict[str, Any] | None:
        return self._db.get_persona_visual_library_item_by_source(
            user_id=str(user_id or "").strip(),
            source_persona_id=str(source_persona_id or "").strip(),
            source_pack_id=str(source_pack_id or "").strip(),
        )

    def update_item(
        self,
        *,
        user_id: str,
        item_id: str,
        title: str | None | object = _UNSET,
        notes: str | None | object = _UNSET,
        tags: list[str] | None | object = _UNSET,
        expected_version: int | None = None,
    ) -> dict[str, Any]:
        try:
            update_kwargs: dict[str, Any] = {}
            if title is not _UNSET:
                update_kwargs["title"] = self._normalize_required_title(title if title is not None else None)
            if notes is not _UNSET:
                update_kwargs["notes"] = self._normalize_optional_notes(notes if notes is not None else None)
            if tags is not _UNSET:
                update_kwargs["tags"] = self._normalize_tags(tags if tags is not None else None)
            updated = self._db.update_persona_visual_library_item(
                user_id=str(user_id or "").strip(),
                item_id=str(item_id or "").strip(),
                expected_version=expected_version,
                **update_kwargs,
            )
        except InputError as exc:
            raise PersonaVisualLibraryServiceError(
                "invalid_library_metadata",
                str(exc),
            ) from exc
        except ConflictError as exc:
            raise PersonaVisualLibraryServiceError(
                "library_item_conflict",
                str(exc),
                details={"item_id": item_id},
            ) from exc
        if not updated:
            raise PersonaVisualLibraryServiceError(
                "library_item_not_found",
                "Persona Visual library item not found for user.",
                details={"item_id": item_id},
            )
        return updated

    def delete_item(
        self,
        *,
        user_id: str,
        item_id: str,
    ) -> bool:
        return self._db.soft_delete_persona_visual_library_item(
            user_id=str(user_id or "").strip(),
            item_id=str(item_id or "").strip(),
        )

    def use_item_for_persona(
        self,
        *,
        user_id: str,
        item_id: str,
        target_persona_id: str,
        title: str | None = None,
    ) -> dict[str, Any]:
        item = self._db.get_persona_visual_library_item(
            user_id=str(user_id or "").strip(),
            item_id=str(item_id or "").strip(),
        )
        if not item:
            raise PersonaVisualLibraryServiceError(
                "library_item_not_found",
                "Persona Visual library item not found for user.",
                details={"item_id": item_id},
            )

        source_persona_id = str(item.get("source_persona_id") or "").strip()
        source_pack_id = str(item.get("source_pack_id") or "").strip()
        if not item.get("source_available") or not source_persona_id or not source_pack_id:
            raise PersonaVisualLibraryServiceError(
                "source_pack_unavailable",
                "Source Persona Visual pack is no longer available.",
                details={"item_id": item_id, "source_pack_id": source_pack_id or None},
            )

        title_value = self._normalize_optional_title(title)
        if title_value is None:
            title_value = str(item.get("title") or item.get("source_pack_title") or "").strip() or None
        try:
            return self._visual_service.duplicate_pack_to_persona(
                source_persona_id=source_persona_id,
                user_id=str(user_id or "").strip(),
                pack_id=source_pack_id,
                target_persona_id=str(target_persona_id or "").strip(),
                title=title_value,
            )
        except PersonaVisualServiceError as exc:
            if exc.code == "pack_not_found":
                raise PersonaVisualLibraryServiceError(
                    "source_pack_unavailable",
                    "Source Persona Visual pack is no longer available.",
                    details={"item_id": item_id, "source_pack_id": source_pack_id},
                ) from exc
            if exc.code in {"target_persona_not_found", "same_persona_target_unsupported"}:
                raise PersonaVisualLibraryServiceError(
                    exc.code,
                    str(exc),
                    details=exc.details,
                ) from exc
            raise PersonaVisualLibraryServiceError(
                exc.code,
                str(exc),
                details=exc.details,
            ) from exc

    @staticmethod
    def _normalize_optional_title(value: str | None) -> str | None:
        if value is None:
            return None
        return PersonaVisualLibraryService._normalize_required_title(value)

    @staticmethod
    def _normalize_required_title(value: str | None) -> str:
        normalized = str(value or "").strip()
        if not normalized:
            raise InputError("title cannot be empty.")  # noqa: TRY003
        if len(normalized) > 200:
            raise InputError("title must be 200 characters or fewer.")  # noqa: TRY003
        return normalized

    @staticmethod
    def _normalize_optional_notes(value: str | None) -> str | None:
        if value is None:
            return None
        normalized = str(value).strip()
        if not normalized:
            return None
        if len(normalized) > 4000:
            raise InputError("notes must be 4000 characters or fewer.")  # noqa: TRY003
        return normalized

    @staticmethod
    def _normalize_tags(value: list[str] | None) -> list[str]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise InputError("tags must be a list.")  # noqa: TRY003
        normalized: list[str] = []
        seen: set[str] = set()
        for raw_tag in value:
            tag = str(raw_tag or "").strip().lower()
            if not tag or tag in seen:
                continue
            if len(tag) > 64:
                raise InputError("tags must be 64 characters or fewer.")  # noqa: TRY003
            seen.add(tag)
            normalized.append(tag)
            if len(normalized) >= 20:
                break
        return normalized
