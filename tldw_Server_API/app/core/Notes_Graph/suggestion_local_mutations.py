"""Guarded local suggestion mutations through the existing Notes stores.

This adapter performs no admission or authority selection. Its caller supplies a
fresh acceptance guard; that guard runs before product locks and again inside
the existing store mutation. Canonical Sync uses its existing coordinators.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
from tldw_Server_API.app.core.DB_Management.chacha.note_link_store import NotesLink
from tldw_Server_API.app.core.DB_Management.chacha.organization_sync_store import (
    NotesOrganizationSyncStore,
    OrganizationResource,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import InputError
from tldw_Server_API.app.core.Sync.v2.materializers.guarded_product_mutation import (
    GuardedProductMutation,
)
from tldw_Server_API.app.core.Sync.v2.models import normalize_sync_timestamp
from tldw_Server_API.app.core.Sync.v2.notes_organization import organization_link_id

if TYPE_CHECKING:
    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB


class LocalSuggestionMutations:
    """Apply only the three review-approved local Notes mutation shapes."""

    def __init__(self, note_db: CharactersRAGDB) -> None:
        self.note_db = note_db
        self._organization = NotesOrganizationSyncStore(note_db)

    def create_related_link(
        self,
        *,
        edge_id: str,
        source_note_id: str,
        target_note_id: str,
        guarded_mutation: GuardedProductMutation,
    ) -> NotesLink:
        """Create one ordinary undirected link and finalize in its transaction."""

        guarded_mutation.require_identity("notes.link", edge_id)
        source, target = sorted((source_note_id, target_note_id))
        timestamp = normalize_sync_timestamp(self.note_db._get_current_utc_timestamp_iso())
        with self.note_db.transaction() as conn:
            guarded_mutation.before(conn)
            result = self.note_db.notes_link_store.upsert(
                edge_id=edge_id,
                payload={
                    "source_note_id": source,
                    "target_note_id": target,
                    "type": "manual",
                    "directed": False,
                    "weight": 1.0,
                    "label": None,
                    "properties": {},
                    "created_at": timestamp,
                    "last_modified": timestamp,
                    "created_by": f"user:{self.note_db.client_id}",
                },
                expected_version=None,
                conn=conn,
                before=guarded_mutation.before,
                after=guarded_mutation.after,
            )
            return result.link

    def create_keyword(
        self,
        *,
        keyword_sync_id: str,
        display: str,
        guarded_mutation: GuardedProductMutation,
    ) -> OrganizationResource:
        """Create or replay the guarded keyword step without finalizing it."""

        guarded_mutation.require_identity("notes.keyword", keyword_sync_id)
        with self.note_db.transaction() as conn:
            guarded_mutation.before(conn)
            return self._organization.apply_resource(
                domain="notes.keyword",
                object_id=keyword_sync_id,
                operation="upsert",
                payload={"keyword": display},
                before=guarded_mutation.before,
            )

    def find_keyword_identity(self, display: str, *, conn: Any | None = None, for_update: bool = False) -> str | None:
        """Reuse local keyword identities without treating SQLite device labels as owners."""

        if self.note_db.backend_type == BackendType.POSTGRESQL:
            resource = self._organization.find_keyword_by_normalized_identity(display, conn=conn, for_update=for_update)
            return resource.sync_id if resource is not None else None
        normalized = self._organization.normalize_keyword_identity(display)
        with self.note_db.transaction():
            for row in self.note_db.list_keywords(limit=self.note_db.count_keywords()):
                if self._organization.normalize_keyword_identity(row["keyword"]) == normalized:
                    return str(row["sync_id"])
        return None

    def link_keyword(
        self,
        *,
        note_id: str,
        keyword_sync_id: str,
        guarded_mutation: GuardedProductMutation,
    ) -> None:
        """Apply the owner-checked membership and its acceptance finalizer."""

        identity = organization_link_id("notes.keyword_link", ["note", note_id, keyword_sync_id])
        guarded_mutation.require_identity("notes.keyword_link", identity)
        with self.note_db.transaction() as conn:
            guarded_mutation.before(conn)
            if self.note_db.backend_type == BackendType.SQLITE:
                keyword = self.note_db.keyword_store.resolve_merge_survivor(keyword_sync_id, conn=conn, for_update=True)
                if keyword is None or keyword["sync_id"] != keyword_sync_id:
                    raise InputError("Suggestion keyword is missing or deleted")
                self.note_db.link_note_to_keyword(note_id, keyword["id"])
                if guarded_mutation.after is not None:
                    guarded_mutation.after(conn, identity)
                return
            self._organization.apply_relationship(
                domain="notes.keyword_link",
                object_id=identity,
                operation="upsert",
                payload={
                    "subject_type": "note",
                    "subject_id": note_id,
                    "keyword_sync_id": keyword_sync_id,
                },
                routing_metadata={},
                before=guarded_mutation.before,
                after=guarded_mutation.after,
            )
