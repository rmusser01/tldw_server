from __future__ import annotations

"""Backend-neutral ChaCha projection seam for Notes organization Sync domains."""

import hashlib
import json
from collections import defaultdict, deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    BackendType,
    ConflictError,
    InputError,
)
from tldw_Server_API.app.core.Sync.v2.models import SyncDomain, SyncOperation
from tldw_Server_API.app.core.Sync.v2.notes_organization import (
    NotesOrganizationValidationError,
    organization_link_id,
    parse_notes_organization_payload,
    validate_organization_object_id,
)

if TYPE_CHECKING:
    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB


@dataclass(frozen=True)
class OrganizationResource:
    domain: SyncDomain
    sync_id: str
    local_id: int
    name: str
    parent_sync_id: str | None
    deleted: bool
    version: int


@dataclass(frozen=True)
class OrganizationRelationship:
    domain: SyncDomain
    object_id: str
    payload: Mapping[str, object]


@dataclass(frozen=True)
class OrganizationSnapshot:
    resources: tuple[OrganizationResource, ...]
    relationships: tuple[OrganizationRelationship, ...]


@dataclass(frozen=True)
class SourceFolderTransitionPlan:
    """One request-bound source delta and its exact product pre/post states."""

    operation: SyncOperation | None
    pre_state_hash: str
    post_state_hash: str
    transition_identity: str


_RESOURCE_TABLES: dict[SyncDomain, tuple[str, str]] = {
    "notes.keyword": ("keywords", "keyword"),
    "notes.keyword_collection": ("keyword_collections", "name"),
    "notes.folder": ("note_folders", "name"),
}


class NotesOrganizationSyncStore:
    """Project canonical Notes organization resources into one user's ChaCha DB."""

    def __init__(self, db: CharactersRAGDB) -> None:
        self._db = db

    @property
    def _owner_id(self) -> str:
        return str(self._db.client_id)

    def _deleted_value(self, deleted: bool) -> bool | int:
        if self._db.backend_type == BackendType.POSTGRESQL:
            return deleted
        return 1 if deleted else 0

    def _table(self, domain: SyncDomain) -> tuple[str, str]:
        try:
            logical_table, name_column = _RESOURCE_TABLES[domain]
        except KeyError as exc:
            raise InputError(f"Unsupported organization resource domain: {domain}") from exc
        return self._db._map_table_for_backend(logical_table), name_column

    @staticmethod
    def _validated_payload(
        domain: SyncDomain,
        operation: SyncOperation,
        object_id: str,
        payload: Mapping[str, object],
    ) -> dict[str, object]:
        try:
            normalized = parse_notes_organization_payload(domain, operation, payload)
            validate_organization_object_id(domain, object_id, normalized)
            return normalized
        except NotesOrganizationValidationError as exc:
            raise InputError(str(exc)) from exc

    def _resource_from_row(
        self,
        conn: Any,
        domain: SyncDomain,
        row: Mapping[str, object],
    ) -> OrganizationResource:
        row = dict(row)
        parent_sync_id: str | None = None
        parent_id = row.get("parent_id")
        if parent_id is not None and domain in {"notes.keyword_collection", "notes.folder"}:
            table, _ = self._table(domain)
            parent = conn.execute(
                f"SELECT sync_id FROM {table} WHERE id = ? AND client_id = ?",  # nosec B608
                (int(parent_id), self._owner_id),
            ).fetchone()
            if not parent:
                raise InputError("Organization parent chain is invalid")
            parent_sync_id = str(parent["sync_id"])
        _, name_column = self._table(domain)
        return OrganizationResource(
            domain=domain,
            sync_id=str(row["sync_id"]),
            local_id=int(row["id"]),
            name=str(row[name_column]),
            parent_sync_id=parent_sync_id,
            deleted=bool(row["deleted"]),
            version=int(row["version"]),
        )

    def _get_resource_locked(
        self,
        conn: Any,
        domain: SyncDomain,
        sync_id: str,
        *,
        for_update: bool = False,
    ) -> OrganizationResource | None:
        table, _ = self._table(domain)
        query = f"SELECT * FROM {table} WHERE sync_id = ? AND client_id = ?"  # nosec B608
        if for_update and self._db.backend_type == BackendType.POSTGRESQL:
            query += " FOR UPDATE"
        row = conn.execute(
            query,
            (sync_id, self._owner_id),
        ).fetchone()
        return self._resource_from_row(conn, domain, row) if row else None

    def get_resource(
        self,
        domain: SyncDomain,
        sync_id: str,
    ) -> OrganizationResource | None:
        """Return one active or deleted resource by canonical identity."""

        with self._db.transaction() as conn:
            return self._get_resource_locked(conn, domain, sync_id)

    def get_resource_row_by_local_id(
        self,
        domain: SyncDomain,
        local_id: int,
        *,
        include_deleted: bool = False,
    ) -> dict[str, Any] | None:
        """Return one owner-scoped resource row by its local product identity."""

        table, _ = self._table(domain)
        query = f"SELECT * FROM {table} WHERE id = ? AND client_id = ?"  # nosec B608
        params: list[object] = [int(local_id), self._owner_id]
        if not include_deleted:
            query += " AND deleted = ?"
            params.append(self._deleted_value(False))
        with self._db.transaction() as conn:
            row = conn.execute(query, tuple(params)).fetchone()
        return dict(row) if row is not None else None

    def snapshot(self) -> OrganizationSnapshot:
        """Return a transactionally consistent organization snapshot."""

        resources: list[OrganizationResource] = []
        relationships: dict[tuple[str, str], OrganizationRelationship] = {}
        with self._db.transaction() as conn:
            for domain in cast(tuple[SyncDomain, ...], tuple(_RESOURCE_TABLES)):
                table, _ = self._table(domain)
                rows = conn.execute(
                    f"SELECT * FROM {table} WHERE client_id = ? ORDER BY id",  # nosec B608
                    (self._owner_id,),
                ).fetchall()
                resources.extend(self._resource_from_row(conn, domain, row) for row in rows)

            keyword_table = self._db._map_table_for_backend("keywords")
            for subject_type, link_table, subject_column in (
                ("note", "note_keywords", "note_id"),
                ("conversation", "conversation_keywords", "conversation_id"),
            ):
                subject_table = "notes" if subject_type == "note" else "conversations"
                rows = conn.execute(
                    f"SELECT l.{subject_column} AS subject_id, k.sync_id AS keyword_sync_id "  # nosec B608
                    f"FROM {link_table} l JOIN {keyword_table} k ON k.id = l.keyword_id "
                    f"JOIN {subject_table} subject ON subject.id = l.{subject_column} "
                    "WHERE k.client_id = ? AND subject.client_id = ?",
                    (self._owner_id, self._owner_id),
                ).fetchall()
                for row in rows:
                    payload = {
                        "subject_type": subject_type,
                        "subject_id": str(row["subject_id"]),
                        "keyword_sync_id": str(row["keyword_sync_id"]),
                    }
                    object_id = organization_link_id(
                        "notes.keyword_link",
                        [subject_type, payload["subject_id"], payload["keyword_sync_id"]],
                    )
                    relationships[("notes.keyword_link", object_id)] = OrganizationRelationship(
                        domain="notes.keyword_link", object_id=object_id, payload=payload
                    )

            collection_rows = conn.execute(
                f"SELECT c.sync_id AS collection_sync_id, k.sync_id AS keyword_sync_id "  # nosec B608
                "FROM collection_keywords l "
                "JOIN keyword_collections c ON c.id = l.collection_id "
                f"JOIN {keyword_table} k ON k.id = l.keyword_id "  # nosec B608
                "WHERE c.client_id = ? AND k.client_id = ?",
                (self._owner_id, self._owner_id),
            ).fetchall()
            for row in collection_rows:
                payload = {
                    "collection_sync_id": str(row["collection_sync_id"]),
                    "keyword_sync_id": str(row["keyword_sync_id"]),
                }
                object_id = organization_link_id(
                    "notes.keyword_collection_link",
                    [payload["collection_sync_id"], payload["keyword_sync_id"]],
                )
                relationships[("notes.keyword_collection_link", object_id)] = OrganizationRelationship(
                    domain="notes.keyword_collection_link", object_id=object_id, payload=payload
                )

            folder_rows = conn.execute(
                "SELECT memberships.note_id, f.sync_id AS folder_sync_id "
                "FROM ("
                "SELECT note_id, folder_id FROM note_folder_memberships "
                "UNION SELECT note_id, folder_id FROM note_folder_source_memberships"
                ") memberships JOIN note_folders f ON f.id = memberships.folder_id "
                "JOIN notes note ON note.id = memberships.note_id "
                "WHERE f.client_id = ? AND note.client_id = ? AND NOT EXISTS ("
                "SELECT 1 FROM note_folder_sync_suppressions suppression "
                "WHERE suppression.note_id = memberships.note_id "
                "AND suppression.folder_id = memberships.folder_id)",
                (self._owner_id, self._owner_id),
            ).fetchall()
            for row in folder_rows:
                payload = {
                    "note_id": str(row["note_id"]),
                    "folder_sync_id": str(row["folder_sync_id"]),
                }
                object_id = organization_link_id(
                    "notes.folder_link", [payload["note_id"], payload["folder_sync_id"]]
                )
                relationships[("notes.folder_link", object_id)] = OrganizationRelationship(
                    domain="notes.folder_link", object_id=object_id, payload=payload
                )

        return OrganizationSnapshot(
            resources=tuple(sorted(resources, key=lambda item: (item.domain, item.sync_id))),
            relationships=tuple(
                relationships[key] for key in sorted(relationships)
            ),
        )

    def relationship_present(
        self,
        *,
        domain: SyncDomain,
        object_id: str,
        payload: Mapping[str, object],
    ) -> bool:
        """Return whether one canonical relationship is effectively present."""

        normalized = self._validated_payload(domain, "upsert", object_id, payload)
        keyword_table = self._db._map_table_for_backend("keywords")
        owner_id = self._owner_id
        with self._db.transaction() as conn:
            if domain == "notes.keyword_link":
                subject_type = str(normalized["subject_type"])
                link_table = (
                    "note_keywords"
                    if subject_type == "note"
                    else "conversation_keywords"
                )
                subject_table = "notes" if subject_type == "note" else "conversations"
                subject_column = (
                    "note_id" if subject_type == "note" else "conversation_id"
                )
                query = (
                    f"SELECT 1 FROM {link_table} link "  # nosec B608
                    f"JOIN {keyword_table} keyword ON keyword.id = link.keyword_id "
                    f"JOIN {subject_table} subject ON subject.id = link.{subject_column} "
                    "WHERE subject.id = ? AND keyword.sync_id = ? "
                    "AND subject.client_id = ? AND keyword.client_id = ? LIMIT 1"
                )
                params = (
                    str(normalized["subject_id"]),
                    str(normalized["keyword_sync_id"]),
                    owner_id,
                    owner_id,
                )
            elif domain == "notes.keyword_collection_link":
                collection_table, _ = self._table("notes.keyword_collection")
                query = (
                    "SELECT 1 FROM collection_keywords link "
                    f"JOIN {collection_table} collection ON collection.id = link.collection_id "  # nosec B608
                    f"JOIN {keyword_table} keyword ON keyword.id = link.keyword_id "
                    "WHERE collection.sync_id = ? AND keyword.sync_id = ? "
                    "AND collection.client_id = ? AND keyword.client_id = ? LIMIT 1"
                )
                params = (
                    str(normalized["collection_sync_id"]),
                    str(normalized["keyword_sync_id"]),
                    owner_id,
                    owner_id,
                )
            elif domain == "notes.folder_link":
                folder_table, _ = self._table("notes.folder")
                query = (
                    f"SELECT 1 FROM {folder_table} folder "  # nosec B608
                    "JOIN notes note ON note.id = ? "
                    "WHERE folder.sync_id = ? "
                    "AND note.client_id = ? AND folder.client_id = ? "
                    "AND ("
                    "EXISTS (SELECT 1 FROM note_folder_memberships membership "
                    "WHERE membership.note_id = note.id AND membership.folder_id = folder.id) "
                    "OR EXISTS (SELECT 1 FROM note_folder_source_memberships source "
                    "WHERE source.note_id = note.id AND source.folder_id = folder.id)"
                    ") AND NOT EXISTS ("
                    "SELECT 1 FROM note_folder_sync_suppressions suppression "
                    "WHERE suppression.note_id = note.id AND suppression.folder_id = folder.id"
                    ") LIMIT 1"
                )
                params = (
                    str(normalized["note_id"]),
                    str(normalized["folder_sync_id"]),
                    owner_id,
                    owner_id,
                )
            else:
                raise InputError(
                    f"Unsupported organization relationship domain: {domain}"
                )
            return conn.execute(query, params).fetchone() is not None

    def _validated_parent_id(
        self,
        conn: Any,
        *,
        domain: SyncDomain,
        object_id: str,
        parent_sync_id: object,
    ) -> int | None:
        if parent_sync_id is None:
            return None
        table, _ = self._table(domain)
        parent = conn.execute(
            f"SELECT id, parent_id, deleted FROM {table} "  # nosec B608
            "WHERE sync_id = ? AND client_id = ?",
            (str(parent_sync_id), self._owner_id),
        ).fetchone()
        if not parent or bool(parent["deleted"]):
            raise InputError("Organization parent is missing or deleted")
        parent_id = int(parent["id"])
        current = conn.execute(
            f"SELECT id FROM {table} WHERE sync_id = ? AND client_id = ?",  # nosec B608
            (object_id, self._owner_id),
        ).fetchone()
        current_id = int(current["id"]) if current else None
        if current_id == parent_id:
            raise InputError("Organization resource cannot be its own parent")

        seen: set[int] = set()
        cursor_id: int | None = parent_id
        while cursor_id is not None:
            if cursor_id in seen or cursor_id == current_id:
                raise InputError("Organization parent would create a cycle")
            seen.add(cursor_id)
            ancestor = conn.execute(
                f"SELECT parent_id FROM {table} WHERE id = ? AND client_id = ?",  # nosec B608
                (cursor_id, self._owner_id),
            ).fetchone()
            if not ancestor:
                raise InputError("Organization parent chain is invalid")
            raw_parent = ancestor["parent_id"]
            cursor_id = int(raw_parent) if raw_parent is not None else None
        return parent_id

    @staticmethod
    def _folder_name(value: object) -> str:
        name = str(value or "").strip()
        if not name or name in {".", ".."} or "/" in name or "\\" in name:
            raise InputError("Folder name must be one non-empty relative path segment")
        return name

    def _apply_folder_locked(
        self,
        conn: Any,
        *,
        object_id: str,
        payload: Mapping[str, object],
        existing: OrganizationResource | None,
    ) -> OrganizationResource:
        name = self._folder_name(payload["name"])
        parent_id = self._validated_parent_id(
            conn,
            domain="notes.folder",
            object_id=object_id,
            parent_sync_id=payload.get("parent_sync_id"),
        )
        parent_path = ""
        if parent_id is not None:
            parent = conn.execute(
                "SELECT path FROM note_folders WHERE id = ? AND client_id = ?",
                (parent_id, self._owner_id),
            ).fetchone()
            parent_path = str(parent["path"])
        root_path = f"{parent_path}/{name}" if parent_path else name
        if len(root_path) > 500:
            raise InputError("Folder path cannot exceed 500 characters")

        now = self._db._get_current_utc_timestamp_iso()
        if existing is None:
            duplicate = conn.execute(
                "SELECT id FROM note_folders "
                "WHERE client_id = ? AND LOWER(path) = LOWER(?) LIMIT 1",
                (self._owner_id, root_path),
            ).fetchone()
            if duplicate:
                raise ConflictError("Folder path already exists", entity="note_folders", entity_id=root_path)
            conn.execute(
                "INSERT INTO note_folders("
                "sync_id, name, path, parent_id, created_at, last_modified, deleted, client_id, version"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    object_id,
                    name,
                    root_path,
                    parent_id,
                    now,
                    now,
                    self._deleted_value(False),
                    self._db.client_id,
                    1,
                ),
            )
            return cast(OrganizationResource, self._get_resource_locked(conn, "notes.folder", object_id))

        rows = [
            dict(row)
            for row in conn.execute(
                "SELECT * FROM note_folders WHERE client_id = ?",
                (self._owner_id,),
            ).fetchall()
        ]
        by_id = {int(row["id"]): row for row in rows}
        children: dict[int, list[int]] = defaultdict(list)
        for row in rows:
            if row["parent_id"] is not None:
                children[int(row["parent_id"])].append(int(row["id"]))
        subtree: list[int] = []
        visited: set[int] = set()
        queue = deque([existing.local_id])
        while queue:
            folder_id = queue.popleft()
            if folder_id in visited or folder_id not in by_id:
                raise InputError("Folder hierarchy contains a cycle or invalid descendant")
            visited.add(folder_id)
            subtree.append(folder_id)
            queue.extend(sorted(children.get(folder_id, [])))

        paths = {existing.local_id: root_path}
        for folder_id in subtree[1:]:
            row = by_id[folder_id]
            ancestor_path = paths[int(row["parent_id"])]
            projected = f"{ancestor_path}/{row['name']}"
            if len(projected) > 500:
                raise InputError("Folder path cannot exceed 500 characters")
            paths[folder_id] = projected
        projected_keys = [path.casefold() for path in paths.values()]
        if len(projected_keys) != len(set(projected_keys)):
            raise ConflictError("Folder subtree paths would conflict", entity="note_folders")
        outside_keys = {
            str(row["path"]).casefold() for row in rows if int(row["id"]) not in paths
        }
        if outside_keys.intersection(projected_keys):
            raise ConflictError("Folder path already exists", entity="note_folders", entity_id=root_path)

        current_row = by_id[existing.local_id]
        unchanged = (
            not bool(current_row["deleted"])
            and str(current_row["name"]) == name
            and current_row["parent_id"] == parent_id
        )
        if unchanged:
            return existing

        for folder_id in subtree:
            conn.execute(
                "UPDATE note_folders SET path = ? WHERE id = ? AND client_id = ?",
                (
                    f"__sync_repath__/{by_id[folder_id]['sync_id']}",
                    folder_id,
                    self._owner_id,
                ),
            )
        conn.execute(
            "UPDATE note_folders SET name = ?, path = ?, parent_id = ?, deleted = ?, "
            "last_modified = ?, client_id = ?, version = ? "
            "WHERE id = ? AND client_id = ?",
            (
                name,
                paths[existing.local_id],
                parent_id,
                self._deleted_value(False),
                now,
                self._db.client_id,
                existing.version + 1,
                existing.local_id,
                self._owner_id,
            ),
        )
        for folder_id in subtree[1:]:
            conn.execute(
                "UPDATE note_folders SET path = ? WHERE id = ? AND client_id = ?",
                (paths[folder_id], folder_id, self._owner_id),
            )
        return cast(OrganizationResource, self._get_resource_locked(conn, "notes.folder", object_id))

    def apply_resource(
        self,
        *,
        domain: SyncDomain,
        object_id: str,
        operation: SyncOperation,
        payload: Mapping[str, object],
        merge_relationship_set_hash: str | None = None,
        before: Callable[[Any], None] | None = None,
        after: Callable[[Any, str], None] | None = None,
    ) -> OrganizationResource:
        """Apply one resource envelope in a ChaCha transaction."""

        if (before is not None or after is not None) and (
            domain != "notes.keyword" or operation != "upsert"
        ):
            raise InputError("Guarded organization resource must be a keyword upsert")
        normalized = self._validated_payload(domain, operation, object_id, payload)
        table, name_column = self._table(domain)
        with self._db.transaction() as conn:
            existing = self._get_resource_locked(
                conn,
                domain,
                object_id,
                for_update=before is not None or after is not None,
            )
            if operation == "tombstone":
                if existing is None:
                    raise InputError("Cannot tombstone an unknown organization resource")
                if merge_relationship_set_hash is not None:
                    if domain != "notes.keyword":
                        raise InputError("Keyword merge precondition domain is invalid")
                    if (
                        self._db.keyword_store.synchronized_relationship_set_hash(
                            existing.local_id,
                            conn=conn,
                        )
                        != merge_relationship_set_hash
                    ):
                        raise ConflictError(
                            "Keyword relationships changed after merge planning"
                        )
                if existing.deleted:
                    return existing
                conn.execute(
                    f"UPDATE {table} SET deleted = ?, last_modified = ?, client_id = ?, version = ? "  # nosec B608
                    "WHERE sync_id = ? AND client_id = ?",
                    (
                        self._deleted_value(True),
                        self._db._get_current_utc_timestamp_iso(),
                        self._db.client_id,
                        existing.version + 1,
                        object_id,
                        self._owner_id,
                    ),
                )
                return cast(OrganizationResource, self._get_resource_locked(conn, domain, object_id))

            if domain == "notes.folder":
                return self._apply_folder_locked(
                    conn, object_id=object_id, payload=normalized, existing=existing
                )

            name = str(normalized[name_column]).strip()
            duplicate = conn.execute(
                f"SELECT id, sync_id FROM {table} WHERE LOWER({name_column}) = LOWER(?) "  # nosec B608
                "AND sync_id <> ? AND client_id = ? LIMIT 1",
                (name, object_id, self._owner_id),
            ).fetchone()
            if duplicate:
                raise ConflictError(
                    f"{domain} name already exists",
                    entity=table,
                    entity_id=name,
                )
            parent_id = None
            if domain == "notes.keyword_collection":
                parent_id = self._validated_parent_id(
                    conn,
                    domain=domain,
                    object_id=object_id,
                    parent_sync_id=normalized.get("parent_sync_id"),
                )
            if before is not None:
                before(conn)
            now = self._db._get_current_utc_timestamp_iso()
            if existing is None:
                columns = ["sync_id", name_column]
                values: list[object] = [object_id, name]
                if domain == "notes.keyword_collection":
                    columns.append("parent_id")
                    values.append(parent_id)
                columns.extend(["created_at", "last_modified", "deleted", "client_id", "version"])
                values.extend([now, now, self._deleted_value(False), self._db.client_id, 1])
                placeholders = ", ".join("?" for _ in columns)
                conn.execute(
                    f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})",  # nosec B608
                    tuple(values),
                )
            else:
                parent_unchanged = (
                    domain != "notes.keyword_collection"
                    or existing.parent_sync_id == normalized.get("parent_sync_id")
                )
                if not existing.deleted and existing.name == name and parent_unchanged:
                    if after is not None:
                        after(conn, object_id)
                    return existing
                set_parts = [f"{name_column} = ?"]
                values = [name]
                if domain == "notes.keyword_collection":
                    set_parts.append("parent_id = ?")
                    values.append(parent_id)
                set_parts.extend(["deleted = ?", "last_modified = ?", "client_id = ?", "version = ?"])
                values.extend(
                    [
                        self._deleted_value(False),
                        now,
                        self._db.client_id,
                        existing.version + 1,
                        object_id,
                        self._owner_id,
                    ]
                )
                conn.execute(
                    f"UPDATE {table} SET {', '.join(set_parts)} "  # nosec B608
                    "WHERE sync_id = ? AND client_id = ?",
                    tuple(values),
                )
            resource = cast(
                OrganizationResource,
                self._get_resource_locked(conn, domain, object_id),
            )
            if resource.deleted or resource.name != name:
                raise ConflictError("Organization resource postcondition is absent")
            if after is not None:
                after(conn, object_id)
            return resource

    def _resource_row_for_relationship(
        self,
        conn: Any,
        domain: SyncDomain,
        sync_id: str,
        *,
        require_active: bool,
        for_update: bool = False,
    ) -> Mapping[str, object]:
        table, _ = self._table(domain)
        query = f"SELECT * FROM {table} WHERE sync_id = ? AND client_id = ?"  # nosec B608
        if for_update and self._db.backend_type == BackendType.POSTGRESQL:
            query += " FOR UPDATE"
        row = conn.execute(
            query,
            (sync_id, self._owner_id),
        ).fetchone()
        if not row or (require_active and bool(row["deleted"])):
            raise InputError(f"Referenced {domain} resource is missing or deleted")
        return row

    def _insert_link(self, conn: Any, table: str, columns: tuple[str, str], values: tuple[object, object]) -> None:
        now = self._db._get_current_utc_timestamp_iso()
        if self._db.backend_type == BackendType.POSTGRESQL:
            sql = (
                f"INSERT INTO {table} ({columns[0]}, {columns[1]}, created_at) "  # nosec B608
                "VALUES (?, ?, ?) ON CONFLICT DO NOTHING"
            )
        else:
            sql = (
                f"INSERT OR IGNORE INTO {table} ({columns[0]}, {columns[1]}, created_at) "  # nosec B608
                "VALUES (?, ?, ?)"
            )
        conn.execute(sql, (*values, now))

    @staticmethod
    def _source_provenance_values(
        provenance: Mapping[str, object],
    ) -> tuple[str, int, str | None, str | None]:
        if set(provenance) not in (
            {"operation", "source_id"},
            {"operation", "source_id", "read_set_hash"},
            {"operation", "source_id", "pre_state_hash", "post_state_hash"},
        ):
            raise InputError("Folder source provenance fields are invalid")
        operation = provenance.get("operation")
        source_id = provenance.get("source_id")
        if operation not in {"source_upsert", "source_delete"}:
            raise InputError("Folder source provenance operation is invalid")
        if isinstance(source_id, bool) or not isinstance(source_id, int) or source_id <= 0:
            raise InputError("Folder source provenance identifier is invalid")
        pre_state_hash = provenance.get("pre_state_hash")
        post_state_hash = provenance.get("post_state_hash")
        legacy_read_set_hash = provenance.get("read_set_hash")
        for state_hash in (pre_state_hash, post_state_hash, legacy_read_set_hash):
            if state_hash is not None and (
                not isinstance(state_hash, str)
                or len(state_hash) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in state_hash
                )
            ):
                raise InputError("Folder source provenance state is invalid")
        if legacy_read_set_hash is not None:
            pre_state_hash = legacy_read_set_hash
        return str(operation), source_id, pre_state_hash, post_state_hash

    @staticmethod
    def _source_folder_state_hash(
        *,
        note_id: str,
        folder_sync_id: str,
        source_id: int,
        operation: str,
        transition_identity: str,
        manual: bool,
        source_ids: set[int],
        suppressed: bool,
    ) -> str:
        encoded = json.dumps(
            {
                "folder_sync_id": folder_sync_id,
                "manual": manual,
                "note_id": note_id,
                "operation": operation,
                "source_id": source_id,
                "source_ids": sorted(source_ids),
                "suppressed": suppressed,
                "transition_identity": transition_identity,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _source_folder_read_set_locked(
        self,
        conn: Any,
        *,
        note_id: str,
        folder_id: int,
    ) -> tuple[bool, set[int], bool]:
        manual = bool(
            conn.execute(
                "SELECT 1 FROM note_folder_memberships membership "
                "JOIN notes note ON note.id = membership.note_id "
                "JOIN note_folders folder ON folder.id = membership.folder_id "
                "WHERE membership.note_id = ? AND membership.folder_id = ? "
                "AND note.client_id = ? AND folder.client_id = ?",
                (note_id, folder_id, self._owner_id, self._owner_id),
            ).fetchone()
        )
        source_rows = conn.execute(
            "SELECT membership.source_id FROM note_folder_source_memberships membership "
            "JOIN notes note ON note.id = membership.note_id "
            "JOIN note_folders folder ON folder.id = membership.folder_id "
            "WHERE membership.note_id = ? AND membership.folder_id = ? "
            "AND note.client_id = ? AND folder.client_id = ?",
            (note_id, folder_id, self._owner_id, self._owner_id),
        ).fetchall()
        suppressed = bool(
            conn.execute(
                "SELECT 1 FROM note_folder_sync_suppressions suppression "
                "JOIN notes note ON note.id = suppression.note_id "
                "JOIN note_folders folder ON folder.id = suppression.folder_id "
                "WHERE suppression.note_id = ? AND suppression.folder_id = ? "
                "AND note.client_id = ? AND folder.client_id = ?",
                (note_id, folder_id, self._owner_id, self._owner_id),
            ).fetchone()
        )
        return manual, {int(row["source_id"]) for row in source_rows}, suppressed

    def _source_folder_state_hash_locked(
        self,
        conn: Any,
        *,
        note_id: str,
        folder_id: int,
        folder_sync_id: str,
        source_id: int,
        operation: str,
        transition_identity: str,
    ) -> str:
        manual, source_ids, suppressed = self._source_folder_read_set_locked(
            conn,
            note_id=note_id,
            folder_id=folder_id,
        )
        return self._source_folder_state_hash(
            note_id=note_id,
            folder_sync_id=folder_sync_id,
            source_id=source_id,
            operation=operation,
            transition_identity=transition_identity,
            manual=manual,
            source_ids=source_ids,
            suppressed=suppressed,
        )

    def _source_folder_transition_guard_locked(
        self,
        conn: Any,
        *,
        note_id: str,
        folder_id: int,
        folder_sync_id: str,
        source_id: int,
        operation: str,
        transition_identity: str,
        pre_state_hash: str,
        post_state_hash: str,
    ) -> bool:
        current_hash = self._source_folder_state_hash_locked(
            conn,
            note_id=note_id,
            folder_id=folder_id,
            folder_sync_id=folder_sync_id,
            source_id=source_id,
            operation=operation,
            transition_identity=transition_identity,
        )
        if current_hash == post_state_hash:
            return False
        if current_hash != pre_state_hash:
            raise ConflictError("Folder source membership changed after planning")
        return True

    def _folder_relationship_rows(
        self,
        conn: Any,
        *,
        note_id: str,
        folder_sync_id: str,
        require_active: bool,
    ) -> tuple[Mapping[str, object], Mapping[str, object]]:
        folder = self._resource_row_for_relationship(
            conn,
            "notes.folder",
            folder_sync_id,
            require_active=require_active,
        )
        note = conn.execute(
            "SELECT id, deleted FROM notes WHERE id = ? AND client_id = ?",
            (note_id, self._owner_id),
        ).fetchone()
        if not note or (require_active and bool(note["deleted"])):
            raise InputError("Referenced folder-link note is missing or deleted")
        return note, folder

    def source_folder_transition(
        self,
        *,
        note_id: str,
        source_id: int,
        folder_sync_id: str,
        present: bool,
    ) -> SyncOperation | None:
        """Return the canonical transition for one prospective source delta."""

        return self.source_folder_transition_plan(
            note_id=note_id,
            source_id=source_id,
            folder_sync_id=folder_sync_id,
            present=present,
            transition_identity=(
                f"source-folder-transition:{note_id}:{folder_sync_id}:"
                f"{source_id}:{int(present)}"
            ),
        ).operation

    def source_folder_transition_plan(
        self,
        *,
        note_id: str,
        source_id: int,
        folder_sync_id: str,
        present: bool,
        transition_identity: str,
    ) -> SourceFolderTransitionPlan:
        """Return one source transition with opaque exact pre/post states."""

        normalized_identity = str(transition_identity).strip()
        if not normalized_identity:
            raise InputError("Folder source transition identity is invalid")
        operation = "source_upsert" if present else "source_delete"
        _, validated_source_id, _, _ = self._source_provenance_values(
            {
                "operation": operation,
                "source_id": source_id,
            }
        )
        with self._db.transaction() as conn:
            _, folder = self._folder_relationship_rows(
                conn,
                note_id=note_id,
                folder_sync_id=folder_sync_id,
                require_active=True,
            )
            folder_id = int(folder["id"])
            manual, source_ids, suppressed = self._source_folder_read_set_locked(
                conn,
                note_id=note_id,
                folder_id=folder_id,
            )
            pre_state_hash = self._source_folder_state_hash(
                note_id=note_id,
                folder_sync_id=folder_sync_id,
                source_id=validated_source_id,
                operation=operation,
                transition_identity=normalized_identity,
                manual=manual,
                source_ids=source_ids,
                suppressed=suppressed,
            )
            before = not suppressed and bool(manual or source_ids)
            post_source_ids = set(source_ids)
            if present:
                post_source_ids.add(validated_source_id)
            else:
                post_source_ids.discard(validated_source_id)
            after = not suppressed and bool(manual or post_source_ids)
            transition: SyncOperation | None = None
            if before != after:
                transition = "upsert" if after else "tombstone"
            post_manual = manual
            post_suppressed = suppressed
            if transition == "upsert":
                post_suppressed = False
            elif transition == "tombstone":
                post_manual = False
                post_suppressed = True
            post_state_hash = self._source_folder_state_hash(
                note_id=note_id,
                folder_sync_id=folder_sync_id,
                source_id=validated_source_id,
                operation=operation,
                transition_identity=normalized_identity,
                manual=post_manual,
                source_ids=post_source_ids,
                suppressed=post_suppressed,
            )
            return SourceFolderTransitionPlan(
                operation=transition,
                pre_state_hash=pre_state_hash,
                post_state_hash=post_state_hash,
                transition_identity=normalized_identity,
            )

    def manual_folder_sync_ids(self, note_id: str) -> set[str]:
        """Return active folders explicitly owned by the note's manual membership set."""

        with self._db.transaction() as conn:
            rows = conn.execute(
                "SELECT folder.sync_id FROM note_folder_memberships membership "
                "JOIN note_folders folder ON folder.id = membership.folder_id "
                "JOIN notes note ON note.id = membership.note_id "
                "WHERE membership.note_id = ? AND folder.deleted = ? "
                "AND note.client_id = ? AND folder.client_id = ?",
                (
                    note_id,
                    self._deleted_value(False),
                    self._owner_id,
                    self._owner_id,
                ),
            ).fetchall()
        return {str(row["sync_id"]) for row in rows}

    def _apply_source_folder_provenance_locked(
        self,
        conn: Any,
        *,
        note_id: str,
        folder: Mapping[str, object],
        operation: str,
        source_id: int,
    ) -> None:
        folder_id = int(folder["id"])
        if operation == "source_delete":
            conn.execute(
                "DELETE FROM note_folder_source_memberships "
                "WHERE note_id = ? AND source_id = ? AND folder_id = ? "
                "AND EXISTS (SELECT 1 FROM notes owner_note "
                "WHERE owner_note.id = note_folder_source_memberships.note_id "
                "AND owner_note.client_id = ?) "
                "AND EXISTS (SELECT 1 FROM note_folders owner_folder "
                "WHERE owner_folder.id = note_folder_source_memberships.folder_id "
                "AND owner_folder.client_id = ?)",
                (note_id, source_id, folder_id, self._owner_id, self._owner_id),
            )
            remaining = conn.execute(
                "SELECT 1 FROM note_folder_source_memberships membership "
                "JOIN notes note ON note.id = membership.note_id "
                "JOIN note_folders folder ON folder.id = membership.folder_id "
                "WHERE membership.source_id = ? AND membership.folder_id = ? "
                "AND note.client_id = ? AND folder.client_id = ? LIMIT 1",
                (source_id, folder_id, self._owner_id, self._owner_id),
            ).fetchone()
            if not remaining:
                conn.execute(
                    "DELETE FROM note_folder_source_keys "
                    "WHERE source_id = ? AND folder_id = ? "
                    "AND EXISTS (SELECT 1 FROM note_folders owner_folder "
                    "WHERE owner_folder.id = note_folder_source_keys.folder_id "
                    "AND owner_folder.client_id = ?)",
                    (source_id, folder_id, self._owner_id),
                )
            return

        folder_key = self._db._note_folder_path_key(str(folder["path"]))
        if not folder_key:
            raise InputError("Folder source provenance path is invalid")
        now = self._db._get_current_utc_timestamp_iso()
        conn.execute(
            "DELETE FROM note_folder_source_keys "
            "WHERE source_id = ? AND (folder_key = ? OR folder_id = ?) "
            "AND EXISTS (SELECT 1 FROM note_folders owner_folder "
            "WHERE owner_folder.id = note_folder_source_keys.folder_id "
            "AND owner_folder.client_id = ?)",
            (source_id, folder_key, folder_id, self._owner_id),
        )
        conn.execute(
            "INSERT INTO note_folder_source_keys(source_id, folder_key, folder_id, created_at) "
            "VALUES (?, ?, ?, ?)",
            (source_id, folder_key, folder_id, now),
        )
        if self._db.backend_type == BackendType.POSTGRESQL:
            sql = (
                "INSERT INTO note_folder_source_memberships"
                "(note_id, source_id, folder_id, created_at) VALUES (?, ?, ?, ?) "
                "ON CONFLICT DO NOTHING"
            )
        else:
            sql = (
                "INSERT OR IGNORE INTO note_folder_source_memberships"
                "(note_id, source_id, folder_id, created_at) VALUES (?, ?, ?, ?)"
            )
        conn.execute(sql, (note_id, source_id, folder_id, now))

    def apply_source_folder_provenance(
        self,
        *,
        note_id: str,
        folder_sync_id: str,
        operation: str,
        source_id: int,
        pre_state_hash: str | None = None,
        post_state_hash: str | None = None,
        transition_identity: str | None = None,
    ) -> bool:
        """Apply provenance-only bookkeeping without changing canonical visibility."""

        provenance: dict[str, object] = {
            "operation": operation,
            "source_id": source_id,
        }
        if pre_state_hash is not None or post_state_hash is not None:
            provenance.update(
                {
                    "pre_state_hash": pre_state_hash,
                    "post_state_hash": post_state_hash,
                }
            )
        normalized_operation, normalized_source_id, normalized_pre, normalized_post = (
            self._source_provenance_values(provenance)
        )
        with self._db.transaction() as conn:
            _, folder = self._folder_relationship_rows(
                conn,
                note_id=note_id,
                folder_sync_id=folder_sync_id,
                require_active=True,
            )
            if normalized_pre is not None and normalized_post is not None:
                normalized_identity = str(transition_identity or "").strip()
                if not normalized_identity:
                    raise InputError("Folder source transition identity is invalid")
                should_apply = self._source_folder_transition_guard_locked(
                    conn,
                    note_id=note_id,
                    folder_id=int(folder["id"]),
                    folder_sync_id=folder_sync_id,
                    source_id=normalized_source_id,
                    operation=normalized_operation,
                    transition_identity=normalized_identity,
                    pre_state_hash=normalized_pre,
                    post_state_hash=normalized_post,
                )
                if not should_apply:
                    return False
            self._apply_source_folder_provenance_locked(
                conn,
                note_id=note_id,
                folder=folder,
                operation=normalized_operation,
                source_id=normalized_source_id,
            )
            if normalized_pre is not None and normalized_post is not None:
                if self._source_folder_state_hash_locked(
                    conn,
                    note_id=note_id,
                    folder_id=int(folder["id"]),
                    folder_sync_id=folder_sync_id,
                    source_id=normalized_source_id,
                    operation=normalized_operation,
                    transition_identity=str(transition_identity),
                ) != normalized_post:
                    raise ConflictError("Folder source membership changed during apply")
            return True

    def apply_relationship(
        self,
        *,
        domain: SyncDomain,
        object_id: str,
        operation: SyncOperation,
        payload: Mapping[str, object],
        routing_metadata: Mapping[str, object],
        origin_provenance: Mapping[str, object] | None = None,
        source_transition_identity: str | None = None,
        before: Callable[[Any], None] | None = None,
        after: Callable[[Any, str], None] | None = None,
    ) -> bool:
        """Apply one relationship envelope in a ChaCha transaction."""

        if (before is not None or after is not None) and (
            domain != "notes.keyword_link" or operation != "upsert"
        ):
            raise InputError("Guarded organization relationship must be a keyword-link upsert")
        del routing_metadata
        normalized = self._validated_payload(domain, operation, object_id, payload)
        require_active = operation == "upsert"
        with self._db.transaction() as conn:
            guarded = before is not None or after is not None
            link_owner_sql = ""
            link_owner_params: tuple[object, ...] = ()
            if domain == "notes.keyword_link":
                keyword = self._resource_row_for_relationship(
                    conn,
                    "notes.keyword",
                    str(normalized["keyword_sync_id"]),
                    require_active=require_active,
                    for_update=guarded,
                )
                subject_type = str(normalized["subject_type"])
                subject_id = str(normalized["subject_id"])
                subject_table = "notes" if subject_type == "note" else "conversations"
                subject_query = (
                    f"SELECT id, deleted FROM {subject_table} "  # nosec B608
                    "WHERE id = ? AND client_id = ?"
                )
                if guarded and self._db.backend_type == BackendType.POSTGRESQL:
                    subject_query += " FOR UPDATE"
                subject = conn.execute(
                    subject_query,
                    (subject_id, self._owner_id),
                ).fetchone()
                if not subject or (require_active and bool(subject["deleted"])):
                    raise InputError("Referenced keyword-link subject is missing or deleted")
                link_table = "note_keywords" if subject_type == "note" else "conversation_keywords"
                subject_column = "note_id" if subject_type == "note" else "conversation_id"
                values = (subject_id, int(keyword["id"]))
                columns = (subject_column, "keyword_id")
                keyword_table, _ = self._table("notes.keyword")
                link_owner_sql = (
                    f" AND EXISTS (SELECT 1 FROM {subject_table} owner_subject "  # nosec B608
                    f"WHERE owner_subject.id = {link_table}.{subject_column} "
                    "AND owner_subject.client_id = ?)"
                    f" AND EXISTS (SELECT 1 FROM {keyword_table} owner_keyword "
                    f"WHERE owner_keyword.id = {link_table}.keyword_id "
                    "AND owner_keyword.client_id = ?)"
                )
                link_owner_params = (self._owner_id, self._owner_id)
            elif domain == "notes.keyword_collection_link":
                collection = self._resource_row_for_relationship(
                    conn,
                    "notes.keyword_collection",
                    str(normalized["collection_sync_id"]),
                    require_active=require_active,
                    for_update=guarded,
                )
                keyword = self._resource_row_for_relationship(
                    conn,
                    "notes.keyword",
                    str(normalized["keyword_sync_id"]),
                    require_active=require_active,
                    for_update=guarded,
                )
                link_table = "collection_keywords"
                columns = ("collection_id", "keyword_id")
                values = (int(collection["id"]), int(keyword["id"]))
                collection_table, _ = self._table("notes.keyword_collection")
                keyword_table, _ = self._table("notes.keyword")
                link_owner_sql = (
                    f" AND EXISTS (SELECT 1 FROM {collection_table} owner_collection "  # nosec B608
                    f"WHERE owner_collection.id = {link_table}.collection_id "
                    "AND owner_collection.client_id = ?)"
                    f" AND EXISTS (SELECT 1 FROM {keyword_table} owner_keyword "
                    f"WHERE owner_keyword.id = {link_table}.keyword_id "
                    "AND owner_keyword.client_id = ?)"
                )
                link_owner_params = (self._owner_id, self._owner_id)
            elif domain == "notes.folder_link":
                note_id = str(normalized["note_id"])
                _, folder = self._folder_relationship_rows(
                    conn,
                    note_id=note_id,
                    folder_sync_id=str(normalized["folder_sync_id"]),
                    require_active=require_active,
                )
                link_table = "note_folder_memberships"
                columns = ("note_id", "folder_id")
                values = (note_id, int(folder["id"]))
            else:
                raise InputError(f"Unsupported organization relationship domain: {domain}")

            if before is not None:
                if self._db.backend_type == BackendType.POSTGRESQL:
                    conn.execute(
                        f"SELECT 1 FROM {link_table} WHERE {columns[0]} = ? "  # nosec B608
                        f"AND {columns[1]} = ?{link_owner_sql} FOR UPDATE",
                        (*values, *link_owner_params),
                    ).fetchone()
                before(conn)

            if domain == "notes.folder_link":
                guarded_post_state: tuple[str, int, str, str, str] | None = None
                if origin_provenance is not None:
                    (
                        provenance_operation,
                        provenance_source_id,
                        pre_state_hash,
                        post_state_hash,
                    ) = (
                        self._source_provenance_values(origin_provenance)
                    )
                    expected_operation = (
                        "source_upsert" if operation == "upsert" else "source_delete"
                    )
                    if provenance_operation != expected_operation:
                        raise InputError(
                            "Folder source provenance does not match canonical operation"
                        )
                    if pre_state_hash is not None and post_state_hash is not None:
                        normalized_identity = str(source_transition_identity or "").strip()
                        if not normalized_identity:
                            raise InputError("Folder source transition identity is invalid")
                        should_apply = self._source_folder_transition_guard_locked(
                            conn,
                            note_id=str(values[0]),
                            folder_id=int(folder["id"]),
                            folder_sync_id=str(normalized["folder_sync_id"]),
                            source_id=provenance_source_id,
                            operation=provenance_operation,
                            transition_identity=normalized_identity,
                            pre_state_hash=pre_state_hash,
                            post_state_hash=post_state_hash,
                        )
                        if not should_apply:
                            return False
                        guarded_post_state = (
                            str(normalized["folder_sync_id"]),
                            provenance_source_id,
                            provenance_operation,
                            normalized_identity,
                            post_state_hash,
                        )
                    elif pre_state_hash is not None:
                        manual, source_ids, suppressed = self._source_folder_read_set_locked(
                            conn,
                            note_id=str(values[0]),
                            folder_id=int(folder["id"]),
                        )
                        legacy_hash = hashlib.sha256(
                            json.dumps(
                                {
                                    "manual": manual,
                                    "source_ids": sorted(source_ids),
                                    "suppressed": suppressed,
                                },
                                sort_keys=True,
                                separators=(",", ":"),
                            ).encode("utf-8")
                        ).hexdigest()
                        if legacy_hash != pre_state_hash:
                            raise ConflictError("Folder source membership changed after planning")
                    self._apply_source_folder_provenance_locked(
                        conn,
                        note_id=str(values[0]),
                        folder=folder,
                        operation=provenance_operation,
                        source_id=provenance_source_id,
                    )
                if operation == "upsert":
                    conn.execute(
                        "DELETE FROM note_folder_sync_suppressions "
                        "WHERE note_id = ? AND folder_id = ? "
                        "AND EXISTS (SELECT 1 FROM notes owner_note "
                        "WHERE owner_note.id = note_folder_sync_suppressions.note_id "
                        "AND owner_note.client_id = ?) "
                        "AND EXISTS (SELECT 1 FROM note_folders owner_folder "
                        "WHERE owner_folder.id = note_folder_sync_suppressions.folder_id "
                        "AND owner_folder.client_id = ?)",
                        (*values, self._owner_id, self._owner_id),
                    )
                    if origin_provenance is None:
                        self._insert_link(conn, link_table, columns, values)
                else:
                    conn.execute(
                        "DELETE FROM note_folder_memberships "
                        "WHERE note_id = ? AND folder_id = ? "
                        "AND EXISTS (SELECT 1 FROM notes owner_note "
                        "WHERE owner_note.id = note_folder_memberships.note_id "
                        "AND owner_note.client_id = ?) "
                        "AND EXISTS (SELECT 1 FROM note_folders owner_folder "
                        "WHERE owner_folder.id = note_folder_memberships.folder_id "
                        "AND owner_folder.client_id = ?)",
                        (*values, self._owner_id, self._owner_id),
                    )
                    self._insert_link(
                        conn,
                        "note_folder_sync_suppressions",
                        ("note_id", "folder_id"),
                        values,
                    )
                if guarded_post_state is not None:
                    (
                        guarded_folder_sync_id,
                        guarded_source_id,
                        guarded_operation,
                        guarded_identity,
                        guarded_post_hash,
                    ) = guarded_post_state
                    if self._source_folder_state_hash_locked(
                        conn,
                        note_id=str(values[0]),
                        folder_id=int(folder["id"]),
                        folder_sync_id=guarded_folder_sync_id,
                        source_id=guarded_source_id,
                        operation=guarded_operation,
                        transition_identity=guarded_identity,
                    ) != guarded_post_hash:
                        raise ConflictError("Folder source membership changed during apply")
            elif operation == "upsert":
                self._insert_link(conn, link_table, columns, values)
            else:
                conn.execute(
                    f"DELETE FROM {link_table} WHERE {columns[0]} = ? "  # nosec B608
                    f"AND {columns[1]} = ?{link_owner_sql}",
                    (*values, *link_owner_params),
                )
            if after is not None:
                present = conn.execute(
                    f"SELECT 1 FROM {link_table} WHERE {columns[0]} = ? "  # nosec B608
                    f"AND {columns[1]} = ?{link_owner_sql} LIMIT 1",
                    (*values, *link_owner_params),
                ).fetchone()
                if present is None:
                    raise ConflictError("Organization relationship postcondition is absent")
                after(conn, object_id)
            return True


__all__ = [
    "NotesOrganizationSyncStore",
    "OrganizationRelationship",
    "OrganizationResource",
    "OrganizationSnapshot",
    "SourceFolderTransitionPlan",
]
