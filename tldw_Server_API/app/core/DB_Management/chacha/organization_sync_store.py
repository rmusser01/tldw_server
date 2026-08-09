from __future__ import annotations

"""Backend-neutral ChaCha projection seam for Notes organization Sync domains."""

from collections import defaultdict, deque
from collections.abc import Mapping
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


_RESOURCE_TABLES: dict[SyncDomain, tuple[str, str]] = {
    "notes.keyword": ("keywords", "keyword"),
    "notes.keyword_collection": ("keyword_collections", "name"),
    "notes.folder": ("note_folders", "name"),
}


class NotesOrganizationSyncStore:
    """Project canonical Notes organization resources into one user's ChaCha DB."""

    def __init__(self, db: CharactersRAGDB) -> None:
        self._db = db

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
                f"SELECT sync_id FROM {table} WHERE id = ?",  # nosec B608
                (int(parent_id),),
            ).fetchone()
            if parent:
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
    ) -> OrganizationResource | None:
        table, _ = self._table(domain)
        row = conn.execute(
            f"SELECT * FROM {table} WHERE sync_id = ?",  # nosec B608
            (sync_id,),
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

    def snapshot(self) -> OrganizationSnapshot:
        """Return a transactionally consistent organization snapshot."""

        resources: list[OrganizationResource] = []
        relationships: dict[tuple[str, str], OrganizationRelationship] = {}
        with self._db.transaction() as conn:
            for domain in cast(tuple[SyncDomain, ...], tuple(_RESOURCE_TABLES)):
                table, _ = self._table(domain)
                rows = conn.execute(f"SELECT * FROM {table} ORDER BY id").fetchall()  # nosec B608
                resources.extend(self._resource_from_row(conn, domain, row) for row in rows)

            keyword_table = self._db._map_table_for_backend("keywords")
            for subject_type, link_table, subject_column in (
                ("note", "note_keywords", "note_id"),
                ("conversation", "conversation_keywords", "conversation_id"),
            ):
                rows = conn.execute(
                    f"SELECT l.{subject_column} AS subject_id, k.sync_id AS keyword_sync_id "  # nosec B608
                    f"FROM {link_table} l JOIN {keyword_table} k ON k.id = l.keyword_id"
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
                f"JOIN {keyword_table} k ON k.id = l.keyword_id"  # nosec B608
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
                "WHERE NOT EXISTS ("
                "SELECT 1 FROM note_folder_sync_suppressions suppression "
                "WHERE suppression.note_id = memberships.note_id "
                "AND suppression.folder_id = memberships.folder_id)"
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
            f"SELECT id, parent_id, deleted FROM {table} WHERE sync_id = ?",  # nosec B608
            (str(parent_sync_id),),
        ).fetchone()
        if not parent or bool(parent["deleted"]):
            raise InputError("Organization parent is missing or deleted")
        parent_id = int(parent["id"])
        current = conn.execute(
            f"SELECT id FROM {table} WHERE sync_id = ?",  # nosec B608
            (object_id,),
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
                f"SELECT parent_id FROM {table} WHERE id = ?",  # nosec B608
                (cursor_id,),
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
                "SELECT path FROM note_folders WHERE id = ?", (parent_id,)
            ).fetchone()
            parent_path = str(parent["path"])
        root_path = f"{parent_path}/{name}" if parent_path else name
        if len(root_path) > 500:
            raise InputError("Folder path cannot exceed 500 characters")

        now = self._db._get_current_utc_timestamp_iso()
        if existing is None:
            duplicate = conn.execute(
                "SELECT id FROM note_folders WHERE LOWER(path) = LOWER(?) LIMIT 1",
                (root_path,),
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

        rows = [dict(row) for row in conn.execute("SELECT * FROM note_folders").fetchall()]
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
                "UPDATE note_folders SET path = ? WHERE id = ?",
                (f"__sync_repath__/{by_id[folder_id]['sync_id']}", folder_id),
            )
        conn.execute(
            "UPDATE note_folders SET name = ?, path = ?, parent_id = ?, deleted = ?, "
            "last_modified = ?, client_id = ?, version = ? WHERE id = ?",
            (
                name,
                paths[existing.local_id],
                parent_id,
                self._deleted_value(False),
                now,
                self._db.client_id,
                existing.version + 1,
                existing.local_id,
            ),
        )
        for folder_id in subtree[1:]:
            conn.execute(
                "UPDATE note_folders SET path = ? WHERE id = ?",
                (paths[folder_id], folder_id),
            )
        return cast(OrganizationResource, self._get_resource_locked(conn, "notes.folder", object_id))

    def apply_resource(
        self,
        *,
        domain: SyncDomain,
        object_id: str,
        operation: SyncOperation,
        payload: Mapping[str, object],
    ) -> OrganizationResource:
        """Apply one resource envelope in a ChaCha transaction."""

        normalized = self._validated_payload(domain, operation, object_id, payload)
        table, name_column = self._table(domain)
        with self._db.transaction() as conn:
            existing = self._get_resource_locked(conn, domain, object_id)
            if operation == "tombstone":
                if existing is None:
                    raise InputError("Cannot tombstone an unknown organization resource")
                if existing.deleted:
                    return existing
                conn.execute(
                    f"UPDATE {table} SET deleted = ?, last_modified = ?, client_id = ?, version = ? "  # nosec B608
                    "WHERE sync_id = ?",
                    (
                        self._deleted_value(True),
                        self._db._get_current_utc_timestamp_iso(),
                        self._db.client_id,
                        existing.version + 1,
                        object_id,
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
                "AND sync_id <> ? LIMIT 1",
                (name, object_id),
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
                    ]
                )
                conn.execute(
                    f"UPDATE {table} SET {', '.join(set_parts)} WHERE sync_id = ?",  # nosec B608
                    tuple(values),
                )
            return cast(OrganizationResource, self._get_resource_locked(conn, domain, object_id))

    def _resource_row_for_relationship(
        self,
        conn: Any,
        domain: SyncDomain,
        sync_id: str,
        *,
        require_active: bool,
    ) -> Mapping[str, object]:
        table, _ = self._table(domain)
        row = conn.execute(
            f"SELECT * FROM {table} WHERE sync_id = ?",  # nosec B608
            (sync_id,),
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

    def apply_relationship(
        self,
        *,
        domain: SyncDomain,
        object_id: str,
        operation: SyncOperation,
        payload: Mapping[str, object],
        routing_metadata: Mapping[str, object],
    ) -> None:
        """Apply one relationship envelope in a ChaCha transaction."""

        del routing_metadata  # Reserved for Tasks 6 and 9.
        normalized = self._validated_payload(domain, operation, object_id, payload)
        require_active = operation == "upsert"
        with self._db.transaction() as conn:
            if domain == "notes.keyword_link":
                keyword = self._resource_row_for_relationship(
                    conn,
                    "notes.keyword",
                    str(normalized["keyword_sync_id"]),
                    require_active=require_active,
                )
                subject_type = str(normalized["subject_type"])
                subject_id = str(normalized["subject_id"])
                subject_table = "notes" if subject_type == "note" else "conversations"
                subject = conn.execute(
                    f"SELECT id, deleted FROM {subject_table} WHERE id = ?",  # nosec B608
                    (subject_id,),
                ).fetchone()
                if not subject or (require_active and bool(subject["deleted"])):
                    raise InputError("Referenced keyword-link subject is missing or deleted")
                link_table = "note_keywords" if subject_type == "note" else "conversation_keywords"
                subject_column = "note_id" if subject_type == "note" else "conversation_id"
                values = (subject_id, int(keyword["id"]))
                columns = (subject_column, "keyword_id")
            elif domain == "notes.keyword_collection_link":
                collection = self._resource_row_for_relationship(
                    conn,
                    "notes.keyword_collection",
                    str(normalized["collection_sync_id"]),
                    require_active=require_active,
                )
                keyword = self._resource_row_for_relationship(
                    conn,
                    "notes.keyword",
                    str(normalized["keyword_sync_id"]),
                    require_active=require_active,
                )
                link_table = "collection_keywords"
                columns = ("collection_id", "keyword_id")
                values = (int(collection["id"]), int(keyword["id"]))
            elif domain == "notes.folder_link":
                folder = self._resource_row_for_relationship(
                    conn,
                    "notes.folder",
                    str(normalized["folder_sync_id"]),
                    require_active=require_active,
                )
                note_id = str(normalized["note_id"])
                note = conn.execute(
                    "SELECT id, deleted FROM notes WHERE id = ?", (note_id,)
                ).fetchone()
                if not note or (require_active and bool(note["deleted"])):
                    raise InputError("Referenced folder-link note is missing or deleted")
                link_table = "note_folder_memberships"
                columns = ("note_id", "folder_id")
                values = (note_id, int(folder["id"]))
            else:
                raise InputError(f"Unsupported organization relationship domain: {domain}")

            if domain == "notes.folder_link":
                if operation == "upsert":
                    conn.execute(
                        "DELETE FROM note_folder_sync_suppressions "
                        "WHERE note_id = ? AND folder_id = ?",
                        values,
                    )
                    self._insert_link(conn, link_table, columns, values)
                else:
                    conn.execute(
                        "DELETE FROM note_folder_memberships "
                        "WHERE note_id = ? AND folder_id = ?",
                        values,
                    )
                    self._insert_link(
                        conn,
                        "note_folder_sync_suppressions",
                        ("note_id", "folder_id"),
                        values,
                    )
            elif operation == "upsert":
                self._insert_link(conn, link_table, columns, values)
            else:
                conn.execute(
                    f"DELETE FROM {link_table} WHERE {columns[0]} = ? AND {columns[1]} = ?",  # nosec B608
                    values,
                )


__all__ = [
    "NotesOrganizationSyncStore",
    "OrganizationRelationship",
    "OrganizationResource",
    "OrganizationSnapshot",
]
