"""Owner-run migrations must leave PostgreSQL usable by a DML-only runtime."""

import configparser
import uuid
from dataclasses import replace
from types import SimpleNamespace

import pytest
from psycopg import sql

from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, CharactersRAGDBError
from tldw_Server_API.app.core.DB_Management.media_db.api import get_media_by_id, search_media
from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase
from tldw_Server_API.app.core.DB_Management.media_db.runtime import factory
from tldw_Server_API.app.core.DB_Management.scope_context import scoped_context

pytestmark = pytest.mark.integration


@pytest.fixture
def migrated_runtime(pg_restricted_backend, pg_database_config, monkeypatch, request):
    """Use the official isolated database and a separate, unprivileged runtime."""
    monkeypatch.delenv("TLDW_CONTENT_PG_ROLE_SWITCH", raising=False)
    role = "tldw_upgrade_runtime_" + uuid.uuid4().hex
    password = uuid.uuid4().hex
    admin = DatabaseBackendFactory.create_backend(pg_database_config)
    owner = pg_restricted_backend
    runtime = media = chacha = None
    with_notes = getattr(request, "param", True)
    note_id = None
    role_created = False
    with scoped_context(user_id=101, org_ids=[], team_ids=[], is_admin=False):
        try:
            media = MediaDatabase(":memory:", client_id="101", backend=owner)
            if with_notes:
                chacha = CharactersRAGDB(":memory:", client_id="101", backend=owner)
            source_id, _, _ = media.add_media_with_keywords(
                title="X01 Rowan Observatory",
                media_type="document",
                content="Migrated Rowan Observatory source",
                visibility="personal",
                owner_user_id=101,
            )
            if chacha is not None:
                note_id = chacha.add_note("X01 migrated note", "Migrated note content")
            media.close_connection()
            if chacha is not None:
                chacha.close_all_connections()
            with admin.transaction() as conn:
                conn.execute(
                    sql.SQL(
                        "CREATE ROLE {} LOGIN PASSWORD {} NOSUPERUSER NOBYPASSRLS NOINHERIT NOCREATEDB NOCREATEROLE"
                    ).format(sql.Identifier(role), sql.Literal(password))
                )
                conn.execute(sql.SQL("GRANT USAGE ON SCHEMA public TO {}").format(sql.Identifier(role)))
                conn.execute(
                    sql.SQL("GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO {}").format(
                        sql.Identifier(role)
                    )
                )
                conn.execute(
                    sql.SQL("GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO {}").format(sql.Identifier(role))
                )
                if with_notes:
                    conn.execute(
                        sql.SQL("REVOKE ALL ON chacha_schema_migration_progress FROM {}").format(sql.Identifier(role))
                    )
                    conn.execute(
                        sql.SQL("REVOKE INSERT, UPDATE, DELETE ON db_schema_version FROM {}").format(
                            sql.Identifier(role)
                        )
                    )
                conn.execute(
                    sql.SQL("REVOKE INSERT, UPDATE, DELETE ON schema_version FROM {}").format(sql.Identifier(role))
                )
            role_created = True
            runtime = DatabaseBackendFactory.create_backend(
                replace(
                    pg_database_config,
                    pg_user=role,
                    pg_password=password,
                    connection_string=None,
                    pool_size=2,
                    max_overflow=2,
                )
            )
            flags = runtime.execute(
                "SELECT rolsuper, rolbypassrls, rolcreatedb, rolcreaterole, "
                "has_schema_privilege(current_user, 'public', 'CREATE') AS can_create, "
                "pg_has_role(current_user, %s, 'MEMBER') AS owner_member, "
                "EXISTS (SELECT 1 FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace "
                "WHERE n.nspname='public' AND c.relowner=current_user::regrole) AS owns_objects "
                "FROM pg_roles WHERE rolname=current_user",
                (owner.config.pg_user,),
            ).first
            # Fixture must prove the actual connection has no migration authority.
            assert flags and not any(flags.values())  # nosec B101
            owner.get_pool().close_all()
            yield SimpleNamespace(
                backend=runtime, owner=owner, admin=admin, role=role, source_id=source_id, note_id=note_id
            )
        finally:
            if chacha is not None:
                chacha.close_all_connections()
            if media is not None:
                media.close_connection()
            if runtime is not None:
                runtime.get_pool().close_all()
            admin.get_pool().close_all()
            admin = DatabaseBackendFactory.create_backend(pg_database_config)
            try:
                if role_created:
                    with admin.transaction() as conn:
                        conn.execute(sql.SQL("DROP OWNED BY {}").format(sql.Identifier(role)))
                        conn.execute(sql.SQL("DROP ROLE IF EXISTS {}").format(sql.Identifier(role)))
            finally:
                admin.get_pool().close_all()


def test_migrated_media_supports_nonowner_runtime_reads_search_and_writes(migrated_runtime):
    """Routine Media construction must not replay owner-only schema changes."""
    f = migrated_runtime
    db = MediaDatabase(":memory:", client_id="101", backend=f.backend)
    try:
        # Existing content must remain readable under the actual DML-only role.
        assert get_media_by_id(db, f.source_id)["content"] == "Migrated Rowan Observatory source"  # nosec B101
        rows, total = search_media(db, "Rowan", search_fields=["title", "content"])
        # Search must retain the migrated FTS behavior, not just raw SELECT.
        assert total == 1 and rows[0]["id"] == f.source_id  # nosec B101
        created, _, _ = db.add_media_with_keywords(
            title="X01 post-upgrade source",
            media_type="document",
            content="New source after migration",
            visibility="personal",
            owner_user_id=101,
        )
        # Real writes must preserve both new and old records.
        assert created != f.source_id and get_media_by_id(db, created)["content"] == "New source after migration"  # nosec B101
    finally:
        db.close_connection()


def test_migrated_media_passes_actual_startup_validation_as_nonowner(migrated_runtime):
    """Startup must validate the migrated schema without requiring its owner."""
    backend = migrated_runtime.backend
    runtime = factory.MediaDbRuntimeConfig(
        default_db_path=":memory:",
        default_config=configparser.ConfigParser(),
        postgres_content_mode=True,
        backend_loader=lambda: backend,
    )
    factory.validate_postgres_content_backend(
        get_content_backend_instance=lambda: backend,
        runtime=runtime,
    )
    # The connected role remains unable to bypass RLS after validation.
    flags = backend.execute("SELECT rolsuper, rolbypassrls FROM pg_roles WHERE rolname=current_user").first
    assert flags and not any(flags.values())  # nosec B101


def test_migrated_chacha_supports_nonowner_runtime_as_connected_control(migrated_runtime):
    """The same role can use the existing catalog-verified ChaCha path."""
    f = migrated_runtime
    db = CharactersRAGDB(":memory:", client_id="101", backend=f.backend)
    try:
        # This control distinguishes a general grant/setup error from Media DDL.
        assert db.get_note_by_id(f.note_id)["content"] == "Migrated note content"  # nosec B101
        created = db.add_note("X01 new runtime note", "New note after migration")
        assert db.get_note_by_id(created)["content"] == "New note after migration"  # nosec B101
    finally:
        db.close_all_connections()


@pytest.mark.parametrize("table", ["schema_version", "db_schema_version", "chacha_schema_migration_progress"])
def test_runtime_cannot_publish_migration_completion(migrated_runtime, table):
    """A DML runtime cannot forge the version trusted by catalog validation."""
    privileges = migrated_runtime.backend.execute(
        "SELECT (has_table_privilege(current_user,%s,'INSERT,UPDATE,DELETE,TRUNCATE') OR "
        "has_any_column_privilege(current_user,%s,'INSERT,UPDATE')) AS can_write, "
        "has_table_privilege(current_user,%s,'SELECT') AS can_read",
        (table, table, table),
    ).first
    # The private journal stays unreadable; only version metadata is exposed.
    assert privileges == {"can_write": False, "can_read": table != "chacha_schema_migration_progress"}  # nosec B101


@pytest.mark.parametrize(
    "drift",
    [
        "ALTER TABLE note_attachments NO FORCE ROW LEVEL SECURITY",
        "ALTER TABLE note_tasks NO FORCE ROW LEVEL SECURITY",
        "CREATE POLICY notes_open ON notes USING (true) WITH CHECK (true)",
        "ALTER TABLE note_attachments OWNER TO {admin}",
        "GRANT SELECT ON chacha_schema_migration_progress TO {runtime}",
        "GRANT UPDATE(version) ON db_schema_version TO {runtime}",
    ],
    ids=["attachment-force", "task-force", "extra-policy", "registry-owner", "journal-acl", "version-column-grant"],
)
def test_nonowner_chacha_reopen_rejects_catalog_drift(migrated_runtime, drift):
    """Separating runtime authority must retain the existing catalog guards."""
    f = migrated_runtime
    with f.admin.transaction() as conn:
        statement = sql.SQL(drift).format(runtime=sql.Identifier(f.role), admin=sql.Identifier(f.admin.config.pg_user))
        conn.execute(statement)
    try:
        with pytest.raises(CharactersRAGDBError, match="catalog|RLS|ownership|drift|authority"):
            CharactersRAGDB(":memory:", client_id="101", backend=f.backend)
    finally:
        if "OWNER TO" in drift:
            with f.admin.transaction() as conn:
                conn.execute(
                    sql.SQL("ALTER TABLE note_attachments OWNER TO {}").format(sql.Identifier(f.owner.config.pg_user))
                )


@pytest.mark.parametrize(
    "drift",
    [
        "DROP TRIGGER update_media_fts_tsv_trigger ON media",
        "ALTER TABLE media DISABLE TRIGGER update_media_fts_tsv_trigger",
        "DROP INDEX idx_media_media_fts_tsv",
        "ALTER TABLE media DROP COLUMN source_hash CASCADE",
        "DROP TABLE audio_presets",
        "ALTER TABLE media NO FORCE ROW LEVEL SECURITY",
        "DROP POLICY media_visibility_access ON media",
        "CREATE POLICY media_open ON media USING (true) WITH CHECK (true)",
        "ALTER POLICY media_visibility_access ON media USING (true)",
        "ALTER POLICY media_visibility_access ON media WITH CHECK (true)",
        "GRANT UPDATE(version) ON schema_version TO {runtime}",
        "ALTER TABLE document_reading_progress DROP COLUMN cfi",
        "ALTER TABLE document_annotations DROP COLUMN client_id CASCADE",
        "DROP INDEX idx_annotations_client_id",
        (
            "DROP INDEX idx_annotations_client_id",
            "CREATE INDEX idx_annotations_client_id ON document_annotations(user_id,media_id,client_id)",
        ),
        (
            "DROP INDEX idx_annotations_client_id",
            "CREATE UNIQUE INDEX idx_annotations_client_id ON document_annotations(user_id,client_id)",
        ),
        (
            "DROP INDEX idx_annotations_client_id",
            "CREATE UNIQUE INDEX idx_annotations_client_id ON document_annotations(user_id,media_id,client_id) WHERE client_id <> 'excluded-intent'",
        ),
        "ALTER TABLE claims DROP COLUMN review_notes",
        "ALTER TABLE data_table_columns DROP COLUMN format",
        (
            "DROP TRIGGER update_media_fts_tsv_trigger ON media",
            "CREATE TRIGGER update_media_fts_tsv_trigger BEFORE INSERT OR UPDATE ON media "
            "FOR EACH ROW WHEN (false) EXECUTE FUNCTION update_media_fts_tsv_function()",
        ),
        (
            "DROP TRIGGER update_media_fts_tsv_trigger ON media",
            "CREATE TRIGGER update_media_fts_tsv_trigger BEFORE INSERT OR UPDATE OF title ON media "
            "FOR EACH ROW EXECUTE FUNCTION update_media_fts_tsv_function()",
        ),
        "CREATE OR REPLACE FUNCTION public.tldw_try_extract_normalized_transcript_text(input_text TEXT) "
        "RETURNS TEXT LANGUAGE plpgsql IMMUTABLE STRICT SET search_path=pg_catalog "
        "AS $function$ BEGIN RETURN 'wrong'; END; $function$",
        "UPDATE schema_version SET version=25",
    ],
    ids=[
        "missing-trigger",
        "disabled-trigger",
        "missing-index",
        "missing-column",
        "missing-addon",
        "no-force",
        "missing-policy",
        "extra-policy",
        "weakened-read",
        "weakened-write",
        "version-column-grant",
        "document-column",
        "annotation-client-column",
        "annotation-client-index",
        "annotation-nonunique-index",
        "annotation-wrong-key-index",
        "annotation-partial-index",
        "claims-column",
        "data-column",
        "trigger-when",
        "trigger-update-of",
        "helper-body",
        "owner-migration-needed",
    ],
)
def test_nonowner_media_reopen_refuses_drift_without_repair(migrated_runtime, drift):
    """An ordinary runtime refuses broken catalogs instead of weakening isolation."""
    from tldw_Server_API.app.core.DB_Management.media_db.errors import DatabaseError

    f = migrated_runtime
    with f.admin.transaction() as conn:
        for statement in drift if isinstance(drift, tuple) else (drift,):
            conn.execute(sql.SQL(statement).format(runtime=sql.Identifier(f.role)))
    with pytest.raises(DatabaseError, match="runtime|schema-owner"):
        MediaDatabase(":memory:", client_id="101", backend=f.backend)
    stored = f.admin.execute("SELECT content FROM media WHERE id=%s", (f.source_id,)).scalar
    # Failed open must not alter the retained source.
    assert stored == "Migrated Rowan Observatory source"  # nosec B101


@pytest.mark.parametrize("migrated_runtime", [False], indirect=True)
def test_media_only_schema_does_not_require_conversation_policy(migrated_runtime):
    """A standalone Media schema also supports the ordinary runtime login."""
    f = migrated_runtime
    db = MediaDatabase(":memory:", client_id="101", backend=f.backend)
    try:
        # Conversation policies are optional; migrated source access is not.
        assert get_media_by_id(db, f.source_id)["content"] == "Migrated Rowan Observatory source"  # nosec B101
    finally:
        db.close_connection()


@pytest.mark.parametrize("clause", ["USING", "WITH CHECK"])
def test_media_refuses_weakened_shared_conversation_policy(migrated_runtime, clause):
    """The legitimate optional policy name cannot disguise a permissive rule."""
    from tldw_Server_API.app.core.DB_Management.media_db.errors import DatabaseError

    f = migrated_runtime
    with f.admin.transaction() as conn:
        conn.execute(sql.SQL("ALTER POLICY sync_log_tenant_isolation ON sync_log {} (true)").format(sql.SQL(clause)))
    with pytest.raises(DatabaseError, match="policy authority drifted for sync_log/sync_log_tenant_isolation"):
        MediaDatabase(":memory:", client_id="101", backend=f.backend)


def test_owner_upgrade_refreshes_legacy_sync_scope_predicates_for_runtime(migrated_runtime):
    """The v0.1.32 policy forms must migrate before a nonowner runtime reopens."""
    f = migrated_runtime
    legacy = {
        "sync_scope_admin": "COALESCE(current_setting('app.is_admin', true), '0') = '1'",
        "sync_scope_personal": "sync_log.client_id = current_setting('app.current_user_id', true)",
        "sync_scope_org": "sync_log.org_id IS NOT NULL AND sync_log.org_id = ANY(COALESCE(string_to_array(NULLIF(current_setting('app.org_ids', true), ''), ',')::BIGINT[], ARRAY[]::BIGINT[]))",
        "sync_scope_team": "sync_log.team_id IS NOT NULL AND sync_log.team_id = ANY(COALESCE(string_to_array(NULLIF(current_setting('app.team_ids', true), ''), ',')::BIGINT[], ARRAY[]::BIGINT[]))",
    }
    with f.admin.transaction() as conn:
        for name, predicate in legacy.items():
            conn.execute(
                sql.SQL("ALTER POLICY {} ON sync_log USING ({}) WITH CHECK ({})").format(
                    sql.Identifier(name), sql.SQL(predicate), sql.SQL(predicate)
                )
            )
    owner_backend = DatabaseBackendFactory.create_backend(f.owner.config)
    try:
        owner = MediaDatabase(":memory:", client_id="101", backend=owner_backend)
        owner.close_connection()
    finally:
        owner_backend.get_pool().close_all()
    runtime = MediaDatabase(":memory:", client_id="101", backend=f.backend)
    try:
        assert get_media_by_id(runtime, f.source_id)["content"] == "Migrated Rowan Observatory source"  # nosec B101 # real owner-upgrade preservation assertion.
    finally:
        runtime.close_connection()


def test_nonowner_settings_roundtrip_does_not_require_schema_creation(migrated_runtime):
    """Ordinary runtime settings reads and versioned writes need no schema authority."""
    f = migrated_runtime
    owner_backend = DatabaseBackendFactory.create_backend(f.owner.config)
    try:
        owner = CharactersRAGDB(":memory:", client_id="101", backend=owner_backend)
        try:
            character = owner.add_character_card({"name": "X01 settings owner"})
            conversation = owner.add_conversation({"title": "X01 retained settings", "character_id": character})
            assert owner.upsert_conversation_settings(conversation, {"temperature": 0.2})  # nosec B101 # real owner seed.
        finally:
            owner.close_connection()
    finally:
        owner_backend.get_pool().close_all()
    runtime = CharactersRAGDB(":memory:", client_id="101", backend=f.backend)
    try:
        runtime._ensure_conversation_settings_table()
        before = runtime.get_conversation_settings(conversation)
        assert before and before["settings"] == {"temperature": 0.2}  # nosec B101 # actual restricted read.
        with runtime.transaction() as conn:
            runtime._ensure_conversation_settings_table(connection=conn)
            assert runtime.upsert_conversation_settings(
                conversation,
                {"temperature": 0.4},
                conn=conn,
                expected_settings_version=before["settings_version"],
            )  # nosec B101 # actual caller-owned versioned restricted write.
        after = runtime.get_conversation_settings(conversation)
        assert (
            after
            and after["settings"] == {"temperature": 0.4}
            and after["settings_version"] == before["settings_version"] + 1
        )  # nosec B101 # data and version preservation.
    finally:
        runtime.close_connection()


@pytest.mark.parametrize(
    "url",
    ["x01-new.txt", "https://example.com/x01-shared-source", None],
    ids=["same-filename", "same-remote-url", "same-generated-local-url"],
)
def test_two_ordinary_owners_can_persist_the_same_private_source(migrated_runtime, url):
    """A shared database must preserve each owner's independent source identity."""
    from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseError
    from tldw_Server_API.app.core.DB_Management.media_db.errors import DatabaseError as MediaDatabaseError

    f = migrated_runtime
    saved = {}
    for actor in (101, 202):
        with scoped_context(user_id=actor, org_ids=[], team_ids=[], is_admin=False):
            db = MediaDatabase(":memory:", client_id=str(actor), backend=f.backend)
            try:
                try:
                    media_id, _uuid, _message = db.add_media_with_keywords(
                        url=url,
                        title="X01 independently owned source",
                        media_type="document",
                        content="Rowan Observatory opened in 1988. Tours start on Friday.",
                        visibility="personal",
                        owner_user_id=actor,
                    )
                except (DatabaseError, MediaDatabaseError) as error:
                    raise AssertionError(
                        {
                            "phase": "private-source-persist",
                            "actor": actor,
                            "error": type(error).__name__,
                            "cause": type(error.__cause__ or error.__context__).__name__,
                        }
                    ) from None
                own = get_media_by_id(db, media_id)
                assert own["owner_user_id"] == actor  # nosec B101 -- actual authenticated owner, never a foreign row reuse.
                saved[actor] = media_id
                if actor == 202:
                    assert get_media_by_id(db, saved[101]) is None  # nosec B101 -- RLS must retain private source isolation.
            finally:
                db.close_connection()
    assert saved[101] != saved[202]  # nosec B101 -- independent personal records in the shared store.
    with scoped_context(user_id=101, org_ids=[], team_ids=[], is_admin=False):
        db = MediaDatabase(":memory:", client_id="101", backend=f.backend)
        try:
            assert get_media_by_id(db, saved[101])["owner_user_id"] == 101  # nosec B101 -- first owner's source remains intact.
            assert get_media_by_id(db, saved[202]) is None  # nosec B101 -- symmetric foreign-owner denial.
        finally:
            db.close_connection()


@pytest.mark.parametrize("visibility", ["org", "team", "admin-readable"])
def test_ingestion_does_not_reuse_a_visible_foreign_source(migrated_runtime, visibility):
    """Sharing read access does not transfer an ingestion identity or authorize an implicit overwrite."""
    f = migrated_runtime
    with scoped_context(user_id=101, org_ids=[37], team_ids=[63], is_admin=False):
        db = MediaDatabase(":memory:", client_id="101", backend=f.backend)
        try:
            original, _uuid, _message = db.add_media_with_keywords(
                url="https://example.com/x01-visible-source",
                title="Original shared source",
                media_type="document",
                content="Original shared body.",
                visibility="personal" if visibility == "admin-readable" else visibility,
                owner_user_id=101,
            )
            before = get_media_by_id(db, original)
        finally:
            db.close_connection()
    with scoped_context(user_id=202, org_ids=[37], team_ids=[63], is_admin=visibility == "admin-readable"):
        db = MediaDatabase(":memory:", client_id="202", backend=f.backend)
        try:
            assert get_media_by_id(db, original) is not None  # nosec B101 -- actual role-visible foreign source before ingestion.
            new, _uuid, _message = db.add_media_with_keywords(
                url="https://example.com/x01-visible-source",
                title="Independent private source",
                media_type="document",
                content="Independent replacement body.",
                visibility="personal",
                owner_user_id=202,
                overwrite=True,
            )
            assert new != original, {"foreignIdentityReused": new == original}  # nosec B101 -- visible foreign records are not ingestion targets.
            assert get_media_by_id(db, new)["owner_user_id"] == 202  # nosec B101 -- ownership remains the actual ingesting principal.
        finally:
            db.close_connection()
    with scoped_context(user_id=101, org_ids=[37], team_ids=[63], is_admin=False):
        db = MediaDatabase(":memory:", client_id="101", backend=f.backend)
        try:
            assert get_media_by_id(db, original) == before  # nosec B101 -- no foreign touch, reparenting or history mutation.
        finally:
            db.close_connection()


def test_same_owner_source_replay_and_overwrite_retain_identity(migrated_runtime):
    """Omitted explicit-owner arguments must still reuse the active owner's identity."""
    db = MediaDatabase(":memory:", client_id="101", backend=migrated_runtime.backend)
    try:
        first, uuid, _message = db.add_media_with_keywords(
            url="https://example.com/x01-replay", title="Before", content="Before body."
        )
        same, same_uuid, _message = db.add_media_with_keywords(
            url="https://example.com/x01-replay", title="Before", content="Before body."
        )
        changed, changed_uuid, _message = db.add_media_with_keywords(
            url="https://example.com/x01-replay", title="After", content="After body.", overwrite=True
        )
        assert (same, same_uuid, changed, changed_uuid) == (first, uuid, first, uuid)  # nosec B101 -- ordinary replay and legitimate overwrite stay one identity.
        assert get_media_by_id(db, first)["content"] == "After body."  # nosec B101 -- actual own overwrite remains functional.
    finally:
        db.close_connection()


@pytest.mark.parametrize(
    "client,explicit,expected",
    [
        ("101", None, "101"),
        ("00101", None, "101"),
        (101, None, "101"),
        ("legacy-device", None, "legacy-device"),
        (" legacy-device ", None, " legacy-device "),
        ("device", 202, "202"),
        ("device", " 202 ", "202"),
    ],
)
def test_media_owner_identity_matches_inserted_ownership(client, explicit, expected):
    from tldw_Server_API.app.core.DB_Management.media_db.dedupe_urls import media_owner_identity

    assert media_owner_identity(client, explicit) == expected  # nosec B101 -- lookup identity must equal persisted owner/client fallback.


@pytest.mark.parametrize(
    "client,explicit", [(None, None), ("", None), (" ", None), (True, None), ("device", ""), ("device", True)]
)
def test_media_owner_identity_rejects_unresolved_new_ownership(client, explicit):
    from tldw_Server_API.app.core.DB_Management.media_db.dedupe_urls import media_owner_identity
    from tldw_Server_API.app.core.DB_Management.media_db.errors import InputError

    with pytest.raises(InputError):
        media_owner_identity(client, explicit)


@pytest.mark.parametrize("migrated_runtime", [False], indirect=True)
@pytest.mark.parametrize("lookup", ["url", "content_hash", "id-and-url"])
def test_legacy_identity_lookup_preserves_owner_and_explicit_id_visibility(migrated_runtime, lookup):
    from tldw_Server_API.app.core.DB_Management.media_db.api import check_media_exists

    f = migrated_runtime
    with scoped_context(user_id=101, org_ids=[37], team_ids=[], is_admin=False):
        db = MediaDatabase(":memory:", client_id="101", backend=f.backend)
        try:
            original, _, _ = db.add_media_with_keywords(
                url="https://example.com/x01-legacy-lookup",
                title="Shared source",
                content="Shared source body",
                visibility="org",
                owner_user_id=101,
            )
            content_hash = get_media_by_id(db, original)["content_hash"]
        finally:
            db.close_connection()
    with scoped_context(user_id=202, org_ids=[37], team_ids=[], is_admin=False):
        db = MediaDatabase(":memory:", client_id="202", backend=f.backend)
        try:
            criteria = (
                {"url": "https://example.com/x01-legacy-lookup"}
                if lookup != "content_hash"
                else {"content_hash": content_hash}
            )
            if lookup == "id-and-url":
                criteria["media_id"] = original
            found = check_media_exists(db, **criteria)
            assert found == (original if lookup == "id-and-url" else None)  # nosec B101 -- ID read visibility is separate from source identity.
        finally:
            db.close_connection()


@pytest.mark.parametrize("migrated_runtime", [False], indirect=True)
def test_media_runtime_refuses_global_url_expression_uniqueness(migrated_runtime):
    from tldw_Server_API.app.core.DB_Management.media_db.errors import DatabaseError

    f = migrated_runtime
    with f.admin.transaction() as conn:
        conn.execute("CREATE UNIQUE INDEX x01_foreign_global_url ON public.media (lower(url))")
    with pytest.raises(DatabaseError, match="schema-owner"):
        MediaDatabase(":memory:", client_id="101", backend=f.backend)
    assert f.admin.execute("SELECT to_regclass('public.x01_foreign_global_url') IS NOT NULL").scalar  # nosec B101 -- unknown operator indexes are refused, never dropped.


@pytest.mark.parametrize("migrated_runtime", [False], indirect=True)
@pytest.mark.parametrize(
    "drift",
    [
        ("DROP INDEX ux_media_owner_url",),
        ("DROP INDEX ux_media_owner_url", "CREATE UNIQUE INDEX ux_media_owner_url ON public.media(client_id,url)"),
        ("DROP INDEX ux_media_clone_url",),
        (
            "DROP INDEX ux_media_clone_url",
            "CREATE UNIQUE INDEX ux_media_clone_url ON public.media(url) WHERE system_operation_id IS NOT NULL",
        ),
        (
            "DROP INDEX ux_media_clone_url",
            "CREATE INDEX ux_media_clone_url ON public.media(url) WHERE url LIKE 'tldw-clone://workspace/%'",
        ),
        ("CREATE UNIQUE INDEX x01_global_url ON public.media(url)",),
    ],
    ids=["missing-owner", "wrong-owner", "missing-clone", "pending-only-clone", "nonunique-clone", "global-url"],
)
def test_media_identity_catalog_drift_refuses_warm_runtime(migrated_runtime, drift):
    from tldw_Server_API.app.core.DB_Management.media_db.errors import DatabaseError

    f = migrated_runtime
    warm = MediaDatabase(":memory:", client_id="101", backend=f.backend)
    warm.close_connection()
    with f.admin.transaction() as conn:
        for statement in drift:
            conn.execute(statement)
    before = f.admin.execute("SELECT row_to_json(m) AS row FROM public.media m ORDER BY id").rows
    with pytest.raises(DatabaseError, match="schema-owner"):
        MediaDatabase(":memory:", client_id="101", backend=f.backend)
    assert f.admin.execute("SELECT row_to_json(m) AS row FROM public.media m ORDER BY id").rows == before  # nosec B101 -- refusal never mutates source data.


@pytest.mark.parametrize("migrated_runtime", [False], indirect=True)
def test_owner_media_identity_migration_preserves_legacy_rows_and_history(migrated_runtime):
    f = migrated_runtime
    with f.admin.transaction() as conn:
        conn.execute("UPDATE media SET owner_user_id=NULL,client_id='',deleted=TRUE WHERE id=%s", (f.source_id,))
        conn.execute("DROP INDEX ux_media_owner_url")
        conn.execute("DROP INDEX ux_media_clone_url")
        conn.execute("ALTER TABLE media ADD CONSTRAINT media_url_key UNIQUE(url)")
    before = {
        table: f.admin.execute(
            sql.SQL("SELECT row_to_json(t) AS row FROM {} t ORDER BY row_to_json(t)::text")
            .format(sql.Identifier(table))
            .as_string()
        ).rows
        for table in ("media", "documentversions", "sync_log")
    }
    owner_backend = DatabaseBackendFactory.create_backend(f.owner.config)
    owner_db = None
    try:
        owner_db = MediaDatabase(":memory:", client_id="101", backend=owner_backend)
    finally:
        try:
            if owner_db is not None:
                owner_db.close_connection()
        finally:
            owner_backend.get_pool().close_all()
    after = {
        table: f.admin.execute(
            sql.SQL("SELECT row_to_json(t) AS row FROM {} t ORDER BY row_to_json(t)::text")
            .format(sql.Identifier(table))
            .as_string()
        ).rows
        for table in before
    }
    assert after == before  # nosec B101 -- IDs, UUIDs, ownership, deletion, URLs, versions and sync history remain exact.
    runtime = MediaDatabase(":memory:", client_id="101", backend=f.backend)
    try:
        original = before["media"][0]["row"]
        created, _, _ = runtime.add_media_with_keywords(
            url=original["url"], title="New owner source", content="Independent new body"
        )
        assert created != f.source_id  # nosec B101 -- unresolved deleted legacy records are never claimed by a new owner.
        assert (
            f.admin.execute("SELECT row_to_json(m) AS row FROM media m WHERE id=%s", (f.source_id,)).first["row"]
            == original
        )  # nosec B101 -- legacy compatibility namespace is preserved.
    finally:
        runtime.close_connection()


@pytest.mark.parametrize("migrated_runtime", [False], indirect=True)
def test_media_runtime_allows_nonunique_covering_url_index(migrated_runtime):
    f = migrated_runtime
    with f.admin.transaction() as conn:
        conn.execute("CREATE UNIQUE INDEX x01_id_covering_url ON media(id) INCLUDE(url)")
    db = MediaDatabase(":memory:", client_id="101", backend=f.backend)
    db.close_connection()


@pytest.mark.parametrize("migrated_runtime", [False], indirect=True)
def test_concurrent_same_owner_source_saves_converge(migrated_runtime, monkeypatch):
    import time
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseError as BackendDatabaseError

    f = migrated_runtime
    databases = [MediaDatabase(":memory:", client_id="101", backend=f.backend) for _ in range(2)]
    inserted, release = Event(), Event()
    original_log = databases[0]._log_sync_event

    def hold_insert(conn, entity, entity_uuid, operation, version, payload):
        original_log(conn, entity, entity_uuid, operation, version, payload)
        if entity == "Media" and operation == "create":
            inserted.set()
            assert release.wait(15)  # nosec B101 -- bounded real uncommitted insert, no artificial source behavior.

    monkeypatch.setattr(databases[0], "_log_sync_event", hold_insert)

    def save(db):
        with scoped_context(user_id=101, org_ids=[], team_ids=[], is_admin=False):
            return db.add_media_with_keywords(
                url="https://example.com/x01-concurrent", title="Concurrent source", content="Shared same-owner content"
            )

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(save, databases[0])
            second = None
            try:
                assert inserted.wait(15)  # nosec B101 -- the first real insertion must precede the competing request.
                second = executor.submit(save, databases[1])
                deadline = time.monotonic() + 10
                while time.monotonic() < deadline:
                    waiting = f.admin.execute(
                        "SELECT 1 FROM pg_stat_activity WHERE usename=%s AND wait_event_type='Lock' AND state='active'",
                        (f.role,),
                    ).rows
                    if waiting:
                        break
                    time.sleep(0.02)
                assert waiting  # nosec B101 -- actual database lock receipt proves overlapping requests.
            finally:
                release.set()
            one = first.result(timeout=15)
            try:
                two = second.result(timeout=15)
            except BackendDatabaseError as error:
                raise AssertionError(
                    {
                        "competingSaveError": type(error).__name__,
                        "cause": type(error.__cause__ or error.__context__).__name__,
                    }
                ) from None
        assert one[:2] == two[:2]  # nosec B101 -- both requests acknowledge the same durable identity.
        assert (
            f.admin.execute("SELECT count(*) FROM media WHERE url=%s", ("https://example.com/x01-concurrent",)).scalar
            == 1
        )  # nosec B101 -- no duplicate source after the real race.
    finally:
        release.set()
        for db in databases:
            db.close_connection()


@pytest.mark.parametrize("migrated_runtime", [False], indirect=True)
@pytest.mark.parametrize("source_kind", ["remote", "hash-column", "metadata-fallback"])
@pytest.mark.parametrize("actor", [101, 202], ids=["same-owner", "visible-foreign-owner"])
@pytest.mark.asyncio
async def test_audio_precheck_uses_source_owner(migrated_runtime, monkeypatch, tmp_path, source_kind, actor):
    import asyncio
    import hashlib
    import json

    from tldw_Server_API.app.core.DB_Management import media_db
    from tldw_Server_API.app.core.Ingestion_Media_Processing import persistence

    f = migrated_runtime
    local = tmp_path / "x01-audio.wav"
    payload = b"X01 source identity bytes, never sent to a model"
    local.write_bytes(payload)
    source_hash = hashlib.sha256(payload).hexdigest()
    url = "https://example.com/x01-audio.wav" if source_kind == "remote" else local.name
    source = url if source_kind == "remote" else str(local)
    with scoped_context(user_id=101, org_ids=[37], team_ids=[], is_admin=False):
        db = MediaDatabase(":memory:", client_id="101", backend=f.backend)
        try:
            existing, _, _ = db.add_media_with_keywords(
                url=url,
                title="Shared audio",
                content="Retained transcript",
                media_type="audio",
                source_hash=source_hash if source_kind == "hash-column" else None,
                safe_metadata=json.dumps({"source_hash": source_hash}, separators=(",", ":")),
                owner_user_id=101,
                visibility="org",
            )
            before = get_media_by_id(db, existing)
        finally:
            db.close_connection()
    runtime = factory.MediaDbRuntimeConfig(
        default_db_path=":memory:",
        default_config=configparser.ConfigParser(),
        postgres_content_mode=True,
        backend_loader=lambda: f.backend,
    )
    monkeypatch.setattr(media_db.api, "build_media_runtime_config", lambda: runtime)
    references = {source: url}
    with scoped_context(user_id=actor, org_ids=[37], team_ids=[], is_admin=False):
        results = await persistence.process_batch_media(
            media_type="audio",
            urls=[source] if source_kind == "remote" else [],
            uploaded_file_paths=[] if source_kind == "remote" else [source],
            source_to_ref_map=references,
            form_data=SimpleNamespace(overwrite_existing=False),
            chunk_options=None,
            loop=asyncio.get_running_loop(),
            db_path=":memory:",
            client_id=str(actor),
            temp_dir=tmp_path,
            cancel_check=lambda: True,
        )
    assert len(results) == 1 and results[0]["status"] == ("Skipped" if actor == 101 else "Cancelled")  # nosec B101 -- a visible foreign source must not satisfy the real precheck.
    assert results[0]["db_id"] == (existing if actor == 101 else None)  # nosec B101 -- only the current owner's source can be reused.
    assert references == {source: url}  # nosec B101 -- no swallowed SQL/parameter error may masquerade as processing eligibility.
    with scoped_context(user_id=101, org_ids=[37], team_ids=[], is_admin=False):
        db = MediaDatabase(":memory:", client_id="101", backend=f.backend)
        try:
            assert get_media_by_id(db, existing) == before  # nosec B101 -- prechecks/cancellation never touch the shared record.
        finally:
            db.close_connection()


@pytest.mark.parametrize("migrated_runtime", [False], indirect=True)
def test_media_identity_owner_refusal_rolls_back_entire_schema_change(migrated_runtime):
    from tldw_Server_API.app.core.DB_Management.media_db.errors import DatabaseError

    f = migrated_runtime
    with f.admin.transaction() as conn:
        conn.execute("DROP INDEX ux_media_owner_url")
        conn.execute("DROP INDEX ux_media_clone_url")
        conn.execute("ALTER TABLE media ADD CONSTRAINT media_url_key UNIQUE(url)")
        conn.execute("CREATE UNIQUE INDEX x01_unknown_url ON media(lower(url))")
    owner_backend = DatabaseBackendFactory.create_backend(f.owner.config)
    try:
        with pytest.raises(DatabaseError, match="schema-owner"):
            MediaDatabase(":memory:", client_id="101", backend=owner_backend)
    finally:
        owner_backend.get_pool().close_all()
    indexes = f.admin.execute(
        "SELECT indexname,indexdef FROM pg_indexes WHERE schemaname='public' AND tablename='media'"
    ).rows
    names = {row["indexname"] for row in indexes}
    assert (
        "media_url_key" in names
        and "x01_unknown_url" in names
        and "ux_media_owner_url" not in names
        and "ux_media_clone_url" not in names
    )  # nosec B101 -- rejected migration commits neither new indexes nor removal of old guarantees.
