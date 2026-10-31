"""PostgreSQL email identity reads retain precedence and isolation with fewer RPCs."""

from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import replace
from uuid import uuid4

import pytest

from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.media_db.api import fetch_keywords_for_media
from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase
from tldw_Server_API.app.core.DB_Management.scope_context import scoped_context

pytestmark = [pytest.mark.integration, pytest.mark.postgres]


@pytest.fixture
def identity_db(pg_database_config, monkeypatch):
    """Use the standard isolated PG fixture and an actual non-bypass request role."""
    monkeypatch.setenv("TLDW_CONTENT_PG_ROLE_SWITCH", "1")
    role = "email_identity_" + uuid4().hex[:12]
    config = replace(pg_database_config, pool_size=1)
    backend = DatabaseBackendFactory.create_backend(config)
    db = MediaDatabase(":memory:", client_id="42", backend=backend)
    ident = backend.escape_identifier(role)
    try:
        with backend.transaction() as conn:
            backend.execute(f"CREATE ROLE {ident} NOLOGIN NOSUPERUSER NOBYPASSRLS", connection=conn)
            backend.execute(f"GRANT USAGE ON SCHEMA public TO {ident}", connection=conn)
            backend.execute(f"GRANT ALL ON ALL TABLES IN SCHEMA public TO {ident}", connection=conn)
            backend.execute(f"GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO {ident}", connection=conn)
        with scoped_context(user_id=42, session_role=role):
            yield db, role, config
    finally:
        db.close_connection()
        with backend.transaction() as conn:
            backend.execute(f"DROP OWNED BY {ident}", connection=conn)
            backend.execute(f"DROP ROLE IF EXISTS {ident}", connection=conn)
        backend.get_pool().close_all()


def _metadata(provider_id="wanted-provider", message_id="<wanted@example.test>", *, source="inbox", provider="upload"):
    return {"source_key": source, "email_source_provider": provider,
            "email": {"source_message_id": provider_id, "message_id": message_id}}


def _add(db, metadata=None, *, body="Synthetic identity body", overwrite=False, filename="same.eml", owner=42,
         keywords=None):
    return db.add_media_with_keywords(
        url=filename, title="Synthetic identity", media_type="email", content=body, keywords=keywords or [],
        safe_metadata=json.dumps(metadata or _metadata()), owner_user_id=owner, overwrite=overwrite,
    )


def _seed(db, role, metadata=None, *, native=True, owner=42, tenant="42", keep_url=False):
    metadata = metadata or _metadata()
    with scoped_context(user_id=42, session_role=role, is_admin=True):
        # Seed distinct legacy rows before assigning deliberately overlapping
        # normalized identities for the precedence regressions.
        creation_metadata = metadata
        if native:
            creation_metadata = {**metadata, "email": {
                "source_message_id": uuid4().hex, "message_id": f"<{uuid4().hex}@example.test>",
            }}
        media_id = _add(db, creation_metadata, filename=uuid4().hex + ".eml", owner=owner)[0]
        if native:
            db.upsert_email_message_graph(
                media_id=media_id, tenant_id=tenant, provider=metadata["email_source_provider"],
                source_key=metadata["source_key"], metadata=metadata, body_text="Synthetic identity body",
            )
        if not keep_url:
            db.execute_query("UPDATE Media SET url = ? WHERE id = ?", ("email://old-native/" + str(media_id), media_id))
    return media_id


def _observe_identity_reads(db, monkeypatch):
    reads = []
    original = db._fetchone_with_connection

    def observe(conn, query, params=None):
        if "FROM Media m" in query and "m.system_operation_id IS NULL" in query:
            reads.append(query)
        return original(conn, query, params)

    monkeypatch.setattr(db, "_fetchone_with_connection", observe)
    return reads


def test_normalized_rfc_match_requires_one_identity_read(identity_db, monkeypatch):
    db, role, _ = identity_db
    expected = _seed(db, role, _metadata(provider_id="old-provider"))
    reads = _observe_identity_reads(db, monkeypatch)
    result = _add(db)
    assert result[0] == expected
    assert len(reads) == 1


def test_new_email_keeps_locked_recheck_with_two_identity_reads(identity_db, monkeypatch):
    db, _, _ = identity_db
    reads = _observe_identity_reads(db, monkeypatch)
    result = _add(db)
    assert result[0] is not None
    assert len(reads) == 2


def test_canonical_url_precedes_both_normalized_identities(identity_db):
    db, role, _ = identity_db
    canonical = _seed(db, role, native=False, keep_url=True)
    _seed(db, role, _metadata(message_id="<different-source@example.test>"))
    _seed(db, role, _metadata(provider_id="different-provider"))
    assert _add(db)[0] == canonical


def test_provider_identity_precedes_rfc_identity(identity_db):
    db, role, _ = identity_db
    provider = _seed(db, role, _metadata(message_id="<different-source@example.test>"))
    _seed(db, role, _metadata(provider_id="different-provider"))
    assert _add(db)[0] == provider


@pytest.mark.parametrize("excluded", ["owner", "deleted", "system_operation", "type", "tenant", "source", "provider", "source_tenant"])
def test_normalized_lookup_retains_all_identity_predicates(identity_db, excluded):
    db, role, _ = identity_db
    metadata = _metadata(source="other-source" if excluded == "source" else "inbox",
                         provider="gmail" if excluded == "provider" else "upload")
    old = _seed(db, role, metadata, owner=43 if excluded == "owner" else 42,
                tenant="other-tenant" if excluded == "tenant" else "42")
    # Admin scope makes excluded Media rows visible, so the explicit predicates
    # must reject them independently of the real forced-RLS policy.
    with scoped_context(user_id=42, session_role=role, is_admin=True):
        if excluded == "deleted":
            db.execute_query("UPDATE Media SET deleted = TRUE WHERE id = ?", (old,))
        elif excluded == "system_operation":
            db.execute_query("UPDATE Media SET system_operation_id = ?, system_operation_kind = ?, "
                             "system_source_identity = ?, system_content_hash = ? WHERE id = ?",
                             ("synthetic-operation", "shared_workspace_clone", "synthetic-source", "0" * 64, old))
        elif excluded == "type":
            db.execute_query("UPDATE Media SET type = 'document' WHERE id = ?", (old,))
        elif excluded == "source_tenant":
            db.execute_query("UPDATE email_sources SET tenant_id = 'other-tenant' WHERE id IN "
                             "(SELECT source_id FROM email_messages WHERE media_id = ?)", (old,))
        assert _add(db)[0] != old


@pytest.mark.parametrize("valid", [True, False])
def test_original_filename_fallback_still_validates_legacy_metadata(identity_db, valid):
    db, role, _ = identity_db
    old = _seed(db, role, native=False)
    with scoped_context(user_id=42, session_role=role, is_admin=True):
        db.execute_query("UPDATE Media SET url = 'same.eml' WHERE id = ?", (old,))
        if not valid:
            db.execute_query("UPDATE DocumentVersions SET safe_metadata = ? WHERE media_id = ?",
                             (json.dumps(_metadata(provider_id="unrelated-provider")), old))
    assert (_add(db)[0] == old) is valid


def test_normalized_match_preserves_overwrite_behavior(identity_db):
    db, role, _ = identity_db
    expected = _seed(db, role, _metadata(provider_id="old-provider"))
    result = _add(db, body="Replacement synthetic identity body", overwrite=True)
    assert result[0] == expected
    assert db.get_media_by_id(expected)["content"] == "Replacement synthetic identity body"


@pytest.mark.parametrize("overwrite", [False, True])
def test_locked_recheck_sees_real_competing_insert(identity_db, monkeypatch, overwrite):
    db, _, config = identity_db
    backend = DatabaseBackendFactory.create_backend(config)
    # Bootstrap DDL uses the standard fixture's schema owner, then the actual
    # competing write below runs under the same non-bypass request role.
    with scoped_context(user_id=42, is_admin=True):
        contender = MediaDatabase(":memory:", client_id="42", backend=backend)
    winner = []
    lock = db._media_insert_lock

    @contextmanager
    def competing_insert():
        winner.append(_add(contender, body="Winning synthetic identity body")[0])
        with lock:
            yield

    try:
        monkeypatch.setattr(db, "_media_insert_lock", competing_insert())
        result = _add(db, body="Losing synthetic identity body", overwrite=overwrite)
        assert result[0] == winner[0]
        assert "concurrent insert" in result[2]
        assert db.get_media_by_id(winner[0])["content"] == "Winning synthetic identity body"
    finally:
        contender.close_connection()
        backend.get_pool().close_all()


def test_new_empty_keywords_need_no_initial_reads_and_can_be_added_and_removed(identity_db, monkeypatch):
    db, _, _ = identity_db
    reads = []
    fetchall = db._fetchall_with_connection

    def observe_all(conn, query, params=None):
        if "FROM MediaKeywords" in query:
            reads.append(query)
        return fetchall(conn, query, params)

    monkeypatch.setattr(db, "_fetchall_with_connection", observe_all)
    media_id = _add(db)[0]
    assert reads == []
    assert fetch_keywords_for_media(db, media_id) == []
    db.update_keywords_for_media(media_id, ["SyntheticTag"])
    assert fetch_keywords_for_media(db, media_id) == ["synthetictag"]
    db.update_keywords_for_media(media_id, [])
    assert fetch_keywords_for_media(db, media_id) == []


def test_nonempty_creation_and_empty_overwrite_still_replace_keywords(identity_db):
    db, _, _ = identity_db
    media_id = _add(db, keywords=["SyntheticTag"])[0]
    assert fetch_keywords_for_media(db, media_id) == ["synthetictag"]
    assert _add(db, body="Replacement synthetic body", overwrite=True)[0] == media_id
    assert fetch_keywords_for_media(db, media_id) == []


def test_new_media_and_later_keyword_links_rollback_together(identity_db):
    db, _, _ = identity_db
    with pytest.raises(RuntimeError, match="Synthetic rollback"):
        with db.transaction():
            media_id = _add(db)[0]
            db.update_keywords_for_media(media_id, ["SyntheticTag"])
            raise RuntimeError("Synthetic rollback")
    assert db.get_media_by_id(media_id) is None
    assert db.execute_query("SELECT COUNT(*) AS n FROM MediaKeywords WHERE media_id = ?", (media_id,)).fetchone()["n"] == 0
