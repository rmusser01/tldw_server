"""PostgreSQL acceleration preserves exact predicates and statement scope."""
from __future__ import annotations

from dataclasses import replace
from uuid import uuid4

import pytest

from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseConfig, DatabaseError
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase
from tldw_Server_API.app.core.DB_Management.media_db.schema.email_schema_structures import ensure_postgres_email_schema
from tldw_Server_API.app.core.DB_Management.scope_context import scoped_context

pytestmark = [pytest.mark.integration, pytest.mark.postgres]


@pytest.fixture
def accelerated_db(pg_database_config: DatabaseConfig, monkeypatch):
    monkeypatch.setenv('TLDW_CONTENT_PG_ROLE_SWITCH', '1')
    role = 'email_rls_' + uuid4().hex[:12]
    backend = DatabaseBackendFactory.create_backend(replace(pg_database_config, pool_size=1))
    db = MediaDatabase(db_path=':memory:', client_id='42', backend=backend)
    ident = backend.escape_identifier(role)
    try:
        with backend.transaction() as conn:
            backend.execute(f'CREATE ROLE {ident} NOLOGIN NOSUPERUSER NOBYPASSRLS', connection=conn)
            backend.execute(f'GRANT USAGE ON SCHEMA public TO {ident}', connection=conn)
            backend.execute(f'GRANT SELECT ON ALL TABLES IN SCHEMA public TO {ident}', connection=conn)
        ids = {}
        fixtures = (
            ('alpha', 'alphabet', 'ordinary body', 42, 'personal', [], [], ['Inbox', 'Inboxes']),
            ('body', 'other subject', 'BodyOnlyNeedle', 42, 'personal', [], [], ['Inbox']),
            ('seam', 'left', 'right', 42, 'personal', [], [], ['Team']),
            ('hidden', 'alphabet', 'BodyOnlyNeedle', 43, 'personal', [], [], ['Inbox']),
            ('team', 'Team shared', 'shared team body', 43, 'team', [17], [], ['Shared']),
            ('org', 'Org shared', 'shared org body', 43, 'org', [], [19], ['Shared']),
        )
        for key, subject, body, owner, visibility, teams, orgs, labels in fixtures:
            with scoped_context(user_id=owner, team_ids=teams, org_ids=orgs, is_admin=True):
                media_id, _, _ = db.add_media_with_keywords(title=subject, content=body, media_type='email',
                                                           keywords=[], owner_user_id=owner, visibility=visibility)
                db.upsert_email_message_graph(media_id=media_id, metadata={'email': {'subject': subject,
                    'from': 'DisplayNeedle <sender@example.test>', 'to': 'reader@example.test',
                    'message_id': f'<{key}@example.test>'}}, body_text=body, tenant_id='email-benchmark:42',
                    source_key='acceleration-tests', source_message_id=key, labels=labels)
                ids[key] = media_id
        with backend.transaction() as conn:
            backend.execute('ANALYZE Media', connection=conn)
            backend.execute('ANALYZE email_messages', connection=conn)
        yield db, role, ids
    finally:
        db.close_connection()
        with backend.transaction() as conn:
            backend.execute(f'DROP OWNED BY {ident}', connection=conn)
            backend.execute(f'DROP ROLE IF EXISTS {ident}', connection=conn)
        backend.get_pool().close_all()


def test_rls_settings_are_statement_initplans_and_refresh_on_reuse(accelerated_db):
    db, role, ids = accelerated_db
    observed = []
    for scope in ({'user_id': 42}, {'user_id': 43}, {'user_id': 42, 'team_ids': [17], 'org_ids': [19]},
                  {'user_id': 42, 'is_admin': True}, {'user_id': None}):
        with scoped_context(**scope, session_role=role):
            with db.transaction() as conn:
                rows = db.backend.execute('SELECT id FROM Media ORDER BY id', connection=conn).rows
                observed.append({row['id'] for row in rows})
                plan = db.backend.execute('EXPLAIN (FORMAT JSON) SELECT id FROM Media', connection=conn).rows[0]['QUERY PLAN'][0]
                assert 'InitPlan' in str(plan)
    assert observed == [{ids['alpha'], ids['body'], ids['seam']}, {ids['hidden']},
                        {ids['alpha'], ids['body'], ids['seam'], ids['team'], ids['org']}, set(ids.values()), set()]


@pytest.mark.parametrize('query,keys', [
    ('subject:lpha', ['alpha']), ('BodyOnlyNeedle', ['body']), ('"left right"', []),
    ('label:Inbo', ['alpha', 'body']), ('-label:Inbo', ['seam']), ('label:absent', []),
    ('-label:absent', ['alpha', 'body', 'seam']),
    ('subject:lpha OR BodyOnlyNeedle', ['alpha', 'body']),
    ('BodyOnlyNeedle -label:Team', ['body']), ('from:DisplayNeedle', ['alpha', 'body', 'seam']),
    ('label:Inbo -label:Team', ['alpha', 'body']),
    ('label:Team OR label:Inbo', ['alpha', 'body', 'seam']),
])
def test_acceleration_preserves_substrings_boundaries_label_duplicates_and_scope(accelerated_db, query, keys):
    db, role, ids = accelerated_db
    with scoped_context(user_id=42, session_role=role):
        rows, total = db.search_email_messages(query=query, tenant_id='email-benchmark:42')
    assert total == len(keys) and {row['media_id'] for row in rows} == {ids[key] for key in keys}


def test_label_query_binds_resolved_ids_in_same_search_transaction(accelerated_db, monkeypatch):
    db, role, _ = accelerated_db
    original = db._fetchone_with_connection
    bound_ids = []

    def fetch(conn, sql, params=None):
        if sql.startswith('SELECT COUNT('):
            bound_ids.extend(value for value in params or () if isinstance(value, list))
        return original(conn, sql, params)

    monkeypatch.setattr(db, '_fetchone_with_connection', fetch)
    with scoped_context(user_id=42, session_role=role):
        assert db.search_email_messages(query='label:Inbo', tenant_id='email-benchmark:42')[1] == 2
    assert len(bound_ids) == 1 and len(bound_ids[0]) == 2


def test_repeated_search_uses_parameter_aware_plans_without_leaking_session_setting(accelerated_db, monkeypatch):
    db, role, ids = accelerated_db
    original = db._fetchone_with_connection
    planning = []

    def observe(conn, sql, params=None):
        if sql.startswith('SELECT COUNT('):
            planning.append(db.backend.execute(
                "SELECT current_setting('plan_cache_mode') AS mode", connection=conn,
            ).rows[0]['mode'])
        return original(conn, sql, params)

    monkeypatch.setattr(db, '_fetchone_with_connection', observe)
    with scoped_context(user_id=42, session_role=role):
        for _ in range(23):
            rows, total = db.search_email_messages(query='label:Inbo', tenant_id='email-benchmark:42')
            assert total == 2 and {row['media_id'] for row in rows} == {ids['alpha'], ids['body']}
        with db.transaction() as conn:
            restored = db.backend.execute(
                "SELECT current_setting('plan_cache_mode') AS mode", connection=conn,
            ).rows[0]['mode']
    assert planning == ['force_custom_plan'] * 23 and restored == 'auto'


def test_failed_search_restores_planning_mode_before_pooled_reuse(accelerated_db, monkeypatch):
    db, role, _ = accelerated_db
    original = db._fetchone_with_connection

    def fail_count(conn, sql, params=None):
        if sql.startswith('SELECT COUNT('):
            return db.backend.execute('SELECT 1 / 0', connection=conn)
        return original(conn, sql, params)

    monkeypatch.setattr(db, '_fetchone_with_connection', fail_count)
    with scoped_context(user_id=42, session_role=role):
        with pytest.raises(DatabaseError):
            db.search_email_messages(query='label:Inbo', tenant_id='email-benchmark:42')
        with db.transaction() as conn:
            setting = db.backend.execute(
                "SELECT current_setting('plan_cache_mode') AS mode", connection=conn,
            ).rows[0]['mode']
        assert setting == 'auto'


def test_optional_trigram_denial_rolls_back_savepoint_and_keeps_search_working(accelerated_db, monkeypatch):
    db, role, _ = accelerated_db
    original = db.backend.execute
    denied = []

    def execute(sql, params=None, connection=None, **kwargs):
        if sql.startswith('CREATE EXTENSION'):
            denied.append(True)
            # Cause a real aborted PostgreSQL transaction, not only a Python exception.
            return original('SELECT 1 / 0', connection=connection)
        return original(sql, params, connection=connection, **kwargs)

    monkeypatch.setattr(db.backend, 'execute', execute)
    with db.transaction() as conn:
        ensure_postgres_email_schema(db, conn)
        assert db.backend.execute('SELECT 1 AS alive', connection=conn).rows == [{'alive': 1}]
    assert denied
    with scoped_context(user_id=42, session_role=role):
        assert db.search_email_messages(query='BodyOnlyNeedle', tenant_id='email-benchmark:42')[1] == 1


def test_trigram_index_preserves_long_native_subject(accelerated_db):
    db, role, _ = accelerated_db
    subject = 'x' * (16384 - len('LongSubjectNeedle')) + 'LongSubjectNeedle'
    with scoped_context(user_id=42, is_admin=True):
        media_id, _, _ = db.add_media_with_keywords(
            title='Long synthetic subject', content='body', media_type='email',
            keywords=[], owner_user_id=42, visibility='personal',
        )
        db.upsert_email_message_graph(
            media_id=media_id,
            metadata={'email': {'subject': subject, 'message_id': '<long@example.test>'}},
            body_text='body', tenant_id='email-benchmark:42', source_key='acceleration-tests',
            source_message_id='long', labels=[],
        )
    with scoped_context(user_id=42, session_role=role):
        rows, total = db.search_email_messages(
            query='subject:LongSubjectNeedle', tenant_id='email-benchmark:42',
        )
    assert total == 1 and rows[0]['subject'] == subject
