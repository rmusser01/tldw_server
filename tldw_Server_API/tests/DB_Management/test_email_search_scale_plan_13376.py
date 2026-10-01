"""Reverse graph filtering must avoid per-message work without changing results."""
import pytest

from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase
from tldw_Server_API.app.core.DB_Management.media_db.runtime.email_benchmark_fixture import seed_email_benchmark_fixture


@pytest.mark.parametrize('query,expected', [('from:sender0@bench.example', 12), ('to:recipient0@bench.example', 6)])
def test_graph_filter_plan_materializes_matching_ids_once(tmp_path, monkeypatch, query, expected):
    db = MediaDatabase(db_path=str(tmp_path/'fixture.db'), client_id='42')
    try:
        seed_email_benchmark_fixture(db, tenant_id='email-benchmark:42', message_target=120, sender_pool=10, recipient_pool=20)
        original = db._fetchone_with_connection
        plans = []
        def fetch(conn, sql, params=None):
            if sql.startswith('SELECT COUNT('):
                plans.extend(dict(row) for row in conn.execute('EXPLAIN QUERY PLAN '+sql, params or ()))
            return original(conn, sql, params)
        monkeypatch.setattr(db, '_fetchone_with_connection', fetch)
        assert db.search_email_messages(query=query, tenant_id='email-benchmark:42')[1] == expected
        assert not any('CORRELATED' in row['detail'] for row in plans), plans
    finally:
        db.close_connection()


def test_broad_label_count_uses_covering_reverse_links_without_duplicates(tmp_path, monkeypatch):
    db = MediaDatabase(db_path=str(tmp_path / 'labels.db'), client_id='42')
    try:
        seed_email_benchmark_fixture(db, tenant_id='email-benchmark:42', message_target=120)
        original = db._fetchone_with_connection
        plans = []

        def fetch(conn, sql, params=None):
            if sql.startswith('SELECT COUNT('):
                plans.extend(dict(row) for row in conn.execute('EXPLAIN QUERY PLAN ' + sql, params or ()))
            return original(conn, sql, params)

        monkeypatch.setattr(db, '_fetchone_with_connection', fetch)
        # The broad substring matches several distinct labels; each email is counted once.
        rows, total = db.search_email_messages(query='label:%', tenant_id='email-benchmark:42', limit=200)
        assert total == len({row['media_id'] for row in rows}) == 120
        assert any('SEARCH eml USING COVERING INDEX' in row['detail'] for row in plans), plans
    finally:
        db.close_connection()
