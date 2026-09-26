"""Legacy search avoids repeated whole-index scans and private INFO logs."""
from __future__ import annotations

import sqlite3

import pytest
from loguru import logger

from tldw_Server_API.app.core.DB_Management.media_db.errors import DatabaseError
from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase
from tldw_Server_API.app.core.DB_Management.media_db.repositories.media_search_repository import MediaSearchRepository
from tldw_Server_API.app.core.DB_Management.scope_context import scoped_context


@pytest.fixture
def email_db(tmp_path):
    db = MediaDatabase(db_path=str(tmp_path / 'legacy.db'), client_id='42')
    try:
        for index in range(120):
            db.add_media_with_keywords(title=f'Report {index}', content=f'Synthetic report body {index}',
                                       media_type='email', keywords=[], owner_user_id=42)
        yield db
    finally:
        db.close_connection()


def _capture_plans(db, monkeypatch):
    original = db.execute_query
    plans = {'count': [], 'page': []}

    def execute(sql, params=None):
        if sql.startswith('SELECT COUNT(') or sql.startswith('SELECT DISTINCT m.id'):
            kind = 'count' if sql.startswith('SELECT COUNT(') else 'page'
            plans[kind] = db.backend.execute('EXPLAIN QUERY PLAN ' + sql, params, connection=db.get_connection()).rows
        return original(sql, params)

    monkeypatch.setattr(db, 'execute_query', execute)
    return plans


def test_legacy_fts_count_scans_matching_index_once(email_db, monkeypatch):
    plans = _capture_plans(email_db, monkeypatch)
    with scoped_context(user_id=42):
        rows, total = MediaSearchRepository(email_db).search('Report', media_types=['email'], results_per_page=20)
    assert total == 120 and len(rows) == 20
    scans = [row['detail'] for row in plans['count'] if row['detail'].startswith(('SCAN ', 'SEARCH '))]
    assert scans[0].startswith('SCAN fts VIRTUAL TABLE'), plans['count']
    assert '=M' not in scans[0], plans['count']


def test_legacy_page_fetches_latest_metadata_by_media_identity(email_db, monkeypatch):
    plans = _capture_plans(email_db, monkeypatch)
    rows, total = MediaSearchRepository(email_db).search('Report', results_per_page=20)
    assert total == 120 and len(rows) == 20
    details = [row['detail'] for row in plans['page']]
    assert not any('SCAN latest_source_metadata' in detail for detail in details), plans['page']
    assert any('SEARCH dv USING INDEX' in detail and 'media_id=?' in detail for detail in details), plans['page']


def test_latest_metadata_ignores_deleted_versions_and_retains_missing_versions(email_db):
    email_db.create_document_version(1, 'version two', safe_metadata='{"chosen":2}')
    removed = email_db.create_document_version(1, 'version three', safe_metadata='{"chosen":3}')
    email_db.soft_delete_document_version(removed['uuid'])
    # A media row without document versions must still be returned by a LEFT JOIN.
    email_db.execute_query('DELETE FROM DocumentVersions WHERE media_id = ?', (2,))
    rows, total = MediaSearchRepository(email_db).search(None, media_ids_filter=[1, 2], results_per_page=10)
    metadata = {row['id']: row['safe_metadata'] for row in rows}
    assert total == 2 and metadata == {1: {'chosen': 2}, 2: None}


def test_relevance_ranking_and_paging_keep_search_scope(email_db):
    repo = MediaSearchRepository(email_db)
    with scoped_context(user_id=42):
        first, total = repo.search('Report', sort_by='relevance', results_per_page=10, boost_fields={'title': 2})
        second, second_total = repo.search('Report', sort_by='relevance', page=2, results_per_page=10)
    assert total == second_total == 120
    assert len(first) == len(second) == 10
    assert not {row['id'] for row in first}.intersection(row['id'] for row in second)
    assert all('relevance_score' in row for row in first)
    with scoped_context(user_id=43):
        assert repo.search('Report', sort_by='relevance') == ([], 0)


@pytest.mark.parametrize('path', ['fts', 'like', 'count_fallback', 'page_fallback', 'database_error', 'unexpected_error'])
def test_legacy_search_info_logs_omit_queries_titles_and_exception_data(tmp_path, monkeypatch, path):
    sentinel = 'SensitiveEmailSearchSentinel13376'
    db = MediaDatabase(db_path=str(tmp_path / 'logs.db'), client_id='42')
    try:
        db.add_media_with_keywords(title=sentinel, content=sentinel, media_type='email', keywords=[], author=sentinel)
        original = db.execute_query
        raised = False

        def execute(sql, params=None):
            nonlocal raised
            is_page = sql.startswith('SELECT DISTINCT m.id')
            if not raised and path in {'count_fallback', 'page_fallback', 'database_error', 'unexpected_error'}:
                if path == 'page_fallback' and not is_page:
                    return original(sql, params)
                raised = True
                if path in {'count_fallback', 'page_fallback'}:
                    raise sqlite3.OperationalError('fts5: syntax error ' + sentinel)
                if path == 'database_error':
                    raise sqlite3.OperationalError(sentinel)
                raise RuntimeError(sentinel)
            return original(sql, params)

        monkeypatch.setattr(db, 'execute_query', execute)
        output = []
        sink = logger.add(output.append, level='INFO', format='{level} {message} {extra}', backtrace=True, diagnose=True,
                          filter=lambda record: record['name'].endswith('media_search_repository'))
        try:
            repo = MediaSearchRepository(db)
            if path.endswith('_error'):
                with pytest.raises(DatabaseError):
                    repo.search(sentinel)
            else:
                rows, total = repo.search(sentinel, search_fields=['author'] if path == 'like' else None,
                                          sort_by='relevance' if path == 'page_fallback' else 'last_modified_desc')
                assert total == len(rows) == 1
        finally:
            logger.remove(sink)
        assert output
        assert sentinel not in ''.join(output)
    finally:
        db.close_connection()
