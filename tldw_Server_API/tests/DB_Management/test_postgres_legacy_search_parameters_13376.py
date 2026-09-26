"""Legacy PostgreSQL FTS values follow SQL predicate and ranking order."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from tldw_Server_API.app.core.DB_Management.media_db.repositories.media_search_repository import MediaSearchRepository
from tldw_Server_API.app.core.DB_Management.scope_context import scoped_context
from tldw_Server_API.tests.DB_Management.test_postgres_email_search_acceleration_13376 import (
    accelerated_db as _accelerated_db,
)

# Reuse the existing generated-role fixture without duplicating its RLS setup.
accelerated_db = _accelerated_db

pytestmark = [pytest.mark.integration, pytest.mark.postgres]


@pytest.mark.parametrize('filters', [
    {'media_types': ['email']},
    {'media_types': ['email'], 'use_identity': True},
    {'media_types': ['email'], 'date_range': {'start_date': datetime(2000, 1, 1, tzinfo=timezone.utc),
                                            'end_date': datetime(2100, 1, 1, tzinfo=timezone.utc)}},
    {'media_types': ['email'], 'must_not_have_keywords': ['absent']},
    {'media_types': ['email'], 'must_have_keywords': ['keep']},
])
def test_postgres_legacy_fts_keeps_existing_filter_parameter_order(accelerated_db, filters):
    db, role, ids = accelerated_db
    filters = dict(filters)
    if filters.get('must_have_keywords'):
        with scoped_context(user_id=42, is_admin=True):
            db.update_keywords_for_media(ids['alpha'], ['keep'])
    if filters.pop('use_identity', False):
        filters['media_ids_filter'] = [ids['alpha']]
    with scoped_context(user_id=42, session_role=role):
        rows, total = MediaSearchRepository(db).search('alphabet', **filters)
    assert total == 1 and [row['id'] for row in rows] == [ids['alpha']]


@pytest.mark.parametrize('sort_by', ['relevance', 'last_modified_desc'])
def test_postgres_legacy_fts_count_rank_and_pages_share_type_filter(accelerated_db, sort_by):
    db, role, ids = accelerated_db
    repo = MediaSearchRepository(db)
    with scoped_context(user_id=42, session_role=role):
        first, total = repo.search('alphabet OR BodyOnlyNeedle', media_types=['email'], sort_by=sort_by,
                                  boost_fields={'title': 2}, results_per_page=1)
        second, second_total = repo.search('alphabet OR BodyOnlyNeedle', media_types=['email'], sort_by=sort_by,
                                          boost_fields={'title': 2}, page=2, results_per_page=1)
    assert total == second_total == 2
    assert {first[0]['id'], second[0]['id']} == {ids['alpha'], ids['body']}
    if sort_by == 'relevance':
        assert all('relevance_score' in row for row in first + second)
    with scoped_context(user_id=43, session_role=role):
        assert repo.search('alphabet', media_types=['email'])[1] == 1


def test_postgres_legacy_fts_keeps_following_author_like_parameter(accelerated_db):
    db, role, _ = accelerated_db
    with scoped_context(user_id=42, is_admin=True):
        expected, _, _ = db.add_media_with_keywords(
            title='alphabet author match', content='Unique author filter fixture body', author='alphabet',
            media_type='email', keywords=[], owner_user_id=42,
        )
    with scoped_context(user_id=42, session_role=role):
        rows, total = MediaSearchRepository(db).search(
            'alphabet', media_types=['email'], search_fields=['title', 'author'], sort_by='relevance',
        )
    assert total == 1 and rows[0]['id'] == expected


def test_postgres_legacy_quoted_phrase_keeps_term_order_and_repeated_numbers(accelerated_db):
    db, role, _ = accelerated_db
    phrase = 'Unique synthetic archive body batch 0 message 0'
    with scoped_context(user_id=42, is_admin=True):
        expected, _, _ = db.add_media_with_keywords(
            title='ArchiveThroughput exact', content=phrase + '.',
            media_type='email', keywords=[], owner_user_id=42,
        )
        db.add_media_with_keywords(
            title='ArchiveThroughput distractor', content='Unique synthetic archive body batch 0 message 1.',
            media_type='email', keywords=[], owner_user_id=42,
        )
        db.add_media_with_keywords(
            title='ArchiveThroughput distant', content=phrase.replace('archive body', 'archive distant body'),
            media_type='email', keywords=[], owner_user_id=42,
        )
        db.add_media_with_keywords(
            title='ArchiveThroughput rearranged', content='message 0 batch 0 body archive synthetic Unique.',
            media_type='email', keywords=[], owner_user_id=42,
        )
    with scoped_context(user_id=42, session_role=role):
        rows, total = MediaSearchRepository(db).search('"' + phrase + '"', media_types=['email'])
    assert total == 1 and [row['id'] for row in rows] == [expected]


@pytest.mark.parametrize('boosts,preferred', [
    (None, None), ({'title': 10, 'content': 1}, 'title'),
    ({'title': 1, 'content': 10}, 'content'),
])
def test_postgres_legacy_default_and_boosted_ranking_preserve_field_preference(accelerated_db, boosts, preferred):
    db, role, _ = accelerated_db
    with scoped_context(user_id=42, is_admin=True):
        title_id, _, _ = db.add_media_with_keywords(
            title='RankRatioNeedle', content='Unique rank title fixture body',
            media_type='email', keywords=[], owner_user_id=42,
        )
        body_id, _, _ = db.add_media_with_keywords(
            title='Body ranking fixture', content='RankRatioNeedle',
            media_type='email', keywords=[], owner_user_id=42,
        )
    with scoped_context(user_id=42, session_role=role):
        rows, total = MediaSearchRepository(db).search(
            'RankRatioNeedle', media_types=['email'], sort_by='relevance', boost_fields=boosts,
        )
    assert total == 2 and {row['id'] for row in rows} == {title_id, body_id}
    assert all(row['relevance_score'] >= 0 for row in rows)
    if preferred is not None:
        assert rows[0]['id'] == {'title': title_id, 'content': body_id}[preferred]
