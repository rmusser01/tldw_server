"""The fast synthetic setup must preserve real persisted search/detail behavior."""
import json
from datetime import datetime, timezone

import pytest

from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase
from tldw_Server_API.app.core.DB_Management.media_db.runtime.email_benchmark_fixture import seed_email_benchmark_fixture


@pytest.fixture
def database(tmp_path):
    db = MediaDatabase(db_path=str(tmp_path / 'fixture.db'), client_id='42')
    yield db
    db.close_connection()


def test_bulk_fixture_retains_legacy_native_metadata_and_search(database):
    report = seed_email_benchmark_fixture(database, tenant_id='email-benchmark:42', message_target=120,
                                         batch_size=37, sender_pool=10, recipient_pool=20)
    rows, total = database.search_email_messages(query='from:sender0@bench.example', tenant_id='email-benchmark:42', limit=50)
    assert total == 12
    assert report['messages'] == 120
    assert report['attachments'] == 24
    assert report['batches'] == 4
    assert database.search_email_messages(query='', tenant_id='other')[1] == 0
    row = rows[0]
    saved = database.execute_query('SELECT m.content, dv.content AS version_content, dv.safe_metadata FROM Media m JOIN DocumentVersions dv ON dv.media_id=m.id WHERE m.id=?', (row['media_id'],)).fetchone()
    assert saved['content'] == saved['version_content']
    assert json.loads(saved['safe_metadata'])['email']['subject'] == row['subject']
    assert datetime.fromisoformat(row['internal_date']) <= datetime.now(timezone.utc)
    assert database.search_email_messages(query='has:attachment', tenant_id='email-benchmark:42')[1] == 24


def test_bulk_fixture_refuses_non_synthetic_tenant_without_writing(database):
    with pytest.raises(ValueError, match='synthetic tenant'):
        seed_email_benchmark_fixture(database, tenant_id='user:42', message_target=20)
    assert database.execute_query('SELECT COUNT(*) AS n FROM Media').fetchone()['n'] == 0


def test_bulk_fixture_refuses_existing_media_and_retains_it(database):
    media_id, _, _ = database.add_media_with_keywords(url='email://existing', title='Existing', media_type='email', content='preserved', keywords=[])
    with pytest.raises(ValueError, match='empty database'):
        seed_email_benchmark_fixture(database, tenant_id='email-benchmark:42', message_target=20)
    assert database.execute_query('SELECT content FROM Media WHERE id=?', (media_id,)).fetchone()['content'] == 'preserved'


def test_default_ten_operator_mix_has_populated_results(database):
    from Helper_Scripts.benchmarks.email_search_bench import _build_default_query_mix, _fetch_fixture_profile
    seed_email_benchmark_fixture(database, tenant_id='email-benchmark:42', message_target=120,
                                 batch_size=37, sender_pool=10, recipient_pool=20)
    cases = _build_default_query_mix(_fetch_fixture_profile(database, 'email-benchmark:42'))
    assert len(cases) == 10
    for case in cases:
        assert database.search_email_messages(query=case.query, tenant_id='email-benchmark:42')[1] > 0, case
    negation = next(case.query for case in cases if case.name == 'mixed_text_and_negation')
    original_query = negation.rsplit(' -from:', 1)[0]
    assert database.search_email_messages(query=negation, tenant_id='email-benchmark:42')[1] < database.search_email_messages(query=original_query, tenant_id='email-benchmark:42')[1]


@pytest.mark.parametrize("query, expected", [("Benchmark", 120), ("Report", 20)])
def test_bulk_fixture_populates_real_legacy_body_and_title_search(database, query, expected):
    seed_email_benchmark_fixture(database, tenant_id='email-benchmark:42', message_target=120)
    rows, total = database.search_media_db(search_query=query, results_per_page=1)
    assert total == expected
    assert len(rows) == 1


def test_bulk_fixture_collects_statistics_for_reverse_participant_lookup(database):
    seed_email_benchmark_fixture(database, tenant_id='email-benchmark:42', message_target=120)
    rows = database.execute_query("SELECT idx FROM sqlite_stat1 WHERE tbl=?", ('email_message_participants',)).fetchall()
    assert 'idx_email_participant_reverse' in {row['idx'] for row in rows}
