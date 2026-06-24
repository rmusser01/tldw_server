import pytest

from tldw_Server_API.app.core.Governance.store import GovernanceStore

pytestmark = pytest.mark.unit


async def test_open_gap_upsert_deduplicates_same_fingerprint(tmp_path):
    db_path = tmp_path / "gov_gaps.db"
    store = GovernanceStore(sqlite_path=str(db_path))
    await store.ensure_schema()

    first = await store.upsert_open_gap(
        question="Which HTTP client should we use?",
        category="dependencies",
        org_id=7,
        team_id=11,
    )
    second = await store.upsert_open_gap(
        question=" Which HTTP client should we use?  ",  # normalized same question
        category="dependencies",
        org_id=7,
        team_id=11,
    )

    assert first.id == second.id
    assert first.question_fingerprint == second.question_fingerprint
    assert first.status == "open"


async def test_open_gap_upsert_distinct_scope_creates_new_gap(tmp_path):
    db_path = tmp_path / "gov_gaps_scope.db"
    store = GovernanceStore(sqlite_path=str(db_path))
    await store.ensure_schema()

    team_a = await store.upsert_open_gap(
        question="How should errors be handled?",
        category="error_handling",
        org_id=7,
        team_id=101,
    )
    team_b = await store.upsert_open_gap(
        question="How should errors be handled?",
        category="error_handling",
        org_id=7,
        team_id=202,
    )

    assert team_a.id != team_b.id


async def test_open_gap_rejects_negative_numeric_scope_ids(tmp_path):
    db_path = tmp_path / "gov_gaps_negative_scope.db"
    store = GovernanceStore(sqlite_path=str(db_path))
    await store.ensure_schema()

    with pytest.raises(ValueError, match="org_id must be non-negative"):
        await store.upsert_open_gap(
            question="How should shared policy apply?",
            category="general",
            org_id=-1,
        )


async def test_open_gap_rejects_boolean_numeric_scope_ids(tmp_path):
    db_path = tmp_path / "gov_gaps_boolean_scope.db"
    store = GovernanceStore(sqlite_path=str(db_path))
    await store.ensure_schema()

    with pytest.raises(ValueError, match="team_id must be an integer"):
        await store.upsert_open_gap(
            question="How should shared policy apply?",
            category="general",
            team_id=True,
        )


async def test_open_gap_normalizes_blank_text_scope_to_null(tmp_path):
    db_path = tmp_path / "gov_gaps_blank_scope.db"
    store = GovernanceStore(sqlite_path=str(db_path))
    await store.ensure_schema()

    gap = await store.upsert_open_gap(
        question="How should workspace policy apply?",
        category="general",
        workspace_id="   ",
    )

    assert gap.workspace_id is None
