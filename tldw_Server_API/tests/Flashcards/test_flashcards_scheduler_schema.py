import json

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.Flashcards.scheduler_sm2 import (
    SchedulerSettingsError,
    scheduler_settings_to_json,
)


@pytest.fixture
def chacha_db(tmp_path):
    db = CharactersRAGDB(str(tmp_path / "flashcards-scheduler.db"), client_id="flashcard-scheduler-tests")
    try:
        yield db
    finally:
        db.close_connection()


def test_scheduler_schema_columns_exist_on_fresh_db(chacha_db: CharactersRAGDB):
    deck_columns = {
        row["name"] for row in chacha_db.execute_query("PRAGMA table_info('decks')").fetchall()
    }
    flashcard_columns = {
        row["name"] for row in chacha_db.execute_query("PRAGMA table_info('flashcards')").fetchall()
    }
    review_columns = {
        row["name"] for row in chacha_db.execute_query("PRAGMA table_info('flashcard_reviews')").fetchall()
    }

    assert "scheduler_settings_json" in deck_columns
    assert "scheduler_type" in deck_columns
    assert "parent_deck_id" in deck_columns
    assert "queue_state" in flashcard_columns
    assert "step_index" in flashcard_columns
    assert "suspended_reason" in flashcard_columns
    assert "scheduler_state_json" in flashcard_columns
    assert "scheduler_type" in review_columns
    assert "previous_queue_state" in review_columns
    assert "next_queue_state" in review_columns
    assert "previous_due_at" in review_columns
    assert "next_due_at" in review_columns


def test_new_deck_persists_default_scheduler_settings(chacha_db: CharactersRAGDB):
    deck_id = chacha_db.add_deck("Scheduler Defaults")
    deck = chacha_db.get_deck(deck_id)

    assert deck is not None
    raw_settings = deck.get("scheduler_settings_json")
    assert isinstance(raw_settings, str)

    settings = json.loads(raw_settings)
    assert deck["scheduler_type"] == "sm2_plus"
    assert set(settings.keys()) == {"sm2_plus", "fsrs"}

    sm2_settings = settings["sm2_plus"]
    assert sm2_settings["new_steps_minutes"] == [1, 10]
    assert sm2_settings["relearn_steps_minutes"] == [10]
    assert sm2_settings["graduating_interval_days"] == 1
    assert sm2_settings["easy_interval_days"] == 4
    assert sm2_settings["easy_bonus"] == pytest.approx(1.3)
    assert sm2_settings["interval_modifier"] == pytest.approx(1.0)
    assert sm2_settings["max_interval_days"] == 36500
    assert sm2_settings["leech_threshold"] == 8
    assert sm2_settings["enable_fuzz"] is False

    fsrs_settings = settings["fsrs"]
    assert fsrs_settings["target_retention"] == pytest.approx(0.9)
    assert fsrs_settings["maximum_interval_days"] == 36500
    assert fsrs_settings["enable_fuzz"] is False


def test_scheduler_settings_to_json_validates_fsrs_envelope():
    with pytest.raises(SchedulerSettingsError):
        scheduler_settings_to_json({"fsrs": {"target_retention": 1.2}})

    with pytest.raises(SchedulerSettingsError):
        scheduler_settings_to_json({"fsrs": "{not-valid-json"})
