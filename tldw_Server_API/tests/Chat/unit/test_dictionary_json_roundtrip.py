"""Dictionary JSON keeps browser numeric meaning and exported timing policy."""

import json

import pytest

from tldw_Server_API.app.core.Character_Chat.chat_dictionary import ChatDictionaryService
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, InputError

pytestmark = pytest.mark.integration
TIMING = {"sticky": 3, "cooldown": 4, "delay": 2}


@pytest.fixture(params=["sqlite", "postgres"])
def dictionary_service(request, tmp_path):
    """Use actual storage; PostgreSQL uses the official restricted per-test role."""
    backend = request.getfixturevalue("pg_restricted_backend") if request.param == "postgres" else None
    db = CharactersRAGDB(tmp_path / "dictionary-json.db", client_id="1", **({"backend": backend} if backend else {}))
    try:
        yield ChatDictionaryService(db)
    finally:
        db.close_all_connections()


def test_json_probability_one_survives_browser_number_encoding(dictionary_service):
    service = dictionary_service
    source = service.create_dictionary("Browser source")
    service.add_entry(source, pattern="B03-RAW-742", replacement="B03-COOKED-742", probability=1.0)
    exported = service.export_to_json(source)
    exported["name"] = "Browser imported"
    # JSON.stringify serializes numeric 1.0 as 1; both represent certainty here.
    exported["entries"][0]["probability"] = int(exported["entries"][0]["probability"])
    imported = service.import_from_json(json.loads(json.dumps(exported, default=str)))
    actual = service.get_entries(imported)[0]
    assert actual["probability"] == 1.0  # nosec B101 - persisted probability must survive browser JSON roundtrip.
    assert service.process_text("B03-RAW-742", dictionary_id=imported) == "B03-COOKED-742"  # nosec B101 - certainty gives deterministic real processing, without mocking RNG.


def test_json_export_uses_structured_timed_effects(dictionary_service):
    service = dictionary_service
    source = service.create_dictionary("Timed source")
    service.add_entry(source, pattern="raw", replacement="cooked", timed_effects=TIMING)
    exported = service.export_to_json(source)
    assert exported["entries"][0]["timed_effects"] == TIMING  # nosec B101 - the shipped JSON importer accepts a timing object, not an encoded string.


def test_json_import_retains_timed_effects(dictionary_service):
    service = dictionary_service
    imported = service.import_from_json(
        {
            "name": "Timed imported",
            "entries": [
                {"pattern": "raw", "replacement": "cooked", "timed_effects": TIMING},
            ],
        }
    )
    actual = service.get_entries(imported)[0]
    assert json.loads(actual["timed_effects"]) == TIMING  # nosec B101 - import must retain actual saved timing policy rather than dropping it.


@pytest.mark.parametrize(("value", "expected"), [(50, 0.5), (100, 1.0), (0.5, 0.5)])
def test_json_import_preserves_legacy_percentage_and_fraction_values(dictionary_service, value, expected):
    service = dictionary_service
    imported = service.import_from_json(
        {
            "name": "Compatible probability",
            "entries": [
                {"pattern": "raw", "replacement": "cooked", "probability": value},
            ],
        }
    )
    assert service.get_entries(imported)[0]["probability"] == expected  # nosec B101 - retain existing percentage imports and ordinary fractional JSON values.


def test_json_import_accepts_previously_exported_timing_string(dictionary_service):
    service = dictionary_service
    imported = service.import_from_json(
        {
            "name": "Archived timing",
            "entries": [
                {"pattern": "raw", "replacement": "cooked", "timed_effects": json.dumps(TIMING)},
            ],
        }
    )
    assert json.loads(service.get_entries(imported)[0]["timed_effects"]) == TIMING  # nosec B101 - existing exported files remain readable after the structured export correction.


@pytest.mark.parametrize("timing", [{"delay": -1}, {"delay": True}, "not-json"])
def test_json_import_rejects_invalid_timing_without_partial_dictionary(dictionary_service, timing):
    service = dictionary_service
    with pytest.raises(InputError):
        service.import_from_json(
            {
                "name": "Invalid timing",
                "entries": [
                    {"pattern": "raw", "replacement": "cooked", "timed_effects": timing},
                ],
            }
        )
    assert not any(row["name"] == "Invalid timing" for row in service.list_dictionaries(include_inactive=True))  # nosec B101 - rejected import must clean its newly created owned dictionary.


def test_direct_legacy_integer_writer_retains_percentage_meaning(dictionary_service):
    service = dictionary_service
    source = service.create_dictionary("Legacy integer writer")
    service.add_entry(source, pattern="raw", replacement="cooked", probability=1)
    assert service.get_entries(source)[0]["probability"] == 0.01  # nosec B101 - the fix belongs at the JSON boundary, not the legacy shared writer.
