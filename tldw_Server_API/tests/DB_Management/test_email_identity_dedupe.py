"""Email identities must survive the legacy Media write preceding native upsert."""

import json
from contextlib import nullcontext

import pytest

from tldw_Server_API.app.core.DB_Management.media_db.errors import InputError
from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase

pytestmark = pytest.mark.unit


@pytest.fixture
def db(tmp_path):
    database = MediaDatabase(db_path=str(tmp_path / "identity.sqlite"), client_id="tenant-a")
    yield database
    database.close_connection()


def add(db, message_id, *, source="inbox", provider_id=None, body="same body", filename="same.eml", owner=None):
    metadata = {"source_key": source, "email": {"message_id": message_id, "source_message_id": provider_id}}
    return db.add_media_with_keywords(
        url=filename,
        media_type="email",
        title="synthetic",
        content=body,
        safe_metadata=json.dumps(metadata),
        owner_user_id=owner,
    )[0]


def test_same_source_distinct_ids_do_not_merge_by_url_or_body(db):
    first = add(db, "<first@example.test>")
    second = add(db, "<second@example.test>")
    assert first != second
    assert add(db, "<first@example.test>") == first


def test_identity_is_independent_of_member_filename_with_explicit_source(db):
    first = add(db, "<first@example.test>", filename="one.eml")
    assert add(db, "<first@example.test>", filename="renamed.eml") == first


def test_provider_id_takes_precedence_when_rfc_header_changes(db):
    first = add(db, "<old@example.test>", provider_id="provider-1")
    assert add(db, "<new@example.test>", provider_id="provider-1") == first


def test_identical_id_and_body_remain_source_and_owner_scoped(db):
    first = add(db, "<first@example.test>", source="inbox", owner=1)
    assert add(db, "<first@example.test>", source="sent", owner=1) != first
    assert add(db, "<first@example.test>", source="inbox", owner=2) != first


def test_no_id_fallback_dedupes_only_same_body_within_source(db):
    first = add(db, None)
    assert add(db, None, filename="renamed.eml") == first
    assert add(db, None, body="different body") != first
    assert add(db, None, source="different-source") != first


def test_non_email_content_dedupe_is_unchanged(db):
    first = db.add_media_with_keywords(url="one.txt", media_type="document", content="same document")[0]
    assert db.add_media_with_keywords(url="two.txt", media_type="document", content="same document")[0] == first


def test_document_cannot_overwrite_email_with_identical_body(db):
    email_id = add(db, "<first@example.test>")
    document_id = db.add_media_with_keywords(url="one.txt", media_type="document", content="same body", overwrite=True)[
        0
    ]
    assert document_id != email_id


@pytest.mark.parametrize("rollback", [False, True])
def test_overwrite_preserves_metadata_and_updates_highlights_atomically(db, monkeypatch, rollback):
    from tldw_Server_API.app.core.DB_Management.media_db.api import get_document_version
    from tldw_Server_API.app.core.DB_Management.media_db.repositories import media_repository

    media_id = add(db, "<first@example.test>")
    db.execute_query(
        "INSERT INTO reading_highlights(user_id,item_id,quote,created_at,content_hash_ref) VALUES (?,?,?,?,?)",
        (str(db.client_id), media_id, "same body", "2026-01-01", "old-hash"),
    )

    class CollectionsProbe:
        @classmethod
        def from_backend(cls, **kwargs):
            raise AssertionError("Highlight update must reuse the Media transaction connection")

    monkeypatch.setattr(media_repository, "load_collections_database_cls", lambda: CollectionsProbe)
    metadata = {"source_key": "inbox", "email": {"message_id": "<first@example.test>", "subject": "Updated"}}
    with pytest.raises(InputError) if rollback else nullcontext():
        with db.transaction():
            updated_id = db.add_media_with_keywords(
                url="same.eml",
                media_type="email",
                content="updated body",
                overwrite=True,
                safe_metadata=json.dumps(metadata),
            )[0]
            assert updated_id == media_id
            assert json.loads(get_document_version(db, media_id)["safe_metadata"])["email"]["subject"] == "Updated"
            if rollback:
                raise InputError("Synthetic rollback")
    highlight = db.execute_query("SELECT state FROM reading_highlights WHERE item_id = ?", (media_id,)).fetchone()
    assert highlight["state"] == ("active" if rollback else "stale")
    assert db.get_media_by_id(media_id)["content"] == ("same body" if rollback else "updated body")


@pytest.mark.parametrize("native", [False, True])
def test_legacy_filename_row_is_reused_when_identity_is_known(db, native):
    media_id = add(db, "<legacy@example.test>")
    db.execute_query("UPDATE Media SET url = ?, version = version + 1 WHERE id = ?", ("same.eml", media_id))
    if native:
        db.upsert_email_message_graph(
            media_id=media_id,
            tenant_id="tenant-a",
            source_key="inbox",
            metadata={"email": {"message_id": "<legacy@example.test>"}},
            body_text="same body",
        )
    assert add(db, "<legacy@example.test>") == media_id


def test_existing_normalized_rfc_identity_handles_changed_provider_id(db):
    media_id = add(db, "<legacy@example.test>", provider_id="old-provider-id")
    db.upsert_email_message_graph(
        media_id=media_id,
        tenant_id="tenant-a",
        source_key="inbox",
        metadata={"email": {"message_id": "<legacy@example.test>", "source_message_id": "old-provider-id"}},
        body_text="same body",
    )
    assert add(db, "<legacy@example.test>", provider_id="new-provider-id") == media_id


@pytest.mark.parametrize(
    "change",
    [{"tenant_id": "other"}, {"source_key": "other"}, {"metadata": {"email": {"message_id": "<other@example.test>"}}}],
)
def test_normalized_media_fallback_cannot_replace_another_identity(db, change):
    media_id = add(db, "<first@example.test>")
    kwargs = {
        "media_id": media_id,
        "tenant_id": "tenant-a",
        "source_key": "inbox",
        "metadata": {"email": {"message_id": "<first@example.test>"}},
        "body_text": "same body",
    }
    original = db.upsert_email_message_graph(**kwargs)
    with pytest.raises(InputError):
        db.upsert_email_message_graph(**{**kwargs, **change})
    detail = db.get_email_message_detail(email_message_id=original["email_message_id"], tenant_id="tenant-a")
    assert detail["message_id"] == "<first@example.test>"


def test_ingestion_safe_metadata_preserves_email_identity_and_attachment_fields():
    from tldw_Server_API.app.core.Ingestion_Media_Processing.persistence import build_safe_metadata_subset

    result = build_safe_metadata_subset(
        {
            "filename": "one.eml",
            "source_key": "inbox",
            "labels": ["Inbox"],
            "email": {
                "message_id": "<one@example.test>",
                "from": "sender@example.test",
                "attachments": [{"name": "one.bin", "size": 5}],
                "headers_map": {"Subject": "Synthetic"},
            },
            "untrusted_extra": "drop this",
        }
    )
    assert result["email"]["message_id"] == "<one@example.test>"
    assert result["email"]["attachments"] == [{"name": "one.bin", "size": 5}]
    assert result["source_key"] == "inbox"
    assert result["labels"] == ["Inbox"]
    assert "untrusted_extra" not in result


def test_native_provider_internal_date_takes_precedence_and_normalizes_utc(db):
    media_id = add(db, "<dated@example.test>")
    graph = db.upsert_email_message_graph(
        media_id=media_id,
        source_key="inbox",
        metadata={
            "email": {
                "message_id": "<dated@example.test>",
                "internal_date": "2026-02-10T09:30:00-05:00",
                "date": "Mon, 01 Jan 2024 00:00:00 +0000",
            }
        },
        body_text="same body",
    )
    detail = db.get_email_message_detail(email_message_id=graph["email_message_id"])
    assert detail["internal_date"] == "2026-02-10T14:30:00+00:00"


@pytest.mark.parametrize(
    "date, expected",
    [
        ("2026-02-10T14:30:00Z", "2026-02-10T14:30:00+00:00"),
        ("2026-02-10T09:30:00-05:00", "2026-02-10T14:30:00+00:00"),
        ("not-a-date", None),
    ],
)
def test_email_date_parser_accepts_canonical_iso_and_rejects_invalid(db, date, expected):
    assert db._parse_email_internal_date(date) == expected


@pytest.mark.parametrize("date", ["0001-01-01T00:00:00+01:00", "9999-12-31T23:59:59-01:00"])
def test_email_date_parser_rejects_utc_overflow(db, date):
    assert db._parse_email_internal_date(date) is None
