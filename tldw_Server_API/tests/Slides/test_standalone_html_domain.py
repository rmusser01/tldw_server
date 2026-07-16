import hashlib
import json
import sqlite3

import pytest

from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.Slides.slides_db import (
    ConflictError,
    InputError,
    SlidesDatabase,
)


def _structured_slides() -> str:
    return json.dumps(
        [
            {
                "order": 0,
                "layout": "title",
                "title": "Deck",
                "content": "",
                "speaker_notes": None,
                "metadata": {},
            }
        ]
    )


def _create_structured(db: SlidesDatabase, *, presentation_id: str = "structured"):
    return db.create_presentation(
        presentation_id=presentation_id,
        title="Structured",
        description=None,
        theme="black",
        marp_theme=None,
        settings=None,
        studio_data=None,
        slides=_structured_slides(),
        slides_text="structured searchable",
        source_type="manual",
        source_ref=None,
        source_query=None,
        custom_css=None,
    )


def _standalone_fields(html_document: str = "<!doctype html><title>Deck</title>") -> dict:
    encoded = html_document.encode("utf-8")
    return {
        "content_kind": "standalone_html",
        "html_document": html_document,
        "html_sha256": hashlib.sha256(encoded).hexdigest(),
        "html_bytes": len(encoded),
        "html_slide_count": 1,
        "generation_job_uuid": "job-uuid-1",
        "generation_provenance_json": json.dumps(
            {
                "source_kind": "prompt",
                "provider": "openai",
                "model": "test-model",
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
    }


def _create_standalone(
    db: SlidesDatabase,
    *,
    presentation_id: str = "standalone",
    generation_job_uuid: str = "job-uuid-1",
):
    fields = _standalone_fields()
    fields["generation_job_uuid"] = generation_job_uuid
    return db.create_presentation(
        presentation_id=presentation_id,
        title="Standalone",
        description=None,
        theme="black",
        marp_theme=None,
        settings=None,
        studio_data=None,
        slides=json.dumps([]),
        slides_text="standalone searchable",
        source_type="prompt",
        source_ref=None,
        source_query=None,
        custom_css=None,
        **fields,
    )


def test_complete_structured_and_standalone_rows_are_discriminated(tmp_path):
    db = SlidesDatabase(db_path=tmp_path / "Slides.db", client_id="tester")

    structured = _create_structured(db)
    html = _create_standalone(db)

    assert structured.content_kind == "structured_slides"
    assert structured.slides is not None and structured.html_document is None
    assert html.content_kind == "standalone_html"
    assert json.loads(html.slides) == [] and html.html_document is not None
    db.close_connection()


@pytest.mark.parametrize(
    ("field_overrides", "match"),
    [
        ({"content_kind": "unknown"}, "content_kind"),
        (
            {
                "content_kind": "structured_slides",
                "html_document": "<!doctype html>",
            },
            "structured_slides",
        ),
        ({"html_document": "   "}, "nonblank"),
        ({"html_sha256": "0" * 64}, "html_sha256"),
        ({"html_bytes": 1}, "html_bytes"),
        ({"generation_job_uuid": None}, "generation_job_uuid"),
        ({"generation_provenance_json": None}, "generation_provenance_json"),
    ],
)
def test_create_rejects_split_brain_or_incomplete_candidates(
    tmp_path,
    field_overrides,
    match,
):
    db = SlidesDatabase(db_path=tmp_path / "Slides.db", client_id="tester")
    fields = _standalone_fields()
    fields.update(field_overrides)

    with pytest.raises(InputError, match=match):
        db.create_presentation(
            presentation_id="invalid",
            title="Invalid",
            description=None,
            theme="black",
            marp_theme=None,
            settings=None,
            studio_data=None,
            slides=json.dumps([]),
            slides_text="invalid",
            source_type="prompt",
            source_ref=None,
            source_query=None,
            custom_css=None,
            **fields,
        )

    with pytest.raises(KeyError):
        db.get_presentation_by_id("invalid", include_deleted=True)
    db.close_connection()


def test_create_rejects_nonempty_structured_payload_for_standalone(tmp_path):
    db = SlidesDatabase(db_path=tmp_path / "Slides.db", client_id="tester")

    with pytest.raises(InputError, match="slides"):
        db.create_presentation(
            presentation_id="invalid",
            title="Invalid",
            description=None,
            theme="black",
            marp_theme=None,
            settings=None,
            studio_data=None,
            slides=_structured_slides(),
            slides_text="invalid",
            source_type="prompt",
            source_ref=None,
            source_query=None,
            custom_css=None,
            **_standalone_fields(),
        )
    db.close_connection()


def test_update_validates_the_merged_candidate_inside_the_transaction(tmp_path):
    db = SlidesDatabase(db_path=tmp_path / "Slides.db", client_id="tester")
    structured = _create_structured(db)

    with pytest.raises(InputError, match="structured_slides"):
        db.update_presentation(
            presentation_id=structured.id,
            update_fields={"description": "changed", "html_document": "<!doctype html>"},
            expected_version=structured.version,
        )

    unchanged = db.get_presentation_by_id(structured.id)
    assert unchanged.description is None
    assert unchanged.version == structured.version
    db.close_connection()


def test_generation_job_uuid_is_unique_when_nonnull(tmp_path):
    db = SlidesDatabase(db_path=tmp_path / "Slides.db", client_id="tester")
    _create_standalone(db, presentation_id="first", generation_job_uuid="shared-job")

    with pytest.raises(ConflictError):
        _create_standalone(db, presentation_id="second", generation_job_uuid="shared-job")
    db.close_connection()


def test_source_free_projection_queries_do_not_load_html_or_version_payload(tmp_path):
    db = SlidesDatabase(db_path=tmp_path / "Slides.db", client_id="tester")
    html = _create_standalone(db)
    statements: list[str] = []
    db.get_connection().set_trace_callback(statements.append)

    summaries, total = db.list_presentation_summaries(
        limit=10,
        offset=0,
        include_deleted=False,
        sort_column="created_at",
        sort_direction="DESC",
    )
    search_rows, search_total = db.search_presentation_summaries(
        query="standalone",
        limit=10,
        offset=0,
        include_deleted=False,
    )
    kind = db.get_presentation_kind(html.id)
    versions, version_total = db.list_presentation_version_metadata(
        presentation_id=html.id,
        limit=10,
        offset=0,
    )

    assert total == search_total == version_total == 1
    assert type(summaries[0]).__name__ == "PresentationSummaryRow"
    assert type(search_rows[0]).__name__ == "PresentationSummaryRow"
    assert type(kind).__name__ == "PresentationKindRow"
    assert type(versions[0]).__name__ == "PresentationVersionMetadataRow"
    assert not hasattr(summaries[0], "html_document")
    select_sql = "\n".join(
        statement for statement in statements if statement.lstrip().upper().startswith("SELECT")
    ).lower()
    assert "html_document" not in select_sql
    assert "payload_json" not in select_sql
    db.close_connection()


def test_receipt_and_input_getters_use_explicit_typed_projections(tmp_path):
    db = SlidesDatabase(db_path=tmp_path / "Slides.db", client_id="tester")
    with db.transaction() as conn:
        conn.execute(
            """
            INSERT INTO slides_generation_receipts (
                id, owner_user_id, digest_key_id,
                idempotency_key_hmac_sha256, jobs_idempotency_key,
                client_request_hmac_sha256, execution_hmac_sha256,
                receipt_status, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "receipt-1",
                "owner-1",
                "key-v1",
                "a" * 64,
                "slides:v1:test",
                "b" * 64,
                "c" * 64,
                "claimed",
                "2026-01-01T00:00:00+00:00",
                "2026-01-01T00:00:00+00:00",
            ),
        )
        conn.execute(
            """
            INSERT INTO slides_generation_inputs (
                receipt_id, source_kind, source_text, source_hmac_sha256,
                source_bytes, provenance_json, html_options_json, provider,
                model, adapter_id, endpoint_identity, system_prompt,
                prompt_sha256, prompt_contract_version, input_expires_at,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "receipt-1",
                "prompt",
                "source",
                "d" * 64,
                6,
                "{}",
                "{}",
                "openai",
                "test-model",
                "openai_official_chat_v1",
                "https://api.openai.com:443/v1/chat/completions",
                "system prompt",
                "e" * 64,
                "slides.standalone_html.v1",
                "2026-01-02T00:00:00+00:00",
                "2026-01-01T00:00:00+00:00",
            ),
        )

    receipt = db.get_generation_receipt("receipt-1", owner_user_id="owner-1")
    generation_input = db.get_generation_input(
        "receipt-1",
        owner_user_id="owner-1",
    )

    assert type(receipt).__name__ == "SlidesGenerationReceiptRow"
    assert type(generation_input).__name__ == "SlidesGenerationInputRow"
    assert receipt.owner_user_id == "owner-1"
    assert generation_input.source_text == "source"
    with pytest.raises(KeyError):
        db.get_generation_receipt("receipt-1", owner_user_id="other-owner")
    with pytest.raises(KeyError):
        db.get_generation_input("receipt-1", owner_user_id="other-owner")
    db.close_connection()


def test_resolve_slides_db_path_does_not_create_user_directory(tmp_path, monkeypatch):
    user_dir = tmp_path / "user-7"
    monkeypatch.setattr(
        DatabasePaths,
        "resolve_user_base_directory",
        staticmethod(lambda _user_id: user_dir),
    )

    resolved = DatabasePaths.resolve_slides_db_path(7)

    assert resolved == user_dir / DatabasePaths.SLIDES_DB_NAME
    assert not user_dir.exists()
