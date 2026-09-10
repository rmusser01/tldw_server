import asyncio
import hashlib
import importlib.util
import json
import re
import sqlite3

import pytest

from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.Slides.slides_db import (
    ConflictError,
    InputError,
    SlidesDatabase,
    SlidesDatabaseError,
    decode_presentation_version_payload,
)
from tldw_Server_API.app.core.Slides.standalone_html_validator import (
    validate_standalone_html,
)


def _valid_html(*, title: str = "Standalone", body_text: str = "Visible search text") -> str:
    return (
        '<!doctype html><html><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{title}</title><style>.slide{{color:#111}}</style></head>"
        '<body><header class="deck-header">Chrome text</header>'
        f'<section class="slide"><h1>{body_text}</h1>'
        '<aside class="notes">Private speaker note</aside></section>'
        "<script>document.addEventListener('keydown', () => {});</script>"
        "</body></html>"
    )


def _provenance() -> dict:
    return {
        "schema_version": 1,
        "source_kind": "prompt",
        "source_ref": None,
        "source_snapshot_hmac_sha256": "a" * 64,
        "digest_key_id": "slides-generation-v1",
        "source_bytes": 10,
        "provider": "openai",
        "model": "test-model",
        "adapter_id": "openai_official_chat_v1",
        "endpoint_identity": "https://api.openai.com:443/v1/chat/completions",
        "prompt_sha256": "b" * 64,
    }


class _InlineValidationPool:
    def __init__(self, db: SlidesDatabase) -> None:
        self.db = db
        self.calls: list[str | bytes] = []

    async def validate(self, document: str | bytes):
        assert not self.db.get_connection().in_transaction
        self.calls.append(document)
        return validate_standalone_html(document)


def _run(awaitable):
    return asyncio.run(awaitable)


def _service(db: SlidesDatabase, pool: _InlineValidationPool | None = None):
    assert (
        importlib.util.find_spec("tldw_Server_API.app.core.Slides.presentation_service") is not None
    ), "Task 3 requires the shared presentation_service seam"
    from tldw_Server_API.app.core.Slides.presentation_service import PresentationService

    return PresentationService(db, validation_pool=pool or _InlineValidationPool(db))


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
    assert versions[0].title == "Standalone"
    assert versions[0].deleted == 0
    assert not hasattr(summaries[0], "html_document")
    select_sql = "\n".join(
        statement for statement in statements if statement.lstrip().upper().startswith("SELECT")
    ).lower()
    assert "html_document" not in select_sql
    assert "payload_json" not in select_sql
    db.close_connection()


def test_standalone_source_identity_projection_is_source_free(tmp_path):
    db = SlidesDatabase(db_path=tmp_path / "Slides.db", client_id="tester")
    html = _create_standalone(db)
    statements: list[str] = []
    db.get_connection().set_trace_callback(statements.append)

    identity = db.get_presentation_source_identity(html.id)

    assert identity.id == html.id
    assert identity.content_kind == "standalone_html"
    assert identity.version == html.version
    assert identity.title == html.title
    assert identity.html_sha256 == html.html_sha256
    assert identity.html_bytes == html.html_bytes
    select_sql = "\n".join(
        statement for statement in statements if statement.lstrip().upper().startswith("SELECT")
    ).lower()
    assert "html_document" not in select_sql
    assert "payload_json" not in select_sql
    assert "generation_provenance_json" not in select_sql
    assert re.search(r"\bslides\b", select_sql) is None
    db.close_connection()


def test_health_probe_executes_only_a_source_free_existence_query(tmp_path):
    db = SlidesDatabase(db_path=tmp_path / "Slides.db", client_id="tester")
    _create_standalone(db)
    statements: list[str] = []
    db.get_connection().set_trace_callback(statements.append)

    db.probe_health()

    select_sql = [statement.lower() for statement in statements if statement.lstrip().upper().startswith("SELECT")]
    assert select_sql == ["select 1 from presentations limit 1"]
    assert "html_document" not in select_sql[0]
    assert "slides" not in select_sql[0]
    db.close_connection()


def test_health_probe_normalizes_sqlite_failure_without_sensitive_exception_chain(
    tmp_path,
    monkeypatch,
):
    db = SlidesDatabase(db_path=tmp_path / "Slides.db", client_id="tester")
    sentinel = "SECRET-/private/slides-health.db"

    class _FailingConnection:
        def execute(self, _query: str):
            raise sqlite3.OperationalError(sentinel)

    monkeypatch.setattr(db, "get_connection", lambda: _FailingConnection())

    with pytest.raises(SlidesDatabaseError, match="^slides_health_probe_failed$") as exc_info:
        db.probe_health()

    chain = [exc_info.value]
    while chain[-1].__cause__ is not None or chain[-1].__context__ is not None:
        chain.append(chain[-1].__cause__ or chain[-1].__context__)
    assert sentinel not in " ".join(repr(exc) for exc in chain)
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


def test_summary_kind_filtering_happens_before_count_and_pagination(tmp_path):
    db = SlidesDatabase(db_path=tmp_path / "Slides.db", client_id="tester")
    _create_structured(db, presentation_id="z-structured")
    _create_standalone(db, presentation_id="a-html")
    statements: list[str] = []
    db.get_connection().set_trace_callback(statements.append)

    html_rows, html_total = db.list_presentation_summaries(
        limit=1,
        offset=0,
        include_deleted=False,
        sort_column="title",
        sort_direction="ASC",
        accepted_content_kinds=frozenset({"standalone_html"}),
    )
    structured_rows, structured_total = db.list_presentation_summaries(
        limit=1,
        offset=0,
        include_deleted=False,
        sort_column="title",
        sort_direction="ASC",
        accepted_content_kinds=frozenset({"structured_slides"}),
    )
    both_rows, both_total = db.list_presentation_summaries(
        limit=1,
        offset=0,
        include_deleted=False,
        sort_column="title",
        sort_direction="ASC",
        accepted_content_kinds=frozenset({"structured_slides", "standalone_html"}),
    )

    assert html_total == structured_total == 1
    assert html_rows[0].content_kind == "standalone_html"
    assert structured_rows[0].content_kind == "structured_slides"
    assert both_total == 2 and len(both_rows) == 1
    sql = "\n".join(statements).lower()
    assert sql.count("content_kind") >= 6
    db.close_connection()


def test_shared_service_derives_html_metadata_and_fts_without_hidden_text(tmp_path):
    db = SlidesDatabase(db_path=tmp_path / "Slides.db", client_id="tester")
    service = _service(db)
    document = _valid_html(title="Cafe\u0301", body_text="Needle visible")

    row = service.create_standalone_for_worker(
        presentation_id="html",
        html_document=document,
        validation_result=validate_standalone_html(document),
        generation_job_uuid="job-html",
        generation_provenance=_provenance(),
    )

    assert row.title == "Caf\u00e9"
    assert row.html_bytes == len(document.encode("utf-8"))
    assert row.html_sha256 == hashlib.sha256(document.encode("utf-8")).hexdigest()
    assert row.html_slide_count == 1
    assert row.slides_text == "Needle visible"
    found, total = db.search_presentation_summaries(
        query="Needle",
        limit=10,
        offset=0,
        include_deleted=False,
        accepted_content_kinds=frozenset({"standalone_html"}),
    )
    assert total == 1 and found[0].id == row.id
    for hidden in ("Chrome", "Private", "script", "style"):
        hidden_rows, hidden_total = db.search_presentation_summaries(
            query=hidden,
            limit=10,
            offset=0,
            include_deleted=False,
            accepted_content_kinds=frozenset({"standalone_html"}),
        )
        assert hidden_total == 0 and hidden_rows == []
    db.close_connection()


def test_html_source_save_is_noop_for_exact_source_and_snapshots_changed_source(tmp_path):
    db = SlidesDatabase(
        db_path=tmp_path / "Slides.db",
        client_id="tester",
        standalone_html_version_retention=2,
    )
    service = _service(db)
    original = _valid_html(title="One", body_text="first")
    created = service.create_standalone_for_worker(
        presentation_id="html",
        html_document=original,
        validation_result=validate_standalone_html(original),
        generation_job_uuid="job-html",
        generation_provenance=_provenance(),
    )

    same = _run(
        service.save_html_source(
            presentation_id=created.id,
            html_document=original,
            expected_version=created.version,
        )
    )
    assert same.version == created.version
    assert db.list_presentation_version_metadata(presentation_id=created.id, limit=10, offset=0)[1] == 1

    changed = _run(
        service.save_html_source(
            presentation_id=created.id,
            html_document=_valid_html(title="Deux", body_text="second"),
            expected_version=created.version,
        )
    )
    latest = _run(
        service.save_html_source(
            presentation_id=created.id,
            html_document=_valid_html(title="三", body_text="third"),
            expected_version=changed.version,
        )
    )
    versions, total = db.list_presentation_version_metadata(presentation_id=created.id, limit=10, offset=0)
    assert latest.version == 3
    assert total == 2
    assert [item.version for item in versions] == [3, 2]
    with pytest.raises(KeyError, match="presentation_version_not_found"):
        db.get_presentation_version(presentation_id=created.id, version=1)

    payload = json.loads(db.get_presentation_version(presentation_id=created.id, version=latest.version).payload_json)
    assert payload["snapshot_schema_version"] == 1
    assert payload["content_kind"] == "standalone_html"
    assert payload["html_document"].startswith("<!doctype html>")
    assert "slides" not in payload
    assert "\\u4e09" not in json.dumps(payload, ensure_ascii=False)
    assert len(json.dumps(payload, ensure_ascii=False).encode("utf-8")) <= (2 * 1_048_576 + 65_536)
    db.close_connection()


def test_exact_source_save_repairs_corrupt_derived_metadata_in_new_version(tmp_path):
    db = SlidesDatabase(db_path=tmp_path / "Slides.db", client_id="tester")
    service = _service(db)
    document = _valid_html(title="Canonical", body_text="searchable")
    created = service.create_standalone_for_worker(
        presentation_id="html",
        html_document=document,
        validation_result=validate_standalone_html(document),
        generation_job_uuid="job-html",
        generation_provenance=_provenance(),
    )
    with db.transaction(immediate=True) as conn:
        conn.execute(
            "UPDATE presentations SET title = ?, html_sha256 = ?, html_bytes = ?, "
            "html_slide_count = ?, slides_text = ? WHERE id = ?",
            ("Forged", "0" * 64, 1, 30, "forged", created.id),
        )

    repaired = _run(
        service.save_html_source(
            presentation_id=created.id,
            html_document=document,
            expected_version=created.version,
        )
    )
    derived = validate_standalone_html(document)

    assert repaired.version == created.version + 1
    assert (
        repaired.title,
        repaired.html_sha256,
        repaired.html_bytes,
        repaired.html_slide_count,
        repaired.slides_text,
    ) == (
        derived.title,
        derived.html_sha256,
        derived.html_bytes,
        derived.slide_count,
        derived.indexable_text,
    )
    db.close_connection()


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("content_kind", "structured_slides", "content_kind_immutable"),
        ("generation_job_uuid", "other-job", "generation_job_uuid_immutable"),
        (
            "generation_provenance_json",
            json.dumps({"source_kind": "media"}),
            "generation_provenance_immutable",
        ),
    ],
)
def test_generic_mutation_cannot_change_html_kind_or_generation_identity(tmp_path, field, value, match):
    db = SlidesDatabase(db_path=tmp_path / "Slides.db", client_id="tester")
    row = _create_standalone(db)

    with pytest.raises(InputError, match=match):
        db.update_presentation(
            presentation_id=row.id,
            update_fields={field: value},
            expected_version=row.version,
        )

    unchanged = db.get_presentation_by_id(row.id)
    assert unchanged.version == row.version
    assert unchanged.content_kind == "standalone_html"
    assert unchanged.generation_job_uuid == "job-uuid-1"
    db.close_connection()


@pytest.mark.parametrize(
    "update_fields",
    [
        {"title": "Forged title"},
        {"slides_text": "forged index text"},
        {"html_sha256": "0" * 64},
        {"html_document": _valid_html(title="Forged source")},
        {"deleted": 1},
    ],
)
def test_generic_mutation_rejects_all_standalone_content_changes(tmp_path, update_fields):
    db = SlidesDatabase(db_path=tmp_path / "Slides.db", client_id="tester")
    row = _create_standalone(db)

    with pytest.raises(InputError, match="operation_not_supported_for_content_kind"):
        db.update_presentation(
            presentation_id=row.id,
            update_fields=update_fields,
            expected_version=row.version,
        )

    assert db.get_presentation_by_id(row.id) == row
    db.close_connection()


def test_generic_mutation_keeps_only_standalone_delete_restore_transition(tmp_path):
    db = SlidesDatabase(db_path=tmp_path / "Slides.db", client_id="tester")
    row = _create_standalone(db)

    deleted = db.soft_delete_presentation(row.id, row.version)
    restored = db.restore_presentation(row.id, deleted.version)

    assert deleted.deleted == 1
    assert restored.deleted == 0
    assert restored.version == row.version + 2
    db.close_connection()


def test_soft_restore_rechecks_exact_source_after_pool_validation(tmp_path):
    db = SlidesDatabase(db_path=tmp_path / "Slides.db", client_id="tester")
    original = _valid_html(title="Original", body_text="original")
    created = _service(db).create_standalone_for_worker(
        presentation_id="standalone",
        html_document=original,
        validation_result=validate_standalone_html(original),
        generation_job_uuid="job-uuid-1",
        generation_provenance=_provenance(),
    )
    deleted = db.soft_delete_presentation(created.id, created.version)
    replacement = _valid_html(title="Replacement", body_text="replacement")
    replacement_result = validate_standalone_html(replacement)

    class _StoredSourceChangingPool(_InlineValidationPool):
        async def validate(self, document: str | bytes):
            result = await super().validate(document)
            with self.db.transaction(immediate=True) as conn:
                conn.execute(
                    """
                    UPDATE presentations
                    SET title = ?, html_document = ?, html_sha256 = ?,
                        html_bytes = ?, html_slide_count = ?, slides_text = ?
                    WHERE id = ?
                    """,
                    (
                        replacement_result.title,
                        replacement,
                        replacement_result.html_sha256,
                        replacement_result.html_bytes,
                        replacement_result.slide_count,
                        replacement_result.indexable_text,
                        created.id,
                    ),
                )
            return result

    service = _service(db, _StoredSourceChangingPool(db))

    with pytest.raises(InputError, match="standalone_html_response_invalid"):
        _run(
            service.restore_presentation(
                presentation_id=created.id,
                expected_version=deleted.version,
            )
        )

    current = db.get_presentation_by_id(created.id, include_deleted=True)
    assert current.html_document == replacement
    assert current.deleted == 1
    assert current.version == deleted.version
    db.close_connection()


def test_soft_restore_keeps_fts_snapshot_and_sync_behavior(tmp_path):
    db = SlidesDatabase(db_path=tmp_path / "Slides.db", client_id="tester")
    pool = _InlineValidationPool(db)
    service = _service(db, pool)
    source = _valid_html(title="Restorable", body_text="restorable phrase")
    created = service.create_standalone_for_worker(
        presentation_id="standalone",
        html_document=source,
        validation_result=validate_standalone_html(source),
        generation_job_uuid="job-uuid-1",
        generation_provenance=_provenance(),
    )
    deleted = service.delete_presentation(
        presentation_id=created.id,
        expected_version=created.version,
    )

    restored = _run(
        service.restore_presentation(
            presentation_id=created.id,
            expected_version=2,
        )
    )

    assert isinstance(deleted, dict)
    assert restored.deleted == 0
    assert restored.version == 3
    assert pool.calls == [source]
    found, total = db.search_presentation_summaries(
        query="restorable",
        limit=10,
        offset=0,
        include_deleted=False,
    )
    assert total == 1
    assert found[0].id == created.id
    snapshot = json.loads(
        db.get_presentation_version(
            presentation_id=created.id,
            version=restored.version,
        ).payload_json
    )
    assert snapshot["title"] == "Restorable"
    assert snapshot["deleted"] == 0
    latest_sync = (
        db.get_connection()
        .execute(
            """
        SELECT operation, version FROM sync_log
        WHERE entity_uuid = ? ORDER BY change_id DESC LIMIT 1
        """,
            (created.id,),
        )
        .fetchone()
    )
    assert tuple(latest_sync) == ("restore", restored.version)
    db.close_connection()


def test_operation_error_preserves_bounded_actual_future_kind(tmp_path):
    db = SlidesDatabase(db_path=tmp_path / "Slides.db", client_id="tester")
    service = _service(db)

    error = service.operation_not_supported("future_kind", "read")

    assert error.code == "operation_not_supported_for_content_kind"
    assert error.operation == "read"
    assert error.content_kind == "future_kind"
    db.close_connection()


def test_validated_source_result_must_match_before_atomic_save(tmp_path):
    db = SlidesDatabase(db_path=tmp_path / "Slides.db", client_id="tester")
    row = _create_standalone(db)
    source = _valid_html(title="Candidate")
    wrong_result = validate_standalone_html(_valid_html(title="Different"))

    with pytest.raises(InputError, match="standalone_html_validation_result_mismatch"):
        db.save_standalone_html_source(
            presentation_id=row.id,
            html_document=source,
            validation_result=wrong_result,
            expected_version=row.version,
        )

    assert db.get_presentation_by_id(row.id) == row
    db.close_connection()


def test_generation_job_uuid_conflict_is_not_confused_with_primary_key_conflict(tmp_path):
    db = SlidesDatabase(db_path=tmp_path / "Slides.db", client_id="tester")
    _create_standalone(db, presentation_id="first", generation_job_uuid="job-one")

    with pytest.raises(ConflictError, match="generation_job_uuid_conflict"):
        _create_standalone(db, presentation_id="second", generation_job_uuid="job-one")
    with pytest.raises(ConflictError, match="presentation already exists"):
        _create_standalone(db, presentation_id="first", generation_job_uuid="job-two")
    db.close_connection()


def test_saved_standalone_invariant_failure_is_fixed_source_free_and_skips_pool(tmp_path):
    from tldw_Server_API.app.core.Slides.presentation_service import (
        PresentationServiceError,
    )

    db = SlidesDatabase(db_path=tmp_path / "Slides.db", client_id="tester")
    created = _create_standalone(db)
    sentinel = "SECRET-MALFORMED-PROVENANCE"
    with db.transaction(immediate=True) as conn:
        conn.execute(
            "UPDATE presentations SET generation_provenance_json = ? WHERE id = ?",
            ('{"private":"' + sentinel, created.id),
        )
    pool = _InlineValidationPool(db)
    service = _service(db, pool)

    with pytest.raises(PresentationServiceError) as exc_info:
        _run(service.validate_saved_standalone(db.get_presentation_by_id(created.id)))

    error = exc_info.value
    assert getattr(error, "code", None) == "standalone_html_response_invalid"
    assert getattr(error, "status_code", None) == 500
    chain = [error]
    while chain[-1].__cause__ is not None or chain[-1].__context__ is not None:
        chain.append(chain[-1].__cause__ or chain[-1].__context__)
    assert sentinel not in " ".join(repr(exc) for exc in chain)
    assert pool.calls == []
    db.close_connection()


def test_saved_invariant_rejects_oversize_provenance_before_json_decode(tmp_path, monkeypatch):
    from tldw_Server_API.app.core.Slides import slides_db as slides_db_module

    db = SlidesDatabase(db_path=tmp_path / "Slides.db", client_id="tester")
    created = _create_standalone(db)
    oversized = json.dumps({"private": "x" * 4096})
    with db.transaction(immediate=True) as conn:
        conn.execute(
            "UPDATE presentations SET generation_provenance_json = ? WHERE id = ?",
            (oversized, created.id),
        )
    row = db.get_presentation_by_id(created.id)
    real_loads = json.loads

    def _guarded_loads(value):
        if value == oversized:
            raise AssertionError("oversize provenance reached JSON decoder")
        return real_loads(value)

    monkeypatch.setattr(slides_db_module.json, "loads", _guarded_loads)

    assert db.presentation_row_invariant_holds(row) is False
    db.close_connection()


def test_restore_html_snapshot_is_same_kind_atomic_and_preserves_generation_identity(tmp_path):
    db_path = tmp_path / "Slides.db"
    db = SlidesDatabase(db_path=db_path, client_id="owner-client")
    service = _service(db)
    created = service.create_standalone_for_worker(
        presentation_id="html",
        html_document=(original := _valid_html(title="Original", body_text="first")),
        validation_result=validate_standalone_html(original),
        generation_job_uuid="job-html",
        generation_provenance=_provenance(),
    )
    changed = _run(
        service.save_html_source(
            presentation_id=created.id,
            html_document=_valid_html(title="Changed", body_text="second"),
            expected_version=created.version,
        )
    )
    db.close_connection()
    db = SlidesDatabase(db_path=db_path, client_id="restoring-client")
    service = _service(db)

    restored = _run(
        service.restore_version(
            presentation_id=created.id,
            version=1,
            expected_version=changed.version,
        )
    )

    assert restored.title == "Original"
    assert restored.version == 3
    assert restored.id == created.id
    assert restored.client_id == "owner-client"
    assert restored.created_at == created.created_at
    assert restored.content_kind == "standalone_html"
    assert restored.generation_job_uuid == "job-html"
    assert restored.generation_provenance_json == created.generation_provenance_json

    version_two = db.get_presentation_version(presentation_id=created.id, version=2)
    corrupt = json.loads(version_two.payload_json)
    corrupt["content_kind"] = "structured_slides"
    with db.transaction(immediate=True) as conn:
        conn.execute(
            "UPDATE presentations_versions SET payload_json = ? " "WHERE presentation_id = ? AND version = ?",
            (json.dumps(corrupt), created.id, 2),
        )
    with pytest.raises(InputError, match="version_content_kind_mismatch"):
        _run(
            service.restore_version(
                presentation_id=created.id,
                version=2,
                expected_version=restored.version,
            )
        )
    assert db.get_presentation_by_id(created.id).version == restored.version
    db.close_connection()


def test_restore_rechecks_exact_snapshot_after_out_of_transaction_validation(tmp_path):
    db = SlidesDatabase(db_path=tmp_path / "Slides.db", client_id="tester")
    setup_service = _service(db)
    original = _valid_html(title="Original", body_text="first")
    created = setup_service.create_standalone_for_worker(
        presentation_id="html",
        html_document=original,
        validation_result=validate_standalone_html(original),
        generation_job_uuid="job-html",
        generation_provenance=_provenance(),
    )
    changed = _run(
        setup_service.save_html_source(
            presentation_id=created.id,
            html_document=_valid_html(title="Changed", body_text="second"),
            expected_version=created.version,
        )
    )

    class _SnapshotChangingPool(_InlineValidationPool):
        async def validate(self, document: str | bytes):
            result = await super().validate(document)
            with self.db.transaction(immediate=True) as conn:
                conn.execute(
                    "UPDATE presentations_versions SET payload_json = payload_json || ' ' "
                    "WHERE presentation_id = ? AND version = 1",
                    (created.id,),
                )
            return result

    service = _service(db, _SnapshotChangingPool(db))
    with pytest.raises(InputError, match="version_payload_invalid"):
        _run(
            service.restore_version(
                presentation_id=created.id,
                version=1,
                expected_version=changed.version,
            )
        )

    assert db.get_presentation_by_id(created.id).version == changed.version
    db.close_connection()


def test_malformed_snapshot_failure_retains_no_source_exception_context(tmp_path):
    db = SlidesDatabase(db_path=tmp_path / "Slides.db", client_id="tester")
    service = _service(db)
    document = _valid_html()
    created = service.create_standalone_for_worker(
        presentation_id="html",
        html_document=document,
        validation_result=validate_standalone_html(document),
        generation_job_uuid="job-html",
        generation_provenance=_provenance(),
    )
    sentinel = "SECRET-SNAPSHOT-SOURCE"
    with db.transaction(immediate=True) as conn:
        conn.execute(
            "UPDATE presentations_versions SET payload_json = ? " "WHERE presentation_id = ? AND version = 1",
            ('{"html_document":"' + sentinel, created.id),
        )

    with pytest.raises(InputError, match="version_payload_invalid") as exc_info:
        _run(
            service.restore_version(
                presentation_id=created.id,
                version=1,
                expected_version=created.version,
            )
        )

    chain = [exc_info.value]
    while chain[-1].__cause__ is not None or chain[-1].__context__ is not None:
        chain.append(chain[-1].__cause__ or chain[-1].__context__)
    assert not any(isinstance(exc, json.JSONDecodeError) for exc in chain)
    assert sentinel not in " ".join(repr(exc) for exc in chain)
    db.close_connection()


def test_recursive_snapshot_decoder_uses_fixed_source_free_error():
    sentinel = "SECRET-RECURSIVE-SNAPSHOT"
    payload_json = '{"html_document":"' + sentinel + '","nested":' + "[" * 1100 + "0" + "]" * 1100 + "}"
    assert len(payload_json.encode("utf-8")) < 4096

    with pytest.raises(InputError, match="^version_payload_invalid$") as exc_info:
        decode_presentation_version_payload(payload_json)

    chain = [exc_info.value]
    while chain[-1].__cause__ is not None or chain[-1].__context__ is not None:
        chain.append(chain[-1].__cause__ or chain[-1].__context__)
    assert not any(isinstance(exc, RecursionError) for exc in chain)
    assert sentinel not in " ".join(repr(exc) for exc in chain)


def test_default_standalone_snapshot_retention_is_25(tmp_path):
    db = SlidesDatabase(db_path=tmp_path / "Slides.db", client_id="tester")
    service = _service(db)
    document = _valid_html(title="Version 1")
    row = service.create_standalone_for_worker(
        presentation_id="html",
        html_document=document,
        validation_result=validate_standalone_html(document),
        generation_job_uuid="job-html",
        generation_provenance=_provenance(),
    )

    for version in range(2, 27):
        row = _run(
            service.save_html_source(
                presentation_id=row.id,
                html_document=_valid_html(title=f"Version {version}"),
                expected_version=row.version,
            )
        )

    versions, total = db.list_presentation_version_metadata(
        presentation_id=row.id,
        limit=30,
        offset=0,
    )
    assert total == 25
    assert [item.version for item in versions] == list(range(26, 1, -1))
    db.close_connection()


def test_snapshot_ceiling_failure_rolls_back_source_update(tmp_path, monkeypatch):
    from tldw_Server_API.app.core.Slides import slides_db as slides_db_module

    db = SlidesDatabase(db_path=tmp_path / "Slides.db", client_id="tester")
    service = _service(db)
    original = _valid_html(title="Original")
    created = service.create_standalone_for_worker(
        presentation_id="html",
        html_document=original,
        validation_result=validate_standalone_html(original),
        generation_job_uuid="job-html",
        generation_provenance=_provenance(),
    )
    monkeypatch.setattr(slides_db_module, "_STANDALONE_HTML_SNAPSHOT_MAX_BYTES", 128)

    with pytest.raises(InputError, match="standalone_html_storage_limit"):
        _run(
            service.save_html_source(
                presentation_id=created.id,
                html_document=_valid_html(title="Changed"),
                expected_version=created.version,
            )
        )

    current = db.get_presentation_by_id(created.id)
    assert current.html_document == original
    assert current.version == created.version
    assert (
        db.list_presentation_version_metadata(
            presentation_id=created.id,
            limit=10,
            offset=0,
        )[1]
        == 1
    )
    db.close_connection()


def test_html_delete_returns_metadata_tombstone_and_preserves_snapshot_semantics(tmp_path):
    db = SlidesDatabase(db_path=tmp_path / "Slides.db", client_id="tester")
    service = _service(db)
    created = service.create_standalone_for_worker(
        presentation_id="html",
        html_document=(document := _valid_html()),
        validation_result=validate_standalone_html(document),
        generation_job_uuid="job-html",
        generation_provenance=_provenance(),
    )
    tombstone = service.delete_presentation(
        presentation_id=created.id,
        expected_version=created.version,
    )

    assert tombstone == {
        "id": created.id,
        "content_kind": "standalone_html",
        "deleted_at": tombstone["deleted_at"],
    }
    versions, total = db.list_presentation_version_metadata(presentation_id=created.id, limit=10, offset=0)
    assert total == 2
    assert [row.version for row in versions] == [2, 1]
    db.close_connection()
