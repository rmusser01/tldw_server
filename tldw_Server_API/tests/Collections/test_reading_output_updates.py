"""Managed archive metadata updates never rename or convert the owned file."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from tldw_Server_API.app.api.v1.endpoints import outputs
from tldw_Server_API.app.core.DB_Management.Collections_DB import CollectionsDatabase
from tldw_Server_API.tests.Collections.test_reading_atomic_delete import snapshot
from tldw_Server_API.tests.Collections.test_reading_output_deletion import archive
from tldw_Server_API.tests.Collections.test_reading_output_disposal_routes import client as client
from tldw_Server_API.tests.Collections.test_reading_revision_mutations import db as db
from tldw_Server_API.tests.Collections.test_reading_revision_mutations import (
    make_archive_output,
    make_reading,
    mutate_output,
)

pytestmark = pytest.mark.unit
pytest_plugins = ["tldw_Server_API.tests._plugins.authnz_full_fixtures"]


@pytest.mark.parametrize("operation", ["rename", "service", "path", "format"])
def test_managed_db_file_changes_reject_entire_update(db, tmp_path, operation):
    _, _, output = archive(db, tmp_path)
    before = snapshot(db)
    with pytest.raises(RuntimeError, match="^reading_archive_file_immutable$"):
        if operation in {"rename", "service"}:
            mutate_output(db, output.id, operation)
        else:
            db.update_output_artifact(
                output.id,
                title="Must roll back",
                storage_path="changed.md" if operation == "path" else None,
                format_="html" if operation == "format" else None,
                retention_until="2030-01-01T00:00:00",
            )
    assert snapshot(db) == before
    assert (tmp_path / output.storage_path).exists()


def test_managed_db_unchanged_file_fields_are_noops(db, tmp_path):
    _, _, output = archive(db, tmp_path)
    before = snapshot(db)
    assert db.update_output_artifact(output.id, storage_path=output.storage_path, format_=output.format) == output
    assert snapshot(db) == before


def test_managed_dispatch_uses_one_explicit_connection(db, tmp_path, monkeypatch):
    parent, _, output = archive(db, tmp_path)
    statements = []
    connections = []
    execute = db.backend.execute

    def trace(query, params=None, *, connection=None, **kwargs):
        statements.append(query)
        connections.append(connection)
        return execute(query, params, connection=connection, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(db.backend, "execute", trace)
        changed = db.update_managed_reading_output(output.id, title="Changed", format_=output.format)
    assert changed.title == "Changed"
    assert "UPDATE" in statements[0] and "reading_revision_clock" in statements[0]
    assert connections[0] is not None
    assert all(conn is connections[0] for conn in connections)
    assert db.get_content_item(parent.id).revision == parent.revision + 1


def test_db_rechecks_late_ownership_before_file_field_update(db, monkeypatch):
    parent = make_reading(db)
    output = make_archive_output(db)
    registered = []
    other = CollectionsDatabase.from_backend(user_id=db.user_id, backend=db.backend)

    # Run before the transaction starts: SQLite BEGIN IMMEDIATE already owns
    # its write fence. This models ownership changing since the caller's read.
    resolve = db.resolve_output_storage_path

    def resolve_after_registration(path):
        with ThreadPoolExecutor(max_workers=1) as workers:
            workers.submit(
                other.register_reading_output_ownership,
                parent.id,
                output.id,
                expected_revision=parent.revision,
                storage_namespace_id="test-volume",
            ).result(timeout=15)
        registered.append(snapshot(other))
        return resolve(path)

    monkeypatch.setattr(db, "resolve_output_storage_path", resolve_after_registration)
    with pytest.raises(RuntimeError, match="^reading_archive_file_immutable$"):
        db.update_output_artifact(output.id, title="No", storage_path="no.md")
    assert snapshot(db) == registered[0]


@pytest.mark.parametrize("same_format", [False, True])
def test_http_managed_title_retention_update_never_resolves_files(db, tmp_path, client, monkeypatch, same_format):
    parent, _, output = archive(db, tmp_path)
    path = tmp_path / output.storage_path
    contents = path.read_bytes()

    def unexpected_path_access(*_args, **_kwargs):
        pytest.fail("Managed metadata update must not normalize or resolve a file path")

    monkeypatch.setattr(outputs, "_normalize_output_storage_path_for_user", unexpected_path_access)
    monkeypatch.setattr(outputs, "_resolve_output_path_for_user", unexpected_path_access)
    payload = {"title": "Readable display title", "retention_until": "2030-01-01T00:00:00"}
    if same_format:
        payload["format"] = output.format
    response = client.patch(f"/outputs/{output.id}", json=payload)
    assert response.status_code == 200, response.text
    assert response.json()["title"] == payload["title"]
    assert response.json()["storage_path"] == output.storage_path
    assert db.get_content_item(parent.id).revision == parent.revision + 1
    assert path.read_bytes() == contents
    before_replay = snapshot(db)
    assert client.patch(f"/outputs/{output.id}", json=payload).status_code == 200
    assert snapshot(db) == before_replay


@pytest.mark.parametrize("compound", [False, True])
def test_http_managed_conversion_rejects_before_any_mutation(db, tmp_path, client, compound):
    _, _, output = archive(db, tmp_path)
    before = snapshot(db)
    paths = {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}
    payload = {"format": "html"}
    if compound:
        payload.update(title="Must not rename", retention_until="2030-01-01T00:00:00")
    response = client.patch(f"/outputs/{output.id}", json=payload)
    assert response.status_code == 409, response.text
    assert response.json() == {"detail": "reading_archive_file_immutable"}
    assert snapshot(db) == before
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()} == paths


def test_http_managed_update_rollback_preserves_file_and_sanitizes_error(db, tmp_path, client, monkeypatch):
    _, _, output = archive(db, tmp_path)
    before = snapshot(db)
    path = tmp_path / output.storage_path
    contents = path.read_bytes()
    advance = db._advance_reading_parent
    logged = []

    def fail(item_id, conn):
        advance(item_id, conn)
        raise RuntimeError("secret title at /private/output.md")

    monkeypatch.setattr(db, "_advance_reading_parent", fail)
    monkeypatch.setattr(outputs.logger, "error", lambda message: logged.append(message))
    response = client.patch(f"/outputs/{output.id}", json={"title": "Changed"})
    assert response.status_code == 409
    assert response.json() == {"detail": "conflict_on_update"}
    assert snapshot(db) == before
    assert path.read_bytes() == contents
    assert logged == ["outputs update failed"]


@pytest.mark.parametrize("case", ["missing", "foreign", "deleted"])
def test_http_inaccessible_managed_output_is_nonmutating_404(db, tmp_path, client, case):
    _, _, output = archive(db, tmp_path)
    output_id = output.id
    if case == "missing":
        output_id += 10000
    elif case == "deleted":
        db.delete_output_artifact(output_id)
    else:
        client.app.dependency_overrides[outputs.get_collections_db_for_user] = lambda: CollectionsDatabase.from_backend(
            user_id="781", backend=db.backend
        )
    before = snapshot(db)
    response = client.patch(f"/outputs/{output_id}", json={"title": "No", "format": "html"})
    assert response.status_code == 404, response.text
    assert snapshot(db) == before
    assert (tmp_path / output.storage_path).exists()


@pytest.mark.parametrize("type_", ["reading_archive", "newsletter_markdown"])
@pytest.mark.parametrize("convert", [False, True])
def test_http_unowned_rename_and_conversion_keep_existing_behavior(db, tmp_path, client, type_, convert):
    output = db.create_output_artifact(type_=type_, title="Old", format_="md", storage_path="old.md")
    path = tmp_path / output.storage_path
    path.write_text("# Body", encoding="utf-8")
    payload = {"title": "New"}
    if convert:
        payload["format"] = "html"
    response = client.patch(f"/outputs/{output.id}", json=payload)
    assert response.status_code == 200, response.text
    changed = db.get_output_artifact(output.id)
    assert changed.title == "New"
    assert changed.format == ("html" if convert else "md")
    assert not path.exists()
    assert "Body" in (tmp_path / changed.storage_path).read_text(encoding="utf-8")
    assert db.backend.execute("SELECT COUNT(*) FROM reading_output_ownership", ()).scalar == 0
