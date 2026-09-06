"""Managed archive metadata updates never rename or convert the owned file."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from tldw_Server_API.app.api.v1.endpoints import outputs
from tldw_Server_API.app.core.DB_Management.Collections_DB import CollectionsDatabase
from tldw_Server_API.app.services import outputs_service
from tldw_Server_API.app.services.reading_artifact_cleanup_service import provision_reading_storage_namespace
from tldw_Server_API.tests.Collections.test_output_file_operations_db import insert_binding
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


@pytest.fixture(autouse=True)
def existing_output_root(tmp_path, monkeypatch):
    monkeypatch.setattr(outputs_service, "_existing_outputs_dir_for_user", lambda user: tmp_path, raising=False)


def activate(db, root):
    namespace = provision_reading_storage_namespace(root)
    insert_binding(
        db,
        storage_namespace_id=namespace,
        operation_bytes=4096,
        user_pending_bytes=16384,
        text_input_bytes=512,
        text_output_bytes=1024,
        free_space_margin_bytes=1,
    )
    return namespace


@pytest.mark.parametrize("problem", ["late_owner", "source_alias", "destination"])
def test_activated_http_patch_rejects_managed_file_races_without_changing_bytes(
    db, tmp_path, client, monkeypatch, problem
):
    output = db.create_output_artifact(
        type_="reading_archive" if problem == "late_owner" else "report",
        title="Old",
        format_="md",
        storage_path="old.md",
    )
    (tmp_path / "old.md").write_text("original", encoding="utf-8")
    namespace = activate(db, tmp_path)
    payload = {"title": "New"}
    baseline = []
    if problem == "late_owner":
        parent = make_reading(db)
        dispatch = db.update_managed_reading_output

        def register_after_dispatch(*args, **kwargs):
            result = dispatch(*args, **kwargs)
            db.register_reading_output_ownership(
                parent.id, output.id, expected_revision=parent.revision, storage_namespace_id=namespace
            )
            baseline.append(snapshot(db))
            return result

        monkeypatch.setattr(db, "update_managed_reading_output", register_after_dispatch)
    else:
        _, _, owned = archive(db, tmp_path)
        if problem == "source_alias":
            # Pre-existing legacy alias: normal writers already reject new aliases.
            db.backend.execute("UPDATE outputs SET storage_path = ? WHERE id = ?", (owned.storage_path, output.id))
        else:
            payload["title"] = (tmp_path / owned.storage_path).stem
        baseline.append(snapshot(db))
    files = {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}
    before = db.get_output_artifact(output.id)
    response = client.patch(f"/outputs/{output.id}", json=payload)
    assert response.status_code == 409, response.text
    assert db.get_output_artifact(output.id) == before
    assert snapshot(db) == baseline[0]
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()} == files


@pytest.mark.parametrize("convert", [False, True])
def test_activated_http_patch_commits_one_compound_replacement(db, tmp_path, client, monkeypatch, convert):
    activate(db, tmp_path)
    output = db.create_output_artifact(type_="report", title="Old", format_="md", storage_path="old.md")
    (tmp_path / "old.md").write_text("# Body", encoding="utf-8")
    payload = {"title": "New", "retention_until": "2030-01-01T00:00:00"}
    if convert:
        payload["format"] = "html"
    committed = []
    apply_operation = db.apply_output_file_operation

    def capture_commit(token, namespace, **kwargs):
        result = apply_operation(token, namespace, **kwargs)
        committed.append(db.get_output_file_operation(token, namespace))
        return result

    monkeypatch.setattr(db, "apply_output_file_operation", capture_commit)
    response = client.patch(f"/outputs/{output.id}", json=payload)
    assert response.status_code == 200, response.text
    changed = db.get_output_artifact(output.id)
    assert changed.title == "New" and changed.format == ("html" if convert else "md")
    assert "Body" in (tmp_path / changed.storage_path).read_text(encoding="utf-8")
    assert not (tmp_path / "old.md").exists()
    assert len(committed) == 1 and committed[0]["phase"] == "committed"
    assert committed[0]["effects_pending"] == 0
    assert db.backend.execute("SELECT COUNT(*) FROM output_file_operations").scalar == 0
    assert (
        db.backend.execute("SELECT retention_until FROM outputs WHERE id = ?", (output.id,)).scalar
        == payload["retention_until"]
    )


def test_activated_rename_reserves_actual_copy_bytes(db, tmp_path, client):
    activate(db, tmp_path)
    db.backend.execute("UPDATE output_storage_bindings SET user_pending_bytes = 32")
    output = db.create_output_artifact(type_="report", title="Old", format_="md", storage_path="old.md")
    (tmp_path / "old.md").write_bytes(b"seven!!")
    response = client.patch(f"/outputs/{output.id}", json={"title": "New"})
    assert response.status_code == 200, response.text
    assert (tmp_path / "New.md").read_bytes() == b"seven!!"


@pytest.mark.parametrize("limit", ["input", "output"])
def test_activated_conversion_limit_preserves_original_and_aborts_stage(db, tmp_path, client, limit):
    activate(db, tmp_path)
    query = (
        "UPDATE output_storage_bindings SET text_input_bytes = 2"
        if limit == "input"
        else "UPDATE output_storage_bindings SET text_output_bytes = 2"
    )
    db.backend.execute(query)
    output = db.create_output_artifact(type_="report", title="Old", format_="md", storage_path="old.md")
    (tmp_path / "old.md").write_bytes(b"# Body")
    response = client.patch(f"/outputs/{output.id}", json={"title": "New", "format": "html"})
    assert response.status_code == 413, response.text
    assert response.json() == {"detail": "output_size_limit"}
    assert db.get_output_artifact(output.id) == output
    assert (tmp_path / "old.md").read_bytes() == b"# Body" and not (tmp_path / "New.html").exists()
    assert db.backend.execute("SELECT phase FROM output_file_operations").scalar == "aborting"


@pytest.mark.parametrize("payload", [{}, {"format": "md"}, {"title": "OLD"}, {"retention_until": "2030-01-01"}])
def test_activated_metadata_patch_needs_no_volume(db, tmp_path, client, monkeypatch, payload):
    activate(db, tmp_path)
    output = db.create_output_artifact(type_="report", title="Old", format_="md", storage_path="old.md")

    def unavailable(*args, **kwargs):
        pytest.fail("Metadata PATCH must not acquire a volume or file")

    from tldw_Server_API.app.services import outputs_service

    monkeypatch.setattr(outputs_service, "_outputs_dir_for_user", unavailable)
    response = client.patch(f"/outputs/{output.id}", json=payload)
    assert response.status_code == 200, response.text
    assert db.get_output_artifact(output.id).storage_path == "old.md"
    assert db.backend.execute("SELECT COUNT(*) FROM output_file_operations").scalar == 0


def test_activated_patch_preserves_shared_unowned_source(db, tmp_path, client):
    activate(db, tmp_path)
    output = db.create_output_artifact(type_="report", title="Old", format_="md", storage_path="old.md")
    other = db.create_output_artifact(type_="report", title="Other", format_="md", storage_path="old.md")
    (tmp_path / "old.md").write_bytes(b"shared")
    response = client.patch(f"/outputs/{output.id}", json={"title": "New"})
    assert response.status_code == 200, response.text
    assert db.get_output_artifact(other.id) == other
    assert (tmp_path / "old.md").read_bytes() == (tmp_path / "New.md").read_bytes() == b"shared"


def test_activated_patch_never_provisions_a_missing_volume(db, tmp_path, client, monkeypatch):
    activate(db, tmp_path)
    output = db.create_output_artifact(type_="report", title="Old", format_="md", storage_path="old.md")
    (tmp_path / "old.md").write_bytes(b"original")
    missing = tmp_path / "offline"
    monkeypatch.setattr(outputs_service, "_existing_outputs_dir_for_user", lambda user: missing)
    response = client.patch(f"/outputs/{output.id}", json={"title": "New"})
    assert response.status_code == 503, response.text
    assert db.get_output_artifact(output.id) == output
    assert (tmp_path / "old.md").read_bytes() == b"original"
    assert not missing.exists()


def test_activated_conversion_reads_and_writes_multiple_bounded_chunks(db, tmp_path, client):
    activate(db, tmp_path)
    db.backend.execute(
        "UPDATE output_storage_bindings SET operation_bytes = 8388608, user_pending_bytes = 16777216, "
        "text_input_bytes = 2097152, text_output_bytes = 2097152"
    )
    output = db.create_output_artifact(type_="report", title="Old", format_="html", storage_path="old.html")
    body = "x" * (1024 * 1024 + 23)
    (tmp_path / "old.html").write_text("<p>" + body + "</p>", encoding="utf-8")
    response = client.patch(f"/outputs/{output.id}", json={"title": "New", "format": "md"})
    assert response.status_code == 200, response.text
    assert (tmp_path / "New.md").read_text(encoding="utf-8") == body
    assert not (tmp_path / "old.html").exists()


def test_activated_patch_commit_failure_preserves_original_and_sanitizes_error(db, tmp_path, client, monkeypatch):
    activate(db, tmp_path)
    output = db.create_output_artifact(type_="report", title="Old", format_="md", storage_path="old.md")
    (tmp_path / "old.md").write_bytes(b"original")

    def unavailable(*args, **kwargs):
        raise RuntimeError("private body at /secret/location")

    monkeypatch.setattr(db, "apply_output_file_operation", unavailable)
    response = client.patch(f"/outputs/{output.id}", json={"title": "New", "format": "html"})
    assert response.status_code == 409, response.text
    assert response.json() == {"detail": "output_operation_conflict"}
    assert db.get_output_artifact(output.id) == output
    assert (tmp_path / "old.md").read_bytes() == b"original"
    assert db.backend.execute("SELECT phase FROM output_file_operations").scalar == "aborting"


@pytest.mark.parametrize("problem", ["unsupported", "unknown", "ownership_without_binding"])
def test_inconsistent_storage_never_falls_back_to_legacy_patch(db, tmp_path, client, monkeypatch, problem):
    if problem == "ownership_without_binding":
        archive(db, tmp_path)
    else:
        activate(db, tmp_path)
    output = db.create_output_artifact(type_="report", title="Old", format_="md", storage_path="old.md")
    (tmp_path / "old.md").write_bytes(b"original")
    if problem == "unsupported":
        db.backend.execute("UPDATE output_storage_bindings SET protocol_version = 999")
    elif problem == "unknown":

        def unavailable():
            raise RuntimeError("private storage error")

        monkeypatch.setattr(db, "get_output_read_namespace", unavailable)
    response = client.patch(f"/outputs/{output.id}", json={"title": "New"})
    assert response.status_code == 503, response.text
    assert response.json() == {"detail": "output_storage_unavailable"}
    assert db.get_output_artifact(output.id) == output
    assert (tmp_path / "old.md").read_bytes() == b"original"


async def test_cancelled_conversion_cannot_publish_after_renderer_returns(db, tmp_path, monkeypatch):
    activate(db, tmp_path)
    output = db.create_output_artifact(type_="report", title="Old", format_="md", storage_path="old.md")
    (tmp_path / "old.md").write_bytes(b"original")
    started, release, finished = Event(), Event(), Event()

    def render(*args):
        started.set()
        try:
            assert release.wait(10)
            return b"converted"
        finally:
            finished.set()

    monkeypatch.setattr(outputs_service, "_convert_protected_output_text", render)
    task = asyncio.create_task(
        outputs_service.update_protected_output(
            db,
            int(db.user_id),
            output,
            title="New",
            format_="html",
        )
    )
    try:
        assert await asyncio.to_thread(started.wait, 10)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        release.set()
        assert await asyncio.to_thread(finished.wait, 10)
    assert db.get_output_artifact(output.id) == output
    assert (tmp_path / "old.md").read_bytes() == b"original"
    assert not (tmp_path / "New.html").exists()
    assert db.backend.execute("SELECT phase FROM output_file_operations").scalar == "aborting"


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
