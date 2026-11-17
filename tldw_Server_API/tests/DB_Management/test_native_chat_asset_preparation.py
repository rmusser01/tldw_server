"""Real validated uploads cannot become prepared native assets by filename alone."""

from __future__ import annotations

import asyncio
import hashlib
import io
import threading

import pytest
from PIL import Image
from starlette.datastructures import UploadFile

from tldw_Server_API.app.core.Ingestion_Media_Processing import Upload_Sink as sink
from tldw_Server_API.tests.DB_Management.test_native_chat_asset_io import (
    _expire,
    _io,
    _reserve,
    owner,
)

pytest_plugins = ("tldw_Server_API.tests.DB_Management.test_native_chat_asset_io",)
pytestmark = pytest.mark.integration


def png() -> bytes:
    stream = io.BytesIO()
    Image.new("RGB", (2, 2), "red").save(stream, format="PNG")
    return stream.getvalue()


def pdf() -> bytes:
    import fitz

    with fitz.open() as document:
        document.new_page().insert_text((20, 20), "Original document")
        return document.tobytes()


def case(db, tmp_path, monkeypatch, payload, mime, representation):
    generation = _reserve(db, payload=payload, mime_type=mime, representation=representation)
    root = tmp_path / "user"
    root.mkdir()
    return _io(db, root, monkeypatch), generation, root


def row(db):
    with db.transaction() as conn:
        return dict(db.native_assets._candidate_row("1", "candidate-one", conn=conn))


async def prepare(adapter, generation, payload, filename):
    with io.BytesIO(payload) as stream:
        return await adapter.prepare_upload(
            owner(),
            "native_fork_v1",
            "fork-one",
            "sha256:one",
            generation,
            "candidate-one",
            upload=UploadFile(file=stream, filename=filename),
        )


def test_filename_fallback_is_not_content_detection(tmp_path, monkeypatch):
    monkeypatch.setattr(sink, "puremagic", None)
    monkeypatch.setattr(sink, "_get_python_magic_module", lambda: None)
    path = tmp_path / "source.pdf"
    path.write_bytes(b"extracted words only")
    result = sink.FileValidator().validate_file(path, media_type_key="pdf")
    assert result.mime_detection_source == "fallback"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload,mime,representation,filename",
    [
        (png(), "image/png", "embedded_image_v1", "source.png"),
        (png(), "image/png", "generated_image_revision_v1", "chosen.png"),
        ("Résumé 😀".encode(), "text/plain", "document_text_v1", "processed.txt"),
        (b"Original text", "text/plain", "document_original_v1", "source.txt"),
    ],
    ids=["embedded", "generated", "unicode-text", "original-text"],
)
async def test_prepares_exact_actual_uploaded_bytes(
    native_fork_db, tmp_path, monkeypatch, payload, mime, representation, filename
):
    adapter, generation, root = case(native_fork_db, tmp_path, monkeypatch, payload, mime, representation)
    result = await prepare(adapter, generation, payload, filename)
    candidate = row(native_fork_db)
    assert result.state == candidate["state"] == "prepared"
    assert (root / "chat_assets" / candidate["storage_key"]).read_bytes() == payload
    assert candidate["expected_hash"] == hashlib.sha256(payload).hexdigest()


@pytest.mark.asyncio
async def test_prepares_actual_pdf_original(native_fork_db, tmp_path, monkeypatch):
    payload = pdf()
    adapter, generation, root = case(
        native_fork_db, tmp_path, monkeypatch, payload, "application/pdf", "document_original_v1"
    )
    assert (await prepare(adapter, generation, payload, "source.pdf")).state == "prepared"
    assert (root / "chat_assets" / row(native_fork_db)["storage_key"]).read_bytes() == payload


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload,mime,representation,filename",
    [
        (b"extracted words only", "application/pdf", "document_original_v1", "source.pdf"),
        (png(), "image/jpeg", "embedded_image_v1", "source.jpg"),
        (b"\x89PNG\r\n\x1a\ntruncated", "image/png", "embedded_image_v1", "source.png"),
        (b"\xff\xfeinvalid utf8", "text/plain", "document_text_v1", "processed.txt"),
        (b"words", "application/pdf", "document_text_v1", "processed.pdf"),
        (b"words", "text/plain", "file_v1", "processed.txt"),
        (b"words", "text/plain", "unavailable_v1", "processed.txt"),
    ],
    ids=[
        "text-as-pdf",
        "wrong-image-mime",
        "truncated-image",
        "invalid-utf8",
        "false-text-mime",
        "unknown",
        "unavailable",
    ],
)
async def test_refuses_spoofed_or_unsupported_representation(
    native_fork_db, tmp_path, monkeypatch, payload, mime, representation, filename
):
    adapter, generation, root = case(native_fork_db, tmp_path, monkeypatch, payload, mime, representation)
    with pytest.raises(ValueError, match="native_asset_|native_upload_"):
        await prepare(adapter, generation, payload, filename)
    assert row(native_fork_db)["state"] == "reserved"
    assert not list(root.glob("chat_assets/blobs/**/*.blob"))


@pytest.mark.asyncio
async def test_binary_original_refuses_detector_fallback(native_fork_db, tmp_path, monkeypatch):
    payload = pdf()
    monkeypatch.setattr(sink, "puremagic", None)
    monkeypatch.setattr(sink, "_get_python_magic_module", lambda: None)
    monkeypatch.setattr(sink, "_default_file_validator", sink.FileValidator())
    adapter, generation, root = case(
        native_fork_db, tmp_path, monkeypatch, payload, "application/pdf", "document_original_v1"
    )
    with pytest.raises(ValueError, match="native_asset_mime_unverified"):
        await prepare(adapter, generation, payload, "source.pdf")
    assert row(native_fork_db)["state"] == "reserved"


@pytest.mark.asyncio
async def test_actual_image_uses_destination_cap_not_flashcard_cap(native_fork_db, tmp_path, monkeypatch):
    payload = png() + b"x" * (3 * 1024 * 1024)
    adapter, generation, root = case(native_fork_db, tmp_path, monkeypatch, payload, "image/png", "embedded_image_v1")
    assert (await prepare(adapter, generation, payload, "source.png")).state == "prepared"
    assert (root / "chat_assets" / row(native_fork_db)["storage_key"]).read_bytes() == payload


@pytest.mark.asyncio
async def test_actual_upload_does_not_replace_reserved_identity(native_fork_db, tmp_path, monkeypatch):
    adapter, generation, root = case(native_fork_db, tmp_path, monkeypatch, b"first", "text/plain", "document_text_v1")
    with pytest.raises(ValueError, match="native_asset_content_mismatch"):
        await prepare(adapter, generation, b"other", "processed.txt")
    assert row(native_fork_db)["state"] == "reserved"
    assert not list(root.glob("chat_assets/blobs/**/*.blob"))


@pytest.mark.asyncio
async def test_held_upload_rechecks_original_generation(native_fork_db, tmp_path, monkeypatch):
    payload = b"first"
    adapter, generation, root = case(native_fork_db, tmp_path, monkeypatch, payload, "text/plain", "document_text_v1")

    class HeldUpload:
        filename = "processed.txt"
        sent = False

        async def read(self, size):
            if self.sent:
                return b""
            self.sent = True
            _expire(native_fork_db)
            return payload

    with pytest.raises(ValueError, match="stale_attempt"):
        await adapter.prepare_upload(
            owner(), "native_fork_v1", "fork-one", "sha256:one", generation, "candidate-one", upload=HeldUpload()
        )
    assert row(native_fork_db)["state"] == "reserved"


@pytest.mark.asyncio
async def test_limit_change_after_publication_cannot_prepare(native_fork_db, tmp_path, monkeypatch):
    from tldw_Server_API.app.core.config import settings
    from tldw_Server_API.app.core.Sync.v2.blob_store import LocalSyncBlobStore

    payload = png()
    adapter, generation, root = case(native_fork_db, tmp_path, monkeypatch, payload, "image/png", "embedded_image_v1")
    publish = LocalSyncBlobStore._commit_native_upload

    def changed_limit(**kwargs):
        result = publish(**kwargs)
        monkeypatch.setitem(settings, "MAX_MESSAGE_IMAGE_BYTES", len(payload) - 1)
        return result

    monkeypatch.setattr(LocalSyncBlobStore, "_commit_native_upload", staticmethod(changed_limit))
    with pytest.raises(ValueError, match="native_asset_size_limit"):
        await prepare(adapter, generation, payload, "source.png")
    assert row(native_fork_db)["state"] == "reserved"
    assert list(root.glob("chat_assets/blobs/**/*.blob"))
    with native_fork_db.transaction() as conn:
        quota = conn.execute(
            "SELECT state FROM native_chat_quota_intents WHERE candidate_id = ?", ("candidate-one",)
        ).fetchone()
    assert quota["state"] == "confirmed"


@pytest.mark.asyncio
async def test_publication_failure_retries_same_candidate(native_fork_db, tmp_path, monkeypatch):
    from tldw_Server_API.app.core.Sync.v2.blob_store import LocalSyncBlobStore

    payload = b"exact text"
    adapter, generation, root = case(native_fork_db, tmp_path, monkeypatch, payload, "text/plain", "document_text_v1")
    publish = LocalSyncBlobStore._commit_native_upload

    def fail(**kwargs):
        raise OSError("synthetic native publication failure")

    monkeypatch.setattr(LocalSyncBlobStore, "_commit_native_upload", staticmethod(fail))
    with pytest.raises(OSError, match="synthetic native"):
        await prepare(adapter, generation, payload, "processed.txt")
    assert row(native_fork_db)["state"] == "reserved"
    monkeypatch.setattr(LocalSyncBlobStore, "_commit_native_upload", staticmethod(publish))
    assert (await prepare(adapter, generation, payload, "processed.txt")).state == "prepared"
    assert len(list(root.glob("chat_assets/blobs/**/*.blob"))) == 1
    with native_fork_db.transaction() as conn:
        count = conn.execute(
            "SELECT COUNT(*) AS n FROM native_chat_quota_intents WHERE candidate_id = ?", ("candidate-one",)
        ).fetchone()["n"]
    assert count == 1


@pytest.mark.asyncio
async def test_cancelled_request_keeps_staging_until_worker_drains(native_fork_db, tmp_path, monkeypatch):
    payload = b"exact text"
    adapter, generation, root = case(native_fork_db, tmp_path, monkeypatch, payload, "text/plain", "document_text_v1")
    started, release, finished = threading.Event(), threading.Event(), threading.Event()
    paths = []
    original = adapter._prepare_upload

    def held(*args):
        paths.append(args[6])
        started.set()
        assert release.wait(15)
        try:
            return original(*args)
        finally:
            finished.set()

    monkeypatch.setattr(adapter, "_prepare_upload", held)
    work = asyncio.create_task(prepare(adapter, generation, payload, "processed.txt"))
    try:
        assert await asyncio.to_thread(started.wait, 10)
        work.cancel()
        await asyncio.sleep(0)
        work.cancel()
        await asyncio.sleep(0)
        staged_exists = paths[0].exists()
    finally:
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await work
        assert await asyncio.to_thread(finished.wait, 10)
    assert staged_exists
    assert row(native_fork_db)["state"] == "reserved"
    assert not list(root.glob("chat_assets/blobs/**/*.blob"))


def test_shared_parser_refuses_header_valid_truncated_jpeg():
    from tldw_Server_API.app.core.Utils.image_validation import validate_uploaded_image_bytes

    stream = io.BytesIO()
    Image.new("RGB", (16, 16), "red").save(stream, format="JPEG")
    payload = stream.getvalue()
    sos = payload.index(b"\xff\xda")
    end_of_header = sos + 2 + int.from_bytes(payload[sos + 2 : sos + 4], "big")
    truncated = payload[: end_of_header + 1]
    with Image.open(io.BytesIO(truncated)) as opened:
        assert opened.format == "JPEG"
    assert not validate_uploaded_image_bytes(truncated, "image/jpeg", max_bytes=5 * 1024 * 1024)[0]


@pytest.mark.asyncio
async def test_reloaded_document_limit_prevents_prepared_cas(native_fork_db, tmp_path, monkeypatch):
    from tldw_Server_API.app.core import config
    from tldw_Server_API.app.core.Sync.v2.blob_store import LocalSyncBlobStore

    private_config = tmp_path / "private-config.txt"
    private_config.write_text("[Media-Processing]\nmax_pdf_file_size_mb=50\n")
    monkeypatch.setenv("TLDW_CONFIG_FILE", str(private_config))
    monkeypatch.setattr(config, "_CONFIG_PARSER_CACHE", None)
    payload = pdf()
    adapter, generation, root = case(
        native_fork_db, tmp_path, monkeypatch, payload, "application/pdf", "document_original_v1"
    )
    publish = LocalSyncBlobStore._commit_native_upload

    def lowered_limit(**kwargs):
        result = publish(**kwargs)
        private_config.write_text("[Media-Processing]\nmax_pdf_file_size_mb=0\n")
        config.refresh_config_cache()
        return result

    monkeypatch.setattr(LocalSyncBlobStore, "_commit_native_upload", staticmethod(lowered_limit))
    try:
        with pytest.raises(ValueError, match="native_asset_size_limit"):
            await prepare(adapter, generation, payload, "source.pdf")
        assert row(native_fork_db)["state"] == "reserved"
        assert list(root.glob("chat_assets/blobs/**/*.blob"))
    finally:
        config.refresh_config_cache()


@pytest.mark.asyncio
async def test_actual_sink_reads_are_bounded_by_reserved_native_bytes(native_fork_db, tmp_path, monkeypatch):
    payload = b"first"
    adapter, generation, root = case(native_fork_db, tmp_path, monkeypatch, payload, "text/plain", "document_text_v1")
    requests = []

    class OversizedUpload:
        filename = "processed.txt"
        sent = False

        async def read(self, size):
            requests.append(size)
            if self.sent:
                return b""
            self.sent = True
            return b"x" * size

    with pytest.raises(ValueError, match="native_upload_|native_asset_"):
        await adapter.prepare_upload(
            owner(), "native_fork_v1", "fork-one", "sha256:one", generation, "candidate-one", upload=OversizedUpload()
        )
    assert max(requests) <= len(payload) + 1
    assert row(native_fork_db)["state"] == "reserved"
    assert not list(root.glob("chat_assets/blobs/**/*.blob"))


@pytest.mark.asyncio
async def test_structured_text_original_cannot_use_filename_fallback(native_fork_db, tmp_path, monkeypatch):
    payload = b"extracted words only"
    monkeypatch.setattr(sink, "puremagic", None)
    monkeypatch.setattr(sink, "_get_python_magic_module", lambda: None)
    monkeypatch.setattr(sink, "_default_file_validator", sink.FileValidator())
    adapter, generation, root = case(native_fork_db, tmp_path, monkeypatch, payload, "text/rtf", "document_original_v1")
    with pytest.raises(ValueError, match="native_asset_mime_unverified"):
        await prepare(adapter, generation, payload, "original.rtf")
    assert row(native_fork_db)["state"] == "reserved"
    assert not list(root.glob("chat_assets/blobs/**/*.blob"))


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["image", "document"])
async def test_final_locked_admission_rechecks_current_limit(native_fork_db, tmp_path, monkeypatch, kind):
    from tldw_Server_API.app.core import config

    if kind == "image":
        payload, mime, representation, filename = png(), "image/png", "embedded_image_v1", "source.png"
    else:
        payload, mime, representation, filename = pdf(), "application/pdf", "document_original_v1", "source.pdf"
        private_config = tmp_path / "private-config.txt"
        private_config.write_text("[Media-Processing]\nmax_pdf_file_size_mb=50\n")
        monkeypatch.setenv("TLDW_CONFIG_FILE", str(private_config))
        monkeypatch.setattr(config, "_CONFIG_PARSER_CACHE", None)
    adapter, generation, root = case(native_fork_db, tmp_path, monkeypatch, payload, mime, representation)
    read = native_fork_db.native_assets.read_candidate_for_write
    lowered = False

    def changed_after_locks(*args, **kwargs):
        nonlocal lowered
        current = read(*args, **kwargs)
        if not lowered and list(root.glob("chat_assets/blobs/**/*.blob")):
            lowered = True
            if kind == "image":
                monkeypatch.setitem(config.settings, "MAX_MESSAGE_IMAGE_BYTES", len(payload) - 1)
            else:
                private_config.write_text("[Media-Processing]\nmax_pdf_file_size_mb=0\n")
                config.refresh_config_cache()
        return current

    monkeypatch.setattr(native_fork_db.native_assets, "read_candidate_for_write", changed_after_locks)
    try:
        with pytest.raises(ValueError, match="native_asset_size_limit"):
            await prepare(adapter, generation, payload, filename)
        assert lowered
        assert row(native_fork_db)["state"] == "reserved"
        assert list(root.glob("chat_assets/blobs/**/*.blob"))
    finally:
        if kind == "document":
            config.refresh_config_cache()


@pytest.mark.asyncio
async def test_cancellation_after_last_chunk_refuses_final_publication(native_fork_db, tmp_path, monkeypatch):
    from tldw_Server_API.app.core.Sync.v2.blob_store import LocalSyncBlobStore

    payload = b"exact text"
    adapter, generation, root = case(native_fork_db, tmp_path, monkeypatch, payload, "text/plain", "document_text_v1")
    started, release = threading.Event(), threading.Event()
    write = LocalSyncBlobStore._write_native_chunk

    def held(**kwargs):
        result = write(**kwargs)
        started.set()
        assert release.wait(15)
        return result

    monkeypatch.setattr(LocalSyncBlobStore, "_write_native_chunk", staticmethod(held))
    work = asyncio.create_task(prepare(adapter, generation, payload, "processed.txt"))
    try:
        assert await asyncio.to_thread(started.wait, 10)
        work.cancel()
        await asyncio.sleep(0)
    finally:
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await work
    assert row(native_fork_db)["state"] == "reserved"
    assert not list(root.glob("chat_assets/blobs/**/*.blob"))
    assert list(root.glob("chat_assets/_uploads/**/*.part"))


@pytest.mark.asyncio
@pytest.mark.parametrize("marker", [True, False], ids=["epub-marker", "missing-marker"])
async def test_real_epub_container_preserves_house_marker_validation(native_fork_db, tmp_path, monkeypatch, marker):
    import zipfile

    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        if marker:
            archive.writestr("mimetype", "application/epub+zip")
        archive.writestr("META-INF/container.xml", '<container version="1.0"/>')
        archive.writestr("chapter.xhtml", '<html xmlns="http://www.w3.org/1999/xhtml"><body>Original</body></html>')
    payload = stream.getvalue()
    monkeypatch.setattr(sink.puremagic, "from_file", lambda *args, **kwargs: "application/zip")
    monkeypatch.setattr(sink, "_default_file_validator", sink.FileValidator())
    adapter, generation, root = case(
        native_fork_db, tmp_path, monkeypatch, payload, "application/epub+zip", "document_original_v1"
    )
    if marker:
        assert (await prepare(adapter, generation, payload, "source.epub")).state == "prepared"
        assert (root / "chat_assets" / row(native_fork_db)["storage_key"]).read_bytes() == payload
    else:
        with pytest.raises(ValueError, match="native_upload_|native_asset_"):
            await prepare(adapter, generation, payload, "source.epub")
        assert row(native_fork_db)["state"] == "reserved"
        assert not list(root.glob("chat_assets/blobs/**/*.blob"))


@pytest.mark.asyncio
@pytest.mark.parametrize("fence", ["expired", "reclaiming"])
async def test_durable_fence_during_chunk_stops_later_byte_calls(native_fork_db, tmp_path, monkeypatch, fence):
    from datetime import datetime, timezone

    from tldw_Server_API.app.core.Sync.v2.blob_store import STREAM_CHUNK_SIZE, LocalSyncBlobStore

    payload = b"a" * (STREAM_CHUNK_SIZE * 2 + 3)
    adapter, generation, root = case(native_fork_db, tmp_path, monkeypatch, payload, "text/plain", "document_text_v1")
    write = LocalSyncBlobStore._write_native_chunk
    publish = LocalSyncBlobStore._commit_native_upload
    indexes, publications = [], []

    def fenced_chunk(**kwargs):
        result = write(**kwargs)
        indexes.append(kwargs["chunk_index"])
        if kwargs["chunk_index"] == 0:
            _expire(native_fork_db)
            if fence == "reclaiming":
                with native_fork_db.transaction() as conn:
                    native_fork_db.native_assets.claim_reclamation(
                        owner(),
                        "native_fork_v1",
                        "fork-one",
                        "sha256:one",
                        "candidate-one",
                        now=datetime.now(timezone.utc),
                        conn=conn,
                    )
        return result

    def observed_publish(**kwargs):
        publications.append(kwargs["upload_id"])
        return publish(**kwargs)

    monkeypatch.setattr(LocalSyncBlobStore, "_write_native_chunk", staticmethod(fenced_chunk))
    monkeypatch.setattr(LocalSyncBlobStore, "_commit_native_upload", staticmethod(observed_publish))
    with pytest.raises(ValueError, match="stale_attempt|candidate_state_conflict"):
        await prepare(adapter, generation, payload, "processed.txt")
    assert indexes == [0]
    assert not publications
    assert not list(root.glob("chat_assets/blobs/**/*.blob"))
    assert row(native_fork_db)["state"] == ("reserved" if fence == "expired" else "reclaiming")


def test_oversized_compressed_epub_marker_is_refused_before_open(tmp_path, monkeypatch):
    import zipfile

    path = tmp_path / "oversized-marker.epub"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("mimetype", b"x" * (4 * 1024 * 1024))
    assert path.stat().st_size < 20 * 1024 * 1024
    original = zipfile.ZipFile.open
    opened = []

    def observed_open(self, name, *args, **kwargs):
        opened.append(name)
        return original(self, name, *args, **kwargs)

    monkeypatch.setattr(zipfile.ZipFile, "open", observed_open)
    assert not sink._is_epub_archive(path)
    assert not opened


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["chunk", "final"])
async def test_cancellation_during_admission_refuses_next_byte_call(native_fork_db, tmp_path, monkeypatch, phase):
    from tldw_Server_API.app.core.Sync.v2.blob_store import LocalSyncBlobStore

    payload = b"exact text"
    adapter, generation, root = case(native_fork_db, tmp_path, monkeypatch, payload, "text/plain", "document_text_v1")
    started, release, validated = threading.Event(), threading.Event(), threading.Event()
    policy, descriptor = adapter._upload_policy, adapter._descriptor
    write = LocalSyncBlobStore._write_native_chunk
    written = []

    def checked_policy(*args):
        result = policy(*args)
        if threading.current_thread() is not threading.main_thread():
            validated.set()
        return result

    def held_admission(*args):
        result = descriptor(*args)
        if validated.is_set() and not started.is_set() and (phase == "chunk" or written):
            started.set()
            assert release.wait(15)
        return result

    def observed_write(**kwargs):
        result = write(**kwargs)
        written.append(kwargs["chunk_index"])
        return result

    monkeypatch.setattr(adapter, "_upload_policy", checked_policy)
    monkeypatch.setattr(adapter, "_descriptor", held_admission)
    monkeypatch.setattr(LocalSyncBlobStore, "_write_native_chunk", staticmethod(observed_write))
    work = asyncio.create_task(prepare(adapter, generation, payload, "processed.txt"))
    try:
        assert await asyncio.to_thread(started.wait, 10)
        work.cancel()
        await asyncio.sleep(0)
    finally:
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await work
    assert row(native_fork_db)["state"] == "reserved"
    assert not list(root.glob("chat_assets/blobs/**/*.blob"))
    assert written == ([] if phase == "chunk" else [0])
