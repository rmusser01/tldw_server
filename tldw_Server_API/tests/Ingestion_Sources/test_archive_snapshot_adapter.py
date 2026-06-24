from __future__ import annotations

import io
import tarfile
import zipfile

import pytest


@pytest.mark.asyncio
@pytest.mark.unit
async def test_archive_refresh_keeps_previous_snapshot_when_candidate_fails(tmp_path):
    from tldw_Server_API.app.core.Ingestion_Sources.archive_snapshot import apply_archive_candidate

    current_snapshot = {"id": 3, "status": "active"}

    with pytest.raises(ValueError, match="Invalid ZIP archive"):
        await apply_archive_candidate(
            source_id=11,
            archive_bytes=b"not-a-zip",
            filename="broken.zip",
            current_snapshot=current_snapshot,
        )

    assert current_snapshot == {"id": 3, "status": "active"}


@pytest.mark.unit
def test_archive_media_snapshot_supports_pdf_and_epub_with_collected_failures(monkeypatch):
    import tldw_Server_API.app.core.Ingestion_Sources.archive_snapshot as archive_snapshot

    archive_buffer = io.BytesIO()
    with zipfile.ZipFile(archive_buffer, "w") as archive:
        archive.writestr("export/report.pdf", b"%PDF-1.4 fake\n")
        archive.writestr("export/book.epub", b"fake-epub")
        archive.writestr("export/bad.pdf", b"%PDF-1.4 broken\n")
    archive_bytes = archive_buffer.getvalue()

    def _fake_process_pdf(file_input, *, filename, **kwargs):
        del file_input, kwargs
        if filename == "bad.pdf":
            return {
                "status": "Error",
                "error": "pdf parse failed",
                "warnings": ["pdf parse failed"],
            }
        return {
            "status": "Success",
            "content": f"content for {filename}",
            "metadata": {"title": filename, "author": "PDF Author", "raw": {"pages": 1}},
            "parser_used": "pymupdf4llm",
            "input_ref": filename,
        }

    def _fake_process_epub(file_path, **kwargs):
        del file_path, kwargs
        return {
            "status": "Success",
            "content": "epub body",
            "metadata": {"title": "Book Title", "author": "EPUB Author", "raw": {"chapters": 2}},
            "parser_used": "filtered",
        }

    monkeypatch.setattr(archive_snapshot, "process_pdf", _fake_process_pdf)
    monkeypatch.setattr(archive_snapshot, "process_epub", _fake_process_epub)

    items, failures = archive_snapshot.build_archive_snapshot_from_bytes_with_failures(
        archive_bytes=archive_bytes,
        filename="documents.zip",
        sink_type="media",
    )

    assert set(items) == {"book.epub", "report.pdf"}
    assert set(failures) == {"bad.pdf"}
    assert items["report.pdf"]["source_format"] == "pdf"
    assert items["book.epub"]["source_format"] == "epub"
    assert failures["bad.pdf"]["error"] == "pdf parse failed"


@pytest.mark.unit
def test_archive_snapshot_supports_tar_gz_members_for_notes():
    from tldw_Server_API.app.core.Ingestion_Sources.archive_snapshot import (
        build_archive_snapshot_from_bytes_with_failures,
    )

    archive_buffer = io.BytesIO()
    with tarfile.open(fileobj=archive_buffer, mode="w:gz") as archive:
        payload = b"# Alpha\n\nfrom tar archive\n"
        member = tarfile.TarInfo("export/alpha.md")
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))

    items, failures = build_archive_snapshot_from_bytes_with_failures(
        archive_bytes=archive_buffer.getvalue(),
        filename="notes.tar.gz",
        sink_type="notes",
    )

    assert failures == {}
    assert set(items) == {"alpha.md"}
    assert items["alpha.md"]["source_format"] == "md"
    assert "from tar archive" in items["alpha.md"]["text"]


@pytest.mark.unit
def test_validate_archive_members_rejects_tar_symlink():
    from tldw_Server_API.app.core.Ingestion_Sources.archive_snapshot import (
        validate_archive_members,
    )

    archive_buffer = io.BytesIO()
    with tarfile.open(fileobj=archive_buffer, mode="w:gz") as archive:
        member = tarfile.TarInfo("export/link.md")
        member.type = tarfile.SYMTYPE
        member.linkname = "target.md"
        archive.addfile(member)

    with pytest.raises(ValueError, match="symbolic link"):
        validate_archive_members(
            archive_buffer.getvalue(),
            filename="unsafe.tar.gz",
        )


@pytest.mark.unit
def test_archive_snapshot_preserves_suffix_colliding_member_content(monkeypatch):
    import tldw_Server_API.app.core.Ingestion_Sources.archive_snapshot as archive_snapshot

    archive_buffer = io.BytesIO()
    with zipfile.ZipFile(archive_buffer, "w") as archive:
        archive.writestr("export/a.md", "# Root\n")
        archive.writestr("export/dir/a.md", "# Nested\n")

    def _fake_convert_document_to_text(path):
        return path.read_text(encoding="utf-8"), "md", {}

    monkeypatch.setattr(
        archive_snapshot,
        "convert_document_to_text",
        _fake_convert_document_to_text,
    )

    items, failures = archive_snapshot.build_archive_snapshot_from_bytes_with_failures(
        archive_bytes=archive_buffer.getvalue(),
        filename="notes.zip",
        sink_type="notes",
    )

    assert failures == {}
    assert items["a.md"]["text"] == "# Root\n"
    assert items["dir/a.md"]["text"] == "# Nested\n"


@pytest.mark.unit
def test_validate_archive_members_rejects_zip_member_over_size_limit(monkeypatch):
    from tldw_Server_API.app.core.Ingestion_Sources.archive_snapshot import (
        validate_archive_members,
    )

    monkeypatch.setenv("INGESTION_SOURCES_ARCHIVE_MEMBER_MAX_BYTES", "8")
    archive_buffer = io.BytesIO()
    with zipfile.ZipFile(archive_buffer, "w") as archive:
        archive.writestr("export/large.md", b"123456789")

    with pytest.raises(ValueError, match="exceeds archive member size limit"):
        validate_archive_members(
            archive_buffer.getvalue(),
            filename="notes.zip",
        )


@pytest.mark.unit
def test_validate_archive_members_rejects_tar_total_uncompressed_limit(monkeypatch):
    from tldw_Server_API.app.core.Ingestion_Sources.archive_snapshot import (
        validate_archive_members,
    )

    monkeypatch.setenv("INGESTION_SOURCES_ARCHIVE_TOTAL_MAX_BYTES", "8")
    archive_buffer = io.BytesIO()
    with tarfile.open(fileobj=archive_buffer, mode="w:gz") as archive:
        first_payload = b"12345"
        first = tarfile.TarInfo("export/first.md")
        first.size = len(first_payload)
        archive.addfile(first, io.BytesIO(first_payload))
        second_payload = b"6789"
        second = tarfile.TarInfo("export/second.md")
        second.size = len(second_payload)
        archive.addfile(second, io.BytesIO(second_payload))

    with pytest.raises(ValueError, match="exceeds archive total uncompressed size limit"):
        validate_archive_members(
            archive_buffer.getvalue(),
            filename="notes.tar.gz",
        )


@pytest.mark.unit
def test_validate_archive_members_rejects_zip_actual_bytes_over_total_limit(monkeypatch):
    import tldw_Server_API.app.core.Ingestion_Sources.archive_snapshot as archive_snapshot

    class _FakeZipInfo:
        filename = "export/spoofed.md"
        file_size = 1
        flag_bits = 0
        external_attr = 0

        def is_dir(self):
            return False

    class _FakeZipFile:
        def __init__(self, *args, **kwargs):
            del args, kwargs

        def __enter__(self):
            return self

        def __exit__(self, *args):
            del args
            return False

        def infolist(self):
            return [_FakeZipInfo()]

        def open(self, member, mode):
            del member, mode
            return io.BytesIO(b"12345")

    monkeypatch.setenv("INGESTION_SOURCES_ARCHIVE_TOTAL_MAX_BYTES", "4")
    monkeypatch.setenv("INGESTION_SOURCES_ARCHIVE_MEMBER_MAX_BYTES", "10")
    monkeypatch.setattr(archive_snapshot.zipfile, "ZipFile", _FakeZipFile)

    with pytest.raises(ValueError, match="exceeds archive total uncompressed size limit"):
        archive_snapshot.validate_archive_members(
            b"fake-zip",
            filename="notes.zip",
        )


@pytest.mark.unit
def test_validate_archive_members_rejects_tar_actual_bytes_over_total_limit(monkeypatch):
    import tldw_Server_API.app.core.Ingestion_Sources.archive_snapshot as archive_snapshot

    class _FakeTarInfo:
        name = "export/spoofed.md"
        size = 1

        def isdir(self):
            return False

        def issym(self):
            return False

        def islnk(self):
            return False

        def isfile(self):
            return True

    class _FakeTarFile:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            del args
            return False

        def __iter__(self):
            yield _FakeTarInfo()

        def extractfile(self, member):
            del member
            return io.BytesIO(b"12345")

    def _fake_tar_open(*args, **kwargs):
        del args, kwargs
        return _FakeTarFile()

    monkeypatch.setenv("INGESTION_SOURCES_ARCHIVE_TOTAL_MAX_BYTES", "4")
    monkeypatch.setenv("INGESTION_SOURCES_ARCHIVE_MEMBER_MAX_BYTES", "10")
    monkeypatch.setattr(archive_snapshot.tarfile, "open", _fake_tar_open)

    with pytest.raises(ValueError, match="exceeds archive total uncompressed size limit"):
        archive_snapshot.validate_archive_members(
            b"fake-tar",
            filename="notes.tar.gz",
        )


@pytest.mark.unit
def test_validate_archive_members_rejects_excessive_member_count(monkeypatch):
    from tldw_Server_API.app.core.Ingestion_Sources.archive_snapshot import (
        validate_archive_members,
    )

    monkeypatch.setenv("INGESTION_SOURCES_ARCHIVE_MAX_MEMBERS", "1")
    archive_buffer = io.BytesIO()
    with zipfile.ZipFile(archive_buffer, "w") as archive:
        archive.writestr("export/first.md", b"first")
        archive.writestr("export/second.md", b"second")

    with pytest.raises(ValueError, match="exceeds archive member count limit"):
        validate_archive_members(
            archive_buffer.getvalue(),
            filename="notes.zip",
        )
