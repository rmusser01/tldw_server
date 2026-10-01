import io
import stat
import tarfile
import zipfile

import pytest

pytestmark = pytest.mark.unit


def test_validate_file_mime_mismatch_hard_fail(tmp_path, monkeypatch):


    """Reject when magic-detected MIME disagrees with allowed and do not accept fallback."""
    # Arrange: create a non-PDF file with .pdf extension
    p = tmp_path / "fake.pdf"
    p.write_bytes(b"not-a-pdf")

    # Monkeypatch puremagic in the module to simulate a strong MIME detection
    import tldw_Server_API.app.core.Ingestion_Media_Processing.Upload_Sink as US

    class DummyMagic:
        @staticmethod
        def from_file(path, mime=True):
            return "application/x-msdownload"

    monkeypatch.setattr(US, "puremagic", DummyMagic)

    from tldw_Server_API.app.core.Ingestion_Media_Processing.Upload_Sink import FileValidator

    v = FileValidator()
    res = v.validate_file(p, original_filename="fake.pdf", media_type_key="pdf")
    assert not res, f"Expected failure; issues: {res.issues}"
    assert any("Detected MIME" in i for i in res.issues), res.issues


def test_validate_audio_file_accepts_mp4a_latm(tmp_path, monkeypatch):


    """Accept AAC-in-MP4 streams commonly detected as audio/mp4a-latm."""
    p = tmp_path / "sample.m4a"
    p.write_bytes(b"fake-audio")

    import tldw_Server_API.app.core.Ingestion_Media_Processing.Upload_Sink as US

    class DummyMagic:
        @staticmethod
        def from_file(path, mime=True):
            return "audio/mp4a-latm"

    monkeypatch.setattr(US, "puremagic", DummyMagic)

    from tldw_Server_API.app.core.Ingestion_Media_Processing.Upload_Sink import FileValidator

    v = FileValidator()
    res = v.validate_file(p, original_filename="sample.m4a", media_type_key="audio")
    assert res, res.issues
    assert (res.detected_mime_type or "").lower() == "audio/mp4a-latm"


def test_validate_archive_rejects_encrypted_zip(tmp_path, monkeypatch):


    """Explicitly reject encrypted ZIP entries using flag_bits check."""
    zpath = tmp_path / "enc.zip"
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("ok.txt", "hello")

    # Provide a fake encrypted entry via infolist monkeypatch
    class DummyInfo:
        filename = "secret.txt"
        file_size = 10
        external_attr = 0
        flag_bits = 0x1  # Encrypted

        def is_dir(self):

            return False

    def fake_infolist(self):

        return [DummyInfo()]

    monkeypatch.setattr(zipfile.ZipFile, "infolist", fake_infolist, raising=False)

    from tldw_Server_API.app.core.Ingestion_Media_Processing.Upload_Sink import FileValidator

    v = FileValidator()
    res = v.validate_archive_contents(zpath)
    assert not res, "Encrypted archive should be rejected"
    assert any("encrypted member" in i.lower() for i in res.issues), res.issues


def test_validate_archive_flags_zip_symlink(tmp_path, monkeypatch):


    """Symlink entries in ZIP are flagged and skipped."""
    zpath = tmp_path / "sym.zip"
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("ok.txt", "hello")

    # Provide a symlink-like entry via external_attr high bits
    mode = stat.S_IFLNK | 0o777

    class DummyInfo:
        filename = "link"
        file_size = 0
        external_attr = (mode & 0xFFFF) << 16
        flag_bits = 0

        def is_dir(self):

            return False

    def fake_infolist(self):

        return [DummyInfo()]

    monkeypatch.setattr(zipfile.ZipFile, "infolist", fake_infolist, raising=False)

    from tldw_Server_API.app.core.Ingestion_Media_Processing.Upload_Sink import FileValidator

    v = FileValidator()
    res = v.validate_archive_contents(zpath)
    assert not res, "ZIP symlink entry should be flagged and cause validation failure"
    assert any("symbolic link" in i.lower() for i in res.issues), res.issues


def test_validate_archive_flags_tar_bad_types(tmp_path, monkeypatch):


    """Non-file TAR members are flagged and not extracted."""
    tpath = tmp_path / "bad.tar"
    with tarfile.open(tpath, "w") as tf:
        data = b"hello"
        ti = tarfile.TarInfo(name="ok.txt")
        ti.size = len(data)
        tf.addfile(ti, io.BytesIO(data))

    # Fake getmembers to return a symlink-like member
    class DummyTarInfo:
        name = "weird"
        size = 0
        type = b"?"  # unknown type marker for message

        def isdir(self):

            return False

        def issym(self):

            return True

        def islnk(self):

            return False

        def isfile(self):

            return False

    def fake_getmembers(self):

        return [DummyTarInfo()]

    monkeypatch.setattr(tarfile.TarFile, "getmembers", fake_getmembers, raising=False)

    from tldw_Server_API.app.core.Ingestion_Media_Processing.Upload_Sink import FileValidator

    v = FileValidator()
    res = v.validate_archive_contents(tpath)
    assert not res, "TAR with non-file member should be rejected"
    assert any("link entry" in i.lower() or "unsupported member type" in i.lower() for i in res.issues), res.issues


def test_svg_treated_as_xml_and_allowed(tmp_path):


    """SVG is handled under XML rules and accepted (sanitization path available)."""
    svg_path = tmp_path / "img.svg"
    svg_path.write_text("<svg xmlns='http://www.w3.org/2000/svg'><title>T</title></svg>")

    from tldw_Server_API.app.core.Ingestion_Media_Processing.Upload_Sink import FileValidator

    v = FileValidator()
    res = v.validate_file(svg_path, original_filename="img.svg", media_type_key="xml")
    assert res, res.issues
    # Expect MIME either detected as image/svg+xml or via fallback
    assert (res.detected_mime_type or "").lower() in ("image/svg+xml", "text/xml", "application/xml")


def test_missing_mime_libraries_warn_about_extension_fallback(monkeypatch):
    from loguru import logger

    from tldw_Server_API.app.core.Ingestion_Media_Processing import Upload_Sink as sink

    monkeypatch.setattr(sink, "puremagic", None)
    monkeypatch.setattr(sink, "_get_python_magic_module", lambda: None)
    output = []
    token = logger.add(
        output.append, level="WARNING", filter=lambda record: record["name"] == sink.__name__
    )
    try:
        sink.FileValidator()
    finally:
        logger.remove(token)
    messages = " ".join(message.record["message"] for message in output).lower()
    assert "mime" in messages and "extension" in messages
    assert "permit scanner errors" not in messages
