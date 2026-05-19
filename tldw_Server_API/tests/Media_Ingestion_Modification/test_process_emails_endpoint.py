from io import BytesIO
import zipfile
import pytest


class _LoggerStub:
    def __init__(self) -> None:
        self.debugs: list[str] = []
        self.debug_kwargs: list[dict[str, object]] = []

    def info(self, *_args: object, **_kwargs: object) -> None:
        return None

    def debug(self, message: str, *args: object, **kwargs: object) -> None:
        self.debugs.append(message.format(*args, **kwargs) if args or kwargs else message)
        self.debug_kwargs.append(dict(kwargs))


def _assert_sanitized_debug_log(logger: _LoggerStub, expected: str) -> None:
    target_kwargs = [
        kwargs for message, kwargs in zip(logger.debugs, logger.debug_kwargs) if message == expected
    ]
    assert target_kwargs, logger.debugs
    rendered = "\n".join(message for message in logger.debugs if message == expected)
    assert "exploded" not in rendered
    assert "/private/" not in rendered
    assert all(not kwargs for kwargs in target_kwargs)


def test_process_emails_endpoint_basic(client_user_only):


    # Build a minimal EML file
    content = (
        b"From: Alice <alice@example.com>\r\n"
        b"To: Bob <bob@example.com>\r\n"
        b"Subject: Test Email\r\n"
        b"MIME-Version: 1.0\r\n"
        b"Content-Type: text/plain; charset=utf-8\r\n\r\n"
        b"Hello Bob, this is a test.\r\n"
    )

    files = {
        "files": ("test.eml", BytesIO(content), "message/rfc822"),
    }

    r = client_user_only.post("/api/v1/media/process-emails", files=files)
    assert r.status_code in (200, 207)
    data = r.json()
    assert isinstance(data.get("results"), list)
    assert len(data["results"]) >= 1
    first = data["results"][0]
    assert first.get("media_type") == "email"
    md = first.get("metadata", {})
    assert md.get("email", {}).get("subject") == "Test Email"


def test_process_emails_sanitizes_processor_failure(client_user_only, monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints.media import process_emails as process_emails_mod

    def fail_process_email_task(**_kwargs):
        raise RuntimeError("email parser failed at /private/mail.eml")

    monkeypatch.setattr(
        process_emails_mod.email_lib,
        "process_email_task",
        fail_process_email_task,
    )

    content = (
        b"From: Alice <alice@example.com>\r\n"
        b"To: Bob <bob@example.com>\r\n"
        b"Subject: Test Email\r\n"
        b"MIME-Version: 1.0\r\n"
        b"Content-Type: text/plain; charset=utf-8\r\n\r\n"
        b"Hello Bob, this is a test.\r\n"
    )
    files = {
        "files": ("test.eml", BytesIO(content), "message/rfc822"),
    }

    response = client_user_only.post("/api/v1/media/process-emails", files=files)

    assert response.status_code == 207, response.text
    payload = response.json()
    result = payload["results"][0]
    assert result["status"] == "Error"
    assert result["error"] == "Email processing failed"
    assert "email parser failed" not in response.text
    assert "/private/mail.eml" not in response.text


def test_process_emails_rechunk_failure_log_is_sanitized(client_user_only, monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints.media import process_emails as process_emails_mod
    import tldw_Server_API.app.core.Chunking as chunking_mod

    logger_stub = _LoggerStub()

    def stub_process_email_task(**_kwargs):
        return {
            "status": "Success",
            "content": "Hello Bob, this is a test.",
            "metadata": {"email": {"subject": "Test Email"}},
        }

    def fail_improved_chunking_process(*_args, **_kwargs):
        raise RuntimeError("email rechunk exploded at /private/chunks")

    monkeypatch.setattr(process_emails_mod, "logger", logger_stub)
    monkeypatch.setattr(process_emails_mod.email_lib, "process_email_task", stub_process_email_task)
    monkeypatch.setattr(chunking_mod, "improved_chunking_process", fail_improved_chunking_process)

    content = (
        b"From: Alice <alice@example.com>\r\n"
        b"To: Bob <bob@example.com>\r\n"
        b"Subject: Test Email\r\n"
        b"MIME-Version: 1.0\r\n"
        b"Content-Type: text/plain; charset=utf-8\r\n\r\n"
        b"Hello Bob, this is a test.\r\n"
    )
    files = {
        "files": ("test.eml", BytesIO(content), "message/rfc822"),
    }

    response = client_user_only.post(
        "/api/v1/media/process-emails",
        files=files,
        data={"perform_chunking": "true"},
    )

    assert response.status_code == 200, response.text
    _assert_sanitized_debug_log(logger_stub, "Optional email re-chunking failed")


def test_process_emails_template_classifier_failure_log_is_sanitized(client_user_only, monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints.media import process_emails as process_emails_mod

    logger_stub = _LoggerStub()

    class _MediaProxy:
        def __init__(self, wrapped):
            self._wrapped = wrapped

        def __getattr__(self, name):
            if name == "TemplateClassifier":
                raise RuntimeError("template classifier exploded at /private/templates")
            return getattr(self._wrapped, name)

    def stub_process_email_task(**_kwargs):
        return {
            "status": "Success",
            "content": "Hello Bob, this is a test.",
            "metadata": {"email": {"subject": "Test Email"}},
        }

    monkeypatch.setattr(process_emails_mod, "logger", logger_stub)
    monkeypatch.setattr(process_emails_mod, "media_mod", _MediaProxy(process_emails_mod.media_mod))
    monkeypatch.setattr(process_emails_mod.email_lib, "process_email_task", stub_process_email_task)

    content = (
        b"From: Alice <alice@example.com>\r\n"
        b"To: Bob <bob@example.com>\r\n"
        b"Subject: Test Email\r\n"
        b"MIME-Version: 1.0\r\n"
        b"Content-Type: text/plain; charset=utf-8\r\n\r\n"
        b"Hello Bob, this is a test.\r\n"
    )
    files = {
        "files": ("test.eml", BytesIO(content), "message/rfc822"),
    }

    response = client_user_only.post(
        "/api/v1/media/process-emails",
        files=files,
        data={"perform_chunking": "true"},
    )

    assert response.status_code == 200, response.text
    _assert_sanitized_debug_log(logger_stub, "TemplateClassifier not available")


def test_process_emails_first_filename_failure_log_is_sanitized(client_user_only, monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints.media import process_emails as process_emails_mod

    logger_stub = _LoggerStub()

    class _SavedFilesInfo:
        def __init__(self, item):
            self._item = item

        def __iter__(self):
            return iter([self._item])

        def __bool__(self):
            return True

        def __getitem__(self, index):
            if index == 0:
                raise RuntimeError("filename lookup exploded at /private/filelist")
            raise IndexError(index)

    async def fake_save_uploaded_files(files, temp_dir, validator, allowed_extensions):
        saved_path = temp_dir / "test.eml"
        saved_path.write_bytes(
            b"From: Alice <alice@example.com>\r\n"
            b"To: Bob <bob@example.com>\r\n"
            b"Subject: Test Email\r\n"
            b"MIME-Version: 1.0\r\n"
            b"Content-Type: text/plain; charset=utf-8\r\n\r\n"
            b"Hello Bob, this is a test.\r\n"
        )
        return _SavedFilesInfo({"path": str(saved_path), "original_filename": "test.eml"}), []

    def stub_process_email_task(**_kwargs):
        return {
            "status": "Success",
            "content": "Hello Bob, this is a test.",
            "metadata": {"email": {"subject": "Test Email"}},
        }

    monkeypatch.setattr(process_emails_mod, "logger", logger_stub)
    monkeypatch.setattr(process_emails_mod, "save_uploaded_files", fake_save_uploaded_files)
    monkeypatch.setattr(process_emails_mod.email_lib, "process_email_task", stub_process_email_task)

    files = {
        "files": ("test.eml", BytesIO(b"ignored"), "message/rfc822"),
    }

    response = client_user_only.post(
        "/api/v1/media/process-emails",
        files=files,
        data={"perform_chunking": "true"},
    )

    assert response.status_code == 200, response.text
    _assert_sanitized_debug_log(logger_stub, "Could not determine first filename")


def _build_zip_of_emls() -> bytes:


    # Build two simple EMLs in a zip archive (in-memory)
    eml1 = (
        b"From: A <a@example.com>\r\n"
        b"To: B <b@example.com>\r\n"
        b"Subject: Zip One\r\n"
        b"MIME-Version: 1.0\r\n"
        b"Content-Type: text/plain; charset=utf-8\r\n\r\n"
        b"Body one.\r\n"
    )
    eml2 = (
        b"From: C <c@example.com>\r\n"
        b"To: D <d@example.com>\r\n"
        b"Subject: Zip Two\r\n"
        b"MIME-Version: 1.0\r\n"
        b"Content-Type: text/plain; charset=utf-8\r\n\r\n"
        b"Body two.\r\n"
    )
    bio = BytesIO()
    with zipfile.ZipFile(bio, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('one.eml', eml1)
        zf.writestr('two.eml', eml2)
    bio.seek(0)
    return bio.getvalue()


def test_process_emails_endpoint_zip_archive(client_user_only):


    zip_bytes = _build_zip_of_emls()
    files = {
        "files": ("emails.zip", BytesIO(zip_bytes), "application/zip"),
    }
    r = client_user_only.post(
        "/api/v1/media/process-emails",
        files=files,
        data={
            "accept_archives": "true",
            "perform_chunking": "true",
        },
    )
    assert r.status_code in (200, 207)
    data = r.json()
    res = data.get("results")
    assert isinstance(res, list) and len(res) >= 2
    subjects = sorted([item.get("metadata", {}).get("email", {}).get("subject") for item in res if isinstance(item, dict)])
    assert subjects[0] == "Zip One" and subjects[1] == "Zip Two"
    # Assert archive grouping keyword is present on each child
    for item in res:
        if isinstance(item, dict):
            kws = item.get("keywords") or []
            assert "email_archive:emails" in kws, f"Archive keyword missing in child: {kws}"


def _build_mbox_two_emails() -> bytes:


    # Build a small mbox file with two minimal emails via mailbox
    import mailbox as _mailbox
    import tempfile as _tempfile
    from email.message import EmailMessage

    with _tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp_path = tmp.name
    try:
        mbox = _mailbox.mbox(tmp_path)
        # Email 1
        msg1 = EmailMessage()
        msg1["From"] = "A <a@example.com>"
        msg1["To"] = "B <b@example.com>"
        msg1["Subject"] = "Mbox One"
        msg1.set_content("Hello from mbox one.")
        mbox.add(msg1)
        # Email 2
        msg2 = EmailMessage()
        msg2["From"] = "C <c@example.com>"
        msg2["To"] = "D <d@example.com>"
        msg2["Subject"] = "Mbox Two"
        msg2.set_content("Hello from mbox two.")
        mbox.add(msg2)
        mbox.flush()
        mbox.close()
        with open(tmp_path, "rb") as f:
            return f.read()
    finally:
        import os as _os
        try:
            _os.unlink(tmp_path)
        except Exception:
            _ = None


def test_process_emails_endpoint_mbox_archive(client_user_only):


    mbox_bytes = _build_mbox_two_emails()
    files = {
        "files": ("emails.mbox", BytesIO(mbox_bytes), "application/mbox"),
    }
    r = client_user_only.post(
        "/api/v1/media/process-emails",
        files=files,
        data={
            "accept_mbox": "true",
            "perform_chunking": "true",
        },
    )
    assert r.status_code in (200, 207)
    data = r.json()
    res = data.get("results")
    assert isinstance(res, list) and len(res) >= 2
    subjects = sorted([item.get("metadata", {}).get("email", {}).get("subject") for item in res if isinstance(item, dict)])
    assert subjects[0] == "Mbox One" and subjects[1] == "Mbox Two"
    # Assert mbox grouping keyword is present on each child
    for item in res:
        if isinstance(item, dict):
            kws = item.get("keywords") or []
            assert "email_mbox:emails" in kws, f"MBOX keyword missing in child: {kws}"


def test_process_emails_endpoint_mbox_guardrail_too_many_messages(client_user_only):


    # Lower guardrail for internal files to a small number, then exceed it
    import mailbox as _mailbox
    import tempfile as _tempfile
    from email.message import EmailMessage
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Email import Email_Processing_Lib as email_lib

    # Monkeypatch guardrail limits to keep the test lightweight
    archive_cfg = email_lib.DEFAULT_MEDIA_TYPE_CONFIG.get('archive', {})
    orig_max_files = archive_cfg.get('max_internal_files', 100)
    try:
        archive_cfg['max_internal_files'] = 5

        with _tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name
        try:
            mbox = _mailbox.mbox(tmp_path)
            # Create 6 messages to exceed the limit of 5
            for i in range(6):
                msg = EmailMessage()
                msg["From"] = f"X <x{i}@example.com>"
                msg["To"] = "Y <y@example.com>"
                msg["Subject"] = f"Msg {i}"
                msg.set_content("Hi")
                mbox.add(msg)
            mbox.flush()
            mbox.close()
            with open(tmp_path, "rb") as f:
                mbox_bytes = f.read()
        finally:
            import os as _os
            try:
                _os.unlink(tmp_path)
            except Exception:
                _ = None

        files = {
            "files": ("emails.mbox", BytesIO(mbox_bytes), "application/mbox"),
        }
        r = client_user_only.post(
            "/api/v1/media/process-emails",
            files=files,
            data={
                "accept_mbox": "true",
                "perform_chunking": "false",
            },
        )
        assert r.status_code in (200, 207)
        data = r.json()
        res = data.get("results") or []
        # Expect at least one Error item indicating too many messages
        errors = [it for it in res if isinstance(it, dict) and it.get("status") == "Error" and "too many messages" in str(it.get("error", "")).lower()]
        assert errors, f"Expected guardrail error for too many messages, got: {res}"
    finally:
        archive_cfg['max_internal_files'] = orig_max_files


def test_process_emails_endpoint_mbox_guardrail_oversized_bytes(client_user_only):


    # Lower size guardrail to 1 MB and build a ~1.5 MB mbox to trigger size error
    import mailbox as _mailbox
    import tempfile as _tempfile
    from email.message import EmailMessage
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Email import Email_Processing_Lib as email_lib

    archive_cfg = email_lib.DEFAULT_MEDIA_TYPE_CONFIG.get('archive', {})
    orig_max_size_mb = archive_cfg.get('max_internal_uncompressed_size_mb', 200)
    try:
        archive_cfg['max_internal_uncompressed_size_mb'] = 1  # 1 MB

        # Build one big message so the resulting mbox exceeds 1 MB
        big_payload = ("X" * (1024 * 1024 + 500 * 1024))  # ~1.5 MB text
        with _tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name
        try:
            mbox = _mailbox.mbox(tmp_path)
            msg = EmailMessage()
            msg["From"] = "Big <big@example.com>"
            msg["To"] = "Dest <dest@example.com>"
            msg["Subject"] = "BigMsg"
            msg.set_content(big_payload)
            mbox.add(msg)
            mbox.flush()
            mbox.close()
            with open(tmp_path, "rb") as f:
                mbox_bytes = f.read()
        finally:
            import os as _os
            try:
                _os.unlink(tmp_path)
            except Exception:
                _ = None

        files = {
            "files": ("emails.mbox", BytesIO(mbox_bytes), "application/mbox"),
        }
        r = client_user_only.post(
            "/api/v1/media/process-emails",
            files=files,
            data={
                "accept_mbox": "true",
                "perform_chunking": "false",
            },
        )
        assert r.status_code in (200, 207)
        data = r.json()
        res = data.get("results") or []
        # Expect a single error result for size guardrail
        assert len(res) >= 1 and isinstance(res[0], dict)
        err = res[0]
        assert err.get("status") == "Error"
        assert "exceeds limit" in str(err.get("error", "")).lower()
    finally:
        archive_cfg['max_internal_uncompressed_size_mb'] = orig_max_size_mb


def _build_zip_with_emls(n: int, payload_size: int = 32) -> bytes:
    # Build a zip with n EML files, each with payload_size bytes body
    bio = BytesIO()
    with zipfile.ZipFile(bio, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        for i in range(n):
            body = ("X" * payload_size).encode("utf-8")
            eml = (
                b"From: A <a@example.com>\r\n"
                b"To: B <b@example.com>\r\n"
                + f"Subject: Z{i}\r\n".encode("utf-8")
                + b"MIME-Version: 1.0\r\n"
                + b"Content-Type: text/plain; charset=utf-8\r\n\r\n"
                + body
            )
            zf.writestr(f"m{i}.eml", eml)
    bio.seek(0)
    return bio.getvalue()


def test_process_emails_endpoint_zip_guardrail_too_many_files(client_user_only):


    from tldw_Server_API.app.core.Ingestion_Media_Processing.Email import Email_Processing_Lib as email_lib
    archive_cfg = email_lib.DEFAULT_MEDIA_TYPE_CONFIG.get('archive', {})
    orig_max_files = archive_cfg.get('max_internal_files', 100)
    try:
        archive_cfg['max_internal_files'] = 1
        zip_bytes = _build_zip_with_emls(2, payload_size=64)
        files = {
            "files": ("emails.zip", BytesIO(zip_bytes), "application/zip"),
        }
        r = client_user_only.post(
            "/api/v1/media/process-emails",
            files=files,
            data={
                "accept_archives": "true",
            },
        )
        assert r.status_code in (200, 207)
        data = r.json()
        res = data.get("results") or []
        assert len(res) >= 1 and isinstance(res[0], dict)
        err = res[0]
        assert err.get("status") == "Error"
        assert "too many files" in str(err.get("error", "")).lower()
    finally:
        archive_cfg['max_internal_files'] = orig_max_files


def test_process_emails_endpoint_zip_guardrail_oversize(client_user_only):


    from tldw_Server_API.app.core.Ingestion_Media_Processing.Email import Email_Processing_Lib as email_lib
    archive_cfg = email_lib.DEFAULT_MEDIA_TYPE_CONFIG.get('archive', {})
    orig_max_size_mb = archive_cfg.get('max_internal_uncompressed_size_mb', 200)
    try:
        archive_cfg['max_internal_uncompressed_size_mb'] = 1
        # Build one large eml (~1.5MB body)
        big_body_len = 1024 * 1024 + 500 * 1024
        zip_bytes = _build_zip_with_emls(1, payload_size=big_body_len)
        files = {
            "files": ("emails.zip", BytesIO(zip_bytes), "application/zip"),
        }
        r = client_user_only.post(
            "/api/v1/media/process-emails",
            files=files,
            data={
                "accept_archives": "true",
            },
        )
        assert r.status_code in (200, 207)
        data = r.json()
        res = data.get("results") or []
        assert len(res) >= 1 and isinstance(res[0], dict)
        err = res[0]
        assert err.get("status") == "Error"
        assert "exceeds limit" in str(err.get("error", "")).lower()
    finally:
        archive_cfg['max_internal_uncompressed_size_mb'] = orig_max_size_mb


@pytest.mark.performance
def test_process_emails_endpoint_zip_large_container(client_user_only):
    # Build 120 small EMLs and ensure the endpoint expands and processes them
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Email import Email_Processing_Lib as email_lib
    archive_cfg = email_lib.DEFAULT_MEDIA_TYPE_CONFIG.get('archive', {})
    orig_max_files = archive_cfg.get('max_internal_files', 100)
    try:
        archive_cfg['max_internal_files'] = 200
        zip_bytes = _build_zip_with_emls(120, payload_size=64)
        files = {
            "files": ("emails.zip", BytesIO(zip_bytes), "application/zip"),
        }
        r = client_user_only.post(
            "/api/v1/media/process-emails",
            files=files,
            data={
                "accept_archives": "true",
                "perform_chunking": "false",
            },
        )
        assert r.status_code in (200, 207)
        data = r.json()
        res = data.get("results") or []
        # Expect at least 120 children
        assert isinstance(res, list) and len(res) >= 120
    finally:
        archive_cfg['max_internal_files'] = orig_max_files


@pytest.mark.performance
def test_process_emails_endpoint_mbox_large_container(client_user_only):
    # Build an mbox with 120 small messages; ensure expansion handles volume
    import mailbox as _mailbox
    import tempfile as _tempfile
    from email.message import EmailMessage
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Email import Email_Processing_Lib as email_lib

    archive_cfg = email_lib.DEFAULT_MEDIA_TYPE_CONFIG.get('archive', {})
    orig_max_files = archive_cfg.get('max_internal_files', 100)
    try:
        archive_cfg['max_internal_files'] = 200
        with _tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name
        try:
            mbox = _mailbox.mbox(tmp_path)
            for i in range(120):
                msg = EmailMessage()
                msg["From"] = f"X <x{i}@example.com>"
                msg["To"] = "Y <y@example.com>"
                msg["Subject"] = f"Msg {i}"
                msg.set_content("Hi")
                mbox.add(msg)
            mbox.flush()
            mbox.close()
            with open(tmp_path, "rb") as f:
                mbox_bytes = f.read()
        finally:
            import os as _os
            try:
                _os.unlink(tmp_path)
            except Exception:
                _ = None

        files = {
            "files": ("emails.mbox", BytesIO(mbox_bytes), "application/mbox"),
        }
        r = client_user_only.post(
            "/api/v1/media/process-emails",
            files=files,
            data={
                "accept_mbox": "true",
                "perform_chunking": "false",
            },
        )
        assert r.status_code in (200, 207)
        data = r.json()
        res = data.get("results") or []
        assert isinstance(res, list) and len(res) >= 120
    finally:
        archive_cfg['max_internal_files'] = orig_max_files


@pytest.mark.requires_pypff
@pytest.mark.skipif(__import__('importlib').util.find_spec('pypff') is None, reason="pypff is not installed")
def test_process_emails_endpoint_pst_with_pypff_extraction(client_user_only):
    # This test only runs when pypff is installed on the system.
    # Use a tiny fake byte buffer; handler will try to open and likely error as invalid PST.
    # The assertion is focused on exercising the pypff code path under real install conditions.
    pst_bytes = b"!pst"
    files = {
        "files": ("emails.pst", BytesIO(pst_bytes), "application/octet-stream"),
    }
    r = client_user_only.post(
        "/api/v1/media/process-emails",
        files=files,
        data={
            "accept_pst": "true",
            "perform_chunking": "false",
        },
    )
    assert r.status_code in (200, 207)
    data = r.json()
    res = data.get("results") or []
    assert isinstance(res, list) and len(res) >= 1
    # Either we parse some messages or return an 'Invalid PST/OST file' error, but not the feature-flag message.
    first = res[0]
    assert 'support not enabled' not in str(first.get('error', '')).lower()


@pytest.mark.requires_pypff
@pytest.mark.skipif(__import__('os').environ.get('PST_FIXTURE_PATH') in (None, ''), reason="No PST_FIXTURE_PATH provided")
@pytest.mark.skipif(__import__('importlib').util.find_spec('pypff') is None, reason="pypff is not installed")
def test_process_emails_endpoint_pst_recipients_and_date_strict(client_user_only):
    # Requires a tiny valid PST fixture at PST_FIXTURE_PATH with at least one message
    import os
    pst_path = os.environ.get('PST_FIXTURE_PATH')
    assert os.path.isfile(pst_path), f"Fixture not found: {pst_path}"
    with open(pst_path, 'rb') as f:
        pst_bytes = f.read()
    files = {
        "files": ("fixture.pst", BytesIO(pst_bytes), "application/octet-stream"),
    }
    r = client_user_only.post(
        "/api/v1/media/process-emails",
        files=files,
        data={
            "accept_pst": "true",
            "perform_chunking": "false",
        },
    )
    assert r.status_code in (200, 207)
    data = r.json()
    res = data.get("results") or []
    assert isinstance(res, list) and len(res) >= 1
    item = None
    for entry in res:
        if not isinstance(entry, dict):
            continue
        if entry.get("status") == "Error":
            continue
        md = entry.get("metadata") or {}
        emd = md.get("email") or {}
        if emd:
            item = emd
            break
    assert item is not None, "No successful email metadata found in PST results"
    # Ensure recipients and date appear (format-agnostic checks)
    assert (item.get('to') or item.get('cc') or item.get('bcc')), f"No recipients found in metadata: {item}"
    assert item.get('date'), f"No date found in metadata: {item}"


def test_process_emails_endpoint_pst_feature_flag_behavior(client_user_only):


    # Without pypff installed, uploading a small .pst with accept_pst=true should return informative error and grouping keyword
    placeholder = b"!pst placeholder!"  # not a real PST
    files = {
        "files": ("emails.pst", BytesIO(placeholder), "application/octet-stream"),
    }
    r = client_user_only.post(
        "/api/v1/media/process-emails",
        files=files,
        data={
            "accept_pst": "true",
        },
    )
    assert r.status_code in (200, 207)
    data = r.json()
    res = data.get("results") or []
    assert len(res) >= 1 and isinstance(res[0], dict)
    item = res[0]
    assert item.get("status") == "Error"
    assert "pst/ost support not enabled" in str(item.get("error", "")).lower()
    kws = item.get("keywords") or []
    assert "email_pst:emails" in kws, f"PST grouping keyword missing: {kws}"
