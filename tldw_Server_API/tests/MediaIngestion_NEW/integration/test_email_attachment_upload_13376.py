"""Policy options reach real /media/add parsing and both persistence graphs."""
import pytest

from tldw_Server_API.tests.Media_Ingestion_Modification.test_email_attachment_policy_13376 import _message
from tldw_Server_API.tests.MediaIngestion_NEW.integration.test_email_offline_ingestion import (
    offline_client as offline_client,
)
from tldw_Server_API.tests.MediaIngestion_NEW.integration.test_email_offline_ingestion import upload


@pytest.mark.integration
@pytest.mark.parametrize('options,expected_children,reason', [
    ({'extract_attachments': 'true'}, 1, None),
    ({'ingest_attachments': 'true', 'extract_attachments': 'false'}, 0, 'disabled'),
    ({'extract_attachments': 'true', 'attachment_mime_denylist': 'message/rfc822'}, 0, 'mime_denied'),
    ({'extract_attachments': 'true', 'attachment_mime_allowlist': 'application/pdf'}, 0, 'mime_not_allowed'),
])
def test_real_upload_persists_only_selected_nested_email(offline_client, options, expected_children, reason):
    result = upload(offline_client, 'outer.eml', _message(), **options)
    assert len(result.get('child_db_results') or []) == expected_children
    descriptor = result['metadata']['email']['attachments'][0]
    assert descriptor.get('extraction_reason') == reason
    assert descriptor['extraction_status'] == ('captured' if expected_children else 'skipped')
    search = offline_client.get('/api/v1/email/search', params={'q': 'subject:Inner'})
    assert search.status_code == 200
    assert len(search.json()['items']) == expected_children
    parent = offline_client.get('/api/v1/email/messages/' + str(result['email_message_id']))
    assert parent.status_code == 200
    assert 'PRIVATE ATTACHMENT BODY' not in parent.json()['body_text']


@pytest.mark.integration
@pytest.mark.parametrize("container", ["zip", "mbox"])
def test_archive_upload_persists_selected_attachment_with_immediate_parent(offline_client, container):
    import io
    import zipfile

    data = _message()
    if container == "zip":
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w") as archive:
            archive.writestr("outer.eml", data)
        result = upload(offline_client, "synthetic.zip", stream.getvalue(),
                        accept_archives="true", extract_attachments="true", max_depth="2")
    else:
        result = upload(offline_client, "synthetic.mbox",
            b"From synthetic@example.invalid Sat Jan 01 00:00:00 2022\n" + data + b"\n",
            accept_mbox="true", extract_attachments="true", max_depth="2")
    found = offline_client.get("/api/v1/email/search", params={"q": "subject:Inner"})
    assert found.status_code == 200
    assert len(found.json()["items"]) == 1
    persisted = result["child_db_results"]
    assert len(persisted) == 2
    root_uuid = next(item["media_uuid"] for item in persisted if item["title"] == "Outer synthetic")
    nested = offline_client.get("/api/v1/email/messages/" + str(found.json()["items"][0]["email_message_id"])).json()
    assert nested["raw_metadata"]["parent_media_uuid"] == root_uuid


@pytest.mark.integration
@pytest.mark.parametrize("max_depth,expected", [("2", 0), ("3", 1)])
def test_eml_upload_persists_only_depth_selected_grandchild(offline_client, max_depth, expected):
    result = upload(offline_client, "outer.eml", _message(grandchild=True),
                    extract_attachments="true", max_depth=max_depth)
    found = offline_client.get("/api/v1/email/search", params={"q": "subject:Deep"})
    assert found.status_code == 200
    assert len(found.json()["items"]) == expected
    if expected:
        persisted = result["child_db_results"]
        assert len(persisted) == 2
        parent_uuid = next(item["media_uuid"] for item in persisted if item["title"] == "Inner synthetic")
        deep = offline_client.get("/api/v1/email/messages/" + str(found.json()["items"][0]["email_message_id"])).json()
        assert deep["raw_metadata"]["parent_media_uuid"] == parent_uuid


@pytest.mark.integration
@pytest.mark.parametrize("container", ["zip", "mbox"])
def test_archive_explicit_metadata_only_keeps_root_without_attachment(offline_client, container):
    import io
    import zipfile

    data = _message()
    if container == "zip":
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w") as archive:
            archive.writestr("outer.eml", data)
        result = upload(offline_client, "synthetic.zip", stream.getvalue(), accept_archives="true",
                        ingest_attachments="true", extract_attachments="false")
    else:
        result = upload(offline_client, "synthetic.mbox",
            b"From synthetic@example.invalid Sat Jan 01 00:00:00 2022\n" + data + b"\n",
            accept_mbox="true", ingest_attachments="true", extract_attachments="false")
    assert len(result["child_db_results"]) == 1
    found = offline_client.get("/api/v1/email/search", params={"q": "subject:Inner"})
    assert found.status_code == 200
    assert found.json()["items"] == []


@pytest.mark.integration
def test_archive_same_named_attachments_keep_distinct_parent_source_paths(offline_client):
    import io
    import zipfile

    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("first.eml", _message())
        archive.writestr("second.eml", _message())
    result = upload(offline_client, "synthetic.zip", stream.getvalue(),
                    accept_archives="true", extract_attachments="true", max_depth="2")
    found = offline_client.get("/api/v1/email/search", params={"q": "subject:Inner"})
    assert found.status_code == 200
    assert len(found.json()["items"]) == 2
    details = [offline_client.get("/api/v1/email/messages/" + str(item["email_message_id"])).json()
               for item in found.json()["items"]]
    root_uuids = {item["media_uuid"] for item in result["child_db_results"] if item["title"] == "Outer synthetic"}
    assert {item["raw_metadata"]["parent_media_uuid"] for item in details} == root_uuids
    assert len({item["source"]["source_key"] for item in details}) == 2


def _bodyless_message(*, nested=False, empty_child=False):
    """Keep genuine parser-generated headers while removing one MIME text body."""
    from email import policy
    from email.parser import BytesParser

    message = BytesParser(policy=policy.default).parsebytes(_message(grandchild=nested))
    target = message.get_payload()[1].get_payload()[0] if empty_child else message
    target.get_payload()[0].set_content("")
    return message.as_bytes()


@pytest.mark.integration
@pytest.mark.parametrize("empty_child", [False, True])
def test_real_upload_persists_bodyless_email_node_and_selected_descendants(offline_client, empty_child):
    result = upload(offline_client, "outer.eml", _bodyless_message(nested=True, empty_child=empty_child),
                    extract_attachments="true", max_depth="3")
    assert result["db_id"] is not None
    found = offline_client.get("/api/v1/email/search", params={"q": ""})
    assert found.status_code == 200
    assert len(found.json()["items"]) == 3
    details = {item["subject"]: item for item in [
        offline_client.get("/api/v1/email/messages/" + str(row["email_message_id"])).json()
        for row in found.json()["items"]]}
    blank = details["Inner synthetic" if empty_child else "Outer synthetic"]
    assert blank["body_text"] == ""
    assert details["Inner synthetic"]["raw_metadata"]["parent_media_uuid"] == details["Outer synthetic"]["media"]["uuid"]
    assert details["Deep synthetic"]["raw_metadata"]["parent_media_uuid"] == details["Inner synthetic"]["media"]["uuid"]


@pytest.mark.integration
@pytest.mark.parametrize("container", ["zip", "mbox"])
def test_archive_persists_bodyless_root_without_creating_container_envelope(offline_client, container):
    import io
    import zipfile

    data = _bodyless_message()
    if container == "zip":
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w") as archive:
            archive.writestr("outer.eml", data)
        result = upload(offline_client, "synthetic.zip", stream.getvalue(),
                        accept_archives="true", extract_attachments="true", max_depth="2")
    else:
        result = upload(offline_client, "synthetic.mbox",
            b"From synthetic@example.invalid Sat Jan 01 00:00:00 2022\n" + data + b"\n",
            accept_mbox="true", extract_attachments="true", max_depth="2")
    assert result.get("db_id") is None
    assert len(result["child_db_results"]) == 2
    found = offline_client.get("/api/v1/email/search", params={"q": ""})
    assert found.status_code == 200
    assert len(found.json()["items"]) == 2
    details = {item["subject"]: item for item in [
        offline_client.get("/api/v1/email/messages/" + str(row["email_message_id"])).json()
        for row in found.json()["items"]]}
    assert details["Outer synthetic"]["body_text"] == ""
    assert details["Inner synthetic"]["raw_metadata"]["parent_media_uuid"] == details["Outer synthetic"]["media"]["uuid"]


@pytest.mark.integration
@pytest.mark.parametrize("bad_status,bad_metadata", [
    ("Success", {}),
    ("Success", {"email": {"subject": "Forged blank"}}),
    ("Error", {"parser_used": "builtin-email", "email": {"format": "text/plain", "headers_map": {}}}),
])
def test_archive_skips_invalid_empty_nodes_and_preserves_failed_archive_guard(offline_client, monkeypatch, bad_status, bad_metadata):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Email import Email_Processing_Lib as lib

    good = lib.process_email_task(file_bytes=_message(), filename="outer.eml", perform_chunking=False)
    invalid = {"status": bad_status, "media_type": "email", "content": "", "metadata": bad_metadata}
    monkeypatch.setattr(lib, "process_eml_archive_bytes", lambda **_kwargs: [invalid, good])
    result = upload(offline_client, "synthetic.zip", b"synthetic archive bytes", accept_archives="true")
    expected_subjects = [] if bad_status == "Error" else ["Outer synthetic"]
    assert len(result.get("child_db_results") or []) == len(expected_subjects)
    found = offline_client.get("/api/v1/email/search", params={"q": ""})
    assert found.status_code == 200
    assert [item["subject"] for item in found.json()["items"]] == expected_subjects


@pytest.mark.integration
@pytest.mark.parametrize("metadata", [{}, {"email": {}}, {"email": {"subject": "Synthetic envelope"}}])
def test_empty_or_malformed_email_envelope_does_not_create_media(offline_client, monkeypatch, metadata):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Email import Email_Processing_Lib as lib

    monkeypatch.setattr(lib, "process_email_task", lambda **_kwargs: {
        "status": "Success", "media_type": "email", "content": "", "metadata": metadata})
    result = upload(offline_client, "outer.eml", _message(), extract_attachments="true")
    assert result.get("db_id") is None
    found = offline_client.get("/api/v1/email/search", params={"q": ""})
    assert found.status_code == 200
    assert found.json()["items"] == []
