"""Legacy metadata compatibility when rebuilding normalized email content."""

import json

import pytest

from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase
from tldw_Server_API.app.core.DB_Management.media_db.runtime.email_persisted_content import read_persisted_email_content


@pytest.mark.unit
def test_persisted_email_read_preserves_legacy_graph_metadata_when_version_stripped_it(tmp_path):
    db = MediaDatabase(db_path=str(tmp_path / "legacy.db"), client_id="legacy-test")
    metadata = {
        "email": {
            "message_id": "<legacy@example.test>",
            "subject": "Original subject",
            "from": "sender@example.test",
            "attachments": [{"name": "original.bin"}],
        }
    }
    try:
        media_id, _, _ = db.add_media_with_keywords(
            url="email://legacy",
            title="Original subject",
            media_type="email",
            content="Original body",
            keywords=[],
            safe_metadata=json.dumps(metadata),
        )
        db.upsert_email_message_graph(
            media_id=media_id, metadata=metadata, body_text="Original body", tenant_id="owner"
        )
        # Upgrade fixture: older safe-metadata policy dropped email fields, while graph retained them.
        db.execute_query(
            "UPDATE DocumentVersions SET safe_metadata = ?, version = version + 1 WHERE media_id = ?",
            (json.dumps({"title": "Original subject"}), media_id),
            commit=True,
        )
        saved_metadata, body = read_persisted_email_content(db, media_id, tenant_id="owner")
        assert saved_metadata["email"] == metadata["email"]
        assert body == "Original body"
        other_metadata, _ = read_persisted_email_content(db, media_id, tenant_id="other")
        assert "email" not in other_metadata
    finally:
        db.close_connection()
