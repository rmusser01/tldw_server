# tests/MediaDB2/test_media_files.py
"""
Unit tests for MediaFiles table operations in Media_DB_v2.

Tests cover:
- Inserting media file records
- Retrieving media files by media_id and type
- Listing all files for a media item
- has_original_file check
- Soft-delete and include_deleted behavior
"""
import pytest
from unittest.mock import MagicMock

from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase
from tldw_Server_API.app.core.DB_Management.media_db.errors import DatabaseError
from tldw_Server_API.app.core.DB_Management.media_db.repositories.media_files_repository import (
    MediaFilesRepository,
)


@pytest.fixture
def memory_db():
    """Create an in-memory database instance."""
    db = MediaDatabase(db_path=":memory:", client_id="test_client")
    yield db
    db.close_connection()


@pytest.fixture
def db_with_media(memory_db):
    """Create a database with a single media item for testing."""
    media_id, _, _ = memory_db.add_media_with_keywords(
        title="Test PDF Document",
        content="This is the content of the test document.",
        media_type="pdf",
        url="file:///test/document.pdf",
        keywords=["test", "pdf"]
    )
    return memory_db, media_id


class TestInsertMediaFile:
    """Tests for insert_media_file method."""

    @pytest.mark.unit
    def test_insert_media_file_returns_uuid(self, db_with_media):
        """Test that insert_media_file returns a valid UUID."""
        db, media_id = db_with_media

        file_uuid = db.insert_media_file(
            media_id=media_id,
            file_type="original",
            storage_path="user1/media/1/original.pdf",
            original_filename="document.pdf",
            file_size=12345,
            mime_type="application/pdf",
            checksum="abc123def456"
        )

        assert file_uuid is not None
        assert len(file_uuid) == 36  # UUID format: 8-4-4-4-12

    @pytest.mark.unit
    def test_insert_media_file_minimal_params(self, db_with_media):
        """Test inserting a file with only required parameters."""
        db, media_id = db_with_media

        file_uuid = db.insert_media_file(
            media_id=media_id,
            file_type="original",
            storage_path="path/to/file.pdf"
        )

        assert file_uuid is not None
        # Verify we can retrieve it
        record = db.get_media_file(media_id, "original")
        assert record is not None
        assert record["storage_path"] == "path/to/file.pdf"

    @pytest.mark.unit
    def test_insert_multiple_file_types(self, db_with_media):
        """Test inserting multiple file types for same media."""
        db, media_id = db_with_media

        db.insert_media_file(
            media_id=media_id,
            file_type="original",
            storage_path="path/original.pdf"
        )
        db.insert_media_file(
            media_id=media_id,
            file_type="thumbnail",
            storage_path="path/thumb.png"
        )

        files = db.get_media_files(media_id)
        assert len(files) == 2
        file_types = {f["file_type"] for f in files}
        assert file_types == {"original", "thumbnail"}


class TestGetMediaFile:
    """Tests for get_media_file method."""

    @pytest.mark.unit
    @pytest.mark.parametrize("delete_latest", [False, True])
    def test_get_media_file_selects_latest_matching_registration(self, db_with_media, delete_latest):
        db, media_id = db_with_media
        db.insert_media_file(media_id, "original", "previous.pdf")
        db.insert_media_file(media_id, "original", "latest.pdf")
        db.insert_media_file(media_id, "thumbnail", "thumbnail.png")
        if delete_latest:
            latest = next(row for row in db.get_media_files(media_id) if row["storage_path"] == "latest.pdf")
            db.soft_delete_media_file(latest["id"])

        expected_path = "previous.pdf" if delete_latest else "latest.pdf"
        assert db.get_media_file(media_id, "original")["storage_path"] == expected_path
        assert db.get_media_file(media_id, "original", include_deleted=True)["storage_path"] == "latest.pdf"

    @pytest.mark.unit
    def test_get_media_file_returns_record(self, db_with_media):
        """Test that get_media_file returns the correct record."""
        db, media_id = db_with_media

        db.insert_media_file(
            media_id=media_id,
            file_type="original",
            storage_path="path/to/file.pdf",
            original_filename="my_document.pdf",
            file_size=5000,
            mime_type="application/pdf",
            checksum="sha256hash"
        )

        record = db.get_media_file(media_id, "original")

        assert record is not None
        assert record["storage_path"] == "path/to/file.pdf"
        assert record["original_filename"] == "my_document.pdf"
        assert record["file_size"] == 5000
        assert record["mime_type"] == "application/pdf"
        assert record["checksum"] == "sha256hash"

    @pytest.mark.unit
    def test_get_media_file_returns_none_when_missing(self, db_with_media):
        """Test that get_media_file returns None for non-existent file."""
        db, media_id = db_with_media

        record = db.get_media_file(media_id, "original")
        assert record is None

    @pytest.mark.unit
    def test_get_media_file_respects_file_type(self, db_with_media):
        """Test that get_media_file correctly filters by file_type."""
        db, media_id = db_with_media

        db.insert_media_file(
            media_id=media_id,
            file_type="original",
            storage_path="path/original.pdf"
        )
        db.insert_media_file(
            media_id=media_id,
            file_type="thumbnail",
            storage_path="path/thumb.png"
        )

        original = db.get_media_file(media_id, "original")
        thumbnail = db.get_media_file(media_id, "thumbnail")

        assert original["storage_path"] == "path/original.pdf"
        assert thumbnail["storage_path"] == "path/thumb.png"


class TestGetMediaFiles:
    """Tests for get_media_files method."""

    @pytest.mark.unit
    def test_get_media_files_returns_all_files(self, db_with_media):
        """Test that get_media_files returns all files for media."""
        db, media_id = db_with_media

        db.insert_media_file(media_id=media_id, file_type="original", storage_path="a.pdf")
        db.insert_media_file(media_id=media_id, file_type="thumbnail", storage_path="b.png")
        db.insert_media_file(media_id=media_id, file_type="preview", storage_path="c.jpg")

        files = db.get_media_files(media_id)
        assert len(files) == 3

    @pytest.mark.unit
    def test_get_media_files_returns_empty_list_when_none(self, db_with_media):
        """Test that get_media_files returns empty list when no files."""
        db, media_id = db_with_media

        files = db.get_media_files(media_id)
        assert files == []


class TestHasOriginalFile:
    """Tests for has_original_file method."""

    @pytest.mark.unit
    def test_has_original_file_returns_false_when_none(self, db_with_media):
        """Test that has_original_file returns False when no file."""
        db, media_id = db_with_media

        assert db.has_original_file(media_id) is False

    @pytest.mark.unit
    def test_has_original_file_returns_true_when_exists(self, db_with_media):
        """Test that has_original_file returns True when file exists."""
        db, media_id = db_with_media

        db.insert_media_file(
            media_id=media_id,
            file_type="original",
            storage_path="path/file.pdf"
        )

        assert db.has_original_file(media_id) is True

    @pytest.mark.unit
    def test_has_original_file_ignores_other_types(self, db_with_media):
        """Test that has_original_file only checks for 'original' type."""
        db, media_id = db_with_media

        db.insert_media_file(
            media_id=media_id,
            file_type="thumbnail",  # Not 'original'
            storage_path="path/thumb.png"
        )

        assert db.has_original_file(media_id) is False


class TestSoftDeleteMediaFile:
    """Tests for soft_delete_media_file method."""

    @pytest.mark.unit
    def test_soft_delete_hides_file(self, db_with_media):
        """Test that soft-deleted files are not returned by default."""
        db, media_id = db_with_media

        db.insert_media_file(
            media_id=media_id,
            file_type="original",
            storage_path="path/file.pdf"
        )

        record = db.get_media_file(media_id, "original")
        file_id = record["id"]

        db.soft_delete_media_file(file_id)

        # Should not find deleted record by default
        assert db.get_media_file(media_id, "original") is None
        assert db.has_original_file(media_id) is False

    @pytest.mark.unit
    def test_soft_delete_visible_with_include_deleted(self, db_with_media):
        """Test that soft-deleted files are visible with include_deleted=True."""
        db, media_id = db_with_media

        db.insert_media_file(
            media_id=media_id,
            file_type="original",
            storage_path="path/file.pdf"
        )

        record = db.get_media_file(media_id, "original")
        file_id = record["id"]

        db.soft_delete_media_file(file_id)

        # Should find with include_deleted=True
        record = db.get_media_file(media_id, "original", include_deleted=True)
        assert record is not None
        assert record["deleted"] == 1


class TestMediaFilesRepository:
    @pytest.mark.unit
    def test_hard_delete_rolls_back_when_sync_logging_fails(self, db_with_media, monkeypatch):
        db, media_id = db_with_media
        db.insert_media_file(media_id, "original", "old.pdf")
        old_id = db.get_media_file(media_id)["id"]
        db.insert_media_file(media_id, "original", "new.pdf")

        def fail_sync(*_args, **_kwargs):
            raise RuntimeError("sync log unavailable")

        with monkeypatch.context() as patch:
            patch.setattr(db, "_log_sync_event", fail_sync)
            with pytest.raises(DatabaseError, match="sync log unavailable"):
                db.soft_delete_media_file(old_id, hard_delete=True)

        assert len(db.get_media_files(media_id, include_deleted=True)) == 2
        db.soft_delete_media_file(old_id, hard_delete=True)
        assert [row["storage_path"] for row in db.get_media_files(media_id, include_deleted=True)] == ["new.pdf"]

    @pytest.mark.unit
    def test_single_file_hard_delete_is_scoped_and_idempotent(self, db_with_media):
        db, media_id = db_with_media
        old_uuid = db.insert_media_file(media_id, "original", "old.pdf")
        old_id = db.get_media_file(media_id)["id"]
        new_uuid = db.insert_media_file(media_id, "original", "new.pdf")
        thumbnail_uuid = db.insert_media_file(media_id, "thumbnail", "thumbnail.png")

        db.soft_delete_media_file(old_id, hard_delete=True)
        db.soft_delete_media_file(old_id, hard_delete=True)

        assert {row["uuid"] for row in db.get_media_files(media_id, include_deleted=True)} == {
            new_uuid, thumbnail_uuid,
        }
        deletes = [row for row in db.get_sync_log_entries()
                   if row["entity_uuid"] == old_uuid and row["operation"] == "delete"]
        assert len(deletes) == 1
        assert deletes[0]["payload"]["hard_delete"] is True

    @pytest.mark.unit
    def test_repository_lists_active_files(self, db_with_media):
        db, media_id = db_with_media
        repo = MediaFilesRepository.from_legacy_db(db)

        repo.insert(media_id=media_id, file_type="original", storage_path="repo/original.pdf")
        repo.insert(media_id=media_id, file_type="thumbnail", storage_path="repo/thumb.png")

        assert repo.has_original_file(media_id) is True
        assert len(repo.list_for_media(media_id)) == 2

    @pytest.mark.unit
    def test_repository_soft_delete_hides_file(self, db_with_media):
        db, media_id = db_with_media
        repo = MediaFilesRepository.from_legacy_db(db)

        repo.insert(media_id=media_id, file_type="original", storage_path="repo/original.pdf")
        record = repo.get_for_media(media_id, "original")

        repo.soft_delete(record["id"])

        assert repo.get_for_media(media_id, "original") is None
        assert repo.get_for_media(media_id, "original", include_deleted=True)["deleted"] == 1

    @pytest.mark.unit
    def test_repository_insert_logs_dict_payloads(self, db_with_media):
        db, media_id = db_with_media
        repo = MediaFilesRepository.from_legacy_db(db)

        file_uuid = repo.insert(
            media_id=media_id,
            file_type="original",
            storage_path="repo/original.pdf",
        )

        log_entry = next(
            entry
            for entry in db.get_sync_log_entries()
            if entry["entity"] == "MediaFiles" and entry["entity_uuid"] == file_uuid
        )

        assert log_entry["operation"] == "create"
        assert log_entry["payload"]["media_id"] == media_id
        assert log_entry["payload"]["storage_path"] == "repo/original.pdf"

    @pytest.mark.unit
    def test_repository_hard_delete_logs_non_empty_entity_uuid(self, db_with_media):
        db, media_id = db_with_media
        repo = MediaFilesRepository.from_legacy_db(db)

        first_uuid = repo.insert(
            media_id=media_id,
            file_type="original",
            storage_path="repo/original.pdf",
        )
        second_uuid = repo.insert(
            media_id=media_id,
            file_type="thumbnail",
            storage_path="repo/thumb.png",
        )

        initial_change_count = len(db.get_sync_log_entries())
        repo.soft_delete_for_media(media_id, hard_delete=True)
        delete_entries = db.get_sync_log_entries(since_change_id=initial_change_count)

        delete_file_uuids = {
            entry["entity_uuid"]
            for entry in delete_entries
            if entry["entity"] == "MediaFiles" and entry["operation"] == "delete"
        }

        assert delete_file_uuids == {first_uuid, second_uuid}

    @pytest.mark.unit
    def test_soft_delete_increments_version(self, db_with_media):
        """Test that soft-delete increments the version number."""
        db, media_id = db_with_media

        db.insert_media_file(
            media_id=media_id,
            file_type="original",
            storage_path="path/file.pdf"
        )

        record_before = db.get_media_file(media_id, "original")
        file_id = record_before["id"]
        version_before = record_before["version"]

        db.soft_delete_media_file(file_id)

        record_after = db.get_media_file(media_id, "original", include_deleted=True)
        assert record_after["version"] == version_before + 1

    @pytest.mark.unit
    def test_soft_delete_nonexistent_is_safe(self, db_with_media):
        """Test that soft-deleting a non-existent file doesn't raise."""
        db, media_id = db_with_media

        # Should not raise
        db.soft_delete_media_file(99999)


class TestGetMediaFilesWithDeleted:
    """Tests for get_media_files with include_deleted parameter."""

    @pytest.mark.unit
    def test_get_media_files_excludes_deleted_by_default(self, db_with_media):
        """Test that get_media_files excludes deleted files by default."""
        db, media_id = db_with_media

        db.insert_media_file(media_id=media_id, file_type="original", storage_path="a.pdf")
        db.insert_media_file(media_id=media_id, file_type="thumbnail", storage_path="b.png")

        # Delete one
        record = db.get_media_file(media_id, "original")
        db.soft_delete_media_file(record["id"])

        # Should only return non-deleted
        files = db.get_media_files(media_id)
        assert len(files) == 1
        assert files[0]["file_type"] == "thumbnail"

    @pytest.mark.unit
    def test_get_media_files_includes_deleted_when_requested(self, db_with_media):
        """Test that get_media_files includes deleted files when requested."""
        db, media_id = db_with_media

        db.insert_media_file(media_id=media_id, file_type="original", storage_path="a.pdf")
        db.insert_media_file(media_id=media_id, file_type="thumbnail", storage_path="b.png")

        # Delete one
        record = db.get_media_file(media_id, "original")
        db.soft_delete_media_file(record["id"])

        # Should return both with include_deleted
        files = db.get_media_files(media_id, include_deleted=True)
        assert len(files) == 2


class TestMediaFileRuntimeHelperForwarding:
    @pytest.mark.unit
    def test_insert_media_file_forwards_to_repository(self, monkeypatch):
        import importlib

        helper_module = importlib.import_module(
            "tldw_Server_API.app.core.DB_Management.media_db.runtime.media_file_ops"
        )

        db = object()
        repo = MagicMock()
        repo.insert.return_value = "helper-uuid"
        from_legacy_db = MagicMock(return_value=repo)
        monkeypatch.setattr(MediaFilesRepository, "from_legacy_db", from_legacy_db)

        result = helper_module.insert_media_file(
            db,
            11,
            "original",
            "files/original.pdf",
            original_filename="original.pdf",
            file_size=123,
            mime_type="application/pdf",
            checksum="abc",
        )

        assert result == "helper-uuid"
        from_legacy_db.assert_called_once_with(db)
        repo.insert.assert_called_once_with(
            media_id=11,
            file_type="original",
            storage_path="files/original.pdf",
            original_filename="original.pdf",
            file_size=123,
            mime_type="application/pdf",
            checksum="abc",
        )

    @pytest.mark.unit
    def test_get_media_file_forwards_to_repository(self, monkeypatch):
        import importlib

        helper_module = importlib.import_module(
            "tldw_Server_API.app.core.DB_Management.media_db.runtime.media_file_ops"
        )

        db = object()
        repo = MagicMock()
        repo.get_for_media.return_value = {"id": 7}
        from_legacy_db = MagicMock(return_value=repo)
        monkeypatch.setattr(MediaFilesRepository, "from_legacy_db", from_legacy_db)

        result = helper_module.get_media_file(db, 11, "thumbnail", include_deleted=True)

        assert result == {"id": 7}
        from_legacy_db.assert_called_once_with(db)
        repo.get_for_media.assert_called_once_with(
            media_id=11,
            file_type="thumbnail",
            include_deleted=True,
        )

    @pytest.mark.unit
    def test_get_media_files_and_has_original_file_forwards_to_repository(self, monkeypatch):
        import importlib

        helper_module = importlib.import_module(
            "tldw_Server_API.app.core.DB_Management.media_db.runtime.media_file_ops"
        )

        db = object()
        repo = MagicMock()
        repo.list_for_media.return_value = [{"id": 1}, {"id": 2}]
        repo.has_original_file.return_value = True
        from_legacy_db = MagicMock(return_value=repo)
        monkeypatch.setattr(MediaFilesRepository, "from_legacy_db", from_legacy_db)

        files = helper_module.get_media_files(db, 11, include_deleted=True)
        has_original = helper_module.has_original_file(db, 11)

        assert files == [{"id": 1}, {"id": 2}]
        assert has_original is True
        assert from_legacy_db.call_args_list == [((db,),), ((db,),)]
        repo.list_for_media.assert_called_once_with(media_id=11, include_deleted=True)
        repo.has_original_file.assert_called_once_with(11)

    @pytest.mark.unit
    def test_soft_delete_media_file_and_media_files_for_media_forwards_to_repository(self, monkeypatch):
        import importlib

        helper_module = importlib.import_module(
            "tldw_Server_API.app.core.DB_Management.media_db.runtime.media_file_ops"
        )

        db = object()
        repo = MagicMock()
        from_legacy_db = MagicMock(return_value=repo)
        monkeypatch.setattr(MediaFilesRepository, "from_legacy_db", from_legacy_db)

        helper_module.soft_delete_media_file(db, 91)
        helper_module.soft_delete_media_files_for_media(db, 11, hard_delete=True)

        assert from_legacy_db.call_args_list == [((db,),), ((db,),)]
        repo.soft_delete.assert_called_once_with(91, hard_delete=False)
        repo.soft_delete_for_media.assert_called_once_with(media_id=11, hard_delete=True)
