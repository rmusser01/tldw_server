# tests/Skills/unit/test_skills_service.py
#
# Unit tests for the SkillsService class
#
import asyncio
import shutil
import stat
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from contextlib import contextmanager, suppress
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from loguru import logger

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    CharactersRAGDBError,
    ConflictError,
)
from tldw_Server_API.app.core.Infrastructure.distributed_lock import FileLock
from tldw_Server_API.app.core.Skills.exceptions import (
    SkillConflictError,
    SkillNotFoundError,
    SkillParseError,
    SkillsError,
    SkillStorageError,
    SkillValidationError,
)
from tldw_Server_API.app.core.Skills.skills_service import (
    SkillMetadata,
    SkillsService,
    _public_import_preview_error,
)

pytestmark = pytest.mark.unit


@contextmanager
def _capture_skills_service_logs():
    """Capture service logs without changing the process-wide logger format."""
    messages: list[str] = []
    sink_id = logger.add(
        lambda message: messages.append(str(message)),
        filter=lambda record: record["name"]
        == "tldw_Server_API.app.core.Skills.skills_service",
    )
    try:
        yield messages
    finally:
        logger.remove(sink_id)


async def _restart_and_sync(service: SkillsService) -> SkillsService:
    """Create a replacement service and trigger its first registry sync."""
    restarted = SkillsService(
        user_id=service.user_id,
        base_path=service.base_path,
        db=service._get_db(),
        integrity_resolver=service.integrity_resolver,
    )
    await restarted.list_skills()
    return restarted


class TestSkillMetadata:
    """Tests for SkillMetadata class."""

    def test_to_dict_and_from_dict_roundtrip(self):
        """Test that metadata can be serialized and deserialized."""
        now = datetime.now()
        original = SkillMetadata(
            id="test-uuid",
            name="test-skill",
            description="A test skill",
            argument_hint="[arg]",
            disable_model_invocation=True,
            user_invocable=False,
            allowed_tools=["Read", "Grep"],
            model="gpt-4",
            context="fork",
            directory_path="/path/to/skill",
            content_hash="abc123",
            created_at=now,
            last_modified=now,
            version=2,
        )

        data = original.to_dict()
        restored = SkillMetadata.from_dict(data)

        assert restored.id == original.id
        assert restored.name == original.name
        assert restored.description == original.description
        assert restored.argument_hint == original.argument_hint
        assert restored.disable_model_invocation == original.disable_model_invocation
        assert restored.user_invocable == original.user_invocable
        assert restored.allowed_tools == original.allowed_tools
        assert restored.model == original.model
        assert restored.context == original.context
        assert restored.directory_path == original.directory_path
        assert restored.content_hash == original.content_hash
        assert restored.version == original.version


class TestSkillsService:
    """Tests for the SkillsService class."""

    @pytest.fixture
    def temp_base_path(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def service(self, temp_base_path):
        """Create a SkillsService instance for testing."""
        db_path = temp_base_path / "ChaChaNotes.db"
        chacha_db = CharactersRAGDB(db_path=db_path, client_id="test_client")
        service = SkillsService(user_id=1, base_path=temp_base_path, db=chacha_db)
        yield service
        chacha_db.close_connection()

    def test_cleanup_failure_log_omits_paths_and_exception_details(
        self,
        service,
        monkeypatch,
    ):
        """Cleanup retries identify the operation without exposing filesystem details."""
        cleanup_dir = service._ensure_cleanup_directory()
        cleanup_path = cleanup_dir / "cleanup-entry"
        cleanup_path.mkdir()

        def _fail_cleanup(*_args, **_kwargs):
            raise OSError("permission denied at /private/skills/secret")

        original_rmtree = shutil.rmtree
        monkeypatch.setattr(shutil, "rmtree", _fail_cleanup)
        try:
            with _capture_skills_service_logs() as messages:
                removed = service._remove_cleanup_path_best_effort(cleanup_path)
        finally:
            monkeypatch.setattr(shutil, "rmtree", original_rmtree)

        output = "".join(messages)
        assert removed is False
        assert "cleanup-entry" in output
        assert "OSError" in output
        assert "/private/" not in output
        assert str(service.base_path) not in output

    def test_reconciliation_failure_log_omits_paths_and_exception_details(
        self,
        service,
        monkeypatch,
    ):
        """Startup reconciliation logs safe archive identity and error type only."""
        service._ensure_trash_directory()
        candidate = service.trash_dir / ".purging-archive-123"
        candidate.mkdir()

        def _fail_lookup(_archive_id: str):
            raise CharactersRAGDBError("database failed at /private/skills/registry.db")

        monkeypatch.setattr(service, "_registry_row_for_archive_id", _fail_lookup)
        with _capture_skills_service_logs() as messages:
            service._reconcile_orphaned_archives()

        output = "".join(messages)
        assert "archive-123" in output
        assert "CharactersRAGDBError" in output
        assert "/private/" not in output
        assert str(service.base_path) not in output

    @pytest.mark.asyncio
    async def test_purge_rollback_failure_log_omits_paths_and_exception_details(
        self,
        service,
        monkeypatch,
    ):
        """Rollback failures retain safe skill context without raw exception text."""
        created = await service.create_skill("purge-log", "Body")
        await service.delete_skill("purge-log", expected_version=created["version"])
        deleted_row = service._get_db().get_skill_registry("purge-log", include_deleted=True)
        assert deleted_row is not None
        archive_dir = Path(deleted_row["directory_path"])
        staging_dir = service.trash_dir / f".purging-{deleted_row['uuid']}"
        original_move = service._move_skill_dir

        def _fail_registry_purge(*_args, **_kwargs):
            raise CharactersRAGDBError("database failed at /private/skills/registry.db")

        def _fail_rollback(source: Path, destination: Path) -> None:
            if source == staging_dir and destination == archive_dir:
                raise OSError("rollback failed at /private/skills/archive")
            original_move(source, destination)

        monkeypatch.setattr(service._get_db(), "purge_skill_registry", _fail_registry_purge)
        monkeypatch.setattr(service, "_move_skill_dir", _fail_rollback)

        with _capture_skills_service_logs() as messages:
            with pytest.raises(SkillsError, match="Failed to purge"):
                await service.purge_skill(
                    "purge-log",
                    expected_version=deleted_row["version"],
                )

        output = "".join(messages)
        assert "purge-log" in output
        assert "user 1" in output
        assert "OSError" in output
        assert "/private/" not in output
        assert str(service.base_path) not in output

    @pytest.mark.asyncio
    async def test_create_skill_simple(self, service):
        """Test creating a simple skill."""
        content = """---
name: test-skill
description: A test skill
---

This is the skill content.
$ARGUMENTS will be replaced.
"""
        result = await service.create_skill("test-skill", content)

        assert result["name"] == "test-skill"
        assert result["description"] == "A test skill"
        assert "This is the skill content" in result["content"]
        assert result["version"] == 1

    @pytest.mark.asyncio
    async def test_create_skill_rejects_flow_yaml_name_mismatch(self, service):
        """Canonical create identity must use the backend's parsed YAML value."""
        content = """---
{name: other-skill, description: Mismatch}
---

Body
"""

        with pytest.raises(SkillValidationError, match="must match canonical name"):
            await service.create_skill("canonical-skill", content)

    @pytest.mark.asyncio
    async def test_create_skill_with_supporting_files(self, service):
        """Test creating a skill with supporting files."""
        content = "Skill content"
        supporting = {
            "reference.md": "Reference docs",
            "examples.md": "Example usage",
        }

        result = await service.create_skill(
            "with-files",
            content,
            supporting_files=supporting,
        )

        assert result["name"] == "with-files"
        assert result["supporting_files"] is not None
        assert "reference.md" in result["supporting_files"]
        assert result["supporting_files"]["reference.md"] == "Reference docs"

    @pytest.mark.asyncio
    async def test_create_skill_conflict(self, service):
        """Test that creating a duplicate skill raises ConflictError."""
        content = "Skill content"
        await service.create_skill("duplicate", content)

        with pytest.raises(SkillConflictError, match="already exists"):
            await service.create_skill("duplicate", content)

    @pytest.mark.asyncio
    async def test_create_skill_invalid_name_rejected(self, service):
        """Service-level name validation should reject invalid skill names."""
        with pytest.raises(SkillValidationError, match="Invalid skill name"):
            await service.create_skill("Invalid_Name!", "content")

    @pytest.mark.asyncio
    async def test_create_skill_supporting_file_traversal_rejected(self, service):
        """Supporting file names must not include traversal or path separators."""
        with pytest.raises(SkillValidationError, match="Invalid supporting file name"):
            await service.create_skill(
                "safe-skill",
                "Content",
                supporting_files={"../escape.md": "bad"},
            )

    @pytest.mark.asyncio
    async def test_create_skill_normalizes_name(self, service):
        """Test that skill names are normalized to lowercase."""
        content = "Skill content"
        result = await service.create_skill("MySkill", content)

        assert result["name"] == "myskill"

    @pytest.mark.asyncio
    async def test_get_skill(self, service):
        """Test getting a skill by name."""
        content = """---
description: A test skill
---

Skill content here.
"""
        await service.create_skill("get-test", content)
        result = await service.get_skill("get-test")

        assert result["name"] == "get-test"
        assert result["description"] == "A test skill"
        assert "Skill content here" in result["content"]

    @pytest.mark.asyncio
    async def test_get_skill_not_found(self, service):
        """Test that getting a non-existent skill raises NotFoundError."""
        with pytest.raises(SkillNotFoundError):
            await service.get_skill("nonexistent")

    @pytest.mark.asyncio
    async def test_list_model_visible_skills_page_filters_and_counts_once(self, service, monkeypatch):
        """Catalog pages use one matching registry query and visibility pass."""
        await service.create_skill("visible", "---\nuser-invocable: true\n---\nVisible")
        await service.create_skill("hidden", "---\nuser-invocable: false\n---\nHidden")
        await service.create_skill(
            "manual-only",
            "---\ndisable-model-invocation: true\n---\nManual",
        )
        await service._sync_registry_async(force=True)
        monkeypatch.setattr(service, "_sync_registry_async", AsyncMock())
        calls = 0
        integrity_calls: list[str] = []
        original = service._get_db().list_skill_registry

        def counted_list(*args, **kwargs):
            nonlocal calls
            calls += 1
            return original(*args, **kwargs)

        def allowed(name, *, purpose):
            integrity_calls.append(name)
            assert purpose == "skill_discovery"
            return True

        monkeypatch.setattr(service._get_db(), "list_skill_registry", counted_list)
        monkeypatch.setattr(service, "_is_skill_allowed", allowed)

        items, total = await service.list_model_visible_skills_page(limit=10, offset=0)

        assert [item.name for item in items] == ["visible"]
        assert total == 1
        assert calls == 1
        assert integrity_calls == ["visible"]

    @pytest.mark.asyncio
    async def test_model_visible_page_applies_schema_defaults_to_null_flags(
        self,
        service,
        monkeypatch,
    ):
        """Legacy nullable registry flags retain their documented visibility defaults."""
        monkeypatch.setattr(service, "_sync_registry_async", AsyncMock())
        monkeypatch.setattr(
            service._get_db(),
            "list_skill_registry",
            lambda **_kwargs: [
                {
                    "uuid": "legacy-null-flags",
                    "name": "legacy-visible",
                    "user_invocable": None,
                    "disable_model_invocation": None,
                    "version": 1,
                }
            ],
        )
        monkeypatch.setattr(service, "_is_skill_allowed", lambda *_args, **_kwargs: True)

        items, total = await service.list_model_visible_skills_page()

        assert total == 1
        assert [item.name for item in items] == ["legacy-visible"]
        assert items[0].user_invocable is True
        assert items[0].disable_model_invocation is False

    @pytest.mark.asyncio
    async def test_get_model_visible_skill_metadata_hides_non_model_skills(self, service):
        """Model-disabled skills are indistinguishable from missing catalog entries."""
        await service.create_skill(
            "manual-only",
            "---\ndisable-model-invocation: true\n---\nManual",
        )

        with pytest.raises(SkillNotFoundError):
            await service.get_model_visible_skill_metadata("manual-only")

    @pytest.mark.asyncio
    async def test_model_visible_metadata_hides_integrity_blocked_skills(self, service, monkeypatch):
        """Integrity-blocked skills are excluded from page and exact discovery."""
        await service.create_skill("blocked", "---\n---\nBlocked")
        monkeypatch.setattr(service, "_is_skill_allowed", lambda *_args, **_kwargs: False)

        items, total = await service.list_model_visible_skills_page()

        assert items == []
        assert total == 0
        with pytest.raises(SkillNotFoundError):
            await service.get_model_visible_skill_metadata("blocked")

    @pytest.mark.asyncio
    async def test_model_visible_metadata_omits_skill_content_and_supporting_files(self, service, monkeypatch):
        """Catalog helpers return registry metadata without parsing a full skill."""
        await service.create_skill(
            "metadata-only",
            "---\ndescription: Metadata only\n---\nSecret instructions",
            supporting_files={"reference.md": "Private supporting text"},
        )
        monkeypatch.setattr(service, "_is_skill_allowed", lambda *_args, **_kwargs: True)
        monkeypatch.setattr(
            service,
            "_parse_verified_skill_directory",
            lambda *_args, **_kwargs: pytest.fail("metadata helpers must not parse full Skill directories"),
        )
        monkeypatch.setattr(
            service,
            "_parse_unchecked_skill_directory",
            lambda *_args, **_kwargs: pytest.fail("metadata helpers must not parse full Skill directories"),
        )

        metadata = await service.get_model_visible_skill_metadata("metadata-only")
        items, _ = await service.list_model_visible_skills_page()

        assert [item.name for item in items] == ["metadata-only"]
        assert metadata.description == "Metadata only"
        assert metadata.directory_path
        assert metadata.content_hash
        for item in [metadata, *items]:
            assert not hasattr(item, "content")
            assert not hasattr(item, "raw_content")
            assert not hasattr(item, "supporting_files")

    @pytest.mark.asyncio
    async def test_list_model_visible_skills_page_offloads_verified_load(self, service, monkeypatch):
        """The catalog page's synchronous registry and integrity work leaves the event loop."""
        event_loop_thread = threading.get_ident()
        worker_threads: list[int] = []

        def record_thread(*_args, **_kwargs):
            worker_threads.append(threading.get_ident())
            return [], 0

        monkeypatch.setattr(service, "_list_model_visible_skills_page_sync", record_thread)

        assert await service.list_model_visible_skills_page() == ([], 0)
        assert worker_threads != [event_loop_thread]

    @pytest.mark.asyncio
    async def test_get_skill_verified_load_offload_preserves_content_and_supporting_files(self, service):
        """Verified loads retain the established full-content response after offloading."""
        await service.create_skill(
            "verified-load",
            "---\ndescription: Full payload\n---\nSkill instructions",
            supporting_files={"reference.md": "Reference text"},
        )

        result = await service.get_skill("verified-load")

        assert result["content"] == "Skill instructions"
        assert result["raw_content"] == "---\ndescription: Full payload\n---\nSkill instructions"
        assert result["supporting_files"] == {"reference.md": "Reference text"}

    @pytest.mark.asyncio
    async def test_list_skills(self, service):
        """Test listing skills."""
        await service.create_skill("skill-a", "Content A")
        await service.create_skill("skill-b", "Content B")
        await service.create_skill("skill-c", "Content C")

        skills = await service.list_skills()

        assert len(skills) == 3
        names = [s.name for s in skills]
        assert "skill-a" in names
        assert "skill-b" in names
        assert "skill-c" in names

    @pytest.mark.asyncio
    async def test_list_skills_filters_hidden(self, service):
        """Test that hidden skills are filtered by default."""
        # Create a visible skill
        await service.create_skill(
            "visible",
            """---
user-invocable: true
---
Content""",
        )

        # Create a hidden skill
        await service.create_skill(
            "hidden",
            """---
user-invocable: false
---
Content""",
        )

        # Default should filter hidden
        skills = await service.list_skills()
        names = [s.name for s in skills]
        assert "visible" in names
        assert "hidden" not in names

        # With include_hidden should show all
        skills = await service.list_skills(include_hidden=True)
        names = [s.name for s in skills]
        assert "visible" in names
        assert "hidden" in names

    @pytest.mark.asyncio
    async def test_list_skills_pagination(self, service):
        """Test skill listing with pagination."""
        for i in range(5):
            await service.create_skill(f"skill-{i:02d}", f"Content {i}")

        # Get first page
        page1 = await service.list_skills(limit=2, offset=0)
        assert len(page1) == 2

        # Get second page
        page2 = await service.list_skills(limit=2, offset=2)
        assert len(page2) == 2

        # Names should be different
        names1 = {s.name for s in page1}
        names2 = {s.name for s in page2}
        assert names1.isdisjoint(names2)

    @pytest.mark.asyncio
    async def test_list_skills_search_filters_before_pagination(self, service):
        """Search should find matching skills outside the first unfiltered page."""
        for i in range(12):
            await service.create_skill(
                f"alpha-{i:02d}",
                """---
description: General utility skill
---

Common content
""",
            )
        await service.create_skill(
            "omega-research",
            """---
description: Needle workflow for longform research synthesis
---

Use this for longform synthesis.
""",
        )

        skills = await service.list_skills(q="needle", limit=5, offset=0)

        assert [skill.name for skill in skills] == ["omega-research"]
        assert await service.get_total_count(q="needle") == 1

    @pytest.mark.asyncio
    async def test_list_skills_filters_and_sorts_before_pagination(self, service):
        """Power-user filters and sort should apply before pagination."""
        for i in range(12):
            await service.create_skill(
                f"alpha-{i:02d}",
                """---
description: General utility skill
context: inline
---

Common content
""",
            )
        await service.create_skill(
            "beta-first",
            """---
description: Forked skill with tools
context: fork
allowed-tools:
  - Read
model: gpt-4o
---

Use this with tools.
""",
        )
        await service.create_skill(
            "beta-second",
            """---
description: Another forked skill with tools
context: fork
allowed-tools:
  - Grep
model: gpt-4o
---

Use this with tools too.
""",
        )

        skills = await service.list_skills(
            context="fork",
            has_tools=True,
            model="gpt-4o",
            sort="name",
            order="desc",
            limit=1,
            offset=0,
        )

        assert [skill.name for skill in skills] == ["beta-second"]
        assert await service.get_total_count(
            context="fork",
            has_tools=True,
            model="gpt-4o",
        ) == 2

    @pytest.mark.asyncio
    async def test_list_skills_explicit_hidden_filter_overrides_default_visibility(
        self,
        service,
    ):
        """Filtering for user_invocable=false should find hidden skills."""
        await service.create_skill("visible", """---
user-invocable: true
---
Content""")
        await service.create_skill("hidden", """---
user-invocable: false
---
Content""")

        skills = await service.list_skills(user_invocable=False)

        assert [skill.name for skill in skills] == ["hidden"]
        assert await service.get_total_count(user_invocable=False) == 1

    @pytest.mark.asyncio
    async def test_update_skill_content(self, service):
        """Test updating skill content."""
        await service.create_skill("update-test", "Original content")

        result = await service.update_skill(
            "update-test",
            content="""---
description: Updated description
---

New content here.
""",
        )

        assert result["description"] == "Updated description"
        assert "New content" in result["content"]
        assert result["version"] == 2

    @pytest.mark.asyncio
    async def test_update_skill_rejects_flow_yaml_name_mismatch(self, service):
        """Editing raw source cannot change the canonical registry identity."""
        created = await service.create_skill("canonical-update", "Original")
        content = """---
{name: other-skill, description: Mismatch}
---

Body
"""

        with pytest.raises(SkillValidationError, match="must match canonical name"):
            await service.update_skill(
                "canonical-update",
                content=content,
                expected_version=created["version"],
            )

    @pytest.mark.asyncio
    async def test_update_skill_supporting_files(self, service):
        """Test updating supporting files."""
        await service.create_skill(
            "files-test",
            "Content",
            supporting_files={"old.md": "Old file"},
        )

        result = await service.update_skill(
            "files-test",
            supporting_files={
                "new.md": "New file",
                "old.md": None,  # Delete old file
            },
        )

        assert "new.md" in result["supporting_files"]
        assert "old.md" not in result["supporting_files"]

    @pytest.mark.asyncio
    async def test_update_skill_not_found(self, service):
        """Test that updating a non-existent skill raises NotFoundError."""
        with pytest.raises(SkillNotFoundError):
            await service.update_skill("nonexistent", content="New")

    @pytest.mark.asyncio
    async def test_update_skill_version_conflict(self, service):
        """Test optimistic locking with version mismatch."""
        await service.create_skill("version-test", "Content")

        # Update successfully
        await service.update_skill("version-test", content="Updated", expected_version=1)

        # Try to update with stale version
        with pytest.raises(SkillConflictError, match="modified"):
            await service.update_skill("version-test", content="Again", expected_version=1)

    @pytest.mark.asyncio
    async def test_update_skill_registry_conflict_restores_original_skill_file(self, service, monkeypatch):
        """A registry conflict must not leave the staged SKILL.md content on disk."""
        created = await service.create_skill("rollback-test", "Original content")
        skill_file = service._get_skill_dir("rollback-test") / "SKILL.md"
        original_disk_content = skill_file.read_text(encoding="utf-8")

        def _conflict_update(*_args, **_kwargs):
            raise ConflictError("simulated stale version", entity="skill_registry", entity_id="rollback-test")

        monkeypatch.setattr(service._get_db(), "update_skill_registry", _conflict_update)

        with pytest.raises(SkillConflictError):
            await service.update_skill(
                "rollback-test",
                content="Updated content",
                expected_version=created["version"],
            )

        assert skill_file.read_text(encoding="utf-8") == original_disk_content

    @pytest.mark.asyncio
    async def test_update_skill_unexpected_registry_error_restores_original_skill_file(self, service, monkeypatch):
        """Unexpected registry failures must also restore staged SKILL.md content."""
        created = await service.create_skill("rollback-unexpected", "Original content")
        skill_file = service._get_skill_dir("rollback-unexpected") / "SKILL.md"
        original_disk_content = skill_file.read_text(encoding="utf-8")

        def _unexpected_update(*_args, **_kwargs):
            raise RuntimeError("simulated registry timeout")

        monkeypatch.setattr(service._get_db(), "update_skill_registry", _unexpected_update)

        with pytest.raises(SkillsError, match="simulated registry timeout"):
            await service.update_skill(
                "rollback-unexpected",
                content="Updated content",
                expected_version=created["version"],
            )

        assert skill_file.read_text(encoding="utf-8") == original_disk_content

    @pytest.mark.asyncio
    async def test_delete_skill(self, service):
        """Test deleting a skill."""
        await service.create_skill("delete-test", "Content")

        # Verify it exists
        await service.get_skill("delete-test")

        # Delete it
        await service.delete_skill("delete-test")

        # Verify it's gone
        with pytest.raises(SkillNotFoundError):
            await service.get_skill("delete-test")

    @pytest.mark.asyncio
    async def test_delete_skill_not_found(self, service):
        """Test that deleting a non-existent skill raises NotFoundError."""
        with pytest.raises(SkillNotFoundError):
            await service.delete_skill("nonexistent")

    @pytest.mark.asyncio
    async def test_delete_skill_version_conflict(self, service):
        """Test optimistic locking on delete."""
        await service.create_skill("delete-version", "Content")

        # Update to increment version
        await service.update_skill("delete-version", content="Updated")

        # Try to delete with stale version
        with pytest.raises(SkillConflictError):
            await service.delete_skill("delete-version", expected_version=1)

    @pytest.mark.asyncio
    async def test_delete_skill_registry_conflict_keeps_skill_directory(self, service, monkeypatch):
        """A delete conflict must not remove the skill directory before the DB delete lands."""
        created = await service.create_skill("delete-rollback", "Content")
        skill_dir = service._get_skill_dir("delete-rollback")

        def _conflict_delete(*_args, **_kwargs):
            raise ConflictError("simulated stale delete", entity="skill_registry", entity_id="delete-rollback")

        monkeypatch.setattr(service._get_db(), "mark_skill_registry_deleted", _conflict_delete)

        with pytest.raises(SkillConflictError):
            await service.delete_skill("delete-rollback", expected_version=created["version"])

        assert skill_dir.exists()
        assert (skill_dir / "SKILL.md").read_text(encoding="utf-8") == "Content"

    @pytest.mark.asyncio
    async def test_delete_skill_directory_failure_restores_registry(self, service, monkeypatch):
        """A directory deletion failure must not leave the skill hidden in the registry."""
        created = await service.create_skill("delete-restore", "Content")
        skill_dir = service._get_skill_dir("delete-restore")

        def _fail_move(_source, _destination):
            raise OSError("simulated directory lock")

        monkeypatch.setattr(service, "_move_skill_dir", _fail_move, raising=False)

        with pytest.raises(SkillStorageError, match="simulated directory lock"):
            await service.delete_skill("delete-restore", expected_version=created["version"])

        row = service._get_db().get_skill_registry("delete-restore", include_deleted=False)
        assert row is not None
        assert skill_dir.exists()
        assert (skill_dir / "SKILL.md").read_text(encoding="utf-8") == "Content"

    @pytest.mark.asyncio
    async def test_delete_archives_complete_bundle_and_restore_recovers_it(self, service):
        """Delete and restore move the complete bundle without losing supporting files."""
        created = await service.create_skill(
            "trash-roundtrip",
            "---\ndescription: Round trip\n---\nBody",
            supporting_files={"notes.md": "keep me"},
        )

        await service.delete_skill("trash-roundtrip", expected_version=created["version"])

        assert not service._get_skill_dir("trash-roundtrip").exists()
        deleted_row = service._get_db().get_skill_registry(
            "trash-roundtrip",
            include_deleted=True,
        )
        assert deleted_row is not None
        archive_dir = Path(deleted_row["directory_path"])
        assert archive_dir.parent.resolve() == (service.skills_dir / ".trash").resolve()
        assert (archive_dir / "SKILL.md").is_file()
        assert (archive_dir / "notes.md").read_text(encoding="utf-8") == "keep me"

        trash = await service.list_trash(limit=10, offset=0)
        assert trash[0]["name"] == "trash-roundtrip"
        assert trash[0]["restorable"] is True

        restored = await service.restore_skill(
            "trash-roundtrip",
            expected_version=deleted_row["version"],
        )
        assert restored["supporting_files"] == {"notes.md": "keep me"}
        assert service._get_skill_dir("trash-roundtrip").is_dir()
        assert not archive_dir.exists()

    @pytest.mark.asyncio
    async def test_trash_reports_missing_archive_and_restore_conflicts(self, service):
        """A stale registry row remains visible but never promises an impossible restore."""
        created = await service.create_skill("missing-archive", "Body")
        await service.delete_skill("missing-archive", expected_version=created["version"])
        deleted_row = service._get_db().get_skill_registry("missing-archive", include_deleted=True)
        assert deleted_row is not None
        shutil.rmtree(Path(deleted_row["directory_path"]))

        trash = await service.list_trash(limit=10, offset=0)
        assert trash[0]["restorable"] is False
        assert trash[0]["restore_unavailable_reason"] == "Archived skill files are missing."

        with pytest.raises(SkillConflictError, match="archived files are missing"):
            await service.restore_skill(
                "missing-archive",
                expected_version=deleted_row["version"],
            )

    @pytest.mark.asyncio
    async def test_trash_reports_malformed_archive_as_not_restorable(self, service):
        """Trash must not advertise a bundle that the restore parser will reject."""
        created = await service.create_skill("malformed-archive", "Body")
        await service.delete_skill("malformed-archive", expected_version=created["version"])
        deleted_row = service._get_db().get_skill_registry("malformed-archive", include_deleted=True)
        assert deleted_row is not None
        archive_dir = Path(deleted_row["directory_path"])
        (archive_dir / "SKILL.md").write_text(
            "---\n1: invalid-key\n---\nBody",
            encoding="utf-8",
        )

        trash = await service.list_trash(limit=10, offset=0)

        assert trash[0]["restorable"] is False
        assert trash[0]["restore_unavailable_reason"] == "Archived skill files are invalid."

    @pytest.mark.asyncio
    async def test_trash_rejects_archive_with_mismatched_canonical_name(self, service):
        """Trash validation and restore must enforce the registry identity."""
        created = await service.create_skill(
            "canonical-archive",
            "---\nname: canonical-archive\ndescription: Canonical\n---\n\nBody",
        )
        await service.delete_skill(
            "canonical-archive",
            expected_version=created["version"],
        )
        deleted_row = service._get_db().get_skill_registry(
            "canonical-archive",
            include_deleted=True,
        )
        assert deleted_row is not None
        archive_dir = Path(deleted_row["directory_path"])
        (archive_dir / "SKILL.md").write_text(
            "---\nname: other-skill\ndescription: Other\n---\n\nBody",
            encoding="utf-8",
        )

        trash = await service.list_trash(limit=10, offset=0)

        assert trash[0]["restorable"] is False
        assert trash[0]["restore_unavailable_reason"] == "Archived skill files are invalid."
        with pytest.raises(SkillConflictError, match="archived files are invalid"):
            await service.restore_skill(
                "canonical-archive",
                expected_version=deleted_row["version"],
            )
        assert archive_dir.is_dir()
        assert not service._get_skill_dir("canonical-archive").exists()

    @pytest.mark.asyncio
    async def test_restore_rejects_archive_with_invalid_supporting_file(self, service):
        """Direct restore must reject every bundle reported as non-restorable."""
        created = await service.create_skill(
            "invalid-supporting-archive",
            "Body",
            supporting_files={"notes.md": "valid"},
        )
        await service.delete_skill(
            "invalid-supporting-archive",
            expected_version=created["version"],
        )
        deleted_row = service._get_db().get_skill_registry(
            "invalid-supporting-archive",
            include_deleted=True,
        )
        assert deleted_row is not None
        archive_dir = Path(deleted_row["directory_path"])
        (archive_dir / "notes.md").write_bytes(b"\xff")

        trash = await service.list_trash(limit=10, offset=0)
        assert trash[0]["restorable"] is False

        with pytest.raises(SkillConflictError, match="invalid"):
            await service.restore_skill(
                "invalid-supporting-archive",
                expected_version=deleted_row["version"],
            )

        assert archive_dir.is_dir()
        assert not service._get_skill_dir("invalid-supporting-archive").exists()
        current_row = service._get_db().get_skill_registry(
            "invalid-supporting-archive",
            include_deleted=True,
        )
        assert current_row is not None and current_row["deleted"] is True

    @pytest.mark.asyncio
    async def test_trash_bundle_validation_runs_off_event_loop(self, service, monkeypatch):
        """Listing Trash must not perform recursive bundle reads on the event loop."""
        created = await service.create_skill("offloaded-trash-validation", "Body")
        await service.delete_skill(
            "offloaded-trash-validation",
            expected_version=created["version"],
        )
        event_loop_thread = threading.get_ident()
        validation_threads: list[int] = []
        original_validate = service._is_skill_bundle_valid

        def _track_validation_thread(name: str, bundle_dir: Path) -> bool:
            validation_threads.append(threading.get_ident())
            return original_validate(name, bundle_dir)

        monkeypatch.setattr(service, "_is_skill_bundle_valid", _track_validation_thread)

        await service.list_trash(limit=10, offset=0)

        assert validation_threads
        assert all(thread_id != event_loop_thread for thread_id in validation_threads)

    @pytest.mark.asyncio
    async def test_restore_does_not_resynchronize_inside_trash_lock(self, service, monkeypatch):
        """A committed restore must not contend with its own non-reentrant lock."""
        created = await service.create_skill("restore-without-resync", "Body")
        await service.delete_skill(
            "restore-without-resync",
            expected_version=created["version"],
        )
        deleted_row = service._get_db().get_skill_registry(
            "restore-without-resync",
            include_deleted=True,
        )
        assert deleted_row is not None
        service._sync_interval = 0
        original_sync = service._sync_registry_async
        sync_calls: list[tuple[bool, bool]] = []

        async def _guarded_sync(
            force: bool = False,
            *,
            trash_lock_held: bool = False,
        ) -> None:
            sync_calls.append((force, trash_lock_held))
            if not trash_lock_held:
                raise AssertionError("restore attempted an unlocked nested registry sync")
            await original_sync(force=force, trash_lock_held=trash_lock_held)

        monkeypatch.setattr(service, "_sync_registry_async", _guarded_sync)

        restored = await service.restore_skill(
            "restore-without-resync",
            expected_version=deleted_row["version"],
        )

        assert restored["name"] == "restore-without-resync"
        assert sync_calls == [(True, True)]

    @pytest.mark.asyncio
    async def test_trash_listing_rejects_symlinked_trash_root(self, service):
        """Trash status checks must not follow a replaced Trash root outside skills storage."""
        created = await service.create_skill("symlink-trash", "Body")
        await service.delete_skill("symlink-trash", expected_version=created["version"])
        external_trash = service.skills_dir.parent / "external-trash"
        service.trash_dir.rename(external_trash)
        try:
            service.trash_dir.symlink_to(external_trash, target_is_directory=True)
        except OSError as error:
            pytest.skip(f"TASK-12969: directory symlinks are unavailable: {error}")

        with pytest.raises(SkillStorageError, match="regular directory"):
            await service.list_trash(limit=10, offset=0)

    @pytest.mark.asyncio
    async def test_purge_removes_archive_and_registry_row(self, service):
        """Permanent deletion removes both durable trash storage and registry metadata."""
        created = await service.create_skill("purge-me", "Body")
        await service.delete_skill("purge-me", expected_version=created["version"])
        deleted_row = service._get_db().get_skill_registry("purge-me", include_deleted=True)
        assert deleted_row is not None
        archive_dir = Path(deleted_row["directory_path"])

        await service.purge_skill("purge-me", expected_version=deleted_row["version"])

        assert not archive_dir.exists()
        assert service._get_db().get_skill_registry("purge-me", include_deleted=True) is None

    @pytest.mark.asyncio
    async def test_constructor_defers_startup_maintenance_to_worker_sync(
        self,
        service,
        monkeypatch,
    ):
        """Request-time construction must not scan Trash on the event-loop thread."""
        event_loop_thread = threading.get_ident()
        maintenance_calls: list[tuple[str, int]] = []

        def _track_reconcile(_service: SkillsService) -> None:
            maintenance_calls.append(("reconcile", threading.get_ident()))

        def _track_cleanup(_service: SkillsService) -> None:
            maintenance_calls.append(("cleanup", threading.get_ident()))

        monkeypatch.setattr(SkillsService, "_reconcile_orphaned_archives", _track_reconcile)
        monkeypatch.setattr(SkillsService, "_retry_staged_cleanup", _track_cleanup)

        restarted = SkillsService(
            user_id=service.user_id,
            base_path=service.base_path,
            db=service._get_db(),
            sync_interval=float("inf"),
            integrity_resolver=service.integrity_resolver,
        )

        assert maintenance_calls == []

        await restarted.list_skills()

        assert [name for name, _thread_id in maintenance_calls] == ["reconcile", "cleanup"]
        assert all(thread_id != event_loop_thread for _name, thread_id in maintenance_calls)

    def test_cleanup_retries_readonly_entries(self, service, monkeypatch):
        """Cleanup clears a read-only entry and retries removal on Windows."""
        cleanup_dir = service._ensure_cleanup_directory()
        cleanup_path = cleanup_dir / "readonly-cleanup"
        cleanup_path.mkdir()
        readonly_file = cleanup_path / "readonly.txt"
        readonly_file.write_text("content", encoding="utf-8")
        readonly_file.chmod(stat.S_IREAD)
        original_rmtree = shutil.rmtree

        def _simulate_windows_rmtree(path, *, onerror=None):
            assert onerror is not None

            def _retry_remove(target) -> None:
                target_path = Path(target)
                assert target_path.stat().st_mode & stat.S_IWUSR
                target_path.unlink()

            error = PermissionError("read-only file")
            onerror(_retry_remove, str(readonly_file), (PermissionError, error, None))
            original_rmtree(path)

        # Keep the global shutil patch inside the assertion. Python 3.13 uses
        # the newer ``onexc`` callback while tearing down TemporaryDirectory;
        # leaking this older-signature fake into fixture teardown breaks an
        # otherwise successful test.
        with monkeypatch.context() as scoped_patch:
            scoped_patch.setattr(shutil, "rmtree", _simulate_windows_rmtree)
            assert service._remove_cleanup_path_best_effort(cleanup_path) is True
        assert not cleanup_path.exists()

    def test_cleanup_rejects_symlink_to_sibling_queue_entry(self, service):
        """Cleanup must not follow a queued symlink to another cleanup directory."""
        cleanup_dir = service._ensure_cleanup_directory()
        target = cleanup_dir / "target-cleanup"
        target.mkdir()
        (target / "preserve.txt").write_text("content", encoding="utf-8")
        symlink = cleanup_dir / "linked-cleanup"
        symlink.symlink_to(target, target_is_directory=True)

        assert service._remove_cleanup_path_best_effort(symlink) is False
        assert symlink.is_symlink()
        assert (target / "preserve.txt").read_text(encoding="utf-8") == "content"

    @pytest.mark.asyncio
    async def test_startup_restores_archive_after_interrupted_delete(self, service):
        """Reconciliation restores the only bundle when delete crashed before its DB commit."""
        created = await service.create_skill("interrupted-delete", "Original body")
        row = service._get_db().get_skill_registry("interrupted-delete", include_deleted=True)
        assert row is not None
        skill_dir = service._get_skill_dir("interrupted-delete")
        archive_dir = service._get_archive_dir(row)
        service._move_skill_dir(skill_dir, archive_dir)

        await _restart_and_sync(service)

        assert skill_dir.is_dir()
        assert (skill_dir / "SKILL.md").read_text(encoding="utf-8") == "Original body"
        assert not archive_dir.exists()
        active_row = service._get_db().get_skill_registry("interrupted-delete", include_deleted=True)
        assert active_row is not None
        assert active_row["deleted"] is False
        assert active_row["version"] == created["version"]

    @pytest.mark.asyncio
    async def test_startup_discards_incomplete_prepublication_staging(self, service):
        """A crash while preparing a bundle cannot expose partial files as a skill."""
        staging_dir = service.skills_dir / ".staging-create-interrupted"
        staging_dir.mkdir()
        (staging_dir / "SKILL.md").write_text("Partial replacement", encoding="utf-8")

        await _restart_and_sync(service)

        assert not staging_dir.exists()
        assert service._get_db().get_skill_registry(
            ".staging-create-interrupted",
            include_deleted=True,
        ) is None

    @pytest.mark.asyncio
    async def test_startup_restores_active_bundle_when_replacement_publish_was_interrupted(
        self,
        service,
    ):
        """A crash after moving the active bundle aside rolls back to the original."""
        created = await service.create_skill("replace-crash", "Original body")
        row = service._get_db().get_skill_registry("replace-crash", include_deleted=True)
        assert row is not None
        active_dir = service._get_skill_dir("replace-crash")
        replacement_backup = service._replacement_backup_path(row)
        service._move_skill_dir(active_dir, replacement_backup)
        staging_dir = service.skills_dir / ".staging-replace-crash"
        staging_dir.mkdir()
        (staging_dir / "SKILL.md").write_text("Partial new body", encoding="utf-8")

        await _restart_and_sync(service)

        assert (active_dir / "SKILL.md").read_text(encoding="utf-8") == "Original body"
        assert not replacement_backup.exists()
        assert not staging_dir.exists()
        active_row = service._get_db().get_skill_registry("replace-crash", include_deleted=True)
        assert active_row is not None
        assert active_row["deleted"] is False
        assert active_row["version"] == created["version"]

    @pytest.mark.asyncio
    async def test_startup_rolls_back_replacement_published_before_registry_commit(
        self,
        service,
    ):
        """Registry metadata decides whether an interrupted atomic swap committed."""
        created = await service.create_skill("replace-precommit", "Original body")
        row = service._get_db().get_skill_registry("replace-precommit", include_deleted=True)
        assert row is not None
        active_dir = service._get_skill_dir("replace-precommit")
        replacement_backup = service._replacement_backup_path(row)
        service._move_skill_dir(active_dir, replacement_backup)
        active_dir.mkdir()
        (active_dir / "SKILL.md").write_text(
            "---\nname: replace-precommit\n---\nReplacement body",
            encoding="utf-8",
        )

        await _restart_and_sync(service)

        assert (active_dir / "SKILL.md").read_text(encoding="utf-8") == "Original body"
        assert not replacement_backup.exists()
        active_row = service._get_db().get_skill_registry(
            "replace-precommit",
            include_deleted=True,
        )
        assert active_row is not None
        assert active_row["version"] == created["version"]

    @pytest.mark.asyncio
    async def test_startup_rolls_back_supporting_only_replacement_before_registry_commit(
        self,
        service,
    ):
        """Recovery uses the registry version when SKILL.md itself is unchanged."""
        content = "---\nname: replace-supporting\n---\nStable body"
        created = await service.create_skill(
            "replace-supporting",
            content,
            supporting_files={"guide.md": "Original guide"},
        )
        row = service._get_db().get_skill_registry(
            "replace-supporting",
            include_deleted=True,
        )
        assert row is not None
        active_dir = service._get_skill_dir("replace-supporting")
        replacement_backup = service._replacement_backup_path(row)
        service._move_skill_dir(active_dir, replacement_backup)
        active_dir.mkdir()
        (active_dir / "SKILL.md").write_text(content, encoding="utf-8")
        (active_dir / "guide.md").write_text("Replacement guide", encoding="utf-8")

        await _restart_and_sync(service)

        assert (active_dir / "guide.md").read_text(encoding="utf-8") == "Original guide"
        assert not replacement_backup.exists()
        active_row = service._get_db().get_skill_registry(
            "replace-supporting",
            include_deleted=True,
        )
        assert active_row is not None
        assert active_row["version"] == created["version"]

    @pytest.mark.asyncio
    async def test_startup_discards_backup_after_replacement_registry_commit(
        self,
        service,
    ):
        """A committed replacement survives a crash before old-bundle cleanup."""
        created = await service.create_skill("replace-committed", "Original body")
        db = service._get_db()
        row = db.get_skill_registry("replace-committed", include_deleted=True)
        assert row is not None
        active_dir = service._get_skill_dir("replace-committed")
        replacement_backup = service._replacement_backup_path(row)
        service._move_skill_dir(active_dir, replacement_backup)
        active_dir.mkdir()
        (active_dir / "SKILL.md").write_text(
            "---\nname: replace-committed\n---\nReplacement body",
            encoding="utf-8",
        )
        parsed = service._parse_unchecked_skill_directory("replace-committed", active_dir)
        db.update_skill_registry(
            "replace-committed",
            service._registry_payload("replace-committed", active_dir, parsed),
            expected_version=created["version"],
        )

        await _restart_and_sync(service)

        assert (active_dir / "SKILL.md").read_text(encoding="utf-8").endswith(
            "Replacement body"
        )
        assert not replacement_backup.exists()
        active_row = db.get_skill_registry("replace-committed", include_deleted=True)
        assert active_row is not None
        assert active_row["version"] == created["version"] + 1

    @pytest.mark.asyncio
    async def test_startup_keeps_supporting_only_replacement_after_registry_commit(
        self,
        service,
    ):
        """A committed supporting-file edit is distinguished by its registry version."""
        content = "---\nname: replace-supporting-committed\n---\nStable body"
        created = await service.create_skill(
            "replace-supporting-committed",
            content,
            supporting_files={"guide.md": "Original guide"},
        )
        db = service._get_db()
        row = db.get_skill_registry(
            "replace-supporting-committed",
            include_deleted=True,
        )
        assert row is not None
        active_dir = service._get_skill_dir("replace-supporting-committed")
        replacement_backup = service._replacement_backup_path(row)
        service._move_skill_dir(active_dir, replacement_backup)
        active_dir.mkdir()
        (active_dir / "SKILL.md").write_text(content, encoding="utf-8")
        (active_dir / "guide.md").write_text("Replacement guide", encoding="utf-8")
        parsed = service._parse_unchecked_skill_directory(
            "replace-supporting-committed",
            active_dir,
        )
        db.update_skill_registry(
            "replace-supporting-committed",
            service._registry_payload(
                "replace-supporting-committed",
                active_dir,
                parsed,
            ),
            expected_version=created["version"],
        )

        await _restart_and_sync(service)

        assert (active_dir / "guide.md").read_text(encoding="utf-8") == (
            "Replacement guide"
        )
        assert not replacement_backup.exists()
        active_row = db.get_skill_registry(
            "replace-supporting-committed",
            include_deleted=True,
        )
        assert active_row is not None
        assert active_row["version"] == created["version"] + 1

    @pytest.mark.asyncio
    async def test_startup_preserves_bundles_and_retries_failed_replacement_rollback(
        self,
        service,
        monkeypatch,
    ):
        """A failed backup restore cannot fall through to cleanup or registry sync."""
        created = await service.create_skill("replace-restore-failure", "Original body")
        row = service._get_db().get_skill_registry(
            "replace-restore-failure",
            include_deleted=True,
        )
        assert row is not None
        active_dir = service._get_skill_dir("replace-restore-failure")
        replacement_backup = service._replacement_backup_path(row)
        service._move_skill_dir(active_dir, replacement_backup)
        active_dir.mkdir()
        (active_dir / "SKILL.md").write_text(
            "---\nname: replace-restore-failure\n---\nReplacement body",
            encoding="utf-8",
        )
        restarted = SkillsService(
            user_id=service.user_id,
            base_path=service.base_path,
            db=service._get_db(),
            integrity_resolver=service.integrity_resolver,
        )
        original_move = restarted._move_skill_dir

        def _fail_backup_restore(source: Path, destination: Path) -> None:
            if source == replacement_backup and destination == active_dir:
                raise OSError("simulated backup restore failure")
            original_move(source, destination)

        monkeypatch.setattr(restarted, "_move_skill_dir", _fail_backup_restore)

        with pytest.raises(SkillStorageError, match="replacement recovery is incomplete"):
            await restarted.list_skills()

        assert (active_dir / "SKILL.md").read_text(encoding="utf-8").endswith(
            "Replacement body"
        )
        assert (replacement_backup / "SKILL.md").read_text(
            encoding="utf-8"
        ) == "Original body"
        unchanged_row = service._get_db().get_skill_registry(
            "replace-restore-failure",
            include_deleted=True,
        )
        assert unchanged_row is not None
        assert unchanged_row["deleted"] is False
        assert unchanged_row["version"] == created["version"]

        monkeypatch.setattr(restarted, "_move_skill_dir", original_move)
        await restarted.list_skills()

        assert (active_dir / "SKILL.md").read_text(encoding="utf-8") == "Original body"
        assert not replacement_backup.exists()

    @pytest.mark.asyncio
    async def test_startup_blocks_on_ambiguous_legacy_replacement(self, service):
        """A legacy swap with indistinguishable SKILL.md hashes stays untouched."""
        content = "---\nname: replace-legacy-ambiguous\n---\nStable body"
        created = await service.create_skill(
            "replace-legacy-ambiguous",
            content,
            supporting_files={"guide.md": "Original guide"},
        )
        row = service._get_db().get_skill_registry(
            "replace-legacy-ambiguous",
            include_deleted=True,
        )
        assert row is not None
        active_dir = service._get_skill_dir("replace-legacy-ambiguous")
        replacement_backup = service.skills_dir / f".replacing-{row['uuid']}"
        service._move_skill_dir(active_dir, replacement_backup)
        active_dir.mkdir()
        (active_dir / "SKILL.md").write_text(content, encoding="utf-8")
        (active_dir / "guide.md").write_text("Replacement guide", encoding="utf-8")

        restarted = SkillsService(
            user_id=service.user_id,
            base_path=service.base_path,
            db=service._get_db(),
            integrity_resolver=service.integrity_resolver,
        )
        with pytest.raises(SkillStorageError, match="replacement recovery is incomplete"):
            await restarted.list_skills()

        assert (active_dir / "guide.md").read_text(encoding="utf-8") == (
            "Replacement guide"
        )
        assert (replacement_backup / "guide.md").read_text(encoding="utf-8") == (
            "Original guide"
        )
        unchanged_row = service._get_db().get_skill_registry(
            "replace-legacy-ambiguous",
            include_deleted=True,
        )
        assert unchanged_row is not None
        assert unchanged_row["version"] == created["version"]

    @pytest.mark.asyncio
    async def test_startup_blocks_on_malformed_replacement_metadata(self, service):
        """Malformed recovery metadata is preserved instead of ignored destructively."""
        malformed_backup = service.skills_dir / ".replacing-invalid.version"
        malformed_backup.mkdir()
        (malformed_backup / "SKILL.md").write_text("Preserve me", encoding="utf-8")
        restarted = SkillsService(
            user_id=service.user_id,
            base_path=service.base_path,
            db=service._get_db(),
            integrity_resolver=service.integrity_resolver,
        )

        with pytest.raises(SkillStorageError, match="replacement recovery is incomplete"):
            await restarted.list_skills()

        assert (malformed_backup / "SKILL.md").read_text(encoding="utf-8") == (
            "Preserve me"
        )

    @pytest.mark.asyncio
    async def test_initialized_service_reconciles_replacement_before_debounce(
        self,
        service,
    ):
        """A long-lived service detects replacement markers created after startup."""
        created = await service.create_skill("replace-after-startup", "Original body")
        await service.list_skills()
        assert service._startup_maintenance_complete is True
        row = service._get_db().get_skill_registry(
            "replace-after-startup",
            include_deleted=True,
        )
        assert row is not None
        active_dir = service._get_skill_dir("replace-after-startup")
        replacement_backup = service._replacement_backup_path(row)
        service._move_skill_dir(active_dir, replacement_backup)
        active_dir.mkdir()
        (active_dir / "SKILL.md").write_text(
            "---\nname: replace-after-startup\n---\nReplacement body",
            encoding="utf-8",
        )

        await service.list_skills()

        assert (active_dir / "SKILL.md").read_text(encoding="utf-8") == "Original body"
        assert not replacement_backup.exists()
        unchanged_row = service._get_db().get_skill_registry(
            "replace-after-startup",
            include_deleted=True,
        )
        assert unchanged_row is not None
        assert unchanged_row["version"] == created["version"]

    @pytest.mark.asyncio
    async def test_startup_restores_all_archives_after_interrupted_bulk_delete(self, service):
        """Reconciliation restores every bundle moved before an interrupted bulk DB commit."""
        await service.create_skill("interrupted-bulk-a", "Body A")
        await service.create_skill("interrupted-bulk-b", "Body B")
        archive_dirs: list[Path] = []
        for name in ("interrupted-bulk-a", "interrupted-bulk-b"):
            row = service._get_db().get_skill_registry(name, include_deleted=True)
            assert row is not None
            archive_dir = service._get_archive_dir(row)
            service._move_skill_dir(service._get_skill_dir(name), archive_dir)
            archive_dirs.append(archive_dir)

        await _restart_and_sync(service)

        assert (service._get_skill_dir("interrupted-bulk-a") / "SKILL.md").read_text(
            encoding="utf-8"
        ) == "Body A"
        assert (service._get_skill_dir("interrupted-bulk-b") / "SKILL.md").read_text(
            encoding="utf-8"
        ) == "Body B"
        assert all(not archive_dir.exists() for archive_dir in archive_dirs)

    @pytest.mark.asyncio
    async def test_startup_preserves_archive_when_active_bundle_is_invalid(self, service):
        """Ambiguous active state must not delete the only known valid archive."""
        await service.create_skill("ambiguous-replacement", "Active body")
        row = service._get_db().get_skill_registry("ambiguous-replacement", include_deleted=True)
        assert row is not None
        active_dir = service._get_skill_dir("ambiguous-replacement")
        archive_dir = service._get_archive_dir(row)
        archive_dir.mkdir(parents=True)
        (archive_dir / "SKILL.md").write_text("Archived body", encoding="utf-8")
        (active_dir / "SKILL.md").unlink()

        await _restart_and_sync(service)

        assert active_dir.is_dir()
        assert not (active_dir / "SKILL.md").exists()
        assert (archive_dir / "SKILL.md").read_text(encoding="utf-8") == "Archived body"

    @pytest.mark.asyncio
    async def test_delete_preserves_archive_when_active_bundle_is_invalid(self, service):
        """Delete must fail closed instead of replacing a valid archive with invalid files."""
        created = await service.create_skill("ambiguous-delete", "Active body")
        row = service._get_db().get_skill_registry("ambiguous-delete", include_deleted=True)
        assert row is not None
        active_dir = service._get_skill_dir("ambiguous-delete")
        archive_dir = service._get_archive_dir(row)
        archive_dir.mkdir(parents=True)
        (archive_dir / "SKILL.md").write_text("Archived body", encoding="utf-8")
        (active_dir / "SKILL.md").unlink()

        with pytest.raises(SkillStorageError, match="ambiguous"):
            await service.delete_skill("ambiguous-delete", expected_version=created["version"])

        assert active_dir.is_dir()
        assert not (active_dir / "SKILL.md").exists()
        assert (archive_dir / "SKILL.md").read_text(encoding="utf-8") == "Archived body"
        active_row = service._get_db().get_skill_registry("ambiguous-delete", include_deleted=True)
        assert active_row is not None
        assert active_row["deleted"] is False

    @pytest.mark.asyncio
    async def test_startup_preserves_archive_when_active_bundle_cannot_be_parsed(self, service):
        """A regular but unreadable active SKILL.md must not displace its valid archive."""
        await service.create_skill("unparsable-replacement", "Active body")
        row = service._get_db().get_skill_registry("unparsable-replacement", include_deleted=True)
        assert row is not None
        active_dir = service._get_skill_dir("unparsable-replacement")
        archive_dir = service._get_archive_dir(row)
        archive_dir.mkdir(parents=True)
        (archive_dir / "SKILL.md").write_text("Archived body", encoding="utf-8")
        (active_dir / "SKILL.md").write_bytes(b"\xff")

        await _restart_and_sync(service)

        assert (active_dir / "SKILL.md").read_bytes() == b"\xff"
        assert (archive_dir / "SKILL.md").read_text(encoding="utf-8") == "Archived body"

    @pytest.mark.asyncio
    async def test_startup_preserves_archive_when_active_frontmatter_is_malformed(self, service):
        """Unexpected parser input must fail closed without blocking service startup."""
        await service.create_skill("malformed-replacement", "Active body")
        row = service._get_db().get_skill_registry("malformed-replacement", include_deleted=True)
        assert row is not None
        active_dir = service._get_skill_dir("malformed-replacement")
        archive_dir = service._get_archive_dir(row)
        archive_dir.mkdir(parents=True)
        (archive_dir / "SKILL.md").write_text("Archived body", encoding="utf-8")
        (active_dir / "SKILL.md").write_text("---\n1: invalid-key\n---\nBody", encoding="utf-8")

        await _restart_and_sync(service)

        assert (active_dir / "SKILL.md").read_text(encoding="utf-8").startswith("---\n1:")
        assert (archive_dir / "SKILL.md").read_text(encoding="utf-8") == "Archived body"

    @pytest.mark.asyncio
    async def test_cancelled_trash_lock_waiter_releases_late_acquisition(
        self,
        service,
        monkeypatch,
    ):
        """Cancellation cannot strand a file lock acquired by the worker thread later."""
        holder = FileLock(service.trash_lock_path, timeout=1)
        assert holder.acquire() is True
        acquire_started = threading.Event()
        acquire_finished = threading.Event()
        original_acquire = FileLock.acquire
        tracked_first_waiter = False

        def _tracked_acquire(lock: FileLock) -> bool:
            nonlocal tracked_first_waiter
            if lock.path == service.trash_lock_path and not tracked_first_waiter:
                tracked_first_waiter = True
                acquire_started.set()
                try:
                    return original_acquire(lock)
                finally:
                    acquire_finished.set()
            return original_acquire(lock)

        monkeypatch.setattr(FileLock, "acquire", _tracked_acquire)

        async def _wait_for_lock() -> None:
            async with service._trash_operation_lock():
                pass

        waiter = asyncio.create_task(_wait_for_lock())
        contender: FileLock | None = None
        try:
            assert await asyncio.to_thread(acquire_started.wait, 5)
            waiter.cancel()
            holder.release()
            assert await asyncio.to_thread(acquire_finished.wait, 5)
            with pytest.raises(asyncio.CancelledError):
                await waiter

            contender = FileLock(service.trash_lock_path, timeout=0.3)
            assert await asyncio.to_thread(contender.acquire) is True
        finally:
            holder.release()
            if not waiter.done():
                waiter.cancel()
                with suppress(asyncio.CancelledError):
                    await waiter
            if contender is not None:
                contender.release()

    @pytest.mark.asyncio
    async def test_registry_sync_waits_for_in_flight_delete(self, service):
        """Sync cannot mark a row deleted between the delete move and registry commit."""
        created = await service.create_skill("sync-delete-boundary", "Body")
        row = service._get_db().get_skill_registry("sync-delete-boundary", include_deleted=True)
        assert row is not None
        skill_dir = service._get_skill_dir("sync-delete-boundary")
        archive_dir = service._get_archive_dir(row)
        holder = FileLock(service.trash_lock_path, timeout=1)
        assert holder.acquire() is True
        service._move_skill_dir(skill_dir, archive_dir)
        sync_task = asyncio.create_task(service._sync_registry_async(force=True))

        try:
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(asyncio.shield(sync_task), timeout=0.05)
            service._move_skill_dir(archive_dir, skill_dir)
        finally:
            if archive_dir.exists() and not skill_dir.exists():
                service._move_skill_dir(archive_dir, skill_dir)
            holder.release()
            await sync_task

        active_row = service._get_db().get_skill_registry(
            "sync-delete-boundary",
            include_deleted=True,
        )
        assert active_row is not None
        assert active_row["deleted"] is False
        assert active_row["version"] == created["version"]

    @pytest.mark.asyncio
    async def test_registry_sync_waits_for_in_flight_restore(self, service):
        """Sync cannot reactivate a row between the restore move and registry commit."""
        created = await service.create_skill("sync-restore-boundary", "Body")
        await service.delete_skill("sync-restore-boundary", expected_version=created["version"])
        deleted_row = service._get_db().get_skill_registry(
            "sync-restore-boundary",
            include_deleted=True,
        )
        assert deleted_row is not None
        skill_dir = service._get_skill_dir("sync-restore-boundary")
        archive_dir = service._get_archive_dir(deleted_row)
        holder = FileLock(service.trash_lock_path, timeout=1)
        assert holder.acquire() is True
        service._move_skill_dir(archive_dir, skill_dir)
        sync_task = asyncio.create_task(service._sync_registry_async(force=True))

        try:
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(asyncio.shield(sync_task), timeout=0.05)
            service._move_skill_dir(skill_dir, archive_dir)
        finally:
            if skill_dir.exists() and not archive_dir.exists():
                service._move_skill_dir(skill_dir, archive_dir)
            holder.release()
            await sync_task

        current_row = service._get_db().get_skill_registry(
            "sync-restore-boundary",
            include_deleted=True,
        )
        assert current_row is not None
        assert current_row["deleted"] is True
        assert current_row["version"] == deleted_row["version"]

    @pytest.mark.asyncio
    async def test_cancelled_delete_holds_lock_until_mutation_finishes(
        self,
        service,
        monkeypatch,
    ):
        """Cancellation cannot expose a half-finished delete to another service."""
        created = await service.create_skill("cancelled-delete", "Body")
        row = service._get_db().get_skill_registry("cancelled-delete", include_deleted=True)
        assert row is not None
        skill_dir = service._get_skill_dir("cancelled-delete")
        archive_dir = service._get_archive_dir(row)
        move_started = threading.Event()
        allow_move = threading.Event()
        original_move = service._move_skill_dir

        def _blocked_move(source: Path, destination: Path) -> None:
            if source == skill_dir and destination == archive_dir:
                move_started.set()
                if not allow_move.wait(timeout=5):
                    raise RuntimeError("timed out waiting to continue delete move")
            original_move(source, destination)

        monkeypatch.setattr(service, "_move_skill_dir", _blocked_move)
        delete_task = asyncio.create_task(
            service.delete_skill("cancelled-delete", expected_version=created["version"])
        )
        contender = FileLock(service.trash_lock_path, timeout=0.3)
        contender_acquired = False
        try:
            assert await asyncio.to_thread(move_started.wait, 5)
            delete_task.cancel()
            contender_acquired = await asyncio.to_thread(contender.acquire)
            assert contender_acquired is False
        finally:
            if contender_acquired:
                contender.release()
            allow_move.set()
            with pytest.raises(asyncio.CancelledError):
                await delete_task

        deleted_row = service._get_db().get_skill_registry(
            "cancelled-delete",
            include_deleted=True,
        )
        assert deleted_row is not None
        assert deleted_row["deleted"] is True
        assert not skill_dir.exists()
        assert archive_dir.is_dir()
        released_lock = FileLock(service.trash_lock_path, timeout=0.3)
        assert await asyncio.to_thread(released_lock.acquire) is True
        released_lock.release()

    def test_startup_reconciliation_does_not_interfere_with_in_flight_purge(
        self,
        service,
        monkeypatch,
    ):
        """A second service must not move purge staging before the registry commit."""
        created = asyncio.run(service.create_skill("concurrent-purge", "Body"))
        asyncio.run(
            service.delete_skill(
                "concurrent-purge",
                expected_version=created["version"],
            )
        )
        db = service._get_db()
        deleted_row = db.get_skill_registry("concurrent-purge", include_deleted=True)
        assert deleted_row is not None
        archive_dir = Path(deleted_row["directory_path"])
        staging_dir = service.trash_dir / f".purging-{deleted_row['uuid']}"
        purge_reached_registry = threading.Event()
        allow_registry_purge = threading.Event()
        original_purge = db.purge_skill_registry

        def _pause_registry_purge(*args, **kwargs):
            purge_reached_registry.set()
            if not allow_registry_purge.wait(timeout=5):
                raise RuntimeError("timed out waiting to continue registry purge")
            return original_purge(*args, **kwargs)

        def _run_purge() -> None:
            asyncio.run(
                service.purge_skill(
                    "concurrent-purge",
                    expected_version=deleted_row["version"],
                )
            )

        monkeypatch.setattr(db, "purge_skill_registry", _pause_registry_purge)
        contender_db = None
        contender_future = None
        with ThreadPoolExecutor(max_workers=2) as executor:
            purge_future = executor.submit(_run_purge)
            try:
                assert purge_reached_registry.wait(timeout=5)
                assert staging_dir.is_dir()

                contender_db = CharactersRAGDB(
                    db_path=db.db_path,
                    client_id="concurrent_purge_contender",
                )
                contender = SkillsService(
                    user_id=service.user_id,
                    base_path=service.base_path,
                    db=contender_db,
                    integrity_resolver=service.integrity_resolver,
                )
                contender_future = executor.submit(
                    lambda: asyncio.run(contender.list_skills())
                )
                with pytest.raises(FutureTimeoutError):
                    contender_future.result(timeout=0.2)
            finally:
                allow_registry_purge.set()
            try:
                purge_future.result(timeout=5)
                assert contender_future is not None
                contender_future.result(timeout=5)
            finally:
                if contender_db is not None:
                    contender_db.close_connection()

        assert db.get_skill_registry("concurrent-purge", include_deleted=True) is None
        assert not staging_dir.exists()
        assert not archive_dir.exists()

    def test_replacement_install_holds_lock_against_second_service_sync(
        self,
        service,
        monkeypatch,
    ):
        """A replacement bundle cannot become visible before registry activation commits."""
        created = asyncio.run(service.create_skill("replacement-race", "Old body"))
        asyncio.run(
            service.delete_skill(
                "replacement-race",
                expected_version=created["version"],
            )
        )
        db = service._get_db()
        target_file = service._get_skill_dir("replacement-race") / "SKILL.md"
        write_finished = threading.Event()
        allow_install = threading.Event()
        original_write_text = Path.write_text

        def _pause_replacement_write(path: Path, data: str, *args, **kwargs):
            result = original_write_text(path, data, *args, **kwargs)
            if (
                path.name == "SKILL.md"
                and path.parent.name.startswith(
                    ".staging-create-replacement-race-"
                )
                and data.rstrip().endswith("New body")
            ):
                write_finished.set()
                if not allow_install.wait(timeout=5):
                    raise RuntimeError("timed out waiting to continue replacement install")
            return result

        monkeypatch.setattr(Path, "write_text", _pause_replacement_write)
        contender_db = CharactersRAGDB(
            db_path=db.db_path,
            client_id="replacement_race_contender",
        )
        contender = SkillsService(
            user_id=service.user_id,
            base_path=service.base_path,
            db=contender_db,
            integrity_resolver=service.integrity_resolver,
        )

        def _run_replacement():
            return asyncio.run(
                service.import_skill(
                    content="---\ndescription: Replacement\n---\nNew body",
                    name="replacement-race",
                    overwrite=True,
                )
            )

        try:
            with ThreadPoolExecutor(max_workers=2) as executor:
                replacement_future = executor.submit(_run_replacement)
                assert write_finished.wait(timeout=5)
                contender_future = executor.submit(
                    lambda: asyncio.run(contender.list_skills())
                )
                contender_was_blocked = False
                try:
                    contender_future.result(timeout=0.2)
                except FutureTimeoutError:
                    contender_was_blocked = True
                finally:
                    allow_install.set()

                replacement = replacement_future.result(timeout=5)
                contender_future.result(timeout=5)
        finally:
            allow_install.set()
            contender_db.close_connection()

        assert contender_was_blocked is True
        assert replacement["description"] == "Replacement"
        active_row = db.get_skill_registry("replacement-race", include_deleted=True)
        assert active_row is not None and active_row["deleted"] is False
        assert target_file.read_text(encoding="utf-8").endswith("New body")

    def test_failed_update_cannot_restore_files_after_concurrent_delete(
        self,
        service,
        monkeypatch,
    ):
        """Update rollback must finish before another service can delete the skill."""
        created = asyncio.run(service.create_skill("update-delete-race", "Original"))
        db = service._get_db()
        update_reached_registry = threading.Event()
        allow_update_failure = threading.Event()

        def _fail_registry_update(name, _data, expected_version=None):
            if name == "update-delete-race":
                update_reached_registry.set()
                if not allow_update_failure.wait(timeout=5):
                    raise RuntimeError("timed out waiting to fail update")
                raise CharactersRAGDBError("simulated update failure")
            raise AssertionError(f"unexpected registry update for {name}")

        monkeypatch.setattr(db, "update_skill_registry", _fail_registry_update)
        contender_db = CharactersRAGDB(
            db_path=db.db_path,
            client_id="update_delete_contender",
        )
        contender = SkillsService(
            user_id=service.user_id,
            base_path=service.base_path,
            db=contender_db,
            integrity_resolver=service.integrity_resolver,
        )
        delete_committed = threading.Event()
        original_mark_deleted = contender_db.mark_skill_registry_deleted

        def _track_delete_commit(*args, **kwargs):
            result = original_mark_deleted(*args, **kwargs)
            delete_committed.set()
            return result

        monkeypatch.setattr(contender_db, "mark_skill_registry_deleted", _track_delete_commit)

        def _run_update() -> None:
            asyncio.run(
                service.update_skill(
                    "update-delete-race",
                    content="Updated",
                    expected_version=created["version"],
                )
            )

        def _run_delete() -> None:
            asyncio.run(contender.delete_skill("update-delete-race"))

        try:
            with ThreadPoolExecutor(max_workers=2) as executor:
                update_future = executor.submit(_run_update)
                assert update_reached_registry.wait(timeout=5)
                delete_future = executor.submit(_run_delete)
                delete_committed.wait(timeout=1)
                allow_update_failure.set()

                with pytest.raises(SkillsError, match="Failed to update skill"):
                    update_future.result(timeout=5)
                delete_future.result(timeout=5)
        finally:
            allow_update_failure.set()
            contender_db.close_connection()

        row = db.get_skill_registry("update-delete-race", include_deleted=True)
        assert row is not None
        assert row["deleted"] is True
        assert not service._get_skill_dir("update-delete-race").exists()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("operation", ["restore", "purge"])
    async def test_trash_mutation_rejects_symlink_to_sibling_archive(
        self,
        service,
        operation,
    ):
        """Restore and purge must not follow one archive UUID to another bundle."""
        first = await service.create_skill("alias-first", "First")
        second = await service.create_skill("alias-second", "Second")
        await service.delete_skill("alias-first", expected_version=first["version"])
        await service.delete_skill("alias-second", expected_version=second["version"])
        first_row = service._get_db().get_skill_registry("alias-first", include_deleted=True)
        second_row = service._get_db().get_skill_registry("alias-second", include_deleted=True)
        assert first_row is not None
        assert second_row is not None
        first_archive = Path(first_row["directory_path"])
        second_archive = Path(second_row["directory_path"])
        shutil.rmtree(first_archive)
        try:
            first_archive.symlink_to(second_archive, target_is_directory=True)
        except OSError:
            pytest.skip("TASK-12969: symlink creation is unavailable on this platform")

        with pytest.raises(SkillStorageError, match="symlink"):
            if operation == "restore":
                await service.restore_skill("alias-first")
            else:
                await service.purge_skill("alias-first")

        assert first_archive.is_symlink()
        assert second_archive.is_dir()
        assert (second_archive / "SKILL.md").read_text(encoding="utf-8") == "Second"

    @pytest.mark.asyncio
    async def test_purge_uses_registry_uuid_instead_of_another_archive_path(self, service):
        """A corrupted directory path must not let one Trash row delete another bundle."""
        first = await service.create_skill("purge-first", "First")
        second = await service.create_skill("purge-second", "Second")
        await service.delete_skill("purge-first", expected_version=first["version"])
        await service.delete_skill("purge-second", expected_version=second["version"])
        first_row = service._get_db().get_skill_registry("purge-first", include_deleted=True)
        second_row = service._get_db().get_skill_registry("purge-second", include_deleted=True)
        assert first_row is not None
        assert second_row is not None
        first_archive = Path(first_row["directory_path"])
        second_archive = Path(second_row["directory_path"])

        service._get_db().execute_query(
            "UPDATE skill_registry SET directory_path = ? WHERE name = ?",
            (str(second_archive), "purge-first"),
            commit=True,
        )

        await service.purge_skill("purge-first", expected_version=first_row["version"])

        assert not first_archive.exists()
        assert second_archive.is_dir()
        assert (second_archive / "SKILL.md").read_text(encoding="utf-8") == "Second"

    @pytest.mark.asyncio
    async def test_create_requires_explicit_replacement_of_trashed_name(self, service):
        """Normal create preserves a recoverable trash item; explicit import overwrite replaces it."""
        created = await service.create_skill("reserved-name", "Old body")
        await service.delete_skill("reserved-name", expected_version=created["version"])

        with pytest.raises(SkillConflictError, match="exists in Trash"):
            await service.create_skill("reserved-name", "New body")

        replaced = await service.import_skill(
            content="---\ndescription: Replacement\n---\nNew body",
            name="reserved-name",
            overwrite=True,
        )
        assert replaced["description"] == "Replacement"
        assert await service.get_trash_count() == 0

    @pytest.mark.asyncio
    async def test_replacement_preserves_active_bundle_after_partial_archive_cleanup(
        self,
        service,
        monkeypatch,
    ):
        """Partial cleanup of the old archive must not roll back a valid replacement."""
        created = await service.create_skill("replacement-failure", "Old body")
        await service.delete_skill("replacement-failure", expected_version=created["version"])
        deleted_row = service._get_db().get_skill_registry(
            "replacement-failure",
            include_deleted=True,
        )
        assert deleted_row is not None
        original_rmtree = shutil.rmtree
        cleanup_failed = False

        def _partially_remove_then_fail(path, *args, **kwargs):
            nonlocal cleanup_failed
            target = Path(path)
            if not cleanup_failed and (
                target.name == str(deleted_row["uuid"])
                or target.parent.name == ".cleanup"
            ):
                cleanup_failed = True
                (target / "SKILL.md").unlink()
                raise OSError("simulated partial archive cleanup failure")
            return original_rmtree(path, *args, **kwargs)

        monkeypatch.setattr(shutil, "rmtree", _partially_remove_then_fail)

        replaced = await service.import_skill(
            content="---\ndescription: Replacement\n---\nNew body",
            name="replacement-failure",
            overwrite=True,
        )

        active_row = service._get_db().get_skill_registry(
            "replacement-failure",
            include_deleted=True,
        )
        assert active_row is not None
        assert active_row["deleted"] is False
        assert replaced["description"] == "Replacement"
        assert service._get_skill_dir("replacement-failure").is_dir()
        assert (service._get_skill_dir("replacement-failure") / "SKILL.md").read_text(
            encoding="utf-8"
        ).endswith("New body")

        monkeypatch.setattr(shutil, "rmtree", original_rmtree)
        await _restart_and_sync(service)
        cleanup_dir = service.trash_dir / ".cleanup"
        assert not cleanup_dir.exists() or not any(cleanup_dir.iterdir())

    @pytest.mark.asyncio
    async def test_purge_does_not_restore_partially_deleted_archive(
        self,
        service,
        monkeypatch,
    ):
        """A committed purge remains purged when recursive cleanup fails part-way."""
        created = await service.create_skill(
            "partial-purge",
            "Old body",
            supporting_files={"notes.md": "supporting content"},
        )
        await service.delete_skill("partial-purge", expected_version=created["version"])
        deleted_row = service._get_db().get_skill_registry(
            "partial-purge",
            include_deleted=True,
        )
        assert deleted_row is not None
        original_rmtree = shutil.rmtree
        cleanup_failed = False

        def _partially_remove_then_fail(path, *args, **kwargs):
            nonlocal cleanup_failed
            target = Path(path)
            if not cleanup_failed and (
                target.name.startswith(".purging-")
                or target.parent.name == ".cleanup"
            ):
                cleanup_failed = True
                (target / "SKILL.md").unlink()
                raise OSError("simulated partial purge cleanup failure")
            return original_rmtree(path, *args, **kwargs)

        monkeypatch.setattr(shutil, "rmtree", _partially_remove_then_fail)

        await service.purge_skill("partial-purge", expected_version=deleted_row["version"])

        assert service._get_db().get_skill_registry(
            "partial-purge",
            include_deleted=True,
        ) is None
        assert not Path(deleted_row["directory_path"]).exists()

        monkeypatch.setattr(shutil, "rmtree", original_rmtree)
        await _restart_and_sync(service)
        cleanup_dir = service.trash_dir / ".cleanup"
        assert not cleanup_dir.exists() or not any(cleanup_dir.iterdir())

    @pytest.mark.asyncio
    async def test_startup_reconciles_purge_when_cleanup_queue_staging_failed(
        self,
        service,
        monkeypatch,
    ):
        """A committed purge orphan is retried even when queue staging failed."""
        created = await service.create_skill("purge-stage-failure", "Old body")
        await service.delete_skill("purge-stage-failure", expected_version=created["version"])
        deleted_row = service._get_db().get_skill_registry(
            "purge-stage-failure",
            include_deleted=True,
        )
        assert deleted_row is not None
        original_stage = service._stage_for_cleanup

        def _fail_purge_queue_staging(source: Path, label: str) -> Path:
            if source.name.startswith(".purging-"):
                raise OSError("simulated cleanup queue staging failure")
            return original_stage(source, label)

        monkeypatch.setattr(service, "_stage_for_cleanup", _fail_purge_queue_staging)

        await service.purge_skill("purge-stage-failure", expected_version=deleted_row["version"])

        staging_dir = service.trash_dir / f".purging-{deleted_row['uuid']}"
        assert staging_dir.is_dir()
        assert service._get_db().get_skill_registry(
            "purge-stage-failure",
            include_deleted=True,
        ) is None

        monkeypatch.setattr(service, "_stage_for_cleanup", original_stage)
        await _restart_and_sync(service)

        assert not staging_dir.exists()

    @pytest.mark.asyncio
    async def test_startup_reconciles_replacement_archive_when_queue_staging_failed(
        self,
        service,
        monkeypatch,
    ):
        """An active replacement keeps running while its stale archive is retried."""
        created = await service.create_skill("replacement-stage-failure", "Old body")
        await service.delete_skill("replacement-stage-failure", expected_version=created["version"])
        deleted_row = service._get_db().get_skill_registry(
            "replacement-stage-failure",
            include_deleted=True,
        )
        assert deleted_row is not None
        archive_dir = Path(deleted_row["directory_path"])
        original_stage = service._stage_for_cleanup

        def _fail_replacement_queue_staging(source: Path, label: str) -> Path:
            if source == archive_dir:
                raise OSError("simulated replacement queue staging failure")
            return original_stage(source, label)

        monkeypatch.setattr(service, "_stage_for_cleanup", _fail_replacement_queue_staging)

        replaced = await service.import_skill(
            content="---\ndescription: Replacement\n---\nNew body",
            name="replacement-stage-failure",
            overwrite=True,
        )

        assert replaced["description"] == "Replacement"
        assert archive_dir.is_dir()
        assert service._get_db().get_skill_registry(
            "replacement-stage-failure",
            include_deleted=True,
        )["deleted"] is False

        monkeypatch.setattr(service, "_stage_for_cleanup", original_stage)
        await _restart_and_sync(service)

        assert not archive_dir.exists()
        assert (service._get_skill_dir("replacement-stage-failure") / "SKILL.md").read_text(
            encoding="utf-8"
        ).endswith("New body")

    @pytest.mark.asyncio
    async def test_startup_restores_trash_after_purge_registry_and_rollback_failures(
        self,
        service,
        monkeypatch,
    ):
        """A failed purge rollback is logged and reconciled into a visible Trash item."""
        created = await service.create_skill("purge-rollback-failure", "Old body")
        await service.delete_skill("purge-rollback-failure", expected_version=created["version"])
        deleted_row = service._get_db().get_skill_registry(
            "purge-rollback-failure",
            include_deleted=True,
        )
        assert deleted_row is not None
        archive_dir = Path(deleted_row["directory_path"])
        staging_dir = service.trash_dir / f".purging-{deleted_row['uuid']}"
        original_move = service._move_skill_dir
        original_purge = service._get_db().purge_skill_registry

        def _fail_registry_purge(*args, **kwargs):
            raise CharactersRAGDBError("simulated registry purge failure")

        def _fail_rollback_move(source: Path, destination: Path) -> None:
            if source == staging_dir and destination == archive_dir:
                raise OSError("simulated rollback move failure")
            original_move(source, destination)

        monkeypatch.setattr(service._get_db(), "purge_skill_registry", _fail_registry_purge)
        monkeypatch.setattr(service, "_move_skill_dir", _fail_rollback_move)

        with pytest.raises(SkillsError, match="Failed to purge"):
            await service.purge_skill("purge-rollback-failure", expected_version=deleted_row["version"])

        assert staging_dir.is_dir()
        assert not archive_dir.exists()

        monkeypatch.setattr(service._get_db(), "purge_skill_registry", original_purge)
        monkeypatch.setattr(service, "_move_skill_dir", original_move)
        await _restart_and_sync(service)

        assert archive_dir.is_dir()
        assert not staging_dir.exists()
        assert (archive_dir / "SKILL.md").read_text(encoding="utf-8") == "Old body"

    @pytest.mark.asyncio
    async def test_bulk_delete_move_failure_rolls_back_all_bundles(self, service, monkeypatch):
        """A filesystem failure cannot leave a partially deleted bulk selection."""
        first = await service.create_skill("bulk-move-a", "Content A")
        second = await service.create_skill("bulk-move-b", "Content B")
        original_move = getattr(service, "_move_skill_dir", None)
        move_count = 0

        def _fail_second_move(source: Path, destination: Path) -> None:
            nonlocal move_count
            move_count += 1
            if move_count == 2:
                raise OSError("simulated second move failure")
            if original_move is not None:
                original_move(source, destination)
            else:
                source.rename(destination)

        monkeypatch.setattr(service, "_move_skill_dir", _fail_second_move, raising=False)

        with pytest.raises(SkillStorageError, match="simulated second move failure"):
            await service.bulk_delete_skills(
                [
                    {"name": "bulk-move-a", "version": first["version"]},
                    {"name": "bulk-move-b", "version": second["version"]},
                ]
            )

        assert service._get_skill_dir("bulk-move-a").is_dir()
        assert service._get_skill_dir("bulk-move-b").is_dir()
        assert service._get_db().get_skill_registry("bulk-move-a", include_deleted=False) is not None
        assert service._get_db().get_skill_registry("bulk-move-b", include_deleted=False) is not None
        assert await service.get_trash_count() == 0

    @pytest.mark.asyncio
    async def test_bulk_delete_validates_all_versions_before_discarding_stale_archive(self, service):
        """A later version conflict must leave every pre-existing archive untouched."""
        first = await service.create_skill("bulk-preflight-a", "Current A")
        second = await service.create_skill("bulk-preflight-b", "Current B")
        first_row = service._get_db().get_skill_registry("bulk-preflight-a", include_deleted=True)
        assert first_row is not None
        stale_archive = service._get_archive_dir(first_row)
        stale_archive.mkdir(parents=True)
        (stale_archive / "SKILL.md").write_text("Archived A", encoding="utf-8")
        await service.update_skill(
            "bulk-preflight-b",
            content="Updated B",
            expected_version=second["version"],
        )

        with pytest.raises(SkillConflictError):
            await service.bulk_delete_skills(
                [
                    {"name": "bulk-preflight-a", "version": first["version"]},
                    {"name": "bulk-preflight-b", "version": second["version"]},
                ]
            )

        assert (stale_archive / "SKILL.md").read_text(encoding="utf-8") == "Archived A"
        assert service._get_skill_dir("bulk-preflight-a").is_dir()

    @pytest.mark.asyncio
    async def test_bulk_delete_preserves_archive_when_active_bundle_is_invalid(self, service):
        """Bulk delete must fail closed before replacing a valid recovery archive."""
        first = await service.create_skill("bulk-ambiguous-a", "Current A")
        second = await service.create_skill("bulk-ambiguous-b", "Current B")
        first_row = service._get_db().get_skill_registry("bulk-ambiguous-a", include_deleted=True)
        assert first_row is not None
        first_dir = service._get_skill_dir("bulk-ambiguous-a")
        archive_dir = service._get_archive_dir(first_row)
        archive_dir.mkdir(parents=True)
        (archive_dir / "SKILL.md").write_text("Archived A", encoding="utf-8")
        (first_dir / "SKILL.md").unlink()

        with pytest.raises(SkillStorageError, match="ambiguous"):
            await service.bulk_delete_skills(
                [
                    {"name": "bulk-ambiguous-a", "version": first["version"]},
                    {"name": "bulk-ambiguous-b", "version": second["version"]},
                ]
            )

        assert first_dir.is_dir()
        assert not (first_dir / "SKILL.md").exists()
        assert (archive_dir / "SKILL.md").read_text(encoding="utf-8") == "Archived A"
        first_current = service._get_db().get_skill_registry(
            "bulk-ambiguous-a",
            include_deleted=True,
        )
        second_current = service._get_db().get_skill_registry(
            "bulk-ambiguous-b",
            include_deleted=True,
        )
        assert first_current is not None and first_current["deleted"] is False
        assert second_current is not None and second_current["deleted"] is False

    @pytest.mark.asyncio
    async def test_bulk_delete_syncs_registry_once(self, service, monkeypatch):
        """Bulk delete should reuse one forced registry sync for the whole request."""
        first = await service.create_skill("bulk-sync-a", "Content A")
        second = await service.create_skill("bulk-sync-b", "Content B")
        sync_calls = 0
        original_sync = service._sync_registry_async

        async def _counted_sync(*args, **kwargs):
            nonlocal sync_calls
            sync_calls += 1
            return await original_sync(*args, **kwargs)

        monkeypatch.setattr(service, "_sync_registry_async", _counted_sync)

        await service.bulk_delete_skills(
            [
                {"name": "bulk-sync-a", "version": first["version"]},
                {"name": "bulk-sync-b", "version": second["version"]},
            ]
        )

        assert sync_calls == 1

    @pytest.mark.asyncio
    async def test_update_skill_supporting_files_only_bumps_version(self, service):
        """Supporting-file-only changes are skill bundle mutations and must be versioned."""
        created = await service.create_skill(
            "support-version",
            "Content",
            supporting_files={"guide.md": "v1"},
        )

        updated = await service.update_skill(
            "support-version",
            supporting_files={"guide.md": "v2"},
            expected_version=created["version"],
        )

        assert updated["version"] == created["version"] + 1
        assert updated["supporting_files"]["guide.md"] == "v2"

    @pytest.mark.asyncio
    async def test_update_skill_supporting_file_write_failure_raises_storage_error(self, service, monkeypatch):
        """Supporting-file write failures must not be reported as successful updates."""
        await service.create_skill("support-failure", "Content")
        bad_path = service.skills_dir / "missing-parent" / "new.md"

        monkeypatch.setattr(service, "_safe_supporting_path", lambda *_args, **_kwargs: bad_path)

        with pytest.raises(SkillStorageError):
            await service.update_skill(
                "support-failure",
                supporting_files={"new.md": "new content"},
            )

    @pytest.mark.asyncio
    async def test_import_skill(self, service):
        """Test importing a skill from content."""
        content = """---
name: imported
description: Imported skill
---

Imported content.
"""
        result = await service.import_skill(content=content)

        assert result["name"] == "imported"
        assert result["description"] == "Imported skill"

    @pytest.mark.asyncio
    async def test_import_skill_with_name_override(self, service):
        """Test importing with name override."""
        content = """---
name: original-name
custom-review-key: preserve-me
---

Content.
"""
        result = await service.import_skill(content=content, name="override-name")

        assert result["name"] == "override-name"
        assert "name: override-name" in result["raw_content"]
        assert "custom-review-key: preserve-me" in result["raw_content"]

        updated = await service.update_skill(
            "override-name",
            content=result["raw_content"],
            expected_version=result["version"],
        )

        assert updated["name"] == "override-name"
        assert updated["version"] == result["version"] + 1

    @pytest.mark.asyncio
    async def test_import_skill_name_override_replaces_null_frontmatter_name(self, service):
        content = """---
name: null
custom-review-key: preserve-me
---

Content.
"""

        result = await service.import_skill(content=content, name="override-null")

        assert "name: override-null" in result["raw_content"]
        assert "name: null" not in result["raw_content"]
        assert "custom-review-key: preserve-me" in result["raw_content"]

    @pytest.mark.asyncio
    async def test_import_skill_invalid_name_param_rejected(self, service):
        """Invalid override names should be rejected by service validation."""
        content = """---
name: valid-name
---
content"""
        with pytest.raises(SkillValidationError, match="Invalid skill name"):
            await service.import_skill(content=content, name="Invalid_Name!")

    @pytest.mark.asyncio
    async def test_import_skill_invalid_frontmatter_name_rejected(self, service):
        """Invalid frontmatter names should be rejected even when importing directly."""
        content = """---
name: Invalid_Name!
---
content"""

        with pytest.raises(SkillValidationError, match="frontmatter skill name"):
            await service.import_skill(content=content)

    @pytest.mark.asyncio
    async def test_import_skill_overwrite(self, service):
        """Test overwriting an existing skill on import."""
        await service.create_skill("existing", "Original")

        content = """---
description: New version
---

New content.
"""
        result = await service.import_skill(
            content=content,
            name="existing",
            overwrite=True,
        )

        assert result["description"] == "New version"
        assert "New content" in result["content"]

    @pytest.mark.asyncio
    async def test_import_overwrite_restores_active_skill_when_replacement_write_fails(
        self,
        service,
        monkeypatch,
    ):
        """A failed replacement must leave the original active instead of in Trash."""
        created = await service.create_skill("rollback-import", "Original")
        original_write_text = Path.write_text

        def _fail_replacement_write(path: Path, data: str, *args, **kwargs):
            if (
                path.name == "SKILL.md"
                and path.parent.name.startswith(
                    ".staging-import-replace-rollback-import-"
                )
                and data == "Replacement"
            ):
                raise OSError("replacement write failed")
            return original_write_text(path, data, *args, **kwargs)

        monkeypatch.setattr(Path, "write_text", _fail_replacement_write)

        with pytest.raises(SkillStorageError, match="Failed to prepare skill bundle"):
            await service.import_skill(
                content="Replacement",
                name="rollback-import",
                overwrite=True,
                expected_version=created["version"],
            )

        current = await service.get_skill("rollback-import", enforce_integrity=False)
        assert current["content"] == "Original"
        assert await service.list_trash(limit=10, offset=0) == []

    @pytest.mark.asyncio
    async def test_import_overwrite_rejects_version_changed_after_preview(self, service):
        """Overwrite confirmation applies only to the exact version shown in preview."""
        created = await service.create_skill("preview-race", "Original")
        preview = await service.preview_import_skill(
            content="Replacement",
            name="preview-race",
        )
        updated = await service.update_skill(
            "preview-race",
            content="Concurrent edit",
            expected_version=created["version"],
        )

        with pytest.raises(SkillConflictError) as exc_info:
            await service.import_skill(
                content="Replacement",
                name="preview-race",
                overwrite=True,
                expected_version=preview["existing_version"],
            )

        assert exc_info.value.expected_version == preview["existing_version"]
        assert exc_info.value.actual_version == updated["version"]
        current = await service.get_skill("preview-race", enforce_integrity=False)
        assert current["content"] == "Concurrent edit"

    @pytest.mark.asyncio
    async def test_import_skill_conflict_without_overwrite(self, service):
        """Test that import fails without overwrite flag."""
        await service.create_skill("existing", "Original")

        with pytest.raises(SkillConflictError):
            await service.import_skill(content="New", name="existing", overwrite=False)

    @pytest.mark.asyncio
    async def test_preview_import_skill_returns_metadata_without_writing(self, service):
        """Previewing an import should parse metadata without creating the skill."""
        content = """---
name: previewed-skill
description: Preview this skill
argument-hint: "[topic]"
allowed-tools: Read, Grep
model: test-model
context: fork
---

Preview content.
"""
        result = await service.preview_import_skill(
            content=content,
            supporting_files={"ref.md": "Reference content"},
        )

        assert result["valid"] is True
        assert result["errors"] == []
        assert result["name"] == "previewed-skill"
        assert result["description"] == "Preview this skill"
        assert result["argument_hint"] == "[topic]"
        assert result["allowed_tools"] == ["Read", "Grep"]
        assert result["model"] == "test-model"
        assert result["context"] == "fork"
        assert result["supporting_file_count"] == 1
        assert result["conflict"] is False
        assert result["can_overwrite"] is False
        assert result["existing_version"] is None
        assert not service._get_skill_dir("previewed-skill").exists()
        assert service._get_db().get_skill_registry("previewed-skill", include_deleted=True) is None

    @pytest.mark.asyncio
    async def test_preview_import_skill_reports_conflict_without_overwriting(self, service):
        """Preview should report conflicts without mutating the existing skill."""
        existing = await service.create_skill("existing", "Original")
        skill_file = service._get_skill_dir("existing") / "SKILL.md"
        original_disk_content = skill_file.read_text(encoding="utf-8")

        result = await service.preview_import_skill(
            content="---\ndescription: Replacement\n---\nReplacement content",
            name="existing",
        )

        assert result["valid"] is True
        assert result["name"] == "existing"
        assert result["description"] == "Replacement"
        assert result["conflict"] is True
        assert result["can_overwrite"] is True
        assert result["existing_version"] == existing["version"]
        assert skill_file.read_text(encoding="utf-8") == original_disk_content
        persisted = await service.get_skill("existing")
        assert persisted["content"] == "Original"

    @pytest.mark.asyncio
    async def test_preview_import_skill_returns_validation_errors_without_writing(self, service):
        """Preview should return validation errors instead of mutating invalid imports."""
        result = await service.preview_import_skill(
            content="---\nname: Invalid_Name!\n---\nInvalid content",
        )

        assert result["valid"] is False
        assert result["name"] is None
        assert result["conflict"] is False
        assert result["can_overwrite"] is False
        assert result["existing_version"] is None
        assert any("frontmatter skill name" in error for error in result["errors"])
        assert not service._get_skill_dir("invalid-name").exists()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "frontmatter_name",
        ["true", "false", "123", "[one, two]", "{nested: value}"],
    )
    async def test_preview_import_rejects_non_string_frontmatter_names(
        self,
        service,
        frontmatter_name,
    ):
        """Preview and import share fail-closed frontmatter name validation."""
        result = await service.preview_import_skill(
            content=f"---\nname: {frontmatter_name}\n---\nBody",
        )

        assert result["valid"] is False
        assert result["errors"] == ["Frontmatter skill name must be a string"]
        assert result["name"] is None
        assert service._get_db().get_skill_registry("invalid-name", include_deleted=True) is None

    @pytest.mark.asyncio
    async def test_preview_import_skill_returns_parse_errors_without_writing(self, service, monkeypatch):
        """Domain parse failures should return review errors without masking server bugs broadly."""
        def fail_parse(*args, **kwargs):
            raise SkillParseError("invalid frontmatter")

        monkeypatch.setattr(service._parser, "parse_content", fail_parse)

        result = await service.preview_import_skill(content="---\nname: parsed\n---\ncontent")

        assert result["valid"] is False
        assert result["name"] is None
        assert result["conflict"] is False
        assert result["errors"] == ["Invalid skill content: invalid frontmatter"]
        assert not service._get_skill_dir("parsed").exists()
        assert service._get_db().get_skill_registry("parsed", include_deleted=True) is None

    def test_public_import_preview_error_preserves_non_traceback_file_messages(self):
        assert _public_import_preview_error("File front-matter is missing required field") == (
            "File front-matter is missing required field"
        )

    def test_public_import_preview_error_filters_traceback_file_frames(self):
        message = 'Traceback (most recent call last):\nFile "/tmp/app.py", line 10, in parse\nInvalid field'

        assert _public_import_preview_error(message) == "Invalid field"

    @pytest.mark.asyncio
    async def test_invalid_import_preview_does_not_resanitize_public_errors(self, service):
        long_detail = "x" * 300
        result = service._invalid_import_preview([f"Invalid skill content: {long_detail}"])

        assert result["errors"] == [f"Invalid skill content: {long_detail}"]

    @pytest.mark.asyncio
    async def test_export_skill(self, service):
        """Test exporting a skill as zip."""
        await service.create_skill(
            "export-test",
            """---
name: export-test
description: Export test
---

Content here.
""",
            supporting_files={"ref.md": "Reference"},
        )

        zip_data = await service.export_skill("export-test")

        # Verify it's valid zip data
        assert zip_data is not None
        assert len(zip_data) > 0
        # Should start with PK (zip magic bytes)
        assert zip_data[:2] == b"PK"

    @pytest.mark.asyncio
    async def test_export_skill_not_found(self, service):
        """Test that exporting a non-existent skill raises NotFoundError."""
        with pytest.raises(SkillNotFoundError):
            await service.export_skill("nonexistent")

    def test_get_context_payload_empty(self, service):
        """Test context payload with no skills."""
        payload = service.get_context_payload()

        assert payload["available_skills"] == []
        assert payload["context_text"] == ""

    @pytest.mark.asyncio
    async def test_get_context_payload_with_skills(self, service):
        """Test context payload with skills."""
        await service.create_skill(
            "skill-a",
            """---
description: Skill A does things
argument-hint: "[arg]"
---
Content A""",
        )

        await service.create_skill(
            "skill-b",
            """---
description: Skill B does other things
---
Content B""",
        )

        payload = service.get_context_payload()

        assert len(payload["available_skills"]) == 2
        assert "<available-skills>" in payload["context_text"]
        assert "skill-a" in payload["context_text"]
        assert "skill-b" in payload["context_text"]
        assert "Skill A does things" in payload["context_text"]

    @pytest.mark.asyncio
    async def test_get_context_payload_async_uses_async_sync(self, service, monkeypatch):
        """Async context payload should use _sync_registry_async (not sync path)."""
        await service.create_skill(
            "async-context-skill",
            """---
description: Async context
---
Body""",
        )

        calls = {"sync": 0, "async": 0}

        def _sync_stub(*_args, **_kwargs):
            calls["sync"] += 1
            raise AssertionError("sync registry should not be called by get_context_payload_async")

        async def _async_stub(*_args, **_kwargs):
            calls["async"] += 1

        monkeypatch.setattr(service, "_sync_registry", _sync_stub)
        monkeypatch.setattr(service, "_sync_registry_async", _async_stub)

        payload = await service.get_context_payload_async()

        assert calls["async"] == 1
        assert calls["sync"] == 0
        assert "async-context-skill" in payload["context_text"]

    @pytest.mark.asyncio
    async def test_get_context_payload_excludes_model_invocation_disabled(self, service):
        """Test that skills with disable_model_invocation are excluded from context."""
        await service.create_skill(
            "visible",
            """---
disable-model-invocation: false
---
Content""",
        )

        await service.create_skill(
            "hidden",
            """---
disable-model-invocation: true
---
Content""",
        )

        payload = service.get_context_payload()

        names = [s["name"] for s in payload["available_skills"]]
        assert "visible" in names
        assert "hidden" not in names

    @pytest.mark.asyncio
    async def test_quarantined_skill_is_filtered_from_context(self, service):
        """Context payload generation must not advertise quarantined skills."""
        from tldw_Server_API.app.core.Context_Integrity.models import (
            ContextIntegrityBootState,
            ContextIntegrityFinding,
        )
        from tldw_Server_API.app.core.Context_Integrity.resolver import (
            ContextIntegrityResolver,
        )

        await service.create_skill(
            "blocked-skill",
            """---
description: Blocked
---
Body""",
        )
        service.integrity_resolver = ContextIntegrityResolver(
            ContextIntegrityBootState(
                mode="enforce",
                degraded=False,
                manifest_sequence=1,
                manifest_digest="sha256:manifest",
                findings=(
                    ContextIntegrityFinding(
                        asset_id="skill:user:1/blocked-skill",
                        state="changed_approved_executable",
                        severity="error",
                        summary="changed",
                        remediation="review",
                        source_type="skill_file",
                    ),
                ),
            )
        )

        payload = service.get_context_payload()

        assert payload["available_skills"] == []
        assert "blocked-skill" not in payload["context_text"]

    @pytest.mark.asyncio
    async def test_quarantined_skill_get_is_blocked(self, service):
        """Direct skill reads must fail closed for quarantined assets."""
        from tldw_Server_API.app.core.Context_Integrity.models import (
            ContextIntegrityBootState,
            ContextIntegrityFinding,
        )
        from tldw_Server_API.app.core.Context_Integrity.resolver import (
            ContextIntegrityBlocked,
            ContextIntegrityResolver,
        )

        await service.create_skill(
            "blocked-skill",
            """---
description: Blocked
---
Body""",
        )
        service.integrity_resolver = ContextIntegrityResolver(
            ContextIntegrityBootState(
                mode="enforce",
                degraded=False,
                manifest_sequence=1,
                manifest_digest="sha256:manifest",
                findings=(
                    ContextIntegrityFinding(
                        asset_id="skill:user:1/blocked-skill",
                        state="changed_approved_executable",
                        severity="error",
                        summary="changed",
                        remediation="review",
                        source_type="skill_file",
                    ),
                ),
            )
        )

        with pytest.raises(ContextIntegrityBlocked):
            await service.get_skill("blocked-skill")

    @pytest.mark.asyncio
    async def test_live_skill_edit_after_boot_is_blocked(self, service):
        """A skill edited after boot must not be read under an old approval digest."""
        from tldw_Server_API.app.core.Context_Integrity.canonicalization import (
            canonical_filesystem_digest,
        )
        from tldw_Server_API.app.core.Context_Integrity.models import (
            ContextIntegrityBootState,
        )
        from tldw_Server_API.app.core.Context_Integrity.resolver import (
            ContextIntegrityBlocked,
            ContextIntegrityResolver,
        )

        initial_content = """---
description: Live
---
Approved body"""
        await service.create_skill("live-skill", initial_content)
        asset_id = "skill:user:1/live-skill"
        approved_digest = canonical_filesystem_digest(
            source_type="skill_file",
            asset_id=asset_id,
            files={"SKILL.md": initial_content.encode("utf-8")},
            metadata={"skill_name": "live-skill"},
        )
        service.integrity_resolver = ContextIntegrityResolver(
            ContextIntegrityBootState(
                mode="enforce",
                degraded=False,
                manifest_sequence=1,
                manifest_digest="sha256:manifest",
                approved_digests_by_asset_id={asset_id: approved_digest},
            )
        )
        (service.skills_dir / "live-skill" / "SKILL.md").write_text(
            "---\ndescription: Live\n---\nModified body",
            encoding="utf-8",
        )

        with pytest.raises(ContextIntegrityBlocked):
            await service.get_skill("live-skill")

    @pytest.mark.asyncio
    async def test_live_skill_edit_after_boot_is_filtered_from_context(self, service):
        """Context payload must omit a skill whose live digest no longer matches approval."""
        from tldw_Server_API.app.core.Context_Integrity.canonicalization import (
            canonical_filesystem_digest,
        )
        from tldw_Server_API.app.core.Context_Integrity.models import (
            ContextIntegrityBootState,
        )
        from tldw_Server_API.app.core.Context_Integrity.resolver import (
            ContextIntegrityResolver,
        )

        initial_content = """---
description: Live
---
Approved body"""
        await service.create_skill("live-skill", initial_content)
        asset_id = "skill:user:1/live-skill"
        approved_digest = canonical_filesystem_digest(
            source_type="skill_file",
            asset_id=asset_id,
            files={"SKILL.md": initial_content.encode("utf-8")},
            metadata={"skill_name": "live-skill"},
        )
        service.integrity_resolver = ContextIntegrityResolver(
            ContextIntegrityBootState(
                mode="enforce",
                degraded=False,
                manifest_sequence=1,
                manifest_digest="sha256:manifest",
                approved_digests_by_asset_id={asset_id: approved_digest},
            )
        )
        (service.skills_dir / "live-skill" / "SKILL.md").write_text(
            "---\ndescription: Live\n---\nModified body",
            encoding="utf-8",
        )

        payload = service.get_context_payload()

        assert payload["available_skills"] == []
        assert "live-skill" not in payload["context_text"]

    @pytest.mark.asyncio
    async def test_list_skills_filters_integrity_before_pagination(self, service):
        """Integrity filtering must happen before list pagination is applied."""
        from tldw_Server_API.app.core.Context_Integrity.canonicalization import (
            canonical_filesystem_digest,
        )
        from tldw_Server_API.app.core.Context_Integrity.models import (
            ContextIntegrityBootState,
        )
        from tldw_Server_API.app.core.Context_Integrity.resolver import (
            ContextIntegrityResolver,
        )

        await service.create_skill("aaa-blocked", "---\ndescription: Blocked\n---\nBlocked")
        allowed_content = "---\ndescription: Allowed\n---\nAllowed"
        await service.create_skill("zzz-allowed", allowed_content)
        allowed_asset_id = "skill:user:1/zzz-allowed"
        allowed_digest = canonical_filesystem_digest(
            source_type="skill_file",
            asset_id=allowed_asset_id,
            files={"SKILL.md": allowed_content.encode("utf-8")},
            metadata={"skill_name": "zzz-allowed"},
        )
        service.integrity_resolver = ContextIntegrityResolver(
            ContextIntegrityBootState(
                mode="enforce",
                degraded=False,
                manifest_sequence=1,
                manifest_digest="sha256:manifest",
                approved_digests_by_asset_id={allowed_asset_id: allowed_digest},
            )
        )

        skills = await service.list_skills(limit=1, offset=0)

        assert [skill.name for skill in skills] == ["zzz-allowed"]
        assert await service.get_total_count() == 1

    @pytest.mark.asyncio
    async def test_create_skill_returns_write_response_under_enforce(self, service):
        """Write APIs can return the newly written skill without approving it for reads."""
        from tldw_Server_API.app.core.Context_Integrity.models import (
            ContextIntegrityBootState,
        )
        from tldw_Server_API.app.core.Context_Integrity.resolver import (
            ContextIntegrityBlocked,
            ContextIntegrityResolver,
        )

        service.integrity_resolver = ContextIntegrityResolver(
            ContextIntegrityBootState(
                mode="enforce",
                degraded=False,
                manifest_sequence=1,
                manifest_digest="sha256:manifest",
                approved_digests_by_asset_id={},
            )
        )

        created = await service.create_skill("pending-skill", "---\ndescription: Pending\n---\nBody")

        assert created["name"] == "pending-skill"
        assert created["description"] == "Pending"
        with pytest.raises(ContextIntegrityBlocked):
            await service.get_skill("pending-skill")

    @pytest.mark.asyncio
    async def test_degraded_global_resolver_blocks_default_service(
        self,
        temp_base_path,
    ):
        """Degraded boot state from the global resolver must fail closed by default."""
        from tldw_Server_API.app.core.Context_Integrity.models import (
            ContextIntegrityBootState,
            ContextIntegrityFinding,
        )
        from tldw_Server_API.app.core.Context_Integrity.resolver import (
            ContextIntegrityBlocked,
            ContextIntegrityResolver,
            clear_global_context_integrity_resolver,
            set_global_context_integrity_resolver,
        )

        chacha_db = CharactersRAGDB(
            db_path=temp_base_path / "ChaChaNotes.db",
            client_id="test_client",
        )
        set_global_context_integrity_resolver(
            ContextIntegrityResolver(
                ContextIntegrityBootState(
                    mode="enforce",
                    degraded=True,
                    manifest_sequence=None,
                    manifest_digest=None,
                    findings=(
                        ContextIntegrityFinding(
                            asset_id="manifest:env",
                            state="signature_invalid",
                            severity="error",
                            summary="bad signature",
                            remediation="fix manifest",
                            source_type="manifest",
                        ),
                    ),
                )
            )
        )
        try:
            service = SkillsService(user_id=1, base_path=temp_base_path, db=chacha_db)
            created = await service.create_skill(
                "degraded-global",
                "---\ndescription: Degraded global\n---\nBody",
            )
            with pytest.raises(ContextIntegrityBlocked):
                await service.get_skill("degraded-global")
        finally:
            clear_global_context_integrity_resolver()
            chacha_db.close_connection()

        assert created["name"] == "degraded-global"

    @pytest.mark.asyncio
    async def test_symlinked_skill_directory_is_not_read_without_resolver(
        self,
        service,
        temp_base_path,
    ):
        """Runtime skill reads must not follow a symlinked skill directory."""
        outside_skill = temp_base_path / "outside-skill"
        outside_skill.mkdir()
        (outside_skill / "SKILL.md").write_text(
            "---\ndescription: Outside\n---\nOutside body",
            encoding="utf-8",
        )
        skill_link = service.skills_dir / "linked-skill"
        try:
            skill_link.symlink_to(outside_skill, target_is_directory=True)
        except (NotImplementedError, OSError):
            pytest.skip("symlink creation is unavailable on this platform")

        with pytest.raises(SkillsError):
            await service.get_skill("linked-skill")

    @pytest.mark.asyncio
    async def test_sync_registry_skips_symlinked_skill_md(self, service, temp_base_path):
        """Sync must not index a skill whose SKILL.md is a symlink."""
        outside_skill_file = temp_base_path / "outside-SKILL.md"
        outside_skill_file.write_text(
            "---\ndescription: Outside\n---\nOutside body",
            encoding="utf-8",
        )
        skill_dir = service.skills_dir / "symlinked-file"
        skill_dir.mkdir()
        try:
            (skill_dir / "SKILL.md").symlink_to(outside_skill_file)
        except (NotImplementedError, OSError):
            pytest.skip("symlink creation is unavailable on this platform")

        skills = await service.list_skills(include_hidden=True)

        assert [skill.name for skill in skills] == []
        assert await service.get_total_count(include_hidden=True) == 0

    @pytest.mark.asyncio
    async def test_get_total_count(self, service):
        """Test getting total skill count."""
        assert await service.get_total_count() == 0

        await service.create_skill("skill-1", "Content")
        await service.create_skill("skill-2", "Content")

        assert await service.get_total_count() == 2

    @pytest.mark.asyncio
    async def test_sync_index_discovers_new_skills(self, service, temp_base_path):
        """Test that index sync discovers skills added to filesystem."""
        # Create a skill directly on disk (bypassing service)
        skills_dir = temp_base_path / "skills" / "manual-skill"
        skills_dir.mkdir(parents=True)
        (skills_dir / "SKILL.md").write_text(
            """---
name: manual-skill
description: Manually created
---
Content"""
        )

        # List should discover it
        skills = await service.list_skills()
        names = [s.name for s in skills]

        assert "manual-skill" in names

    @pytest.mark.asyncio
    async def test_sync_debounce_avoids_redundant_scans(self, temp_base_path):
        """Regression Bug 2: read ops should skip sync when within debounce interval."""
        db_path = temp_base_path / "ChaChaNotes.db"
        chacha_db = CharactersRAGDB(db_path=db_path, client_id="test_client")
        service = SkillsService(user_id=1, base_path=temp_base_path, db=chacha_db, sync_interval=60.0)
        try:
            sync_count = 0
            original_sync = service._sync_registry.__func__  # unbound method

            def counting_sync(self_inner, force=False):
                nonlocal sync_count
                sync_count += 1
                original_sync(self_inner, force=force)

            import types

            service._sync_registry = types.MethodType(counting_sync, service)

            # First call triggers sync
            service.get_context_payload()
            first_count = sync_count

            # Second call (within debounce window) should skip actual scan
            service.get_context_payload()
            assert sync_count == first_count + 1  # Called, but debounce returns early inside
        finally:
            chacha_db.close_connection()

    @pytest.mark.asyncio
    async def test_import_from_zip_invalid_name_rejected(self, service):
        """Regression Bug 6: zip with invalid directory name should be rejected."""
        import zipfile
        from io import BytesIO

        buffer = BytesIO()
        with zipfile.ZipFile(buffer, "w") as zf:
            zf.writestr("Invalid_Name!/SKILL.md", "---\nname: invalid\n---\nContent")
        zip_data = buffer.getvalue()

        with pytest.raises(SkillValidationError, match="Invalid skill name"):
            await service.import_from_zip(zip_data)

    @pytest.mark.asyncio
    async def test_import_from_zip_path_traversal_rejected(self, service):
        """Zip import must reject traversal entries in supporting files."""
        import zipfile
        from io import BytesIO

        buffer = BytesIO()
        with zipfile.ZipFile(buffer, "w") as zf:
            zf.writestr("safe-skill/SKILL.md", "---\nname: safe-skill\n---\nContent")
            zf.writestr("safe-skill/../escape.md", "evil")
        zip_data = buffer.getvalue()

        with pytest.raises(SkillValidationError, match="path traversal"):
            await service.import_from_zip(zip_data)

    @pytest.mark.asyncio
    async def test_import_from_zip_rejects_oversized_skill_md(self, service):
        """Zip import must enforce the SKILL.md content cap before writing files."""
        import zipfile
        from io import BytesIO

        buffer = BytesIO()
        with zipfile.ZipFile(buffer, "w") as zf:
            zf.writestr("large-skill/SKILL.md", "x" * 500_001)
        zip_data = buffer.getvalue()

        with pytest.raises(SkillValidationError, match="SKILL.md.*500KB"):
            await service.import_from_zip(zip_data)

    @pytest.mark.asyncio
    async def test_import_from_zip_rejects_too_many_entries(self, service):
        """Zip import should reject archives with excessive entry counts before reads."""
        import zipfile
        from io import BytesIO

        buffer = BytesIO()
        with zipfile.ZipFile(buffer, "w") as zf:
            zf.writestr("many-entries/SKILL.md", "Content")
            for i in range(100):
                zf.writestr(f"many-entries/extra-{i}.md", "x")
        zip_data = buffer.getvalue()

        with pytest.raises(SkillValidationError, match="too many entries"):
            await service.import_from_zip(zip_data)


class TestSkillSchemaValidation:
    """Tests for schema-level validation (Bug 7 regression)."""

    def test_supporting_files_count_limit(self):
        """Regression Bug 7: too many supporting files should be rejected."""
        import pydantic

        from tldw_Server_API.app.api.v1.schemas.skills_schemas import SkillCreate

        files = {f"file{i:02d}.md": "content" for i in range(25)}
        with pytest.raises(pydantic.ValidationError, match="Too many supporting files"):
            SkillCreate(name="test-skill", content="content", supporting_files=files)

    def test_supporting_files_aggregate_limit(self):
        """Regression Bug 7: total size exceeding 5MB should be rejected."""
        import pydantic

        from tldw_Server_API.app.api.v1.schemas.skills_schemas import SkillCreate

        # Create files that individually pass (< 500KB) but collectively exceed 5MB
        big_content = "x" * 400_000  # ~400KB each
        files = {f"file{i:02d}.md": big_content for i in range(15)}  # ~6MB total
        with pytest.raises(pydantic.ValidationError, match="Total supporting files size"):
            SkillCreate(name="test-skill", content="content", supporting_files=files)

    def test_skill_update_supporting_files_allows_null_delete(self):
        """SkillUpdate should accept null values to indicate delete semantics."""
        from tldw_Server_API.app.api.v1.schemas.skills_schemas import SkillUpdate

        payload = SkillUpdate(supporting_files={"remove.md": None, "keep.md": "updated"})
        assert payload.supporting_files is not None
        assert payload.supporting_files["remove.md"] is None
        assert payload.supporting_files["keep.md"] == "updated"

    def test_skill_import_name_optional_uses_frontmatter(self):
        """SkillImportRequest name should be optional (frontmatter fallback)."""
        from tldw_Server_API.app.api.v1.schemas.skills_schemas import SkillImportRequest

        payload = SkillImportRequest(content="---\nname: from-frontmatter\n---\nBody")
        assert payload.name is None

    def test_skill_import_supporting_files_count_limit(self):
        """Import schema should enforce supporting-files count limit."""
        import pydantic

        from tldw_Server_API.app.api.v1.schemas.skills_schemas import SkillImportRequest

        files = {f"file{i:02d}.md": "content" for i in range(25)}
        with pytest.raises(pydantic.ValidationError, match="Too many supporting files"):
            SkillImportRequest(content="content", supporting_files=files)

    def test_skill_import_supporting_files_aggregate_limit(self):
        """Import schema should enforce supporting-files aggregate size limit."""
        import pydantic

        from tldw_Server_API.app.api.v1.schemas.skills_schemas import SkillImportRequest

        big_content = "x" * 400_000
        files = {f"file{i:02d}.md": big_content for i in range(15)}
        with pytest.raises(pydantic.ValidationError, match="Total supporting files size"):
            SkillImportRequest(content="content", supporting_files=files)


class TestSeedBuiltinSkills:
    """Tests for seed_builtin_skills method."""

    @pytest.fixture
    def temp_base_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def builtin_source_dir(self, temp_base_path):
        builtin_root = temp_base_path / "builtin_source"

        summarize_dir = builtin_root / "summarize"
        summarize_dir.mkdir(parents=True, exist_ok=True)
        (summarize_dir / "SKILL.md").write_text(
            """---
name: summarize
description: Summarize content
---

Summarize this: $ARGUMENTS
""",
            encoding="utf-8",
        )
        (summarize_dir / "guide.md").write_text("Summarization guide", encoding="utf-8")
        templates_dir = summarize_dir / "templates"
        templates_dir.mkdir(parents=True, exist_ok=True)
        (templates_dir / "prompt.txt").write_text("Builtin prompt template", encoding="utf-8")

        code_review_dir = builtin_root / "code-review"
        code_review_dir.mkdir(parents=True, exist_ok=True)
        (code_review_dir / "SKILL.md").write_text(
            """---
name: code-review
description: Review code for issues
---

Review this code: $ARGUMENTS
""",
            encoding="utf-8",
        )
        (code_review_dir / "checklist.md").write_text("Security\nPerformance\nStyle", encoding="utf-8")

        return builtin_root

    @pytest.fixture
    def service(self, temp_base_path, builtin_source_dir, monkeypatch):
        db_path = temp_base_path / "ChaChaNotes.db"
        chacha_db = CharactersRAGDB(db_path=db_path, client_id="test_seed")
        service = SkillsService(user_id=1, base_path=temp_base_path, db=chacha_db)
        monkeypatch.setattr(service, "_get_builtin_skills_dir", lambda: builtin_source_dir)
        yield service
        chacha_db.close_connection()

    @pytest.mark.asyncio
    async def test_seed_builtin_skills_copies_full_directory(self, service):
        """Verify seeding copies SKILL.md, supporting files, and nested content."""
        seeded = await service.seed_builtin_skills()

        assert len(seeded) == 2
        assert "summarize" in seeded
        assert "code-review" in seeded

        summarize_skill = await service.get_skill("summarize")
        assert summarize_skill["name"] == "summarize"
        assert "Summarize this" in summarize_skill["content"]
        assert summarize_skill["supporting_files"] is not None
        assert summarize_skill["supporting_files"]["guide.md"] == "Summarization guide"

        nested_prompt = service.skills_dir / "summarize" / "templates" / "prompt.txt"
        assert nested_prompt.exists()
        assert nested_prompt.read_text(encoding="utf-8") == "Builtin prompt template"

    @pytest.mark.asyncio
    async def test_seed_builtin_skills_no_overwrite(self, service):
        """Verify existing skills are not replaced when overwrite=False."""
        await service.seed_builtin_skills()
        await service.update_skill(
            "summarize",
            "Custom content",
            supporting_files={"guide.md": "Custom guide"},
        )
        custom_prompt = service.skills_dir / "summarize" / "templates" / "prompt.txt"
        custom_prompt.write_text("Custom prompt template", encoding="utf-8")

        seeded = await service.seed_builtin_skills(overwrite=False)
        assert "summarize" not in seeded

        summarize_skill = await service.get_skill("summarize")
        assert "Custom content" in summarize_skill["content"]
        assert summarize_skill["supporting_files"] is not None
        assert summarize_skill["supporting_files"]["guide.md"] == "Custom guide"
        assert custom_prompt.read_text(encoding="utf-8") == "Custom prompt template"

    @pytest.mark.asyncio
    async def test_seed_builtin_skills_overwrite(self, service):
        """Verify overwrite replaces existing skills."""
        await service.seed_builtin_skills()
        await service.update_skill(
            "summarize",
            "Custom content",
            supporting_files={"guide.md": "Custom guide"},
        )
        extra_file = service.skills_dir / "summarize" / "extra.md"
        extra_file.write_text("Should be removed on overwrite", encoding="utf-8")
        custom_prompt = service.skills_dir / "summarize" / "templates" / "prompt.txt"
        custom_prompt.write_text("Custom prompt template", encoding="utf-8")

        seeded = await service.seed_builtin_skills(overwrite=True)
        assert "summarize" in seeded

        summarize_skill = await service.get_skill("summarize")
        assert "Summarize this" in summarize_skill["content"]
        assert summarize_skill["supporting_files"] is not None
        assert summarize_skill["supporting_files"]["guide.md"] == "Summarization guide"
        assert not extra_file.exists()
        assert custom_prompt.read_text(encoding="utf-8") == "Builtin prompt template"

    @pytest.mark.asyncio
    async def test_seed_overwrite_copy_failure_preserves_active_bundle(
        self,
        service,
        builtin_source_dir,
        monkeypatch,
    ):
        """A failed replacement copy must not remove the only active bundle."""
        await service.seed_builtin_skills()
        await service.update_skill("summarize", "Custom active body")
        active_file = service._get_skill_dir("summarize") / "SKILL.md"
        original_copytree = shutil.copytree

        def _copy_then_fail(src, dst, *args, **kwargs):
            result = original_copytree(src, dst, *args, **kwargs)
            if Path(src) == builtin_source_dir / "summarize":
                raise OSError("simulated interrupted replacement copy")
            return result

        monkeypatch.setattr(shutil, "copytree", _copy_then_fail)

        with pytest.raises(SkillStorageError, match="Failed to copy built-in skill"):
            await service.seed_builtin_skills(overwrite=True)

        assert active_file.read_text(encoding="utf-8") == "Custom active body"
        active_row = service._get_db().get_skill_registry("summarize", include_deleted=True)
        assert active_row is not None and active_row["deleted"] is False

    @pytest.mark.asyncio
    async def test_seed_missing_only_preserves_deleted_builtin_in_trash(self, service):
        """Missing-only seeding must not reactivate or duplicate a deleted built-in."""
        await service.seed_builtin_skills()
        summarize = await service.get_skill("summarize")
        await service.delete_skill("summarize", expected_version=summarize["version"])
        deleted_row = service._get_db().get_skill_registry("summarize", include_deleted=True)
        assert deleted_row is not None
        archive_dir = Path(deleted_row["directory_path"])

        seeded = await service.seed_builtin_skills(overwrite=False)

        assert "summarize" not in seeded
        assert archive_dir.is_dir()
        assert not service._get_skill_dir("summarize").exists()
        preserved_row = service._get_db().get_skill_registry("summarize", include_deleted=True)
        assert preserved_row is not None
        assert preserved_row["deleted"] is True

    @pytest.mark.asyncio
    async def test_seed_overwrite_replaces_deleted_builtin_without_orphaning_archive(self, service):
        """Explicit overwrite should replace a deleted built-in and remove its old archive."""
        await service.seed_builtin_skills()
        summarize = await service.get_skill("summarize")
        await service.delete_skill("summarize", expected_version=summarize["version"])
        deleted_row = service._get_db().get_skill_registry("summarize", include_deleted=True)
        assert deleted_row is not None
        archive_dir = Path(deleted_row["directory_path"])

        seeded = await service.seed_builtin_skills(overwrite=True)

        assert "summarize" in seeded
        assert not archive_dir.exists()
        restored = await service.get_skill("summarize")
        assert "Summarize this" in restored["content"]
        assert restored["supporting_files"]["guide.md"] == "Summarization guide"
        assert (service._get_skill_dir("summarize") / "templates" / "prompt.txt").is_file()

    def test_seed_overwrite_holds_lock_while_installing_deleted_builtin(
        self,
        service,
        builtin_source_dir,
        monkeypatch,
    ):
        """A second service cannot reconcile a built-in replacement before activation."""
        asyncio.run(service.seed_builtin_skills())
        summarize = asyncio.run(service.get_skill("summarize"))
        asyncio.run(
            service.delete_skill(
                "summarize",
                expected_version=summarize["version"],
            )
        )

        target_dir = service._get_skill_dir("summarize")
        copy_finished = threading.Event()
        allow_activation = threading.Event()
        original_copytree = shutil.copytree

        def _pause_replacement_copy(src, dst, *args, **kwargs):
            result = original_copytree(src, dst, *args, **kwargs)
            if (
                Path(src) == builtin_source_dir / "summarize"
                and Path(dst).name.startswith(".staging-seed-summarize-")
            ):
                copy_finished.set()
                if not allow_activation.wait(timeout=5):
                    raise RuntimeError("timed out waiting to activate built-in replacement")
            return result

        monkeypatch.setattr(shutil, "copytree", _pause_replacement_copy)
        db = service._get_db()
        contender_db = CharactersRAGDB(
            db_path=db.db_path,
            client_id="seed_replacement_contender",
        )
        contender = SkillsService(
            user_id=service.user_id,
            base_path=service.base_path,
            db=contender_db,
            integrity_resolver=service.integrity_resolver,
        )

        try:
            with ThreadPoolExecutor(max_workers=2) as executor:
                seed_future = executor.submit(
                    lambda: asyncio.run(service.seed_builtin_skills(overwrite=True))
                )
                assert copy_finished.wait(timeout=5)
                contender_future = executor.submit(
                    lambda: asyncio.run(contender.list_skills())
                )
                contender_was_blocked = False
                try:
                    contender_future.result(timeout=0.2)
                except FutureTimeoutError:
                    contender_was_blocked = True
                finally:
                    allow_activation.set()

                seeded = seed_future.result(timeout=5)
                contender_future.result(timeout=5)
        finally:
            allow_activation.set()
            contender_db.close_connection()

        assert contender_was_blocked is True
        assert "summarize" in seeded
        active_row = db.get_skill_registry("summarize", include_deleted=True)
        assert active_row is not None and active_row["deleted"] is False
        assert target_dir.is_dir()

    @pytest.mark.asyncio
    async def test_seed_overwrite_cleans_failed_deleted_replacement_and_remains_retryable(
        self,
        service,
        builtin_source_dir,
        monkeypatch,
    ):
        """A failed built-in copy leaves the archived skill restorable and retryable."""
        await service.seed_builtin_skills()
        summarize = await service.get_skill("summarize")
        await service.delete_skill(
            "summarize",
            expected_version=summarize["version"],
        )
        deleted_row = service._get_db().get_skill_registry(
            "summarize",
            include_deleted=True,
        )
        assert deleted_row is not None
        archive_dir = Path(deleted_row["directory_path"])
        destination_dir = service._get_skill_dir("summarize")
        original_copytree = shutil.copytree

        def _copy_then_fail(src, dst, *args, **kwargs):
            result = original_copytree(src, dst, *args, **kwargs)
            if (
                Path(src) == builtin_source_dir / "summarize"
                and Path(dst).name.startswith(".staging-seed-summarize-")
            ):
                raise OSError("interrupted built-in copy")
            return result

        monkeypatch.setattr(shutil, "copytree", _copy_then_fail)

        with pytest.raises(SkillStorageError, match="Failed to copy built-in skill"):
            await service.seed_builtin_skills(overwrite=True)

        assert archive_dir.is_dir()
        assert not destination_dir.exists()
        restored = await service.restore_skill(
            "summarize",
            expected_version=deleted_row["version"],
        )
        await service.delete_skill(
            "summarize",
            expected_version=restored["version"],
        )

        monkeypatch.setattr(shutil, "copytree", original_copytree)
        seeded = await service.seed_builtin_skills(overwrite=True)

        assert "summarize" in seeded
        assert destination_dir.is_dir()

    @pytest.mark.asyncio
    async def test_seed_overwrite_preserves_preexisting_destination_on_copy_conflict(
        self,
        service,
    ):
        """Cleanup must not delete a destination that predates the copy attempt."""
        await service.seed_builtin_skills()
        summarize = await service.get_skill("summarize")
        await service.delete_skill(
            "summarize",
            expected_version=summarize["version"],
        )
        destination_dir = service._get_skill_dir("summarize")
        destination_dir.mkdir()
        sentinel = destination_dir / "preserve.txt"
        sentinel.write_text("preexisting", encoding="utf-8")

        with pytest.raises(SkillStorageError, match="Failed to copy built-in skill"):
            await service.seed_builtin_skills(overwrite=True)

        assert sentinel.read_text(encoding="utf-8") == "preexisting"
