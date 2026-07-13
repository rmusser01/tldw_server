# tests/Skills/unit/test_skills_service.py
#
# Unit tests for the SkillsService class
#
import tempfile
import threading
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import ConflictError
from tldw_Server_API.app.core.Skills.exceptions import (
    SkillConflictError,
    SkillsError,
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

        from tldw_Server_API.app.core.Skills import skills_service as skills_service_mod

        original_rmtree = skills_service_mod.shutil.rmtree

        def _fail_rmtree(path, *args, **kwargs):
            if Path(path) == skill_dir:
                raise OSError("simulated directory lock")
            return original_rmtree(path, *args, **kwargs)

        monkeypatch.setattr(skills_service_mod.shutil, "rmtree", _fail_rmtree)

        with pytest.raises(SkillStorageError, match="simulated directory lock"):
            await service.delete_skill("delete-restore", expected_version=created["version"])

        row = service._get_db().get_skill_registry("delete-restore", include_deleted=False)
        assert row is not None
        assert skill_dir.exists()
        assert (skill_dir / "SKILL.md").read_text(encoding="utf-8") == "Content"

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
---

Content.
"""
        result = await service.import_skill(content=content, name="override-name")

        assert result["name"] == "override-name"

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
