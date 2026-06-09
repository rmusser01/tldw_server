# tests/Skills/unit/test_skills_service.py
#
# Unit tests for the SkillsService class
#
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.Skills.exceptions import (
    SkillConflictError,
    SkillNotFoundError,
    SkillValidationError,
)
from tldw_Server_API.app.core.Skills.skills_service import SkillMetadata, SkillsService


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
        await service.create_skill("visible", """---
user-invocable: true
---
Content""")

        # Create a hidden skill
        await service.create_skill("hidden", """---
user-invocable: false
---
Content""")

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
        await service.create_skill("skill-a", """---
description: Skill A does things
argument-hint: "[arg]"
---
Content A""")

        await service.create_skill("skill-b", """---
description: Skill B does other things
---
Content B""")

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
        await service.create_skill("visible", """---
disable-model-invocation: false
---
Content""")

        await service.create_skill("hidden", """---
disable-model-invocation: true
---
Content""")

        payload = service.get_context_payload()

        names = [s["name"] for s in payload["available_skills"]]
        assert "visible" in names
        assert "hidden" not in names

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
        (skills_dir / "SKILL.md").write_text("""---
name: manual-skill
description: Manually created
---
Content""")

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
