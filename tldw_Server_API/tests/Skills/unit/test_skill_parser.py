# tests/Skills/unit/test_skill_parser.py
#
# Unit tests for the SkillParser class
#
import tempfile
from pathlib import Path

import pytest

from tldw_Server_API.app.core.Skills.exceptions import SkillParseError
from tldw_Server_API.app.core.Skills.skill_parser import (
    SkillFrontmatter,
    SkillParser,
)


class TestSkillFrontmatter:
    """Tests for SkillFrontmatter parsing."""

    def test_from_dict_basic(self):
        """Test basic frontmatter parsing."""
        data = {
            "name": "test-skill",
            "description": "A test skill",
        }
        fm = SkillFrontmatter.from_dict(data)

        assert fm.name == "test-skill"
        assert fm.description == "A test skill"
        assert fm.disable_model_invocation is False
        assert fm.user_invocable is True
        assert fm.context == "inline"

    def test_from_dict_with_hyphens(self):
        """Test that hyphenated keys are normalized to underscores."""
        data = {
            "argument-hint": "[arg1]",
            "disable-model-invocation": True,
            "user-invocable": False,
            "allowed-tools": "Read, Grep, Glob",
        }
        fm = SkillFrontmatter.from_dict(data)

        assert fm.argument_hint == "[arg1]"
        assert fm.disable_model_invocation is True
        assert fm.user_invocable is False
        assert fm.allowed_tools == ["Read", "Grep", "Glob"]

    def test_from_dict_allowed_tools_as_list(self):
        """Test allowed_tools as a list."""
        data = {
            "allowed-tools": ["Read", "Grep", "Bash(git *)"],
        }
        fm = SkillFrontmatter.from_dict(data)

        assert fm.allowed_tools == ["Read", "Grep", "Bash(git *)"]

    def test_from_dict_fork_context(self):
        """Test fork context mode."""
        data = {
            "context": "fork",
        }
        fm = SkillFrontmatter.from_dict(data)

        assert fm.context == "fork"


class TestSkillParser:
    """Tests for the SkillParser class."""

    @pytest.fixture
    def parser(self):
        return SkillParser()

    def test_parse_content_simple(self, parser):
        """Test parsing simple content without frontmatter."""
        content = "This is a simple skill content."
        parsed = parser.parse_content(content, default_name="simple")

        assert parsed.frontmatter.name == "simple"
        assert parsed.content == content
        assert parsed.raw_content == content

    def test_parse_content_with_frontmatter(self, parser):
        """Test parsing content with YAML frontmatter."""
        content = """---
name: my-skill
description: A useful skill
argument-hint: "[level]"
---

Execute the task with $ARGUMENTS detail level.
"""
        parsed = parser.parse_content(content)

        assert parsed.frontmatter.name == "my-skill"
        assert parsed.frontmatter.description == "A useful skill"
        assert parsed.frontmatter.argument_hint == "[level]"
        assert "Execute the task" in parsed.content
        assert "$ARGUMENTS" in parsed.content

    def test_parse_content_with_crlf_frontmatter(self, parser):
        """Test parsing YAML frontmatter with Windows line endings."""
        content = "---\r\nname: crlf-skill\r\n---\r\nBody\r\n"

        parsed = parser.parse_content(content)

        assert parsed.frontmatter.name == "crlf-skill"
        assert parsed.content == "Body"

    def test_parse_content_frontmatter_overrides_default_name(self, parser):
        """Test that frontmatter name overrides default name."""
        content = """---
name: frontmatter-name
---

Content here.
"""
        parsed = parser.parse_content(content, default_name="default-name")

        assert parsed.frontmatter.name == "frontmatter-name"

    def test_parse_content_uses_default_name_when_not_in_frontmatter(self, parser):
        """Test that default name is used when not in frontmatter."""
        content = """---
description: No name field
---

Content here.
"""
        parsed = parser.parse_content(content, default_name="default-name")

        assert parsed.frontmatter.name == "default-name"

    def test_parse_content_empty_raises_error(self, parser):
        """Test that empty content raises SkillParseError."""
        with pytest.raises(SkillParseError, match="empty"):
            parser.parse_content("")

        with pytest.raises(SkillParseError, match="empty"):
            parser.parse_content("   \n  ")

    def test_parse_content_invalid_yaml_raises_error(self, parser):
        """Test that invalid YAML frontmatter raises SkillParseError."""
        content = """---
name: [invalid yaml
description:
---

Content
"""
        with pytest.raises(SkillParseError, match="YAML"):
            parser.parse_content(content)

    def test_parse_content_invalid_context_raises_error(self, parser):
        """Test that invalid context value raises SkillParseError."""
        content = """---
context: invalid
---

Content
"""
        with pytest.raises(SkillParseError, match="context"):
            parser.parse_content(content)

    def test_parse_content_extracts_description_from_first_paragraph(self, parser):
        """Test that description is extracted from first paragraph if not in frontmatter."""
        content = """---
name: no-desc
---

This is the first paragraph of the skill.
It continues on this line.

This is the second paragraph.
"""
        parsed = parser.parse_content(content)

        assert parsed.frontmatter.description is not None
        assert "first paragraph" in parsed.frontmatter.description

    def test_parse_content_generates_hash(self, parser):
        """Test that content hash is generated."""
        content = "Test content"
        parsed = parser.parse_content(content, default_name="test")

        assert parsed.content_hash is not None
        assert len(parsed.content_hash) == 64  # SHA256 hex

    def test_parse_directory(self, parser):
        """Test parsing a skill directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            skill_dir = Path(tmpdir) / "my-skill"
            skill_dir.mkdir()

            # Write SKILL.md
            skill_file = skill_dir / "SKILL.md"
            skill_file.write_text("""---
name: my-skill
description: A test skill
---

Skill instructions here.
""")

            # Write a supporting file
            ref_file = skill_dir / "reference.md"
            ref_file.write_text("Reference documentation.")

            parsed = parser.parse_directory(skill_dir)

            assert parsed.frontmatter.name == "my-skill"
            assert parsed.frontmatter.description == "A test skill"
            assert "Skill instructions" in parsed.content
            assert "reference.md" in parsed.supporting_files
            assert parsed.supporting_files["reference.md"] == "Reference documentation."

    def test_parse_directory_uses_dirname_as_default_name(self, parser):
        """Test that directory name is used as default skill name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            skill_dir = Path(tmpdir) / "dirname-skill"
            skill_dir.mkdir()

            skill_file = skill_dir / "SKILL.md"
            skill_file.write_text("Simple content without frontmatter.")

            parsed = parser.parse_directory(skill_dir)

            assert parsed.frontmatter.name == "dirname-skill"

    def test_parse_directory_missing_skill_md_raises_error(self, parser):
        """Test that missing SKILL.md raises FileNotFoundError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            skill_dir = Path(tmpdir) / "no-skill"
            skill_dir.mkdir()

            with pytest.raises(FileNotFoundError):
                parser.parse_directory(skill_dir)

    def test_serialize_skill_minimal(self, parser):
        """Test serializing a skill with minimal frontmatter."""
        fm = SkillFrontmatter(name="simple")
        content = "Skill content"

        result = parser.serialize_skill(fm, content)

        assert "---\n" in result
        assert "name: simple" in result
        assert "Skill content" in result

    def test_serialize_skill_full(self, parser):
        """Test serializing a skill with full frontmatter."""
        fm = SkillFrontmatter(
            name="full-skill",
            description="A full skill",
            argument_hint="[arg]",
            disable_model_invocation=True,
            user_invocable=False,
            allowed_tools=["Read", "Grep"],
            model="gpt-4",
            context="fork",
        )
        content = "Full skill content"

        result = parser.serialize_skill(fm, content)

        assert "name: full-skill" in result
        assert "description: A full skill" in result
        assert "argument-hint:" in result
        assert "disable-model-invocation: true" in result
        assert "user-invocable: false" in result
        assert "allowed-tools:" in result
        assert "model: gpt-4" in result
        assert "context: fork" in result
        assert "Full skill content" in result

    def test_serialize_skill_omits_defaults(self, parser):
        """Test that default values are omitted from serialization."""
        fm = SkillFrontmatter(
            name="defaults",
            disable_model_invocation=False,  # Default
            user_invocable=True,  # Default
            context="inline",  # Default
        )
        content = "Content"

        result = parser.serialize_skill(fm, content)

        assert "disable-model-invocation" not in result
        assert "user-invocable: false" not in result  # Only false is written
        assert "context: fork" not in result  # Only fork is written
