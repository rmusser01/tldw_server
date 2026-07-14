# app/core/Skills/skill_parser.py
#
# Parse SKILL.md files with YAML frontmatter
#
"""
Skill Parser
============

Parses SKILL.md files that follow the Claude Code format:

```markdown
---
name: my-skill
description: What this skill does
argument-hint: "[arg1] [arg2]"
disable-model-invocation: false
user-invocable: true
allowed-tools: Read, Grep, Glob, Bash(git *)
model: default
context: inline
---

Skill instructions here...
$ARGUMENTS will be replaced with the arguments.
${0}, ${1}, etc. for specific arguments.
```
"""

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml
from loguru import logger

from tldw_Server_API.app.core.Skills.exceptions import SkillParseError


@dataclass
class SkillFrontmatter:
    """Parsed frontmatter from a SKILL.md file."""
    name: Optional[str] = None
    description: Optional[str] = None
    argument_hint: Optional[str] = None
    disable_model_invocation: bool = False
    user_invocable: bool = True
    allowed_tools: Optional[list[str]] = None
    model: Optional[str] = None
    context: str = "inline"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SkillFrontmatter":
        """Create frontmatter from parsed YAML dict with key normalization."""
        # Normalize keys: replace hyphens with underscores
        normalized = {}
        for key, value in data.items():
            norm_key = key.replace("-", "_")
            normalized[norm_key] = value

        # Handle allowed-tools which can be string or list
        allowed_tools = normalized.get("allowed_tools")
        if isinstance(allowed_tools, str):
            # Split comma-separated string into list
            allowed_tools = [t.strip() for t in allowed_tools.split(",") if t.strip()]
        elif allowed_tools is not None and not isinstance(allowed_tools, list):
            allowed_tools = [str(allowed_tools)]

        return cls(
            name=normalized.get("name"),
            description=normalized.get("description"),
            argument_hint=normalized.get("argument_hint"),
            disable_model_invocation=bool(normalized.get("disable_model_invocation", False)),
            user_invocable=bool(normalized.get("user_invocable", True)),
            allowed_tools=allowed_tools,
            model=normalized.get("model"),
            context=normalized.get("context", "inline"),
        )


def _split_frontmatter(content: str) -> tuple[str, str] | None:
    first_line_end = content.find("\n")
    if first_line_end < 0 or content[:first_line_end].strip() != "---":
        return None

    yaml_start = first_line_end + 1
    line_start = yaml_start
    while True:
        line_end = content.find("\n", line_start)
        if line_end < 0:
            return None
        if content[line_start:line_end].strip() == "---":
            return content[yaml_start:line_start].rstrip("\r\n"), content[line_end + 1 :]
        line_start = line_end + 1


@dataclass
class ParsedSkill:
    """Fully parsed skill from SKILL.md and supporting files."""
    frontmatter: SkillFrontmatter
    content: str  # Markdown content after frontmatter
    raw_content: str  # Original full content including frontmatter
    supporting_files: dict[str, str] = field(default_factory=dict)  # filename -> content
    content_hash: Optional[str] = None  # SHA256 of raw_content for change detection

    def __post_init__(self):
        if self.content_hash is None:
            self.content_hash = hashlib.sha256(self.raw_content.encode()).hexdigest()


class SkillParser:
    """Parser for SKILL.md files with YAML frontmatter."""

    def parse_content(self, content: str, default_name: Optional[str] = None) -> ParsedSkill:
        """
        Parse SKILL.md content string.

        Args:
            content: The raw SKILL.md file content
            default_name: Name to use if not specified in frontmatter

        Returns:
            ParsedSkill with frontmatter and markdown content

        Raises:
            SkillParseError: If parsing fails
        """
        if not content or not content.strip():
            raise SkillParseError("SKILL.md content is empty")

        raw_content = content
        frontmatter = SkillFrontmatter()
        markdown_content = content

        # Try to extract frontmatter
        frontmatter_parts = _split_frontmatter(content)
        if frontmatter_parts:
            frontmatter_yaml, markdown_content = frontmatter_parts
            try:
                parsed_yaml = yaml.safe_load(frontmatter_yaml)
                if parsed_yaml and isinstance(parsed_yaml, dict):
                    frontmatter = SkillFrontmatter.from_dict(parsed_yaml)
            except yaml.YAMLError as e:
                logger.warning(f"Failed to parse SKILL.md frontmatter: {e}")
                raise SkillParseError(f"Invalid YAML frontmatter: {e}", detail=str(e)) from e

        # Apply default name if not in frontmatter
        if not frontmatter.name and default_name:
            frontmatter.name = default_name

        # Validate context value
        if frontmatter.context not in ("inline", "fork"):
            raise SkillParseError(
                f"Invalid context value: {frontmatter.context}. Must be 'inline' or 'fork'."
            )

        # If no description, try to extract from first paragraph
        if not frontmatter.description:
            frontmatter.description = self._extract_first_paragraph(markdown_content)

        return ParsedSkill(
            frontmatter=frontmatter,
            content=markdown_content.strip(),
            raw_content=raw_content,
        )

    def parse_directory(self, skill_dir: Path) -> ParsedSkill:
        """
        Parse a skill directory containing SKILL.md and optional supporting files.

        Args:
            skill_dir: Path to the skill directory

        Returns:
            ParsedSkill with frontmatter, content, and supporting files

        Raises:
            SkillParseError: If parsing fails
            FileNotFoundError: If SKILL.md doesn't exist
        """
        skill_file = skill_dir / "SKILL.md"
        if not skill_file.exists():
            raise FileNotFoundError(f"SKILL.md not found in {skill_dir}")

        try:
            content = skill_file.read_text(encoding="utf-8")
        except Exception as e:
            raise SkillParseError(f"Failed to read SKILL.md: {e}", detail=str(e)) from e

        # Parse the main file
        parsed = self.parse_content(content, default_name=skill_dir.name)

        # Load supporting files
        supporting_files: dict[str, str] = {}
        for f in skill_dir.iterdir():
            if f.is_file() and f.name != "SKILL.md":
                # Only include text files
                if f.suffix.lower() in (".md", ".txt", ".json", ".yaml", ".yml", ".py", ".sh"):
                    try:
                        supporting_files[f.name] = f.read_text(encoding="utf-8")
                    except Exception as e:
                        logger.warning(f"Failed to read supporting file {f}: {e}")

        parsed.supporting_files = supporting_files
        return parsed

    def _extract_first_paragraph(self, content: str) -> Optional[str]:
        """Extract first non-empty paragraph from markdown content."""
        if not content:
            return None

        lines = content.strip().split("\n")
        paragraph_lines = []

        for line in lines:
            stripped = line.strip()
            # Skip headers
            if stripped.startswith("#"):
                if paragraph_lines:
                    break
                continue
            # Empty line ends paragraph
            if not stripped:
                if paragraph_lines:
                    break
                continue
            paragraph_lines.append(stripped)

        if paragraph_lines:
            result = " ".join(paragraph_lines)
            # Truncate if too long
            if len(result) > 500:
                result = result[:497] + "..."
            return result
        return None

    def rewrite_frontmatter_name(self, content: str, name: str) -> str:
        """Rewrite a declared frontmatter name while preserving all YAML fields."""
        frontmatter_parts = _split_frontmatter(content)
        if frontmatter_parts is None:
            return content

        frontmatter_yaml, markdown_content = frontmatter_parts
        try:
            parsed_yaml = yaml.safe_load(frontmatter_yaml) or {}
        except yaml.YAMLError as e:
            raise SkillParseError(f"Invalid YAML frontmatter: {e}", detail=str(e)) from e
        if not isinstance(parsed_yaml, dict):
            raise SkillParseError("YAML frontmatter must be a mapping")
        if parsed_yaml.get("name") == name:
            return content

        parsed_yaml["name"] = name
        rewritten_yaml = yaml.safe_dump(
            parsed_yaml,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        ).rstrip("\n")
        return f"---\n{rewritten_yaml}\n---\n{markdown_content}"

    def serialize_skill(
        self,
        frontmatter: SkillFrontmatter,
        content: str,
    ) -> str:
        """
        Serialize frontmatter and content back to SKILL.md format.

        Args:
            frontmatter: The skill frontmatter
            content: The markdown content

        Returns:
            Complete SKILL.md file content
        """
        # Build frontmatter dict, excluding None values and defaults
        fm_dict: dict[str, Any] = {}

        if frontmatter.name:
            fm_dict["name"] = frontmatter.name
        if frontmatter.description:
            fm_dict["description"] = frontmatter.description
        if frontmatter.argument_hint:
            fm_dict["argument-hint"] = frontmatter.argument_hint
        if frontmatter.disable_model_invocation:
            fm_dict["disable-model-invocation"] = True
        if not frontmatter.user_invocable:
            fm_dict["user-invocable"] = False
        if frontmatter.allowed_tools:
            fm_dict["allowed-tools"] = ", ".join(frontmatter.allowed_tools)
        if frontmatter.model:
            fm_dict["model"] = frontmatter.model
        if frontmatter.context != "inline":
            fm_dict["context"] = frontmatter.context

        # Build the file content
        if fm_dict:
            yaml_str = yaml.dump(fm_dict, default_flow_style=False, allow_unicode=True)
            return f"---\n{yaml_str}---\n\n{content}"
        else:
            return content
