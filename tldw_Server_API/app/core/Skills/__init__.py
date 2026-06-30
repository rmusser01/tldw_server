# app/core/Skills/__init__.py
#
# Skills module for SKILL.md support
#
# This module implements Claude Code-style skills with server-side storage.
# Skills are stored per-user on the server, auto-loaded into chat context,
# and invoked via a dedicated Skill tool.
#
"""
Skills Module
=============

Provides SKILL.md first-class support for tldw_server:

- **SkillParser**: Parse SKILL.md files with YAML frontmatter
- **SkillsService**: CRUD operations and file management
- **SkillExecutor**: Execute skills with argument substitution

Usage:
    from tldw_Server_API.app.core.Skills import SkillsService, SkillParser, SkillExecutor

    # Parse a skill file
    parser = SkillParser()
    parsed = parser.parse_content(skill_content)

    # Manage skills
    service = SkillsService(user_id=1, base_path=Path("Databases/user_databases/1"), db=chacha_db)
    skill = await service.create_skill("my-skill", content)

    # Execute a skill
    executor = SkillExecutor()
    result = await executor.execute(skill, args="arg1 arg2")
"""

from tldw_Server_API.app.core.Skills.context_integration import (
    add_skill_tool_to_tools_list,
    add_skill_tool_to_tools_list_async,
    build_system_message_with_skills,
    build_system_message_with_skills_async,
    get_skill_tool_definition,
    get_skills_context_text,
    get_skills_context_text_async,
    handle_skill_tool_call,
)
from tldw_Server_API.app.core.Skills.exceptions import (
    SkillConflictError,
    SkillExecutionError,
    SkillNotFoundError,
    SkillParseError,
    SkillsError,
    SkillValidationError,
)
from tldw_Server_API.app.core.Skills.skill_executor import SKILL_TOOL_DEFINITION, SkillExecutor
from tldw_Server_API.app.core.Skills.skill_parser import ParsedSkill, SkillParser
from tldw_Server_API.app.core.Skills.skills_service import SkillsService

__all__ = [
    # Parser
    "SkillParser",
    "ParsedSkill",
    # Service
    "SkillsService",
    # Executor
    "SkillExecutor",
    "SKILL_TOOL_DEFINITION",
    # Exceptions
    "SkillsError",
    "SkillNotFoundError",
    "SkillValidationError",
    "SkillConflictError",
    "SkillParseError",
    "SkillExecutionError",
    # Context integration
    "get_skills_context_text",
    "get_skills_context_text_async",
    "build_system_message_with_skills",
    "build_system_message_with_skills_async",
    "get_skill_tool_definition",
    "handle_skill_tool_call",
    "add_skill_tool_to_tools_list",
    "add_skill_tool_to_tools_list_async",
]
