# app/core/Skills/context_integration.py
#
# Integration helpers for injecting skills context into chat
#
"""
Skills Context Integration
==========================

Provides helpers for integrating skills into the chat system:
- Context injection for system messages
- Skill tool definition for tool-enabled chats
- Skill invocation handling
"""

from pathlib import Path
from typing import Any, Optional

from loguru import logger

from tldw_Server_API.app.core.Skills.skill_executor import (
    SKILL_TOOL_DEFINITION,
    RequestContext,
    SkillExecutor,
)
from tldw_Server_API.app.core.Skills.skills_service import SkillsService


def get_skills_context_text(user_id: int, base_path: Path, db: Any | None = None) -> str:
    """
    Get formatted skills context for injection into system message.

    Args:
        user_id: The user ID
        base_path: Base path for user databases

    Returns:
        Formatted skills context string, or empty string if no skills
    """
    try:
        service = SkillsService(user_id=user_id, base_path=base_path, db=db)
        payload = service.get_context_payload()
        return payload.get("context_text", "")
    except Exception as e:
        logger.warning(f"Failed to get skills context for user {user_id}: {e}")
        return ""


def build_system_message_with_skills(
    base_system_message: Optional[str],
    user_id: int,
    base_path: Path,
    db: Any | None = None,
) -> str:
    """
    Build a system message that includes available skills.

    Args:
        base_system_message: The original system message (can be None)
        user_id: The user ID
        base_path: Base path for user databases

    Returns:
        System message with skills context appended
    """
    skills_context = get_skills_context_text(user_id, base_path, db=db)

    parts = []
    if base_system_message:
        parts.append(base_system_message)
    if skills_context:
        parts.append(skills_context)

    return "\n\n".join(parts) if parts else ""


def get_skill_tool_definition() -> dict[str, Any]:
    """
    Get the Skill tool definition for tool-enabled chats.

    Returns:
        Tool definition dict in OpenAI function calling format
    """
    return SKILL_TOOL_DEFINITION


async def handle_skill_tool_call(
    skill_name: str,
    args: str,
    user_id: int,
    base_path: Path,
    db: Any | None = None,
    request_context: RequestContext | None = None,
) -> dict[str, Any]:
    """
    Handle a Skill tool invocation from the LLM.

    Args:
        skill_name: The skill to execute
        args: Arguments for the skill
        user_id: The user ID
        base_path: Base path for user databases

    Returns:
        Result dict with rendered prompt and metadata
    """
    from tldw_Server_API.app.core.Skills.exceptions import SkillNotFoundError, SkillsError

    try:
        service = SkillsService(user_id=user_id, base_path=base_path, db=db)
        skill_data = await service.get_skill(skill_name)

        executor = SkillExecutor()
        result = await executor.execute(
            skill_data=skill_data,
            arguments=args or "",
            context=request_context,
        )

        return {
            "success": True,
            "skill_name": result.skill_name,
            "rendered_prompt": result.rendered_prompt,
            "allowed_tools": result.allowed_tools,
            "model_override": result.model_override,
            "execution_mode": result.execution_mode,
            "fork_output": result.fork_output,
        }
    except SkillNotFoundError:
        logger.warning(f"Skill not found: {skill_name}")
        return {
            "success": False,
            "error": f"Skill '{skill_name}' not found",
        }
    except SkillsError:
        logger.error("Error executing skill '{}'", skill_name)
        return {
            "success": False,
            "error": "skill_execution_failed",
        }


def add_skill_tool_to_tools_list(
    tools: Optional[list[dict[str, Any]]],
    user_id: int,
    base_path: Path,
    db: Any | None = None,
) -> list[dict[str, Any]]:
    """
    Add the Skill tool to a tools list if the user has skills.

    Args:
        tools: Existing tools list (can be None)
        user_id: The user ID
        base_path: Base path for user databases

    Returns:
        Tools list with Skill tool added (if user has skills)
    """
    result = list(tools) if tools else []

    # Check if user has any skills
    try:
        service = SkillsService(user_id=user_id, base_path=base_path, db=db)
        skills = service.get_context_payload().get("available_skills", [])
        if skills:
            # Add Skill tool if not already present
            skill_tool = get_skill_tool_definition()
            tool_names = []
            for tool in result:
                if not isinstance(tool, dict):
                    continue
                func = tool.get("function")
                if isinstance(func, dict) and func.get("name"):
                    tool_names.append(func.get("name"))
                elif tool.get("name"):
                    tool_names.append(tool.get("name"))
            if "Skill" not in tool_names:
                result.append(skill_tool)
    except Exception as e:
        logger.warning(f"Failed to check skills for user {user_id}: {e}")

    return result


#
# End of context_integration.py
#######################################################################################################################
