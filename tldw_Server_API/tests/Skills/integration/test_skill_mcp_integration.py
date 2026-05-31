# tests/Skills/integration/test_skill_mcp_integration.py
#
# Integration tests for Skills context_integration module
#
import tempfile
from pathlib import Path

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.Skills.context_integration import (
    add_skill_tool_to_tools_list,
    build_system_message_with_skills,
    get_skill_tool_definition,
    get_skills_context_text,
    handle_skill_tool_call,
)
from tldw_Server_API.app.core.Skills.exceptions import SkillsError
from tldw_Server_API.app.core.Skills.skills_service import SkillsService

pytestmark = pytest.mark.integration


@pytest.fixture()
def temp_env():
    """Provide temp dir with a SkillsService that has skills."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base_path = Path(tmpdir)
        db_path = base_path / "ChaChaNotes.db"
        db = CharactersRAGDB(db_path=db_path, client_id="test_mcp")
        service = SkillsService(user_id=1, base_path=base_path, db=db)
        yield {
            "base_path": base_path,
            "db": db,
            "service": service,
            "user_id": 1,
        }
        db.close_connection()


@pytest.fixture()
def env_with_skills(temp_env):
    """Provide temp env with 2 skills already created."""
    import asyncio
    service = temp_env["service"]
    loop = asyncio.new_event_loop()
    loop.run_until_complete(service.create_skill(
        "summarize",
        "---\ndescription: Summarize text\nargument-hint: \"[text]\"\n---\nSummarize: $ARGUMENTS",
    ))
    loop.run_until_complete(service.create_skill(
        "review",
        "---\ndescription: Code review\n---\nReview the code:\n$ARGUMENTS",
    ))
    loop.close()
    return temp_env


class TestGetSkillsContextText:
    def test_returns_formatted_string(self, env_with_skills):
        env = env_with_skills
        text = get_skills_context_text(env["user_id"], env["base_path"], db=env["db"])

        assert "<available-skills>" in text
        assert "summarize" in text
        assert "review" in text
        assert "Summarize text" in text

    def test_returns_empty_when_no_skills(self, temp_env):
        env = temp_env
        text = get_skills_context_text(env["user_id"], env["base_path"], db=env["db"])
        assert text == ""


class TestBuildSystemMessageWithSkills:
    def test_appends_to_base(self, env_with_skills):
        env = env_with_skills
        result = build_system_message_with_skills(
            "You are a helpful assistant.",
            env["user_id"],
            env["base_path"],
            db=env["db"],
        )

        assert result.startswith("You are a helpful assistant.")
        assert "<available-skills>" in result
        assert "summarize" in result

    def test_returns_base_when_no_skills(self, temp_env):
        env = temp_env
        result = build_system_message_with_skills(
            "Base message",
            env["user_id"],
            env["base_path"],
            db=env["db"],
        )
        assert result == "Base message"


class TestGetSkillToolDefinition:
    def test_has_correct_schema(self):
        tool_def = get_skill_tool_definition()

        assert tool_def["type"] == "function"
        func = tool_def["function"]
        assert func["name"] == "Skill"
        assert "skill" in func["parameters"]["properties"]
        assert "args" in func["parameters"]["properties"]
        assert "skill" in func["parameters"]["required"]


class TestHandleSkillToolCall:
    @pytest.mark.asyncio
    async def test_success(self, env_with_skills):
        env = env_with_skills
        result = await handle_skill_tool_call(
            skill_name="summarize",
            args="this is my text",
            user_id=env["user_id"],
            base_path=env["base_path"],
            db=env["db"],
        )

        assert result["success"] is True
        assert result["skill_name"] == "summarize"
        assert "this is my text" in result["rendered_prompt"]
        assert result["execution_mode"] == "inline"

    @pytest.mark.asyncio
    async def test_not_found(self, temp_env):
        env = temp_env
        result = await handle_skill_tool_call(
            skill_name="nonexistent",
            args="",
            user_id=env["user_id"],
            base_path=env["base_path"],
            db=env["db"],
        )

        assert result["success"] is False
        assert "not found" in result["error"]

    @pytest.mark.asyncio
    async def test_sanitizes_backend_errors(self, monkeypatch, temp_env):
        env = temp_env

        async def fail_get_skill(self, name):
            raise SkillsError("skills backend exploded at /private/skills-context-cache")

        monkeypatch.setattr(SkillsService, "get_skill", fail_get_skill)

        result = await handle_skill_tool_call(
            skill_name="summarize",
            args="",
            user_id=env["user_id"],
            base_path=env["base_path"],
            db=env["db"],
        )

        assert result == {"success": False, "error": "skill_execution_failed"}


class TestAddSkillToolToToolsList:
    def test_adds_when_skills_exist(self, env_with_skills):
        env = env_with_skills
        tools = [{"type": "function", "function": {"name": "Read"}}]
        result = add_skill_tool_to_tools_list(
            tools, env["user_id"], env["base_path"], db=env["db"],
        )

        tool_names = []
        for t in result:
            func = t.get("function", {})
            name = func.get("name") or t.get("name")
            if name:
                tool_names.append(name)

        assert "Skill" in tool_names
        assert "Read" in tool_names
        assert len(result) == 2  # Original + Skill

    def test_skips_when_no_skills(self, temp_env):
        env = temp_env
        tools = [{"type": "function", "function": {"name": "Read"}}]
        result = add_skill_tool_to_tools_list(
            tools, env["user_id"], env["base_path"], db=env["db"],
        )

        # Skill tool should NOT be added since no skills exist
        tool_names = []
        for t in result:
            func = t.get("function", {})
            name = func.get("name") or t.get("name")
            if name:
                tool_names.append(name)

        assert "Skill" not in tool_names
        assert len(result) == 1
