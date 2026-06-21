# tests/Skills/unit/test_skill_executor.py
#
# Unit tests for the SkillExecutor class
#
import pytest

from tldw_Server_API.app.core.Skills.skill_executor import (
    SKILL_TOOL_DEFINITION,
    RequestContext,
    SkillExecutor,
)


class TestSkillExecutor:
    """Tests for the SkillExecutor class."""

    @pytest.fixture
    def executor(self):
        return SkillExecutor()

    class TestArgumentSubstitution:
        """Tests for argument substitution."""

        @pytest.fixture
        def executor(self):
            return SkillExecutor()

        def test_substitute_arguments_full(self, executor):
            """Test $ARGUMENTS substitution."""
            content = "Do something with $ARGUMENTS please."
            result = executor.substitute_arguments(content, "arg1 arg2 arg3")

            assert result == "Do something with arg1 arg2 arg3 please."

        def test_substitute_arguments_empty(self, executor):
            """Test $ARGUMENTS with empty arguments."""
            content = "Do something with $ARGUMENTS please."
            result = executor.substitute_arguments(content, "")

            assert result == "Do something with  please."

        def test_substitute_arguments_indexed(self, executor):
            """Test ${0}, ${1}, ${2} substitution."""
            content = "First: ${0}, Second: ${1}, Third: ${2}"
            result = executor.substitute_arguments(content, "apple banana cherry")

            assert result == "First: apple, Second: banana, Third: cherry"

        def test_substitute_arguments_indexed_bracket_form(self, executor):
            """Test $ARGUMENTS[0], $ARGUMENTS[1] substitution."""
            content = "First: $ARGUMENTS[0], Second: $ARGUMENTS[1]"
            result = executor.substitute_arguments(content, "apple banana")

            assert result == "First: apple, Second: banana"

        def test_substitute_arguments_out_of_range(self, executor):
            """Test out-of-range indexed arguments return empty string."""
            content = "Has: ${0}, Missing: ${1}"
            result = executor.substitute_arguments(content, "only-one")

            assert result == "Has: only-one, Missing: "

        def test_substitute_arguments_quoted(self, executor):
            """Test arguments with quotes are handled correctly."""
            content = "Process: ${0}"
            result = executor.substitute_arguments(content, '"multi word arg"')

            assert result == "Process: multi word arg"

        def test_substitute_arguments_mixed(self, executor):
            """Test mixed argument styles."""
            content = "All: $ARGUMENTS, First: ${0}, Second: $ARGUMENTS[1]"
            result = executor.substitute_arguments(content, "one two three")

            assert result == "All: one two three, First: one, Second: two"

        def test_substitute_arguments_no_placeholders(self, executor):
            """Test content without placeholders is unchanged."""
            content = "No placeholders here."
            result = executor.substitute_arguments(content, "ignored args")

            assert result == "No placeholders here."

        def test_substitute_arguments_none_content(self, executor):
            """Test empty content returns empty string."""
            result = executor.substitute_arguments("", "args")
            assert result == ""

        def test_substitute_dollar_amount_not_replaced(self, executor):
            """Regression: bare $100, $50 etc. must NOT be treated as indexed args."""
            content = "The cost is $100 and the fee is $50."
            result = executor.substitute_arguments(content, "ignored")

            assert result == "The cost is $100 and the fee is $50."

        def test_substitute_braces_indexed(self, executor):
            """Test that ${N} brace-delimited syntax works correctly."""
            content = "Arg0=${0}, Arg1=${1}, literal $99"
            result = executor.substitute_arguments(content, "hello world")

            assert result == "Arg0=hello, Arg1=world, literal $99"

    class TestToolResolution:
        """Tests for allowed-tools resolution."""

        @pytest.fixture
        def executor(self):
            return SkillExecutor()

        def test_resolve_allowed_tools_simple(self, executor):
            """Test simple tool name resolution."""
            allowed = ["Read", "Grep", "Glob"]
            result = executor.resolve_allowed_tools(allowed)

            assert result == ["Read", "Grep", "Glob"]

        def test_resolve_allowed_tools_with_patterns(self, executor):
            """Test tool patterns like Bash(git *)."""
            allowed = ["Read", "Bash(git *)", "Bash(npm run *)"]
            result = executor.resolve_allowed_tools(allowed)

            assert "Read" in result
            assert "Bash(git *)" in result
            assert "Bash(npm run *)" in result

        def test_resolve_allowed_tools_empty(self, executor):
            """Test empty allowed tools returns empty list."""
            result = executor.resolve_allowed_tools(None)
            assert result == []

            result = executor.resolve_allowed_tools([])
            assert result == []

        def test_resolve_allowed_tools_filters_unavailable(self, executor):
            """Test that unavailable tools are filtered when available list provided."""
            allowed = ["Read", "Grep", "Write"]
            available = ["Read", "Grep"]
            result = executor.resolve_allowed_tools(allowed, available)

            assert "Read" in result
            assert "Grep" in result
            assert "Write" not in result

    class TestPatternMatching:
        """Tests for tool command pattern matching."""

        @pytest.fixture
        def executor(self):
            return SkillExecutor()

        def test_matches_simple_tool(self, executor):
            """Test simple tool name matching."""
            assert executor.matches_tool_pattern("Read", "", "Read") is True
            assert executor.matches_tool_pattern("Write", "", "Read") is False

        def test_matches_bash_git_pattern(self, executor):
            """Test Bash(git *) pattern."""
            pattern = "Bash(git *)"

            assert executor.matches_tool_pattern("Bash", "git status", pattern) is True
            assert executor.matches_tool_pattern("Bash", "git commit -m 'test'", pattern) is True
            assert executor.matches_tool_pattern("Bash", "npm install", pattern) is False
            assert executor.matches_tool_pattern("Read", "git status", pattern) is False

        def test_matches_npm_run_pattern(self, executor):
            """Test Bash(npm run *) pattern."""
            pattern = "Bash(npm run *)"

            assert executor.matches_tool_pattern("Bash", "npm run test", pattern) is True
            assert executor.matches_tool_pattern("Bash", "npm run build", pattern) is True
            assert executor.matches_tool_pattern("Bash", "npm install", pattern) is False

        def test_matches_exact_command(self, executor):
            """Test exact command matching."""
            pattern = "Bash(make build)"

            assert executor.matches_tool_pattern("Bash", "make build", pattern) is True
            assert executor.matches_tool_pattern("Bash", "make test", pattern) is False

    class TestFilterTools:
        """Tests for tool filtering."""

        @pytest.fixture
        def executor(self):
            return SkillExecutor()

        def test_filter_tools_no_restrictions(self, executor):
            """Test that all tools pass with no restrictions."""
            tools = [
                {"name": "Read"},
                {"name": "Write"},
                {"name": "Bash"},
            ]
            result = executor.filter_tools_for_skill(tools, [])

            assert len(result) == 3

        def test_filter_tools_with_restrictions(self, executor):
            """Test filtering tools based on allowed list."""
            tools = [
                {"name": "Read"},
                {"name": "Write"},
                {"name": "Bash"},
            ]
            allowed = ["Read", "Bash"]
            result = executor.filter_tools_for_skill(tools, allowed)

            names = [t["name"] for t in result]
            assert "Read" in names
            assert "Bash" in names
            assert "Write" not in names

        def test_filter_tools_with_patterns(self, executor):
            """Test that patterns still allow the base tool."""
            tools = [
                {"name": "Read"},
                {"name": "Bash"},
            ]
            allowed = ["Read", "Bash(git *)"]
            result = executor.filter_tools_for_skill(tools, allowed)

            names = [t["name"] for t in result]
            assert "Read" in names
            assert "Bash" in names


class TestSkillExecution:
    """Tests for skill execution."""

    @pytest.fixture
    def executor(self):
        return SkillExecutor()

    @pytest.mark.asyncio
    async def test_execute_inline(self, executor):
        """Test inline execution mode."""
        skill_data = {
            "name": "test-skill",
            "content": "Do something with $ARGUMENTS",
            "context": "inline",
            "allowed_tools": ["Read"],
            "model": None,
        }

        result = await executor.execute(skill_data, "my-args")

        assert result.skill_name == "test-skill"
        assert result.rendered_prompt == "Do something with my-args"
        assert result.execution_mode == "inline"
        assert "Read" in result.allowed_tools

    @pytest.mark.asyncio
    async def test_execute_fork(self, executor, monkeypatch):
        """Test fork execution mode."""
        skill_data = {
            "name": "fork-skill",
            "content": "Forked task: $ARGUMENTS",
            "context": "fork",
            "allowed_tools": ["Read", "Grep"],
            "model": "gpt-4",
        }

        async def _fake_chat_call(**_kwargs):
            return {"choices": [{"message": {"content": "fork output"}}]}

        from tldw_Server_API.app.core.Chat import chat_service as chat_service_mod
        monkeypatch.setattr(chat_service_mod, "perform_chat_api_call_async", _fake_chat_call)

        class _ToolExecutorStub:
            async def list_tools(self, *, user_id=None, client_id=None):
                return {"tools": []}

            async def execute(self, **_kwargs):
                return {"ok": True}

        ctx = RequestContext(
            user_id=1,
            default_provider="openai",
            tool_executor=_ToolExecutorStub(),
            tool_definitions=[],
        )

        result = await executor.execute(skill_data, "task-args", context=ctx)

        assert result.skill_name == "fork-skill"
        assert result.execution_mode == "fork"
        assert result.model_override is None
        assert result.rendered_prompt == "Forked task: task-args"
        assert result.fork_output == "fork output"

    @pytest.mark.asyncio
    async def test_execute_dry_run_renders_fork_without_model_or_tools(self, executor, monkeypatch):
        """Dry-rendering a fork skill must not invoke model calls or tool execution."""
        skill_data = {
            "name": "fork-dry-run-skill",
            "content": "Forked task: $ARGUMENTS",
            "context": "fork",
            "allowed_tools": ["Read", "Grep"],
            "model": "gpt-4",
        }

        async def _unexpected_chat_call(**_kwargs):
            raise AssertionError("dry_run must not call the model adapter")

        from tldw_Server_API.app.core.Chat import chat_service as chat_service_mod
        monkeypatch.setattr(chat_service_mod, "perform_chat_api_call_async", _unexpected_chat_call)

        class _UnexpectedToolExecutor:
            async def list_tools(self, **_kwargs):
                raise AssertionError("dry_run must not list tools")

            async def execute(self, **_kwargs):
                raise AssertionError("dry_run must not execute tools")

        ctx = RequestContext(
            user_id=1,
            default_provider="openai",
            available_tools=["Read", "Grep"],
            tool_executor=_UnexpectedToolExecutor(),
        )

        result = await executor.execute(
            skill_data,
            "task-args",
            context=ctx,
            dry_run=True,
        )

        assert result.skill_name == "fork-dry-run-skill"
        assert result.rendered_prompt == "Forked task: task-args"
        assert result.allowed_tools == ["Read", "Grep"]
        assert result.model_override == "gpt-4"
        assert result.execution_mode == "fork"
        assert result.fork_output is None
        assert result.dry_run is True

    @pytest.mark.asyncio
    async def test_execute_with_model_override(self, executor):
        """Test that model override is preserved."""
        skill_data = {
            "name": "model-skill",
            "content": "Content",
            "context": "inline",
            "allowed_tools": None,
            "model": "claude-3-opus",
        }

        result = await executor.execute(skill_data, "")

        assert result.model_override == "claude-3-opus"


class TestForkExceptionLogging:
    """Regression tests for fork mode exception handling (Bug 4)."""

    @pytest.fixture
    def executor(self):
        return SkillExecutor()

    @pytest.mark.asyncio
    async def test_fork_unexpected_tool_error_logged(self, executor, monkeypatch):
        """Test that unexpected exceptions in fork tool execution are logged with traceback."""
        import json
        from io import StringIO

        from loguru import logger

        skill_data = {
            "name": "fork-error-skill",
            "content": "Do something",
            "context": "fork",
            "allowed_tools": [],
            "model": None,
        }

        call_count = 0

        async def _fake_chat_call(**_kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call returns a tool call
                return {
                    "choices": [{
                        "message": {
                            "content": None,
                            "tool_calls": [{
                                "id": "call_1",
                                "function": {
                                    "name": "SomeTool",
                                    "arguments": json.dumps({"key": "value"}),
                                },
                            }],
                        }
                    }]
                }
            # Second call returns final text
            return {"choices": [{"message": {"content": "done"}}]}

        from tldw_Server_API.app.core.Chat import chat_service as chat_service_mod
        monkeypatch.setattr(chat_service_mod, "perform_chat_api_call_async", _fake_chat_call)

        class _BrokenToolExecutor:
            async def list_tools(self, **_kw):
                return {"tools": []}

            async def execute(self, **_kw):
                raise RuntimeError("Unexpected kaboom!")

        ctx = RequestContext(
            user_id=1,
            default_provider="openai",
            tool_executor=_BrokenToolExecutor(),
            tool_definitions=[],
        )

        # Capture loguru output
        log_output = StringIO()
        handler_id = logger.add(log_output, format="{message}", level="WARNING")
        try:
            result = await executor.execute(skill_data, "", context=ctx)
        finally:
            logger.remove(handler_id)

        # Fork should complete (error is returned to LLM, not raised)
        assert result.execution_mode == "fork"
        assert result.fork_output == "done"
        # The warning should have been logged
        captured = log_output.getvalue()
        assert "kaboom" in captured or "Unexpected" in captured


class TestSkillToolDefinition:
    """Tests for the Skill tool definition."""

    def test_skill_tool_definition_structure(self):
        """Test that SKILL_TOOL_DEFINITION has correct structure."""
        assert "type" in SKILL_TOOL_DEFINITION
        assert SKILL_TOOL_DEFINITION["type"] == "function"

        func = SKILL_TOOL_DEFINITION["function"]
        assert func["name"] == "Skill"
        assert "description" in func
        assert "parameters" in func

        params = func["parameters"]
        assert params["type"] == "object"
        assert "skill" in params["properties"]
        assert "args" in params["properties"]
        assert "skill" in params["required"]
