from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from tldw_Server_API.app.core.Chat_Macros.branch_runner import BranchPromptResult
from tldw_Server_API.app.core.Chat_Macros.context_snapshot import (
    MacroContextSnapshot,
    build_macro_context_snapshot,
    snapshot_from_mapping,
)
from tldw_Server_API.app.core.Chat_Macros.exceptions import MacroValidationError
from tldw_Server_API.app.core.Chat_Macros.executor import ChatMacroExecutor, MacroExecutorSettings
from tldw_Server_API.app.core.Chat_Macros.models import MacroDefinition
from tldw_Server_API.app.core.Chat_Macros.output_profiles import MacroOutputProfile
from tldw_Server_API.app.core.Chat_Macros.parser import load_macro_definition, parse_macro_args
from tldw_Server_API.app.core.Chat_Macros.repository import ChatMacroRepository
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

pytestmark = pytest.mark.unit


BUILTIN_WRAPUP_PATH = (
    Path(__file__).parents[3]
    / "app"
    / "core"
    / "Chat_Macros"
    / "builtin"
    / "wrapup"
    / "MACRO.yaml"
)


class RecordingBranchRunner:
    def __init__(
        self,
        *,
        fail_first_for: tuple[str, ...] = (),
        always_fail_for: tuple[str, ...] = (),
        delay_seconds: float = 0.0,
    ) -> None:
        self.fail_first_for = fail_first_for
        self.always_fail_for = always_fail_for
        self.delay_seconds = delay_seconds
        self.prompts: list[str] = []
        self.attempts: dict[str, int] = {}
        self.active = 0
        self.max_active = 0

    async def run_branch(
        self,
        *,
        prompt: str,
        snapshot: MacroContextSnapshot,
        model_selection: dict[str, Any],
    ) -> BranchPromptResult:
        self.prompts.append(prompt)
        self.attempts[prompt] = self.attempts.get(prompt, 0) + 1
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            if self.delay_seconds:
                await asyncio.sleep(self.delay_seconds)
            if any(marker in prompt for marker in self.always_fail_for):
                raise RuntimeError(f"failed: {prompt}")
            if any(marker in prompt for marker in self.fail_first_for) and self.attempts[prompt] == 1:
                raise RuntimeError(f"transient: {prompt}")
            return BranchPromptResult(
                text=f"output for {prompt}",
                citations=[{"source_id": "s1"}],
                usage={"prompt_tokens": 3, "completion_tokens": 5},
            )
        finally:
            self.active -= 1


class CancelOnFirstBranchRunner:
    def __init__(self, repo: ChatMacroRepository, run_id: str) -> None:
        self.repo = repo
        self.run_id = run_id
        self.prompts: list[str] = []

    async def run_branch(
        self,
        *,
        prompt: str,
        snapshot: MacroContextSnapshot,
        model_selection: dict[str, Any],
    ) -> BranchPromptResult:
        self.prompts.append(prompt)
        if len(self.prompts) == 1:
            self.repo.request_cancel(self.run_id)
        await asyncio.sleep(0.01)
        return BranchPromptResult(text=f"output for {prompt}")


class SecretFailingBranchRunner:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def run_branch(
        self,
        *,
        prompt: str,
        snapshot: MacroContextSnapshot,
        model_selection: dict[str, Any],
    ) -> BranchPromptResult:
        self.prompts.append(prompt)
        raise RuntimeError("provider failed with api_key=sk-live-secret token=super-secret")


class CancelAndFailFirstAttemptRunner:
    def __init__(self, repo: ChatMacroRepository, run_id: str) -> None:
        self.repo = repo
        self.run_id = run_id
        self.attempts = 0

    async def run_branch(
        self,
        *,
        prompt: str,
        snapshot: MacroContextSnapshot,
        model_selection: dict[str, Any],
    ) -> BranchPromptResult:
        self.attempts += 1
        if self.attempts == 1:
            self.repo.request_cancel(self.run_id)
            raise RuntimeError("cancel after failed provider attempt")
        return BranchPromptResult(text=f"unexpected retry output for {prompt}")


@pytest.fixture()
def raw_db(tmp_path):
    db = CharactersRAGDB(str(tmp_path / "macros.db"), client_id="macro_executor_test")
    try:
        yield db
    finally:
        db.close_connection()


@pytest.fixture()
def repo(raw_db):
    return ChatMacroRepository(raw_db)


@pytest.fixture()
def wrapup_definition() -> MacroDefinition:
    return load_macro_definition(BUILTIN_WRAPUP_PATH.read_text())


def _create_run(
    repo: ChatMacroRepository,
    definition: MacroDefinition,
    *,
    normalized_args: dict[str, Any] | None = None,
    context_snapshot: dict[str, Any] | None = None,
    model_selection: dict[str, Any] | None = None,
    output_profile: str = "default",
):
    return repo.create_run(
        user_id="1",
        macro_name=definition.name,
        macro_command=definition.command,
        macro_source="builtin",
        macro_version=definition.builtin_version,
        macro_digest="test-digest",
        normalized_args=normalized_args or {},
        status="pending",
        surface="chat",
        conversation_id="conv-1",
        workspace_id="workspace-1",
        acp_session_id=(context_snapshot or {}).get("acp_session_id"),
        output_profile=output_profile,
        context_snapshot=context_snapshot or _snapshot_dict(),
        model_selection=model_selection or {"api_provider": "openai", "model": "gpt-test"},
    )


def _snapshot_dict(**overrides: Any) -> dict[str, Any]:
    data = {
        "conversation_id": "conv-1",
        "workspace_id": "workspace-1",
        "acp_session_id": None,
        "messages": [{"id": "m1", "role": "user", "excerpt": "Please wrap this up."}],
        "selected_message_ids": ["m1"],
        "selected_source_ids": {"rag": ["rag-1"], "media": ["media-1"]},
        "model_selection": {"api_provider": "openai", "model": "gpt-test"},
        "output_profile": "default",
        "token_estimate": 120,
        "acp": {},
    }
    data.update(overrides)
    return data


def _executor(
    repo: ChatMacroRepository,
    definition: MacroDefinition,
    runner: RecordingBranchRunner,
    **kwargs: Any,
) -> ChatMacroExecutor:
    return ChatMacroExecutor(
        repository=repo,
        macro_loader=lambda _run: definition,
        branch_runner=runner,
        **kwargs,
    )


def test_context_snapshot_bounds_messages_and_redacts_secrets():
    snapshot = build_macro_context_snapshot(
        chat_db=None,
        conversation_id="conv-1",
        workspace_id="workspace-1",
        acp_session_id="acp-1",
        request_messages=[
            {"id": "m1", "role": "user", "content": f"{'A' * 40} api_key=sk-live-secret"},
            {"id": "m2", "role": "assistant", "content": [{"type": "text", "text": "B" * 80}]},
        ],
        model_selection={"api_provider": "openai", "model": "gpt-test", "api_key": "sk-secret"},
        output_profile="compact",
        request_metadata={
            "selected_message_ids": ["m1"],
            "selected_rag_ids": ["rag-1"],
            "selected_media_ids": ["media-1"],
            "token": "secret-token",
            "nested": {"password": "secret-password", "safe": "kept"},
        },
        max_excerpt_chars=24,
    )
    payload = snapshot.model_dump(mode="json")

    assert payload["conversation_id"] == "conv-1"
    assert payload["workspace_id"] == "workspace-1"
    assert payload["acp_session_id"] == "acp-1"
    assert payload["selected_message_ids"] == ["m1"]
    assert payload["selected_source_ids"] == {"rag": ["rag-1"], "media": ["media-1"]}
    assert payload["model_selection"] == {"api_provider": "openai", "model": "gpt-test"}
    assert payload["output_profile"] == "compact"
    assert len(payload["messages"][0]["excerpt"]) <= 24
    assert "sk-live-secret" not in str(payload)
    assert "secret-token" not in str(payload)
    assert "secret-password" not in str(payload)


def test_snapshot_from_mapping_redacts_stored_message_excerpts():
    snapshot = snapshot_from_mapping(
        {
            "conversation_id": "conv-1",
            "messages": [
                {"id": "m1", "role": "user", "excerpt": "Bearer abc123 and password=hunter2"}
            ],
            "selected_message_ids": ["m1"],
            "selected_source_ids": {},
            "model_selection": {"model": "gpt-test"},
            "output_profile": "default",
            "token_estimate": 1,
        }
    )
    payload = snapshot.model_dump(mode="json")

    assert "abc123" not in str(payload)
    assert "hunter2" not in str(payload)
    assert "[redacted]" in str(payload)


def test_snapshot_from_mapping_redacts_selected_id_values():
    snapshot = snapshot_from_mapping(
        {
            "conversation_id": "conv-1",
            "messages": [],
            "selected_message_ids": ["m1", "Bearer abc123"],
            "selected_source_ids": {
                "rag": ["rag-1", "api_key=sk-live-secret"],
                "media": ["token=stored-secret"],
            },
            "model_selection": {"model": "gpt-test"},
            "output_profile": "default",
            "token_estimate": 1,
        }
    )
    payload = snapshot.model_dump(mode="json")

    assert "abc123" not in str(payload)
    assert "sk-live-secret" not in str(payload)
    assert "stored-secret" not in str(payload)
    assert "[redacted]" in str(payload)


def test_builtin_wrapup_supports_include_branches_but_rejects_unimplemented_sync(
    wrapup_definition,
):
    args = parse_macro_args("--include-branches", wrapup_definition.args)

    assert args["include_branches"] is True
    with pytest.raises(MacroValidationError, match="unknown macro argument"):
        parse_macro_args("--sync", wrapup_definition.args)


@pytest.mark.asyncio
async def test_executor_fails_early_when_model_unavailable_and_does_not_start_branches(repo, wrapup_definition):
    runner = RecordingBranchRunner()
    run = _create_run(repo, wrapup_definition)
    executor = _executor(
        repo,
        wrapup_definition,
        runner,
        model_available=lambda _selection: False,
    )

    saved = await executor.execute_run(run.run_id)

    assert saved.status == "failed"
    assert saved.error_code == "model_unavailable"
    assert runner.prompts == []


@pytest.mark.asyncio
async def test_executor_fails_early_when_token_cap_is_exceeded(repo, wrapup_definition):
    runner = RecordingBranchRunner()
    run = _create_run(repo, wrapup_definition, context_snapshot=_snapshot_dict(token_estimate=10_000))
    executor = _executor(
        repo,
        wrapup_definition,
        runner,
        settings=MacroExecutorSettings(max_total_estimated_tokens=100),
    )

    saved = await executor.execute_run(run.run_id)

    assert saved.status == "failed"
    assert saved.error_code == "token_cap_exceeded"
    assert runner.prompts == []


@pytest.mark.asyncio
async def test_token_cap_counts_snapshot_context_for_each_branch(repo, wrapup_definition):
    runner = RecordingBranchRunner()
    run = _create_run(
        repo,
        wrapup_definition,
        context_snapshot=_snapshot_dict(token_estimate=30),
    )
    executor = _executor(
        repo,
        wrapup_definition,
        runner,
        settings=MacroExecutorSettings(max_total_estimated_tokens=100),
    )

    saved = await executor.execute_run(run.run_id)

    assert saved.status == "failed"
    assert saved.error_code == "token_cap_exceeded"
    assert runner.prompts == []


@pytest.mark.asyncio
async def test_executor_fails_early_when_branch_cap_is_exceeded(repo, wrapup_definition):
    runner = RecordingBranchRunner()
    run = _create_run(
        repo,
        wrapup_definition,
        normalized_args={"question": ["extra 1", "extra 2", "extra 3"]},
    )
    executor = _executor(
        repo,
        wrapup_definition,
        runner,
        settings=MacroExecutorSettings(max_branches=4),
    )

    saved = await executor.execute_run(run.run_id)

    assert saved.status == "failed"
    assert saved.error_code == "branch_limit_exceeded"
    assert runner.prompts == []


@pytest.mark.asyncio
async def test_executor_does_not_start_branches_for_pre_cancelled_run(repo, wrapup_definition):
    runner = RecordingBranchRunner()
    run = _create_run(repo, wrapup_definition)
    repo.request_cancel(run.run_id)
    executor = _executor(repo, wrapup_definition, runner)

    saved = await executor.execute_run(run.run_id)

    assert saved.status == "cancelled"
    assert runner.prompts == []


@pytest.mark.asyncio
async def test_executor_does_not_restart_terminal_run(repo, wrapup_definition):
    runner = RecordingBranchRunner()
    run = _create_run(repo, wrapup_definition)
    repo.update_run_status(run.run_id, status="completed")
    executor = _executor(repo, wrapup_definition, runner)

    saved = await executor.execute_run(run.run_id)

    assert saved.status == "completed"
    assert runner.prompts == []
    assert repo.list_branches(run.run_id) == []


@pytest.mark.asyncio
async def test_terminal_run_retry_skips_loader_and_snapshot_planning(repo, wrapup_definition):
    runner = RecordingBranchRunner()
    run = _create_run(repo, wrapup_definition)
    repo.update_run_status(run.run_id, status="completed")

    def fail_loader(_run):
        raise AssertionError("terminal retries must not load macro definitions")

    executor = ChatMacroExecutor(
        repository=repo,
        macro_loader=fail_loader,
        branch_runner=runner,
    )

    saved = await executor.execute_run(run.run_id)

    assert saved.status == "completed"
    assert runner.prompts == []


@pytest.mark.asyncio
async def test_completed_unposted_run_retries_post_back_without_replanning(repo, wrapup_definition):
    posted: list[str] = []
    post_keys: list[str] = []
    runner = RecordingBranchRunner()
    run = _create_run(repo, wrapup_definition)
    repo.store_final_output(
        run.run_id,
        final_output="stored final output",
        final_output_format="markdown",
    )
    repo.update_run_status(run.run_id, status="completed")

    def fail_loader(_run):
        raise AssertionError("post-only retries must not load macro definitions")

    async def post_back(*, run_id: str, final_output: str, post_idempotency_key: str) -> str:
        posted.append(final_output)
        post_keys.append(post_idempotency_key)
        return "msg-final"

    executor = ChatMacroExecutor(
        repository=repo,
        macro_loader=fail_loader,
        branch_runner=runner,
        post_back=post_back,
    )

    saved = await executor.execute_run(run.run_id)

    assert posted == ["stored final output"]
    assert post_keys == [f"chat_macro:{run.run_id}:final"]
    assert saved.status == "completed"
    assert saved.final_message_id == "msg-final"
    assert runner.prompts == []


@pytest.mark.asyncio
async def test_wrapup_dev_handoff_preset_selects_handoff_branches(repo, wrapup_definition):
    runner = RecordingBranchRunner()
    run = _create_run(repo, wrapup_definition, normalized_args={"preset": "dev_handoff"})
    executor = _executor(repo, wrapup_definition, runner)

    await executor.execute_run(run.run_id)

    step_ids = {branch.step_id for branch in repo.list_branches(run.run_id)}
    assert step_ids == {"summary", "changes", "verification", "risks", "next_steps"}


@pytest.mark.asyncio
async def test_repeated_custom_questions_append_custom_branches(repo, wrapup_definition):
    runner = RecordingBranchRunner()
    run = _create_run(
        repo,
        wrapup_definition,
        normalized_args={"question": ["What changed?", "What is next?"]},
    )
    executor = _executor(repo, wrapup_definition, runner)

    await executor.execute_run(run.run_id)

    branches = repo.list_branches(run.run_id)
    custom_branches = {branch.step_id: branch for branch in branches if branch.step_id.startswith("custom_")}
    assert set(custom_branches) == {"custom_1", "custom_2"}
    assert custom_branches["custom_1"].label == "Custom 1"
    assert custom_branches["custom_2"].label == "Custom 2"
    assert any("What changed?" in prompt for prompt in runner.prompts)


@pytest.mark.asyncio
async def test_include_branches_adds_appendix_only_when_profile_allows(repo, wrapup_definition):
    runner = RecordingBranchRunner()
    settings = MacroExecutorSettings(
        output_profiles={
            "default": MacroOutputProfile(name="default", include_branch_outputs=False),
            "verbose": MacroOutputProfile(name="verbose", include_branch_outputs=True),
        },
    )
    default_run = _create_run(
        repo,
        wrapup_definition,
        normalized_args={"include_branches": True},
        output_profile="default",
    )
    verbose_run = _create_run(
        repo,
        wrapup_definition,
        normalized_args={"include_branches": True},
        output_profile="verbose",
    )
    executor = _executor(repo, wrapup_definition, runner, settings=settings)

    default_saved = await executor.execute_run(default_run.run_id)
    verbose_saved = await executor.execute_run(verbose_run.run_id)

    assert "## Branch Outputs" not in default_saved.final_output
    assert "## Branch Outputs" in verbose_saved.final_output


@pytest.mark.asyncio
async def test_sync_mode_runs_only_under_configured_thresholds(repo, wrapup_definition):
    runner = RecordingBranchRunner()
    settings = MacroExecutorSettings(sync_max_branches=4, sync_max_estimated_tokens=2_000)
    sync_run = _create_run(
        repo,
        wrapup_definition,
        normalized_args={"sync": True},
        context_snapshot=_snapshot_dict(token_estimate=200),
    )
    large_run = _create_run(
        repo,
        wrapup_definition,
        normalized_args={"sync": True},
        context_snapshot=_snapshot_dict(token_estimate=900),
    )
    executor = _executor(repo, wrapup_definition, runner, settings=settings)

    saved_sync = await executor.execute_run(sync_run.run_id)
    saved_large = await executor.execute_run(large_run.run_id)

    assert saved_sync.status == "completed"
    assert saved_large.status == "failed"
    assert saved_large.error_code == "sync_limit_exceeded"


@pytest.mark.asyncio
async def test_branches_run_up_to_max_concurrency(repo, wrapup_definition):
    runner = RecordingBranchRunner(delay_seconds=0.01)
    run = _create_run(repo, wrapup_definition)
    executor = _executor(
        repo,
        wrapup_definition,
        runner,
        settings=MacroExecutorSettings(max_concurrency=2),
    )

    await executor.execute_run(run.run_id)

    assert runner.max_active == 2


@pytest.mark.asyncio
async def test_cancel_requested_while_queued_prevents_later_branches(repo, wrapup_definition):
    run = _create_run(repo, wrapup_definition)
    runner = CancelOnFirstBranchRunner(repo, run.run_id)
    executor = _executor(
        repo,
        wrapup_definition,
        runner,
        settings=MacroExecutorSettings(max_concurrency=1),
    )

    saved = await executor.execute_run(run.run_id)
    branches = repo.list_branches(run.run_id)

    assert saved.status == "cancelled"
    assert len(runner.prompts) == 1
    assert any(branch.status == "completed" for branch in branches)
    assert any(branch.status == "cancelled" for branch in branches)


@pytest.mark.asyncio
async def test_failed_branch_retries_once_and_partial_failure_is_rendered(repo, wrapup_definition):
    runner = RecordingBranchRunner(
        fail_first_for=("Extract decisions",),
        always_fail_for=("Extract action items",),
    )
    run = _create_run(repo, wrapup_definition)
    executor = _executor(repo, wrapup_definition, runner)

    saved = await executor.execute_run(run.run_id)
    branches = {branch.step_id: branch for branch in repo.list_branches(run.run_id)}

    decisions_prompt = next(prompt for prompt in runner.prompts if "Extract decisions" in prompt)
    action_prompt = next(prompt for prompt in runner.prompts if "Extract action items" in prompt)
    assert runner.attempts[decisions_prompt] == 2
    assert runner.attempts[action_prompt] == 2
    assert branches["decisions"].status == "completed"
    assert branches["action_items"].status == "failed"
    assert "## Failed Branches" in saved.final_output
    assert "Action items" in saved.final_output


@pytest.mark.asyncio
async def test_cancel_after_failed_branch_attempt_prevents_retry(repo, wrapup_definition):
    run = _create_run(repo, wrapup_definition)
    runner = CancelAndFailFirstAttemptRunner(repo, run.run_id)
    executor = ChatMacroExecutor(
        repository=repo,
        macro_loader=lambda _run: wrapup_definition,
        branch_runner=runner,
        settings=MacroExecutorSettings(max_concurrency=1),
    )

    saved = await executor.execute_run(run.run_id)

    assert runner.attempts == 1
    assert saved.status == "cancelled"


@pytest.mark.asyncio
async def test_all_branches_failed_stores_failure_report(repo, wrapup_definition):
    runner = RecordingBranchRunner(always_fail_for=("Summarize", "Extract", "Identify"))
    run = _create_run(repo, wrapup_definition)
    executor = _executor(repo, wrapup_definition, runner)

    saved = await executor.execute_run(run.run_id)

    assert saved.status == "failed"
    assert "All branch prompts failed" in saved.final_output
    assert "## Summary" not in saved.final_output


@pytest.mark.asyncio
async def test_branch_timeout_is_enforced_and_recorded(
    repo: ChatMacroRepository,
    wrapup_definition: MacroDefinition,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timeouts: list[float | None] = []

    async def fake_wait_for(awaitable: Any, timeout: float | None) -> Any:
        timeouts.append(timeout)
        awaitable.close()
        raise TimeoutError

    monkeypatch.setattr(asyncio, "wait_for", fake_wait_for)
    wrapup_definition.execution.timeout_seconds = 1
    run = _create_run(repo, wrapup_definition)
    executor = _executor(repo, wrapup_definition, RecordingBranchRunner())

    saved = await executor.execute_run(run.run_id)
    branches = repo.list_branches(run.run_id)

    assert saved.status == "failed"
    assert timeouts and set(timeouts) == {1}
    assert all(branch.error_code == "timeout" for branch in branches)


@pytest.mark.asyncio
async def test_branch_failure_errors_are_redacted_before_persistence(repo, wrapup_definition):
    runner = SecretFailingBranchRunner()
    run = _create_run(repo, wrapup_definition)
    executor = _executor(repo, wrapup_definition, runner)

    saved = await executor.execute_run(run.run_id)
    branches = repo.list_branches(run.run_id)
    persisted = str([branch.error_message for branch in branches]) + str(saved.final_output)

    assert saved.status == "failed"
    assert "sk-live-secret" not in persisted
    assert "super-secret" not in persisted
    assert "[redacted]" in persisted


@pytest.mark.asyncio
async def test_merge_failure_preserves_successful_branch_outputs_in_run_detail(repo, wrapup_definition):
    async def fail_merge(**_kwargs: Any) -> str:
        raise RuntimeError("merge unavailable")

    runner = RecordingBranchRunner()
    run = _create_run(repo, wrapup_definition)
    executor = _executor(repo, wrapup_definition, runner, merge_runner=fail_merge)

    saved = await executor.execute_run(run.run_id)
    branches = repo.list_branches(run.run_id)

    assert saved.status == "failed"
    assert "Merge failed" in saved.final_output
    assert all(branch.output_text for branch in branches)


@pytest.mark.asyncio
async def test_cancel_requested_during_merge_failure_wins_over_merge_failed(repo, wrapup_definition):
    runner = RecordingBranchRunner()
    run = _create_run(repo, wrapup_definition)

    async def cancel_then_fail_merge(**_kwargs: Any) -> str:
        repo.request_cancel(run.run_id)
        raise RuntimeError("merge unavailable")

    executor = _executor(repo, wrapup_definition, runner, merge_runner=cancel_then_fail_merge)

    saved = await executor.execute_run(run.run_id)

    assert saved.status == "cancelled"
    assert saved.error_code == "cancelled"
    assert saved.final_output == "Macro run cancelled."


@pytest.mark.asyncio
async def test_final_output_is_stored_before_post_back(repo, wrapup_definition):
    observed = {}

    async def post_back(*, run_id: str, final_output: str, post_idempotency_key: str) -> str:
        assert post_idempotency_key == f"chat_macro:{run_id}:final"
        observed["stored_before_post"] = repo.get_run(run_id).final_output == final_output
        return "msg-final"

    runner = RecordingBranchRunner()
    run = _create_run(repo, wrapup_definition)
    executor = _executor(repo, wrapup_definition, runner, post_back=post_back)

    saved = await executor.execute_run(run.run_id)

    assert observed["stored_before_post"] is True
    assert saved.final_message_id == "msg-final"


@pytest.mark.asyncio
async def test_cancel_after_final_output_storage_suppresses_post_back(repo, wrapup_definition, monkeypatch):
    posted: list[str] = []
    run = _create_run(repo, wrapup_definition)
    original_store_final_output = repo.store_final_output

    def cancel_after_store(*args: Any, **kwargs: Any):
        saved = original_store_final_output(*args, **kwargs)
        repo.request_cancel(run.run_id)
        return saved

    async def post_back(*, run_id: str, final_output: str, post_idempotency_key: str) -> str:
        assert post_idempotency_key == f"chat_macro:{run_id}:final"
        posted.append(final_output)
        return "msg-final"

    monkeypatch.setattr(repo, "store_final_output", cancel_after_store)
    runner = RecordingBranchRunner()
    executor = _executor(repo, wrapup_definition, runner, post_back=post_back)

    saved = await executor.execute_run(run.run_id)
    persisted = repo.get_run(run.run_id)

    assert saved.status == "cancelled"
    assert posted == []
    assert persisted.final_message_id is None


@pytest.mark.asyncio
async def test_required_acp_branch_is_failed_when_acp_is_unavailable(repo):
    raw = """
schema_version: 1
name: acp_required
command: acp_required
execution:
  branch_strategy: auto
steps:
  - id: forked
    type: branch_prompt
    label: Forked
    output: forked
    branch_strategy: acp_fork
    prompt: Must fork.
  - id: merge
    type: merge
    consumes: [forked]
    output: final
"""
    definition = load_macro_definition(raw)
    runner = RecordingBranchRunner()
    run = _create_run(repo, definition, context_snapshot=_snapshot_dict(acp_session_id=None, acp={}))
    executor = _executor(repo, definition, runner)

    saved = await executor.execute_run(run.run_id)
    branch = repo.list_branches(run.run_id)[0]

    assert runner.prompts == []
    assert branch.status == "failed"
    assert branch.error_code == "acp_unavailable"
    assert branch.usage["branch_strategy"]["required_failed"] is True
    assert saved.status == "failed"
