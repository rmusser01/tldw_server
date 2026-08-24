"""Chat macro execution orchestration."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from .acp_adapter import resolve_acp_branch_capability, select_branch_strategy
from .branch_runner import BranchPromptRunner
from .context_snapshot import MacroContextSnapshot, redact_sensitive_text, snapshot_from_mapping
from .exceptions import MacroExecutionError, MacroStorageError
from .models import MacroBranchRecord, MacroDefinition, MacroRunRecord, MacroStep
from .output_profiles import (
    DEFAULT_OUTPUT_PROFILE,
    MacroOutputProfile,
    render_output_profile,
)
from .repository import ChatMacroRepository

MacroLoader = Callable[[MacroRunRecord], MacroDefinition]
ModelAvailable = Callable[[dict[str, Any]], bool]
MergeRunner = Callable[..., str | Awaitable[str]]
PostBack = Callable[..., str | Awaitable[str]]

_TERMINAL_RUN_STATUSES = {"completed", "failed", "cancelled"}


@dataclass(slots=True)
class MacroExecutorSettings:
    """Runtime caps for one macro executor instance."""

    max_branches: int = 6
    max_concurrency: int = 3
    max_total_estimated_tokens: int = 12_000
    sync_max_branches: int = 3
    sync_max_estimated_tokens: int = 4_000
    retain_scratch_branches: bool = False
    output_profiles: dict[str, MacroOutputProfile] = field(
        default_factory=lambda: {"default": DEFAULT_OUTPUT_PROFILE}
    )


class ChatMacroExecutor:
    """Execute a stored macro run using fakeable branch, merge, and post seams."""

    def __init__(
        self,
        *,
        repository: ChatMacroRepository,
        macro_loader: MacroLoader,
        branch_runner: BranchPromptRunner,
        settings: MacroExecutorSettings | None = None,
        model_available: ModelAvailable | None = None,
        merge_runner: MergeRunner | None = None,
        post_back: PostBack | None = None,
    ) -> None:
        self.repository = repository
        self.macro_loader = macro_loader
        self.branch_runner = branch_runner
        self.settings = settings or MacroExecutorSettings()
        self.model_available = model_available or _default_model_available
        self.merge_runner = merge_runner
        self.post_back = post_back

    async def execute_run(self, run_id: str) -> MacroRunRecord:
        """Execute one pending macro run and return the saved run record."""
        run = await self._repo_call("get_run", run_id)
        if run is None:
            raise MacroStorageError(f"macro run not found: {run_id}")
        if run.status == "completed":
            return await self._post_final_output_if_needed(run)
        if run.status in _TERMINAL_RUN_STATUSES:
            return run
        if run.status == "cancel_requested" or run.cancel_requested_at:
            return await self._repo_call(
                "update_run_status",
                run_id,
                status="cancelled",
                error_code="cancelled",
                error_message="Macro run was cancelled before execution started.",
            )

        macro = await asyncio.to_thread(self.macro_loader, run)
        snapshot = snapshot_from_mapping(run.context_snapshot)
        model_selection = dict(run.model_selection or snapshot.model_selection or {})
        try:
            planned_steps = _planned_branch_steps(macro, run.normalized_args, self.settings)
        except MacroExecutionError as exc:
            return await self._repo_call(
                "update_run_status",
                run_id,
                status="failed",
                error_code="branch_limit_exceeded",
                error_message=str(exc),
            )
        if not planned_steps:
            return await self._repo_call(
                "update_run_status",
                run_id,
                status="failed",
                error_code="no_branches_planned",
                error_message="Macro run does not contain any branch prompts.",
            )
        profile = self._resolve_output_profile(run, macro)
        estimated_tokens = _estimated_total_tokens(snapshot, planned_steps)

        if not self.model_available(model_selection):
            return await self._repo_call(
                "update_run_status",
                run_id,
                status="failed",
                error_code="model_unavailable",
                error_message="No usable model is available for this macro run.",
            )
        if estimated_tokens > self.settings.max_total_estimated_tokens:
            return await self._repo_call(
                "update_run_status",
                run_id,
                status="failed",
                error_code="token_cap_exceeded",
                error_message="Macro run exceeds the configured token cap.",
            )
        if _sync_requested(run.normalized_args) and (
            len(planned_steps) > self.settings.sync_max_branches
            or estimated_tokens > self.settings.sync_max_estimated_tokens
        ):
            return await self._repo_call(
                "update_run_status",
                run_id,
                status="failed",
                error_code="sync_limit_exceeded",
                error_message="Macro run is too large for synchronous execution.",
            )

        started = await self._repo_call("update_run_status", run_id, status="running")
        if started.status == "cancel_requested":
            return await self._repo_call(
                "update_run_status",
                run_id,
                status="cancelled",
                error_code="cancelled",
                error_message="Macro run was cancelled before execution started.",
            )
        if started.status != "running":
            return started
        try:
            branches = await self._run_branches(
                run_id=run_id,
                macro=macro,
                steps=planned_steps,
                snapshot=snapshot,
                model_selection=model_selection,
                normalized_args=run.normalized_args,
            )
        except Exception:  # noqa: BLE001 - persistence failures must terminally fail the run
            return await self._repo_call(
                "update_run_status",
                run_id,
                status="failed",
                error_code="branch_persistence_failed",
                error_message="A branch result could not be persisted.",
            )

        completed = [branch for branch in branches if branch.status == "completed"]
        failed = [branch for branch in branches if branch.status != "completed"]
        latest = await self._repo_call("get_run", run_id)
        if (
            latest is not None
            and (latest.status == "cancel_requested" or latest.cancel_requested_at)
        ) or (branches and all(branch.status == "cancelled" for branch in branches)):
            await self._repo_call(
                "store_final_output",
                run_id,
                final_output="Macro run cancelled.",
                final_output_format="markdown",
            )
            return await self._repo_call(
                "update_run_status",
                run_id,
                status="cancelled",
                error_code="cancelled",
                error_message="Macro run was cancelled.",
            )
        if not completed:
            final_output = _all_failed_report(failed)
            await self._repo_call(
                "store_final_output",
                run_id,
                final_output=final_output,
                final_output_format="markdown",
            )
            return await self._repo_call(
                "update_run_status",
                run_id,
                status="failed",
                error_code="all_branches_failed",
                error_message="All branch prompts failed.",
            )

        outputs = {branch.step_id: branch.output_text or "" for branch in completed}
        try:
            final_output = await self._merge_outputs(
                profile=profile,
                outputs=outputs,
                failed_branches=failed,
                branch_outputs=completed if _include_branch_outputs(run.normalized_args, profile) else [],
            )
        except Exception as exc:  # noqa: BLE001 - merge failure should preserve branch detail
            if await self._run_cancel_requested(run_id):
                await self._repo_call(
                    "store_final_output",
                    run_id,
                    final_output="Macro run cancelled.",
                    final_output_format="markdown",
                )
                return await self._repo_call(
                    "update_run_status",
                    run_id,
                    status="cancelled",
                    error_code="cancelled",
                    error_message="Macro run was cancelled during merge.",
                )
            final_output = _merge_failed_report(exc, completed)
            await self._repo_call(
                "store_final_output",
                run_id,
                final_output=final_output,
                final_output_format="markdown",
            )
            return await self._repo_call(
                "update_run_status",
                run_id,
                status="failed",
                error_code="merge_failed",
                error_message="Macro merge failed.",
            )

        await self._repo_call(
            "store_final_output",
            run_id,
            final_output=final_output,
            final_output_format="markdown",
        )
        cancelled = await self._cancelled_run_if_requested(
            run_id,
            message="Macro run was cancelled before final post.",
        )
        if cancelled is not None:
            return cancelled

        completed_run = await self._repo_call("update_run_status", run_id, status="completed")
        if completed_run.status == "cancel_requested" or completed_run.cancel_requested_at:
            return await self._repo_call(
                "update_run_status",
                run_id,
                status="cancelled",
                error_code="cancelled",
                error_message="Macro run was cancelled before final post.",
            )
        if completed_run.status != "completed":
            return completed_run

        return await self._post_final_output_if_needed(completed_run)

    async def _run_branches(
        self,
        *,
        run_id: str,
        macro: MacroDefinition,
        steps: list[MacroStep],
        snapshot: MacroContextSnapshot,
        model_selection: dict[str, Any],
        normalized_args: dict[str, Any],
    ) -> list[MacroBranchRecord]:
        capability = resolve_acp_branch_capability(snapshot)
        semaphore = asyncio.Semaphore(max(1, min(self.settings.max_concurrency, macro.execution.max_concurrency)))

        async def run_step(step: MacroStep) -> MacroBranchRecord:
            cancelled_branch = await self._cancelled_branch_if_requested(run_id, step)
            if cancelled_branch is not None:
                return cancelled_branch

            decision = select_branch_strategy(
                step_strategy=step.branch_strategy,
                macro_strategy=macro.execution.branch_strategy,
                capability=capability,
            )
            strategy_metadata = decision.model_dump(mode="json")
            if decision.required_failed:
                return await self._repo_call(
                    "upsert_branch",
                    run_id,
                    step_id=step.id,
                    label=step.label,
                    status="failed",
                    attempt_count=0,
                    prompt_digest=_digest(step.prompt or ""),
                    usage={"branch_strategy": strategy_metadata},
                    retained=_retain_branch(normalized_args, macro, self.settings),
                    error_code=decision.error_code,
                    error_message=str(decision.metadata.get("reason") or decision.error_code or "failed"),
                )

            async with semaphore:
                cancelled_branch = await self._cancelled_branch_if_requested(run_id, step)
                if cancelled_branch is not None:
                    return cancelled_branch
                return await self._run_branch_with_retries(
                    run_id=run_id,
                    step=step,
                    snapshot=snapshot,
                    model_selection=model_selection,
                    normalized_args=normalized_args,
                    macro=macro,
                    strategy_metadata=strategy_metadata,
                )

        results = await asyncio.gather(*(run_step(step) for step in steps), return_exceptions=True)
        failures = [result for result in results if isinstance(result, BaseException)]
        if failures:
            raise MacroStorageError("one or more branch results could not be persisted") from failures[0]
        return [result for result in results if isinstance(result, MacroBranchRecord)]

    async def _cancelled_branch_if_requested(
        self,
        run_id: str,
        step: MacroStep,
    ) -> MacroBranchRecord | None:
        current = await self._repo_call("get_run", run_id)
        if current is None or (current.status != "cancel_requested" and not current.cancel_requested_at):
            return None
        return await self._repo_call(
            "upsert_branch",
            run_id,
            step_id=step.id,
            label=step.label,
            status="cancelled",
            error_code="cancelled",
            error_message="Macro run was cancelled before this branch started.",
        )

    async def _cancelled_run_if_requested(
        self,
        run_id: str,
        *,
        message: str,
    ) -> MacroRunRecord | None:
        if not await self._run_cancel_requested(run_id):
            return None
        return await self._repo_call(
            "update_run_status",
            run_id,
            status="cancelled",
            error_code="cancelled",
            error_message=message,
        )

    async def _run_cancel_requested(self, run_id: str) -> bool:
        current = await self._repo_call("get_run", run_id)
        return current is not None and (
            current.status == "cancel_requested" or bool(current.cancel_requested_at)
        )

    async def _repo_call(self, method_name: str, *args: Any, **kwargs: Any) -> Any:
        """Run one synchronous repository operation outside the event-loop thread."""
        method = getattr(self.repository, method_name)
        return await asyncio.to_thread(method, *args, **kwargs)

    async def _post_final_output_if_needed(self, run: MacroRunRecord) -> MacroRunRecord:
        if self.post_back is None or run.final_output is None:
            return run
        post_idempotency_key = f"chat_macro:{run.run_id}:final"
        final_message_id = await _maybe_await(
            self.post_back(
                run_id=run.run_id,
                final_output=run.final_output,
                post_idempotency_key=post_idempotency_key,
            )
        )
        if final_message_id:
            return await self._repo_call(
                "mark_final_posted",
                run.run_id,
                final_message_id=str(final_message_id),
                post_idempotency_key=post_idempotency_key,
            )
        return await self._repo_call("get_run", run.run_id) or run

    async def _run_branch_with_retries(
        self,
        *,
        run_id: str,
        step: MacroStep,
        snapshot: MacroContextSnapshot,
        model_selection: dict[str, Any],
        normalized_args: dict[str, Any],
        macro: MacroDefinition,
        strategy_metadata: dict[str, Any],
    ) -> MacroBranchRecord:
        attempts = 0
        last_error = "branch failed"
        last_error_code = "branch_failed"
        max_attempts = max(1, macro.execution.retries_per_branch + 1)
        prompt = step.prompt or ""
        for _ in range(max_attempts):
            cancelled_branch = await self._cancelled_branch_if_requested(run_id, step)
            if cancelled_branch is not None:
                return cancelled_branch
            attempts += 1
            try:
                result = await asyncio.wait_for(
                    self.branch_runner.run_branch(
                        prompt=prompt,
                        snapshot=snapshot,
                        model_selection=model_selection,
                    ),
                    timeout=macro.execution.timeout_seconds,
                )
            except TimeoutError:
                last_error_code = "timeout"
                last_error = f"Branch timed out after {macro.execution.timeout_seconds} seconds."
            except Exception as exc:  # noqa: BLE001 - branch failures are recorded per branch
                last_error_code = "branch_failed"
                last_error = redact_sensitive_text(str(exc) or type(exc).__name__)
            else:
                if result.status != "completed":
                    last_error_code = result.error_code or "branch_failed"
                    last_error = redact_sensitive_text(
                        result.error_message or result.error_code or "branch failed"
                    )
                else:
                    usage = dict(result.usage)
                    usage["branch_strategy"] = strategy_metadata
                    return await self._repo_call(
                        "upsert_branch",
                        run_id,
                        step_id=step.id,
                        label=step.label,
                        status="completed",
                        output_text=result.text,
                        attempt_count=attempts,
                        prompt_digest=_digest(prompt),
                        citations=result.citations,
                        usage=usage,
                        acp_child_session_id=result.acp_child_session_id,
                        retained=_retain_branch(normalized_args, macro, self.settings),
                    )
            if attempts < max_attempts:
                cancelled_branch = await self._cancelled_branch_if_requested(run_id, step)
                if cancelled_branch is not None:
                    return cancelled_branch
                await asyncio.sleep(min(2.0, 0.25 * (2 ** (attempts - 1))))

        return await self._repo_call(
            "upsert_branch",
            run_id,
            step_id=step.id,
            label=step.label,
            status="failed",
            attempt_count=attempts,
            prompt_digest=_digest(prompt),
            usage={"branch_strategy": strategy_metadata},
            retained=_retain_branch(normalized_args, macro, self.settings),
            error_code=last_error_code,
            error_message=last_error,
        )

    async def _merge_outputs(
        self,
        *,
        profile: MacroOutputProfile,
        outputs: Mapping[str, str],
        failed_branches: list[MacroBranchRecord],
        branch_outputs: list[MacroBranchRecord],
    ) -> str:
        failed_payloads = [branch.model_dump(mode="json") for branch in failed_branches]
        branch_payloads = [branch.model_dump(mode="json") for branch in branch_outputs]
        if self.merge_runner is not None:
            return str(
                await _maybe_await(
                    self.merge_runner(
                        profile=profile,
                        outputs=dict(outputs),
                        failed_branches=failed_payloads,
                        branch_outputs=branch_payloads,
                    )
                )
            )
        return render_output_profile(
            profile,
            outputs,
            failed_branches=failed_payloads,
            branch_outputs=branch_payloads,
        )

    def _resolve_output_profile(
        self,
        run: MacroRunRecord,
        macro: MacroDefinition,
    ) -> MacroOutputProfile:
        profile_name = run.output_profile or run.normalized_args.get("output_profile") or macro.output_profile
        return self.settings.output_profiles.get(str(profile_name), self.settings.output_profiles["default"])


def _planned_branch_steps(
    macro: MacroDefinition,
    normalized_args: dict[str, Any],
    settings: MacroExecutorSettings,
) -> list[MacroStep]:
    if macro.command == "wrapup" and normalized_args.get("preset") == "dev_handoff":
        steps = _dev_handoff_steps()
    else:
        steps = [step for step in macro.steps if step.type == "branch_prompt"]
    for index, question in enumerate(normalized_args.get("question") or [], start=1):
        steps.append(
            MacroStep(
                id=f"custom_{index}",
                type="branch_prompt",
                label=f"Custom {index}",
                output=f"custom_{index}",
                prompt=str(question),
            )
        )
    max_branches = max(1, min(settings.max_branches, macro.execution.max_branches))
    if len(steps) > max_branches:
        raise MacroExecutionError("macro branch count exceeds configured limit")
    return steps


def _dev_handoff_steps() -> list[MacroStep]:
    return [
        MacroStep(
            id="summary",
            type="branch_prompt",
            label="Summary",
            output="summary",
            prompt="Summarize the implementation state for developer handoff.",
        ),
        MacroStep(
            id="changes",
            type="branch_prompt",
            label="Changes",
            output="changes",
            prompt="List the concrete code, test, and documentation changes.",
        ),
        MacroStep(
            id="verification",
            type="branch_prompt",
            label="Verification",
            output="verification",
            prompt="Summarize verification commands, results, and remaining skips.",
        ),
        MacroStep(
            id="risks",
            type="branch_prompt",
            label="Risks",
            output="risks",
            prompt="Identify risks, assumptions, and likely follow-up review points.",
        ),
        MacroStep(
            id="next_steps",
            type="branch_prompt",
            label="Next steps",
            output="next_steps",
            prompt="List the next steps needed to continue the work.",
        ),
    ]


def _default_model_available(selection: dict[str, Any]) -> bool:
    del selection
    return True


def _sync_requested(args: dict[str, Any]) -> bool:
    return bool(args.get("sync")) or str(args.get("mode") or "").lower() == "sync"


def _include_branch_outputs(args: dict[str, Any], profile: MacroOutputProfile) -> bool:
    return bool(args.get("include_branches")) and profile.include_branch_outputs


def _retain_branch(
    args: dict[str, Any],
    macro: MacroDefinition,
    settings: MacroExecutorSettings,
) -> bool:
    return bool(args.get("keep_forks")) or macro.execution.retain_scratch_branches or settings.retain_scratch_branches


def _estimated_total_tokens(snapshot: MacroContextSnapshot, steps: list[MacroStep]) -> int:
    prompt_tokens = sum(max(1, len(step.prompt or "") // 4) for step in steps)
    return (int(snapshot.token_estimate or 0) * len(steps)) + prompt_tokens


def _all_failed_report(branches: list[MacroBranchRecord]) -> str:
    lines = ["All branch prompts failed."]
    for branch in branches:
        label = branch.label or branch.step_id
        error = redact_sensitive_text(branch.error_message or branch.error_code or "failed")
        lines.append(f"- {label}: {error}")
    return "\n".join(lines)


def _merge_failed_report(exc: Exception, branches: list[MacroBranchRecord]) -> str:
    blocks = [f"Merge failed: {type(exc).__name__}"]
    for branch in branches:
        if branch.output_text:
            blocks.append(f"## {branch.label or branch.step_id}\n\n{branch.output_text}")
    return "\n\n".join(blocks)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value
