"""Pydantic models for chat macro definitions and run records."""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

COMMAND_PATTERN = r"^[a-z][a-z0-9_]{0,63}$"
ARG_NAME_PATTERN = r"^[a-z][a-z0-9_]{0,63}$"
ARG_OPTION_PATTERN = r"^[a-z][a-z0-9_-]{0,63}$"
_ARG_NAME_RE = re.compile(ARG_NAME_PATTERN)
_ARG_OPTION_RE = re.compile(ARG_OPTION_PATTERN)


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())


class MacroArgSpec(_StrictModel):
    type: Literal["string", "boolean", "integer", "number"]
    default: Any = None
    repeated: bool = False
    aliases: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_default_type(self) -> MacroArgSpec:
        if self.default is None:
            return self
        if self.repeated:
            if not isinstance(self.default, list):
                raise ValueError("repeated arg default must be a list")
            for item in self.default:
                if not matches_arg_type(item, self.type):
                    raise ValueError(f"default item does not match arg type: {self.type}")
            return self
        if not matches_arg_type(self.default, self.type):
            raise ValueError(f"default does not match arg type: {self.type}")
        return self


class MacroStep(_StrictModel):
    id: str = Field(min_length=1, max_length=128)
    type: Literal["prompt", "branch_prompt", "merge", "post_result"]
    label: str | None = Field(default=None, max_length=128)
    output: str | None = Field(default=None, min_length=1, max_length=128)
    consumes: list[str] = Field(default_factory=list)
    prompt: str | None = None
    branch_strategy: Literal["auto", "chat_native", "acp_fork"] | None = None

    @model_validator(mode="after")
    def _requires_output_for_producers(self) -> MacroStep:
        if self.type in {"prompt", "branch_prompt", "merge"} and not self.output:
            raise ValueError(f"{self.type} step requires output")
        return self


class MacroPermissions(_StrictModel):
    tool_calls: list[str] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _reject_capabilities(self) -> MacroPermissions:
        if self.tool_calls:
            raise ValueError("tool_calls are not allowed in chat macro definitions")
        if self.skills:
            raise ValueError("skills are not allowed in chat macro definitions")
        return self


class MacroContext(_StrictModel):
    surfaces: list[str] = Field(default_factory=list)
    include_chat_history: bool = True
    include_workspace_context: str = "auto"
    retrieval: str = "auto"
    snapshot_at_dispatch: bool = True


class MacroExecution(_StrictModel):
    mode_default: Literal["background"] = "background"
    branch_strategy: Literal["auto", "chat_native", "acp_fork"] = "auto"
    max_branches: int = Field(default=6, ge=1)
    max_concurrency: int = Field(default=3, ge=1)
    timeout_seconds: int = Field(default=180, ge=1)
    retries_per_branch: int = Field(default=1, ge=0)
    merge_retries: int = Field(default=1, ge=0)
    partial_failure: Literal["best_effort"] = "best_effort"
    retain_scratch_branches: bool = False


class OutputProfile(_StrictModel):
    name: str = Field(default="default", min_length=1, max_length=128)
    description: str | None = None
    include_branch_outputs: bool = False


class MacroDefinition(_StrictModel):
    schema_version: Literal[1]
    name: str = Field(min_length=1, max_length=128)
    command: str = Field(pattern=COMMAND_PATTERN)
    description: str | None = None
    enabled: bool = True
    builtin_version: int | None = Field(default=None, ge=1)
    args: dict[str, MacroArgSpec] = Field(default_factory=dict)
    context: MacroContext = Field(default_factory=MacroContext)
    execution: MacroExecution = Field(default_factory=MacroExecution)
    steps: list[MacroStep] = Field(default_factory=list)
    output_profile: str = "default"
    permissions: MacroPermissions = Field(default_factory=MacroPermissions)

    @model_validator(mode="after")
    def _validate_arg_options(self) -> MacroDefinition:
        seen: dict[str, str] = {}
        for name, spec in self.args.items():
            if not _ARG_NAME_RE.fullmatch(name):
                raise ValueError(f"invalid arg name: {name}")

            exposed = [name]
            hyphenated_name = name.replace("_", "-")
            if hyphenated_name != name:
                exposed.append(hyphenated_name)
            exposed.extend(spec.aliases)

            for option in exposed:
                if not _ARG_OPTION_RE.fullmatch(option):
                    raise ValueError(f"invalid arg alias: {option}")

                previous_name = seen.get(option)
                if previous_name is not None:
                    if previous_name == name:
                        continue
                    raise ValueError(f"duplicate arg option {option} for {previous_name} and {name}")
                seen[option] = name
        return self

    @model_validator(mode="after")
    def _validate_step_consumes(self) -> MacroDefinition:
        previous_outputs: set[str] = set()
        for step in self.steps:
            if step.type in {"merge", "post_result"}:
                missing = [target for target in step.consumes if target not in previous_outputs]
                if missing:
                    raise ValueError(f"step {step.id} consumes unknown output: {', '.join(missing)}")
            if step.output:
                previous_outputs.add(step.output)
        return self


class MacroRunRecord(_StrictModel):
    run_id: str
    user_id: str
    macro_name: str
    macro_command: str
    macro_source: str | None = None
    macro_version: int | None = None
    macro_digest: str | None = None
    normalized_args: dict[str, Any] = Field(default_factory=dict)
    status: Literal[
        "pending", "running", "cancel_requested", "cancelled", "completed", "failed"
    ] = "pending"
    surface: str | None = None
    source_surface: str | None = None
    conversation_id: str | None = None
    workspace_id: str | None = None
    acp_session_id: str | None = None
    job_id: str | None = None
    output_profile: str | None = None
    context_snapshot: dict[str, Any] | None = None
    model_selection: dict[str, Any] | None = None
    status_message_id: str | None = None
    final_message_id: str | None = None
    final_output: str | None = None
    final_output_format: str | None = None
    final_post_status: str | None = None
    post_idempotency_key: str | None = None
    cancel_requested_at: str | None = None
    error_code: str | None = None
    error_message: str | None = None
    created_at: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    updated_at: str | None = None

    @computed_field
    @property
    def error(self) -> str | None:
        """Expose the canonical run error through the compatibility field."""
        return self.error_message or self.error_code


class MacroBranchRecord(_StrictModel):
    branch_id: str
    run_id: str
    step_id: str
    label: str | None = None
    output_name: str | None = None
    status: Literal["pending", "running", "cancelled", "completed", "failed"] = "pending"
    attempt_count: int = 0
    prompt_digest: str | None = None
    prompt: str | None = None
    output_text: str | None = None
    citations: list[Any] = Field(default_factory=list)
    usage: dict[str, Any] = Field(default_factory=dict)
    acp_child_session_id: str | None = None
    retained: bool = False
    error_code: str | None = None
    error_message: str | None = None
    created_at: str | None = None
    completed_at: str | None = None
    started_at: str | None = None

    @computed_field
    @property
    def output(self) -> str | None:
        """Expose canonical branch output through the compatibility field."""
        return self.output_text

    @computed_field
    @property
    def error(self) -> str | None:
        """Expose the canonical branch error through the compatibility field."""
        return self.error_message or self.error_code

    @computed_field
    @property
    def finished_at(self) -> str | None:
        """Expose canonical completion time through the compatibility field."""
        return self.completed_at


def matches_arg_type(value: Any, arg_type: str) -> bool:
    """Return whether a value matches a supported macro argument type."""
    if arg_type == "string":
        return isinstance(value, str)
    if arg_type == "boolean":
        return isinstance(value, bool)
    if arg_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if arg_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    return False
