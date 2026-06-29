# Skills Runtime Metadata Visibility Design

## Status

Approved for TASK-530.13 implementation.

## Problem

The Skills page currently exposes execution details in scattered ways. Detail responses include `allowed_tools`, `model`, `context`, and `disable_model_invocation`, but list summaries do not include enough information for users to scan runtime impact before opening or running a skill. The Skills table reduces runtime behavior to a "Model use" column that can be misread as an execution permission, and the test-run modal uses generic copy that does not reflect the selected skill.

This creates two user problems:

- Beginner users cannot confidently tell whether a test run may call a model or depend on declared tools.
- Power users cannot scan or verify fork mode, tool declarations, model overrides, and auto-invocation state from the list workflow.

## Goals

- Add structured, read-only runtime declaration metadata to Skills list and detail responses.
- Show a compact runtime summary in the Skills manager without breaking older responses that lack the new field.
- Show selected-skill runtime impact before `Render prompt only` and `Run test`.
- Use honest labels: declarations and possible model calls, not permission guarantees.
- Keep behavior unchanged. This work only exposes metadata that already exists in skill frontmatter/registry rows.

## Non-Goals

- No policy editor.
- No RBAC or authorization changes.
- No mutation of `allowed_tools`, model overrides, or skill execution settings.
- No enforcement changes in `SkillExecutor`.
- No database schema migration or persisted runtime column.
- No redesign of the Skills page layout.

## Runtime Metadata Contract

Add a `SkillRuntimeMetadata` response object:

```json
{
  "execution_mode": "inline",
  "test_run_may_call_model": false,
  "declares_tools": false,
  "declared_tool_count": 0,
  "model_override": null,
  "auto_invocation_enabled": true
}
```

Field meanings:

- `execution_mode`: Existing `context` value, normalized to `inline` or `fork`.
- `test_run_may_call_model`: `true` when `execution_mode` is `fork`. A non-dry test run may call the configured model in fork mode.
- `declares_tools`: `true` when the skill declares one or more `allowed_tools`.
- `declared_tool_count`: Count of declared tool strings.
- `model_override`: Existing skill model override, if set.
- `auto_invocation_enabled`: `true` when `disable_model_invocation` is false. This describes whether the skill can be advertised for model auto-invocation context, not whether a user-triggered test run can call a model.

## API Shape

`SkillSummary` should include:

- Existing fields.
- `allowed_tools: list[str] | None`, optional for list consumers.
- `model: str | None`, optional for list consumers.
- `runtime: SkillRuntimeMetadata`.

`SkillResponse` should include:

- Existing detail fields.
- `runtime: SkillRuntimeMetadata`.

`SkillContextPayload.available_skills` should remain compatible with `SkillSummary`. Runtime metadata in this endpoint is response metadata only; it must not add tool/model text into the LLM-facing `context_text`.

## UI Behavior

Skills manager:

- Add optional table column `Runtime`.
- Column content should compactly summarize:
  - `Fork` or `Inline`
  - `Test may call model` for fork, or `Prompt only by default` for inline
  - `N tools declared` when `declares_tools` is true
  - `Model override` when `model_override` is set
  - `Auto off` when `auto_invocation_enabled` is false
- The column must tolerate legacy responses without `runtime`, `allowed_tools`, or `model` by deriving a conservative fallback from existing fields.

Skill test-run modal:

- Accept selected-skill runtime metadata when available.
- Show runtime impact before action buttons.
- Keep `Render prompt only` copy clearly separate from `Run test`.
- Do not imply that declared tools are actually available, approved, or executable.

Import review:

- Existing import review can continue to show parsed model/tools. Labels should be interpreted as runtime declarations.
- No new import mutation behavior is required for this task.

## Accessibility and Copy

- Table column labels and button labels remain text-based and screen-reader visible.
- Runtime tags must not rely on color alone. The tag text carries the meaning.
- Modal copy should be concise and state actual behavior:
  - Dry render does not invoke fork/model/tool execution.
  - Fork test runs may call the configured model.
  - Declared tools are declarations, not availability guarantees.

## Risks and Mitigations

- Risk: Users interpret `allowed_tools` as permission guarantees.
  - Mitigation: Use `declared tools` in UI labels and schema descriptions.
- Risk: `disable_model_invocation` is misunderstood as "no model call ever."
  - Mitigation: Expose `auto_invocation_enabled` and keep `test_run_may_call_model` tied to fork mode.
- Risk: Frontend tests and older mocks break because summaries gain fields.
  - Mitigation: Make frontend fields optional and add fallback derivation.
- Risk: Runtime metadata drifts between list/detail/context endpoints.
  - Mitigation: Use one backend helper to derive the metadata dictionary from existing values.

## Acceptance Criteria

- Skills list responses include accurate `runtime`, `allowed_tools`, and `model` values.
- Skill detail responses include accurate `runtime`.
- `/skills/context` remains valid and does not change `context_text` semantics.
- Skills manager can show a runtime summary column and remains compatible with missing runtime data.
- SkillPreview displays selected runtime impact before either test action.
- Backend and frontend focused tests cover the new response shape and UI disclosure.
