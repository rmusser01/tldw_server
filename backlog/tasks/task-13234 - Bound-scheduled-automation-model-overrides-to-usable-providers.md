---
id: TASK-13234
title: 'Bound scheduled-automation model overrides to usable providers at authoring time'
status: In Progress
assignee:
  - '@robert'
created_date: '2026-08-30 00:00'
updated_date: '2026-08-30 00:00'
labels:
  - scheduled-tasks
  - automation
  - llm-providers
  - authnz
dependencies: []
---

## Description (the why)

ADR-077 (chatbook task-18940) gives scheduled automations per-task model
selection that rides the definition payload: `input.provider` /
`input.model` / `input.max_tokens` override the automation config defaults
(`automation_executors.resolve_execution_target` precedence: definition
input → config defaults → server-default resolution). The execution half
honors the override, but **nothing bounds it at authoring time**: a
definition can name a provider the server does not configure (and the user
has no BYOK secret for), or a provider an admin override has disabled, and
the failure only surfaces as a failed run after the schedule fires — the
worst possible place to discover a typo.

This is AC#7 of chatbook TASK-18940: "per-task model selection rides the
definition payload (hermes parity), **bounded to the providers the server
account can use**."

## Acceptance Criteria (the what)

- [x] Preview and definition create/update validate `input.provider` when
      present: a provider not in the configured listing (after admin
      overrides: enabled) is a validation **error** (`input.provider`
      unusable), so authoring refuses with the reason instead of a future
      failed run. (Scope amendment: no BYOK exemption in this task — the
      scheduled executor threads no owner credentials, so BYOK-covered
      providers cannot run scheduled work today; accepting them at
      authoring would lie. BYOK-for-scheduled-execution is a phase-2
      design alongside server issue #2805.)
- [x] An admin-override-disabled provider, or a model outside the
      override's `allowed_models`, is a validation **error** (admin policy
      is a hard bound, not a warning)
- [x] A set `input.model` that is not in the resolved provider's known
      model list is a validation **warning** only (model lists drift and
      passthrough providers accept new names; the preview surfaces it, the
      definition still saves)
- [x] Blank/whitespace keys keep today's behavior (fall through to config
      defaults; no new errors)
- [x] The bound check runs at preview, create, and update — the three
      authoring surfaces — and the validation result is recorded in the
      preview's `validation_errors`/`warnings` like existing findings
- [x] Tests cover: usable provider passes; unconfigured provider errors;
      disabled provider errors; model-not-allowed errors (attributed to
      `input.model`); unknown model warns; blank keys unaffected;
      listing-read failure does not brick authoring; legacy previews
      hard-fail at create/update when the target becomes unusable

## Implementation Plan (the how)

1. Add an async bound check to `ScheduledTaskAutomationService` that takes
   the owner id and the normalized payload, extracting
   `input.provider`/`input.model` with the executor's coercion semantics
2. Provider usability: configured listing via `get_configured_providers()`
   + `apply_llm_provider_overrides_to_listing` (admin enablement) +
   `validate_provider_override(provider, model)` for hard policy, and the
   owner's BYOK secrets (`AuthnzUserProviderSecretsRepo`) as the
   server-config-independent path
3. Model-not-in-listing → warning only; unknown provider → error; wire
   both into the existing `_field_error` shapes
4. Call it from `create_preview`, `create_definition`, `update_definition`
   (owner id is already in scope at all three)
5. Tests mirroring the existing preview-validation suite's style, with the
   provider/BYOK/override layers monkeypatched

## Implementation Notes

- `_bound_execution_target_findings` (module-level in the automation
  service) coerces `input.provider`/`input.model` with the executor's
  blank-fallthrough semantics and produces `(errors, warnings)` in the
  shared `_field_error` / preview-warning-string shapes. Hard codes:
  `unusable`, `provider_disabled`, `model_not_allowed`; soft:
  `unknown_model` (warning only — model lists drift).
- Usability source: `get_configured_providers(include_deprecated=True)`
  (deferred import — the listing lives in the API layer) passed through
  `apply_llm_provider_overrides_to_listing`, plus
  `validate_provider_override(resolved, model)` for admin hard policy.
  A listing read failure returns no findings rather than bricking
  authoring; run-time failure semantics are unchanged.
- Wiring: `_normalize_preview` appends the findings (preview turns
  invalid on hard codes); `_create_definition`/`_update_definition` call
  `_require_execution_target_bound` so previews authored before the bound
  existed (or when the target became unusable between preview and
  consume) hard-fail with new error code `execution_target_unusable`,
  mapped to 422 `scheduled_task_execution_target_unusable` in the
  control plane's error table.
- Field attribution follows the error code: `model_not_allowed` lands on
  `input.model`, `provider_disabled`/`unusable` on `input.provider`
  (pinned-key discipline caught by the test matrix).

ADR required: no — implements chatbook ADR-077's AC#7 clause within the
existing scheduled-tasks control-plane design; no new boundary.
