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

- [ ] Preview and definition create/update validate `input.provider` when
      present: a provider that is neither in the configured listing (after
      admin overrides: enabled) nor covered by the owner's BYOK secret is
      a validation **error** (`input.provider` unusable), so authoring
      refuses with the reason instead of a future failed run
- [ ] An admin-override-disabled provider, or a model outside the
      override's `allowed_models`, is a validation **error** (admin policy
      is a hard bound, not a warning)
- [ ] A set `input.model` that is not in the resolved provider's known
      model list is a validation **warning** only (model lists drift and
      passthrough providers accept new names; the preview surfaces it, the
      definition still saves)
- [ ] Blank/whitespace keys keep today's behavior (fall through to config
      defaults; no new errors)
- [ ] The bound check runs at preview, create, and update — the three
      authoring surfaces — and the validation result is recorded in the
      preview's `validation_errors`/`warnings` like existing findings
- [ ] Tests cover: usable provider passes; unconfigured provider errors;
      BYOK-covered provider passes without server config; disabled
      provider errors; model-not-allowed errors; unknown model warns;
      blank keys unaffected

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

(added after implementation)

ADR required: no — implements chatbook ADR-077's AC#7 clause within the
existing scheduled-tasks control-plane design; no new boundary.
