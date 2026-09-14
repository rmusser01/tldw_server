---
id: TASK-13174
title: Expose synchronous EPUB summarization in Service Prompts
status: In Progress
assignee: []
created_date: '2026-09-05 14:31'
updated_date: '2026-09-05 15:15'
labels: []
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/2882'
documentation:
  - Docs/Design/ebook-summary-service-prompt.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Approved bounded follow-on to PDF customization: add owner-specific EPUB system instructions through the existing Service Prompts catalog and shared WebUI/extension Settings.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 EPUB system guidance is editable and resettable through existing Service Prompts in both clients
- [x] #2 One authenticated-owner prompt snapshot is used across files, chapters, and recursive summaries; explicit prompts including empty values win
- [x] #3 Server defaults, disabled analysis, provider configuration, extraction and queued ingestion behavior are preserved
- [x] #4 Focused backend and shared Settings tests, lint, security validation and independent review pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Add regression tests, extend the existing registry/request adapter/Settings metadata, verify and review the scoped change.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Baseline: 108 backend tests passed. RED: 10 expected backend failures for missing EPUB catalog/lookup, explicit-empty and canonical provider handling; new shared Settings EPUB test failed before metadata implementation. GREEN: combined backend120 and shared Settings/service/domain196 passed. Ruff lint/format, compilation and Bandit passed (zero findings). Independent read-only review in progress. Official OpenAPI exporter initially needed its ignored output directory created; rerun in progress. Main checkout remains untouched.

Final: 120 focused backend +45 adjacent EPUB/ingestion/safety/chunking/usage tests passed; shared UI196 passed and WebUI-config targeted4 passed. Ruff, compileall and Bandit zero findings. Official OpenAPI export/typegen/check passed; deleting only new EPUB api_provider reproduces exact base fingerprint. Independent review found no actionable issues. Full repo suites, full frontend typecheck and live browser/provider E2E not run. Temporary dependency symlinks removed before commit; generated full schema/types stay ignored.

Published PR #2882 against dev at user request. Implementation commit cbfdd903ec492fd70ba253e555c6f404eebad5c4. Branch and worktree retained for review follow-up.

Qodo posted three rule comments. Plan: move explicit-empty multipart preservation into get_process_ebooks_form before model validation, with a RED parser contract regression. Remove redundant exact owner lookup sequences in ownership/provider tests while keeping explicit no-read and single-snapshot contract checks. Evaluate architecture request against existing core resolver and API-owned resource lifetime; avoid introducing core-to-API dependencies.

Qodo remediation verified: explicit-empty restoration moved into validated form parsing (RED empty None vs empty-string assertion; GREEN omitted/empty/literal parser cases). Removed redundant exact lookup sequences but retained approved no-read and once-per-request contracts. Architecture disposition independently reviewed: authenticated Request/User DB acquisition and same-worker cleanup stay in API adapter; prompt override/default policy already resides in core. 123 backend tests, Ruff, compileall, Bandit zero findings and unchanged OpenAPI fingerprint check passed. Review follow-up recorded in design doc. Awaiting fresh remote review/checks and human Change summary before merge.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented independent EPUB summarization system instructions through existing Service Prompts storage, catalog and shared Settings. Captures a single authenticated-owner prompt for each synchronous processing request, preserving explicit empty prompts, deployment defaults, disabled analysis, provider credentials and recursive behavior. Canonical provider field matches PDF handling. No new settings system or core ingestion behavior changes; continues the approved incremental prompt-customization workstream.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
