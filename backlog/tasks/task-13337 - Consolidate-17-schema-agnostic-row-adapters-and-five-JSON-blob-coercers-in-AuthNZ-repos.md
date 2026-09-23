---
id: TASK-13337
title: >-
  Consolidate 17 schema-agnostic row adapters and five JSON blob coercers in
  AuthNZ repos
status: In Progress
assignee: []
created_date: '2026-09-22 04:59'
updated_date: '2026-09-23 19:38'
labels:
  - duplication
  - authnz
dependencies: []
references:
  - 'tldw_Server_API/app/core/AuthNZ/repos/shared_workspace_repo.py:121'
  - 'tldw_Server_API/app/core/AuthNZ/repos/org_provider_secrets_repo.py:70'
  - 'tldw_Server_API/app/core/AuthNZ/repos/data_subject_requests_repo.py:44'
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
20 _row_to_dict definitions in repos/, 17 sharing the byte-identical schema-agnostic signature (row: Any) -> dict[str, Any] and containing ZERO column names - driver-row adapters, not schema mappers.

NOTE this is a deliberate carve-out from the standing ruling that *_to_dict is justified-divergence. That ruling is correct for its exemplar (Sync_DB.py:_device_from_row names 13 columns with per-field coercion into a typed SyncDevice). It does not cover these.

They have already split two ways: Class A (8 copies) guards row is None and returns {}, trying dict(row) first; Class B (9 copies) has NO None guard so dict(None) raises TypeError, and inverts the fallback order. Twelve Class-B call sites each replicate the guard instead of it living in the helper, and they disagree on the empty value ({} at two sites, None at three, in ONE repo).

Five JSON-blob coercers live in the same files and travel with them. Four clamp the parsed value to the expected container; data_subject_requests_repo.py:44 does NOT, so a row whose JSON column holds "[]", "null" or "123" yields a non-dict where the caller fallback promised a dict, and _normalize_record then raises AttributeError on a GDPR data-subject-request read path.

Destination: core/AuthNZ/repos/row_mapping.py - guard inside, return {}, clamp to the container.

Source: synthesis F36
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 One row adapter with the None guard inside
- [ ] #2 One JSON coercer that clamps to the expected container
- [ ] #3 Test covers None, bytes, malformed JSON and wrong-shaped JSON
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
PARTIAL - the embedded DEFECT is fixed; the consolidation is not.

FIXED: data_subject_requests_repo._parse_json_field now clamps the parsed value to the container the caller asked for, matching the four sibling copies. It previously returned json.loads(...) raw, so a column holding "[]", "null", "123" or a bare string yielded a list/None/int/str where fallback promised a dict.

CONSEQUENCE IS SHARPER THAN THE FINDING STATED: coverage_metadata is declared dict[str, Any] on the response models (api/v1/schemas/admin_schemas.py:1027 and :1082), so an unclamped non-dict becomes a Pydantic VALIDATION ERROR on a GDPR data-subject-request read, not the AttributeError originally predicted. selected_categories and preview_summary (fallback=[]) have the mirror problem.

Test: tests/AuthNZ/unit/test_data_subject_json_field_clamp.py - 12 cases (dict-fallback clamp, list-fallback clamp, well-formed passthrough, pre-parsed containers, None/malformed). Red before, green after.
Regression, stash-isolated on tests/AuthNZ/unit: WITHOUT the change 18 failed / 1193 passed; WITH it 9 failed / 1202 passed. Fixed 9, broke 0. The remaining 9 are pre-existing.

STILL OPEN - the consolidation itself: repos/row_mapping.py does not exist, 20 _row_to_dict definitions remain across the repos package (17 sharing the byte-identical schema-agnostic signature, split two ways on the None guard), and the other four JSON coercers are unmigrated.

2026-09-23 reconciliation: no ACs met. AC1 NOT met - repos/row_mapping.py does not exist; 20 _row_to_dict definitions still in core/AuthNZ/repos/. AC2 NOT met - JSON coercers still live in 5 repos (shared_workspace, managed_secret_refs, mcp_hub, data_subject_requests, prototype_workspaces); only the data_subject_requests copy was fixed to clamp (8c1a637a2d). AC3 NOT met - tests/AuthNZ/unit/test_data_subject_json_field_clamp.py (12 passed today) covers only the data_subject copy, and it tests None, malformed and wrong-shaped input but no bytes. Left: build row_mapping.py (a row adapter with the None guard built in, plus a clamping coercer), migrate the 17 schema-agnostic adapters and 5 coercers, and add a test that includes bytes.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
