---
id: TASK-13342
title: Consolidate scalar and environment coercion behind one contract
status: To Do
assignee: []
created_date: '2026-09-22 05:10'
updated_date: '2026-09-23 12:32'
labels:
  - refactor
  - security
  - tech-debt
dependencies: []
references:
  - Docs/Design/2026-09-21-scalar-and-env-coercion-consolidation-design.md
  - 'tldw_Server_API/app/core/TTS/adapters/audio_cpp_config.py:18'
  - 'tldw_Server_API/app/core/LLM_Calls/providers/google_adapter.py:74'
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement `Docs/Design/2026-09-21-scalar-and-env-coercion-consolidation-design.md`.

~256 private re-implementations of string->bool, string->int and env-var reads span 37 modules (103 bool coercers / 37 modules, 102 int / 36, 51 env helpers / 18). Three of the copies are wrong and two fail **open** on security-relevant switches:

- `core/TTS/adapters/audio_cpp_config.py:_as_bool` ends `return bool(value)`, so `"n"`, `"none"`, `"disabled"`, `"nope"` all resolve **True**. It gates `allow_remote_base_url`, whose False value is what enables the loopback check — so an operator writing `"n"` for "no" disables the egress guard. Same parser gates `managed` (spawns a subprocess) and `retain_request_artifacts`.
- `core/LLM_Calls/providers/google_adapter.py:_env_flag` ends `return lowered not in {"0","false","no","off",""}`, so `"disabled"` is True. Gates whether caller-supplied URLs are forwarded for Google to fetch server-side.
- `core/RAG/rag_service/request_resolution.py:_is_truthy_value` omits `"y"`, so `SEARCH_QUERY_CLASSIFICATION=y` silently resolves the feature off while `RAG_GUARDRAILS_STRICT=y` resolves on in the same request.

The design selects `core/TTS/utils.py:parse_bool`'s three-way contract as canonical (truthy / falsy / unrecognised-returns-explicit-default), homes it in a new `core/Utils/coercion.py` (explicitly NOT `Utils.py`, which already carries 282 LOC of zero-importer symbols), and adopts it **by ratchet rather than mass refactor** — 256 sites across security switches is not one reviewable commit.

Stages are defined in the design doc. Stage 1 (the three defects) is independent and should not wait for the rest.

Found by the comprehensive core-module review (TASK-13293). All three defects independently verified by the orchestrator.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Stage 1: the two fail-open parsers and the missing "y" are fixed, each test-first
- [x] #2 A test proves an unrecognised allow_remote_base_url value leaves the loopback guard ON
- [ ] #3 Stage 2: core/Utils/coercion.py exists with the documented contract and a table test over every TRUTHY and FALSY token, including that an unmatched token returns the supplied default rather than True
- [ ] #4 Stage 3: core/testing.py:is_truthy and MCP_unified/environment.py:is_truthy delegate to it as two-way wrappers; neither is deleted
- [ ] #5 Stage 4: tests/lint/test_private_coercion_ratchet.py is seeded at current per-module counts and demonstrably fails when a new private coercer is added
- [ ] #6 No configuration key that parses today resolves to a different value after any stage
- [ ] #7 Bandit run for touched scope
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
REOPENED. Closing this as a duplicate of TASK-13322 was wrong, and Qodo caught it on PR #2981.

TASK-13322's file was added in commit 9b4d78bf46, which exists only on the branch fix/core-module-review-batch-1. That branch has no pull request and has not reached dev, so the file is absent from dev and from every branch that will merge. The backlog CLI resolves TASK-13322 because it reads state outside this branch, which is what made the duplicate look real.

Net effect of the closure would have been to delete the only in-repo record of three fail-open coercion defects -- one of them an SSRF escape hatch, where TTS audio_cpp_config._as_bool ends 'return bool(value)' so allow_remote_base_url = n evaluates True and disables the loopback guard (ADR-026 governs). Qodo's description of the consequence was right; only its phrasing ('the claimed canonical duplicate is not represented in the repository') read as if the task had never existed, which is why I initially judged the finding wrong.

This task stays open until either the work is done or TASK-13322 actually lands on dev. If 13322 lands, reconcile then: 13322 carries the root cause (core/testing.py:30 is_truthy imported by 140 production files from a module documented as test-mode helpers) and the OCR three-truthy-sets finding, while this task carries the google_adapter._env_flag and request_resolution._is_truthy_value defects and the explicit fail-closed acceptance criterion. Neither is a superset of the other.

STAGE 1 IMPLEMENTED on branch fix/coercion-fail-closed. All three defects verified empirically before being touched, not taken from the description.

TTS audio_cpp_config._as_bool -- verified fail-open AND verified exploitable end to end. 'n', 'N', 'none', 'nope', 'disabled' all returned True via the trailing 'return bool(value)'. Traced the consequence rather than stopping at the parser: _as_bool feeds allow_remote from extra_params, which feeds validate_base_url(allow_remote_base_url=...), which skips _is_loopback_host when True. A test constructing AudioCppConfig.from_provider_config with base_url=http://attacker.example.com:8080 and allow_remote_base_url='n' was ACCEPTED before the fix. ADR-026 governs that boundary. Fixed by returning the caller's default for unrecognised strings, and by adding 'y'/'n' to the recognised sets.

Google google_adapter._env_flag -- verified. Ended 'lowered not in {"0","false","no","off",""}', so 'disabled', 'nope', 'none', 'n' all returned True. Gates whether caller-supplied URLs are forwarded for Google to fetch server-side. Now returns bool(is_truthy(lowered)), so anything the canonical parser does not accept is false.

RAG request_resolution._is_truthy_value -- verified, and the divergence measured rather than assumed: _is_truthy_value('y') was False while core is_truthy('y') is True. So SEARCH_QUERY_CLASSIFICATION=y resolved off while RAG_GUARDRAILS_STRICT=y resolved on in the same request. Now delegates to is_truthy, which removes the second vocabulary instead of widening it.

A WEAKNESS IN MY OWN TEST, found and fixed. The end-to-end SSRF test originally used allow_remote_base_url='n'. Because the fix also added 'n' to the explicit falsy set, that test passed even with the fail-closed fallback reverted -- I verified this by reverting it and watching the test still pass, so it was not exercising what it claimed. It now parametrises over unrecognised spellings ('nope', 'disabled', 'none', 'maybe', '0 ') which exercise the fallback itself. Re-probed: reverting the fallback now fails 7 tests including 4 of the end-to-end guard cases.

Verification: 35 tests, red before (11 failures across all three defects) and green after. tests/TTS + tests/unit = 55 failed / 1026 passed against a clean-dev baseline of the identical 55 failed / 991 passed, so the delta is exactly the new tests. RAG_NEW/unit's 3 failures also reproduce on dev (postgres auth plus two unrelated). ruff clean on the four touched files; the 22 remaining findings in request_resolution.py match dev exactly, and the one I introduced (I001, from the added import) is fixed.

Stages 2 to 5 -- core/Utils/coercion.py, the delegating wrappers, and the ratchet -- are untouched. AC #1 and #2 are met; #3 to #7 belong to those stages.
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
