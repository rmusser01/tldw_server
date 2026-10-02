---
id: TASK-13342
title: Consolidate scalar and environment coercion behind one contract
status: Done
assignee: []
created_date: '2026-09-22 05:10'
updated_date: '2026-09-29 07:29'
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
- [x] #3 Stage 2: core/Utils/coercion.py exists with the documented contract and a table test over every TRUTHY and FALSY token, including that an unmatched token returns the supplied default rather than True
- [x] #4 Stage 4: tests/lint/test_private_coercion_ratchet.py is seeded at current per-module counts and demonstrably fails when a new private coercer is added
- [x] #5 Bandit run for touched scope
- [x] #6 core/testing.py:is_truthy delegates to core/Utils/coercion; MCP_unified/environment.py keeps a documented copy for the standalone package boundary
- [x] #7 Accepted values change only where recorded: stage 1's two intended fail-open fixes (TTS audio_cpp_config._as_bool and google_adapter._env_flag now fail closed on unrecognised input) and the additive 'y' token; stage 5 migrations each prove no other change, per the design's additive-sets rule
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Superseded by TASK-13322, which was filed first and identifies a better root cause: the de-facto canonical is_truthy lives in core/testing.py, a module documented as test-mode detection, imported by 140 production files -- so engineers reasonably write their own rather than importing production flag semantics from testing.py. 13322 also found a defect this task missed: one truthy set lowercases without stripping, so a trailing space from a docker-compose environment: list flips DOTS_VLLM_USE_DATA_URL to False and sends a server-local path to a remote vLLM.

The design doc Docs/Design/2026-09-21-scalar-and-env-coercion-consolidation-design.md remains the design of record and is now referenced from 13322. It contributes what 13322 does not carry: the three-way contract decision (truthy / falsy / unrecognised returns explicit default) with the argument for why the two-way contract is what makes the fail-open class expressible; the ratchet-over-mass-refactor decision; the TRUTHY/FALSY union derived so no currently-accepted spelling changes meaning; the five staged migration gates; and the explicit exclusions.

Closing as duplicate rather than merging, so one task owns the work.

Notes recorded on dev by the parallel core-review work (merged 2026-09-23):
REOPENED. Closing this as a duplicate of TASK-13322 was wrong, and Qodo caught it on PR #2981.
TASK-13322's file was added in commit 9b4d78bf46, which exists only on the branch fix/core-module-review-batch-1. That branch has no pull request and has not reached dev, so the file is absent from dev and from every branch that will merge. The backlog CLI resolves TASK-13322 because it reads state outside this branch, which is what made the duplicate look real.
Net effect of the closure would have been to delete the only in-repo record of three fail-open coercion defects -- one of them an SSRF escape hatch, where TTS audio_cpp_config._as_bool ends 'return bool(value)' so allow_remote_base_url = n evaluates True and disables the loopback guard (ADR-026 governs). Qodo's description of the consequence was right; only its phrasing ('the claimed canonical duplicate is not represented in the repository') read as if the task had never existed, which is why I initially judged the finding wrong.
This task stays open until either the work is done or TASK-13322 actually lands on dev. If 13322 lands, reconcile then: 13322 carries the root cause (core/testing.py:30 is_truthy imported by 140 production files from a module documented as test-mode helpers) and the OCR three-truthy-sets finding, while this task carries the google_adapter._env_flag and request_resolution._is_truthy_value defects and the explicit fail-closed acceptance criterion. Neither is a superset of the other.

Reconciled 2026-09-28 against dev after #3011 merged TASK-13322 (Done). The Done status here contradicted the REOPENED note above; this task stays open for the work 13322 did not carry. Verified on dev: AC1: TTS audio_cpp_config._as_bool is gone, google_adapter._env_flag is env_bool(name, default=False), and request_resolution._is_truthy_value is parse_bool(..., default=False), all fail-closed via core/Utils/coercion. AC2: TTS_NEW/unit/adapters/test_audio_cpp_config.py::test_negative_or_unknown_allow_remote_tokens_keep_the_loopback_guard. AC3: core/Utils/coercion.py plus the 13322 table test. AC4 amended: core/testing re-exports and delegates, while MCP_unified/environment.is_truthy deliberately copies the vocabulary ('copied, not imported, for the standalone package boundary'), which is a recorded divergence, not a gap. Still open: AC5 (stage 4 private-coercion lint ratchet: tests/lint/test_private_coercion_ratchet.py does not exist on dev), AC6, AC7.

Closed 2026-09-29 after #3049 merged. AC4: the ratchet is on dev and passes (4/4), enforced by backend-required's 'Enforce CI contracts and code ratchets' step. AC5 (Bandit): coercion.py 0 findings; only pytest asserts (B101) in the ratchet file. AC7 amended from 'no key resolves differently after any stage', which contradicted stage 1's intended fail-open fixes, to the design's actual rule: additive sets, and value changes stop and are recorded.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Delivered: stages 1-3 via TASK-13322 (#3011), meaning fail-closed parsers, core/Utils/coercion.py and the table test; stage 4 via #3049, where tests/lint/test_private_coercion_ratchet.py freezes 212 private coercers in 175 modules and runs in backend-required. Stage 5 (migrating modules) proceeds opportunistically under the ratchet, each migration lowering its seed. Bandit on coercion.py: 0 findings; the test file's B101 asserts are expected in pytest.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
