---
id: TASK-13322
title: Consolidate scalar and env coercion into core/Utils/coercion.py
status: Done
assignee: []
created_date: '2026-09-22 04:56'
updated_date: '2026-09-24 00:02'
labels:
  - duplication
  - utils
  - migration
dependencies: []
references:
  - Docs/Design/2026-09-21-scalar-and-env-coercion-consolidation-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
ROOT CAUSE: the de-facto canonical truthiness parser is core/testing.py:30 is_truthy, imported by 140 PRODUCTION files, living in a module whose docstring opens "Lightweight helpers for test-mode detection". Engineers reasonably do not import production flag semantics from testing.py, so they write their own. Measured named private definitions: _env_bool 15, _env_int 23, _env_flag 6, _env_float 4, _coerce_bool 24, _as_bool 19, _to_bool 8, _parse_bool 14 = 113.

Consequences that clear the drop rule:
- THREE truthy sets among five sibling OCR backends. Set C uses lower() with NO strip(), so DOTS_VLLM_USE_DATA_URL with one trailing space (the ordinary result of a docker-compose environment: list) reads False and sends a server-local filesystem path to a remote vLLM. Set C guards eleven such decisions.
- SSRF escape hatch: TTS audio_cpp_config._as_bool falls through to bool(value), so allow_remote_base_url = n evaluates True (runtime-verified) and the adapter may point at an arbitrary non-loopback host. ADR-026 governs.
- LLM_Calls alone has five vocabularies under five names.

core/MCP_unified/environment.py:17 is a DELIBERATE copy for the standalone-package boundary with an identical truthy set - justified-divergence, leave it.

Destination: core/Utils/coercion.py, single responsibility scalar/env coercion, re-exported from core/testing.py for the 140 existing importers. NOT Utils/Utils.py.

Source: synthesis F15
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 core/Utils/coercion.py exists with one documented truthy vocabulary
- [x] #2 core/testing.py re-exports it so existing importers are unaffected
- [x] #3 The five OCR backends and the TTS SSRF gate use it
- [x] #4 Table-driven test covers strip, case, and the y/on/enabled spellings
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Design of record: Docs/Design/2026-09-21-scalar-and-env-coercion-consolidation-design.md (committed 42e219757b). It records the three-way contract decision, the ratchet-over-mass-refactor decision, the TRUTHY/FALSY union derivation, five staged migration gates and the explicit exclusions. TASK-13342 was a duplicate of this task and is closed.

Commit 1e03a1582c. New core/Utils/coercion.py: TRUTHY/FALSY union sets, parse_bool(value,*,default,key=None) three-way (unrecognised -> default, WARNING with key only), env_bool(key,*,default). parse_int/env_int deferred (no caller yet). core/testing.is_truthy/_env_truthy delegate via parse_bool(str(val or ''), default=False), so non-string semantics are unchanged (is_truthy(2) still False); only 'enabled' widens. testing.py re-exports parse_bool/env_bool. MCP_unified/environment.py kept as a deliberate copy, 'enabled' added, and a test asserts its _TRUTHY == coercion.TRUTHY. Migrated: audio_cpp_config (_as_bool deleted; allow_remote_base_url/managed/retain_request_artifacts/lazy_load via parse_bool), all 8 OCR backends (dots, dolphin, points, hunyuan, nemotron had unstripped Set C; chatllm, llamacpp, deepseek _env_bool now wrap env_bool, names kept), google_adapter._env_flag (fail-open fixed), request_resolution._is_truthy_value (accepts y). Behaviour note: unrecognised values on migrated OCR keys now fall back to the key default instead of False. Red-before/green-after: test_audio_cpp_config.py 8 new cases failed on 4a84d02b55, pass now; tests/Utils/test_coercion_adopters.py 6/6 failed on old code, pass now; tests/Utils/test_coercion.py 117 table cases pass. Suite diff (baseline 4a84d02b55 vs branch, -n 8): Media_Ingestion_Modification, MediaIngestion_NEW, TTS_NEW/unit, unit, MCP_unified, Chat, Watchlists, Collections, Telegram, Integrations, MCP_Hub, Workflows, Services, Scheduler, lint: 210 -> 201 FAILED/ERROR lines, zero new (9 fixed are parallel-load flakes; they pass serially on old code). LLM_Calls, LLM_Adapters, RAG_NEW/unit, Utils, Setup, Workspaces, Research_Workspace, Sharing, TTS_NEW/integration, Health, CI, Audio: 4 extra failures on branch reproduced identically on old code when run alone (timing flakes). Web_Scraping excluded: phase4 fixture lock collection error under xdist, unrelated. Bandit -ll on all touched source: no issues. Docs: design doc status paragraph + Utils/README.md. Not done: stage 4 ratchet test (design stage, not an AC).
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
One boolean vocabulary in core/Utils/coercion.py, re-exported from core/testing.py; audio.cpp SSRF gate, Gemini URL betas, eight OCR backends and Search-Agent flags migrated with red-first tests; two fail-open parsers closed. Ratchet (design stage 4) left for a follow-up.
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
