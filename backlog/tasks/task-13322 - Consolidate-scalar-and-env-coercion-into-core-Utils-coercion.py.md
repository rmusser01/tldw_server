---
id: TASK-13322
title: Consolidate scalar and env coercion into core/Utils/coercion.py
status: To Do
assignee: []
created_date: '2026-09-22 04:56'
updated_date: '2026-09-22 05:11'
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
- [ ] #1 core/Utils/coercion.py exists with one documented truthy vocabulary
- [ ] #2 core/testing.py re-exports it so existing importers are unaffected
- [ ] #3 The five OCR backends and the TTS SSRF gate use it
- [ ] #4 Table-driven test covers strip, case, and the y/on/enabled spellings
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Design of record: Docs/Design/2026-09-21-scalar-and-env-coercion-consolidation-design.md (committed 42e219757b). It records the three-way contract decision, the ratchet-over-mass-refactor decision, the TRUTHY/FALSY union derivation, five staged migration gates and the explicit exclusions. TASK-13342 was a duplicate of this task and is closed.
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
