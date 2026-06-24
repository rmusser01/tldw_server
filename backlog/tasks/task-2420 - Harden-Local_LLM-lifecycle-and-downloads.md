---
id: TASK-2420
title: Harden Local_LLM lifecycle and downloads
status: Done
assignee: []
created_date: 2026-06-23 18:25
updated_date: 2026-06-24 03:46
labels:
- local-llm
- llamacpp
- security
dependencies: []
references:
- tldw_Server_API/app/core/Local_LLM
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Verify and remediate Local_LLM review findings around llama.cpp acquisition SSRF protections, llamafile executable provenance, legacy process lifecycle locking, unmanaged PID termination, HuggingFace local-path/cache behavior, URL redaction, and HTTP status parsing. Continue moving llama.cpp lifecycle behavior toward the newer supervisor/process-runner path and consolidate duplicated llama.cpp argument formatting.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Validated findings are either fixed or documented as not applicable with evidence.
- [x] #2 llama.cpp acquisition fetches revalidate redirect/final targets and private-network protections at worker download time.
- [x] #3 Llamafile executable auto-download is opt-in and requires integrity verification before execution.
- [x] #4 Legacy handlers no longer terminate arbitrary unmanaged PIDs and lifecycle state mutations are serialized.
- [x] #5 Duplicated llama.cpp server argument formatting is consolidated for supervisor and legacy compatibility paths.
- [x] #6 Focused tests, Bandit on touched scope, and relevant lint/import checks pass or are documented.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Verified the current Local_LLM code still contains the reviewed issues: acquisition payload validation does not re-run DNS checks and the httpx fetch follows redirects automatically; Llamafile start can auto-download the latest executable without required integrity verification; legacy LlamaCpp/Llamafile/Ollama PID stop paths can target unmanaged processes; legacy LlamaCpp/Llamafile lifecycle state lacks explicit locks; HuggingFace accepts arbitrary local model directories and caches variants without a bound; download failures can log raw URLs; HTTP status fallback regex is double-escaped. Plan: add failing focused regressions, consolidate llama.cpp argument formatting into a shared helper used by both supervisor runner and legacy handler, harden download/provenance behavior, remove unmanaged PID stops, add lifecycle locks, tighten HuggingFace path/cache behavior, and verify with focused pytest plus Bandit.

Implemented shared llama.cpp server argument formatting, hardened acquisition redirect/final-target validation through the central HTTP client, required opt-in plus SHA-256 for llamafile executable auto-downloads, serialized legacy LlamaCpp/Llamafile lifecycle mutation, rejected unmanaged PID/port stops, tightened HuggingFace local path and loaded-model cache behavior, and redacted sensitive URLs/status parsing.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Verified and remediated the validated Local_LLM findings. Focused touched-file suite passed: python -m pytest -p no:unraisableexception tldw_Server_API/tests/LLM_Local/test_llamacpp_acquisition_service.py tldw_Server_API/tests/LLM_Local/test_llamacpp_process_runner.py tldw_Server_API/tests/Local_LLM/test_http_utils.py tldw_Server_API/tests/Local_LLM/test_llamacpp_handler.py tldw_Server_API/tests/Local_LLM/test_llamafile_handler.py tldw_Server_API/tests/Local_LLM/test_ollama_handler.py tldw_Server_API/tests/Local_LLM/test_huggingface_handler_hardening.py -q (100 passed). Py_compile passed for touched Local_LLM modules. Bandit passed on tldw_Server_API/app/core/Local_LLM with 0 findings in /tmp/bandit_local_llm_2420.json; only unrelated handler_utils nosec warnings were emitted.
PR follow-up verification after rebase: `python -m py_compile tldw_Server_API/app/core/Local_LLM/llamacpp_acquisition_service.py tldw_Server_API/app/core/Local_LLM/Ollama_Handler.py` passed. Focused suite passed: `python -m pytest -p no:unraisableexception tldw_Server_API/tests/LLM_Local/test_llamacpp_acquisition_service.py tldw_Server_API/tests/LLM_Local/test_llamacpp_process_runner.py tldw_Server_API/tests/Local_LLM/test_http_utils.py tldw_Server_API/tests/Local_LLM/test_llamacpp_handler.py tldw_Server_API/tests/Local_LLM/test_llamafile_handler.py tldw_Server_API/tests/Local_LLM/test_ollama_handler.py tldw_Server_API/tests/Local_LLM/test_huggingface_handler_hardening.py -q` (102 passed). Bandit passed on `tldw_Server_API/app/core/Local_LLM` with 0 findings in `/tmp/bandit_local_llm_2420_rebase.json`.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
- [x] #7 Focused tests pass.
- [x] #8 Bandit runs on touched Python files.
- [x] #9 Backlog task records verification and final summary.
<!-- DOD:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Reopened for PR follow-up: rebase `codex/local-llm-hardening-2420` on latest `dev`, inspect PR comments/checks, address validated issues, re-run focused verification, and update PR branch.
PR follow-up completed: rebased `codex/local-llm-hardening-2420` onto latest `origin/dev`, removed the unrelated Claims_Extraction design/task commit from the PR branch, and addressed the Qodo review findings. `_HttpxDownloadStream.__aenter__()` now offloads config loading and per-hop URL/DNS validation with `asyncio.to_thread`; `OllamaHandler.stop_server()` now terminates the handler-owned `asyncio.subprocess.Process` directly for managed PID/port stops instead of routing through the optional psutil PID helper.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
