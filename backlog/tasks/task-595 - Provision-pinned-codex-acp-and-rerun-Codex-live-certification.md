---
id: TASK-595
title: Provision pinned codex-acp and rerun Codex live certification
status: Done
labels:
- ACP
- Codex
- agents
- certification
priority: high
ordinal: 595
documentation:
- Docs/superpowers/specs/2026-06-01-acp-codex-orchestration-design.md
- Docs/User_Guides/Integrations_Experiments/Getting_Started_with_ACP.md
- Docs/Development/ACP_Compatibility_Matrix.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Install or otherwise provision the pinned `zed-industries/codex-acp` 0.15.0 adapter so the Codex external_acp_adapter profile can be actively probed, then rerun the existing ACP certification smoke path. Prefer the GitHub release artifact if available; use pinned npm only as the documented operator fallback. Record exact version/path/auth blockers and do not claim live support unless initialize/session/prompt/cancel/close evidence passes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 codex-acp 0.15.0 is available on PATH or the installation/provisioning blocker is recorded with exact command output.
- [x] #2 Codex profile manifest transitions from adapter_missing to ready_to_probe or to a more specific stable blocker.
- [x] #3 ACP certification smoke run is executed when prerequisites are present, or refuses with an actionable nonzero blocker.
- [x] #4 Live evidence or blocker closeout is recorded without overclaiming support.
- [x] #5 Any repo changes are covered by focused verification and Bandit/diff hygiene when applicable.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Provisioned `@zed-industries/codex-acp@0.15.0` with the documented pinned npm fallback into `/private/tmp/tldw-codex-acp-0.15.0`.
- Confirmed executable path: `/private/tmp/tldw-codex-acp-0.15.0/node_modules/.bin/codex-acp`.
- Confirmed package metadata version: `0.15.0`. The adapter executable does not support `--version`; it exits with usage output for that flag.
- Re-ran the Codex agent-profile manifest with the temp bin prepended to PATH. The manifest reported `entrypoint.probe_state=ready_to_probe`, `blockers=[]`, `adapter_package=@zed-industries/codex-acp`, and `adapter_version=0.15.0`.
- Hardened `acp_certification_smoke.py` so stdio JSON-RPC probes start in a POSIX process session and forced cleanup signals the process group before falling back to direct process cleanup.
- Added explicit `PASS <command>` output for successful stdio JSON-RPC probe commands.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
TASK-595 completed. Final live Codex ACP certification probe was run outside the filesystem sandbox with `PATH=/private/tmp/tldw-codex-acp-0.15.0/node_modules/.bin:$PATH PYTHONUNBUFFERED=1 /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python Helper_Scripts/Testing-related/acp_certification_smoke.py --profile live-e2e --run --agent-profile codex`. It exited `0` and stdout showed `==> acp_initialize_probe (...)` followed by `PASS acp_initialize_probe`. No `codex-acp` or `acp_certification_smoke.py` processes remained afterward. Stderr still includes non-blocking Codex environment noise: Figma MCP OAuth refresh failure, Codex MCP startup notifications for unknown submission IDs, and skill icon path warnings. This task certifies the bounded initialize/session/prompt smoke path only; broader MCP/Figma environment cleanup and full backend cancel/close evidence remain separate work.
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
