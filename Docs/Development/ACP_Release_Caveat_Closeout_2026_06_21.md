# ACP Release-Caveat Closeout - 2026-06-21

This records the final reconciliation for
[GitHub #2398](https://github.com/rmusser01/tldw_server/issues/2398). The
goal is to make the release evidence boundaries explicit after the ACP
release-caveat child workstreams closed.

## Child Issue Outcomes

| Issue | Outcome | Release interpretation |
| --- | --- | --- |
| [#2404](https://github.com/rmusser01/tldw_server/issues/2404) | Closed. Live-backend browser E2E passed across ACP Playground, Agent Registry, Agent Tasks, and Research Workspace diagnostics. | User-facing ACP flows have live-backend browser evidence when run with the bundled runner home from `Config_Files/config.txt`. |
| [#2403](https://github.com/rmusser01/tldw_server/issues/2403) | Closed. Go runner verification passed on the recorded macOS host. | The runner build/test refresh is not a remaining release blocker. |
| [#2401](https://github.com/rmusser01/tldw_server/issues/2401) | Closed, with [#2408](https://github.com/rmusser01/tldw_server/issues/2408) split out and closed. | Retention and redaction policy covers ACP session evidence, audit records, diagnostics, artifacts, task previews, and promoted workspace artifacts. |
| [#2400](https://github.com/rmusser01/tldw_server/issues/2400) | Closed. Docker is the only sandbox runtime with current release-host pass evidence. | Release surfaces may claim Docker-backed sandbox runtime lifecycle evidence for the recorded host only. Lima, VZ, all-runtime sandbox support, and named-agent sandbox support remain unverified. |
| [#2402](https://github.com/rmusser01/tldw_server/issues/2402) | Closed. Goose, Hermes, and OpenCode passed `workspace-live-e2e` on the host runner with workspace binding and non-empty MCP server injection. | These agents remain `supported_with_caveats`; artifact-producing workflows, sandbox-backed execution, reviewer-loop behavior, and failure diagnostic payloads remain unverified. |
| [#2399](https://github.com/rmusser01/tldw_server/issues/2399) | Closed. Guardrails preserve conservative Aider, Continue, and custom-profile status. | Aider and Continue remain `documented_unverified`; the seeded `custom` profile remains template-only until a distinct named profile has live evidence. |

## Surface Reconciliation

- `ACP_Production_Readiness.md` now links this closeout record from the issue
  map and status summary.
- `ACP_Compatibility_Matrix.md` already distinguishes host E2E, workspace/MCP
  evidence, sandbox evidence, artifact evidence, reviewer-loop evidence, and
  failure-diagnostic evidence.
- `tldw_Server_API/Config_Files/agents.yaml` records Goose, Hermes, and
  OpenCode workspace/MCP evidence while keeping all three at
  `supported_with_caveats`.
- The setup guide and Agent Registry consume registry compatibility metadata and
  continue to avoid promoting documented-only profiles.

## Remaining Caveats

These are expected support boundaries, not blockers for closing #2398:

- Named downstream agents are not sandbox-supported until they pass
  agent-specific sandbox evidence, preferably `workspace-live-e2e` with
  `ACP_E2E_EXPECT_SANDBOX=1`.
- Goose, Hermes, and OpenCode did not produce ACP artifacts during the
  workspace-live runs, so artifact-producing workflows remain unverified.
- Reviewer-loop behavior and failure diagnostic payloads remain unverified for
  those live-agent profiles because the passing success paths did not exercise
  those cases.
- Aider, Continue, and seeded custom profiles remain conservative
  documented-only entries until concrete ACP commands or named custom profiles
  pass live certification.

## Verification

Closeout validation for this reconciliation slice:

- GitHub GraphQL issue-state check confirmed #2398 is open and #2399, #2400,
  #2401, #2402, #2403, #2404, and #2408 are closed.
- Targeted `rg` audits checked ACP readiness docs, compatibility docs, setup
  guide surfaces, seeded registry metadata, runner config, Agent Registry, and
  ACP setup-guide fallback copy for stale caveats or overclaims.
- Focused registry metadata tests passed for the shipped `agents.yaml` parser
  path.
- Bandit was run on the touched pytest file and reported the existing test-file
  `B101` assertion baseline only. The metadata checks use ordinary pytest
  assertions and avoid pinning the workspace-live evidence to one exact commit.
