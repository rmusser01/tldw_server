# ACP Downstream-Agent Certification Checklist

Use this checklist when updating
`Docs/Development/ACP_Compatibility_Matrix.md` or claiming support for a named
downstream agent. It separates stub protocol evidence from live-agent evidence
so release notes do not overclaim compatibility.

## Certification Modes

| Mode | Use for | Evidence level |
| --- | --- | --- |
| `stub-smoke` | Proving the in-repo server, runner, mocked browser, and support-view paths still work. | `stub_smoke_tested` |
| `live-e2e` | Proving a named downstream agent binary works on a named host/runtime profile. | `live_e2e_tested` |
| `sandbox` | Proving Docker, Lima, VZ, or another configured sandbox runtime works for the agent/profile. | `sandbox_tested` |

The `stub-smoke` mode is CI-friendly and useful for release health. It does not
certify Codex, Claude Code, OpenCode, or a custom third-party binary.

## Smoke Harness

The focused harness lives at:

```bash
python Helper_Scripts/Testing-related/acp_certification_smoke.py --profile stub-smoke
```

It emits a machine-readable or Markdown command manifest. The manifest reuses
existing ACP checks instead of introducing a second test suite.

```bash
# Markdown manifest for release evidence
python Helper_Scripts/Testing-related/acp_certification_smoke.py --profile stub-smoke

# JSON manifest for issue bots or local scripts
python Helper_Scripts/Testing-related/acp_certification_smoke.py --profile stub-smoke --format json

# Run safe stub-smoke commands locally
python Helper_Scripts/Testing-related/acp_certification_smoke.py --profile stub-smoke --run
```

Live E2E evidence requires explicit operator-provided state and refuses to run
without these variables:

```bash
export TLDW_E2E_SERVER_URL=127.0.0.1:8000
export TLDW_E2E_API_KEY=<local-api-key>
export ACP_AGENT_PROFILE=<profile-key-from-matrix>

python Helper_Scripts/Testing-related/acp_certification_smoke.py --profile live-e2e --format json
python Helper_Scripts/Testing-related/acp_certification_smoke.py --profile live-e2e --run
```

## Stub-Smoke Checklist

Run this when updating the stub row or when validating release health without a
live downstream agent:

- Emit the manifest and attach it to the issue/PR evidence.
- Run backend ACP smoke coverage from the manifest.
- Run `tools/tldw-agent/scripts/verify-local-build.sh` from the manifest.
- Run the mocked Agent Tasks browser setup/run/diagnose flow when frontend
  dependencies are installed.
- Record any skipped command with a reason such as missing frontend
  dependencies or unavailable browser runtime.

Minimum capability IDs expected from stub-smoke evidence:

- `init`
- `session_new`
- `prompt`
- `structured_completion`
- `diagnostics`
- `cancel_close`
- `artifacts`
- `review_loop`
- `workspace_env`
- `mcp_injection`
- `redacted_support_view`

`workspace_env` and `mcp_injection` can be limited/mocked in stub-smoke mode.
Do not upgrade those checks to live support without named host/runtime evidence.

## Live-E2E Checklist

Run this when changing a candidate agent row from `documented_unverified` to
`supported`, `supported_with_caveats`, or `experimental`:

- Record host OS, runtime profile, repo commit, branch, and tldw_server config.
- Record downstream agent binary path and version.
- Record required provider credentials or local login state as present without
  exposing secret values.
- Confirm the configured profile can start through `/api/v1/acp/health` or
  `/api/v1/acp/setup-guide`.
- Exercise session create, prompt, structured completion, diagnostics, cancel,
  and close/teardown.
- Exercise artifacts, review loop, workspace env, MCP server injection, and
  redacted support views when those capabilities are part of the claim.
- Record unsupported or skipped capabilities with caveat taxonomy labels from
  `ACP_Compatibility_Matrix.md`.

Live E2E evidence should include the JSON manifest plus command output summary,
not raw secrets or full transcript payloads. Use `?redacted=true` support views
for transcript and artifact snippets in public evidence.

## Sandbox Checklist

Run this only when a support claim includes sandbox behavior:

- Record sandbox runtime (`docker`, `lima`, `vz`, or other configured backend).
- Confirm missing runtime/configuration fails closed with actionable setup
  guidance.
- Confirm workspace bind/mount behavior, cwd, env, and network policy.
- Confirm sandbox-specific diagnostics can distinguish host runtime problems
  from downstream-agent protocol errors.

Sandbox evidence can be separate from host live E2E evidence. A host-supported
agent is not sandbox-supported until these checks pass.

## Evidence Record

Use this template in #1539, per-agent follow-up issues, or PR closeout comments.

```text
Agent:
Profile key:
Support state:
Verification level:
Commit/branch:
Host/runtime:
Agent binary/version:
Config profile:
Manifest command:
Commands run:
Capability results:
Caveats:
Follow-up issue:
```

## Updating The Matrix

1. Update only the row whose evidence changed.
2. Keep support state and verification level aligned with the evidence.
3. Use `documented_unverified` when setup is documented but not verified.
4. Use `unsupported` only for proven protocol incompatibility.
5. Link follow-up issues for every failed or skipped capability that matters to
   the support claim.
