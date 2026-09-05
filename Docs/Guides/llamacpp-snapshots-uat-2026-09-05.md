# Manual llama.cpp snapshot UAT — 2026-09-05

Tracking: TASK-13174. Architecture: [ADR-043](../ADR/043-managed-llamacpp-manual-slot-snapshots.md).

## Scope and environment

The branch was rebased onto `dev` at `2742468a19`. The rebase preserved newer
Admin runtime-unavailable guidance and independent testing lessons. An independent
review found a Windows-class ordinary-profile deletion regression; commit
`3e35a31eff` fixes it with a proven-absent-state check and retained-snapshot guards.
Scoped re-review approved the fix without new findings.

Targeted verification after that fix: 232 backend tests passed, one opt-in live
test skipped, six baseline warnings. A separate profile/cache validation run
passed 21 tests. Four shared Admin/runtime Vitest modules passed 64 tests.
Ruff formatting/lint, compile checks and scoped production Bandit passed.
No full repository sweep was run. Missing-fcntl tests simulate unsupported
platforms; no actual Windows host was tested.

Browser UAT uses the actual Next.js Admin page on loopback, the production
llama.cpp FastAPI router, real supervisor/profile store/runner/snapshot service,
and the supplied native b10816 bundle/model. Only disposable files/profiles are
used. The fixture lives outside the repository at
`/private/tmp/llamacpp-uat-20260905/serve.py`; evidence/report files are in that
directory and `/private/tmp/llamacpp-rebased-verification.md`.

Explicit fixture boundaries:

- A disposable key supplies an admin principal; production AuthNZ provisioning
  and rate limiting are not exercised by this browser run.
- Configuration discovery is process-local; production config writes, downloads,
  asset registration and Use in Chat are blocked by the fixture.
- Only the fixture's service instance admits the exact candidate executable hash.
  The production build allowlist remains empty.
- Egress uses supported settings allowing the one native loopback origin/port.
  Production readiness and snapshot HTTP transport are not replaced.
- Uvicorn uses asyncio because the installed uvloop rejects the runner's
  subprocess `process_group` argument. No application code was changed for this.
- Health responses represent fixture startup readiness. Unrelated notification,
  persona and health/live endpoints are absent and produce visible 404 diagnostics.
  An existing AntD List deprecation is also visible; this is not a clean-console
  or full-application UAT claim.

The native configuration is CPU, context 16384, parallelism 1 and `swa_full=true`.
The public synthetic seed contains 1266 tokens. Build/model hashes and earlier
cold-control evidence are recorded in the [operator guide](llamacpp-manual-snapshots.md).

## Observed browser workflow

| Step | Evidence |
| --- | --- |
| Initial state | Real slot 0 idle with 1266 tokens; zero snapshots and receipts. |
| Save | Actual Save button dispatched one operation; UI showed Saving and blocked conflicting actions. Durable receipt completed with 1266 tokens and one 224.8 MiB saved copy. |
| Stop | Actual Stop button stopped the owned child; snapshot count and receipt count stayed at one. |
| Start | Actual Start button launched a new generation; slot returned empty and saved copy remained compatible. Slot Refresh observed the new state. No new receipt. |
| Restore confirmation | Opening the dialog did not change receipt count. Destination received focus; warning explains cache replacement and lack of message/tool recovery. Escape returned focus to Restore. |
| Restore | Explicit confirmation completed one restore receipt for 1266 tokens. The next synthetic suffix request reported `cache_n=1266`, `prompt_n=10`; this establishes actual reuse for this configuration, not merely successful file loading. |
| Reload | Reopening the panel recovered the completed result; receipt count stayed at two, so reload did not replay Restore. |
| Narrow layout | Actual 390px viewport: panel client width and scroll width both 356px. Dark confirmation and light recovered-result screenshots were visually inspected; no panel horizontal overflow. |
| Pause/Resume | Pause preserved one saved copy and two receipts. Resume started a new generation without another receipt. The identical suffix then reported `cache_n=0`, `prompt_n=1276`, proving the cold control and absence of automatic restore. |
| Delete | Opening confirmation left the saved copy intact and focused Cancel. The permanent-action label included the exact snapshot ID. Explicit confirmation removed the saved copy while retaining two receipts and the running generation. The subsequent real slots response showed idle slot 0 with 1291 tokens, so deletion did not erase its cache. |

The final slot read exceeded an initial five-second client budget; a read-only
retry with a 30-second budget succeeded. No mutation was retried. Fresh final
verification at `3e35a31eff` passed 232 backend tests (one opt-in live-test skip,
six warnings, 5.18 seconds) and 64 UI tests (four files, 9.80 seconds). The real
browser/native UAT above is separate from that skipped opt-in pytest invocation.

The named browser and disposable API/frontend processes were closed after UAT.
Only the public-synthetic saved copy was permanently deleted; user profiles,
original llama.cpp installation, model assets and conversation data were untouched.

## Acceptance limits

One model/configuration cannot establish standard-attention or hybrid/recurrent
coverage. Live Chatbook conversation/message/tool/approval invariance and its
actual slot routing have not been demonstrated. The original snapshot acceptance
task remains open and production support remains gated. The PR must not claim
general model support or conversation resumption from these cache tests.

The rebase also revealed that dev independently reused Backlog IDs 13159–13163.
The five snapshot task files have not been renumbered: the CLI has no ID-renaming
operation and explicit approval for direct task-file migration is outstanding.
TASK-13174 uniquely tracks this verification/PR work. The draft PR also awaits
the repository-required human-written Change summary before merge.
