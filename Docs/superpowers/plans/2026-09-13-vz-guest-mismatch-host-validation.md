# Real VZ Guest Capability Mismatch Validation

Task: TASK-13243.3. Roadmap: #1442, Phase 1 lifecycle evidence.
Approved contract: `Docs/superpowers/specs/2026-05-18-vz-linux-lifecycle-drill-gaps-design.md`,
Guest-Agent Mismatch section. Follow-up to merged PR #2955.

## Stage 1: Verify And Add The Test
**Goal**: Exercise existing mismatch rejection, not add another policy layer.
**Success Criteria**: Explicit manual opt-in; actual VSock metadata omits `exec`;
rejection happens before guest execution and session-control persistence.
**Tests**: Existing portable mismatch tests; new host-test opt-in checks.
**Status**: Complete

Portable baseline passed. Review identified lost-create-reply cleanup; a failing
regression was added, then cleanup was changed to reconcile attempted ownership.

## Stage 2: Run The Real Drill
**Goal**: Boot a purpose-built fault bundle and prove healthy execution afterward.
**Success Criteria**: Missing-capability reason, no guest exec, no reusable control
or leaked VM; healthy session executes twice in one VM on the same helper.
**Tests**: Real Apple Silicon host test, plus a negative control demonstrating that
removing the create-time guard makes the test fail.
**Status**: In Progress (blocked by host disk space)

Offline fault-image installation and preparation-VM cleanup passed. The live
mismatch test and negative control have NOT run: the first live attempt failed
with `ENOSPC` before creating its evidence directory or starting a helper.

Use the existing image-store materializer for separate disposable run bundles.
Build a test-only Go overlay that removes `exec` from advertised capabilities but
leaves the executable handler intact. Install it into an offline disposable disk
through a separate owned preparation VM. Hash the canonical source before and
after. Keep the test agent out of production builds and canonical bundles.

## Stage 3: Review And Retain Evidence
**Goal**: Leave a repeatable test and accurate evidence, not a new runtime feature.
**Success Criteria**: Review complete; portable checks and Bandit pass; helper,
VM, socket, session, and source-immutability receipts preserved.
**Tests**: Focused pytest, formatting, Bandit, diff check, receipt inspection.
**Status**: In Progress

Focused portable suite: 35 passed, 1 intentional host skip. Bandit: zero findings.
Do not mark the task or live acceptance complete until Stage 2 has real receipts.

## Boundaries
- No host reboot, default-helper takeover, network attachment, or production code change.
- Only required-capability mismatch is proven here. Protocol-version mismatch,
  missing agent, and stuck boot/readiness remain distinct unproven live cases.
- An unavailable host is a skip, never acceptance; live evidence requires no skips.
- Do not claim escaped guest-process containment from the earlier pipe-drain fix.
