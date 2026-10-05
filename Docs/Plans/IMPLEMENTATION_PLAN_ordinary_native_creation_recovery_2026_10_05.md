# Ordinary native Chat creation recovery — TASK-13454.8

Base: e1b00cd3ae; latest dev: 27ce9763870e5fb5de40dece8e2744b6b2475c28.
Parent: TASK-13454. Approved five-stage Chat UX remediation and D7 owner model.

## Stage 1: Integrate latest dev ownership
**Goal**: Preserve native ordinary first-send, source-qualified draft binding and exact canceled-load retirement.
**Success Criteria**: Upstream native creation/reopen and existing local/history/New-chat tests pass together. Canceling provisional ownership requires explicit reopening, never automatic replay.
**Tests**: normalChatMode history integration, real H1 controller, clear/navigation, Chat actions.
**Status**: Complete

## Stage 2: Reuse first-create recovery
**Goal**: Wrap ordinary native first-create with the existing durable owner/scope recovery contract and retain original input.
**Success Criteria**: No repeated POST after unknown or lost creation response; late ID retained privately; qualifying bind and native settlement update the existing marker.
**Tests**: Real ordinary sender with actual recovery storage; held POST, cold retry, stale load/New/ABA/account/Stop, confirmed settlement and unavailable storage.
**Status**: Complete

## Stage 3: Verify and review
**Goal**: Verify combined behavior through meaningful regressions, types, lint and two independent reviews.
**Success Criteria**: Relevant checks pass with exact limits recorded. No Critical/Important source findings remain. Browser/native extension/Sync and roleplay startup remain separate program gates.
**Tests**: Focused UI suites, WebUI typecheck, extension entrypoint compile, baseline-relative touched lint and diff checks. Bandit N/A TypeScript/docs.
**Status**: Complete

## Verification and review — 2026-10-05

- Kept incoming native ordinary creation and existing on-device/local ownership; retained exact canceled-load retirement with the stale-selection automatic-load barrier. The incoming idle reset was superseded by that stronger existing policy.
- Reused the durable first-create marker and existing recovery UI for both production caller surfaces. Creation-only body remains the incoming title/source contract; the record does not invent provider/settings supplied to that POST.
- Captured the same dispatch projection and source bind before service setup in both the sender and actual full-page caller. Ten held scope/actor caller cases reproduced unwanted POST before correction; two direct setup New/ABA cases reproduced source replacement. Explicit overrides and unrelated scoped settings remain usable.
- Supplied null IDs follow the canonical normal-target precedence. The regression first reproduced a successful native send leaving Core incorrectly fresh; the corrected actual caller binds the native address.
- Actual Core + mounted H1 tests cover qualified first bind, owned native reopen, setup/Stop/account/New/ABA drift, held capture/bookmark/bound-record IO, newer same/different ID, private late ACK, unavailable storage/locks, lost create, and retained input after an unconfirmed assistant save. Accepted ownership outlives the originating Stop control.
- Final 12 suites: **368 passed**. WebUI typecheck and extension entrypoint compile exit 0. Touched merged diagnostics compared with both merge parents: no added warnings/errors. Both diff checks clean. Both authorized independent source reviews report no remaining Critical/Important issue.
- Logs: `/private/tmp/ordinary-native-creation-recovery-final3-{green,types,extension,lint}.log`; dedicated red logs for awaited setup source, actual caller preparation, and overridden target.
- Bandit N/A: TypeScript/docs scope. No backend implementation changed.

## Evidence limits and remaining program work

Protocol IO/server rows are test doubles. Cold checks remount H1 against durable-storage shims; they do not prove process/browser restart. Surface parameterization plus source-verified caller strings is not packaged-extension acceptance. Native browser, actual backend/provider/Sync, VoiceOver, greeting-only startup, first-visit setup and broader stages remain open. Unknown creation without an ID still requires manual conversation discovery; existing recovery cannot claim universal server replay. Parent TASK-13454 stays In Progress.
