# Explicit character greeting startup

TASK-13454.11; parent TASK-13454/TASK-13456. Approved design Docs/Design/2026-10-04-chat-conversation-ux-remediation-design.md. Latest dev5775d3fbbe integrated at5e66add940. Both authorized independent source reviews completed before implementation. No new API, storage authority, provider adapter or dependency.

## Stage 1: Preview without transcript mutation
**Goal**: Reuse the existing greeting hook/picker for an actual fresh tracked character; selection/reroll are preview only.
**Success Criteria**: No automatic messages/history insertion or direct server append; no-greeting and unsupported canonical seeds have explicit own-first-message paths; existing stored history is untouched.
**Tests**: Red-first actual hook/picker cases for default/alternate/no/long greeting, settings/card drift, user/character display names and scoped stale fetches.
**Status**: Complete

## Stage 2: Deliberate captured native creation
**Goal**: Reuse the existing native sender's creation/recovery/load prefix and stop greeting startup before admission or provider dispatch.
**Success Criteria**: Exact supported seed/provider/model/sampling; first-ID bind after actual owned coherent assistant row; pre-publication capture qualification; repeated/stale/unknown outcomes do not replay creation.
**Tests**: Real H1/Core with transport I/O doubles, default/exact alternate zero index/explicit sampling zeros, held create/capture/bookmark, New/account/settings/tools/card/workspace ABA, missing seed/changed card/unsupported Sync, zero provider calls.
**Status**: Complete

## Stage 3: Integrate the two actual composer surfaces
**Goal**: Put the existing picker opening actions near the composer, outside the live transcript, with primary actions before bounded preview and no generic onboarding above a fresh character introduction.
**Success Criteria**: Start with this greeting and Start with your message are explicit; own-message focuses and preserves exact input; startup awaits owned retainDraft ACK for exact text/collapse before POST; unsupported images block before retention and remain in the active composer; Send/repeat are blocked during preparation. Greeting recovery inputText is empty: no user message was submitted. Settling the stored assistant greeting never claims delivery or erases the separately retained user draft.
**Tests**: Actual full/compact callers, preparation races, draft write failure/recovery and first-ID bind, missing-model/provider prevention, English rendering and narrow layout.
**Status**: Complete

## Stage 4: Verify and record bounded completion
**Goal**: Relevant full suites, both type checks, touched lint, two independent source reviews, actual isolated backend/UI proof and a local commit.
**Success Criteria**: Only demonstrated behavior closes; native package/VoiceOver/full generation/Sync backend and other character start-entry transitions remain explicit parent dependencies. Bandit applies only if Python source is authored.
**Tests**: Relevant regression/type/lint/diff checks, isolated actual creation plus owner capture/reopen; inspect staging and remove only this completed plan after its history is retained.
**Status**: In Progress

## Review decisions

Canonical descriptors and query transport are completed prerequisites and will not be reimplemented. The existing native sender is the shared creation root; a narrowly explicit greeting intent may stop it before its completion suffix without a parallel creator. H1 must qualify the final coherent capture before bookmark/capture publication and onCapture; checking only onLoaded is too late. Changed card content is displayed from the actual accepted row with review feedback; unsupported/missing rows retain known-ID recovery, with no append fallback or second POST.

The existing creation recovery supports empty input and explains it. Preserve actual composer text/metadata through its owned retainDraft and bind, independently of the greeting-only attempt. A greeting ACK is not a user-turn ACK. Source remains recoverable if destination draft persistence fails.

Active global Sync v2 creation bypasses the factory and H1 rejects Sync ownership. This unit will expose unsupported/recovery truthfully and will not claim delivered Sync/backend authority. All-entry-point safe startup also depends on CharacterControlsSheet, Characters card actions and Quick Chat transitions; those are subsequent bounded units, not implicit acceptance here.

## Review verification in progress

Independent review corrections cover the H1 fence captured before draft ACK, image retention limited to the existing text-only contract, ordinary and queued entry guards, source-owned busy state, Retry, qualified keyboard focus handoff, saved-row display-only name projection, unconfirmed-creation blocking, deferred fresh visual controls, and proper English source/extension locale catalogs. Actual full and compact submit adapters verify owned assistant-only bind with zero user append or provider calls. Real storage holds reject newer H1/source intent before creation. Four preexisting full-page roleplay timeouts were reproduced on the unchanged baseline and diagnosed by the independent evidence reviewer as an unstable mocked ownership callback; the fixture now matches production callback lifetime and those cases pass. The core12suites326 checks pass, plus updated full-form/notices2suites61 after a red2 correction to defer the expression-image nudge in the fresh tracked composer. Both final type checks and lint comparison pass with no new diagnostics. Live preview/alternate/no-provider/own-message exact draft focus and 320/390 wrapping are verified. The actual non-Sync factory stores the display name as role; H1 correctly rejects that opening and known-ID review misattributes it to You. Supported live bind remains open under TASK-13454.13. Do not claim the observed case is the separate Sync limitation. Notification error rendering is repaired in separately committed TASK-13454.12. A fresh dev fetch found8aeddeee36 (router contract test isolation only); integration follows this working checkpoint.
