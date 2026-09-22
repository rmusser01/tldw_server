# Release UAT playbook: tiers A, B and C

**Version:** 1.0 · **Created:** 2026-09-17 · **Tracking:** TASK13262

**Purpose:** Give a person or automated runner the same repeatable user workflows, expected outcomes and evidence requirements before a public rollout or release.

**Status:** Workflow specification. This document does not implement a runner or certify that its scenarios currently pass.

## Using this playbook

1. Choose the candidate release and its supported feature/deployment scope; freeze the planned workflow variants before testing.
2. Prepare the four database/auth cells and applicable upgrade targets with the versioned fixtures and declared providers.
3. Run Tier A, then B/C and selected supplemental workflows, preserving the linked journeys and recording each prerequisite failure.
4. Complete X-01 through X-04 where applicable, including human UX review; record every result using the contract below.
5. Repair and retest failures, reconcile planned versus executed coverage, then have the release owner record the decision and any explicit exceptions.

## Contents

1. [Scope and tier definitions](#scope-and-tier-definitions)
2. [Release matrix and execution modes](#release-matrix-and-execution-modes)
3. [Run preparation and fixtures](#run-preparation-and-fixtures)
4. [Rules shared by every workflow](#rules-shared-by-every-workflow)
5. [Tier A: daily use](#tier-a-daily-use)
6. [Tier B: regular use](#tier-b-regular-use)
7. [Tier C: occasional use and administration](#tier-c-occasional-use-and-administration)
8. [Cross-cutting release workflows](#cross-cutting-release-workflows)
9. [Additional shipped features](#additional-shipped-features)
10. [Automation mapping and implementation order](#automation-mapping-and-implementation-order)
11. [Results, issue tracking and release decisions](#results-issue-tracking-and-release-decisions)
12. [Maintenance and source references](#maintenance-and-source-references)

## Scope and tier definitions

The historical A/B/C labels describe frequency of use. They are **not** the names of the current Playwright projects, nor a measure of the severity of a failure. An authorization failure in Tier C is still a release blocker.

| Tier               | Historical feature families                                         | Workflow IDs in this playbook                                                  |
| ------------------ | ------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| A — daily use      | Chat + RAG, Media, Settings, Notes, Flashcards                      | A-01 through A-12                                                              |
| B — regular use    | Characters, Audio, Watchlists, Collections, Content Review          | B-01 through B-09                                                              |
| C — occasional use | Evaluations, Prompt Studio, Agents/ACP, Writing, Admin              | C-01 through C-12; includes MCP and workflow automation as explicit extensions |
| X — release-wide   | Upgrade, account isolation, UX/accessibility, extension integration | X-01 through X-04                                                              |
| S — supplemental   | Other shipped surfaces absent from the historical tier list         | S-01 through S-06                                                              |

The primary surface is the WebUI. X-04 adds extension options/sidepanel coverage when an extension is part of the release. Desktop companions, hosted billing, OS sandboxes and external agent distributions require their own declared release profiles; WebUI success does not certify them.

Before execution, the release owner records every advertised feature as **required**, **excluded from this release**, or **not applicable to this deployment**. An enabled feature without a mapped workflow is a coverage gap. Experimental/advanced status alone does not excuse a broken advertised action: either exercise it with its dependencies or record the limitation in the release decision.

## Release matrix and execution modes

### Required deployment cells

| Cell          | Authentication                                                | Databases                                                 | Actors            |
| ------------- | ------------------------------------------------------------- | --------------------------------------------------------- | ----------------- |
| sqlite-single | Single-user API key/session, as documented for the deployment | SQLite auth and supported content stores                  | Owner             |
| sqlite-multi  | Normal multi-user login/session                               | SQLite auth and supported content stores                  | Admin, Alice, Bob |
| pg-single     | Single-user API key/session                                   | PostgreSQL auth **and PostgreSQL-capable content stores** | Owner             |
| pg-multi      | Normal multi-user login/session                               | PostgreSQL auth **and PostgreSQL-capable content stores** | Admin, Alice, Bob |

Record the actual engine used by each domain. A PostgreSQL auth database with SQLite content is a mixed deployment, not a substitute for a full PostgreSQL cell. Stores that the product only supports on SQLite must be named explicitly rather than falsely described as PostgreSQL.

- Run fresh application-state workflows in all four supported cells. Use real normal initialization and, in multi-user mode, supported admin bootstrap followed by UI-created ordinary accounts.
- For public releases, also run X-01 from the previous public release and the oldest supported upgrade starting point when different. Repeat for each supported database/auth cell. Record unsupported upgrade paths as product support limits, not passing tests.
- A clean-install claim additionally requires the published installer/container procedure on a clean supported environment. Reusing a virtual environment, package tree, browser or model cache only proves fresh **application state**. Record these separately.
- Use the built release frontend/server artifacts for release acceptance. Development-server testing remains useful diagnostic evidence; it does not replace the production build, deployment proxy and timeout path.
- Run the complete core catalog in the primary supported desktop browser. Run X-03 and critical A paths in every additional advertised browser and responsive surface; expand to all workflows affected by browser-specific changes. Fix the selected browser/viewport list before the run.
- Feature-dependent workflows need a declared provider/worker capability profile. At minimum, test one working provider per advertised capability and each changed provider adapter. Record untested providers; one text model does not certify vision, STT, TTS or ACP.

### Three evidence modes

| Mode                          | Environment                                                                          | What it proves                                                                                                                                                   |
| ----------------------------- | ------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| D — deterministic integration | Real browser, app, persistence and workers; versioned controlled downstream services | Repeatable state changes, permissions, error handling, counts and orchestration. List every substituted dependency.                                              |
| L — live integration          | Real browser/app/database and actual configured provider/worker                      | The actual feature works through its provider boundary, including final output and persistence. A catalog or health response is insufficient.                    |
| U — UX review                 | Human review of the same flow, with automation for measurable checks                 | Discoverability, understandable progress/errors, keyboard/focus behavior, responsive layout and output usefulness. Automated screenshots alone are not approval. |

Modes listed on a workflow are required components of its release result. D and L may share a real-server execution for deterministic CRUD; there is no need to duplicate identical work merely to assign two labels. When D substitutes a provider, L needs its own successful live execution. A browser-fulfilled application API only proves a UI contract and must be reported separately from D.

### Suggested cadence

| Run                 | Selection                                                                                                   | Decision                                                                       |
| ------------------- | ----------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| Pull request        | Changed workflows + their prerequisites and failure cases; relevant SQLite and PostgreSQL persistence tests | Fast regression feedback; cannot claim whole-release acceptance.               |
| Nightly             | D coverage for all required A/B/C/S workflows; four deployment cells; scheduled live provider rotation      | Detect drift, flakes and cross-feature regressions.                            |
| Release candidate   | Required A/B/C/X/S workflows, live capability coverage, all four cells, upgrade paths and U review          | Evidence for the public-release decision.                                      |
| Repair verification | Original failing case on the fixed revision, its negative/positive controls, then affected workflows        | Closes an issue; final integrated run still uses one frozen release candidate. |

Prefer isolated parallel cells only when their source/runtime storage, databases, ports, browser contexts and providers are independent. Serialize shared local inference and outage tests. Never stop a shared service to simulate an isolated failure.

## Run preparation and fixtures

### Preparation checklist

1. Record run ID, workflow-spec version/hash, candidate commit, artifact hashes, intended base commit, dependency versions, deployment mode, browser versions, UTC clock/time zone, database versions and provider/model IDs. Verify ancestry instead of assuming a branch began on the latest development commit.
2. Freeze a **planned case manifest**: workflow, variant, cell, phase, provider profile and required evidence modes. Include negative cases and blocked dependencies in the denominator before running anything.
3. Reserve isolated storage/configuration/temp/cache/build paths and ports per cell. Account for source-relative stores as well as configurable database paths. Do not import the mutable working checkout through an editable dependency alias.
4. Use the repository's official PostgreSQL test fixtures/fixture adapter for test database provisioning. Use direct non-superuser, non-BYPASSRLS application roles and verify their live attributes. Never bypass row-level security or use an administrator login to make ordinary-user tests pass. A missing PostgreSQL service is BLOCKED, not a skip converted to PASS.
5. Load secrets through the supported operator mechanism. Keep passwords, API keys, cookies and tokens outside logs, command arguments and published artifacts. Browser setup/login must use the real UI for fresh-setup and authentication cases; pre-seeded auth fixtures cannot certify these workflows.
6. Create the binary fixture pack and controlled test services described below if absent. Hash and version them. This document provides text fixtures and contracts; it does **not** ship audio/video/PDF/PNG-card binaries or a feed/agent/provider service.
7. Verify readiness and perform one real action per required capability. Record unavailable dependencies before dependent cases begin; keep independent cases runnable.
8. Set bounded deadlines, resource/cost limits and the fixture cleanup owner. Use monotonic elapsed time for durations and UTC timestamps for correlation.

### Stable test data

Use a unique namespace such as `uat-<run_id>-<cell>-<actor>-<workflow>`. Keep display names, returned IDs and file hashes in the result. **Public below means safe synthetic data, not shared permission scope**: user-created objects remain private unless sharing is explicitly configured.

| Fixture      | Contents / construction                                                                                                                                                                      | Expected use                                                                                                             |
| ------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| F-SOURCE     | The exact Rowan text below, as UTF-8 TXT and Markdown; prepare equivalent PDF/DOCX and one supported image-based PDF with a documented text-extraction oracle                                | Ingestion, search, citations, analysis, cross-feature handoff                                                            |
| F-DISTRACTOR | `Larch Observatory opened in 2021. Its director is Dr. Tomas Reed. It is in Pine Hollow. Public tours begin on Tuesdays at 09:00.`                                                           | Search discrimination; never substitute these facts for Rowan                                                            |
| F-BIOLOGY    | Five facts below, separated by blank lines                                                                                                                                                   | Exactly five distinct supported study cards                                                                              |
| F-PRIVATE    | Bob-only note/source: `The private UAT access phrase is CANARY-<random-run-suffix>.` Generate the suffix per run; never include it in Alice's inputs                                         | Isolation and negative retrieval controls                                                                                |
| F-IMAGE      | Versioned PNG of a red square on the left and a blue circle on the right on white; no text labels; record dimensions and hash                                                                | Actual attachment, preview, persistence, supported vision answer; inaccessible/corrupt variant                           |
| F-AUDIO      | Independently recorded WAV, mono 16 kHz PCM, speaking the audio sentence below; frozen reference transcript; plus silence, corrupt bytes and a short supported video carrying the same audio | STT, dictation, ingestion and failure behavior. Do not use output from the TTS system under test as the sole STT oracle. |
| F-TTS        | The same audio sentence as text; two genuinely available voices when voice switching is advertised                                                                                           | Synthesis, playback and export                                                                                           |
| F-FEED       | Controlled HTTP/RSS source: revision1 has item1; revision2 adds item2 with a stable unique ID; switchable delay/denial/failure; log real requests                                            | Real watchlist polling, deduplication and notifications. The browser must not fulfill the watchlist API.                 |
| F-CHARACTER  | TestBot instruction below; separate versioned valid character PNG/JSON export fixture and malformed import                                                                                   | Character identity, import/export and conversation transitions                                                           |
| F-EVAL       | Two exact-match examples: answer `ORBIT-742`, expected `ORBIT-742`; answer `ORBIT-999`, expected `ORBIT-742`                                                                                 | Deterministic one-pass/one-fail result; adapt fields to the current documented dataset schema                            |
| F-REPO       | Disposable tiny repository with a public README, one source file and an ignored sentinel file; fixed commit/hash, no credentials                                                             | repo2txt and narrowly scoped agent workspace actions                                                                     |
| F-AGENT      | Versioned controlled ACP/tool worker that can request a read of the public README, request a write to a disposable output file, delay, fail and cancel                                       | Permission and state-machine controls. A separate real advertised agent run is required for L.                           |

**F-SOURCE: exact Rowan text**

```text
Rowan Observatory

Rowan Observatory opened in 2019 in Cedar Ridge. Its director is Dr. Mira Vale.
The observatory's main telescope is named Selene.

Public tours begin every Friday at 18:00. Visitors must reserve a place before
arriving. The tour reference code is ORBIT-742.

The observatory studies variable stars and shares a monthly public report.
The source provides no ticket price and no current weather forecast.
```

Questions and answers: director → Dr. Mira Vale; location → Cedar Ridge; tour time → Friday at 18:00; telescope → Selene; ticket price/current weather → not provided. Semantic comparisons may normalize whitespace, punctuation and time notation; they must not accept changed facts. Preserve exact source bytes for storage roundtrips; use a separately declared normalized-text oracle for format extraction.

**F-BIOLOGY: exact five facts**

```text
The mitochondria is the powerhouse of the cell.

DNA stands for deoxyribonucleic acid.

Photosynthesis converts light energy into chemical energy.

The human body has 206 bones.

Water boils at 100 degrees Celsius at sea level.
```

**Audio sentence:** `Welcome to Rowan Observatory. The telescope is named Selene. Tours start on Friday at six in the evening.`

**Pirate instruction:** `You are a pirate. Respond to everything in pirate speak. Always say ARRR at least once.`

**TestBot instruction:** `You are E2E-TestBot. Always respond with exactly: BEEP BOOP.`

### Default budgets and model-output checks

These are initial **test budgets**, not a claim about published service-level guarantees. A release profile may set stricter or hardware-appropriate values before execution; changes after a failure must remain visible in the report.

- UI action acknowledgement: 2 seconds; ordinary navigation/save: 15 seconds; startup readiness: 120 seconds; short Chat/STT/TTS/generation: 180 seconds; background ingest/batch/agent job: 600 seconds. Capture first progress, first output and completion separately. Timeout is a failed or blocked observation, never silent success.
- Natural-expiry and scheduled-run cases use their observed token lifetime or configured interval plus a declared grace period, rather than the generic job timeout. Wait beyond the last actual issuance plus lifetime plus at least 10 seconds for the expiry case; record any later renewal that changes that deadline.
- Include controlled 45-second nonstream generation and delayed-first-chunk streaming cases. They must either complete within the declared end-to-end budget or show a truthful recoverable timeout while preserving user work. A frontend proxy dropping a request while the server later succeeds is recorded explicitly.
- Never use fixed sleeps to guess completion. Wait for the intended visible state and corresponding canonical result. For a job, outer `completed` plus a nested error/no artifact is not success.
- Strictly assert IDs, authorization, counts, saved text, provenance, versions and valid artifact formats. For model content, require the specified facts, no contradictory facts, valid citations when requested, and no reasoning/transport tokens in saved final answers.
- For instruction-following checks, use exact TestBot output and literal `ARRR` plus pirate style. A correct request with an incorrect model answer is still a failed live outcome; classify the provider/model failure separately from request construction.
- Score live quality cases against a frozen rubric. Record the first result; permit at most one declared recovery attempt, without replacing the first failure. Larger quality samples belong in a separate versioned evaluation dataset. An uncalibrated model judge cannot independently approve release UX or factuality.

## Rules shared by every workflow

### Execution contract

Every section below defines a **goal**, prerequisites, ordered actions with expected results, and required recovery/UX variants. Each variant is a separate planned case. All selected variants and evidence modes must pass before the workflow can be marked PASS.

- **Default actor/cells:** Owner in both single-user cells; Alice in both multi-user cells. Admin is used only where named. Repeat ownership-sensitive checks as Bob through X-02.
- **Default evidence:** before/after UI state, safe request/status and canonical IDs/counts, relevant output/artifact hash, normal reload/reopen result, per-step duration, attributed console/page errors, and issue IDs. A click or HTTP status without the intended visible and saved outcome is insufficient.
- **Default UX:** apply X-03 to each workflow's entry, principal action, progress, error and completion states. Recovery controls must name the problem and next action, preserve appropriate drafts, and avoid duplicate/conflicting notices.
- **Default cleanup:** capture evidence first; remove only recorded run-owned resources through supported product controls. Restore changed settings/permissions/schedules to the recorded baseline. Preserve failed profiles until triage is complete. Dispose of database fixtures through their owner; do not issue broad cleanup against shared data.
- Use user-facing UI for the action under test. Supported APIs may seed unrelated prerequisites in D and corroborate state, but label the adaptation. A fixture API that creates the very artifact being tested cannot count as its UI creation pass.
- Use accessible roles/names and stable test IDs for automation. Resolve IDs from actual responses; never hard-code database IDs. Capture download bytes, not just the download event. Refresh locators after meaningful state changes.
- Record empty, populated, invalid-input, unauthorized, delayed, failure/retry and reload behavior where relevant. If a documented action is absent for a required supported capability, fail or block it explicitly rather than quietly substituting a different feature.

### Dependency and failure handling

Create prerequisite resources per workflow when isolation is useful. Also run these complete linked journeys using the **same returned resource identities**:

- A-05 ingestion → A-07 cited QA/Media-to-Chat → A-09 answer reuse → A-11 study.
- A-08 Biology Note → A-10 five generated cards → A-11 five-card study.
- C-01 Prompt → A-03 Chat; B-01/B-03 Character context → B-02 Character Chat.
- B-07 watchlist → A-05 saved source → B-08 collection/B-09 content review where those handoffs are advertised.
- A-06 source → B-09 analysis/review → A-06 delete/restore, retaining the same source and versions.

When ingestion fails, downstream source cases are BLOCKED with the original issue ID. A separately seeded source can test a downstream component, but cannot turn the broken linked journey into PASS. Likewise, one manually reviewed card does not replace five generated cards. Continue independent cases after recording the failure.

### Owned-runtime shutdown evidence

Close only the run-owned browser contexts and finish active streaming requests before stopping their API. A released listening port alone does not prove worker or database-pool cleanup; retain process exit plus application teardown logs before reusing a profile or releasing its PostgreSQL fixture. A forced stop is recorded as forced, with cleanup unverified.

UAT351 isolated a signal-order failure in the retained Uvicorn 0.35.0 runtime: TERM followed by INT can replay the original TERM before async cleanup runs. Do not use that mixed sequence as a graceful-cleanup check. The existing single-TERM controls completed application, auth/content-pool and registry teardown after the owned browser connections closed. Record the actual server version and graceful-shutdown timeout, and distinguish waiting for live responses from application teardown. Preserve incomplete attempts rather than interpreting an empty port as a successful shutdown.

Capture a harness error separately from a product failure. Do not repeat a partially executed click sequence until the actual state and mutation count are known. A file-picker/dialog interruption may resume the remaining automation after the dialog is handled. After three failed attempts at an issue, stop repeating it, preserve evidence and investigate.

## Tier A: daily use

### A-01 — Fresh setup to first successful Chat

**Goal:** A new user reaches a working first conversation. **Modes:** D, L, U. **Needs:** fresh cell, working text provider.

1. Start the documented deployment with empty application data and a new browser context → the correct single/multi-user setup guidance appears; no old user or content is shown.
2. Single-user: discover/configure the provider, choose a real advertised model, validate and save. Multi-user: perform supported operator bootstrap/configuration, sign in as Admin and create Alice/Bob with ordinary roles → each action has an explicit result; record operator-only steps and required restarts.
3. Complete normal credential handoff/login and enter Chat → access mode, model/provider and current identity are correct; no storage injection or onboarding bypass.
4. Send `Reply with ORBIT-742.` → a real final answer arrives; inspect the selected provider/model and canonical saved conversation; reload and reopen it.
5. Follow the next suggested source action → it opens the intended ingestion screen with usable controls.

**Recovery/UX:** blank/invalid provider URL, unavailable model, missing key, failed validation, page reload mid-setup, setup re-entry after completion. Never persist invalid settings as verified or send the user to an extension-only instruction from the WebUI. Record installation versus reused-dependency coverage separately.

### A-02 — Login, persistence, logout and connection recovery

**Goal:** Access survives supported transitions and ends when the user signs out. **Modes:** D, L, U. **Needs:** accounts from A-01; one saved Note and Chat.

1. Sign in/enter a valid key normally; reload and reopen saved work → the same account and canonical data remain available.
2. Submit an invalid credential in a separate attempt → clear error, no authenticated content and no noisy unauthorized background polling.
3. Logout in multi-user mode, or Disconnect in single-user mode; reload and follow a protected deep link → an actionable access gate appears. Verify the documented distinction between connection-only disconnect and account-data clearing.
4. Reconnect normally → owned work returns without duplication; connection tests describe actual readiness.
5. Stop only the owned API, reload, then restart the same profile and use Retry → truthful unavailable state followed by recovered work; no data reset or unnecessary credential entry.
6. Multi-user: in an independently logged-in context, record actual token issuance time/lifetime without exporting tokens; park with no active application timers, wait past real expiry, then return → actual renewal and owned data access occur. Separately revoke that test session and verify normal sign-in recovery.

**Recovery/UX:** include expired/revoked refresh, interrupted network during renewal, Back after logout and a second account login. Active proactive refresh is not natural-expiry proof. API-key single-user JWT expiry is N/A with a reason. Never read/copy browser auth storage to manufacture this test.

### A-03 — Multi-turn Chat, stream controls and Retry

**Goal:** Conversation context and message identity remain correct across failures. **Modes:** D, L, U. **Needs:** text provider; a fresh ordinary conversation.

1. Send `Remember the tour code ORBIT-742.` then `What is the tour code?` → the second actual request contains the intended history and the final answer is correct.
2. Reload/open from history → saved user/assistant messages and order match the completed turns, with stable IDs and no duplicate answer.
3. Start a sufficiently long answer and use Stop → streaming ends, partial/cancelled state is honest, composer remains usable. Regenerate → branch/replacement behavior matches the documented action and preserves relevant context.
4. Trigger a controlled real backend/provider rejection, then fix availability and choose Retry → actionable original error; the retry keeps the same failed user-turn identity and produces one successful answer.
5. Change model through the actual picker and send another turn → the next request uses that provider/model; persisted settings and displayed selection agree.

**Recovery/UX:** include delayed first chunk, interrupted stream and hidden-tab return. Preserve draft/scroll/focus as promised, keep final answer distinct from reasoning, and record control variants separately. Request interception that alters a request must be disclosed; browser-fulfilled fake success is not live evidence.

### A-04 — Image attachments and capability recovery

**Goal:** Users can attach an actual image and understand model capability limits. **Modes:** D, L, U. **Needs:** F-IMAGE; text-only and working vision profiles when vision is advertised.

1. Attach F-IMAGE with the native chooser → filename/preview/removal controls appear; remove then reattach it and verify the intended file is selected once.
2. With a text-only model, send a question about it → a specific capability error or pre-send guard appears without sending an unsupported completion or duplicating the user turn.
3. Select the supported vision model through normal UI and retry/send → the actual provider receives image content/reference and correctly identifies red square left, blue circle right.
4. Reload the saved conversation → inspect the image element/asset retrieval and message identities, not merely attachment text. Exercise Retry on a failed image turn and check one resulting answer.

**Recovery/UX:** corrupt/oversized image, unavailable asset, interrupted upload and hidden-tab return. If only the text-model guard ran, record vision and image-persistence components as untested/blocked. Image generation is a separate capability; attaching an existing image does not certify it.

### A-05 — Ingest files/URLs and follow background progress

**Goal:** Supported content becomes usable saved media with honest processing status. **Modes:** D, L, U. **Needs:** F-SOURCE formats, F-AUDIO/video, configured processors.

1. Use Quick Ingest to upload F-SOURCE TXT; inspect the review/options before starting → filename, type, provider and selected operations are accurate.
2. Start once, minimize, reopen and reload during progress → the same job/batch identity remains visible; no duplicate submission or lost status.
3. Wait for the terminal result → successful source has a real media ID, correct owner, source text and requested processing artifacts. Distinguish saved-with-warning, fully processed and failed; record indexing readiness separately from source persistence.
4. Repeat declared format variants (Markdown, PDF/DOCX, OCR PDF, audio/video) → extracted/transcribed content meets the fixture oracle; unsupported features remain explicitly blocked.
5. Ingest `https://en.wikipedia.org/wiki/Playwright_(software)` and search/ask what Playwright is using that source → only the actual article counts. Denial content must not be stored as the article; external denial blocks this exact case. A controlled local article provides separate repeatable coverage.

**Recovery/UX:** unsupported/corrupt/oversized file, duplicate upload, queue delay, worker failure, quota denial, provider warning and retry. Ordinary and Admin quota paths must both initialize correctly on fresh PostgreSQL. Inner errors/no media ID must not be disguised by an outer completed job status. Record duplicate warnings and misleading counts as issues.

### A-06 — Media search, batch management, delete and restore

**Goal:** Library actions preserve the right content and respect permissions. **Modes:** D, U. **Needs:** three owned sources, including F-SOURCE/F-DISTRACTOR; recorded IDs/hashes.

1. Search `Cedar Ridge`, clear search and apply/remove a tag filter → only the intended source matches; totals, selection and empty states agree.
2. Select two sources and apply/remove a batch tag → exactly those two change; verify after reload.
3. Edit one source's supported metadata → content/other records remain intact; navigate away with an unsaved edit and exercise keep/discard behavior.
4. Delete a permitted source; repeat with the sole remaining active item → active count reaches zero correctly and Trash shows the original ID plus a meaningful deletion date.
5. Restore that exact item → same source content, identity/provenance and prior analysis versions return; Trash count decreases correctly after reload.
6. Multi-user: attempt deletion without permission, then perform the intended authorized operation as Admin → denial is truthful; permitted actions use actual permissions, not a role change made to bypass the test.

**Recovery/UX:** batch partial failure, stale selection, double-click prevention, confirmation cancel and navigation Back. Permanent deletion, when advertised, is a separate disposable-item variant with explicit confirmation and verified irreversibility; never use it on retained failure evidence.

### A-07 — Search/RAG to cited answer and grounded Chat

**Goal:** An answer is based on the correct owned source and its citations are inspectable. **Modes:** D, L, U. **Needs:** F-SOURCE, F-DISTRACTOR; indexed copy for advertised vector/hybrid modes.

1. Search `Who directs Rowan Observatory, where is it, and when are tours?` in Media-only full-text mode → Rowan is returned and the distractor facts are not substituted.
2. Ask Knowledge QA with the selected real provider → final answer gives the three correct facts and citations tied to the actual source/chunks.
3. Open each cited excerpt, source preview and Open in Media link → IDs, supporting text and target record agree.
4. Immediately select a source and use Chat with this media while content loads, then repeat after it loads → either action waits/disables safely or sends the complete intended source; an ID-only placeholder cannot masquerade as full-content handoff.
5. Send the grounded question from the handoff, reload the conversation → useful answer and intended source context persist. Repeat vector/hybrid/reranking variants only when advertised and ready, recording their settings and retrieval results.
6. Ask the ticket-price question absent from F-SOURCE → answer states the limitation without inventing a price; follow X-02 for foreign private content.

**Recovery/UX:** no results, indexing pending, invalid/stale citation, unavailable retrieval/provider and retry. A lexical fallback must identify its mode and cannot count as successful vector retrieval.

### A-08 — Notes lifecycle, search and export

**Goal:** Users can create, organize and recover their written work. **Modes:** D, U. **Needs:** F-BIOLOGY and two uniquely named short notes.

1. Create a titled note with content and tags → save feedback reflects actual completion; reload and verify exact normalized text, tags and owner.
2. Edit the body/title, add/remove a tag, search by unique title/body text and filter → results and counts match the intended records.
3. Open two clients/tabs for the same note and attempt conflicting edits → documented conflict handling preserves the user's draft or explicitly resolves it; no silent data loss.
4. Export using a supported format → downloaded bytes contain the expected note content/metadata; reopen/parse the file.
5. Exercise delete/restore where supported → active/Trash state and restored content agree. If the actual Notes lifecycle differs from Media, document and assert that contract rather than assuming identical endpoints.

**Recovery/UX:** first-use tour dismissal, blank/over-limit input, failed save, unsaved navigation and reload. A disappearing spinner without confirmed save is insufficient.

### A-09 — Chat answer to Note/card with provenance

**Goal:** Reused answers stay clean and linked to their origin. **Modes:** D, L, U. **Needs:** completed grounded answer from A-07.

1. Use the answer's Save to Notes action → saved Note contains final answer text only, plus the exact source conversation/message identity.
2. Open the Note, reload it and use Open conversation → the original conversation/message is accessible and correct.
3. Use Save to Flashcards on the same answer → review/edit the question and answer before saving; record card, linked Note, conversation and message IDs.
4. Open Study/Manage and follow the card's source link → it opens the intended clean Note; reload retains the same provenance.
5. Repeat the save only when explicitly requested → intentional additional saves are distinguishable from accidental duplicate retries.

**Recovery/UX:** failed save/retry, message menu focus/hover, stale source, account switch mid-handoff. No reasoning, transient status, template residue or another account's draft may enter saved content. If A-07 failed, a separate ordinary-answer reuse check is a component test, not this linked journey's pass.

### A-10 — Generate, review and manage five study cards

**Goal:** A source produces an exact, inspectable set of usable cards. **Modes:** D, L, U. **Needs:** saved F-BIOLOGY Note, real text provider.

1. Open the Note's Generate flashcards action → source text and provenance are prefilled accurately; choose exactly five cards and a unique new deck.
2. Generate → progress remains visible through the declared time budget; review five distinct draft questions/answers, one supported fact per card, with no contradictions or duplicate fact replacing a missing one.
3. Edit one draft; save the batch once → exactly five card IDs and the intended deck are persisted, with the edited content and source links. Reload and recount.
4. Rename the deck, move a card to/from an undecked state and edit a card → counts across filters and Study stay consistent. Restore the five-card deck for its linked A-11 case.
5. Import/export a declared supported format using a separate deck → parsed content, tags, image assets if included, and counts round-trip without silently changing the original deck.

**Recovery/UX:** delayed generation beyond 30 seconds, provider rejection, malformed/partial generated output, duplicate Save and interrupted batch save. A raw proxy500 with later backend success is a failed experience. No drafts/no save means five-card Study is BLOCKED, even if another manually created card works.

### A-11 — Study, scheduling, practice and accurate analytics

**Goal:** Reviews apply once, previews are truthful and progress is accurate. **Modes:** D, L for generated-card lineage, U. **Needs:** A-10 five-card deck; separate manually created controls and one undecked card.

1. Start the five-card due session under the recorded default scheduler policy; reveal and rate each distinct card Easy, waiting for the next identity → five intended review events, five distinct cards and a completed session count of five. Easy keeps this count-control case separate from relearning loops. Reload and verify persisted due dates/history.
2. On a separate control card, review once, then Practice again with Update schedule OFF → the answer can be revealed/rated without another scheduling review or changed due date.
3. Enable Update schedule, submit Good, then use Re-rate and Hard → each intentional scheduling event is recorded once; displayed next intervals reflect the latest saved state and match the resulting schedule within the configured fuzz/time policy. Re-rate is not assumed to erase the prior event.
4. On separate controls, include a real lapse (Again on a previously learned review-state card) and successful Hard/Good/Easy recalls → analytics matches persisted scheduler outcomes; Hard recall is not counted as forgetting merely because of its numeric rating. Button shortcuts and API rating values are different concepts.
5. Exercise a mixed deck/undecked queue and End early with cards remaining → remaining/reviewed counts and completed/ended session state are accurate; one card uses singular wording.
6. Change a disposable deck's supported scheduler/preset, reload, and test conflicting unsaved edits → changes remain scoped to that deck and reviews preserve history. Inspect source links and managed card images.

**Recovery/UX:** double rating, slow review response, failed review retry, stale preview after re-rate, keyboard shortcuts and current-card reload. Do not hard-code a 14-day interval for every scheduler: compare against the actual configured algorithm/authoritative preview with explicit tolerance.

### A-12 — Settings and provider readiness reflect real behavior

**Goal:** Configuration is understandable, persistent and used by the next action. **Modes:** D, L, U. **Needs:** baseline settings snapshot and one working provider.

1. Open Settings from navigation and recovery links → correct route/section appears; no unsupported discovery URLs or hidden loading loops.
2. Edit server/provider/model and relevant feature defaults, validate, save and reload → displayed saved state matches the accepted configuration; secrets remain masked.
3. Perform the affected real action (Chat, ingestion or speech) → the actual request uses the saved setting. Distinguish browser preference from server configuration and operator-only changes.
4. Enter invalid values or switch away with unsaved changes → actionable validation/keep-discard behavior; the previous working configuration remains usable.
5. Reopen Settings after logout/login as another user → only settings in their documented scope survive; no private keys or prior-account defaults leak.

**Recovery/UX:** provider inventory empty, connection outage, validation timeout, blank default model, route aliases and reload. Test advanced settings only when enabled; preserve the baseline for cleanup.

## Tier B: regular use

### B-01 — Character create/edit/import/export lifecycle

**Goal:** A character's identity and behavior settings survive normal management. **Modes:** D, U. **Needs:** F-CHARACTER and a second disposable character.

1. Create TestBot with its exact instruction, name and supported greeting/description fields → one saved character appears with a returned ID.
2. Edit its supported model/generation settings and descriptive fields; reopen/reload → the saved values agree with the editor and library.
3. Export a supported JSON/PNG character card and import it into a separate test namespace → parse/reopen the artifact and compare documented transferable fields. Record any deliberately excluded fields or duplicate-name behavior.
4. Import malformed/unsupported content → a useful error, no half-created character and no loss of the valid original.
5. Delete the disposable copy → only that character disappears; existing conversations follow the documented retention policy.

**Recovery/UX:** empty library, search/filter, save failure, duplicate submit, world-book catalog loading failure and canceling deletion. The character's main save succeeding does not hide a broken attached selector.

### B-02 — Character conversation and ordinary-Chat transitions

**Goal:** The intended character actually controls the selected conversation. **Modes:** D, L, U. **Needs:** B-01 TestBot and an existing ordinary conversation.

1. From the character library, choose Chat with TestBot while an ordinary conversation exists → the UI visibly enters the intended character context; no old persona silently persists.
2. Send `Hello.` → the request targets TestBot's actual character identity and the final answer is exactly `BEEP BOOP.`; verify canonical user/assistant messages.
3. Reload, reopen from history and send again → character identity/instructions remain correct; the conversation is not silently converted to ordinary Chat.
4. Switch to ordinary Chat and then another character → each next request uses the intended context and history; no prior character instruction leaks into the ordinary conversation.
5. Make the model unavailable, follow the offered selection/recovery action, and retry → working configured provider/model aliases are accepted and one answer is saved.

**Recovery/UX:** model-qualified versus raw IDs, delayed first stream chunk, cancelled generation and stale library selection. A readiness dialog is a passing recovery-state check only; successful Character Chat still requires the actual answer.

### B-03 — World books and dictionaries affect only intended context

**Goal:** Users can configure reusable context and verify its scoped effect. **Modes:** D, L, U. **Needs:** two characters and two ordinary conversations.

1. Create a world book with an enabled trigger entry containing a unique fictional fact; list/reopen it → catalog, entry count and text persist on both database engines.
2. Attach it to one character, trigger it in Chat → selected entry/context is included according to the configured matching rules and the answer reflects the fictional fact.
3. Disable/detach it and repeat in a fresh conversation; test the unattached character → the entry is absent from assembled context. Do not rely solely on model wording to prove absence.
4. Create a dictionary rule with an unambiguous input/output token; apply it in its documented scope → the actual transformation is observable; unrelated conversations remain unchanged.
5. Edit, export/import where offered and remove the disposable entries → counts and attachments agree after reload.

**Recovery/UX:** invalid/overlapping rules, empty catalog, database read failure and conflict. When an operation is unsupported, report that capability boundary; do not reinterpret a catalog500 as a legitimate empty state.

### B-04 — Text to speech, voice selection and playback

**Goal:** The chosen voice produces usable audio through the actual provider. **Modes:** D, L, U. **Needs:** F-TTS and a working synthesis provider.

1. Open Speech/TTS and discover voices → loading, provider grouping and readiness are accurate; choose an actual available voice.
2. Enter F-TTS and generate once → progress reaches a real decodable audio artifact with positive duration; record provider/voice, format and hash.
3. Play, pause, resume, seek where supported, and stop → playback follows controls without overlapping audio or stuck state. A person listens for the complete intelligible sentence and major glitches.
4. Export/download and reopen the audio; revisit history/reload → correct artifact and metadata remain accessible.
5. Switch to a second genuinely available voice when advertised and synthesize again → request and rendered selection change; the resulting artifact is independently usable.

**Recovery/UX:** empty text, unavailable voice, provider failure, unsupported format, long input and cancellation. A visible player/duration is necessary but not sufficient for speech-quality acceptance; a silent file cannot pass.

### B-05 — File transcription and microphone dictation

**Goal:** Speech becomes an accurate, usable transcript without losing work. **Modes:** D, L, U. **Needs:** F-AUDIO and working STT; microphone capability for dictation variant.

1. Upload F-AUDIO, choose the actual model/language/options and transcribe → transcript contains Rowan Observatory, Selene and Friday at six/18:00 with no contradictory facts.
2. Compare against the frozen transcript → normalize case/punctuation/number notation; initial target word-error rate is at most 15%, with all key facts required. Record the measured value rather than just non-empty text.
3. Save/export or use the offered downstream handoff → exact reviewed transcript and source identity persist after reload; timestamps, when offered, remain within audio duration and ordered.
4. Grant microphone access normally, speak the sentence, stop recording and review the transcript → recording starts/stops visibly, final text is editable and no recording continues unexpectedly.
5. Deny microphone access and process silence/corrupt bytes in separate cases → clear recovery guidance; no invented transcript or false success.

**Recovery/UX:** interrupted upload/stream, provider unavailable, cancel, oversized input and language switch. Browser-supplied fake microphone audio is D coverage; real recording/playback review is required for an advertised live microphone experience.

### B-06 — Audio Studio project to rendered export

**Goal:** A multi-part audio project can be edited, generated and exported. **Modes:** D, L, U. **Needs:** two short public text segments and required audio providers.

1. Open the current Audio Studio route; create a named Narration project with two segments → project and segment identities persist after reload.
2. Assign a supported voice, generate segments and inspect progress → each segment has the correct artifact/status; individual failure is not hidden by overall project completion.
3. Reorder/edit one segment and regenerate it → only intended content/order changes; prior successful artifacts remain recoverable according to the UI contract.
4. Preview the timeline, render/export and reopen the result → both segments appear in the intended order with intelligible playback and valid output bytes.
5. For each additional advertised workflow (Podcast, Briefing, Music), create one minimum complete project and render its promised output → mode-specific controls lead to an actual result, not merely a selected tab.

**Recovery/UX:** queued/failed segment, cancel/retry, reopen long-running job, export failure and legacy Audiobook links. Current tests map `/audiobook-studio` to `/audio-studio?workflow=narration`; verify the candidate's actual alias. A provider without music capability cannot certify Music through Narration output.

### B-07 — Watchlist source to scheduled item and notification

**Goal:** Monitoring discovers new content once and lets the user inspect it. **Modes:** D with real scheduler/worker, L for advertised external-source adapters, U. **Needs:** F-FEED and enabled watchlist workers/notifications.

1. Create a watchlist, add the controlled source, save its supported schedule/filter → settings persist and ownership is correct.
2. Run it manually against feed revision1 → a real run reaches terminal success, item1 is visible with source provenance and meaningful timestamps.
3. Repeat unchanged revision1 → no duplicate item/notification unless explicitly documented as a new event.
4. Advance F-FEED to revision2 and let a real scheduled interval occur → item2 appears once; notification links to the actual new item/run. Record actual clock/scheduler evidence; an immediate manual Run is not proof of scheduling.
5. Search/filter items, mark notification read and follow the source/ingest action → correct content opens, unread count changes, and saved-source handoff works where offered.
6. Disable the watchlist/schedule, then wait through one planned interval → no new scheduled run is created. Remove the disposable watchlist after evidence capture.

**Recovery/UX:** source denial, bad feed, delayed worker, partial run failure, retry and restart. Controlled source responses are acceptable D inputs; mocked watchlist/notification application APIs do not prove the worker, scheduler or persistence.

### B-08 — Collections organize and share the intended sources

**Goal:** Collection membership and visibility match user intent. **Modes:** D, U. **Needs:** three saved sources and ordinary accounts.

1. Create a named collection with description → one persistent collection, correct owner and empty count.
2. Add two source IDs → displayed membership/count is two; add the same source again and verify documented duplicate prevention.
3. Edit metadata, search/filter, remove one member and reload → membership and counts are correct; removing membership does not delete the source.
4. If sharing is advertised, grant a specific recipient the supported access level and exercise it as Bob → only granted content/actions are available. Revoke access and verify the next protected read follows the product's revocation contract.
5. Delete the disposable collection → collection disappears; underlying source retention matches the documented contract.

**Recovery/UX:** unavailable/deleted member, unauthorized access, batch partial failure and stale counts. Public fixture content is still private until explicitly shared.

### B-09 — Content draft review and source reanalysis

**Goal:** Reviewers can inspect changes and save the right final content. **Modes:** D, L, U. **Needs:** a draft produced through supported ingestion and a saved F-SOURCE record.

1. Open Content Review from the draft-producing flow → correct batch/draft, source, title and editable body appear; the empty state offers a useful entry action.
2. Edit the draft, save, reopen and inspect Diff → original versus edited content is accurate; Reset and unsaved-navigation behavior are explicit.
3. Request AI fix with a bounded instruction → proposed final text is inspectable before Commit; it does not silently overwrite unrelated work or invent source facts.
4. Commit one draft, then exercise a separate two-draft batch → saved media IDs/content match the reviewed drafts, successful items are counted once, failed items remain recoverable. Clearing drafts must not silently delete committed sources.
5. Separately analyze saved F-SOURCE, view it in Multi-Item Review, request a changed analysis and save/reload → new analysis/version appears on the same source with source content intact.
6. Fail both streaming and any supported fallback during the next analysis attempt → an error is shown and the previous saved analysis/version remains unchanged.

**Recovery/UX:** missing original attachment, conflict, failure between generation/save, partial commit and duplicate click. Current Content Review is an edit/diff/commit workflow; do not pretend historical approve/reject queue steps describe this screen. Moderation approval is S-05.

## Tier C: occasional use and administration

### C-01 — Prompt library to applied Chat behavior

**Goal:** Saved reusable instructions are actually applied. **Modes:** D, L, U. **Needs:** pirate instruction and a fresh ordinary Chat.

1. Create and save the named pirate Prompt with supported metadata/tags → sync completes and a persistent ID is returned; search/reopen it.
2. Edit a nonessential description, reload, then choose Use in Chat/System Instruction → the exact intended instruction is visibly applied and present in the actual request.
3. Ask `Tell me about the weather today.` → answer uses pirate style and literal `ARRR`; it must not invent a local forecast when location/current data are absent.
4. Reload the conversation and inspect the Prompt → applied context and saved Prompt identity remain correct.
5. Export/import where supported and delete a disposable copy → downloaded content and library counts agree; original conversation retention follows its contract.

**Recovery/UX:** save/sync failure, premature navigation, template variables left unresolved, unavailable model and account switch. A Prompt appearing in the library is not proof it was applied to Chat.

### C-02 — Prompt Studio variables, tests and revisions

**Goal:** A reusable prompt can be tested and improved reproducibly. **Modes:** D, L, U. **Needs:** Prompt Studio project, F-SOURCE and working model.

1. Open the Studio tab from Prompts (including legacy `/prompt-studio` entry), create a project/prompt with one supported template variable → save/reopen preserves the template and variable definition.
2. Leave the required variable empty and run → validation identifies it without launching a misleading empty request.
3. Supply F-SOURCE as the variable and run a request for its director → actual rendered prompt contains the supplied value once; real result gives Dr. Mira Vale; input/model/result provenance is available.
4. Save an edited revision and rerun the same input → history distinguishes revisions and retains the earlier result instead of overwriting it silently.
5. For advertised comparison/optimization features, execute a bounded comparison with a fixed rubric and budget → candidate/result linkage and selected winner are inspectable; a score alone is not proof of better content.

**Recovery/UX:** provider timeout, invalid template, concurrent edit and long optimization cancellation. Unsupported Studio controls are explicit gaps; plain Prompt CRUD cannot substitute for a Studio test run.

### C-03 — Evaluation dataset to interpretable single/batch results

**Goal:** Users can run a known evaluation and trust its counts and provenance. **Modes:** D, L for model-graded evaluations, U. **Needs:** F-EVAL; an enabled exact-match recipe and a separate model-graded recipe if advertised.

1. Create/import a two-example dataset and validate it → exactly two rows, visible schema errors for a malformed separate import, no silent dropped rows.
2. Configure exact-match evaluation using the documented output mapping and run → one example passes and one fails; totals equal two and aggregate score matches the documented aggregation.
3. Open example details and history, export/reload results → dataset version, input, expected/actual output, recipe and run identity remain linked.
4. Start a larger bounded batch with a controlled failing example/job → progress, partial results, cancellation and retry distinguish completed from failed work; retries do not double-count examples.
5. Run the model-graded recipe against a correct and clearly contradictory Rowan answer → real evaluator output and rationale are available with provider/model/rubric provenance. Review their reasonableness; do not claim evaluator calibration from two examples.

**Recovery/UX:** no provider, invalid recipe, interrupted polling, failed webhook if enabled and inaccessible foreign dataset. A page that shows either validation failure or success is not a successful evaluation case unless the selected case expected that failure.

### C-04 — Agent registry to completed task and retained result

**Goal:** A user can select an available agent and follow actual work. **Modes:** D, L, U. **Needs:** declared agent/runner, F-REPO, isolated project workspace.

1. Refresh the registry and inspect an agent → capabilities/readiness match the running integration; unavailable agents explain setup requirements.
2. Create a project and task asking for a summary of the public README → task shows the selected agent and correct project/workspace.
3. Start the task and follow queued/running/progress states → the real worker reaches a terminal outcome with a useful result and accessible diagnostics/run identity.
4. Reload project/task history → result, ownership and status persist. Start a second disposable task and cancel → execution stops or enters the documented cancellation transition without falsely reporting success.
5. Switch to Bob → Alice's private project/task/result is unavailable; a Bob-owned positive task remains usable.

**Recovery/UX:** missing runner, downstream failure, stale health, retry and long-running navigation. A task row created without actual execution is only creation coverage.

### C-05 — ACP session, tool permission and workspace boundaries

**Goal:** Tool actions honor the user's decision and workspace scope. **Modes:** D, L, U. **Needs:** F-AGENT plus one actual advertised ACP agent for L; disposable workspace.

1. Create an ACP session and send a harmless README-summary request → session identity, streaming output and final result are visible and reloadable as documented.
2. Cause a permission request to write one disposable output file → the UI explains the exact target/action and offers the documented decisions; no write occurs before approval.
3. Deny the request → the agent receives denial, no output file appears, and the conversation remains usable.
4. In a separate case, approve that same narrowly scoped action → exactly the intended file changes; retain before/after hashes and tool/session outcome.
5. Attempt an out-of-workspace/unauthorized operation using the controlled worker → denial is enforced, including direct tool execution boundaries; do not test on real confidential files.
6. Cancel/reconnect a session during a delayed action → no duplicated tool effect or untracked ongoing process; diagnostics remain accessible.

**Recovery/UX:** missing adapter, tool timeout, disconnected stream and repeated permission prompt. Never use actual messaging, purchases or production administration as test side effects. L must identify the real adapter/agent version; a deterministic stub only certifies D.

### C-06 — Writing session to reviewed, saved/exported revision

**Goal:** Assisted writing preserves the user's original work and chosen edits. **Modes:** D, L, U. **Needs:** F-SOURCE paragraph and a fixed rewrite instruction.

1. Create a Writing session and paste the source → title/body/session persist after reload.
2. Request `Rewrite this as two concise sentences without changing any facts.` with the chosen model/template → a real suggestion appears separately from the original until the documented apply action.
3. Accept part/all of the suggestion, edit manually, save and reopen → the user's selected final text persists; unrelated text is unchanged.
4. Exercise undo/revision history and export where offered → versions/downloaded bytes match the chosen content.
5. Fail or cancel the next generation and navigate away with unsaved edits → previous saved content remains; draft handling is explicit and recoverable.

**Recovery/UX:** empty input, invalid template, slow generation, output that changes a fact and selection loss. A POST being fired is insufficient without the resulting suggestion and saved artifact.

### C-07 — Repository to text bundle

**Goal:** A selected repository becomes a correct, bounded export. **Modes:** D, U. **Needs:** F-REPO at a fixed commit; supported local/remote input profile.

1. Choose the repository and explicit include/exclude rules → preview lists the expected README/source files and excludes the sentinel according to the selected policy.
2. Generate the text bundle → actual output includes file boundaries and complete selected content; counts/size agree with included files.
3. Download/reopen the bundle and repeat unchanged inputs → normalized output is stable apart from documented metadata.
4. Change an include rule and regenerate → only the intended file set changes; preview and export remain consistent.

**Recovery/UX:** inaccessible repository, invalid ref/path, binary/oversized file, symlink/outside-root fixture and cancellation. Never point this test at a developer's real home directory or credentials.

### C-08 — Administrator account, role and quota lifecycle

**Goal:** Administration changes the intended permissions without leaking authority. **Modes:** D, U. **Needs:** dedicated Admin, Alice/Bob and a disposable additional account.

1. Sign in normally as Admin, create the disposable ordinary user and log in as that user → account creation, role and actual access agree.
2. Edit a scoped permission/organization membership through supported controls → allowed action succeeds and a valid forbidden action fails; repeat after reload and revocation.
3. Attempt admin navigation and a canonical admin API action as Alice → clear denial with no privileged metadata or side effect. Use a normally authenticated test client for API controls, not browser-token extraction.
4. Create/rotate/revoke a disposable API key where offered → new key works in its intended scope; revoked key fails; secret is shown only according to the documented lifecycle.
5. Set a small test quota and upload below then above it as ordinary user and Admin → actual documented policy is enforced; a quota error is understandable and doesn't create an orphaned source/job. Fresh PostgreSQL schema must initialize through normal setup.
6. Disable/delete the disposable account and check session/access consequences → behavior matches the documented policy and audit/confirmation feedback.

**Recovery/UX:** duplicate user, invalid email/password, denied account action, expired admin session and partial role update. Do not grant an ordinary test user Admin privileges to bypass a failed ordinary-user case. Single-user cells run applicable owner/operator controls; multi-user-only role cases are explicit N/A.

### C-09 — Health, backup/restore and maintenance recovery

**Goal:** Operators can diagnose and recover a deployment without false success. **Modes:** D, U; L for external storage adapters when advertised. **Needs:** run-owned saved data and a separate restore target.

1. Open health/monitoring/data-ops → actual service readiness is readable; offline/unsupported services are distinct from healthy zero values.
2. Create a supported backup/export → wait for actual completion, download/read its manifest and retain hash; verify declared database/content coverage.
3. Restore into the **separate isolated target**, not the active test cell → normal login and representative Chat/Note/media/card reads match baseline identities/content according to the backup contract.
4. Enter maintenance mode in a disposable deployment → ordinary user actions show the intended restrictions; Admin can exit and service recovers without data reset.
5. Trigger one safe test operation/repair preview and inspect history → target/scope is clear; actual result and failure are distinguishable. Destructive operations use only disposable recorded resources.

**Recovery/UX:** corrupt/incompatible backup, insufficient storage, interrupted restore, failed operation and stale health. A downloaded archive or visible backup button alone does not prove restore works. Preserve the original cell throughout.

### C-10 — Model/runtime administration to real inference

**Goal:** Runtime controls and model readiness correspond to usable inference. **Modes:** D, L, U. **Needs:** one declared locally managed model/runtime and bounded resources.

1. Open the applicable model administration page and inspect installed/available models → sizes, status and provider mapping are meaningful; loading and empty states are distinct.
2. Select/start a preapproved test model within the run budget → readiness changes only when the actual runtime is usable; record model/runtime versions.
3. Choose that model in normal Chat and send the fixture question → a real response uses it; admin health alone is not enough.
4. Stop/restart only that owned runtime → normal Chat shows a recoverable unavailable state and succeeds after recovery, preserving the failed turn correctly.
5. Change a supported runtime setting, save/reload and verify its observable effect → displayed configuration and actual process behavior agree.

**Recovery/UX:** model missing, insufficient memory/storage, invalid configuration and failed start. Do not download arbitrary large models or kill unrelated processes. Unsupported hardware/runtime variants remain untested rather than inferred from another backend.

### C-11 — MCP discovery and scoped tool execution

**Goal:** Configured tools can be discovered and used within granted scope. **Modes:** D, L, U. **Needs:** controlled MCP server with harmless echo/read tool; actual advertised integration for L.

1. Configure/connect the server profile → status, tool names/schemas and trust context load; invalid configuration gives actionable guidance.
2. Execute an echo/read of F-REPO's public README with valid arguments → actual tool result matches expected content and is linked to the invoking account/session.
3. Submit invalid arguments and attempt an ungranted tool as Alice → validation/authorization deny the operation without side effects or foreign data.
4. Disconnect the MCP server mid-request, reconnect and retry where offered → truthful failure, preserved input and no duplicate side effect.
5. Save/reload the profile, then revoke access → discovery/execution follows the new scope; secrets are masked and absent from exports/logs.

**Recovery/UX:** empty tool list, stale capabilities, unavailable transport, cancellation and permission prompt focus. An app-level fake tool response does not prove MCP transport or authentication.

### C-12 — Workflow/scheduled task to monitored result

**Goal:** A saved automation runs the intended steps and handles partial failure. **Modes:** D with real orchestration, L for live provider steps, U. **Needs:** disposable workflow, controlled inputs and enabled workers.

1. Create the minimum supported workflow with two dependent steps and a named input → validation catches missing inputs; saved definition reopens unchanged.
2. Run it manually → one execution ID links each step's input/output, ordered dependency and terminal result; a downstream step does not run before its prerequisite succeeds.
3. Fail the first step in a separate case → dependent work is blocked/failed honestly; supported retry resumes according to the documented idempotency policy without duplicating completed side effects.
4. Create a short supported schedule and observe one real due execution → scheduled result links back to the definition; manual Run is not substituted for schedule evidence.
5. Pause/disable the schedule and cancel a delayed execution → no unexpected future run and clear cancellation status; remove only the test automation during cleanup.

**Recovery/UX:** worker restart, missing handler, invalid schedule/time zone and simultaneous manual/scheduled run. Record the actual Jobs/Scheduler backend and worker; do not infer execution from a saved definition alone.

## Cross-cutting release workflows

### X-01 — Upgrade an existing installation without losing work

**Goal:** Supported upgrades preserve real user data and a usable recovery path. **Modes:** D, L for representative post-upgrade provider actions, U. **Needs:** prior-release installation and separate restore environment.

1. On each declared supported starting release/cell, create representative sources/chunks, analysis versions, Notes, Chats, prompts, characters, cards/reviews and multi-user permissions using that release's supported paths → save an inventory of IDs, content hashes, counts and settings.
2. Back up the old deployment and prove restoration in a separate target → recovery baseline is usable before upgrade.
3. Follow the published upgrade/migration procedure to the candidate built artifacts → migration has an explicit outcome; no silently skipped store or elevated runtime role workaround.
4. Login/reload, open each saved artifact and run a new Chat/source/Note/card operation → old data remains correct, new writes work and scoped permissions still hold. Recheck PostgreSQL content stores under the intended runtime role.
5. Simulate an interrupted/failed upgrade only in a disposable clone → documented recovery either resumes safely or restores the backup. Do not invent a downgrade guarantee when migrations are irreversible.

**Recovery/UX:** old browser assets/storage, changed routes, malformed legacy records and settings migration. Report fresh-install and upgrade results separately; neither implies the other.

### X-02 — Reciprocal user, organization and browser-state isolation

**Goal:** Each account can access its own work without inheriting another account's data. **Modes:** D, L for retrieval/Chat ownership, U. **Needs:** multi-user cell; ordinary Alice/Bob; Admin; F-PRIVATE.

1. Alice creates one private source, Note, Chat, card, draft/handoff and job; Bob creates distinct equivalents including F-PRIVATE → each actor has a working own-resource positive read/write control.
2. As Alice, use valid Bob resource IDs through normal UI deep links and separately authenticated supported API clients → no body, title, snippet, counts, export, asset or job progress leaks; denials follow the endpoint's documented403/404 contract. Repeat Bob→Alice.
3. Perform Alice logout→Bob login→reload→Back in the same browser → lists, unsaved drafts, source handoffs, helper panels, autocomplete and notifications contain no Alice-private state. Test a separate context as well; two tabs share storage and are not independent identities.
4. Ask Alice's retrieval/Chat paths about Bob's private phrase without supplying it → no private retrieval result or answer; Bob's own positive read retrieves it. Check source candidates/citations, not just final wording.
5. Share one declared resource/organization role, exercise the permitted scope, then revoke → allowed work stays usable, forbidden mutations/metadata remain denied, revocation takes effect according to the documented cache/session contract.
6. Admin inspects allowed operational metadata → ordinary users remain ordinary; admin success cannot replace either direction of the tenant test.

**Recovery/UX:** account switch during fetch/generation, old tabs, same-name resources, exported artifacts and background jobs. If ingestion/quotas block Media/job creation, those isolation components are BLOCKED even if Notes/Chats pass. Single-user cells are N/A for reciprocal accounts, with connection-state behavior covered by A-02.

### X-03 — Usability, accessibility and responsive review

**Goal:** Every selected workflow is understandable and operable on its declared surfaces. **Modes:** automated checks plus human U review. **Needs:** fixed browser/viewport/theme list and representative empty/loading/error/populated states.

1. Follow the workflow from the normal navigation/empty state without pasted internal IDs → the next action and current scope are understandable; contextual handoffs reach the correct destination.
2. Complete the main action with keyboard only → visible focus, meaningful accessible labels, usable dialogs, Escape/cancel and focus return; no keyboard trap or hidden required control. Run the repository's accessibility checks and triage findings.
3. Review at the release's desktop widths, a narrow mobile width and 200% zoom → essential text/actions remain available without overlap/clipping; selected tabs, sticky panels and dialogs fit. Example starting viewports:1440×900,1024×768,390×844; pin actual values in the manifest.
4. Inspect empty, loading, slow, failed and completed states → loading does not resemble empty data; progress/counts/grammar are correct; duplicate toasts and raw transport traces do not replace actionable feedback.
5. Test long names, multiline/non-ASCII text, dark/light themes and an advertised non-English locale → readable wrapping, stable layout, meaningful plural/date/time formatting and no untranslated internal keys.
6. Review the generated answer/audio/artifact → intended facts are correct and useful; controls and labels explain how to continue, correct or undo work.

Record **functional result** and **UX result** separately. A successful request with an unusable interface is not an overall pass. Accessibility tooling cannot certify all usability; a human review cannot substitute for persistence/permission evidence. These checks are a project acceptance rubric, not a claim of formal accessibility certification.

### X-04 — Extension capture, sidepanel and cross-surface continuity

**Goal:** The shipped extension works with the same supported server without losing context. **Modes:** D, L, U. **Needs:** packaged extension at candidate version, controlled source page, selected text, appropriate host permissions.

1. Install the packaged extension in a fresh browser profile and connect/login normally → correct account/server; unsupported pages explain limitations without requesting unrelated permissions.
2. Select a Rowan passage and invoke the context-menu/sidepanel capture → editable Note or flashcard draft contains the exact selection and source URL. Save, open the full WebUI and verify the same canonical content/provenance.
3. Generate a small draft card batch, review/edit/save and perform a due-card review in the sidepanel → actual provider result, saved counts and one scheduling action; fuller WebUI controls remain reachable.
4. Chat about captured content, close/reopen the sidepanel and reload the source page → appropriate conversation/draft continuity without duplicated messages or stale source context.
5. Switch accounts and repeat WebUI↔extension entry → ownership, server settings and credential lifetime follow the documented cross-surface scope; no assumption that separate profiles share authentication.

**Recovery/UX:** restricted browser pages, no selection, denied permissions, offline server, expired session and narrow sidepanel. WebUI-only testing cannot certify extension storage, service-worker or permission behavior. Not shipped in a release profile means explicit N/A; a shipped but unavailable extension is BLOCKED.

## Additional shipped features

The historical tiers omit several current surfaces. The following supplemental workflows make their release inclusion explicit. Each **named variant** below becomes its own planned case with normal reload, failure/recovery, X-02 where applicable, X-03 and the default evidence/cleanup contract. Do not count a single selected variant as testing the whole row.

| ID / family                            | Repeatable user workflow and required result                                                                                                                                                                                                                                                                                                                                                   | Variants / dependencies                                                                                                                                                                                                                                                                                                                                                                               |
| -------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| S-01 — Quiz and study assistance       | From F-BIOLOGY create/review a quiz; take it with one intentional wrong answer; verify question-level scoring and saved attempt; generate reviewed remediation cards for the missed question; repeat conversion to verify documented deduplication; reload and study the linked cards. Open a study assistant, ask a question, create a two-client conflict and verify pending input recovery. | Quiz, remediation conversion, Study assistant. D/L/U; real model for generation; card/conversion IDs and counts required.                                                                                                                                                                                                                                                                             |
| S-02 — Chatbooks and portable backups  | Create a bundle containing one Chat, Note and source of each advertised supported type; export, inspect manifest and import into a separate account/profile through the supported mapping; verify text, links, assets and ownership. Retry a failed/corrupt import and ensure no false completed job or unexplained duplicate.                                                                 | Each supported advertised import format and full-account export. D/U; separate restore target; never assume API export equals UI roundtrip.                                                                                                                                                                                                                                                           |
| S-03 — Artifact editors                | Create a small artifact from a fixture, edit/reorder one item, save/reload, export and independently open the output; cancel a failing generation while preserving the saved artifact. Use exact cell/slide/card/text comparisons for deterministic edits.                                                                                                                                     | Data Tables:3rows×2columns; Presentations:3slides; Kanban:2lists/3cards with one move; Document Workspace:2page document with one annotation. Each variant D/L/U when generated; no title-only pass.                                                                                                                                                                                                  |
| S-04 — Research and source workspaces  | Create a research/project workspace, add F-SOURCE plus a controlled additional source, run a bounded query/brief, inspect citations and save a resulting Note/report; reload and verify project/source/result identities and user scope; retry a failed provider action without losing prior work.                                                                                             | Each advertised Research Workspace/connector/source-import path. D/L/U; real connector for L, controlled source for repeatability; no public web result guarantees.                                                                                                                                                                                                                                   |
| S-05 — Moderation and claims review    | Create one benign and one flagged synthetic item; inspect matched rule/reason/evidence, approve/reject or correct through the actual review controls, and verify only selected items change. Review a clearly supported and unsupported Rowan claim; save the decision and audit attribution; verify role denial and batch partial failure.                                                    | Moderation review, rules, claims review, specialized power-user route when shipped. D/U and L for model-derived classifications; fictional benign fixtures only.                                                                                                                                                                                                                                      |
| S-06 — Specialized tools and discovery | For each variant, select a supported input, execute its principal action, inspect a concrete result, save/export or follow its promised handoff, reload, then run one invalid-input/unavailable-dependency case. Pin the variant's expected output before execution.                                                                                                                           | Chunking:known text/exact coverage and valid boundaries; model playground:real answer + selected provider; skills:one declared safe skill action + result; domain research tools (researchers/journalists/OSINT):controlled public dataset + cited report; documentation:search + correct linked page; profile/companion:save identity/settings + actual advertised interaction. D/L/U as applicable. |

S-06 is a minimum contract, not a substitute for detailed acceptance of a newly changed subsystem. Before running a specialized variant, add its exact input, oracle, steps and source/test mapping to the release case manifest. If those cannot be defined, mark the variant **unplanned coverage** and block claims for it. Hosted billing, native companion voice and sandbox runtimes require their dedicated existing test plans in addition to this table when shipped.

## Automation mapping and implementation order

### Existing automation is a starting point

Paths and behavior below were inspected on 2026-09-17 against source revision `3189bb06fd38b8bd98a212a98dbadff7d6adfa3b`. These references identify reusable tests; they do **not** assert full coverage of the stricter workflows above. Check the candidate revision before reusing them.

| Playbook coverage    | Existing source / entry point                                                                                                                                                                                                                                                                                                                                   | Limitation to preserve in reports                                                                                                                                           |
| -------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| A-01/A-02/A-12       | [Onboarding UAT scenarios](../../apps/tldw-frontend/e2e/onboarding-uat/scenarios.ts), [setup happy path](../../apps/tldw-frontend/e2e/onboarding-uat/setup-happy-path.spec.ts), [cookie lifecycle](../../apps/tldw-frontend/e2e/single-user-cookie-lifecycle.spec.ts), [Settings](../../apps/tldw-frontend/e2e/workflows/tier-1-critical/settings-core.spec.ts) | Onboarding runner uses mock downstream inference and SQLite single-user profiles. Auth seeding elsewhere cannot certify fresh login, natural expiry or all four cells.      |
| A-03/A-04            | [Real Chat cockpit](../../apps/tldw-frontend/e2e/workflows/chat-cockpit.real-server.spec.ts)                                                                                                                                                                                                                                                                    | Focused command selects five cases; attachment, true hidden-state and successful vision coverage must be explicitly mapped.                                                 |
| A-05/A-07            | [Ingest/search/Chat journey](../../apps/tldw-frontend/e2e/workflows/journeys/ingest-search-chat.spec.ts), [shared real-server workflows](../../apps/test-utils/real-server-workflows.ts)                                                                                                                                                                        | Existing assertions/fixtures may be weaker than complete cited handoff; denied Wikipedia is not a substitute article.                                                       |
| A-06/A-09/B-09       | [Shared real-server wrapper](../../apps/tldw-frontend/e2e/real-server-workflows.spec.ts)                                                                                                                                                                                                                                                                        | Four active tests: Chat→Note, Chat→card, Media Trash, ingestion→analysis/reanalysis. Historical title constants are not registered test counts. Auth/onboarding are seeded. |
| A-08                 | [Notes](../../apps/tldw-frontend/e2e/workflows/tier-1-critical/notes.spec.ts)                                                                                                                                                                                                                                                                                   | Audit exact CRUD/export/conflict coverage and prerequisite creation before mapping a result.                                                                                |
| A-10/A-11            | [Notes→flashcards journey](../../apps/tldw-frontend/e2e/workflows/journeys/notes-flashcards.spec.ts), [Flashcards](../../apps/tldw-frontend/e2e/workflows/tier-2-features/flashcards.spec.ts)                                                                                                                                                                   | A positive count or one card is weaker than exactly five distinct generated/saved/reviewed cards, complete scheduling and analytics checks.                                 |
| B-01/B-02            | [Character management](../../apps/tldw-frontend/e2e/workflows/tier-2-features/characters.spec.ts), [Character journey](../../apps/tldw-frontend/e2e/workflows/journeys/character-chat.spec.ts), [readiness journey](../../apps/tldw-frontend/e2e/workflows/journeys/character-chat-phase7-readiness.spec.ts)                                                    | Skip/recovery branches do not certify actual `BEEP BOOP` or transitions.                                                                                                    |
| B-03                 | [World books](../../apps/tldw-frontend/e2e/workflows/world-books.spec.ts), [dictionaries](../../apps/tldw-frontend/e2e/workflows/dictionaries.spec.ts)                                                                                                                                                                                                          | Include the actual catalog reader, not only character-attached readers; verify context effect/scope separately.                                                             |
| B-04/B-05/B-06       | [TTS](../../apps/tldw-frontend/e2e/workflows/tier-2-features/tts-synthesis.spec.ts), [STT](../../apps/tldw-frontend/e2e/workflows/tier-2-features/stt-transcription.spec.ts), [Audio Studio](../../apps/tldw-frontend/e2e/workflows/tier-2-features/audio-studio.spec.ts)                                                                                       | Route/control/request checks are partial. Artifact decoding, listening/transcript quality, generation completion and export need explicit cases.                            |
| B-07                 | [Watchlist journey](../../apps/tldw-frontend/e2e/workflows/journeys/watchlist-ingest-notify.spec.ts), [watchlist items](../../apps/tldw-frontend/e2e/workflows/watchlists-items.spec.ts)                                                                                                                                                                        | Journey fulfills watchlist and notification APIs in the browser; it cannot certify scheduler/worker integration.                                                            |
| B-08                 | [Collections](../../apps/tldw-frontend/e2e/workflows/collections-stage3.spec.ts)                                                                                                                                                                                                                                                                                | Check real account/permission and source-membership coverage before reuse.                                                                                                  |
| B-09                 | [Content Review](../../apps/tldw-frontend/e2e/workflows/tier-2-features/content-review.spec.ts), [page object](../../apps/tldw-frontend/e2e/utils/page-objects/ContentReviewPage.ts)                                                                                                                                                                            | Existing tests can return when no draft exists or catch an absent API call. Such a result is not completed draft/commit acceptance.                                         |
| C-01/C-02            | [Prompt journey](../../apps/tldw-frontend/e2e/workflows/journeys/prompts-chat.spec.ts), [Prompts workspace](../../apps/tldw-frontend/e2e/workflows/tier-2-features/prompts-workspace.spec.ts)                                                                                                                                                                   | Add actual instruction application, template/variable revision and Studio-run outcomes; library CRUD alone is insufficient.                                                 |
| C-03                 | [Evaluations](../../apps/tldw-frontend/e2e/workflows/tier-2-features/evaluations.spec.ts), [ingest/evaluate/review journey](../../apps/tldw-frontend/e2e/workflows/journeys/ingest-evaluate-review.spec.ts)                                                                                                                                                     | A usable recipe validation outcome can still be a failed requested run; assert exact dataset/results.                                                                       |
| C-04/C-05            | [Agent registry](../../apps/tldw-frontend/e2e/workflows/tier-3-automation/agent-registry.spec.ts), [Agent tasks](../../apps/tldw-frontend/e2e/workflows/tier-3-automation/agent-tasks.spec.ts), [ACP](../../apps/tldw-frontend/e2e/workflows/tier-3-automation/acp-playground.spec.ts)                                                                          | Some cases mock APIs or accept empty/guarded states; actual task execution, permission effects and real adapter remain separate.                                            |
| C-06/C-07            | [Writing](../../apps/tldw-frontend/e2e/workflows/tier-2-features/writing-playground.spec.ts), [repo2txt](../../apps/tldw-frontend/e2e/workflows/tier-5-specialized/repo2txt.spec.ts)                                                                                                                                                                            | Request-fired/page-loaded evidence does not prove a correct saved/exported result.                                                                                          |
| C-08/C-09/C-10       | [Admin suite directory](../../apps/tldw-frontend/e2e/workflows/tier-4-admin), [local runtime administration](../../apps/tldw-frontend/e2e/workflows/llamacpp-runtime-admin.spec.ts)                                                                                                                                                                             | Guarded/skipped admin tests do not prove authorized operations, restore or inference.                                                                                       |
| C-11/C-12            | [MCP Hub](../../apps/tldw-frontend/e2e/workflows/tier-2-features/mcp-hub.spec.ts), [workflow editor](../../apps/tldw-frontend/e2e/workflows/tier-3-automation/workflow-editor.spec.ts), [Chat workflows](../../apps/tldw-frontend/e2e/workflows/tier-3-automation/chat-workflows.spec.ts)                                                                       | Complete worker/tool execution, permission and real-schedule variants need explicit mapping.                                                                                |
| X-01/X-02            | [Fresh UAT matrix](../Reviews/FRESH_INSTALL_UAT_MATRIX_2026_09_17.md), [running issue tracker](../Reviews/FRESH_INSTALL_SINGLE_MULTI_UAT_TRACKER_2026_09_14.md)                                                                                                                                                                                                 | Prior run evidence guides regression cases; it cannot certify a new build. Upgrade coverage and a durable four-cell runner still require implementation.                    |
| X-03/X-04/S families | [Accessibility routes](../../apps/tldw-frontend/e2e/smoke/stage4-axe-high-risk-routes.spec.ts), [current test project config](../../apps/tldw-frontend/playwright.config.ts), [feature guides](../User_Guides/WebUI/Study_Writing_Artifacts.md)                                                                                                                 | Route inventory and screenshot/axe checks are partial; map actual end-to-end variants and human review.                                                                     |

### Commands that already exist

These are command references, not a newly implemented release runner. Run from `apps/tldw-frontend` against the **explicitly prepared isolated target** with its documented environment/credentials. Inspect fixture seeding and downstream mocks before attributing coverage.

```sh
# Inventory the existing registered suites; does not certify execution.
bunx playwright test --list --project=tier-1 --project=tier-2 --project=tier-3 --project=tier-4 --project=tier-5 --project=journeys

# Existing numbered suites and journeys; excludes some root workflow specs.
bun run e2e:all-tiers

# Existing four shared real-server tests; separate from all-tiers.
bunx playwright test e2e/real-server-workflows.spec.ts --project=chromium --workers=1

# Existing bounded real Chat selection; package script checks skipped tests.
bun run e2e:chat-cockpit:real:focused
```

`e2e:critical` currently means numeric tier1 + journeys; `e2e:features` means tier2 + tier3; `e2e:admin` means tier4 + tier5. None is a literal A/B/C mapping. The `uat:live-tiers` runner launches a mock OpenAI service despite its name. The `e2e:onboarding:uat` runner also uses mock downstream inference. `test:integration` points to a helper with an `admin-ui` frontend path and is not this playbook's release runner.

### Catalog and receipt checks

The versioned [catalog](../../apps/tldw-frontend/scripts/live-tier-uat/release-catalog-data.mjs) records all 43 families and their named variants, assertions, evidence modes and applicability. A release plan must declare every deployment cell, surface, phase, upgrade starting point and selected browser, with a reason for each exclusion. Each applicable variant/mode/context needs an explicit disposition. An automated mapping must name its exact registered project, file and complete title path, plus the assertions it claims to cover and the rationale. Review those assertions in the test; matching a title does not prove behavioral coverage.

```sh
# Collection only: no application services or browser execution.
node scripts/list-release-tests.mjs > collection.json

# Validate a complete explicit plan against registrations; no execution claim.
node scripts/assert-release-uat.mjs --catalog planned-catalog.json collection.json

# Validate first-attempt results for one context's exact required case manifest.
node scripts/assert-release-uat.mjs required-cases.json playwright-results.json
```

Advertised format, source-type, workspace, connector and external-adapter variants use the catalog's six fixed instance inventories. Declare each named option in `scope.instances`; each expands into its own applicable context/mode/recovery requirements. An empty inventory requires an explicit exclusion or not-applicable reason. A single generic import test cannot silently stand for several declared formats.

The catalog validator reports unique executions separately from mapped requirements, so a shared test cannot inflate execution counts. Human UX review is separate and can remain explicitly pending in the frozen plan. Unknown, omitted, duplicate or unregistered mappings fail validation; exclusions stay visible. These commands neither create missing workflow tests nor independently verify runtime artifacts. A complete production runner and actual candidate-bound evidence are still required before release certification. The existing `uat:live-tiers` development runner now labels its results diagnostic, compares exact collected cases and preserves individual attempts, including readable partial reports after cancellation.

Use [package.json](../../apps/tldw-frontend/package.json) and [Playwright config](../../apps/tldw-frontend/playwright.config.ts) as the command authority. The current config can start/reuse a frontend server; disable auto-start or provide an explicit owned server configuration for release runs so an unrelated listener cannot be tested accidentally. Never treat `--allow-skips`, an empty selection or a narrowed grep as complete coverage.

### Smallest useful automation implementation

This is follow-on work, not implemented by this document:

1. **Manifest and fixtures:** Check in the versioned binary fixture pack and controlled feed/provider/agent contracts. Add scenario definitions keyed by the stable IDs here. Keep setup facts, expected steps/variants and outcomes machine-readable; do not build a fragile parser that tries to infer actions from prose.
2. **Isolated orchestration:** Reuse reviewed setup/fixture mechanisms for the four database/auth cells; build/start candidate artifacts; record real runtime store/role/provider facts. Add supported upgrade targets and deterministic teardown. Never hard-code paths, PIDs, credentials or historical fixture IDs from a previous UAT.
3. **Tier A first:** Reuse page objects and real-server test helpers; replace silent skips/early returns in selected acceptance cases with explicit case results. Implement linked journeys and original-failure regressions before broadening the catalog.
4. **Tier B/C and supplemental variants:** Add real worker/artifact end conditions, negative controls and parameterized provider profiles. Run natural expiry/scheduled waiting alongside independent work without altering time or shared services.
5. **Evidence and release reporting:** Emit per-step JSON plus readable Markdown/HTML reports, store artifacts by immutable run/case/attempt, reconcile expected versus actual cases, and leave UX review/signoff explicitly pending until completed.

Useful scenario tags: `workflow:A-07`, `tier:A`, `cell:pg-multi`, `phase:fresh`, `mode:live`, `variant:loaded-handoff`, `requires:retrieval`, `actor:alice`. Map every test to the specific steps/variants it proves. Reusing one shared test across several workflows is acceptable only when it satisfies each named assertion; it cannot inflate executed-case counts.

## Results, issue tracking and release decisions

### Outcome vocabulary

| Status         | Meaning                                                                                                                                                             |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| PASS           | All selected steps, variants and required evidence modes meet their oracles on the recorded revision.                                                               |
| FAIL           | An observed product/integration/UX outcome differs from the expected result. Classify the cause separately; do not silently downgrade a provider failure into PASS. |
| BLOCKED        | A required prerequisite/environment/capability is unavailable, or an upstream failure prevents this case. Include blocker ID/reason and owner.                      |
| NOT_RUN        | Planned case not executed, missing evidence, harness interruption or unfinished review. A harness error is not automatically a product bug.                         |
| NOT_APPLICABLE | Feature/case is outside the predeclared supported deployment scope, with an explicit reason. Missing test data/provider is not N/A.                                 |
| PARTIAL        | Aggregate display only: some components pass and others fail/block/remain unrun. Keep the underlying case outcomes; never treat PARTIAL as PASS.                    |

A case may have `functional_status: PASS` and `ux_status: FAIL`; its overall result is FAIL. If a required mode is not run, overall cannot be PASS. Expected negative tests pass only when the exact denial/recovery oracle holds and the positive control demonstrates the feature works when authorized/available.

### Machine-readable report contract

Use these fields in a future runner. They are a specification, not a claim that the current scripts already emit them.

| Record       | Required fields                                                                                                                                                                                                                       |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Run          | `schema_version`, `run_id`, spec hash, source/artifact hashes, supported-release scope, start/end UTC, actual database/auth/store matrix, browser/viewport, reused dependencies, capability/provider versions, budgets, planned cases |
| Case         | immutable `case_id`, workflow ID/version, variant, cell, fresh/upgrade phase, actor role/actual non-secret ID, required modes, dependency case IDs, fixture IDs/hashes, status and reason                                             |
| Step/attempt | ordinal, action, expected oracle, observed result, start/end/duration, safe request/status correlation, canonical resource IDs, mode, attempt number, original failure reference                                                      |
| Evidence     | relative artifact path, media type, SHA-256, redaction review result, scope/limitations; never raw credentials/headers or entire private browser profiles                                                                             |
| Finding      | issue ID, case/step, severity, functional/UX category, expected/actual, reproduction, affected cells/versions, source/provider attribution confidence, owner, repair/retest evidence                                                  |
| Decision     | coverage totals, unresolved findings, approved exceptions, independent UX review, release owner/date, release/reject decision and exact artifact hashes                                                                               |

**Illustrative unexecuted case record** (all `null` values must be filled or explained before acceptance):

```json
{
  "schema_version": 1,
  "run_id": "example-release-run",
  "case_id": "A-07.loaded-handoff.pg-multi.fresh.live.alice",
  "workflow_id": "A-07",
  "workflow_version": "1.0",
  "variant": "loaded-handoff",
  "cell": "pg-multi",
  "phase": "fresh",
  "actor": { "role": "ordinary_user", "id": null },
  "required_modes": ["L", "U"],
  "dependency_case_ids": ["A-05.txt-ingest.pg-multi.fresh.live.alice"],
  "fixture_ids": ["F-SOURCE", "F-DISTRACTOR"],
  "fixture_hashes": {},
  "source_commit": null,
  "started_at": null,
  "finished_at": null,
  "functional_status": "NOT_RUN",
  "ux_status": "NOT_RUN",
  "overall_status": "NOT_RUN",
  "steps": [],
  "attempts": [],
  "resource_ids": {},
  "evidence": [],
  "issue_ids": [],
  "limitations": ["Example only; no execution evidence."]
}
```

Use the same immutable case identity for retries, with a new attempt number; a changed candidate revision belongs to a new run. Preserve first failures. A retry pass is reported as recovery/flaky history, not a clean first-attempt pass. Intentional user Retry is itself a separate scenario whose initial failure is part of its expected test setup.

### Required readable report

Publish a summary table with columns:

```text
Workflow | Variant | Cell | Fresh/Upgrade | Mode | Functional | UX | Overall | Issue | Evidence
```

Include totals for planned, passed, failed, blocked, not run and justified N/A; count unique case IDs, not API requests or retry attempts. State the denominator explicitly: `pass rate = PASS / (planned - justified N/A)`. It is informational, not the release gate. A 99% pass rate can still contain a critical data-isolation failure.

Reconcile the planned manifest against actual results before publishing. Missing/duplicate case IDs, zero tests, silent early returns, unaccounted skips, stale artifacts and results from a different source revision are report failures. Do not fill an unexecuted cell with an earlier run's result.

### Running issue tracker

For each observed bug, failure or UX problem:

1. Search existing findings/tasks for the same cause/scope; record a new reproduction against that issue when appropriate. A different untested code path can be a separate finding even if a related repair previously passed.
2. Record concrete trigger, expected/actual, artifact/step, affected cell, severity, ownership and blocked descendants. Keep environment/harness failures separate, with any user-facing consequence retained.
3. Associate code/documentation changes with Backlog tasks per repository policy. Link plans, fixes and verification rather than rewriting old evidence.
4. Close only after the original scenario passes on the fixed revision with the necessary regression/negative controls. State the verified scope; a bounded test does not prove an entire feature.

### Proposed release gate

These are the playbook's default decision rules; the release owner adopts or amends them **before** execution.

- No untriaged failures, missing case results or missing required live/UX evidence.
- No open P0/P1 findings: data loss/leak, unauthorized action, broken startup/login, or unusable advertised core workflow. Severity follows actual impact, not A/B/C frequency.
- Required cases pass across their supported cells and upgrade paths. BLOCKED/NOT_RUN is not acceptance; restore the dependency and rerun, or explicitly remove the affected capability from the release scope and validate the resulting disabled/limited UX.
- P2 exceptions require a named release owner, user-visible impact, bounded workaround, linked repair task and deadline. P3 cosmetic issues may remain tracked with explicit disposition. Exceptions remain visible; they do not relabel failed cases PASS.
- After repairs, run the affected original cases and dependent journeys on one final frozen candidate. Run the declared complete release selection after shared infrastructure/auth/persistence changes; do not combine different repaired commits into an imaginary passing build.
- Attach the production artifact hashes, database/auth coverage, dependency/provider limits, UX reviewer and release decision. Completion of this process is evidence for a release decision, not permission for an agent to publish or merge.

### Evidence retention and cleanup

Store artifacts under a unique run directory such as `output/playwright/release-uat/<run_id>/<case_id>/<attempt>/`. Keep safe selected request bodies and canonical responses, screenshots/trace excerpts, final model output, downloaded artifacts and an SHA-256 manifest. Retain minimal evidence that actually supports the claim.

Review traces, screenshots, exports and logs for credentials and unrelated/private content before sharing. Use a file allowlist and secret-pattern/known-value scans; never publish an entire browser profile or private runtime directory. Record redaction and lossless compression. Keep an immutable failure packet and a separate repair packet.

Cleanup verifies resource ownership first, disables only test schedules, stops only test-owned processes and releases PostgreSQL fixtures through their official lifecycle. Preserve failed data until diagnosis/retest requirements are met. A cleanup failure is visible in the report and must not trigger broader deletion of unrelated resources.

## Maintenance and source references

### Updating this playbook

- Keep workflow IDs stable. Add a versioned variant for a new behavior; retire obsolete cases with a reason rather than reusing their IDs for a different feature.
- For each changed advertised feature, update its workflow/oracle, fixture version, automation mapping and UX states in the same change. A new route alone is not a complete workflow.
- Keep one source of truth for fixture data and expected facts once implemented; generated runner definitions must refer to the same versioned catalog rather than duplicating prose and drifting.
- Turn accepted UAT findings into explicit regression variants. Preserve the old failing evidence and note the fixed candidate/verified scope. The current run's issue counts and pass/fail state belong in its tracker, not in this reusable playbook.
- Review external provider changes and actual capability contracts at each release. Pin the tested model/adapter versions and record limits; do not silently reinterpret a blocked capability as optional after the run begins.

### Source hierarchy and provenance

Use the release's documented product contract and observable supported UI first, corroborated by current implementation and tests. Historical plans explain intent but may describe superseded routes or behavior. Conflicts become explicit decisions/issues; never change an expected outcome merely to match a bug.

- [Historical A/B/C coverage design](../Plans/2026-03-12-e2e-test-coverage-expansion-design.md): original 15 feature-family mapping; not a current exhaustive feature inventory.
- [Current frontend testing guide](../../apps/Testing_Guide.md), [Playwright configuration](../../apps/tldw-frontend/playwright.config.ts), [package commands](../../apps/tldw-frontend/package.json): actual execution entry points; inspect version drift.
- [User fixture helpers](../../apps/tldw-frontend/e2e/utils/helpers.ts), [test fixtures](../../apps/tldw-frontend/e2e/utils/fixtures.ts): auth seeding, readiness and skip boundaries.
- [Live-tier runner](../../apps/tldw-frontend/scripts/live-tier-uat/run.mjs), [onboarding runner](../../apps/tldw-frontend/scripts/onboarding-uat/run.mjs): downstream mock boundaries that must remain explicit.
- [Flashcards Study Guide](../User_Guides/WebUI_Extension/Flashcards_Study_Guide.md): scheduler, ratings, card assets and extension capture contracts.
- [Study, Writing and Artifacts](../User_Guides/WebUI/Study_Writing_Artifacts.md), [Audio and Speech](../User_Guides/WebUI/Audio_Speech_Audiobooks.md), [Automation/Admin](../User_Guides/WebUI/Automation_Admin_Operations.md): feature scope. Current Audio Studio test/page object takes precedence over older audiobook route descriptions.
- [Evaluation User Guide](../User_Guides/Server/Evaluations_User_Guide.md), [ACP setup guide](../User_Guides/Integrations_Experiments/Getting_Started_with_ACP.md): capability prerequisites and documented output semantics.
- [2026-09-17 native matrix](../Reviews/FRESH_INSTALL_UAT_MATRIX_2026_09_17.md), [running UAT tracker](../Reviews/FRESH_INSTALL_SINGLE_MULTI_UAT_TRACKER_2026_09_14.md): source for failure-class regressions and evidence lessons; not acceptance for future releases.

**Authoring verification:** Check unique IDs, complete A/B/C family coverage, local references, JSON examples, required workflow fields, current command names and accidental placeholders before finalizing revisions. Documentation-only validation does not run the workflows or constitute product acceptance.
