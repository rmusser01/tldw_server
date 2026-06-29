# Pre-Main UAT Matrix Design

Date: 2026-06-29

Task: TASK-12062

PR under test: rmusser01/tldw_server#1982 (`dev` -> `main`)

## Purpose

Before PR #1982 is merged to `main`, run a focused user acceptance test that
proves the release candidate works for both basic and advanced users across the
supported single-user setup paths. The pass should find bugs, classify them,
fix valid issues with minimal patches, and preserve enough evidence to justify
the final merge decision.

This is not a full audit of every WebUI route. It is a release-gate journey
matrix built around the workflows most likely to block real use after the merge.

## Release Gates

The UAT is blocking if any of these fail without a documented, approved
exception:

- Docker single-user + WebUI can start, authenticate, reach the API, and run the
  basic and advanced UAT journeys against isolated disposable data.
- Local single-user + WebUI can start, authenticate, reach the API, and run the
  basic and advanced UAT journeys against isolated disposable data.
- OpenAI can complete the provider preflight and the required live answer paths.
- `llama.cpp` at `127.0.0.1:9099` can complete the provider preflight and the
  required live answer paths.
- The basic user can ingest a disposable document, search or ask about it, and
  use roleplay-focused character chat features.
- The advanced user can run the power knowledge workflow, including advanced
  controls, evidence/citations, note handoff or equivalent review output, and
  destructive operations on UAT-created data only.

## Matrix Control

The matrix must avoid uncontrolled combinatorial expansion. Coverage is bounded
as follows:

- Run the full basic and advanced journeys on both Docker single-user and local
  single-user environments at desktop viewport.
- Run mobile checks only on the highest-risk screens and controls:
  onboarding/first-value entry, document/media search, Knowledge QA answer and
  evidence controls, character chat creation/selection, and character chat
  conversation.
- Require both OpenAI and `llama.cpp` to pass provider preflight in both
  environments where configuration is expected to be visible.
- Require both providers to pass the core answer path and character-chat answer
  path at least once per environment. Do not duplicate every advanced UI
  substep for every provider unless a provider-specific issue is suspected.

## Personas And Journeys

### Basic User: First Value Plus Roleplay

Goal: prove a new or casual user can get value without understanding the whole
system.

Steps:

1. Start from a clean isolated profile.
2. Confirm the app shows a coherent entry path and does not expose stale local
   data.
3. Ingest a small disposable document with a unique UAT run id in the title,
   content, and tags.
4. Verify the document appears in media/search surfaces.
5. Ask a simple question about the document through the normal app path and
   verify a live provider answer references the disposable content.
6. Create or import a disposable roleplay character.
7. Start a character chat and verify roleplay-oriented affordances:
   character selection, persona/role metadata, initial prompt/context handling,
   message send, provider response, persistence after navigation/reload, and
   recovery/error messaging.
8. Exercise mobile usability on the first-value and character-chat screens.
9. Clean up UAT-created data where product flows support cleanup.

### Advanced User: Power Knowledge Workflow

Goal: prove a power user can inspect, constrain, and reuse knowledge results.

Steps:

1. Create a separate disposable dataset with multiple small documents, tags, and
   content distinctions that can be filtered.
2. Exercise media search and advanced filters.
3. Open Knowledge QA and select or constrain sources.
4. Ask targeted questions that should produce short deterministic answers.
5. Verify evidence/citation/source controls are visible and coherent.
6. Use note handoff, export, review, or the closest available downstream
   workflow for the generated result.
7. Exercise destructive operations only on UAT-created objects:
   delete/trash/restore, clear test outputs, remove test character or notes, and
   verify cleanup behavior.
8. Check keyboard flow and mobile behavior for critical controls.

## Provider Handling

OpenAI and `llama.cpp` are both live blocking gates. The UAT should use short,
deterministic prompts such as "answer with one sentence" or "include the token
`uat-<run-id>` if the source contains it" to reduce provider nondeterminism.

Provider failures must be classified before fixing:

- Configuration or reachability failure.
- App request/streaming failure.
- Empty, malformed, or unusable model response.
- Provider-specific behavior that the UI fails to explain.
- UX-only problem around provider failure, retry, or model selection.

If a provider is down or credentials are unavailable, the UAT stops and records a
blocking provider/setup defect rather than silently falling back to mocks.

## Environment Isolation

The UAT must not inspect or mutate existing user data.

Before running each environment:

- Record commit SHA, branch, environment type, API URL, WebUI URL, auth mode,
  and provider target.
- Record the runtime database/config paths that will be used.
- Use a unique run id for every disposable title, tag, source name, note,
  character, chat, and artifact.
- Prefer fresh runtime directories or container volumes. If a fully fresh store
  is unavailable, verify that all mutations are scoped by the run id.

After running each environment:

- Attempt cleanup through product-supported flows.
- Verify cleanup only targeted UAT-created objects.
- Record any cleanup failure as a finding.

## Evidence

Store UAT evidence under:

`Docs/Product/WebUI/evidence/pre_main_uat/<run-id>/`

Each matrix slice should record:

- Environment: Docker or local, provider, viewport, commit SHA.
- Persona and scenario.
- Steps performed and pass/fail status.
- Screenshots for important UI states and all failures.
- Console errors, page errors, failed API requests, and relevant backend log
  excerpts with secrets redacted.
- Created data identifiers and cleanup result.
- Reproduction steps for each finding.

The final report should include a concise issue table with one status per
finding:

- Fixed.
- Skipped with reason.
- Deferred with reason and explicit approval requirement if it affects release.

## Severity And Fix Policy

Findings are classified as:

- P0 blocker: startup, auth, provider use, document ingest, first answer, or
  character chat cannot proceed.
- P1 release blocker: core workflow succeeds only with broken UX, missing
  persistence, corrupted data, unsafe behavior, or major provider inconsistency.
- P2 fix before main if practical: valid workflow issue with clear impact and a
  workaround.
- P3 document or defer: polish, copy, low-frequency edge, or non-blocking
  inconsistency.

The user requested fixing all valid findings. Apply that as:

- P0-P2 findings are fixed in this workstream unless the fix would be large or
  risky enough to reshape release behavior.
- P3 findings are fixed when the patch is small and safe; otherwise they are
  documented.
- Stop and ask before making a large, risky, or architecture-shaping change even
  if the finding is valid.

## Fix And Verification Loop

For each valid issue:

1. Verify it against current code and current UAT evidence.
2. Create or update Backlog tracking before editing files.
3. Patch minimally using existing patterns.
4. Add or run focused tests when applicable.
5. Rerun the affected UAT slice.
6. Rerun enough of the matrix or smoke suite to ensure no regression.
7. Update the evidence report with fixed, skipped, or deferred status.

## Existing Harnesses To Reuse

Prefer existing app harnesses where they match the UAT intent:

- `apps/tldw-frontend/scripts/onboarding-uat/run.mjs`
- `apps/tldw-frontend/scripts/chat-uat-driver.mjs`
- `apps/tldw-frontend/scripts/media-uat-driver.mjs`
- `apps/tldw-frontend/scripts/chars-uat-driver.mjs`
- `apps/tldw-frontend/scripts/media-multi-uat-driver.mjs`
- `apps/tldw-frontend/scripts/research-workspace-uat-runner.mjs`
- Targeted Playwright specs under `apps/tldw-frontend/e2e/workflows/` and
  `apps/tldw-frontend/e2e/smoke/`

Manual or custom browser probes are acceptable for gaps, but they should produce
the same evidence shape as the existing drivers.

## Out Of Scope

- Exhaustive route-by-route UX review.
- Non-single-user deployment UAT.
- Browser extension UAT unless a WebUI finding points to shared UI parity risk.
- Mutating existing user data.
- Replacing live provider gates with mocked responses.
- Broad refactors unrelated to confirmed UAT findings.

## Completion Criteria

The UAT workstream is complete when:

- The design and implementation plan are committed.
- Docker and local environments have completed the bounded UAT matrix.
- OpenAI and `llama.cpp` blocking gates have passed or have explicit approved
  exceptions.
- Valid findings are fixed or explicitly documented as skipped/deferred with
  reason.
- Evidence artifacts and final report are committed.
- Final verification commands and matrix pass/fail results are recorded.
