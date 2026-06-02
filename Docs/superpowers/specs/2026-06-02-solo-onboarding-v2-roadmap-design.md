# Solo Onboarding V2 Roadmap Design

Date: 2026-06-02
Status: Draft for review
Backlog: TASK-504
Reference: PR #2214, onboarding confidence flow

## Summary

Solo onboarding now has a coherent first-run journey from setup path selection through provider validation, first chat, and a guided first-source milestone. The next work should improve confidence, recovery, and first value through four sequential PRs:

1. Repeatable onboarding UAT harness.
2. Guided diagnostics and recovery.
3. First-value starter questions.
4. Local model guided alternative V2.

The harness ships first. Every later feature PR must extend the harness before implementing product behavior, so onboarding changes are validated against repeatable end-to-end user journeys instead of one-off manual walkthroughs.

## Goals

- Make solo onboarding easier to validate repeatedly from a clean installation state.
- Preserve the merged first-run architecture: backend-authoritative setup/readiness/completion, frontend-local transient UI state.
- Use the repository mock OpenAI-compatible API server for provider behavior during UAT.
- Capture enough evidence to debug failures without immediately rerunning a full walkthrough.
- Improve user recovery for setup, provider, first-chat, and first-source failures.
- Turn first-source success into immediate user value.
- Treat hosted providers and local OpenAI-compatible providers as peer setup choices while keeping local runtime installation outside app ownership.

## Non-Goals

- Do not replace the existing unit, integration, or Playwright smoke suites.
- Do not make the new UAT harness a blocking CI gate in its first PR.
- Do not use Playwright route mocks to fake provider behavior in the UAT harness.
- Do not call real hosted LLM APIs in default UAT runs.
- Do not mutate a developer's real `.env`, `config.txt`, databases, uploads, or model directories.
- Do not install or manage Ollama, llama.cpp, or other local runtimes.
- Do not make RAG, storage paths, STT, or TTS mandatory for first-chat completion.

## Roadmap

The roadmap should ship as one umbrella effort with four independent, reviewable PRs.

### PR1: Repeatable Onboarding UAT Harness

PR1 creates the manual/dev UAT command and evidence contract. It should be usable before CI adoption and should run from a clean isolated runtime profile.

Default process topology:

- `mock_openai_server` provides OpenAI-compatible chat, models, embeddings, streaming, and failure behavior.
- `tldw_Server_API` runs as the real backend against temp config, temp env, isolated databases, and isolated storage paths.
- Next WebUI runs as the real frontend.
- Playwright drives the browser against the WebUI.

Provider behavior must come from static `mock_openai_server` config files per scenario. The harness may still use ordinary Playwright helpers for auth seeding, viewport setup, screenshots, and UI interaction, but it must not route-intercept provider validation/chat behavior that should flow through the real backend and mock server.

Tier A scenarios for the first harness:

- Hosted OpenAI-style happy path against `mock_openai_server`.
- Local OpenAI-compatible happy path against `mock_openai_server`.
- First-source paste-text path.
- First-source file-upload path.
- First-source web-URL path using a local public fixture page under `apps/tldw-frontend/public/e2e/`.
- Desktop and mobile viewport coverage.
- Provider validation failure recovery.
- First-chat transient failure then retry success.
- Ingest failure then retry success.

Tier A is the required manual run for PR1. Later PRs can add Tier B scenarios for broader manual coverage, and a future CI transition can promote stable Tier A or Tier B scenarios into non-blocking or blocking CI jobs.

The first-source fixture should be a short structured research note with headings, bullets, dates, claims, and action items. It should be deterministic and meaningful enough for later starter-question and grounded-chat assertions.

Ingest failure scenarios should be deterministic. Prefer local fixture controls, isolated runtime configuration, or backend-supported test controls over external network failures or timing-sensitive races.

Artifacts per run:

- Screenshots for each scenario step.
- JSON summary with scenario id, viewport, result, durations, failure category, and artifact paths.
- Backend log.
- Frontend log.
- Mock OpenAI server log.
- Browser console and network failure summary.

Pass criteria:

- Expected setup, chat, and source states are reached for every required scenario.
- Required artifacts are written.
- No critical browser console errors.
- No failed required API calls.
- No secret leakage in logs or artifacts.
- Any skipped optional branch is explicitly recorded in the JSON summary.

Recommended command shape:

- `bun run e2e:onboarding:uat` from `apps/tldw-frontend` for the manual harness.
- A harness runner script owns process startup, port selection, isolated temp runtime profile creation, cleanup, and artifact directory layout.
- A future non-blocking CI job can run the same command once the manual harness is stable.

### PR2: Guided Diagnostics And Recovery

PR2 starts by adding UAT scenarios for blocked and degraded setup states. Then it adds inline diagnostics and recovery actions that use backend setup/readiness/first-chat signals.

Harness-first scenarios:

- Provider auth failure from `mock_openai_server`.
- Model unavailable from mock `/v1/models` or chat response.
- Local endpoint unreachable.
- Config write failure or read-only runtime profile path.
- Backend readiness overlays such as restart-needed, network-unavailable, downloads-disabled, or package-installs-disabled.
- First-chat failure category maps to visible recovery actions.

Product behavior:

- Add a compact "what is blocking setup" diagnostic surface inside the focused setup shell.
- Each issue shows status, plain-language cause, why it blocks or does not block first chat, primary action, and optional secondary action.
- Recovery actions stay inline where possible: edit provider, switch provider, retry validation, retry first chat, check endpoint, open `/setup`, or skip setup when allowed.
- Frontend maps backend categories and overlays into clearer copy; it does not invent readiness conditions.
- Optional lanes remain visibly deferrable unless the user opted into them.

Acceptance criteria:

- Every supported failure category has specific copy and at least one valid next action.
- Recovery actions keep the user in the setup shell unless they explicitly choose `/setup`.
- Raw exception details, stack traces, secrets, request headers, and filesystem paths are not shown in primary UI.
- UAT artifacts prove each failure and recovery branch.

### PR3: First-Value Starter Questions

PR3 starts by extending UAT scenarios so the harness asserts first-source value behavior. Then it adds starter questions once source readiness is confirmed.

Harness-first scenarios:

- Paste-text first source becomes queryable and shows starter questions.
- File first source becomes queryable and shows starter questions.
- Web URL fixture source becomes queryable and shows starter questions.
- Source still processing does not show starter questions.
- Source ingest failure does not show starter questions.
- Clicking a starter question opens chat with the expected prompt and source context.

Product behavior:

- After first-source readiness is confirmed, show two or three starter questions.
- V1 questions should be safe templates, not generated questions:
  - "Summarize this source."
  - "List the key claims."
  - "What should I remember?"
- Surrounding copy can include the source title or label.
- Do not show starter questions until the source has a media id and readiness says it can be used.
- Route starter questions through the existing chat/source workflow.
- If grounded chat readiness cannot be guaranteed, show "Open source" or "View source" instead of claiming source chat is ready.

Acceptance criteria:

- Starter questions appear only after source readiness is confirmed.
- Clicking a starter question creates or opens the expected chat context.
- UAT summary records the clicked starter and whether chat received the expected prompt/context.
- The structured research-note fixture supports meaningful assertions.

### PR4: Local Model Guided Alternative V2

PR4 starts by adding local OpenAI-compatible setup scenarios to the UAT harness. Then it improves local endpoint guidance, validation, model discovery, manual fallback, and recovery copy.

Harness-first scenarios:

- Local OpenAI-compatible endpoint reachable, `/v1/models` succeeds, user picks a discovered model, first chat succeeds.
- Local endpoint reachable but `/v1/models` unsupported or fails, user manually enters a model, first chat succeeds.
- Local endpoint unreachable, UI shows endpoint recovery copy and retry.
- Local endpoint reachable but model unavailable, UI asks user to pick or enter another model.
- Local provider selected as default and later switched back to hosted provider without stale validation state.
- Mobile local setup remains usable.

Product behavior:

- Local install remains a peer setup-path choice.
- The app guides and verifies; it does not install or manage local runtimes.
- Endpoint input explains expected OpenAI-compatible base URLs and common examples.
- Validation prefers non-generative checks: health if supported, `/v1/models` where supported, and API shape checks where model listing is unavailable.
- Model discovery populates a picker while preserving manual model entry.
- Failure copy is practical: service not running, wrong host or port, missing `/v1`, model not pulled or loaded, auth/token required.
- Recovery buttons are explicit: retry, edit endpoint, switch provider, continue with manual model when allowed, and open local setup guide/docs.

Acceptance criteria:

- Local setup completes through discovered-model and manual-model paths.
- Hosted and local validation state cannot contaminate each other.
- UAT artifacts prove local success and local recovery cases.
- No installation ownership is implied.

## Shared Harness Contracts

The harness should use isolated runtime profiles. Each run gets temp config, temp env, temp databases, temp uploads/storage, temp logs, selected mock-server config, and an artifact directory. Cleanup should remove temp runtime data by default while preserving artifacts.

Secrets:

- Use synthetic keys only, such as `sk-uat-mock-openai`.
- Redact API keys, auth headers, `.env` values, and config secrets from logs and summaries.
- Fail the run if a synthetic secret appears unredacted in captured artifacts.

Mock server configs:

- Keep static configs under a clear onboarding UAT fixture directory.
- Name configs by scenario, for example hosted success, local success, auth failure, transient chat failure, model unavailable.
- Prefer deterministic responses and bounded delays.

Evidence layout:

- Use a stable artifact root such as `Docs/Product/WebUI/evidence/onboarding_uat/<run-id>/` for human-reviewed runs, or `test-results/onboarding-uat/<run-id>/` for ephemeral local runs.
- The JSON summary is the source of truth for pass/fail.
- Screenshots and logs are supporting evidence.

## Sequencing

PR1 should not attempt to implement diagnostics, starter questions, or local setup improvements beyond what is necessary to make current behavior testable. PR2 through PR4 must each extend the harness first, then implement product behavior.

Suggested PR titles:

1. `test: add repeatable solo onboarding UAT harness`
2. `feat: add guided onboarding diagnostics and recovery`
3. `feat: offer first-source starter questions after onboarding`
4. `feat: improve guided local model onboarding`

## Open Questions For Implementation Planning

- Exact command names and artifact roots should be finalized in the PR1 implementation plan.
- The harness runner can be a Node/TypeScript script or Python script; choose based on existing frontend E2E conventions during planning.
- Some readiness overlay scenarios may need backend test-only controls or isolated config toggles. PR2 should identify whether existing readiness APIs can produce them without new test support.
- First-source queryability must use the existing media readiness contract where possible. If it is insufficient, PR3 should add a narrow backend readiness signal rather than frontend inference.
- If hosted-provider validation remains local-syntax-only, PR1 should validate hosted success at first chat and use local OpenAI-compatible validation or first-chat failure for live mock-server auth failure coverage. The harness should not pretend a live hosted preflight occurred when the backend did not make one.

## Verification Strategy

PR1 verification:

- Harness unit tests for config/profile/artifact helpers.
- One focused manual harness run in local development.
- Existing onboarding E2E remains passing.
- `git diff --check`.

PR2 through PR4 verification:

- New or updated harness scenario fails before product implementation where feasible.
- Product implementation makes the scenario pass.
- Focused frontend unit tests for UI state and copy.
- Focused backend tests for any new setup/readiness response fields.
- Existing onboarding E2E and UAT harness run.
- Bandit on touched backend production code.
- `git diff --check`.
