# UAT390 — exact source grounding acceptance repair

Backlog: TASK-13260.278.3. Approved scope: RELEASE_UAT_PLAYBOOK A-05 TXT,
A-07 Media-only FTS/citations/loaded Media-to-Chat/reload and absent-price control.
The Wikipedia article, immediate-loading race, vector/hybrid, A-09 reuse,
native onboarding and database matrix remain separately measured gates.

## Stage 1: Trace the existing contracts
**Goal**: Trace ingestion, source selection, citations, Chat and persistence.
**Success Criteria**: Assertions use actual UI actions and canonical returned IDs.
**Tests**: Read existing page objects and Media/Knowledge/Chat tests.
**Status**: Complete

The legacy journey accepts a job ID fallback, arbitrary answer prose as search
results and a generic topic answer in a blank Chat. Reuse Quick Ingest and Chat
page objects, but independently corroborate the exact opened source. Use the
Knowledge QA source ID attributes and saved RAG context instead of result count.
No production changes or browser-fulfilled application success responses.

## Stage 2: Add discriminating acceptance checks
**Goal**: Freeze Rowan/Larch bytes; reject wrong IDs, uncited/generic/contradictory
answers, incomplete handoff and changed/duplicate saved messages.
**Success Criteria**: Causal oracle tests reject each missing guarantee; journey
preserves the same identities through real UI/API reads and reload.
**Tests**: UAT390 oracle unit tests and deliberate oracle mutation check.
**Status**: Complete

## Stage 3: Verify and report
**Goal**: Focused unit checks, discovery, lint/type validation and security scan.
**Success Criteria**: Record actual outcomes; do not claim unrun integration or
live model quality. Integrated runs require explicitly owned WebUI/API/worker
and provider, selected with TLDW_UAT390_MODE=deterministic or live and
TLDW_UAT390_PROVIDER / TLDW_UAT390_MODEL. Deterministic means the actual backend
uses a controlled downstream provider; it never means stubbing application APIs.
**Tests**: Unit, Playwright discovery, focused lint, TS check and Bandit scope.
**Status**: In Progress

### Focused verification — 2026-09-21

- Vitest UAT390: 29/29 passed. Removing only the canonical citation source-ID
  check caused both unrelated-ID and chunk-ID regressions to fail (2 failed,
  27 excluded by filter); restored code passed all 29. Mutation output:
  /tmp/uat390-mutation-red.txt.
- ESLint on the three TS files: exit 0. Prettier check: pass. Diff whitespace
  check: pass. Playwright --project=journeys --list: exactly one linked case.
- Frontend tsc --noEmit --incremental false: exit 0 before the final request
  observation/type-only additions; parent runs the combined final check.
- Bandit was invoked through the project venv on the touched TS files. It
  reported 3 Python-AST parse errors and no findings: TypeScript is not
  supported by Bandit, so this is not a successful security analysis. Report:
  /tmp/bandit_uat390.json. No production Python or application code changed.
- Runtime warnings: existing Node localStorage experimental warning and
  module.register deprecation; no assertion failures in the restored suite.

### Remaining acceptance gates

No owned integrated API/worker/provider/WebUI was supplied for this subtask.
Do not mark the SQLite/PostgreSQL linked journey or live-model quality as
passed. Native fresh-auth/multi-user login, immediate-loading handoff, external
Wikipedia, vector/hybrid and linked A-09/A-11 remain separate. Current harness
uses existing seeded API-key auth and removes its application API fulfillment.
The test retains fixture hashes, canonical media/QA/Chat/message IDs and saved
text in uat390-linked-evidence.json for dependent work and failure review.

Verified contracts: MediaDetailResponse.media_id/source.title/content.text;
POST media/search returns items with id; messages-with-context returns an
array, chats/:id/messages returns a messages envelope; both persist sender
and conversation_id. No job IDs, flat guessed media fields or UI-only role
fields are accepted at these boundaries.

Final negative-control review added an invented-free-admission case. It failed
against the previous price oracle, then passed after the unsupported-price
check was extended. Final focused run: **30/30** tests passed; ESLint and
Prettier passed. Parent retains responsibility for final combined type checking
and owned integrated/native execution. The earlier 29-test/mutation counts
above describe the recorded intermediate run, not the final suite size.

### Literal citation membership — 2026-09-22

Four injected/altered/paraphrased/foreign-title controls fail before canonical title/body membership is required. All50grounding tests pass afterward; retained CI5dc06 actual Media and QA responses also pass the real oracle. The visible source preview must equal the saved cited excerpt. This closes that assertion gap without claiming arbitrary chunk identity or full A-07/four-cell acceptance. Scoped independent review is clean; next first-attempt CI remains pending.
