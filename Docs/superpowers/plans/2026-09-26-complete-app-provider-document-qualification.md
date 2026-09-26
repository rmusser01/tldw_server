# Complete-app provider and document qualification (TASK-13376)

The user approved the bounded media-readiness correction on 2026-09-26.
The approved distribution design remains the product contract. Qualification
uses ordinary WebUI controls on a fresh signed extracted candidate outside the
checkout, with the repository's deterministic mock provider. No manual backend
master key, frontend/server wiring, publication, or requirement waiver is part
of this work.

## Stage 1: Correct the demonstrated session-readiness failure
**Goal**: Recognize existing single-user cookie-session auth in the post-setup
media-readiness hook and still require successful media access before ready.
**Success Criteria**: Cookie-session success, pending requests, rejected sessions,
server failures and existing key/JWT paths have behavioral coverage; scoped lint
and formatting pass; review finds no unresolved important issue.
**Tests**: Readiness hook and Home setup-flow suites. Compare any broader failure
against the unchanged source. Bandit is not applicable to this TypeScript-only
production change; record that explicitly.
**Status**: Complete

Verification: new hook cases red 5 failed / 7 passed, then hook and Home setup
suites 18 passed. Scoped ESLint and matching shared-code formatting pass. One
read-only reviewer independently reran 18 tests and found no issues. The broader
core-route-identity suite has the same setup-heading failure on unchanged
`d3d1083adc` (1 failed / 6 passed); it mocks the modified hook. That baseline
failure remains open and is not counted as a passing test.

## Stage 2: Qualify the ordinary fresh browser workflow
**Goal**: Build and verify a candidate from clean committed source; complete
provider setup, Markdown ingestion, lexical search and application chat through
the real WebUI.
**Success Criteria**: Exact candidate source, manifest, platform, browser and
ordinary user steps recorded. No setup/application error or manual master-key
requirement is counted as success. Any newly demonstrated behavior change is
presented for review before implementation.
**Tests**: Existing candidate signature/lifecycle/browser checks, then ordinary
provider validation/save/first chat, document upload/search/application chat with
the configured repository mock. Only disposable owned test resources are used.
**Status**: In Progress

Fresh candidate source `cb581a1e16032df27e4026afd784c9a682b584c1` passed
built-backend qualification, 13 lifecycle and 38 initial-browser checks.
Final signature and local artifact hashes verified; pipeline exited 0 and
removed its owned fixtures and signing key. Manifest SHA256:
`4e5c4173e5437196412de3211a83534f7adb7f4b44b94995d127386a79214de7`.
The final archive is extracted outside the checkout at
`/private/tmp/task13376-workflow-cb581a1e16/bundle` for the separate ordinary
provider/document workflow. Initial-wizard scope stays unchanged; full
qualification remains false. Native CI run `36265212062` is still in progress.

Ordinary WebUI provider validation/save and first test chat passed. Home now
accepts the live cookie session. Markdown ingestion and lexical content search
passed, and ordinary chat returned the mock response. Model discovery still
falsely reports no providers/models because its separate precheck omits the
cookie session. Additional behavior changes await review/approval; see
`Docs/superpowers/reviews/2026-09-26-complete-app-provider-document-qualification.md`.

## Stage 3: Verify persistence and record the actual result
**Goal**: Stop/start retains provider configuration and document data, with an
accurate acceptance record and recoverable evidence.
**Success Criteria**: Repeat ordinary search/chat after restart; task criteria
checked only for demonstrated results. Full provider/document qualification
stays false until all required workflow checks pass. Native-platform/core-format
requirements and G12 remain separate required product gates.
**Tests**: Signed stop/start helpers plus browser search/chat with retained data;
verify scoped cleanup and unchanged unrelated services. Preserve evidence and
plans; commit task and acceptance updates with the related work.
**Status**: In Progress

Signed stop/start passed and the document remained searchable. New chat failed
after restart: the provider settings were written to packaged `config.txt`,
outside the persistent config volume, and were absent after container
recreation. Full qualification stays false. amd64 native initial-browser
qualification also failed `manual_master_key_absent_2`; its cause is unverified.
Preserve the private evidence/state and record these blockers before any fix.
