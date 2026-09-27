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

## Stage 2: Correct the approved model and configuration root causes
**Goal**: Make cookie-authenticated model discovery use real server requests,
and persist editable setup configuration in the existing config volume.
**Success Criteria**: Cookie and unauthenticated caches remain separate;
rejected sessions and key/JWT paths retain coverage. Managed startup seeds
only missing packaged configuration assets, preserves existing credentials and
settings, and fails clearly on initialization errors. Non-managed startup is
unchanged.
**Tests**: `TldwModels.test.ts`, the managed entrypoint/config initialization
tests, existing public Docker entrypoint tests, release helper contracts,
scoped lint/format and Bandit on new Python code. Actual signed recreation
and ordinary provider/document chat remain required in Stages 3 and 4.
**Status**: Complete

The user approved both root-cause corrections on 2026-09-26. They are tracked
as TASK-13376.7 (model catalog) and TASK-13376.8 (persistent configuration).
Model discovery changes are confined to its existing auth precheck/cache scope.
Managed startup will resolve configuration through the existing
`TLDW_CONFIG_DIR` override into `/app/managed-config`, initialize missing
packaged assets there before auth/server startup, and preserve existing files.
No auth bypass, strict-model-selection relaxation or provider-status API change.

Model/readiness verification now passes 52 tests. Corrected persistence tests
first failed 9 / passed 2, then the managed config, legacy Docker entrypoint,
release helper and setup-writer suites passed 108 tests. Python lint/format,
shell syntax and Bandit pass (zero production findings). Scoped TypeScript
ESLint has zero errors and the existing unrelated `inputMods` unused-variable
warning; new code introduces no warning. Independent review found no critical
or important findings and independently passed 40 model / 14 managed-config
tests. Its minor diagnostics finding is corrected: safe validation details or
asset/errno are reported without file contents (new assertion red, then 108
tests green again). Existing browser-helper contract tests passed 59 tests after
the sandbox's localhost-listen restriction required an authorized rerun.
The lifecycle restart check verifies saved provider fields as well as the
existing data sentinel. Fresh signed-source verification is recorded below.

The user approved reclaiming unused Docker build cache after the host disk
limit blocked the fresh build. `docker buildx prune --builder default --force`
reclaimed 14.72 GB. Exact before/after identities matched for all 32 images,
2 containers and 6 volumes; no image, container or volume pruning occurred.
Private cleanup evidence is retained at
`/private/tmp/task13376-cache-cleanup-9709d0dcb7`.

## Stage 3: Qualify the ordinary fresh browser workflow
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
**Status**: Complete

Fresh candidate source `cb581a1e16032df27e4026afd784c9a682b584c1` passed
built-backend qualification, 13 lifecycle and 38 initial-browser checks.
Final signature and local artifact hashes verified; pipeline exited 0 and
removed its owned fixtures and signing key. Manifest SHA256:
`4e5c4173e5437196412de3211a83534f7adb7f4b44b94995d127386a79214de7`.
The final archive is extracted outside the checkout at
`/private/tmp/task13376-workflow-cb581a1e16/bundle` for the separate ordinary
provider/document workflow. Initial-wizard scope stays unchanged; full
qualification remains false. Native CI run `36265212062` ended with amd64 browser
failure and an arm64 180-minute build timeout; neither result is waived.

Ordinary WebUI provider validation/save and first test chat passed. Home now
accepts the live cookie session. Markdown ingestion and lexical content search
passed, and ordinary chat returned the mock response. Model discovery still
falsely reports no providers/models because its separate precheck omits the
cookie session. Additional behavior changes await review/approval; see
`Docs/superpowers/reviews/2026-09-26-complete-app-provider-document-qualification.md`.

The earlier `cb581a1e16` result above is retained as failure history. After the
two approved corrections, source `9709d0dcb7e846d3a7366a3412afa933208f4b10`
passed built-backend checks, 13 lifecycle checks and 38 initial-browser checks
on local linux/arm64. The final signed archive was extracted into fresh private
state outside the checkout at `/private/tmp/task13376-workflow-9709d0dcb7`.
Ordinary WebUI provider validation/save/wizard chat, Markdown upload, lexical
search and application chat all passed. Provider status was Healthy without
the prior false missing-provider/model banner. No manual server API key or
frontend/server URL wiring was used. This local mock-provider result does not
qualify commercial providers, vector retrieval or chat answer quality.

## Stage 4: Verify persistence and record the actual result
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

That failure belongs to the earlier `cb581a1e16` candidate. On the corrected
`9709d0dcb7` candidate, signed stop/start recreated the application containers;
ordinary full-text search returned the retained document and exact marker,
and a distinct post-restart chat request returned the mock response. Provider
settings were not re-entered. Screenshots and `workflow-evidence.json` record
the successful bounded ordinary workflow separately from initial-wizard
evidence. Owned application/mock/registry containers were removed; named data
and configuration volumes, private state, evidence and images were retained.
The two unrelated PostgreSQL containers remain running.

Native run `36280791905` failed amd64 `manual_master_key_absent_2` after its
13 lifecycle checks passed. Arm64 reached the 180-minute job deadline during
Bun install; its root cause remains unverified. Windows syntax passed, which
does not qualify a Windows Docker host. Local signed ordinary workflow passed;
full product qualification, native/core-format gates and G12 remain open.
Stage 4 remains In Progress for those unresolved qualification results.

A separate setup-loading defect has two failing read-only diagnostic cases
(18 existing cases pass): incomplete initial state/metadata exposes the manual
API-key form. The proposed correction is pending user approval and has not been
implemented. Existing browser acceptance checks remain unchanged.

The user subsequently approved that bounded loading correction, tracked as
TASK-13376.9. The route will retain its loader while either initial readiness
value is missing and loading remains active, without exposing manual connection,
operator recovery or a premature load error. Once loading finishes, existing
missing-data/error recovery remains available; refreshes with both existing
values retain their current usable surface. Route tests first reproduce the
initial/partial-state and failure transition; then a minimal rendering guard is
verified with the onboarding-hook and setup-choice suites and independent review.
Browser acceptance checks and native qualification requirements remain unchanged.

TASK-13376.9 completed at source `3ef013bd94d389454e0b10b67060f9a95e9089af`:
60 affected route/hook/choice tests passed after the new cases failed first;
independent review found no actionable issues and independently passed 24 route
tests. Scoped lint had no rule findings and the normal managed production WebUI
build, token synchronization and bundle budgets passed. Bandit is inapplicable
to this TypeScript-only correction; existing whole-frontend typecheck limitations
remain. Fresh signed local qualification was blocked by the recurring frozen
Bun install after resolved/extracted 256. Under the reviewed bound, the verified
owned Buildx process was stopped after 1090 seconds without output; the pipeline
exited 130 and removed its registry, retaining recovery evidence and unrelated
services. No signed candidate or fresh browser/lifecycle result is claimed.
Native run `36293327915` was In Progress at the final check, with Windows syntax
passed and Linux candidate builds running. Stage 4 remains In Progress; the
earlier successful ordinary workflow is specific to `9709d0dcb7`.

Continuation verified native amd64 source `3ef013bd94`: all 13 lifecycle and
38 browser checks passed, including `manual_master_key_absent_2`. The downloaded
artifact ZIP digest, signed manifest and eight bundled file hashes verified;
the unchanged promotion gate refused G12=false. This remains initial-wizard
evidence with planned full setup false. Arm64 was still building at the latest
check, Windows helper syntax passed, and the full native/core-format/G12 matrix
remains open. TASK-13376.10 ran one separately scoped retained-evidence frozen
Bun install: it passed in 85.08 seconds (92-second monitor), identified active
Puppeteer/canvas children and did not reproduce the stall. Timeout retention
was verified with a sleep fixture. Exact owned diagnostic cleanup preserved all
baseline Docker resources and private evidence. No runtime/dependency change,
further candidate retry, requirement waiver or new ordinary workflow is claimed.
Stage 4 remains In Progress; see the acceptance review for exact hashes and limits.

TASK-13376.11 captured the actual BuildKit stall in one unchanged-input diagnostic:
timeout exit 124 at 300 seconds, Buildx exit 1 at 303 seconds, and 966 private
evidence records retained. The pending child was Puppeteer 24.36.0
`node install.mjs`, actually running Bun; CPU and I/O were unchanged from 232
through 299 seconds. Canvas had exited. Baseline Docker resources
were preserved. This identifies the pending script; its internal cause remains
unproven. A prepared Docker-only skip-download proposal awaits requester review
and remains unapplied. No second install/candidate retry or CI cancellation.
Both existing native runs still have arm64 building with amd64 and Windows syntax
passed. The broader workflow/latest-source and release gates remain open.


TASK-13376.12 completed the approved command-local Docker Puppeteer skip at
clean source `41e3b9bc0598071034f250702ccc71a7b862a7e3`. Local production install
completed in 57.06 seconds; the pipeline passed built-backend checks, 13 lifecycle
and 38 initial-browser checks. Native run 36328715529 succeeded on amd64 and
arm64 with the same 13/38 checks each; Windows helper syntax and combined platform
job passed. Independently verified signatures, file/archive hashes and paired
inventory are recorded in the acceptance review. Linux initial-browser blockers
are cleared on this source; native Windows Docker host execution and the full
native/core-format/G12 matrix remain separate gates.

A new fresh signed ordinary WebUI run on this source passed provider validation,
save, wizard chat, Markdown upload (1 succeeded/0 failed, 5 seconds UI elapsed),
lexical content search, application chat and signed container recreation.
Post-restart search and a distinct new chat passed without re-entering settings;
provider stayed Healthy. Owned cleanup preserved baseline resources and named
instance volumes. The upload's additional Search in Knowledge action exposed a
reproducible 401 on QA history despite a valid cookie session. Private controlled
and actual-browser evidence isolates a token-guard auth-order defect. The prepared
canonical-principal resolution correction awaits review/approval; no auth code
changed. New ordinary evidence conservatively remains passed=false and planned
setup false because of the observed application error. Stage 4 remains In Progress.
