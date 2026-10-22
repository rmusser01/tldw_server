# UAT C03 Evaluations cookie authentication

Related task: TASK-13260.278.18.83.45. Root recorded this new actual failure before edits and owns the parent task.

## Evidence and approved design

Actual Recipes and Datasets GETs return HTTP 401, Missing API key or token, while the same single-user cookie session works elsewhere. The Evaluations client uses apiSend and canonical request-core cookie transport, which sends same-origin cookies and omits Authorization/X-API-KEY. Evaluations verify_api_key rejects missing headers before canonical authentication; get_eval_request_user duplicates this rejection.

Recognize only single_user requests with the configured cookie and completely absent Authorization/X-API-KEY as cookie candidates. Preserve the existing Evaluations IP check, then invoke canonical get_request_user with explicit empty dependency arguments. The cookie is never a credential string or owner; the validated canonical user supplies identity. Remove only the duplicate get_eval_request_user header-only rejection. Explicit header/JWT behavior and error mappings remain; permissions, admin, ownership, rate limits and CSRF stay in their existing owners.

ADR check: ADR required: no. Existing [ADR-044](../ADR/044-cookie-session-governance-owner-preflight.md) governs canonical cookie-owner resolution, request-state reuse and explicit-header precedence; this aligns Evaluations with existing authentication and leaves public/persistence/security policies intact. Root owns the related task assessment.

## Stage 1: Diagnose and causal regressions
**Goal**: Prove the header-only rejection using actual persisted sessions and served Recipes/Datasets handlers.
**Success Criteria**: Supported configured Python 3.12 tests fail on legitimate cookie reads; independent denial and security controls retain their outcomes.
**Tests**: Genuine session mint, both list routes and slash redirects; absent/tampered/expired/revoked/wrong-device sessions; explicit invalid/empty/malformed headers, IP denied, primary API-key/Bearer compatibility; owner/permission/CSRF checks; canonical infrastructure failure remains closed.
**Status**: Complete

## Stage 2: Minimal shared boundary repair
**Goal**: Add the narrow cookie candidate branch after the existing IP guard and remove duplicate header rejection.
**Success Criteria**: New regressions and genuinely affected existing auth/permission/owner checks pass; no endpoint/UI/auth resolver changes.
**Tests**: Normal configured pytest, existing exact activated Python 3.12.13, outer 600 seconds and case 300 seconds, maximum three issue attempts.
**Status**: Complete

## Stage 3: Static qualification, review and runtime handoff
**Goal**: Freeze production/tests for root review and controlled runtime recovery.
**Success Criteria**: Scoped baseline/current Ruff and Bandit (including new test), Python syntax pass, independent review clear; root controlled runtime load and actual Recipes/Datasets recovery completed.
**Tests**: Existing activated main Python 3.12 for tests/Ruff/compile; verified existing primary venv STATIC Bandit only. No live/browser/runtime/model/config/dependency/parent-task/Git/proof actions. Prior A10 and all apps remain frozen.
**Status**: In Progress

## Baseline scope

Production SHA256 before edits: 0eeaf816e01eb97a663c4cc14dadb42672703618803ea89dd64d82f8ae2b2cb0. Existing Python 3.12.13 activated; normal configured Ruff --no-cache finds inherited production I001 import ordering. New test import ordering corrected before behavioral invocation. Existing primary STATIC scanner verified Python 3.11.13 / Bandit 1.9.3: baseline production/new test has zero findings/errors, 546 AST lines, 21 narrow B101 annotations and zero broad nosec; scanner emits comment-word parsing warnings for the repository assertion-comment convention, not new security codes.

Initial baseline output-capture wrapper used too small an output budget and its JSON decoder saw the truncation warning; repeat capture with a sufficient budget succeeded. This was a pre-test operator reporting failure, not a causal test failure.

## Attempt 1: causal red

Unchanged production and normal configured Python 3.12 fixtures/plugins: 13 failed / 24 passed, pytest 201.16s / outer 209.416s, 4950 configured warnings. The real session mint succeeds and actual Recipes/Datasets list GETs (with and without slash redirects) return the exact 401 missing-credentials boundary. Other affected owner/permission/IP/canonical-failure assertions are blocked at that same header-only seam; independent invalid-session and explicit-header/primary-key controls remain passed. No timeout/setup error was reported. The output wrapper retained the summary and final failure's real mint200/list401 evidence; not every full failure capture survived its bounded tail. This is not full production lifecycle/governance qualification.

Applied exactly the approved shared-boundary changes after the verified baseline hash: narrow configured cookie candidate, unchanged IP guard before canonical user resolution, and deletion of the duplicate header-only rejection. Existing explicit header/JWT branches and all separate guards remain.

## Current static scope and review handoff

Full parsed AST comparison confirms that restoring only the new cookie-candidate assignment/admission branch and deleting the old duplicate header rejection makes baseline/current AST identical. Current production SHA256: 2d10bfd70b6c75fc004893e71b64cb6cfc4a91d6c8bd980f816ab2436b650123. Current new-test SHA256: b82461202413de22874a966b42919f8540e143839d1fb800f54f244e7c5fc367.

Python 3.12 in-memory compile PASS for both files. Configured Ruff current matches the single inherited production I001; new test has zero findings. Existing primary STATIC Bandit current source+test: zero findings/errors, 542 AST lines, same21 narrow B101 assertion skips, zero broad nosec; 93 assertion-comment parser warnings, no other warnings/security codes. No production suppression was introduced.

Coverage boundary: the imported supported fixture bootstraps real AuthNZ SQLite and real session manager, mints through the real endpoint, and runs actual Recipes/Datasets handlers/services through the minimal app with real CSRF. It does not qualify the full deployed application ingress/lifecycle, PostgreSQL, external evaluators/models or actual catalog C03 acceptance. Permission tests retain real session validation but vary the canonical User factory's claims; the canonical-unavailable control substitutes only that boundary with a503. Root owns independent review, controlled runtime replacement and live recovery. No prior A10/client scope changed.

## Attempt 2: final affected green and frozen handoff

Normal configured activated Python 3.12.13 pytest PASS: 50 passed, zero failed/skipped, pytest197.80s / outer205.791s, 4943 configured warnings (details suppressed by repository addopts). Scope: new37 real-cookie/denial/security cases; existing TestAuthentication6 (primary key, JWT success/expired/invalid/inactive, IP allowlist), existing evaluation auth/permission2, benchmark auth5. No evaluator/model/runtime/UI/prior A10 controls were replayed. Two bounded behavioral invocations used; no third needed. All causal list positives, permission checks, IP-denial code, canonical failure propagation and real owned dataset/CSRF guards now pass.

Production and tests remain at the hashes above. Static results and AST scope comparator retain validity because production/test contents did not change after them. No new source/test security or lint findings; inherited production I001 remains. No Critical/Important/Minor finding identified in self-review. Independent root review requested; Stage3 remains InProgress until independent review, controlled runtime load and actual C03 Recipes/Datasets recovery are complete.

Root independent source review CLEAR: full production/function/new37case module and connected canonical resolver/cached principal/Recipes/Datasets permission/owner consumers read. Actual live classic dataset/evaluation/exact_match run succeeded; root retains50distinctpass evidence. Separate completed-progress0 is a new schema-only unit; no cookie test replay or source changes. Stage3 remains subject to root's catalog acceptance record.
