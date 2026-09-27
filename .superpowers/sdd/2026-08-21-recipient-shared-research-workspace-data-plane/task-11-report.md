# Task 11 Report

## Result

Fix Round 2 execution passed and is pending controller review. The parent task remains In Progress. The explicit Chrome CDP acceptance runner exercised the real multi-user SQLite backend and Next.js WebUI using `local-llm` / `Qwen2.5-0.5B-Instruct`.

## Live Evidence

- Final evidence validation exited `0` with no failures.
- All 15 acceptance checks passed.
- The strict ledger is clean: no undeclared HTTP failure, request failure, console error, page error, runtime overlay, removed full-media call, local-workspace call, or forbidden mutation/tool request.
- Chats settings requests were bounded to two `200` responses.
- The idempotency race produced `409/200/200/409`; replay turn hashes match and the final changed-fingerprint request returned `409`.
- Evidence contains no credentials or absolute machine paths.
- Desktop shared-source, desktop cited-answer, mobile preview, and revoked-state screenshots were visually inspected.

## Lifecycle Hardening

The live runner now records route-transition traffic in a transient observer before attaching the strict interaction ledger. That observer still fails on all console, page, runtime-overlay, and unexpected HTTP errors. Only a `net::ERR_ABORTED` GET caused by route teardown is explicitly excluded; cancelled mutations and every other request failure remain fatal. Deterministic contracts cover ordering, transition failure cleanup, console/page/HTTP failure rejection, and the narrow abort scope.

## Verification

- Focused Vitest runner suite: 49 tests passed.
- Required focused frontend suite: 84 tests passed.
- The two named package tests passed separately from their actual workspace-relative paths: 13 tests passed.
- Final UAT: passed with all evidence checks above.

## Scope

No PR was created or pushed. Run-specific fixture cleanup metadata remains outside version control. The two unrelated untracked watchlist templates remain excluded.

## Fix Round 1/5

Reviewed base: `c718edb37406c78abf5f6813ab98612fd1bb7099`.

### Accepted Findings

- Direct/background GET coalescing and cooldown keys now include the normalized resolved server and a non-secret auth-scope fingerprint. Ambiguous cookie-backed multi-user requests do not coalesce. Regressions cover different servers, different principals, same-scope coalescing, cooldown isolation, and normalized expected statuses.
- Assistant restoration now has a generation-aware serialized selection commit. A cancelled restore is distinct from a missing restore and cannot fall through to recent-chat initialization. Tests cover cancellation during persistence and explicit server-chat, URL-character, settings-return, and handoff selections while retaining local Chats behavior.
- Transition observation is operation-based and fail closed. Owner revocation and member Chats use separate destination policies with exact fixture workspace/migration identity checks, named read-only bootstrap operations, bounded declared writes, status/method/path/origin checks, bounded counts, and exact registered route-teardown cancellations. Wrong IDs, old recipient media routes, local workspace traffic, undeclared mutations/tools/aborts, unknown origins, console/page errors, and overlays remain fatal. Evidence stores only operation names and counts.
- Expected errors use multiset operation correlation. The two race conflicts have distinct operation IDs and body hashes, and console errors require exact correlation. Evidence validation requires `passed`, exactly the canonical 15 true acceptance keys, the exact five screenshots, ready provider/model metadata, isolation and provider-context proofs, closed clean ledgers, settings/race invariants, environment/run hashes, and a closed schema.
- A transparent in-memory forwarding probe sends unchanged requests to the actual model and persists only bounded counts, hashes, and booleans. It fails on zero traffic, changed bodies, sentinel content, mutation payloads, or tool payloads.
- Mobile core and preview evidence are separate. JSON/logs omit prompt and answer bodies; screenshots intentionally retain the visible transcript.

### TDD And Focused Verification

Changed behavior tests:

```bash
cd apps/tldw-frontend
bunx vitest run ../packages/ui/src/services/__tests__/background-proxy.test.ts ../packages/ui/src/hooks/__tests__/useSelectedAssistant.test.tsx ../packages/ui/src/hooks/__tests__/usePlaygroundSessionPersistence.test.tsx ../packages/ui/src/components/Option/Playground/__tests__/Playground.coordinator.integration.test.tsx __tests__/shared-research-workspace-cdp-uat.test.ts __tests__/local-llm-forwarding-probe.test.ts --maxWorkers=1 --no-file-parallelism
```

Result: `6` files, `191 passed`.

Required focused command:

```bash
bunx vitest run __tests__/shared-research-workspace-cdp-uat.test.ts __tests__/research-workspace-uat-runner.test.ts ../../packages/ui/src/components/Option/ResearchWorkspace/__tests__/ShareDialog.test.tsx ../../packages/ui/src/services/tldw/__tests__/request-core.refresh-timeout.test.ts __tests__/components/notification-lifecycle-provider.test.tsx --maxWorkers=1 --no-file-parallelism
```

Result: `3` matched files, `111 passed`; the two `../../packages/ui` arguments are stale relative paths. Running their actual workspace paths passed `13` more tests. RED evidence included stale assistant persistence, restore fallthrough, unsafe cross-scope GET reuse, transition-policy bypasses, duplicate expected failures, malformed evidence, and probe mutation/sentinel/body-change cases before the fixes.

### Live Run

Three distinct live hypotheses were tested: final28 established the owner-management destination-policy mismatch; final29 exposed exact owner font bootstrap operations; final30 proved the owner boundary and exposed the exact member Chats font/capability bootstrap set. The fresh `final31-fix1-1787442542-12131` fixture then passed with exit `0` and no failures.

- All 15 canonical acceptance checks are true.
- Strict and both transition ledgers are closed with no unexpected operation. Owner transition: `56/64`; member Chats transition: `44/64`.
- Settings: two requests, both `200`. Race: `200/409/200/409`, two successes, final `409`, equal turn hashes, and exactly one correlation for each conflict operation.
- Provider: ready `local-llm` / `Qwen2.5-0.5B-Instruct`. Probe: three requests, every input/output body hash equal, no sentinel, mutation, or tool payload.
- Context isolation and provider-context proofs are present. Evidence contains no credential values, sentinel literals, prompt/answer/message bodies, secrets, or absolute machine paths.
- All five screenshots were controller-inspected: desktop workspace, desktop grounded answer, mobile core, mobile preview, and revoked state.
- The final31 cleanup manifest and UAT log are mode `0600`. Controller-required independent verification also confirmed mode `0600` for `/tmp/tldw-shared-uat.Heaplg/cleanup-final27-1787436202-8548.json` and `/tmp/tldw-shared-uat.Heaplg/logs/uat-final27-1787436202-8548.log`.

### Quality Gates

- Focused frontend/UAT ESLint: zero errors. Forced package lint under the frontend config: zero errors with `112` pre-existing warnings outside the changed lines.
- `node --check` passed both UAT scripts.
- No Python file differs from the reviewed base. Repo-venv Ruff found five pre-existing `auth.py` findings and Bandit found eleven pre-existing low-severity `auth.py` B106 findings; base-versus-head byte comparison proves no new Python or security finding. Bandit JSON remained in `/tmp`.
- `git diff --check` passed. No backend unit target was affected because Fix Round 1 changes no Python; the real backend was exercised by final31.
- No PR or push was performed. The two unrelated watchlist templates remain untouched and unstaged.

## Fix Round 2/5

Reviewed base: `5a648f8532cf1f86d4892ccc216cb52a5c8652a2`.

### Findings Addressed

- Direct GET coalescing now resolves one immutable direct configuration snapshot and uses that exact snapshot for both the non-secret server/principal scope and request execution. Runtime messaging, caller-header, cookie-session, missing-config, and anonymous-principal requests do not coalesce unless `noAuth: true` is explicit. A 401 refresh retries with current storage credentials while an old-principal in-flight result remains partitioned.
- Every owner-management and member-Chats transition operation declares exact allowed statuses. API reads and ordinary writes require `200`, migration creation requires `201`, and only named static/font resources allow `200/304`. Proof entries record operation name, bound, count, allowed statuses, and observed statuses; any wrong method/path/ID/origin/status/count remains fatal.
- Evidence validation requires exactly owner/member/nonmember, pairwise-distinct config, cookie, marker, and marker-cookie hashes, same-persona marker equality, and permits only the intentional shared storage-key hash. It requires exactly the two closed transition proofs, exact proof/operation fields, known operation status contracts, and observed-status multiset correlation.
- The provider probe binds only exact `127.0.0.1` or `::1` and accepts only a credential-free, query-free, fragment-free local HTTP loopback upstream at `/v1/chat/completions`. Body bounds, JSON, request count, sentinels, mutations, tools/functions, and byte identity are checked before `fetch`; every violation is local and records no raw body.

### TDD Evidence

- Coalescing RED: the full proxy suite reported `3 failed, 84 passed`; a separate secret-header contract reported `1 failed, 87 skipped`. GREEN: `88 passed`.
- Transition-status RED: `6 failed, 2 passed, 75 skipped` because proofs omitted statuses and `201/302/399/204` were accepted in the wrong contracts. GREEN: `8 passed, 75 skipped`.
- Evidence-validator RED: `2 failed, 11 passed, 82 skipped` on the new canonical proof shape; the strengthened malformed-object contract separately failed `1` test. GREEN: `13 passed, 82 skipped`, then the strengthened malformed-object test passed.
- Probe RED: `16 failed, 2 passed` because prohibited bodies/targets reached fetch or unsafe bind hosts were accepted; main-listen validation separately reported `2 failed, 7 passed, 20 skipped`. GREEN: `29 passed`.
- Final amended matrix with corrected workspace-relative package paths: `260 passed` across seven files. Its component runs were `159 passed` for runner/probe/lifecycle and `101 passed` for proxy/ShareDialog/request-core; the proxy suite was also rerun after lint-only test typing and remained `88 passed`.

### Fresh Live Run

Fresh `final32-fix2-1787446794-16413` passed on the first live Fix Round 2 hypothesis through the loopback probe.

- Evidence status is `passed`; validation is exit `0` with `failures: []`; all canonical 15 acceptance keys are true and exactly five screenshot fields are present.
- The strict ledger is closed and clean. Owner transition recorded `56/64` requests and member Chats recorded `49/64`; both have zero unexpected requests and exact allowed/observed statuses for every used operation.
- Settings are two `200` responses. Race statuses are `200/409/200/409`, with two successes, final `409`, and equal replay turn hashes.
- Provider readiness is truthfully `local-llm` / `Qwen2.5-0.5B-Instruct`. Three requests traversed the probe unchanged; every input/output hash matches and sentinel, mutation, and tool/function checks are clean.
- Exact owner/member/nonmember isolation proof is present. Credential-value, sentinel-literal, and committed-evidence machine-path scans are false. Prompt/answer bodies are absent from JSON and the protected execution log; screenshots intentionally retain the visible transcript.
- All five regenerated screenshots were visually inspected: desktop shared sources, desktop grounded answer/citations, mobile two-tab core, mobile full-screen source preview, and revoked state. No overlap, overflow, extra banner, or revoked-data leak was visible.
- `/tmp/tldw-shared-uat.Heaplg/cleanup-final32-fix2-1787446794-16413.json` and `/tmp/tldw-shared-uat.Heaplg/logs/uat-final32-fix2-1787446794-16413.log` are both mode `0600`.

### Quality Gates

- Task 11 runner/probe ESLint: zero errors or warnings. Forced shared-package lint: zero errors and `95` inherited warnings; the three new test annotations were removed and no changed line reports a warning.
- `node --check` passed both changed `.mjs` scripts. No Python file differs from reviewed base `5a648f8532`, so touched-scope Ruff and Bandit are not applicable; Fix Round 1's base-versus-head Python security comparison remains unchanged.
- `git diff --check` passed. Final status/staging verification and the resulting commit hash are reported by the executor because a commit cannot contain its own hash. No PR/push; the two unrelated watchlist templates remain excluded.
- Status: implementation and live evidence passed; pending controller review, with the Backlog task left In Progress.

## Fix Round 3/5

Reviewed head: `79cf6dc1d4b9ba03c6b1c9fe6595c0a408812865`. Status remains In Progress pending controller review.

### Findings Addressed

- One canonical declaration contract now drives both runtime transition-policy construction and evidence validation. `owner-revocation` requires exactly 37 declarations and `member-chats` exactly 35; each context rejects missing, extra, duplicate, renamed, or cross-context names and requires the canonical allowed-status set and maximum count for every declaration. Existing exact origin/method/path/fixture-ID/status classification remains unchanged.
- The local provider probe now invokes upstream `fetch` with `redirect: "error"`. Redirect responses cannot carry a provider request outside the exact loopback boundary; forwarding failures remain the bounded generic `502` response with no body or target leakage.

### TDD Evidence

Transition validator RED:

```bash
cd apps/tldw-frontend
bunx vitest run __tests__/shared-research-workspace-cdp-uat.test.ts --maxWorkers=1 --no-file-parallelism -t "missing zero-count|owner operation added|member Chats operation added|duplicate or globally"
```

Result: `4 failed, 95 skipped`. Removing a zero-count owner declaration, adding an owner declaration to Chats, adding a Chats declaration to owner, and replacing an owner declaration with a globally known Chats name all incorrectly validated with exit `0`.

Transition validator GREEN: the same command passed `4` tests with `95` skipped. The first full runner-file pass exposed one compatibility regression: an intentionally invalid fourth migration chunk threw during policy construction. A bounded index guard retained the prior classified fail-closed result; the rerun passed `99` tests, and the final file passed `100` after adding the exact maximum-count mutation contract.

Probe redirect RED:

```bash
bunx vitest run __tests__/local-llm-forwarding-probe.test.ts --maxWorkers=1 --no-file-parallelism -t "forwards the exact|fails closed on upstream redirect"
```

Result: `6 failed, 28 skipped`; the fetch init had no redirect mode and all five `301/302/303/307/308` follow branches reached the rejected destination. GREEN: the same command passed `6` with `28` skipped. The strengthened native-fetch integration using a real loopback redirector and an instrumented rejected-host destination passed all five redirect statuses with zero destination contacts (`5 passed, 29 skipped`).

Changed test files:

- `apps/tldw-frontend/__tests__/shared-research-workspace-cdp-uat.test.ts`
- `apps/tldw-frontend/__tests__/local-llm-forwarding-probe.test.ts`

### Fresh Live Run

Fresh `final33-fix3-1787449107-31317` passed through the updated probe on the first live attempt.

- Evidence status is `passed`; independent validation returned exit `0` with `failures: []`; the exact 15 canonical acceptance keys are all true.
- The strict ledger is closed with no request failure, page error, or runtime overlay, and every expected HTTP/console failure has exact operation correlation. Owner transition is `57/64` requests with exactly 37 declarations; member Chats is `48/64` with exactly 35. Both have zero unexpected requests/errors and canonical context-specific declarations/statuses/bounds.
- Settings are two `200` responses. Race statuses are `409/200/200/409`, with two successes, final `409`, distinct conflict operations, and equal replay turn hashes.
- Provider readiness is truthfully `local-llm` / `Qwen2.5-0.5B-Instruct`. Three requests traversed the probe unchanged; all input/output hashes match and sentinel, mutation, tool/function, JSON, and request-bound proofs are clean.
- Exact owner/member/nonmember isolation proof is present. Credential-value and sentinel/body-field scans across evidence/log are clean; the committed evidence machine-path scan is clean.
- All five regenerated screenshots were visually inspected: desktop shared sources, desktop grounded answer/citations, mobile two-tab core, mobile full-screen preview, and revoked state. No overlap, overflow, extra banner, unreadable evidence, or revoked-data leak was visible.
- `/tmp/tldw-shared-uat.Heaplg/cleanup-final33-fix3-1787449107-31317.json` and `/tmp/tldw-shared-uat.Heaplg/logs/uat-final33-fix3-1787449107-31317.log` are both mode `0600`.

### Quality Gates

- Final focused Vitest matrix: `7` files and `270 passed`, including the preserved Round 2 background-proxy suite (`88 passed`).
- `node --check` passed both executable scripts. Focused changed-file ESLint reported zero errors/warnings. Forced shared-package ESLint under the frontend config reported `0 errors, 95 inherited warnings`; those package files are unchanged from the reviewed head.
- No Python file differs from `79cf6dc1d4`, so touched-scope Ruff and Bandit are not applicable. No backend production path changed; the real backend was exercised by final33.
- `git diff --check`, final status/staging verification, and the resulting commit hash are recorded by the executor after the report is staged because a commit cannot contain its own hash. No PR/push; the two unrelated watchlist templates remain excluded.

## Final Whole-Workstream Fix Pass

Reviewed base: `22c9b62f69610b26daff52c4f1e47ea0f2f116d2`. Fix commit: `d51f8d1be5`.

### Findings Addressed

- Implicit all-source selection no longer stores loaded-page IDs. Deselecting one source first materializes the complete unfiltered paginated queryable source snapshot, validates stable totals and summaries, duplicate-free IDs, exact offset progress, no partial errors, and the target's continued presence, then atomically switches to include mode. Any inconsistency leaves all-source mode intact with a bounded recovery error.
- A successful ask response that is malformed, truncated, or has a different `request_id` is now typed as post-commit ambiguous. Retry reuses the exact frozen request object and UUID. Ordinary non-2xx typed API errors retain their existing classification and are not offered as ambiguous retries.
- The chat pane announces and scrolls only when the reducer records the exact newly completed assistant message ID. Bootstrap and reload history cannot consume or synthesize an `Answer added` announcement.
- The live CDP harness now clicks and waits for asynchronous source materialization. Its strict member-Chats ledger permits the exact four read-only bootstrap paths already allowed during the transition observer, while method/path broadening remains rejected.

### TDD And Focused Verification

- Initial final-finding RED: `16 failed, 57 passed` across the shared service, reducer/controller, component, accessibility, and responsive suites.
- Final shared UI/API/locale matrix: `6` files, `78 passed`.
- CDP runner matrix after the live harness findings: `102 passed`.
- Production-only TypeScript check passed before the harness-only JavaScript changes. Targeted harness ESLint, `node --check`, locale byte-parity, evidence validator, bounded credential/path/sentinel/body-field scans, screenshot contract, and `git diff --check` all passed.
- The broader Research Workspace family run retained only 14 unrelated existing local-workspace failures (`845 passed`); no shared recipient test failed. Full frontend typecheck retained only unrelated skills-certification diagnostics with no touched-path diagnostic.
- No Python changed in this pass, so Ruff and Bandit are not applicable.

### Fresh Live Acceptance

The first fresh run correctly failed because the temporary backend process lacked `DEFAULT_MODEL_LOCAL_LLM`; the backend was restarted with the existing local provider/model and the missing non-secret default-model fixture setting. A second run exposed the expected asynchronous deselection contract in Playwright's synchronous `uncheck()` helper, and a later complete run exposed four exact read-only Chats bootstrap requests crossing the transition/strict-ledger timing boundary. Both harness defects received RED/GREEN regression coverage without weakening the product boundary.

Fresh `final34-fix4d` then passed through the real multi-user SQLite backend, Next.js WebUI, Chrome `connectOverCDP`, forwarding probe, and `local-llm` / `Qwen2.5-0.5B-Instruct` target.

- All 15 canonical acceptance checks are true.
- The strict ledger is closed and clean across 282 requests, with zero request failures, page errors, runtime overlays, or classification failures.
- Three provider requests traversed unchanged; input/output hashes match and sentinel, mutation, tool, JSON, and request-bound proofs are clean.
- Race statuses are `409/200/200/409`; replay turn hashes match.
- Evidence validation returned exit `0` with no failures. Credential, absolute-path, sentinel, prompt, and answer-body scans passed.
- All five screenshots were visually inspected: desktop sources, desktop grounded/cited answers, mobile source list, mobile full-screen preview, and revoked state. No overlap, horizontal overflow, extra banner stack, unreadable evidence, or revoked-data leak was visible.
- Temporary WebUI, backend, and forwarding-probe processes were stopped after evidence capture.

### Final Review

- Review package: `review-22c9b62f69..d51f8d1be5.diff`.
- Reviewer: `01a02ca6-a0ad-7022-8bcc-f1cad8e49b25` (`Sartre`).
- Verdict: no Critical, Important, or Minor actionable findings remain; all three original findings are resolved; no forbidden local/shared scope regression found.
- Non-blocking residual: partial errors, summary/offset drift, missing target, and incomplete terminal pagination are explicit fail-closed branches but do not each have a separate direct test. Multi-page success, transport failure, and duplicate IDs are directly covered.

Status: complete. No PR or push was performed. The two unrelated untracked watchlist templates remain untouched and unstaged.
