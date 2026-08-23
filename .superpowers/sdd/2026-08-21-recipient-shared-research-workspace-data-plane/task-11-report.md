# Task 11 Report

## Result

Complete. The explicit Chrome CDP acceptance runner exercised the real multi-user SQLite backend and Next.js WebUI using `local-llm` / `Qwen2.5-0.5B-Instruct`.

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
