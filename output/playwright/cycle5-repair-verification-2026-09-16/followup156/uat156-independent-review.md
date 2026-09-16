# UAT156 / TASK-13260.94 independent review

Reviewed at 2026-09-16T19:31:01.002Z.
Repository: /Users/macbook-dev/Documents/GitHub/tldw_server2
Review began at HEAD 0bdab450057822c123fae4f3f22ec43717460d2e; HEAD at report writing is b7f7f8d40f58649f170a6897fbba839232959543. Other agents committed unrelated work during review. The exact reviewed production/test hashes remained unchanged throughout.
Production baseline: 2d5ad06c86cf279fe0fcc6445009813d97ab1c1b. Its server-chat-mirror.ts is byte-identical to the author's before snapshot and the committed source checked during baseline replay.

## Verdict

No new actionable regression found in the bounded chronology repair. The source change separates canonical creation time from preserved local content in both reconciliation paths and passes the required current scope. Native source-send/reload acceptance remains pending.

One existing malformed-local fallback limitation was found by an extra private boundary probe, reproduced on both baseline and current code, and retained below. This is not a clean-pass claim for every private probe and is not presented as an introduced defect.

This review cannot close UAT013 wrong-source-answer acceptance, UAT103 historical promotion-receipt gaps, or the separate conditional missing-current-ACK/raw-wrapper recovery problem.

## Inspection

- Production change is 7 insertions / 1 deletion in server-chat-mirror.ts. No schema, dependency, provider-message, identity-correlation, owner-guard or content-selection behavior changes.
- canonicalCreatedAt accepts finite primitive numbers and rejects absent, nonnumeric and nonfinite inputs. Nullish fallback preserves epoch zero correctly.
- Memory line 150 and mirror line 211 apply the validated canonical timestamp after the protected-local-content spread. That ordering fixes the observed raw RAG user timestamp remaining 3001 while the equal-content assistant adopts canonical 2000; the canonical user timestamp becomes 1000 without replacing its raw question.
- Earlier canonical times are intentionally authoritative creation times, even when local content has a newer revision. Content-edit version protection remains unchanged. Additional probes accept earlier and later canonical times without using a min/max heuristic, preserving the protected local message and version.
- Unknown/equal/newer-version content, attached images, local IDs, canonical receipts, parent links, repeated equal-text drafts, an older unacknowledged plain retrieval-failure row, and owner/conversation/history rejection are covered by the new and adjacent suites.
- beforeAwait content preservation remains in force. The added timestamp-only probe shows a changed local timestamp cannot prevent canonical chronology adoption. Existing mounted loader tests cover typing during mirror/settings awaits and scoped publication guards.
- Actual saveMessage and formatToMessage are used by the new regression; only IndexedDB I/O and unrelated imports are replaced. Date.now()+1/+2 saves match the diagnosed post-stream persistence sequence. The real formatter sorts the persisted rows by createdAt.
- Actual loader maps invalid Date.parse results to undefined at hooks/chat/useServerChatLoader.ts:361,438. It first reconciles memory, then writes the mirror, then reconciles the formatted mirror with beforeMirror at lines 1038 and 1094-1106. The undefined new-row fallback therefore reaches final memory after mirroring.

## Fresh verification

Cwd for Vitest: /Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui

### Current required scope

```sh
./node_modules/.bin/vitest run src/db/dexie/__tests__/server-chat-chronology.test.ts src/db/dexie/__tests__/server-chat-mirror.test.ts src/db/dexie/__tests__/chat-persistence-transaction.test.ts src/hooks/__tests__/useServerChatLoader.test.ts src/hooks/__tests__/useServerChatLoader.scope.test.tsx src/hooks/__tests__/useServerChatLoader.mirror.integration.test.tsx src/hooks/__tests__/useServerChatLoader.images.test.ts --reporter verbose
```

Exit 0: 128 passed, 7 files passed, no skips. Log: /private/tmp/uat156-independent-green.log.

### Independent baseline replay

A temporary pre-load Vite plugin serves exact git-show baseline production source for server-chat-mirror.ts only; the new repository test stays untouched. The loader emits the actual baseline SHA-256 on use.

```sh
node_modules/.bin/vitest run --config /private/tmp/uat156-independent-baseline.vitest.config.mts src/db/dexie/__tests__/server-chat-chronology.test.ts --reporter verbose
```

Exit 1: 13 behavioral failures / 4 controls passed. The first failure explicitly shows assistant2000 before raw-user3001. Additional failures cover protected edits, beforeAwait edits, invalid canonical dates, epoch0 and retained drafts ordering. Log: /private/tmp/uat156-independent-red.log.

### Additional independent timestamp probes

Private loader appends cases from /private/tmp/uat156-independent-extra-cases.ts in memory without changing the repository test file.

```sh
./node_modules/.bin/vitest run --config /private/tmp/uat156-independent-extra.vitest.config.mts src/db/dexie/__tests__/server-chat-chronology.test.ts --reporter verbose
```

Exit 1: 31 passed / 1 failed across the 17 permanent plus 15 private cases. These overlap the 128-test run and must not be added as a combined test total. Passing added cases cover numeric-looking/empty strings, booleans, object, array, Date object, boxed Number and bigint timestamps while preserving a valid local epoch0; captured-local fallback; earlier/later canonical dates with newer protected content; timestamp-only changes; and the real-loader-shaped undefined fallback with repeat-load stability.

The one failure is the existing malformed-local boundary described next. Log: /private/tmp/uat156-independent-extra-final.log.

## Existing limitation: invalid captured-local timestamps are not sanitized

A new incoming canonical row with createdAt=NaN and no existing local row is returned unchanged by reconcileServerChatMessages (line143). If that output is then supplied as localMessages to mirror, the captured local row keeps NaN through captured.createdAt ?? Date.now() (line194). The new canonical validator rejects the remote NaN, but the fallback local.createdAt is also NaN, so it survives at line211. The same broad limitation applies to already-invalid local fallback values; the repair validates canonical input, not historical local timestamp integrity.

This synthetic NaN sequence fails identically on exact baseline and current source. The equivalent sequence with createdAt=undefined, as produced by the real loader's invalid-date mapping, passes both and stabilizes at the first fallback time on reload. No real-loader regression or native corruption path was established. Consequently this is a nonblocking pre-existing boundary limitation, not a request to expand UAT156. Do not claim all malformed local data is repaired.

Baseline classification command:

```sh
./node_modules/.bin/vitest run --config /private/tmp/uat156-independent-extra-baseline.vitest.config.mts src/db/dexie/__tests__/server-chat-chronology.test.ts --testNamePattern 'gives new invalid-dated|assigns a stable new-row' --reporter verbose
```

Exit 1: 1 failed (NaN), 1 passed (undefined), 30 excluded by the name filter. This is deliberately a selected diagnostic, not a broad passing test run. Log: /private/tmp/uat156-independent-boundary-baseline.log.

## Other checks and exact hashes

- Scoped git diff --check: exit 0.
- All 11 file hashes in the author's /private/tmp/source013-chronology-repair-20260916/manifest.json verified successfully.
- Final reviewed source/test hashes match the author's freeze and the initial review hashes.
- Node's experimental localStorage warning appears in the test environment; no current required-scope failure or unhandled-error report.

```text
655099dfa7452eb8105408bfd62cfdd8b2f84da05d5e421c0d5d63fff79dc0c2  apps/packages/ui/src/db/dexie/server-chat-mirror.ts
f0b0d19fb6027bf0ba70367c009c3e49a847193008c180ed076497d23d239a38  apps/packages/ui/src/db/dexie/__tests__/server-chat-chronology.test.ts
9474ea785c871a25954979efc07d34b344f3fbf09b53aec5348de63cbfbbc7c7  baseline server-chat-mirror.ts
```

## Private evidence SHA-256

```text
e6fabc34ac894c60fe61d0c6b1dfbc53384e2de350de881086da5292ce7ec66a  /private/tmp/uat156-independent-green.log
0cf2eb6ed24d0bb133c5af5df97bc71d8d2d127c28f2374628c05ecf7893f557  /private/tmp/uat156-independent-red.log
9474ea785c871a25954979efc07d34b344f3fbf09b53aec5348de63cbfbbc7c7  /private/tmp/uat156-independent-baseline-source.ts
74582268af14ed36323b7c96ec291c8cea1f739c7e214d4257f35c505f97f505  /private/tmp/uat156-independent-baseline.vitest.config.mts
66531ad55889642416355563b777c34012838117c0c249f986f63c17fdb2d46f  /private/tmp/uat156-independent-extra-cases.ts
11afb3e847d2ece306da51202ff453462756c9cd41da0b7cd65b5f8ca2c98e35  /private/tmp/uat156-independent-extra.vitest.config.mts
df292fa8e07cfa4bcd582e25ee278beb9507ee1beec861cee18fdd0f2a411564  /private/tmp/uat156-independent-extra-baseline.vitest.config.mts
9e15340745c6a9cdc0f9254f3a8a21a786416c9b678af44f335c275b95e2331e  /private/tmp/uat156-independent-extra-final.log
55720c3c2e315800a0a3fb3c8e23d41ecf64b5f2e38353ea0c75d1c4eccd80a5  /private/tmp/uat156-independent-boundary-baseline.log
```

## Limits and ownership

- No native browser, app runtime, inference or provider request was started or driven; the disappeared latest native transcript/profile evidence was not reconstructed or certified.
- The new chronology fixture uses an in-memory stand-in for IndexedDB. Real-browser storage behavior and exact source send/reload acceptance remain root-owned.
- No full application build/typecheck, complete repository suite, or independent lint rerun was performed. Author ESLint reports were hash-verified but their zero-error/warning claim was not independently rerun here. Bandit does not analyze TS-only changes; no Python was changed.
- No production, repository test, task, root-document, staging or commit modifications were made by this reviewer. Private probes and this report are under /private/tmp. Other working-tree changes were left untouched.
