# Frontend CI failure: repaired browser event fixture

Task tracking: TASK-13263 / TASK-13263.1 (parent owns shared task updates).
Worktree: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/release-main-0.1.42`
Branch: `codex/release-main-0.1.43`; investigated HEAD `10b11f6acb`.
CI: https://github.com/rmusser01/tldw_server/actions/runs/35538062496/job/106150967108
Job: frontend-unit-tests (5/8).

## Root cause and change

CI reported exactly three failures in `apps/packages/ui/src/services/__tests__/recipe-persistence-owner.surface-bridge.test.tsx`. Each failed on real composer unmount in `chat-account-boundary.ts:40`: `browser?.storage?.onChanged?.removeListener is not a function`.

The suite's local `wxt/browser` event fixture supplied `addListener` only. Real browser events and the web shim's createEventTarget both supply `removeListener`; `useChatDraftOwner` correctly disposes that listener when the real ComposerHarness unmounts.

One-line test-only fix at line59: event fixture now supplies `removeListener: vi.fn()` alongside addListener. No production change, assertions removed, skips, timeout increase, workflow/ratchet bypass or suppression. Existing cleanup and three previously failing surface scenarios remain the regression checks.

Changed file only:
`apps/packages/ui/src/services/__tests__/recipe-persistence-owner.surface-bridge.test.tsx`.

## Verification

Fetched failed job log through `gh run view 35538062496 --job 106150967108 --log-failed`: `/tmp/release-frontend-ci-job.log`.

From `apps/tldw-frontend`, reproduced exact failure before edit:

```sh
./node_modules/.bin/vitest run --root ../packages/ui --config ../packages/ui/vitest.config.ts src/services/__tests__/recipe-persistence-owner.surface-bridge.test.tsx
```

Result: **3 failed, 1 passed**, identical removeListener cleanup error. Log `/tmp/release-frontend-ci-red.log`.

After edit:

```sh
./node_modules/.bin/vitest run --root ../packages/ui --config ../packages/ui/vitest.config.ts src/services/__tests__/recipe-persistence-owner.surface-bridge.test.tsx src/services/__tests__/recipe-persistence-owner.contract.test.ts src/services/__tests__/recipe-persistence-owner-contract.test.ts src/entries/__tests__/background.recipe-persistence-owner.test.ts
```

Result: **82 passed, 4 files passed**, Vitest4.0.18, 2.77s. Log `/tmp/release-frontend-ci-green.log`. Node emitted its existing localStorage experimental warning; no test failures or unhandled errors.

From `apps`:

```sh
tldw-frontend/node_modules/.bin/eslint --config tldw-frontend/eslint.config.mjs packages/ui/src/services/__tests__/recipe-persistence-owner.surface-bridge.test.tsx
```

Exit0, no file diagnostics. ESLint's Next plugin prints a shared-package cwd notice about no apps/pages directory; no rule disabled. Log `/tmp/release-frontend-ci-eslint.log`. Initial ESLint invocation from frontend cwd ignored the outside file; corrected invocation above actually linted it.

`git diff --check` passed.

Required Bandit invocation used server project venv on touched TSX file; Bandit is Python-only and returned a syntax parse error, so its zero findings are **not** a valid TSX security scan. Artifact `/tmp/release-frontend-ci-bandit.json`. Change is test fixture only; no runtime/security boundary modifications.

No full shard or whole suite rerun locally; the exact failing suite and related owner contract/background coverage passed. Remote CI rerun/publication remains parent-owned. No commit or push performed.
