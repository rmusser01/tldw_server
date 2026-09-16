# UAT135 / TASK13260.74

Frozen 2026-09-16T13:40:48.746Z. Parent independently accepted134 before releasing shared connection files. Its prior manifest remains historical; this manifest contains subsequent135 shared bytes.

## Correction

Opaque Failed-to-fetch/network/abort errors cannot establish a CORS denial. Removed the store helper inventing denied origins and disable-CORS advice; original errors remain intact. Retained existing network-pattern recovery checks. Diagnostics now uses current UX/auth failure instead of stale configStep=auth as proof of missing credentials; browser failure guidance is cause-neutral. No credentials, transports or polling policies change in135.

## Verification

- Permanent RED5/4 controls: /private/tmp/cycle5-repair-ui-135-red.log. Actual store -> normalized transport response -> actual mounted Knowledge diagnostics. Four network/timeout variants and reconnect still claimed missing credentials; four auth/permission controls passed.
- GREEN89/4: -green.log includes new diagnostics integration, full connection store,134 actual readiness-refresh boundary, and existing design-system labels. Missing-key early no-request,401/403,explicit allowlist, valid recovery and retry action retained.
- Command from apps/tldw-frontend: bun run test ../packages/ui/src/store/__tests__/connection.diagnostics.integration.test.tsx ../packages/ui/src/store/__tests__/connection.test.ts ../packages/ui/src/store/__tests__/connection.readiness-refresh.test.ts ../packages/ui/src/components/Option/KnowledgeQA/__tests__/SetupDiagnostics.design-system.test.tsx --maxWorkers=1 --no-file-parallelism
- Scoped ESLint0 errors14unchanged warnings; -lint-comparison.json/raw before-after. git diff --check passes. Final whole typecheck running in /private/tmp/cycle5-repair-ui-final-typecheck.log (prior134 exact90 baseline). Bandit not applicable TS-only.

## Limits

The diagnostics test controls normalized apiSend and TldwClient config seams; actual rendering/store classification is exercised. Separate134 tests exercise actual fetch/core/credential path. No browser/live server/inference, runtime changes, staging or commits. Parent native outage/recovery acceptance still required. Existing diagnostic tests that expected fabricated CORS copy now assert original network/abort errors.

## Scope

- apps/packages/ui/src/store/connection.tsx
- apps/packages/ui/src/components/Option/KnowledgeQA/SetupDiagnostics.tsx
- apps/packages/ui/src/store/__tests__/connection.test.ts
- apps/packages/ui/src/store/__tests__/connection.diagnostics.integration.test.tsx
- backlog/tasks/task-13260.74 - Show-accurate-Knowledge-connection-diagnostics-during-an-outage.md

Hashes /private/tmp/cycle5-repair-ui-135-manifest.json.

Final integrated compiler completed: {"current":90,"baseline":90,"added":[],"removed":[]}. Log/comparison /private/tmp/cycle5-repair-ui-final-typecheck{-comparison.json,.log}.
