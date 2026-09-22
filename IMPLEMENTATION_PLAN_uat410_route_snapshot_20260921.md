# Keep navigation snapshots coherent (UAT410)

TASK13260.278.3.4. Diagnostic9 PostgreSQL verifies source citation identity, previews, canonical source opening and saved QA reload, but Chat handoff is redirected back to Media with the Chat query. The WebUI router shim reads path from Next.js and query/hash from the browser; asynchronous navigation can mix different destinations.

## Stage 1: Reproduce the shared owner
**Goal**: Exercise the actual shim with router/browser snapshots skewed in both directions.
**Success Criteria**: Consumer-visible route never combines separate destinations; encoded query and fragment remain intact.
**Tests**: Causal location cases plus existing navigation regressions.
**Status**: Complete

## Stage 2: Minimal shared repair
**Goal**: Derive all location fields from the committed router snapshot.
**Success Criteria**: Existing navigation consumers retain expected URLs and Media-to-Chat uses one coherent destination.
**Tests**: Scoped Vitest, frontend types/lint, review of permalink lifetime.
**Status**: Complete

## Stage 3: Native follow-up
**Goal**: Repeat exact citation/QA/Chat workflow on new attributable production artifacts.
**Success Criteria**: Both SQLite and official PostgreSQL preserve source identity, navigate to Chat and save/reload a grounded turn.
**Tests**: Required bounded UAT390 journey, zero retries, retained diagnostic9 failures.
**Status**: In Progress

## Related UAT411 (TASK13260.278.3.5)
The actual Media component independently reproduces transient id removal during initial hydration (32 controls pass/1failure). The hydration effect queues pending state; permalink persistence runs in that same render without the pending state and removes the incoming id. Capture whether that URL still needs hydration before effects run and defer persistence until hydration owns it. Preserve existing deleted-selection cleanup and strict native ID assertions.

Verification: UAT4103red/13controls; UAT4111red/32controls; UAT4125fixture failures reproduced on committed baseline then repaired without changing assertions. Final57pass across5files,0fail/skip. TypeScript and scoped lint pass (existing warnings). No changed Python/Bandit scope. Root review confirms one snapshot supplies all location fields and incoming URL hydration cannot race permalink cleanup. Native rebuilt acceptance remains open.
