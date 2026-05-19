## Stage 1: Rail-Local Collapse Contract
**Goal**: Define the desired desktop sidechannel collapse and restore behavior in focused tests.
**Success Criteria**: Tests fail because the current cockpit shell lacks rail-local collapse controls and collapsed edge restore handles.
**Tests**: `Playground.cockpit-a11y.test.tsx`, `Playground.cockpit-shell.test.tsx`
**Status**: Complete

## Stage 2: Shell Implementation
**Goal**: Add discoverable collapse controls and edge restore handles to the main `/chat` cockpit shell using existing persisted rail visibility state.
**Success Criteria**: Context and Runtime sidechannels can be collapsed from their rails and restored from visible edge handles; header controls keep working.
**Tests**: Focused Vitest cockpit tests.
**Status**: Complete

## Stage 3: Real-Server Proof
**Goal**: Prove the behavior in the actual running WebUI against the real backend.
**Success Criteria**: Real-server `/chat` Playwright flow covers rail-local collapse, edge restore, existing header controls, and preserved chat/composer/status.
**Tests**: `apps/tldw-frontend/e2e/workflows/chat-cockpit.real-server.spec.ts`
**Status**: Complete

## Stage 4: Tracker Kickoff
**Goal**: Start the fresh post-merge `/chat` cockpit live audit/enhancement tracker from `origin/dev` as the next workstream.
**Success Criteria**: Tracker captures scoped `/chat` cockpit improvement follow-ups separately from this implementation slice.
**Tests**: Backlog/GitHub issue evidence only.
**Status**: Complete
