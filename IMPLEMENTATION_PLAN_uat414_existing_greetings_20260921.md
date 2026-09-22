# UAT414: preserve existing conversation greetings

Task13260.278.12. Bounded continuation of the approved repair sweep.

## Stage 1: Reproduce the actual callers
**Goal**: Verify no synthetic greeting is inserted when an existing user conversation has none.
**Success Criteria**: Failing real caller tests for missing greeting, with new chat and saved greeting controls.
**Tests**: Actual useMessage and useChatActions/extracted mode, visible messages and saved history.
**Status**: Complete

## Stage 2: Minimal correction
**Goal**: Preserve observed greeting records; stop fallback to the Character default after user turns exist in either messages or history.
**Success Criteria**: All callers preserve saved transcript, first-turn greeting and retry/queue behavior.
**Tests**: Causal and surrounding suites, scoped lint/type comparison; Bandit inapplicable to TypeScript.
**Status**: Complete

## Stage 3: Native verification
**Goal**: Package new owned source and repeat no-greeting text/reload/image/reload on SQLite and official PostgreSQL.
**Success Criteria**: No invented greeting, exact canonical IDs/content/image, source/artifact receipts retained.
**Tests**: Native checks with real model; update task/tracker and remove completed plan.
**Status**: In Progress
