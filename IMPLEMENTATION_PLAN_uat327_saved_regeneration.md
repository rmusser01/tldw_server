# UAT327: one saved user turn across successful regeneration

Backlog: TASK-13260.265. Native PostgreSQL evidence `.tmp/uat286-native` proves successful Regenerate inserts a second user row and loses canonical variant ancestry. Existing local-only ACK tests miss the server write.

## Stage 1: reproduce the persistence boundary
**Goal**: Prove successful regeneration must carry the acknowledged reply identity and reuse the saved user.
**Success Criteria**: Causal frontend/transport and actual-database regressions fail before repair; ordinary repeat and failed Retry controls remain distinct.
**Tests**: Saved normal Chat integration, transport payload, real SQLite/PostgreSQL message counts and parent links.
**Status**: Complete

## Stage 2: explicit saved regeneration
**Goal**: Send the saved reply ID, validate its owned current user turn and exact text/images, reuse that user, and save alternate replies with canonical parent links.
**Success Criteria**: Fresh and legacy replies group consistently; stale/foreign or changed content is rejected before persistence; intentional new sends remain new turns.
**Tests**: Causal regressions, affected retry/history/image/variant suites, scoped lint/Bandit and independent review.
**Status**: Complete

## Stage 3: native PostgreSQL acceptance
**Goal**: Verify successful image Send, Regenerate, reload and repeated new Send on committed source.
**Success Criteria**: One original user across regeneration, exact image retained, coherent answer variants after reload; a new repeated question creates its own user turn. Source parity and owned fixture cleanup recorded.
**Tests**: Native browser and canonical message reads against official PostgreSQL fixture.
**Status**: In Progress

The explicit regeneration request carries `metadata.tldw_regenerate_from_message_id`, current system instructions and the original user turn only. The existing server history window supplies preceding context. Additional supplied turns are rejected before message writes; malformed/missing conversation targets cannot create or fork a conversation. Existing persistence of an explicitly chosen workspace model setting remains unchanged. Review corrections went four causal failures to passing; final backend156 (including real PostgreSQL), HTTP/continuation6 and frontend164 pass. Bandit0; Ruff0;14existing ESLint warnings and93existing TypeScript errors unchanged. Independent review is clear. Native acceptance remains outstanding.

Native041cb46 confirms one user and two correctly linked replies, but settled reload appends an unchanged original assistant card after its collapsed2of2 variant group. Two reconciliation tests reproduce this. The fix suppresses only originals represented unchanged inside the incoming group, including every inactive local variant. Review found the inactive-variant gap; three causal failures now pass with a fully saved-group control. Final173frontend tests pass, scoped ESLint0, existing93TypeScript errors unchanged; review clear. The failed candidate remains immutable, all24970tracked entries match, and its owned runtime and official PostgreSQL holder exited cleanly. Stage3 repeats acceptance on the committed follow-up.
