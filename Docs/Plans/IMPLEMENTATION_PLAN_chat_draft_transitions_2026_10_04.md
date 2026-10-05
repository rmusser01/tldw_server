# Explicit conversation draft transitions

**Task/spec:** TASK-13454; approved five-stage Chat UX design. Key lifecycle foundation archived in `f3a346bbb7`; latest dev fetched at `61589721bc` for integration after the reviewed runtime checkpoint.
**Goal:** Wire explicit source-checked draft switching and first-ID binding through actual WebUI and Sidepanel producers, then make retained fresh/temporary drafts reachable across reload/departure.
**Architecture:** Existing option core holds a typed credential-free address plus transient source/revision/kind. Existing third H1 onLoaded callbacks qualify deliberate selection and first creation; raw ID setters, mirror attachment, refresh and onCapture do not imply navigation. Shared composer/persistence carries current text and collapse metadata on bind only if its previously committed source key matches; mounts hydrate normally. Persist verified-owner handles in existing session/tab snapshots, never transient intent or leases. Reuse owner-filtered enumeration of the existing draft bucket for abandoned fresh drafts. Keep source records until replacement retention is proven; local bucket best-effort writes are not that proof.
**ADR assessment:** No new ADR/API or authority/framework. ADR-049/050 fences remain; TASK-13160 unsupported-native capability and unknown-create reconciliation remain separate Stage 1 gates.

## Stage 1: Runtime address and bind regressions
**Goal:** Establish real store/registry coverage of switches, binds and stale acceptance.
**Success Criteria:** Store source/revision/account guards reject obsolete transitions; first-ID bind retains NEXT text/metadata; a remount hydrates normally.
**Tests:** Existing real keyed registry/composer harness and actual store/controller producer suites, with meaningful red cases before production changes.
**Status:** Complete

## Stage 2: Wire accepted production boundaries
**Goal:** Both composer surfaces consume addresses; actual New/character/history/creation/fork/handoff producers publish explicit intents.
**Success Criteria:** A→B→A retains correct drafts, failed/cancelled/obsolete loads retain source, first local/native ID binds current next draft, refresh and mirror attachment leave address unchanged, compact inactive-tab ACK cannot alter active draft.
**Tests:** Shared composer/tab/account tests plus affected actual loader/send/clear/character/fork suites; WebUI typecheck.
**Status:** Complete

## Stage 3: Durable handles and reachable recovery
**Goal:** Fresh/temporary draft-only views restore through verified-owner handles; abandoned fresh drafts remain discoverable.
**Success Criteria:** Existing session/tab structures persist handles immediately, exclude temporary transcripts and transient binds, validate scope/owner, override stale history fallback, and offer filtered retained-draft recovery. Suppress hybrid target-ID/source-address session saves and fence async saves by draft revision.
**Tests:** Fresh/temp reload before debounce, outdated bookmark/recent-chat precedence, owner/workspace/profile mismatch, prior-fresh recovery, restricted storage and logical-tab isolation.
**Status:** In Progress

## Stage 4: Verify and review
**Goal:** Reviewed incremental deliveries without data loss/privacy regressions.
**Success Criteria:** Meaningful affected suites/typecheck/lint/diff/task normalization pass; both authorized independent reviewers resolve Critical/Important findings; local commits record evidence and remaining Stage 1 gates.
**Tests:** Relevant suites and baseline-relative lint. Bandit N/A for TS/docs-only scope. Broader saving, creation-uncertainty, greeting, usability, accessibility and native-shell journeys remain required under the approved program.

Runtime checkpoint: 19 affected suites, 569 tests pass; WebUI typecheck passes; baseline-relative touched lint adds no diagnostics; diff check passes. Both independent reviewers report no remaining Critical/Important runtime findings. Production regressions cover first local/native send with real H1 fences, controller/display target mismatch, Temporary reads, source setup during provisional Sidepanel loads, cancelled close, first-ID bind and hydration readiness before queue/reset. Stage 3 and the broader program remain required; no TASK-13454 acceptance criterion is closed by this checkpoint.
**Status:** In Progress
