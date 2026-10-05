# WebUI memory fixes — TASK-13450

**Goal:** Address all seven reviewed memory issues and open a PR against dev.
**Design:** Docs/Design/WEBUI_MEMORY_FIXES_2026_10_04.md
**Constraints:** Existing dependencies and authentication; isolated worktree; test-first changes; no unrelated UAT work.

## Stage 1: Bound comparison work
**Goal:** Share a bounded diff algorithm and terminate retired workers.
**Success Criteria:** Short-line large inputs cannot allocate a quadratic matrix; abort settles and terminates; closed modals do no work.
**Tests:** Real diff output and large-input checks; worker and modal lifecycle tests.
**Status:** Complete

## Stage 2: Own audio resources
**Goal:** Cancel retired synthesis and release every owned audio URL.
**Success Criteria:** Clear, Stop, replacement and unmount prevent late playback or URL retention.
**Tests:** Playground and document hook regression checks with deferred synthesis/fetch.
**Status:** Complete

## Stage 3: Bound document and media previews
**Goal:** Virtualize continuous PDF canvases and bound/cancel authenticated media downloads.
**Success Criteria:** Rendered canvases stay bounded with thousands of pages; oversized bodies are cancelled without full buffering; selection cleanup aborts downloads.
**Tests:** PDF viewport/navigation and media transport/hook lifecycle checks.
**Status:** Complete

## Stage 4: Bound shared caches
**Goal:** Evict expired and excess chat/character payloads.
**Success Criteria:** Idle expiry, bounded bytes/entries, request-scoped bypass and deduplication work.
**Tests:** Cache lifetime/capacity and actual API client regression checks.
**Status:** Complete

## Stage 5: Verify and publish
**Goal:** Review, run relevant checks, commit and create the dev PR.
**Success Criteria:** Regression suites pass; broader suite results, lint/type checks and security checks recorded; PR attached.
**Tests:** Targeted shared UI suites, broader UI test command, appropriate type/build checks and Bandit applicability.
**Status:** In Progress
