# WebUI / Extension Performance Remediation — Coordination Index (2026-10-06)

Umbrella index for the staged execution plans drafted from the 2026-10-06 WebUI +
browser-extension efficiency review. The review audited four areas
(`apps/tldw-frontend`, `apps/packages/ui` render layer, `apps/packages/ui`
data/persistence layer, `apps/extension` + its `entries/` background code) and
confirmed ~63 findings across five systemic patterns:

1. **Streaming chat re-renders the world per token** — unthrottled full-array
   store updates, an unmemoized 3,100-line row component used as the list row,
   unvirtualized transcripts, and O(index) scans per row per render.
2. **Whole-world read-modify-write persistence** — request-history sidecars,
   workspace-store re-serialization, full `chrome.storage` area scans, and
   blob-per-collection legacy storage on hot paths.
3. **Client-side linear scans where an index fits** — Dexie JS-filter table
   scans, O(n²) dedupes, offset pagination, N+1 mirror reconciliation.
4. **Sequential N-request HTTP pipelines** — per-item ingest POSTs next to
   batch-capable endpoints, per-job status polling, N+1 fetch fan-outs, and
   interval polling where SSE streams already exist.
5. **Content-script eagerness** — full-document serialize + forced layout +
   heavy parse when a selection clip needs none of it; full-text DOM rewrites
   per stream chunk; per-byte base64.

Scope: **`apps/**` only** (WebUI + extension), with one carve-out: the
**admin WebUI** (`admin-ui/**` and admin-facing pages — Monitoring, RBAC
matrix, billing views, Llamacpp settings panels) is owned by the sibling
admin-webui program (plans A–C; review:
`Docs/Reviews/ADMIN_WEBUI_PERF_REVIEW_2026_10_06.md`). `tldw_Server_API/**` is
owned by the backend program — see
[2026-10-06-perf-remediation-coordination-index.md](2026-10-06-perf-remediation-coordination-index.md)
(TASK-13511, batches 0–7). The programs are complementary: several WebUI
findings (server-side batch/aggregation endpoints) land as backend requests and
are cross-referenced below rather than duplicated. Potential shared-file area
with the admin program: `apps/packages/ui/src/components/Common/Settings/**`
(their Plan C Llamacpp hoist-down vs this program's W4 Stage 6) — different
files, but re-check `git status` before each stage per the collision rules.

All finding locations were verified against `dev` @ `7ba48f251e` (2026-10-06).
Line numbers drift; each plan cites a grep anchor alongside line numbers.

## Batch → plan → task map

| Batch | Plan | Scope flavor | Size | Backlog task |
|-------|------|--------------|------|--------------|
| W0 | [Baseline harness](2026-10-06-webui-perf-batch-0-baseline-harness-implementation-plan.md) | Measurement infrastructure | 2–3 days | TASK-13520 |
| W1 | [Chat streaming render path](2026-10-06-webui-perf-batch-1-chat-streaming-render-implementation-plan.md) | Throttle, memo, virtualize | ~1 week | TASK-13521 |
| W2 | [Persistence write-path](2026-10-06-webui-perf-batch-2-persistence-write-path-implementation-plan.md) | History sidecar, workspace store, ModelDb, legacy DB | ~1 week | TASK-13522 |
| W3 | [Storage indexing & data structures](2026-10-06-webui-perf-batch-3-storage-indexing-implementation-plan.md) | Dexie indexes, Map/Set sweep | ~1 week | TASK-13523 |
| W4 | [HTTP architecture](2026-10-06-webui-perf-batch-4-http-architecture-implementation-plan.md) | Batching, parallelism, polling→SSE | ~1.5 weeks | TASK-13524 |
| W5 | [Extension background & content scripts](2026-10-06-webui-perf-batch-5-extension-scripts-implementation-plan.md) | Web clipper, copilot popup, warm alarms | ~3 days | TASK-13525 |

## Recommended sequencing

1. **Batch W0 first** — the streaming-render benchmark and the bundle-budget
   baseline are the verification mechanism for every later "before/after" claim.
2. **W1 next** (biggest perceived-latency win), then **W2** (typing-latency win
   that worsens with every workspace). W1 touches render components; W2 touches
   stores/services — no file overlap, can run in parallel with two executors.
3. **W3 after W2** — both touch `apps/packages/ui/src/db/**`; sequential.
4. **W5 anytime** (disjoint files from W1–W4).
5. **W4 last among client-only batches** — its polling→SSE and batching stages
   are most valuable once W0's baseline can prove them. Stages requiring new
   server endpoints (marked **[BE]** in the plan) are filed against the backend
   program's Batch 3 (N+1 batching, TASK-13515) rather than implemented here.

### Minimum viable slice

If only a subset ships, in order of leverage:
W0 → W1 Stages 1–2 (sidepanel throttle + `PlaygroundMessage` memo) →
W2 Stage 1 (request-history sidecar) → W2 Stage 2 (workspace persist debounce) →
W4 Stage 3 (buddy-host polling gate) → W5 Stage 1 (web-clipper tiered capture).

## Collision rules

- Never modify `tldw_Server_API/**` (backend workstream owns it). Server-side
  needs discovered by W4 stages are requests to the backend program, not edits.
- Re-check `git status` at each stage start; avoid files modified in the
  working tree by other agents.
- W1 and W2 are parallelizable (disjoint files). W2 and W3 both touch
  `src/db/**` — run sequentially. `PlaygroundChat.tsx`, `Message.tsx`, and
  `hooks/useMessage.tsx` are W1-exclusive.
- Behavior-preserving unless a stage explicitly says otherwise. No API contract
  changes; no storage-schema changes except W3 Stage 1 (Dexie index additions,
  versioned via Dexie upgrade — no backend migration involved).
- Every stage lands with its own commit referencing its batch task ID, tests
  written first, existing suites green, Bandit-equivalent (eslint) clean.

## Program-level Definition of Done

Per batch (enforced in each plan file):

- [ ] All stages complete; each stage landed with its own commit referencing the task ID.
- [ ] Named tests written first and passing; no existing tests disabled.
- [ ] W0 benchmarks re-run for the touched paths; delta recorded in
      `Docs/Reviews/PERF_BASELINE_WEBUI_2026_10.md`.
- [ ] `bun run lint` + `vitest run` (scoped) pass; extension batches also pass
      `test:e2e:perf` where the harness covers the path.
- [ ] Backlog task updated: notes, touched files, verification results, final summary.

## Findings → batch coverage

Finding IDs are `WP-NN` (WebUI perf), assigned by the review. Severity in
parens: (C)ritical, (M)oderate, (m)inor.

| Findings | Batch |
|----------|-------|
| WP-01..04 streaming transcripts: unthrottled sidepanel tokens, unmemoized `PlaygroundMessage`, unvirtualized `PlaygroundChat`/`ChatPane` (C) | W1 |
| WP-05..11 per-row O(n) scans, whole-store boolean selectors, unvirtualized shared pane, query-in-reducer, inline highlight/context values, per-row tooltip nodes (M) | W1 |
| WP-12..14 minor render items: flashcard panel list, index keys, unmemoized derived views (m) | W1 |
| WP-15 request-history localStorage sidecar on every HTTP call (C) | W2 |
| WP-16 workspace-store persist chain re-serialization (C) | W2 |
| WP-17 `ModelDb` full-area storage scans, per-model existence reads (C) | W2 |
| WP-18 legacy `chrome.storage` blob collections on hot paths (C) | W2 |
| WP-19 background session-state serialization per poll tick (C) | W2 |
| WP-20..22 settings write-on-read, JSON deep clones in stores, debug snapshots (M/m) | W2 |
| WP-23..24 Dexie chat search full scan + O(n²) dedupe; mirror reconcile O(n·m) + N+1 (C) | W3 |
| WP-25..31 prompt/metadata/offset scans, quota full loads, folder clear-rewrite, unbounded API caches (M) | W3 |
| WP-32..35 import O(n²)s, cleanup parallelization, Map/Set mechanical sweep (m) | W3 |
| WP-36..38 extension ingest sequential pipeline, per-job sequential polling, conference N+1/PATCH/DELETE (C/M) | W4 |
| WP-39..42 buddy 5s poll, research 5s poll beside SSE, notification backup poll, workspace double pollers (C/M) | W4 |
| WP-43..51 search waterfall/dup-fetch, N+1 fan-outs, force-refresh model catalog, fetch-all counting, client-side character filter (M) | W4 |
| WP-52..55 per-request auth storage reads, sequential config resolution, bulk endpoints, agent-loop parallelism (M/m — some **[BE]**) | W4 |
| WP-56..59 web-clipper eager extraction, parser attr pass, copilot full-text rewrite, per-byte base64 (M) | W5 |
| WP-60..63 sidepanel-open per broadcast, hourly forced warm, funnel metrics RMW, config read batching (M/m) | W5 |

## Deliberately deferred (bounded N or low frequency — revisit only if profiles implicate)

- `companion-home.ts` redundant item rebuilds (suppression already correct).
- `services/settings/local-bucket.ts` throttled cleanup parallelization.
- Prompt bulk-actions N parallel requests (parallel already; server bulk
  endpoint tracked as backend Batch 3 follow-up).
- Flashcard study panel unvirtualized transcript (n ≈ 20).
- `agent-loop.ts` sequential tool execution — behavior risk; needs its own
  design discussion before touching (WP-55, parked).

## Machine-readable baseline (filled by Batch W0)

See `Docs/Reviews/PERF_BASELINE_WEBUI_2026_10.md` after W0 lands. Every batch's
benchmark assertions reference that file's baseline numbers.
