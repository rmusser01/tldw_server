# Web Scraping Phase 5 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Separate crawl and process-local queue ownership without losing analyzer, legacy, service, or downstream behavior, while bounding actual work and making interruption truthful.

**Architecture:** Crawl owns discovery/frontiers/checkpoints; jobs owns admission, transitions, subscriptions, and history. Orchestration injects canonical article execution and live parent controls. Services retain verified authorization, summaries, ingestion, persistence, and lifecycle integration.

**Tech Stack:** Existing Python/asyncio/dataclasses, pytest/Hypothesis, HTTPX/curl_cffi, governed Playwright/Chromium, existing extraction executor, Loguru, backlog-py. No new runtime dependency or central Jobs migration.

**Spec:** [Approved Phase 5 design](../specs/2026-10-04-web-scraping-phase-5-crawl-process-local-jobs-design.md).

**Decision:** [Accepted ADR-093](../../ADR/093-web-scraping-crawl-and-process-local-job-ownership.md). No new ADR: this plan implements that ownership decision without changing it.

**Task:** TASK-15432 (planning complete); TASK-15421 (execution). **Status:** Approved for subagent-driven execution by the human requester on 2026-11-01.

## Global Constraints

- Existing public `CrawlRequest` and `CrawlResult` retain constructor and export compatibility.
- Neither `crawl/` nor `jobs/` imports orchestration, application services, or legacy wrappers to execute a page.
- Preserve pre-scrape analyzer configuration, advice precedence, attached results, explicit extraction choices, cookies, deduplication, and non-secret article extra fields/plain-dict metadata.
- Preserve legacy successful-article page counting versus enhanced collected-result counting, traversal order, filters, scores, seed-relative depth, and omitted-versus-explicit options.
- Parent cancellation, deadline, and budget exhaustion cannot become analyzer fail-open, browser fallback, or queue retry. Async cancellation remains `CancelledError`.
- Queue admission is immediate; retry reservations and physical-drain leases are not available to competing submissions. No unbounded external admission waiters.
- Batches materialize only a bounded window; cancellation owns only their accepted children. Never shadow live network, analyzer, summary, or ingestion work.
- Terminal crawl outcome publishes once after crawl/checkpoint finalization and before service post-processing. Later service cancellation does not rewrite it.
- Keep established structured failure lists before and during traversal; use a typed partial-result interruption only where no such representation exists.
- Enhanced progress JSON remains a snapshot. Legacy local `resume_file` remains supported; checkpoints do not promise exactly-once I/O/ingestion or crash-proof lifetime accounting.
- Running synchronous work is drained, not claimed killed. Preserve executor/browser permit, reload, shutdown, fork, attestation, and guarded-egress behavior.
- Every repository edit has a Backlog task first. Track each implementation task under a separately reserved child task; use backlog-py and existing ADR assessment, TDD, review, verification, Bandit, and commit gates.
- No automatic merge. AI-authored PRs require the requester's own `Change summary` explaining what and why under the repository's merge policy.

## Review Focus

1. Queue full while every worker retries: reservations prevent deadlock; stop settles every subscription once (Task 3, `test_full_queue_retry_stop_interleaving`).
2. Cancellation after some batch jobs were accepted: retain completed articles, cancel/observe only owned unfinished children, and keep physical capacity charged (Task 8, `test_partial_admission_abort_owns_children`).
3. Parent stop inside optional analysis or a fallback: no subsequent network/browser attempt or local fail-open success (Task 5, `test_parent_stop_escapes_all_fallbacks`).
4. Resume with an in-flight page and a still-draining prior execution: refuse overlap, preserve frontier order and known spent work, never serialize credentials (Task 7, `test_resume_waits_for_physical_owner`).
5. Cancellation after crawl publication during summary/write: immutable nested crawl outcome, canceled service operation, no new writes, and committed data preserved (Task 10, `test_late_service_cancel_keeps_crawl_outcome`).

---

## Baselines And Execution Preparation

The design worktree is `codex/web-scraping-phase-5-design`, based on
`75ab224081bf140ef52017c1a9b0a04f6878d488`. Read-only integration inspection used
`origin/dev` at `54b321dcec8cdbb6358581f42c1548696260570b` on 2026-11-01.
This plan does not claim the branch is rebased or the new behavior is tested.

After plan approval, use the git-worktrees skill to prepare an isolated execution
branch with prefix `codex/`. Fetch latest `dev`; record its immutable SHA and recheck
the files/interfaces below against it. Use a clean execution worktree, replay the
scoped approved docs/task commits, and resolve documentation collisions before code.
Do not rebase/reset the dirty main checkout. If latest code changes an approved
contract, return to review rather than silently editing this design. Reserve child
task IDs above local/dev/all paginated open-PR task IDs; preserve approval evidence.

All commands below run at the execution worktree root after activating its project
venv (`source .venv/bin/activate`), or the existing parent repository venv when the
worktree has none. Never install a second dependency stack just for the worktree.
Use fake clocks/events/barriers, not sleeps or external network, for unit tests.

## Concrete Safety Defaults

These are proposed implementation values for this plan's approval, not measurements
or new public configuration keys. Define frozen validated `CrawlLimits`,
`QueueLimits`, and `CheckpointLimits`. Reject booleans as integers, non-finite
durations, negative/zero values (except existing zero history retention), and
incoherent worker/count settings. Boundary tests cover exact-limit acceptance and
one-over rejection. Existing per-page limits are unchanged.

| Owner / field | Default | Meaning |
| --- | ---: | --- |
| Crawl `page_attempts` | 1,000 | Each started logical page attempt, including job retry |
| Crawl `network_requests` | 20,000 | Each admitted governed HTTP request/redirect/subresource/WebSocket handshake |
| Crawl `browser_attempts` | 1,000 | Each article/preflight browser acquisition attempt |
| Crawl `active_probes` | 2,000 | Each active probe operation, including an external-tool invocation |
| Crawl `transfer_bytes` | 1,073,741,824 (1 GiB) | Aggregate observed acquisition bytes; raw HTTP chunks, CDP transfer/frame bytes, bounded tool output |
| Crawl `discovered_candidates` | 50,000 | Candidates inspected, including duplicate/rejected discoveries; not only accepted links |
| Crawl `max_url_bytes` | 65,536 (64 KiB) | UTF-8 ceiling per raw/normalized target, resolution base, request/final location or retained URL identity; execution ruling during Task 6 |
| Crawl `discovered_url_bytes` | 67,108,864 (64 MiB) | Cumulative UTF-8 discovery locations and normalized candidate/parent pairs, separate from transfer bytes; execution ruling during Task 6 |
| Crawl `frontier_entries` | 10,000 | Current queued plus in-flight entries; released on committed disposition |
| Crawl `execution_seconds` | 1,800 | Active execution including admission/rate/retry waits, not paused checkpoint time |
| Queue `max_pending` | 128 | Pending slots plus reserved return-to-queue slots of active retryable jobs |
| Queue `max_live` | 256 | Accepted jobs, including logically terminal but physically draining work |
| Queue `max_payload_bytes` | 262,144 (256 KiB) | One normalized private execution payload |
| Queue `max_total_payload_bytes` | 16,777,216 (16 MiB) | Aggregate accepted private execution payloads, including drain |
| Queue `max_history_bytes` | 67,108,864 (64 MiB) | Serialized allowlisted completed-history projections |
| Existing worker/history count defaults | 5 / 1,000 | Preserve current constructor options, including retention 0 |
| Batch producer window | `min(max_concurrent, max_pending, max_live)` | Default 5; also limited by remaining parent page allowance |
| Checkpoint `max_file_bytes` | 134,217,728 (128 MiB) | Bounded load and streaming write; overflow is explicit checkpoint failure |
| Checkpoint collection ceilings | 50,000 candidates / 10,000 frontier / 1,000 results | Align with default crawl dimensions; validate against injected effective limits |
| Logical cleanup allowance | 2 seconds | Existing bounded-cleanup convention, separate from execution deadline |
| HTTP streaming chunk ceiling | 65,536 bytes | Bounded temporary read chunk; stop before retaining a chunk over allowance |
| Existing article/browser byte defaults | 16,777,216 / 67,108,864 | Per-page caps remain in `ArticleLimits`; parent budget is additional |

Source/override rule: defaults live in the owning internal limits dataclasses.
Trusted composition-time injection may replace them with other finite validated
values; do not read them from arbitrary client options/context metadata. Preserve
existing worker/count-retention parameters and article configuration precedence.
Existing `max_pages` does not redefine or replenish parent limits. A request for more
work than an allowance stops explicitly with partial/failed outcome, not silent
success. No new environment variables, HTTP fields, or admin settings in this phase.
Checkpoint ceilings follow an injected larger crawl allowance only when its trusted
checkpoint limits are raised explicitly too; never silently discard stored articles.

Accounting measures admitted operations and observed bytes, not kernel packet
counts or total process RSS. Browser callbacks can observe an already-received
chunk before closing transport; no subsequent content is retained after exhaustion.
Opaque external tools have invocation/time/output bounds, not a claim to measure
their internal HTTP requests or wire bytes. Preserve their existing enablement and
egress checks; report that accounting scope explicitly rather than inventing counts.

## File Map

All module paths in task blocks are relative to
`tldw_Server_API/app/core/Web_Scraping/` unless prefixed `tldw_Server_API/`.
All test names are under `tldw_Server_API/tests/`. New packages receive minimal
`__init__.py` exports only; do not make them re-export legacy modules.

| Files | Responsibility |
| --- | --- |
| `contracts/execution.py` (new) | Internal discovery/page/outcome/report values, not replacement public models |
| `runtime/controls.py` (new) | Lower-level control protocol, reservations, generic execution/drain handle, typed parent stop |
| `crawl/models.py`, `profiles.py`, `frontier.py`, `discovery.py`, `engine.py` (new) | Immutable resolved plans, exact compatibility profiles, bounded strategies and injected traversal |
| `crawl/checkpoints.py` (new) | Versioned parsing, legacy migration, writer ownership and atomic checkpoint storage |
| `jobs/models.py`, `admission.py`, `queue.py`, `projections.py` (new) | Existing public enums/job values plus private payload, capacity, queue state and safe history |
| `orchestration/crawl_context.py`, `crawl.py`, `batch.py` (new) | Live controls, canonical acquisition composition, bounded batch production |
| `orchestration/article.py`, `article_browser.py`; `runtime/fetch.py` | Optional internal controls/discovery hooks on existing acquisition, no second article engine |
| `preflight/context.py`, `facade.py`, `adapters/{http,browser,external_tools}.py`, `runner.py` | Parent/local composition and parent-stop escape without changing standalone analyzer behavior |
| `tldw_Server_API/app/core/http_client.py` | Narrow generic streaming observer hook; no dependency on Web_Scraping |
| `Article_Extractor_Lib.py`, `enhanced_web_scraping.py`, `contracts/conversion.py` | Signature-preserving compatibility adapters and exports |
| `tldw_Server_API/app/services/{enhanced_web_scraping_service,web_scraping_service}.py` | Trusted composition, outcome propagation and existing post-processing |
| `tldw_Server_API/app/api/v1/endpoints/web_scraping.py`, `media/ingest_web_content.py` | Verified lifecycle authorization and HTTP/ingestion outcome projection |
| `tldw_Server_API/app/api/v1/schemas/media_request_models.py` | Only additive receipt/schema changes where response models otherwise drop outcome |
| `Web_Scraping/phase5_fakes.py` (test helper, new) | Fake clock, barrier executor/transport, controlled pipeline fixture and call ledger |
| `Docs/Design/WebScraping.md`, inventory Markdown/JSON | Refresh actual dependencies and supported contracts, not aspirational completion |

Check existing adapter/document names on the execution baseline before editing;
if a listed file has moved, record the exact replacement in this plan and child
task before its first edit. Consumer changes are limited to actual inventory rows
that unwrap/catch outcomes; do not churn unrelated import users.

## Stage 1: Characterization And Internal Contracts

**Goal:** Freeze supported behavior and define the interfaces subsequent tasks consume.
**Success Criteria:** Profile matrix and private values tested; existing public constructors/signatures unchanged.
**Tests:** Task 1 parity matrix and Task 2 limit/outcome/accounting tests.
**Status:** Complete (Tasks 1-2 independently reviewed; baseline gates remain recorded).

### Task 1: Characterize Every Entry Profile

**Files:** Create tests `Web_Scraping/phase5_fakes.py`, `test_phase5_profile_contracts.py`;
extend `WebScraping/test_phase1_compatibility_contracts.py`,
`Web_Scraping/test_crawl_config_precedence.py`, `test_recursive_crawl_semantics.py`,
`WebScraping/test_refactor_import_inventory.py`.

**Interfaces:** Consumes current public functions/classes. Produces test helper
`make_harness(*, profile: str, outcomes: Sequence[dict[str, Any]]) -> Phase5Harness`
with fake clock, transport/executor release events, calls, captured callbacks,
credentials, and canonical result fixtures. Its `run()` calls the current adapter;
later tasks add `run_report()` without changing captured baseline expectations.
Profile manifest names: `legacy_recursive`, `enhanced_recursive`, `legacy_sitemap`,
`legacy_url_level`, `enhanced_sitemap`, `enhanced_url_level`, `enhanced_batch`.

- [x] **Step 1: Write characterization assertions and the manifest-completeness test.**

```python
def test_profile_manifest_covers_public_crawl_imports(profile_manifest, inventory):
    assert inventory.crawl_entrypoints - profile_manifest.entrypoints == set()

async def test_legacy_midcrawl_transport_failure_contract(harness):
    articles = await harness.legacy_transport_failure_after_article()
    assert articles[0]["content"] == "kept article"
    assert articles[1]["error"] == "browser_transport_unavailable"

async def test_profile_page_counters(harness):
    assert await harness.attempted_urls("legacy_recursive", max_pages=1) == ["/bad", "/good"]
    assert await harness.attempted_urls("enhanced_recursive", max_pages=1) == ["/bad"]
```

Keep the current structured failure's `error` key, not a renamed error field.
Add signature/export/constructor
snapshots, FIFO/best-first order, depth/scoring/tags, all sitemap formats, robots,
sync/async envelopes, nested metadata/extra fields, callbacks, and option presence.
Mark false completion/checkpoint deletion and unsafe projections as intentional
corrections with separate regression tests in owning tasks, not frozen parity.

- [x] **Step 2: Run `python -m pytest tldw_Server_API/tests/Web_Scraping/test_phase5_profile_contracts.py -q`.**
Expected: incomplete manifest test fails until all current inventory entry points
are mapped; characterization assertions pass against existing behavior.
- [x] **Step 3: Complete the test-only profile manifest/helper from actual imports and three existing patterns: legacy traversal, enhanced traversal, and service precedence.**
No production refactor in this task; record each consumer suite needed in Task 11.
- [x] **Step 4: Rerun the command and the three extended compatibility suites.** Focused/compatibility suites pass; unchanged inventory artifact line drift remains disclosed for Task 11. No live network or side effects.
- [x] **Step 5: Commit the explicit changed test paths and child task:** `test(web-scraping): characterize Phase 5 compatibility profiles`.

### Task 2: Define Controls, Limits, And Internal Reports

**Files:** Create `contracts/execution.py`, `runtime/controls.py`, `crawl/models.py`,
`crawl/profiles.py`, `orchestration/crawl_context.py`; create tests
`Web_Scraping/test_phase5_contracts.py`, `test_phase5_controls.py`.

**Interfaces:** All new types below are internal; preserve `contracts/requests.py`
and `results.py` public constructor fields. Define:

```python
Counter = Literal["page_attempts", "network_requests", "browser_attempts", "active_probes", "transfer_bytes", "discovered_candidates"]
class Reservation:  # runtime/controls.py; unused reservations roll back once
    def commit(self) -> None: ...
    def close(self) -> None: ...
class ParentControls(Protocol):
    def check(self) -> None: ...
    def reserve(self, counter: Counter, amount: int = 1) -> Reservation: ...
    def charge_bytes(self, amount: int) -> None: ...
    def cap_timeout(self, requested: float | None) -> float: ...
@dataclass(frozen=True)
class WorkHandle(Generic[T]):
    result: asyncio.Future[T]
    drained: asyncio.Future[None]
    cancel: Callable[[], None]
```

`ParentStop(reason: str)` is a typed exception restricted to deadline/budget reasons;
cancellation uses `CancelledError`. Controls are thread-safe, not serialized.
`CrawlContext(limits: CrawlLimits, *, clock: Callable[[], float])` implements them
and exposes `cancel() -> None`, `attach(handle: WorkHandle[Any]) -> None`,
`snapshot() -> BudgetSnapshot`, `publish(outcome: CrawlOutcome) -> CrawlOutcome`.
`publish` atomically honors earlier cancellation and never rewrites a terminal value.
`BudgetSnapshot` records counters, remaining_seconds, and accounting completeness.
Define `CrawlControls(ParentControls, Protocol)` in `crawl/models.py` with the same
`cancel`, `attach`, `snapshot`, and `publish` signatures. The engine consumes this
protocol; only orchestration constructs `CrawlContext`. Execution ruling: add
internal `note_successful_result() -> None` on both protocol and context, serialized
with stop/publication. It marks successful logical-result retention, not a page
counter; late marks after stop/terminal publication are rejected. Publication can
then report partial presence accurately when cancellation/exhaustion wins over a
completed candidate. Task 6 calls the marker at result commit without an intervening await.

In `contracts/execution.py`: frozen `DiscoveredLink(url: str, parent_url: str)`,
`DiscoveryResult(request_url: str, final_url: str, links: tuple[DiscoveredLink, ...])`,
`PageData(article: dict[str, Any], discovery: DiscoveryResult)`,
`CrawlOutcome(state: Literal["completed", "partial", "failed", "cancelled"], stop_reason: str, has_partial_results: bool)`,
`CrawlReport(articles: tuple[dict[str, Any], ...], outcome: CrawlOutcome, diagnostics: tuple[str, ...] = ())`,
`CrawlInterrupted(report: CrawlReport)` carrying sanitized partial results.
Validate reasons against the approved spec; failure entries alone are not articles.

`CrawlPlan(request: CrawlRequest, profile: str, options: Mapping[str, Any], provenance: Mapping[str, str], limits: CrawlLimits)`
is frozen, credential-free, with deeply frozen options. `resolve_crawl_plan(request:
CrawlRequest, *, profile: str, options: Mapping[str, Any], supplied: frozenset[str],
config: Mapping[str, Any], limits: CrawlLimits) -> CrawlPlan` retains presence and
existing precedence. Execution credentials travel separately, never in `options`.

- [x] **Step 1: Write limit, counter, terminal-publication, and presence tests.**

```python
def test_started_work_not_refunded(ctx):
    lease = ctx.reserve("page_attempts")
    lease.commit()
    lease.close()
    assert ctx.snapshot().page_attempts == 1

def test_failure_entries_do_not_make_partial(outcome_factory):
    report = outcome_factory.interrupted([{"error": "unavailable"}])
    assert report.outcome.state == "failed"
    assert report.outcome.has_partial_results is False

def test_terminal_publication_immutable(ctx, completed):
    assert ctx.publish(completed) is completed
    ctx.cancel()
    assert ctx.publish(completed) is completed
```

Also test all default values, invalid/finite values, pre-start rollback, concurrent
reserve boundary, byte overflow latch, cancel-before-publish, and explicit default
versus omitted options. Use Hypothesis for reservation/close invariants.
- [x] **Step 2: Run `python -m pytest tldw_Server_API/tests/Web_Scraping/test_phase5_contracts.py tldw_Server_API/tests/Web_Scraping/test_phase5_controls.py -q`.** Expected: missing internal types/methods fail.
- [x] **Step 3: Implement these values and controls with one lock protecting cumulative counters, stop latch, and publication; keep active-capacity leases distinct.** Copy profile resolution from characterized rules rather than normalizing differences.
- [x] **Step 4: Rerun Task 2 tests and Task 1 parity.** 233 passed after review fixes; public signatures unchanged.
- [x] **Step 5: Commit explicit paths and child task:** `feat(web-scraping): define bounded crawl execution contracts`.

## Stage 2: Queue Ownership And Service Authorization

**Goal:** Independently tested bounded queue and verified lifecycle access, without an early traversal switch.
**Success Criteria:** Admission/retry/stop/drain races settle once; lifecycle access uses trusted principals.
**Tests:** Tasks 3-4 queue/projection/security suites and existing retention/admin suites.
**Status:** Complete (Tasks 3-4 independently reviewed; physical acquisition wiring follows in Stage 3).

### Task 3: Own Admission, Retry, Completion, And Drain

**Files:** Create `jobs/models.py`, `admission.py`, `queue.py`, `projections.py`;
create tests `Web_Scraping/test_phase5_queue_admission.py`, `test_phase5_queue_lifecycle.py`,
`test_phase5_queue_history.py`. Do not switch the enhanced wrapper yet.

**Interfaces:** Move/copy existing `JobPriority`, `JobStatus`, and `ScrapingJob`
definitions without changing constructor/default/public values; Task 9 re-exports
them. Private `JobPayload` is a validated normalized snapshot of operation, URL,
owner and execution options/credentials, with a precomputed UTF-8 JSON byte size;
callbacks/live controls are separately injected, not serialized or sized as JSON.
Reject unsupported arbitrary objects instead of `str()`/`default=str` coercion.
Temporary sizing bytes are discarded; private values are never logged.

`ProcessLocalJobQueue(*, execute: Callable[[JobPayload, ParentControls | None], WorkHandle[PageData]],
max_concurrent: int = 5, completed_retention: int = 1000,
limits: QueueLimits = QueueLimits())` exposes async `start() -> None`,
`stop() -> QueueStopReport`, `add_job(job: ScrapingJob) -> asyncio.Future[dict[str, Any]]`,
`cancel_job(job_id: str) -> ScrapingJob | None`, plus internal async
`submit_page(job: ScrapingJob, *, controls: ParentControls) -> WorkHandle[PageData]`.
Keep existing public polling/status methods through Task 9 adapters.
`QueueStopReport(state: str, draining_jobs: int)` distinguishes stopping/stopped.
Typed `QueueRejected(code: Literal["queue_overloaded", "queue_stopping", "job_payload_too_large"])`
is raised before accepting a rejected job.

Each accepted job owns private completion and one admission lease. The shared
internal admission routine creates all requested subscriptions atomically, before
execution is runnable; `submit_page` cannot miss a fast result or history eviction.
Caller-facing futures are detached subscriptions, not that private completion.
Controls live in a private execution binding, not the JSON payload; retain that
binding through drain and account its lifetime with the live-job limit. Queue history
keeps only allowlisted public article/status fields; discovery is live-delivery-only.
Bound history serialization incrementally too; do not allocate a second unbounded
JSON result just to learn it exceeds the retention ceiling.
Execution handles retain live/payload charges through physical drain, regardless
of public terminal state. Retry reuses payload and pending reservation; it does
not submit a second job. Use stable priority plus sequence tie-breakers.

- [x] **Step 1: Write deterministic state/lease tests.**

```python
async def test_full_queue_retry_stop_interleaving(queue_harness):
    q = queue_harness.full_with_retrying_workers()
    await q.release_attempt_failures_then_stop()
    assert q.each_accepted_job_terminal_once()
    assert q.each_subscription_settled_once()
    assert q.retry_put_waiters == 0

async def test_detached_waiter_does_not_cancel_work(queue_harness):
    waiter = await queue_harness.submit_blocked()
    waiter.cancel()
    await queue_harness.finish_execution()
    assert queue_harness.job_status == "completed"

async def test_draining_job_holds_payload_and_live_lease(queue_harness):
    await queue_harness.cancel_running_thread()
    assert queue_harness.live_leases == 1
    assert queue_harness.payload_bytes > 0
    await queue_harness.release_thread()
    assert queue_harness.live_leases == 0
```

Cover all priority levels/shared callers, individual/aggregate bytes/count limits,
no rejected residue, retry-vs-stop, submit-vs-stop, cancel-vs-result, polling versus
execution timeout, startup idempotency, retention zero/count/bytes, and huge result
delivery even when history cannot retain it. Hypothesis checks state transitions.
- [x] **Step 2: Run `python -m pytest tldw_Server_API/tests/Web_Scraping/test_phase5_queue_admission.py tldw_Server_API/tests/Web_Scraping/test_phase5_queue_lifecycle.py tldw_Server_API/tests/Web_Scraping/test_phase5_queue_history.py -q`.** Expected: missing queue API fails.
- [x] **Step 3: Implement atomic non-waiting admission and terminalization; run execution outside the lifecycle lock.** Close new/retry admission before stop, terminalize accepted pending jobs, cooperatively cancel active handles, observe subscriptions, and retain drain callbacks/credentials only until release. Do not cancel shared workers to cancel one job.
- [x] **Step 4: Rerun Task 3 tests.** 123 queue and 343 focused tests pass after review fixes, including unchanged wrapper contracts. No pending tasks/futures, lease underflow or retry deadlock in those tests.
- [x] **Step 5: Commit explicit paths and child task:** `feat(web-scraping): isolate bounded process-local job ownership`.

### Task 4: Enforce Lifecycle Authorization And Safe Projections

**Files:** Modify `tldw_Server_API/app/services/enhanced_web_scraping_service.py`,
`tldw_Server_API/app/api/v1/endpoints/web_scraping.py`; complete `jobs/projections.py`;
create tests `Services/test_phase5_webscraping_authorization.py`,
`Web_Scraping/test_phase5_public_projections.py`; extend existing
`Services/test_enhanced_webscraping_admin_claims.py`.

**Interfaces:** Consume verified auth dependency/current service principal helper,
not client role/owner dictionaries. Produce
`public_job(job: ScrapingJob, *, crawl_outcome: CrawlOutcome | None = None) -> dict[str, Any]`
and `public_progress(snapshot: Mapping[str, Any]) -> dict[str, Any]` as allowlists.
Private execution payloads never reach them. Keep current owner/admin status and
cancel methods/signatures; shared initialize/shutdown require platform-admin.

- [x] **Step 1: Write security/projection tests.**

```python
async def test_claimed_admin_in_options_cannot_shutdown(client, ordinary_user):
    response = await client.shutdown_as(ordinary_user, options={"role": "admin"})
    assert response.status_code == 403

def test_nested_execution_secrets_never_project(projection_fixture, caplog):
    public = projection_fixture.job_and_progress()
    assert "sentinel-cookie" not in json.dumps(public)
    assert "sentinel-auth" not in caplog.text
```

Add verified admin/owner success, wrong owner/not-found semantics, nested URL
userinfo/header/option credentials, exception logging, no raw discovery HTML in
history, and no removal of intentionally requested public article/raw content.
- [x] **Step 2: Run `python -m pytest tldw_Server_API/tests/Services/test_phase5_webscraping_authorization.py tldw_Server_API/tests/Web_Scraping/test_phase5_public_projections.py -q`.** RED 15 failed; added nested-redaction and failed-cleanup review regressions also reproduced before fixes.
- [x] **Step 3: Implement authorization before side effects and explicit public projections; reuse established auth dependencies and sanitizers.** Serialized/idempotent initialization, owned drain guards and explicit failed-cleanup retry; ordinary automatic startup preserved.
- [x] **Step 4: Run new tests and existing admin/retention regressions.** 471 pre-review covering tests and 197 post-fix covering tests pass; Bandit zero findings; warning sources categorized. Fresh review I1-I2 fixed and independently re-reviewed; deferred startup/shutdown transition minor remains for final review.
- [x] **Step 5: Commit explicit paths and child task:** `cc4db584dc`, `afe62367b6`; TASK-15423 Done after independent gate.

## Stage 3: Governed Crawl And Continuation

**Goal:** Connect actual acquisition to parent controls, then build traversal/checkpoints/batches behind injected capabilities.
**Success Criteria:** No budget bypass, lost in-flight page, orphan child, or duplicate discovery fetch.
**Tests:** Tasks 5-8 unit/property/transport/checkpoint suites with existing Phase 3/4 regressions.
**Status:** Complete (Tasks 5-8 independently reviewed; bounded production and cancellation fixes accepted).

### Task 5: Integrate Parent Controls With Canonical Acquisition

**Files:** Modify `orchestration/article.py`, `article_browser.py`, `runtime/fetch.py`,
`preflight/context.py`, `facade.py`, `runner.py`, `adapters/http.py`,
`adapters/browser.py`, `adapters/external_tools.py`, and
`tldw_Server_API/app/core/http_client.py`; create tests
`Web_Scraping/test_phase5_controlled_acquisition.py`, `test_phase5_preflight_parent_controls.py`,
`http_client/test_crawl_stream_controls.py`.

**Interfaces:** Consume `ParentControls`, `WorkHandle`, `PageData` (Task 2).
Add internal `start_page(url: str, custom_cookies: list[dict[str, Any]] | None, *,
controls: ParentControls, request_context: RuntimeRequestContext,
discover: Callable[[str, str, str], DiscoveryResult],
plan: ArticlePlan | None = None,
allow_llm_extraction: bool = True) -> WorkHandle[PageData]` in `orchestration/article.py`.
Factor through existing `_run_article`/dependencies, not a parallel pipeline;
public `scrape_article`, blocking/sync/raw-browser returns/signatures stay unchanged.
Private acquisition returns bounded effective-location/link sidecar from the same
document before extraction; discard HTML afterward, retaining public raw content
only when already requested. `DiscoveryResult` is not public article metadata.
The injected `discover(document, request_url, final_url)` callback keeps this task
independent of Task 6's real parsers; tests use bounded fake discovery, Task 6 wires
the real profile-aware implementation. `start_page` reserves/commits one logical
page attempt per invocation, including retries; neither queue nor frontier charges
that same attempt a second time.
The optional resolved article plan carries existing explicit user-agent/header,
backend and analyzer choices; orchestration resolves it through the current
resolver with private credentials, not by stuffing credentials into `CrawlPlan`.

Bind controls to runtime adapter instances, not frozen `FetchRequest.metadata`.
Add optional internal parent argument to `build_execution_context` and
`PreflightRuntimeControls`, default None for standalone compatibility. Local and
parent reservations both succeed before starting; unused reservations roll back.

The shared HTTP boundary accepts generic optional `on_response_chunk:
Callable[[int], None] | None` in its existing bounded sync and async paths and
private bounded-body helpers. No Web_Scraping import. Invoke before appending each
bounded raw chunk; preserve the existing response API and redirect/retry semantics.
Use the existing single-attempt simple API for article requests and `RetryPolicy(attempts=1)`
for probes. Charge requests at the actual per-hop adapter, never again at the caller;
charge fallback/retry operations only when they start. Do not broadly rewrite
shared HTTP retry machinery. Observer/parent stop exceptions escape sanitization
and curl fallback, including deferred curl callback error propagation.
If the shared client wraps callback failures, use a generic
`ResponseObserverAborted` there and preserve the private cause for the runtime
adapter to rethrow; never import `ParentStop` into the shared client or expose the
cause in logs/public errors. Add sync and async transport tests for that path.

Article/preflight browser adapters reserve each browser attempt and each routed
HTTP/WebSocket handshake; wire-byte callbacks feed one ledger per acquisition.
Avoid double counting CDP data plus response body snapshots. Existing CDP/frame
limits remain; close admission on stop, cancel queued acquisition, keep drain
ownership through callbacks/extraction exit. External probes reserve one invocation,
cap subprocess lifetime by remaining time, read stdout/stderr incrementally under
byte/output allowance instead of unbounded `communicate()`, and preserve cleanup.

- [x] **Step 1: Write adapter-spanning stop/byte/discovery tests.**

```python
@pytest.mark.parametrize("stage", ["preflight_http", "preflight_browser", "external_tool", "http_fallback", "browser_retry"])
async def test_parent_stop_escapes_all_fallbacks(stage, acquisition_harness):
    report = await acquisition_harness.exhaust_parent_at(stage)
    assert report.reason == "budget_exhausted"
    assert report.calls_after_stop == []

async def test_authenticated_discovery_uses_same_document(acquisition_harness):
    page = await acquisition_harness.redirected_private_page()
    assert acquisition_harness.document_fetches == 1
    assert page.discovery.final_url == "https://example.test/private/final"
    assert page.discovery.links[0].parent_url == page.discovery.final_url
```

Add concurrent limits, per-hop redirects/robots/probes, chunk boundary/curl callback
abort, WebSockets, active timeout, analyzer-local recoverable error versus parent
stop, advice/attached-result parity, sync thread late completion, private identity
propagation, and no body/sidecar retention in public history. For chunk observation,
assert no over-budget chunk appended; do not claim zero kernel buffering overshoot.
- [x] **Step 2: Run the three named acquisition/preflight/shared-HTTP suites.** Initial RED 19 failures; added deterministic review regressions also reproduced before fixes.
- [x] **Step 3: Implement narrow internal hooks/discovery sidecar and actual per-operation drain.** Fresh review I1-I3 cancellation/commit/WebSocket/cleanup defects fixed and independently re-reviewed.
- [x] **Step 4: Run new and existing regressions.** Historical full scope 3288 passed, 26 skipped, 5 failed; three earlier-Phase5 integration failures corrected, two baseline arXiv assertions retained for Task 11. Final 773 covering tests pass; Chromium 3 pass/no skips; touched Bandit zero findings. Inventory/full-repository baseline gates remain visible, no full-scope green claim.
- [x] **Step 5: Commit explicit paths and child task:** `e3b4515f54`, `db27425016`, `f2abedc4e1`; TASK-15424 Done after fresh review and fix gate.

### Task 6: Extract Bounded Discovery And Profile Traversal

**Files:** Create `crawl/frontier.py`, `discovery.py`, `engine.py`,
`orchestration/crawl.py`; create tests `Web_Scraping/test_phase5_frontier.py`,
`test_phase5_discovery.py`, `test_phase5_crawl_engine.py`.

**Interfaces:** `FrontierEntry(url: str, parent_url: str | None, depth: int, sequence: int, score: float)`
in `crawl/models.py`. `Frontier(plan: CrawlPlan)` offers `admit(entry) -> bool`,
`take() -> FrontierEntry | None`, `commit(entry, page: PageData) -> None`,
`restore_unfinished() -> None`, `snapshot() -> FrontierSnapshot`.
`FrontierSnapshot` holds ordered queued/inflight/completed dispositions and results.
`discover(document: str, *, request_url: str, final_url: str, plan: CrawlPlan,
controls: ParentControls) -> DiscoveryResult` reuses supported XML/HTML parsing and
profile URL rules, checking each candidate before collection growth.
`run_crawl(plan: CrawlPlan, *, context: CrawlControls,
acquire: Callable[[FrontierEntry], Awaitable[WorkHandle[PageData]]],
checkpoint: CheckpointPort | None = None) -> CrawlReport` is the sole engine.
Define structural `CheckpointPort` and `CheckpointState` in `crawl/models.py` with
Task 7's load/save/clear/close methods; Task 6 tests supply an in-memory fake.
Define the credential-free `CheckpointScope` value there too, so the port's
`load(*, plan, scope)` has a concrete type before storage is implemented.
The lower engine sees protocols, not orchestration imports. Concrete context is
supplied by orchestration against `CrawlControls`.

`orchestration/crawl.py` produces `execute_crawl(request: CrawlRequest, *, profile:
str, options: Mapping[str, Any], supplied: frozenset[str], config: Mapping[str, Any],
credentials: Mapping[str, Any], limits: CrawlLimits = CrawlLimits(),
checkpoint: CheckpointPort | None = None) -> CrawlReport`, composing Task 5 with
the `discover` callback closed over this plan/context.
Private credentials are snapshotted separately and used for robots/discovery/pages.
The direct acquirer is an async adapter returning Task 5's handle; the engine awaits
admission, attaches ownership, then awaits the result. Candidate accounting belongs
to discovery (seed admission is charged once by the engine); frontier only owns
occupancy, not a second candidate charge. Robots/sitemap acquisition uses governed
runtime controls without pretending those transport calls are extracted pages.

**Task 6 Execution Rulings (Controller Decisions, 2026-11-01):** The approved
bounded-discovery/no-functionality-loss requirements expose three missing private
contracts. These are builder safety/composition decisions, not new public settings
or an assertion that the human selected these additional numbers.

1. Add trusted validated `CrawlLimits.max_url_bytes=65_536` and
   `discovered_url_bytes=67_108_864`. The former applies to raw references and
   bases before resolution, then normalized targets, retained parents and
   request/final locations. Check length before encoding/resolving to avoid an
   oversized derived temporary; UTF-8 is the exact final byte metric, not RSS.
   Add a separate cumulative `discovered_url_bytes` counter to parent controls and
   `BudgetSnapshot` (default zero for internal constructor compatibility). Charge
   request and final sidecar locations once per discovery call and each normalized
   candidate plus its resolution parent before filtering/deduplication/collection
   growth; duplicates remain charged. The seed is charged once by the engine with
   its existing candidate charge. Do not charge a frontier reference again or
   refund cumulative work when entries leave; all URL bookkeeping must reuse this
   admitted bounded domain. Invalid references count as encounters but are not
   retained; ordinary malformed references keep their existing profile rejection
   behavior. An oversized URL or collective overflow produces explicit
   `budget_exhausted`, never silent truncation or invented transfer usage. Validate
   checkpoint file capacity against these injected byte ceilings too; restore
   known cumulative accounting, explicitly mark legacy unknown history. Chosen
   local safety defaults are generous per-URL and cap amplification independently
   of wire bytes; they are not HTTP standards. Exact/one-over and many-relative-
   links/long-base tests use small trusted limits, no large allocation.
2. Add private frozen `CrawlHooks` in `crawl/models.py`: optional synchronous
   `url_filter: Callable[[str], bool]`,
   `progress_callback: Callable[[str], None]` (legacy messages), and
   `on_progress: Callable[[Mapping[str, Any]], None]` (owned snapshot updates).
   Default all to None, hide callable reprs, and pass an optional `hooks` keyword
   through internal orchestration/engine/discovery only where needed. It is live
   execution state, never plan options, credentials, checkpoints, metadata or public
   payload. Preserve profile-specific invocation order: stripped sitemap loc filter
   before collection/truthy max_urls; recursive filter after established built-in
   filtering/scoring and before candidate admission, not on the seed. Preserve
   legacy progress messages but emit no completed notification on interruption.
   Release hooks with actual owned drain; do not add a new callback framework.
3. Add private `start_page(..., discovery_on_success: bool = False)` and the narrow
   dependency flag needed to select callback timing in the existing `_extract`.
   Default remains the existing pre-extraction callback for standalone/legacy
   compatibility. Enhanced recursive/url-level composition selects True: invoke
   discovery only after extraction returns the same truthy `extraction_successful`
   predicate used by the existing traversal, using the still-bounded same document
   and effective location, then discard HTML. No filter/candidate debit for a failed
   enhanced extraction. Do not substitute the stricter retained-data predicate for
   profile success counting; `note_successful_result` still marks actual retained
   successful article data. If discovery interrupts this acquisition before
   PageData reaches the engine, its frontier entry remains unfinished for checkpoint
   restoration; only previously logically committed articles count as retained
   partials. Never mark the current unfinished page completed or emit a completion
   callback. Add a failed-extraction/linked-page budget regression and a successful
   discovery-interruption/unfinished-frontier regression. No second page fetch.
4. Add optional private `seeds: Iterable[str] | None = None` to internal engine /
   orchestration composition. None selects the request seed; an explicitly empty
   iterable means no inputs. Consume lazily under candidate/URL-byte/frontier and
   eventual Task 8 producer-window limits, never eagerly materialize all inputs.
   `enhanced_batch` preserves repeated inputs, per-input failures and input order;
   do not apply recursive URL deduplication to distinct batch positions. A live
   iterable is not credentials/hooks/options or checkpoint content. This adds no
   new batch resume promise: unsupported continuation combinations fail explicitly
   rather than silently losing unconsumed input.
5. Define a private structural `TraversalPolicy` and frozen `TraversalDecision`
   in `crawl/models.py`; the port has one async
   `evaluate(entry: FrontierEntry, *, phase: Literal["seed", "discovery", "visit"])`
   method returning the decision. The decision supplies allowed status, relevance
   and ordering scores, existing safe crawl metadata and a bounded machine reason
   where needed. Orchestration implements it from existing filters/scorers and
   governed policy/robots with private credentials; lower modules import only the
   structural types. Optional policy injection on `run_crawl` supports pure fakes;
   production composition always supplies the real port. Keep async policy work
   before the logical-result commit marker, not between marker and disposition.
   Preserve per-profile built-in/score/robots/custom-filter order and seed-denial
   outcome versus an ordinary filtered candidate; no new HTTP authorization
   policy or second page/transport debit. Do not move these responsibilities onto
   resource controls or introduce a new plugin registry.
   Policy/robots transport must also have a per-operation WorkHandle registered
   before dispatch, not only counted requests. Add private
   `CrawlControls.owned_handles() -> tuple[WorkHandle[Any], ...]` returning the
   current physical-live handles from this crawl owner under its existing lock.
   Engine stop/finalization includes that snapshot as well as acquired pages for
   cancellation and checkpoint-close-after-drain; it is not executor-global
   outstanding work, public state or serialized data. Completed handles are pruned
   only on actual drain, preserving prompt private-graph release.
6. Output retention is not the strict actual-data marker: `CrawlReport.articles`
   must retain every entry the selected legacy/enhanced profile previously returned,
   including required structured failures and truthy extraction-success entries
   with empty content. Only `has_partial_results` and `note_successful_result` use
   the stricter successful-data predicate. Do not drop compatibility payloads to
   make outcome classification simpler.

The cost of a wrong ruling is private interface/limit rework before adapter cutover;
none authorizes a public signature change, new client knob or silent loss of hooks.

- [x] **Step 1: Write profile strategy and atomic frontier tests.**

```python
def test_unfinished_restores_admission_order(frontier):
    first, second = frontier.take(), frontier.take()
    frontier.restore_unfinished()
    assert [frontier.take(), frontier.take()] == [first, second]

async def test_candidate_limit_counts_duplicates(crawl_harness):
    report = await crawl_harness.repeat_one_link(50_001)
    assert report.outcome.stop_reason == "budget_exhausted"
    assert crawl_harness.max_frontier_size <= 10_000
```

Add all Task 1 profile fixtures, failure inclusion, robots/sitemap policy and
credentials, cyclic/duplicate frontiers, final/request URL distinction, scores,
relative depth, no false completed callback, and normal page-cap versus interruption.
Use Hypothesis for normalization/order/disposition/accounting invariants.
- [x] **Step 2: Run the three named frontier/discovery/engine suites.** Initial RED three missing-module errors; final brief suite 58 pass, all-profile focused 147 pass before independent review.
- [x] **Step 3: Move traversal/parsing rules behind injected acquisition.** No lower owner/wrapper import or duplicate discovery fetch. New bounded normalization, private hooks/policy/seeds and actual policy-work drain follow execution rulings.
- [x] **Step 4: Rerun direct/queued profile and interruption coverage.** Fresh review I1-I3 fixed and independently re-reviewed; final 636 covering tests pass, scoped Bandit zero findings. Historical broad 3427 pass/26 skip/two baseline arXiv failures and inventory gate remain Task 11 obligations. Sequential direct latency/concurrency implication carried to Task 8/final review, no throughput-parity claim.
- [x] **Step 5: Commit explicit paths and child task:** `87c794be0f`, `0702dc3ee2`, `55033a803b`; TASK-15425 Done after fresh review/fix gate.

### Task 7: Implement Safe Legacy Continuation

**Files:** Create `crawl/checkpoints.py`; complete checkpoint protocol/model in
`crawl/models.py`, integration in `crawl/engine.py`; create tests
`Web_Scraping/test_phase5_checkpoints.py`, `test_phase5_checkpoint_security.py`.

**Interfaces:** `CheckpointStore.load(*, plan: CrawlPlan, scope: CheckpointScope) -> CheckpointState | None`,
`save(state: CheckpointState) -> None`, `clear() -> None`, `close() -> None`.
`CheckpointScope` is a trusted owner/crawl namespace plus an explicitly trusted
local-path mode, never a scope derived from file contents or HTTP options.
`CheckpointStore.open(path: Path, *, scope: CheckpointScope,
limits: CheckpointLimits) -> CheckpointStore` acquires single-writer ownership.
`CheckpointStore` implements `CheckpointPort`; its models stay in `crawl/models.py`
so Task 6 does not import a not-yet-created storage implementation.
`CheckpointState` contains schema_version=1, identity/seed/profile/options binding,
safe provenance, frontier snapshot, result/dedup state, BudgetSnapshot, stop reason,
and `accounting_complete: bool`. No credentials/live controls/monotonic timestamps.

Hold process-local owner registry plus cross-process file lock through physical
drain. Reuse existing repository lock utility if its ownership/platform tests fit;
otherwise implement stdlib POSIX/Windows lock adapters in this file with the same
exclusive lifetime, not a new dependency or fixed `.tmp` sentinel. Service paths
are trusted owner-scoped, refuse symlinks/escapes via descriptor/path checks;
trusted local mode preserves explicit legacy paths without exposing a remote API.
Read at most limit+1 bytes before JSON parse; validate schema, bounded collections,
finite counters/options, and binding before any URL execution. Stream JSON to a
unique temporary file with a byte ceiling, flush, atomically replace, remove temp
on failure; retain prior valid state. Re-admit resumed URLs with current egress.

**Task 7 Execution Rulings (Controller Decisions, 2026-11-01):**

- Extend private `CheckpointScope` with `root: Path | None = None` (hidden repr).
  This is the already owner/crawl-scoped authorized directory supplied by trusted
  composition, not a global root inferred from the candidate path. Non-local
  `CheckpointStore.open` requires an existing authorized root; composition creates
  that root deliberately. Canonicalize that trusted root for platform aliases,
  then refuse descendant/checkpoint/lock/temp symlinks or escapes with the required
  descriptor/path checks. Explicit trusted-local mode retains legacy absolute or
  current-directory-relative paths without requiring this service root. Root and
  trusted-local authorization mode are never loaded from checkpoint JSON or client
  options; stored owner/namespace are binding checks only, never a grant. Memory
  ports can keep the default None root because they do not open filesystem paths.
- If the platform cannot provide the descriptor-relative no-follow operations
  required for service-root confinement, that private mode fails closed with a
  sanitized checkpoint failure; do not add an unsafe lexical/path-check fallback.
  Explicit trusted-local legacy mode still uses native POSIX/Windows locking.
  This is a documented service-root capability limitation, not verified Windows
  service-root support or permission to break Windows legacy resume adapters.
  Task 9/10 composition must select the correct trusted mode explicitly, never
  from client options. Native Windows service-root support remains unverified.
- Keep store methods synchronous as specified, but engine composition may offload
  potentially blocking filesystem operations through a narrow owned-operation
  helper reusing existing runtime PhysicalWork/WorkHandle and crawl registration.
  Register before dispatch, including eager task factories. Do not invent a new
  executor or charge file I/O as pages/network/transfer. A canceled awaiter never
  releases the writer while an actual load/write/replace/clear can still act;
  include these handles in the existing per-crawl drain/close protocol. Separate
  bounded logical cleanup return from physical ownership, preserve primary failure
  and old valid state, and test held filesystem calls plus cancellation. Do not
  silently replace the structural port with service/orchestration imports below it.

These close the existing trusted-root and sync-protocol ownership gaps, without
adding a remote resume API or changing public legacy signatures. The cost of a
wrong ruling is private scope/helper rework before Task 9 adapter cutover.

- [x] **Step 1: Write continuation, corruption, and failure-precedence tests.**

```python
async def test_resume_waits_for_physical_owner(checkpoint_harness):
    await checkpoint_harness.interrupt_inflight()
    assert checkpoint_harness.try_resume().code == "checkpoint_in_use"
    await checkpoint_harness.release_drain()
    resumed = checkpoint_harness.try_resume()
    assert resumed.frontier_urls == ["/first", "/second"]
    assert resumed.spent_attempts == 1

async def test_save_failure_does_not_mask_cancellation(checkpoint_harness):
    with pytest.raises(asyncio.CancelledError):
        await checkpoint_harness.cancel_with_replace_failure()
    assert checkpoint_harness.previous_checkpoint_valid()
    assert checkpoint_harness.outcome.stop_reason == "cancelled"
```

Add legacy visited/to_visit/articles/pages migration in exact order, unknown history
annotation, no invented consumption, known exhausted budgets remain exhausted,
paused time excluded, successful dedup commit, unsupported version/size/binding,
malformed counters, credential-free JSON, concurrent processes, symlink escape,
failed replace/serialization/clear, checkpoint overflow, and interrupted transport
retention. New writer fixes future in-flight loss; it cannot recover missing old URLs.
- [x] **Step 2: Run the two named checkpoint suites.** RED two missing-module collection failures; strict schema/aggregate cleanup/packaging defects also reproduced before correction.
- [x] **Step 3: Implement versioned/legacy loading, atomic storage and owned finalization.** Single aggregate two-second cleanup deadline, current-policy replay and actual writer drain retain unfinished state without masking primary stops.
- [x] **Step 4: Rerun checkpoint/engine/lock/package coverage.** Final 346 covering tests pass; scoped quality/Bandit clean. New run_crawl reduced to 265 lines, no ceiling increase; six other branch size regressions remain certification work. Native Windows not verified; service-root capability limitation and M1 fork warning remain explicit.
- [x] **Step 5: Commit explicit paths and child task:** `efbd0e70fc`, `70e58f4a86`; TASK-15426 Done after fresh task/fix review, including regular-package shipping correction.

### Task 8: Compose Bounded Batches And Queue-Owned Pages

**Files:** Create `orchestration/batch.py`; connect `orchestration/crawl.py` and
`jobs/queue.py` internal subscription delivery; create tests
`Web_Scraping/test_phase5_batch_window.py`, `test_phase5_owned_crawl_jobs.py`.

**Interfaces:** `execute_batch(urls: Iterable[str], *, plan: CrawlPlan,
context: CrawlContext, queue: ProcessLocalJobQueue,
make_job: Callable[[str], ScrapingJob]) -> CrawlReport` consumes lazily without a
second full payload/future list. Keep a bounded owned job map and ordered result
indices; collect incrementally, retaining only the caller's required output list.
Async `queued_page(entry: FrontierEntry, *, context: CrawlContext,
queue: ProcessLocalJobQueue, make_job: Callable[[str], ScrapingJob]) -> WorkHandle[PageData]`
is internal orchestration glue using `submit_page` so admission and private
subscription creation are atomic, not public sidecar history. The execute callback
receives the bound controls and calls `start_page` once for each physical job attempt.

**Task 8 Execution Rulings (Controller Decisions, 2026-11-01):**

- Add an optional trusted private `execute` override to `submit_page` and the
  internal `queued_page` / `execute_batch` glue where needed. Its signature is
  the existing queue callback type:
  `Callable[[JobPayload, ParentControls | None], WorkHandle[PageData]]`.
  Capture the selected override (or constructor default when None) once in the
  admitted private entry before it can run. Keep `add_job`, public constructors,
  legacy return conventions and standalone default execution unchanged. Do not put
  callbacks/plans/live controls in JobPayload, metadata, context dictionaries or
  a hidden global registry. Normalized execution inputs still obey count/byte
  admission; the trusted callable is a separate capability. Retain it for retry
  and actual drain, then clear it and its captured private graph even if a caller
  retains a result/cancellation handle. Add callback-isolation, retry and weakref
  drain-release tests.
- Internal crawl composition may accept an optional queue for `queued_page`, with
  direct acquisition remaining the default. A small read-only queue producer-window
  capability is permitted if needed to avoid reading private admission internals;
  it never grants admission or creates a waiter. Keep the exact default window
  `min(max_concurrent, max_pending, max_live)` and remaining parent-page bound.
  A barrier test must prove available capacity is filled before releasing the
  first result; an upper bound alone would incorrectly approve serial execution.
  Preserve the documented Task 6 recursive selection order and do not claim
  recursive throughput parity from standalone producer tests.
- There is exactly one terminal publisher per logical operation. Nested queue
  composition must not publish a second outcome or mark a frontier result
  successful before its actual atomic commit. Reuse the existing acquisition and
  profile-conversion logic, not a second scraping pipeline.
- Add a private lazy owner-loop `CrawlContext.stop_signal() -> asyncio.Future[None]`
  for cancellation and latched parent-stop notification. It is shared/observed,
  not canceled by consumer waits; thread-safe publication uses the existing owner
  lock and loop handoff. Construction outside a running loop must remain valid.
  Execution/admission progress waits include this signal as well as owned result
  or drain progress and the remaining execution timeout, so context cancellation
  can promptly enter bounded cleanup even when physical drain remains held.
  This notification never means work drained or grants new capacity. Do not poll
  the context, create an unbounded waiter registry, or serialize the signal.

These expose required private execution binding without changing public wrapper
signatures or adding unbounded callback/admission registries. The cost of a wrong
ruling is private glue/entry rework before Task 9 adapter cutover.

- [x] **Step 1: Write multi-window and partial-admission ownership tests.**

```python
async def test_batch_larger_than_window_preserves_order(batch_harness):
    report = await batch_harness.run(23, window=5, reverse_completion=True)
    assert [a["input_index"] for a in report.articles] == list(range(23))
    assert batch_harness.max_materialized_jobs <= 5

async def test_partial_admission_abort_owns_children(batch_harness):
    report = await batch_harness.abort_after_two_admissions()
    assert report.articles[0]["content"] == "already completed"
    assert batch_harness.unrelated_job_cancel_calls == 0
    assert batch_harness.unobserved_subscriptions == 0
    assert batch_harness.draining_payload_still_charged
```

Add queue pressure with own work (await completion, then attempt), pressure without
own work (immediate overload), input generator consumption bound, parent deadline
during wait, oversized next payload, per-page failures, ordinary waiter detachment,
explicit crawl cancellation, and full-queue retry reservation reuse.
- [x] **Step 2: Run `python -m pytest tldw_Server_API/tests/Web_Scraping/test_phase5_batch_window.py tldw_Server_API/tests/Web_Scraping/test_phase5_owned_crawl_jobs.py -q`.** RED evidence recorded in the Task 8 report; independent review's two cancellation regressions also reproduced before fixes.
- [x] **Step 3: Implement the producer window and owned-child cleanup without lock-held waits or polling loops.** On abort stop production, cancel unfinished owned jobs and observe subscriptions under separate cleanup allowance. Preserve completed partials; retain physical drain leases/handles beyond logical return.
- [x] **Step 4: Rerun Task 8 and queue lifecycle tests.** Final 403 covering tests and 780 Phase 5 tests pass. Fresh scoped re-review closes I1 stop-aware waits and I2 composed child-local cancellation; scoped Bandit/lint/format/compile clean. Earlier six size debts and inventory/consumer/environment gates remain Task 11 obligations; no recursive-throughput parity claim.
- [x] **Step 5: Commit explicit paths and child task:** `e1a4fa97c9`, `a19f6302f4`, with controller stop-signal ruling `f64b72aa91`; full task and fix-range reviews recorded in the plan-scoped SDD ledger.

## Stage 4: Compatibility Cutovers And Certification

**Goal:** Switch wrappers/services once, propagate outcomes, and certify real consumers.
**Success Criteria:** Public parity except approved safety fixes; report/receipt fidelity and architecture guards pass.
**Tests:** Tasks 9-11 full compatibility, security, browser/lifecycle and affected-consumer coverage.
**Status:** In Progress (Tasks 9-11 independently accepted; final whole-branch review remains).

### Task 9: Switch Legacy And Enhanced Adapters

**Files:** Modify `Article_Extractor_Lib.py`, `enhanced_web_scraping.py`,
`contracts/conversion.py`; extend `Web_Scraping/test_phase5_profile_contracts.py`;
create `Web_Scraping/test_phase5_adapter_outcomes.py`.

**Task 9 Execution Ruling (Controller Decision, 2026-11-01):**
The cutover may add a trusted optional private article-result transform
`Callable[[str, dict[str, Any]], dict[str, Any]] | None` to crawl composition
and the private `article.start_page` path. It runs exactly once per completed
canonical page attempt, after final extraction/content conversion, before queue
retry classification, engine success/result commit and checkpoint state. With a
transform present, defer same-document discovery until after that transform and
apply the existing success/depth/cap gates to the transformed result. A duplicate
must not consume unused discovery/candidate budgets or trigger extraction fallback
merely because its compatibility mapping is unsuccessful. Keep no-transform
standalone behavior unchanged. Preserve established duplicate mappings and analyzer
attachment, non-secret successful extra fields, instance deduplicator state and
retry behavior. The callable is a composition capability, never a plan option,
credential, checkpoint field or global registry. Retain only actual-work ownership,
forward cancellation/physical drain, and release private graphs after drain.
Add duplicate/no-expansion, duplicate-only interruption partial/checkpoint,
per-attempt retry and callback-lifetime tests. This closes a proven no-functionality-
loss gap; the cost of a wrong ruling is private hook/ordering rework, not a new
public contract. Raise any further concrete private-composition gap before
silently dropping existing instance behavior such as rate limiting.

The enhanced adapter may snapshot CookieManager's stored lists once into private
credentials under `stored_cookies_by_netloc`, selecting by the exact existing
`urlparse(target).netloc` lookup for each admitted page or discovery request.
Do not flatten a seed's stored cookies across hosts/ports. Preserve the established
precedence: stored cookies, then explicit cookies, then applicable route cookies
for article acquisition (discovery keeps its characterized behavior). Canonical
redirect credential stripping remains authoritative; this is not permission to
reattach secrets on a cross-boundary redirect. Snapshot data is neither public
metadata nor checkpoint/config/provenance; each normalized queued payload contains
only its selected execution cookies and remains byte-admitted, never the whole jar.
Retain selected secrets only through actual work/drain and clear private graphs.
Add exact-host/port/external-child selection, precedence, snapshot-mutation,
redirect-stripping and projection/checkpoint secrecy tests. A private snapshot
helper on CookieManager is permitted without changing its existing public API.
This preserves an existing instance credential source; wrong scope selection would
cost private composition rework and is a security review blocker, not an accepted
tradeoff. Complete an instance-capability audit before further composition so
remaining rate, method, policy, headers, cookies, dedup and progress requirements
are handled together rather than discovered only after switching each wrapper.

**Task 9 Review-Fix Rulings (2026-11-01):**

- Bound synchronous bridge owners with one private process-owned immediate
  admission gate, default capacity 4 (matching the canonical extractor's existing
  default, but a separate lease to avoid nested extraction deadlock). Acquire
  before thread/loop dispatch; no queued waiters or per-invocation fresh allowance.
  Excess admission uses the existing sanitized `queue_overloaded` typed failure
  and closes any unstarted coroutine. Hold the lease through actual loop, async
  generator and default-executor retirement, not logical result delivery. An
  internal test seam may set a validated positive finite capacity before work;
  it must not reset live/draining capacity. Handle PID/fork changes without using
  inherited locks or refunding a child's capacity from a parent lease. No new
  ThreadPoolExecutor or extraction-manager lifecycle. Cover held-work exhaustion,
  prompt rejection without new thread/loop, startup failure, and recovery after
  actual drain. This is the approved bounded-overload correction, not permission
  to abandon work or unlock checkpoints early.
- Preserve enhanced browser inputs through additive private immutable
  `DirectBrowserProfile` fields for scoped extra headers and validated selected
  proxy configuration, plus target-normalized merged cookies. Defaults leave
  standalone capture unchanged. Compose stored < explicit < route article cookies;
  route-only and ordinary name/value records must be valid browser cookies for
  the admitted URL. Apply secrets at the guarded request boundary, not as
  unrestricted context-wide headers: cross-boundary redirects/subresources must
  not inherit or reacquire original secrets. Existing proxy validation, egress,
  browser transport attestation, transfer bounds and WebSocket guards remain
  authoritative. No wrapper imports in lower layers, raw browser fallback, or
  public request/profile signature removal. Strict guarded-browser boundary tests
  must check actual cookies/headers/proxy and redirects, not only HTTP fixtures.
  Extend `orchestration/article_models.py` and `article_browser.py` narrowly for
  this capability; report a concrete attestation gap rather than bypassing it.
- Carry a trusted private extraction-method intent through `start_page` and
  article dependencies/preparation into real preflight advice, independently of
  backend intent. Default remains `auto`; enhanced recursive preserves its prior
  explicit `trafilatura` intent and advice attachment. Do not force a backend to
  simulate a method or change standalone/analyzer precedence. Test the public
  recursive path with real advice and JS-positive analysis plus HTTP/browser
  acquisition fakes, including explicit route/config backend combinations.
- Normalize both existing enhanced cookie forms (name/value records and shorthand
  mappings) before stored/explicit/route precedence, while preserving exact-netloc
  selection, byte admission and redirect stripping. Do not fix the loss by
  broadening credential scope. Cover public crawl and sitemap with both stored
  and explicit shorthand and a colliding route cookie.

These repair review Findings 1-4 under the no-functionality-loss and bounded-work
requirements. The cost of a wrong ruling is private ownership/profile rework;
the independent scoped re-review must verify all four and new breakage. Decompose
`execute_crawl` rather than grow it past the 300-line new-function ceiling.

**Task 9 Browser Execution Clarification (Review Finding 2):**
The installed Playwright route API is insufficient for per-redirect credential
rewriting. A private owned Chromium CDP Fetch interception collaborator is allowed
under the existing browser acquisition/callback/transfer lifecycle, not a second
HTTP transport. It must intercept each HTTP dispatch before network continuation,
including redirects, subresources and relevant child targets. Preserve the final
URL/document and normal browser execution: no auxiliary fetch/fulfill pipeline,
duplicate requests or global secret headers. Coordinate or replace overlapping
Playwright HTTP routing so two interceptors cannot deadlock or double-charge work.
Keep existing WebSocket and transfer accounting and service-worker blocking.
Child targets must be guarded before execution/dispatch; a main-page session alone
does not certify cross-process frames/workers/popups. Unsupported interception
fails closed with a sanitized capability failure, never a truncated success.

Reuse the existing callback capacity (64 by default, validated maximum 1024) for
live interception callbacks; additionally cap retained target/session ownership at
64 per acquisition. Exhaustion closes admission and preserves physical drain.
Bound retained redirect/request metadata under parent network/URL/byte limits;
do not add an unbounded session, request or command-future registry. Teardown must
stop interception admission, settle/abort paused requests, own pending callbacks
and detach sessions without allowing a late continuation or early lease release.
A focused `orchestration/article_browser_dispatch.py` helper and narrow structural
runtime browser port extensions are permitted to keep existing files/functions
cohesive and below their ceilings. First prove the native interception behavior in
small local Chromium tests before committing a broad implementation; after three
failed same-issue attempts stop and reassess. Required native cases include allowed
redirect chains, cross-host/port secret stripping without reattachment, denied
redirect with zero destination requests, and guarded child-resource dispatch.
Strict fake callback/cancellation/capacity tests complement, not replace, these.

Selected proxies keep the existing deployment distinction. `url_guarded` remains
the explicitly unattested single-user/compat profile: validated configured proxy
use must not be mislabeled DNS/peer-attested or silently removed. For an explicitly
selected proxy under `attested_proxy`, extend the in-process attestation/decision
with an optional hidden immutable binding to the canonical selected proxy config;
only the trusted governed transport supplies evidence. A different or absent
binding fails closed as existing `browser_transport_unavailable`; allowlisting
alone is not evidence. No selected-proxy metadata, credentials or binding goes
into public capability output/logs/checkpoints. Preserve zero-argument providers,
existing constructors via defaulted additive fields and the no-explicit-proxy
decision contract. Normalize/validate selected proxy configuration using existing
shared HTTP policy before browser dispatch. Cover matching, mismatching, absent,
invalid and legacy/unattested cases; do not weaken strict/public-capture policy.
`browser_transport.py` is a narrowly authorized additional fix path for this
binding. This refines the existing attestation rule, not permission to invent an
attestation or bypass it. Any inability to establish native interception or proxy
binding is an explicit unresolved blocker, not a waived safety/functionality loss.

Primary references checked with installed Playwright source and official protocol
documentation: [Playwright Route](https://playwright.dev/python/docs/api/class-route),
[BrowserContext](https://playwright.dev/python/docs/api/class-browsercontext),
[Chromium protocol schema](https://raw.githubusercontent.com/ChromeDevTools/devtools-protocol/master/json/browser_protocol.json).

**Current Human Direction And Minimal Fix (2026-11-02):**
The user rejected a new browser-control subsystem and reaffirmed no functionality
loss. The separate CDP connection/debugging-endpoint proposal and its approval
wait are withdrawn. Do not build that subsystem or repeat the failed browser-level
flattened-autoattach approach. Continue the approved refactor using existing
capabilities; do not treat the withdrawn proposal as permission to stop work.

Root-cause trace identifies an enhanced-versus-direct profile binding error:
Phase 4 deliberately leaves route headers/cookies/proxies ineffective in direct
capture, whereas the existing enhanced browser path supports those inputs.
Preserve these as separate compatibility behaviors instead of forcing every call
through the direct profile or changing standalone defaults. Narrow additive private
browser profile fields and target cookie normalization remain allowed; use the
existing guard/acquisition/cleanup owners and preserve analyzer/method precedence.
Caller cookies retain normal browser domain/path/secure/same-site semantics per
the Phase 4 contract, not a new manually copied Cookie-header scheme. This does
not broaden CookieManager's exact-netloc lookup or route/explicit/stored precedence.

A fresh native diagnostic proved that the existing owned page CDP session can
intercept both HTTP redirect dispatches alongside existing context routing and
strip custom Authorization/X-Private overrides on an origin change. It made two
actual HTTP requests, one context-route call and two Fetch request-pause calls,
preserving final URL and article HTML, with no additional browser control port.
This is evidence for a smaller implementation path, not full browser certification.
Use that existing session/request interface for needed per-dispatch header policy;
extend existing page/frame/context ownership only where actual parity/guard tests
require it. Do not infer that a page-only test proves every child-target case.
Retain callback/session bounds, no double request charging, WebSocket/transfer
guards, deployment-specific proxy admission and actual drain cleanup. Preserve
proxy support in its existing allowed deployment, never manufacture attestation.
No new HTTP fetch/fulfill pipeline, debug endpoint, external dependency, alternate
browser engine or parallel live execution path.

Implement a minimal failing browser-profile regression and fix at the composition
boundary. Verify cookies, headers, proxies, real allowed/denied redirects and
relevant frame/worker/subresource cases using native Chromium plus strict fakes.
Distinguish existing compatibility/security semantics from new guarantees added
by earlier controller proposals; do not silently waive a real regression. Keep
the three earlier unsuccessful experiments as evidence, not a reason to retry
their unsupported interface. The existing-session hypothesis is a reassessment
after the required architecture discussion, not a fourth flattened-wire attempt.

Further causal controls permit retaining the enhanced adapter's existing initial
context-header setup where first-popup parity requires it. This is not permission
for unrestricted credential forwarding: before-dispatch guards must still strip
private headers on cross-origin requests and preserve stripping across redirects
back to the original origin. Keep one explicit policy writer, avoid route-level
credential reinjection/replay, and prove initial popup plus redirected child
requests natively. Direct capture must continue to ignore route headers; do not
apply the enhanced setup to it. Normal browser cookies remain separate from this
header policy. Earlier blanket context-header wording was too broad for the
existing enhanced behavior; the ban on unguarded secrets remains unchanged.

Native mixed-page controls also distinguish the installed headless shell's stall
from adapter logic: bare acquisition completes, simple interception stalls, while
the installed full Chromium completes the same modes. A focused, verified launch
selection of the supported full-Chromium headless channel for enhanced acquisition
is permitted, using the same browser package/launcher and documented installation.
Do not add another engine, download, connection or subsystem, change direct defaults,
silently add an asset dependency, or weaken waits/assertions. Keep the diagnostic
comparison and add the corresponding regression before changing the launch option.

Tasks 1-8 remain accepted; Task 9 Findings 1/3/4 remain independently addressed.
Finding 2 is active work and must pass fresh scoped re-review before Task 9
acceptance. Tasks 10-11 and final integration/certification remain required.
Preserve the plan-scoped reports/probes/ledger; no completion/merge claim.

**Task 9 Default Batch Integration Gate (2026-11-02):**
**Latest Integration Ruling (2026-11-02, Supersedes Page-Only Guard Setup):**
Use Chromium's existing browser-scope Fetch interception through the already
owned browser connection, installed before context/page network activity. Native
diagnostics passed six popup redirect cases, preserving initial credentials,
stripping them across and back from another origin, and rejecting denied targets
before destination I/O. The late page-target attachment was the wrong integration
boundary; no Playwright startup hook or additional connection is justified by
this reproduced bug. Reuse the existing acquisition lease, bounded callbacks,
policy decisions and cleanup. Remove failed target-startup machinery that is no
longer needed for dispatch; keep only independently necessary bounded transfer
observation. Preserve direct versus enhanced input semantics and every existing
browser, cookie, proxy and analyzer capability. A browser-scope interceptor must
only operate on the browser owned by this acquisition, not a shared foreign owner.

Account for CDP cumulative `Network.loadingFinished.encodedDataLength` without
double counting chunks: first-navigation `dataReceived` events can be incomplete.
Do not wait for a renderer-dependent Network-enable acknowledgement while
holding the request that allows its navigation to proceed. Keep native commands
owned through acknowledgement or disposal. Native streaming stop, aggregate
limits, failure/cancellation, capacity, actual drain, cookies/headers/proxy and
frame/worker/popup regression gates remain required; diagnostic success alone
does not certify the product. No auxiliary HTTP fetch/fulfill pipeline or new
control subsystem. Test existing launch defaults again after eliminating the
unnecessary startup coordination; retain a changed launch selection only if
current causal evidence still requires it. Cost if wrong: narrow existing-owner
interception/accounting rework, not a new architecture.

**Cleanup Integration Ruling (2026-11-02):**
The browser-scope interceptor belongs to the dedicated browser owner. Native
controls show that retiring that owned browser with the guard still installed,
then retiring its closed context/session handles, resolves the OOPIF context-close
stall. Preserve the existing direct-profile cleanup behavior. Do not detach or
disable the guard before the owned browser is actually retired, increase cleanup
grace, or mistake an empty registry for physical exit.

The cancellation control also identifies an adapter completion bug: swallowing
a failed `Route.abort()` leaves Playwright's route handler awaiting an unsettled
completion future. Fix through supported route/framework completion semantics,
never by mutating Playwright internals. A local route fallback is only permissible
after successful owned-browser retirement and confirmed disconnection, when it
cannot dispatch network traffic; otherwise preserve fail-closed rejection.
Require a native held-callback barrier proving bounded caller cancellation,
zero late destination I/O, the lease retained while the callback is held, and
capacity recovered only after its actual exit. This proof must pass before the
owner-order fix is adopted. No startup hook or new subsystem is needed for this
integration repair. Cost if wrong: narrow cleanup/route completion rework.

**Failed-Close Retirement Evidence (2026-11-03):**
Successful native close acknowledgment plus confirmed disconnection is sufficient,
but not the only positive exit evidence: this acquisition's observed native
`disconnected` event with `is_connected() is False` also proves actual exit if
the close command rejected. A bare false state read plus failed close does not.
Keep the guard installed and all physical ownership retained without that proof;
a still-running close command retains its lease even after the native exit event.
An early terminal route error waits on this same existing owner event before
local framework completion. Native rejected/held-close and early-error barriers
verify zero late I/O and actual-drain capacity recovery. No retry loop, poller,
extra connection or startup hook. This refines exit evidence, not ownership rules.

**Legacy Browser Availability Parity (B5):**
Retain the enhanced wrapper's supported uninitialized-browser HTTP/trafilatura
fallback. The old public `scrape_article(method="playwright")` with no initialized
instance browser performs the transport-admission check, then can succeed with
one HTTP request; the new default batch must not replace that working case with
repeated failed browser launches. Preserve the same policy/advice/metadata and
single canonical page-attempt ownership. A transport or policy denial,
`ParentStop`, cancellation or exhausted budget must never become a fallback.
Do not require another installed dependency or new browser startup for a case
that previously worked through HTTP. Test both that legacy availability case and
the initialized native-browser path; do not remove the native parity/security
tests to obtain a green result. This is a verified compatibility repair, not a
new extraction mode or startup hook.

The real default enhanced batch callback still bypasses canonical parent controls:
a matching plan/context network limit of one permits two acquisition calls,
charges zero requests, and reports completion. Fix this before Task 9 acceptance
with a separate focused commit and review. If browser startup ordering remains
blocked after capped attempts, continue this independent repair without accepting
or weakening the browser finding. Bind the
default batch/queue path to existing canonical page composition; page attempts
must be debited once by the actual page owner, not queue and page independently.
Preserve enhanced instance routing, extraction method/advice, cookies, headers,
proxy, rate limiting, deduplication, context, retry and result ordering. Trusted
duck-typed parents and replaced `_execute_job` callbacks remain supported; do not
silently delete callback behavior or claim their opaque work is canonical.
Use real default adapter/acquisition integration tests with only outbound I/O
faked: request and transfer budgets, analyzer stop without fail-open/fallback,
exact page-attempt counts including retry, cancellation and physical drain.
Retain private payload byte admission and callback/credential references only
through actual drain. No additional queue, execution engine, background subsystem
or shadow requests. This repairs the approved governance boundary, not scope for
new functionality. Cost if wrong: narrow adapter/composition rework.

**Interfaces:** Consume `execute_crawl`, `execute_batch`, `ProcessLocalJobQueue`,
and profile resolver. Preserve every existing signature/export/factory in Task 1.
`to_legacy_crawl(report: CrawlReport, *, profile: str) -> list[dict[str, Any]]`
in `contracts/conversion.py` returns normal shapes and established failure lists;
otherwise raises `CrawlInterrupted(report)` on non-normal non-cancelled exits.
Never invent metadata-only articles. Enhanced service internals get reports before
public adapter conversion. Keep progress `save_progress/load_progress` as snapshots.

- [x] **Step 1: Write before/after cutover regression assertions.**

```python
async def test_midcrawl_transport_failure_remains_failure_list(harness):
    result = await harness.run_legacy_midcrawl_transport_failure()
    assert result == harness.characterized_articles_plus_transport_error
    assert harness.completed_callback_calls == 0
    assert harness.checkpoint_exists

async def test_ambiguous_incomplete_list_raises_with_partials(harness):
    with pytest.raises(CrawlInterrupted) as raised:
        await harness.run_direct_deadline_after_article()
    assert raised.value.report.articles[0]["content"] == "kept article"
    assert raised.value.report.outcome.stop_reason == "deadline_exhausted"
```

Also cover pre-admission failures, sync calls without nested event loop errors,
constructor/export identity, queue retries/status/retention, progress load round
trips, extra fields, cookies/deduplication, and no new internal wrapper imports.
- [x] **Step 2: Run `python -m pytest tldw_Server_API/tests/Web_Scraping/test_phase5_adapter_outcomes.py -q`.** Expected: old ambiguous success/callback behavior fails, established failure list still passes.
- [x] **Step 3: Switch adapters to canonical composition and re-export moved queue values through lossless shims.** Delete only now-unreachable private implementations with proven coverage. Keep unrelated cookie/dedup/extraction utilities in place. No dual-run flag or shadow requests.
- [x] **Step 4: Run Task 1/9 plus `python -m pytest tldw_Server_API/tests/WebScraping/test_phase1_compatibility_contracts.py tldw_Server_API/tests/Web_Scraping/test_recursive_crawl.py tldw_Server_API/tests/Web_Scraping/test_recursive_crawl_semantics.py tldw_Server_API/tests/Web_Scraping/test_crawl_config_precedence.py -q`.** Expected: pass with documented approved differences only.
- [x] **Step 5: Commit a standalone rollback boundary:** `refactor(web-scraping): switch crawl and queue compatibility adapters` plus child task. Before reverting, quiesce/drain and preserve versioned checkpoints; old code must not read new files as legacy data. Use the new reader or a separately tested export, never silent downgrade.

### Task 10: Carry Outcomes Through Services And Ingestion

**Files:** Modify `tldw_Server_API/app/services/enhanced_web_scraping_service.py`,
`web_scraping_service.py`, `tldw_Server_API/app/api/v1/endpoints/web_scraping.py`,
`media/ingest_web_content.py`; update response schema only if needed in
`tldw_Server_API/app/api/v1/schemas/media_request_models.py`; create tests
`Services/test_phase5_crawl_outcomes.py`, `Web_Scraping/test_phase5_ingestion_outcomes.py`.

**Task 10 Execution Ruling (Controller Decision, 2026-11-01):**
Resolve Task 4 deferred lifecycle finding M3 while this service is owned: once
shutdown enters the serialized service boundary, its service-owned operation
includes waiting for already-owned startup and then stopping/draining resources.
Cancellation of the request waiter still propagates to that waiter but must not
abandon the shutdown transition before a cleanup task exists. Preserve failed-stop
retry, no duplicate startup/cleanup, ordinary authorized automatic initialization,
verified-admin remote control, and physical stopping until actual drain. Add a
barrier regression with canceled startup waiter, entered shutdown, canceled
shutdown waiter, eventual startup release, exactly one actual stop and truthful
drain/reinitialization state. This closes the previously reviewed transition gap
without a new lifecycle API. Cost if wrong: narrow task/lock ownership rework;
do not cancel startup or release resources early merely to make teardown return.

**Interfaces:** Services consume `CrawlReport` before list reduction.
`public_crawl_outcome(outcome: CrawlOutcome) -> dict[str, Any]` in
`jobs/projections.py` returns exactly state/stop_reason/has_partial_results.
`raise_for_empty_failed_crawl(report: CrawlReport) -> None` in the enhanced service
uses existing service/HTTP exception abstractions. Keep request-presence parsing.
Add `crawl_outcome` to enhanced envelopes, ephemeral stored payloads AND receipts,
persistent/duplicate receipts, ingestion receipts, and owned progress/job views.
Keep separate service operation status and storage disposition. Bare-list internal
consumers use established failure list or `CrawlInterrupted`, never discard outcome.

**Projection Ownership Clarification (2026-11-03):**
Project an aggregate crawl outcome where that report is actually owned: service
task progress, envelopes and receipts. Existing queue jobs are individual pages,
not service-operation/crawl jobs; their live controls are deliberately cleared
after physical drain and are absent from safe polling/history copies. Do not
infer an aggregate outcome from a page's status, retain live controls for polling,
or add a job-to-crawl registry to manufacture that association. The existing
optional typed `public_job(..., crawl_outcome=...)` uses the common projection
when an owning caller actually supplies an outcome; standalone page-job status
keeps its established contract. This clarifies the existing ownership boundary,
not a new aggregate job API or a removal of progress/receipt data.

HTTP failures without articles: overload/stopping/transport 503; policy 403;
deadline 504; budget 429; payload 413; sanitized unexpected/checkpoint failure 500.
Partial articles stay available; only active callers may summarize/store successful
articles under existing independent service controls. Failure-only entries cause
no success storage/summary. No post-processing can mutate terminal crawl outcome.

- [x] **Step 1: Write every reason/receipt/cancellation mapping test.**

```python
async def test_ephemeral_receipt_preserves_partial_outcome(service_harness):
    receipt = await service_harness.partial_ephemeral()
    payload = await service_harness.retrieve(receipt)
    assert receipt["status"] == "ephemeral-ok"
    assert receipt["crawl_outcome"] == payload["crawl_outcome"]
    assert payload["crawl_outcome"]["state"] == "partial"

async def test_late_service_cancel_keeps_crawl_outcome(service_harness):
    await service_harness.publish_crawl_then_block_write()
    with pytest.raises(asyncio.CancelledError):
        await service_harness.cancel_service()
    assert service_harness.progress["crawl_outcome"]["state"] == "completed"
    assert service_harness.progress["service_status"] == "cancelled"
    assert service_harness.new_writes_after_cancel == 0
    assert service_harness.already_committed_article_exists
```

Add cancellation-before-publication, summary failure, duplicate/persistent receipt,
ingestion unwrap, failed-empty HTTP table, partial-plus-secondary-checkpoint-error,
no-article failures, existing mid-crawl error list, storage failure not rewriting
outcome, blocking-write drain, and provenance/summary/cookie/dedup regression.
- [x] **Step 2: Run `python -m pytest tldw_Server_API/tests/Services/test_phase5_crawl_outcomes.py tldw_Server_API/tests/Web_Scraping/test_phase5_ingestion_outcomes.py -q`.** Expected: missing outcome propagation/status separation fails.
- [x] **Step 3: Compose reports and explicit outcome/error projections before existing post-processing, then preserve them through each receipt and stored payload.** Reuse database abstractions; no new persistence engine or re-ingestion authorization from checkpoints.
- [x] **Step 4: Run new tests, Task 4 security tests, and existing enhanced/legacy scraping service and media ingestion contract tests identified in Task 1.** Expected: active-caller partial processing retained, no false success, no new service side effects after cancellation.
- [x] **Step 5: Commit standalone service cutover and child task:** `fix(web-scraping): preserve crawl outcomes through storage and ingestion`.

### Task 11: Certify Architecture, Consumers, And Physical Lifecycle

**Files:** Extend `WebScraping/test_refactor_import_inventory.py`,
`Web_Scraping/test_phase4_consumer_imports.py`; create
`Web_Scraping/test_phase5_architecture.py`,
`WebScraping/integration/test_phase5_crawl_lifecycle.py`; modify only affected
consumer files enumerated by Task 1 inventory; refresh
`Docs/Design/WebScraping.md`,
`Docs/Design/WebScraping_Refactor_Import_Inventory.md`,
`Docs/Design/web_scraping_refactor_import_inventory.json`.

**Interfaces:** AST architecture guard enforces no imports from crawl/jobs into
orchestration/services/wrappers (including importlib literal imports), no cycles,
and one canonical page execution route. Consumer tests retain current analyzer
advice/results, cancellation/error conventions and non-secret extra fields.
No central Jobs migration or search-provider decomposition.

- [ ] **Step 1: Write architecture and combined lifecycle tests.**

```python
def test_lower_modules_do_not_import_execution_owners(import_graph):
    assert import_graph.forbidden_lower_edges() == []

async def test_shutdown_does_not_release_live_thread_capacity(browser_harness):
    await browser_harness.cancel_crawl_and_stop_service()
    assert browser_harness.queue_state == "stopping"
    assert browser_harness.executor_active_permits == 1
    await browser_harness.release_extraction_thread()
    assert browser_harness.queue_state == "stopped"
    assert browser_harness.late_results_installed == 0
```

Guard tests include negative fixture modules proving forbidden edges are detected.
Use local controlled browser integration for redirects, subresources/WebSockets,
transfer stop, cancellation and same-process resume exclusion. Retain shared
executor reload/shutdown/fork and attestation suites; do not replace them with fakes.
- [ ] **Step 2: Run new architecture/integration tests.** Expected: guard negative fixtures fail when the checker is disabled; any uncovered wiring/drain regression fails behavior assertions. No artificial production breakage is needed for already-green invariants.
- [ ] **Step 3: Fix scoped consumer adaptation gaps, refresh actual docs and regenerate inventory once.**

```bash
python Helper_Scripts/web_scraping_refactor_inventory.py --root . --json Docs/Design/web_scraping_refactor_import_inventory.json --markdown Docs/Design/WebScraping_Refactor_Import_Inventory.md
```

- [ ] **Step 4: Run certification and record exact results/skips in the child task.**

```bash
python -m pytest tldw_Server_API/tests/Web_Scraping/ tldw_Server_API/tests/WebScraping/ tldw_Server_API/tests/http_client/ -q
python -m pytest tldw_Server_API/tests/Services/ -k 'webscrap or web_scrap or phase5' -q
python -m pytest tldw_Server_API/tests/Resource_Governance/test_rg_cutover_evals_authnz_character_web.py -q
```

Also run exact Security, Watchlists, Workflows, Collections, RAG, Web Search,
evaluation and ingestion tests from Task 1's refreshed import inventory. Record
explicit paths/node IDs, including any absent category (not an invented test path).
Run existing browser smoke and executor fork/reload/shutdown tests with required
local Chromium available. A missing mandatory environment is a recorded gate,
not a passing certification. Compare baseline failures before calling them new.

Run compile/lint on all touched Python and scoped pre-commit on explicit paths.
Run Bandit in the activated venv over touched production packages/files, including
shared HTTP, services and endpoints; save JSON to `/tmp/bandit_web_phase5.json`.
Fix new findings and document pre-existing findings without suppressing tests.
Run backlog-py normalization, task-format CI ratchet, relative links, and
`git diff --check`. Require a fresh whole-branch review after task reviews, then
rerun affected verification after review fixes. Stop after three failed attempts
on an issue and document/reassess per AGENTS.md.
- [ ] **Step 5: Commit explicit certification/docs/consumer paths and child task:** `test(web-scraping): certify Phase 5 consumers and lifecycle`. Update all four stage statuses only with recorded evidence. Keep this approved plan as the task-linked design/execution record; remove only disposable task-owned working plans if repository policy requires it.

## Cutover And Recovery Gates

Stage 1 contracts and Stage 2 new queue can land without switching traversal.
Stage 2 authorization fixes are independently deployable. Stage 3 adds tested
canonical internals without shadow-running live requests. Task 9 switches wrapper
ownership once; Task 10 switches service report handling once. Keep those commits
separate from relocation so regressions have identifiable rollback boundaries.

Do not roll back to an unsafe queue/projection path as an unreviewed emergency
shortcut. Quiesce affected service admission and drain owned work before changing
queue ownership. Preserve versioned checkpoints; a pre-Phase-5 binary cannot be
assumed to load them. If rollback is required, keep the validated reader available
or use a separately tested trusted local export after credentials are reacquired.
Do not delete checkpoint/result data or claim rollback reverses committed ingestion.

Before PR readiness, fetch `dev` again, review/rebase in the isolated branch as
appropriate, rerun affected certification, and check all PR comments. Report draft,
ready, checks, merge and cleanup states from actual GitHub state, not inference.
No merge before the human-written change-summary gate. Durable Jobs and broad
Phase 6/7 cleanup remain separate work items.

## Plan Self-Review And Coverage

| Approved requirement | Owning tasks |
| --- | --- |
| Public/profile/presence/metadata parity and consumer inventory | 1, 2, 9, 11 |
| Finite defaults, parent accounting, analyzer and transport controls | 2, 5, 6 |
| Queue admission, retry reservation, subscriptions, retention, drain | 3, 8, 11 |
| Verified ownership/lifecycle authorization and secret-safe projections | 4, 7, 10 |
| Discovery identity, filters/robots, traversal, successful/result counters | 1, 5, 6, 9 |
| Versioned/legacy checkpoints, unknown history, safe storage and exclusion | 7, 9, 11 |
| Bounded producers and accepted-child ownership | 3, 8 |
| Immutable terminal outcome, primary failure precedence, legacy errors | 2, 7, 9, 10 |
| Partial/failed/cancelled service, ephemeral, persistent, ingestion outcomes | 10 |
| Browser/executor/consumer regression and no reverse dependencies | 5, 11 |

Self-review completed 2026-11-01 for coverage, decisive red/green steps,
type/interface consistency, the five Review Focus cases, and proportion to the
approved spec. Corrections made during review: atomic page subscriptions rather
than post-admission lookup, protocol-only lower dependencies, async admission
composition, resolved article-plan propagation, single accounting owners, and
actual conversion/architecture document paths. This record is
planning evidence only: the future tests above have not been implemented or run.
Any new architecture decision or material compatibility deviation returns to
design review; task-level implementation details stay within the approved plan.
