# Web_Scraping Phase 5: Crawl And Process-Local Jobs

**Date:** 2026-10-04
**Last amended:** 2026-11-01
**Status:** Approved by the human requester on 2026-11-01
**Task:** TASK-15282
**Implementation:** Not started; implementation-plan approval and execution-method selection required
**Code baseline:** `75ab224081bf140ef52017c1a9b0a04f6878d488` (`origin/dev` inspected on 2026-10-04)
**Integration check:** `54b321dcec8cdbb6358581f42c1548696260570b` inspected on 2026-11-01; design branch not rebased

## Intent And Success Criteria

Make web scraping maintainable, easily extendable, and stable without losing
existing functionality. Phase 5 separates crawl/discovery and job-state ownership
from article extraction while retaining the current process-local queue.

The user selected retaining that queue now and considering migration to the
central durable Jobs backend later. Preserving the pre-scrape analyzer is a hard
requirement, including its configuration, advice precedence, and attached results.
Two conversational review passes supplied the execution, authorization,
serialization, and continuation refinements incorporated below.

Success means:

- Each acquired page uses the canonical, governed article orchestration.
- Crawl and queue state have clear owners and injected execution dependencies.
- Actual crawl work, not only successful articles, is bounded and cancellable.
- Existing public imports, constructors, signatures, result shapes, and documented
  entry-point differences survive the cutover.
- Interrupted legacy recursive crawls can continue without silently losing the
  interrupted page; progress snapshots are not misrepresented as checkpoints.
- Authorization, credential redaction, terminal outcomes, and resource drain are
  deterministic and covered by focused regression tests.

## Scope

In scope are sitemap and link discovery, traversal/frontiers, filtering/scoring,
crawl budgets and cancellation, process-local queue priority/retry/lifecycle,
status/progress projections, legacy recursive checkpoints, and the adapters and
consumer changes needed to establish these boundaries.

Summarization, ingestion, persistence, and application lifecycle integration remain
service responsibilities. Cookies, deduplication, raw-browser APIs, extraction
methods, and preflight functionality remain supported through their existing
governed mechanisms; this phase does not replace those implementations.

Out of scope:

- Central `app/core/Jobs` migration, distributed scheduling, durable job receipts,
  restart recovery of queued jobs, or a new public pause/resume API.
- Search-provider decomposition (Phase 6), wholesale wrapper/dead-code cleanup
  (Phase 7), a new extraction engine, or a new persistence subsystem.
- Exactly-once fetching or ingestion across abrupt process death.
- Broad fairness, deduplication, cookie-storage, or configuration redesign.

## Architecture Decision Assessment

**ADR required:** Yes.
**Accepted ADR:** [ADR-085](../../ADR/085-web-scraping-crawl-and-process-local-job-ownership.md).

The durable rule is that crawl and queue own their state and receive execution
callbacks, while orchestration composes them with canonical article execution.
This also defines cancellation and continuation ownership. The human requester
approved the written design and ADR on 2026-11-01; no implementation is claimed.

Existing decisions continue to govern:

- [ADR-003](../../ADR/003-jobs-vs-scheduler-default.md): defaults for new Jobs and
  Scheduler work. Retaining an existing process-local queue is scoped to this
  refactor, not a new default for user-visible background features.
- [ADR-021](../../ADR/021-services-lifecycle-startup-and-shutdown.md): service
  lifecycle ownership and bounded teardown.
- [ADR-026](../../ADR/026-security-outbound-egress-and-ssrf-policy.md) and
  [ADR-042](../../ADR/042-browser-transport-admission-and-attestation.md): guarded
  egress and browser admission.
- [ADR-051](../../ADR/051-retry-backoff-schedules.md): outbound versus contention
  backoff schedules; this split does not introduce a conflicting retry policy.
- [ADR-002](../../ADR/002-backlog-md-task-tracking.md) and
  [ADR-059](../../ADR/059-backlog-py-task-editor-cutover.md): task tracking/editor.

No accepted ADR rationale is changed or superseded by this specification.

## Approach And Alternatives

Use staged consolidation: first characterize the entry points, then move coherent
stateful responsibilities behind explicit interfaces, and finally switch adapters
and consumers through verified cutovers.

| Approach | Assessment |
| --- | --- |
| Mechanical relocation only | Smaller immediate edits, but preserves intertwined execution, queue, and traversal ownership and their races. |
| Staged consolidation | Selected: establishes clear ownership incrementally while preserving entry-point behavior. |
| Immediate unified replacement, including durable Jobs | Combines incompatible concerns and migration risks; durable Jobs was explicitly deferred by the user. |

## Components And Dependency Direction

Paths below are relative to `tldw_Server_API/app/core/Web_Scraping/`.

| Component | Owns | Dependencies |
| --- | --- | --- |
| `crawl/` | Discovery, normalized frontier entries, traversal strategies, filtering/scoring composition, checkpoint state | Pure contracts and injected page acquisition/policy/checkpoint capabilities |
| `jobs/` | Existing process-local queue, priority, retries, job transitions, completion subscriptions, bounded history | Pure contracts and an injected execution callback |
| `orchestration/` | Resolved crawl plan, live crawl context, composition with canonical per-page article execution | Crawl, jobs, and existing article/runtime/preflight boundaries |
| Services | Trusted principal resolution, lifecycle authorization, summaries, ingestion, persistence, public projections | Canonical orchestration plus compatibility adapters where still required |
| Legacy modules | Existing imports/signatures and conversion into/out of canonical execution | Canonical APIs; no independent new crawl or extraction engine |

`Web_Scraping/jobs/` is distinct from the central `app/core/Jobs` package. Neither
`crawl/` nor `jobs/` imports orchestration, application services, or legacy wrappers
to execute a page. Orchestration supplies that capability. New lower-level modules
must not import legacy modules; cutover tests prevent new cycles and reverse
dependencies. Existing unaffected utilities are not relocated simply for naming
consistency.

### Internal Contracts

- Existing public `CrawlRequest` and `CrawlResult` retain constructor and export
  compatibility. They are not silently replaced by a parallel public model.
- An immutable internal crawl plan records resolved options, the entry-point
  compatibility profile, and request/config provenance. Conversion retains
  omitted versus supplied values wherever the existing API exposes that distinction.
- A live `CrawlContext` owns cancellation, execution deadline, resource accounting,
  and crawl-owned task/job/drain handles. It is not serialized as request metadata.
- A bounded discovery result carries links and document-location provenance from
  admitted acquisition, independently of the public article result.
- An internal crawl execution report carries articles plus their terminal outcome
  to services without changing public `CrawlRequest`/`CrawlResult` constructors.
  Compatibility conversion must not erase that report before service decisions.
- Private job execution controls hold credentials and executable state. Public
  job/progress models are explicit allowlisted projections, not dataclass dumps.

Use the existing immutable article/preflight/runtime models where appropriate.
Do not add abstractions that merely rename a function or duplicate those contracts.

## Data Flow And Pre-Scrape Analysis

1. A service or local compatibility adapter resolves existing options and creates
   the plan and live context. Services derive owner/admin identity from the trusted
   authenticated principal, never from an arbitrary options dictionary.
2. Crawl admits a seed or discovered candidate through the selected compatibility
   profile, current URL/robots policy, and parent resource controls.
3. Orchestration submits or directly executes a canonical per-page article request,
   retaining credentials, context, policy, and analyzer configuration.
4. The standard article plan performs optional governed preflight, routing,
   acquisition, extraction, and enrichment. Existing explicit-choice versus
   analyzer-advice precedence remains unchanged; queue retries do not bypass it.
5. Admitted acquisition produces bounded discovery information. Crawl updates its
   frontier and result state; queue records its own completion state independently.
6. Compatibility conversion preserves the public envelope and plain-dict metadata.
   Services then perform their existing summary/ingestion/persistence work.

Discovery uses the acquired document's effective location and the selected
compatibility profile's link-resolution behavior. Request URL, final URL, and
parent/depth provenance are not conflated. Do not add an unauthenticated second
fetch to recover links from an extracted-text-only result. Any necessary additional
acquisition uses the same credential/policy scope and consumes the same budgets.

Raw acquisition HTML must not be added to job history or progress as a discovery
sidecar. Intentionally requested public article/raw-browser content remains part
of its existing API; this prohibition must not erase supported output formats.

Parent cancellation, deadline expiry, and resource exhaustion are control outcomes,
not analyzer-local recoverable failures. They stop admission and cannot trigger
analyzer fail-open, browser fallback, or queue retry. Normal analyzer-local behavior
continues within the remaining parent allowance.

## Crawl Execution And Accounting

Preserve each entry point's `max_pages` meaning. Legacy recursive extraction counts
successful articles; enhanced traversal currently caps collected results. A shared
frontier does not silently make these counters equivalent. Characterization tests
record the exact counter, ordering, and failure inclusion of each profile.

Separately enforce validated, finite parent safety limits on page attempts, network
and probe work, bytes, discovered candidates, and frontier size. The implementation
plan must identify the numeric defaults and their source/override rules before code
approval; this spec defines required dimensions and semantics, not an arbitrary
new public configuration surface. Invalid limits fail validation rather than
silently disabling protection.

Accounting rules:

- Reserve admission atomically before launching work. Competing pages/retries must
  not collectively overrun a parent limit.
- Parent and local controls compose: both must admit work before I/O. Reservation
  failures before work starts roll back unused reservations consistently.
- Started attempts remain charged when fetching/extraction fails or a waiter leaves.
  Redirects, retries, discovery fetches, robots work, and preflight consume their
  relevant transport/probe allowances; logical pages and physical work are separate.
- Assign one accounting owner per physical operation to avoid double charging a
  nested helper. Capacity leases and cumulative usage counters are separate concepts.
- Enforce bytes during acquisition, not after an oversized response has been fully
  buffered. Discovery admission and frontier checks occur before growing collections.
- Page timeouts are capped by the remaining parent execution deadline. Queue waits,
  rate-limit delays, and retry delays count while the crawl execution is active.
- Cleanup uses existing bounded cleanup semantics; expiration of the execution
  allowance does not suppress resource cleanup.

No result-only success count, retry loop, or link-expansion path can provide
unbounded work around these controls. Resource exhaustion has an explicit reason;
the admission boundary does not silently truncate results and report full success.

## Process-Local Queue Lifecycle

Queue states are `running`, `stopping`, and `stopped`. Startup/service initialization
is serialized and idempotent; workers are not duplicated by concurrent requests.
Stop atomically closes submission and retry admission. A rejected submission does
not become an accepted orphan, and a failed retry admission terminalizes its job.

### Aggregate Admission And Overload

Per-crawl controls do not bound a queue shared by many independent requests.
Require finite, validated queue-wide limits on pending jobs, total live jobs
(including physically draining jobs), each normalized execution payload's size,
and aggregate retained execution-payload bytes. All priorities share those limits;
creating another crawl or using a higher priority cannot create a fresh allowance.
Keep `max_concurrent` as a distinct execution limit and validate that worker
configuration can make progress. The implementation plan selects numeric defaults
and their configuration provenance before code approval.

Admission is immediate: atomically reserve the required capacity and publish the
job, or reject it with a sanitized typed `queue_overloaded` failure before it is
accepted. Accepted submissions keep the legacy Future-facing return contract;
rejected submissions retain no job, future, or credential-bearing payload. Do not
hold the lifecycle lock while awaiting a bounded queue's `put`, and do not create
an unbounded population of external admission waiters. This explicit overload
failure is an approved safety correction, not an implicit retry request.
An individually oversized normalized payload instead fails validation as
`job_payload_too_large`, without reserving capacity; unlike aggregate queue pressure,
it is not a transient overload and maps to HTTP 413 at the service boundary.

Each accepted job holds one admission lease across pending, active, and retry
transitions. A retry reuses that lease rather than competing with new submissions
for capacity or duplicating its payload. Terminalization releases capacity only
when the corresponding execution material can actually be discarded; physically
draining work remains charged until it exits. Stop/retry/cancel races release each
lease exactly once and cannot deadlock all workers attempting to requeue.
The pending-slot allowance includes return-to-queue reservations held by active
retry-eligible jobs. New submissions cannot borrow those reservations; a retry can
requeue immediately even when all unreserved pending capacity is full. A terminal
job cannot retry, but its live-job/byte charge remains until any physical drain ends.

Completed history has an independent byte ceiling as well as its existing count
retention. Evict oldest entries to satisfy both; a result too large to retain is
still delivered to its live subscriber but need not be cached. Preserve existing
zero-retention and not-found-after-eviction behavior. These bounds cover queue-owned
data; they are not a promise to bound the entire process's resident memory.

Every accepted job reaches one terminal state, and every live completion
subscription is settled at most once. Terminalization atomically updates job state,
removes pending/active bookkeeping, and publishes completion. Already canceled
subscriptions are observed and removed without invalid-state errors. Retention
eviction is independent of settling subscribers and releasing execution resources.

Retain priority values, ordering, retry limits, result/error conventions, and public
status values except for the explicitly approved correctness/security corrections.
Do not redesign scheduling fairness during this split. Polling timeouts, execution
timeouts, cancellation, and unexpected failures have distinct handling; an execution
timeout must not disappear into the idle-poll path.

### Bounded Batch Production

Batch and crawl orchestration must not enqueue an entire input and only then await
its jobs. Use a finite producer window, bounded by the batch's work allowance and
the shared queue's count/byte limits. Materialize job payloads and completion
subscriptions only for that window, collect completed results incrementally, and
preserve the profile's existing result order and per-page failure entries. A batch
within its work limits may exceed the window size without being rejected merely
because all of its URLs cannot fit in the queue simultaneously.

Queue admission itself remains immediate. When the producer reaches its own window
or encounters capacity pressure while it still owns in-flight jobs, it waits for
an owned completion before attempting further admission. This is bounded producer
pacing, not an external admission-waiter queue or a retry of an accepted job. Each
new attempt follows owned progress; do not poll, sleep-loop, or bypass shared
capacity. If admission is rejected with no in-flight owned work to await, report
the overload immediately. All waits remain subject to the parent deadline and
cancellation; hold no queue lifecycle lock while awaiting a completion.

Track accepted children as a batch-owned group from their admission point. If
cancellation, parent exhaustion, or an unrecoverable admission failure aborts the
batch, stop producing, cancel only its unfinished children, and observe/settle all
their subscriptions through bounded logical cleanup. Preserve completed articles
and failure entries in the internal report or existing public failure contract.
Physical drains retain their queue/executor leases and may outlive the response;
they are not abandoned or represented as completed results. Ordinary per-page
failures still follow the existing profile and do not automatically abort siblings.

### Cancellation Ownership

| Event | Effect |
| --- | --- |
| Standalone caller stops awaiting | Cancel/detach that subscription; queue-owned work continues unless explicitly canceled. |
| Explicit `cancel_job` | Stop matching pending/running work cooperatively; never cancel the shared worker or global executor. |
| Owning crawl is canceled | Close its admission and cancel its owned jobs/tasks; unrelated work remains intact. |
| Shared service shutdown | Close global queue admission and perform the authorized service-owned stop/drain sequence. |

Keep the legacy awaitable/Future-facing signature through a compatibility adapter,
but do not expose the queue's sole internal completion primitive. `CancelledError`
remains cancellation within async execution; it is not converted to a normal
successful list element. Public best-effort cancellation acknowledgement does not
claim a running synchronous function has been preempted.

### Terminal State Versus Physical Drain

An extraction thread may outlive its canceled coroutine. Retain its drain handle,
context references, and existing executor permit until the underlying work exits.
Do not refund active capacity early, install its late result, or resurrect the job.
Do not tear down dependencies that still-owned work needs while claiming they are
drained.

The queue remains `stopping` while its owned work is draining. Bounded service
teardown may return a truthful drain-pending outcome; it cannot report physical
`stopped` merely because all public futures are terminal. A replacement queue must
not bypass the shared executor's outstanding-permit ownership. This phase does not
introduce thread-killing or a new executor lifecycle.

## Authorization And Public State

Shared lifecycle endpoints require a verified platform-admin principal. Individual
job status/cancellation retain owner-or-admin authorization. Resolve authorization
before acting on the job or shared service; client-provided owner/role fields are
not authoritative. Preserve established not-found/unauthorized response semantics
while fixing any lifecycle access gaps.

Public state contains only documented status, timing, safe counters, public result
fields, and sanitized failure information. Do not return cookie values,
authorization headers, execution callbacks, credentials embedded in options, or
private runtime objects through job/status/progress serialization. The redaction
boundary includes nested values and logs, not just top-level cookie fields.

Keep private execution material only as long as accepted or draining work needs it;
terminal public history does not retain credential-bearing execution options.
Bounded history retains the existing retention behavior, including zero-retention
support. Progress and diagnostic projections do not grow an independent unbounded
copy of article content or discovery state.

## Progress And Continuation

### Distinct Artifacts

A progress snapshot is an observation of current state. Enhanced `save_progress`
and `load_progress` remain snapshot operations and do not become durable resume.

A recursive-crawl checkpoint contains enough frontier/result state for its supported
continuation path. Retain legacy checkpoint loading and explicit local `resume_file`
support. New writes use a versioned format and preserve ordered frontier restoration.
Do not expose arbitrary filesystem paths as a new remote checkpoint API.

### Versioned Checkpoint Contract

The checkpoint records:

- Schema version, crawl identity, normalized seed and compatibility-profile binding.
- Effective behavior-affecting options and their safe provenance, excluding credentials.
- Ordered queued entries, in-flight entries, completed dispositions, and committed
  result/deduplication state needed by the existing continuation API.
- Known consumed counters, remaining execution time, accounting completeness, and
  the last stop/checkpoint disposition.

An admitted page is not completed merely because its URL is in a visited set.
Checkpoint result commit, frontier expansion, and completed disposition form one
consistent snapshot. On cooperative interruption, safely restore unfinished entries
in their original admission order; keep spent attempts charged. Completed results
are not appended again just because their URLs remain discoverable.

Within one process, do not resume a crawl while its previous owned execution is
still draining. Keep the crawl/checkpoint ownership handle until that drain ends;
a terminal caller future alone is not permission to start a second execution of
the same in-flight frontier.

This is resumability, not exactly-once I/O: interrupted fetches may be repeated and
work after the last committed snapshot may be replayed after abrupt process death.
Do not promise crash-proof lifetime accounting from periodic JSON checkpoints.
Known committed consumption is never silently reset. Any uncertain post-checkpoint
consumption is marked unknown, and continuation remains explicitly bounded without
claiming that missing historical work was measured. A future durable execution
design must address the stronger crash-consistent ledger guarantee separately.

Store remaining active-execution duration, not a process's monotonic deadline.
Reconstruct a monotonic deadline on resume. Time while paused between executions is
excluded; it does not restore time already spent in a known execution. Resume does
not replenish known exhausted limits or change behavior-affecting options silently.

### Legacy Migration

Load validated legacy `visited`, `to_visit`, `scraped_articles`, and `pages_scraped`
without discarding their existing result/frontier meaning. Restore frontier order.
Missing attempt/byte/time history is explicitly unknown, never manufactured from a
successful-page count. Legacy continuation receives a bounded current-execution
allowance and writes the versioned format thereafter, retaining the unknown-history
annotation. Existing legacy checkpoints cannot recover an interrupted URL that was
never recorded; the new writer prevents that loss prospectively.

### Storage And Trust

Service-owned paths come from a trusted owner/crawl namespace. A checkpoint's own
owner field is not sufficient authorization. Validate seed/profile/options binding,
schema version, field types, depth/counter ranges, collection lengths, and file size
before loading executable state; re-admit pending URLs under current egress policy.
Reject unsupported versions and invalid structure with sanitized diagnostics.

Resume reacquires private credentials from the same trusted owner-scoped mechanism
or explicit local caller input; it does not recover cookie/header secrets from the
checkpoint. Missing or invalid credentials follow the existing acquisition failure
contract and cannot cause an unauthenticated discovery fallback. Re-evaluate current
credential and policy scope even when seed/options binding matches the checkpoint.

Use exclusive single-writer ownership for each checkpoint, including process-level
coordination if a path can be shared across workers. Unique temporary files and
atomic replacement prevent partial replacement; a fixed shared `.tmp` name is not
a lock. Keep the previous valid checkpoint if serialization/replacement fails. Do
not follow untrusted service paths or symlink escapes. Explicit local-path support
is a trusted local API compatibility capability, not permission to accept a remote
arbitrary path. Credential-bearing URLs/private controls are not serialized into
server-owned public progress or histories.

### Stop Outcomes And Retention

| Stop reason | Meaning | Checkpoint action |
| --- | --- | --- |
| Frontier exhausted | Requested traversal finished; individual page failures may remain in results. | Clear only after normal finalization succeeds. |
| Profile page cap reached | Requested bounded traversal finished according to that entry point's counter. | Clear only after normal finalization succeeds. |
| Caller/crawl cancellation | Interrupted; preserve async cancellation semantics. | Retain a valid continuation checkpoint when supported. |
| Deadline or parent resource exhaustion | Partial work; no fallback or retry around parent limits. | Retain; known exhausted limits remain exhausted on resume. |
| Browser transport unavailable or crawl-level policy/runtime failure | Interrupted/failed, not a successful completion callback. | Retain last valid checkpoint and safe partial results where the profile permits. |
| Checkpoint write failure | Resume persistence failed. | Do not delete the previous valid checkpoint; surface this failure without masking the primary exception/cancellation. |

Per-page failures follow the existing profile's result/continuation policy and do
not automatically become a crawl-level abort. Internal outcomes carry explicit
reasons; adapters preserve public list/dict shapes and do not fabricate a success
callback for interruption. Result collection is separate from service ingestion;
the checkpoint does not authorize re-ingesting already committed service outputs.

### Service And Public Outcome Contract

The internal report's authoritative outcome has `state` (`completed`, `partial`,
`failed`, or `cancelled`), an allowlisted `stop_reason`, and
`has_partial_results`. It contains no article copies or execution credentials.
`partial` means an interrupted traversal with collected articles, not a normal
page-cap stop; a completed traversal can still contain individual page failures.
For this classification, collected articles means successfully extracted article
data; failure-only page entries are retained as diagnostics according to the
profile, but do not alone turn a failed crawl into a partial one.
Allowlisted interruption reasons include `deadline_exhausted`, `budget_exhausted`,
`transport_unavailable`, `policy_denied`, `execution_failed`, `checkpoint_failed`,
`queue_overloaded`, `queue_stopping`, `job_payload_too_large`, and `cancelled`.
Normal completion uses `frontier_exhausted` or `page_cap_reached`. Unknown internal exceptions become a
sanitized `execution_failed` outcome, not an arbitrary exception string.

Publish the terminal crawl outcome once, after crawl result/checkpoint finalization
has settled and before service summarization or persistence starts. Physical drain
may continue under its existing ownership. Cancellation and terminal publication
share one atomic transition: cancellation committed first produces a canceled crawl;
an already published terminal outcome is immutable. A later cancellation during
service post-processing still propagates `CancelledError` and cancels the service
operation, but it does not rewrite a completed or partial crawl outcome. Owned
progress/job projections distinguish their service-operation status from the nested
`crawl_outcome`; a canceled service may therefore contain a completed crawl outcome.

Services must consume the report before reducing it to article lists. Preserve
existing result envelopes and add the bounded `crawl_outcome` object to enhanced
service responses, ephemeral stored results and receipts, persistent receipts, and
owned progress/job projections. Existing `ephemeral-ok`, `persist-ok`, and
`duplicate` values remain storage dispositions; they must not overwrite or stand
in for the crawl outcome. Retrieving an ephemeral payload preserves the same
outcome as its receipt. An ingestion adapter such as `_ingest_crawl_articles`
must inspect and propagate the outcome before unwrapping `articles`/`results`.

| Boundary/outcome | Public behavior | Summary and persistence behavior |
| --- | --- | --- |
| Normal traversal or requested profile page cap | Existing successful list/envelope shape; services additionally expose `crawl_outcome.state=completed`. | Preserve requested summaries and existing ephemeral/persistent/ingestion behavior, including individual article failure handling. |
| Interrupted traversal with collected articles | Service returns its existing envelope with `crawl_outcome.state=partial` and the explicit reason; it does not claim traversal completion. | Preserve collected articles. If the caller remains active, requested summaries and storage may process successfully extracted articles under their own existing service controls; never refetch/retry the stopped crawl. Carry the partial outcome through storage and ingestion receipts. |
| Failed traversal without articles | Report `state=failed`; expose the sanitized reason instead of manufacturing a successful empty crawl. With no collected articles, queue overload/stopping and unavailable transport map to HTTP 503, policy denial to 403, parent deadline to 504, parent resource exhaustion to 429, and oversized job payload to 413; unexpected crawl failure uses the existing sanitized 500 path. | Do not invoke article summarization or article persistence. Existing error diagnostics may be recorded without creating a success receipt. |
| Caller/crawl cancellation before terminal crawl publication | Propagate `CancelledError`; there is no normal successful response. The crawl outcome records `state=cancelled` and `stop_reason=cancelled`; owned status/progress records the canceled operation. | Stop scheduling new summaries or writes; preserve an available crawl checkpoint. Work already committed is not rolled back, and already-running blocking writes retain their existing drain semantics. |
| Caller cancellation during service post-processing after terminal crawl publication | Propagate `CancelledError` and mark the service operation canceled; preserve the previously published `crawl_outcome` unchanged in owned status/progress. | Stop scheduling new summaries or writes. Preserve already committed results and existing blocking-write drain semantics; cancellation does not restart or reclassify the finished crawl. |
| Legacy/direct list-returning API after an interrupted traversal | Keep list returns for normal completion and established structured failure-list contracts both before and during traversal, including collected articles followed by an existing transport-failure entry. Only non-normal exits without an established failure representation raise a sanitized typed interruption carrying the outcome and partial results. | Direct helpers do not gain service side effects. Services use the canonical report to preserve the interruption reason independently of the compatible list representation. |

The new typed interruption for formerly ambiguous incomplete list returns is an
explicit correctness change, not a removal of article data. Cancellation remains
`CancelledError`, not that interruption type. Preserve the legacy recursive helper's
existing mid-crawl `browser_transport_unavailable` entry and its preceding articles;
do not replace that already structured failure with a new exception. Fix the false
completion callback and checkpoint retention independently of its list shape.
Failure-list compatibility does not extend to enhanced paths that formerly returned
an ambiguous empty list: those paths expose a failed outcome through their report
or typed direct-call failure. Do not invent fake articles for crawl metadata.
Record the exact before/after mapping in each profile's contract tests.

Summary/storage failures remain separate from the crawl outcome: a later storage
success cannot turn a partial crawl into a completed one, and a failed summary or
write must use its existing observable failure channel. Where a compatibility
consumer ultimately requires a bare list, use its established structured failure
representation, or a typed failure carrying partial results where no such contract
exists, rather than losing the interruption during envelope unwrapping. Later
post-processing failure or cancellation cannot mutate the published crawl outcome.

If checkpoint persistence fails while another interruption is already primary,
keep that primary state/reason and report the sanitized checkpoint failure as
secondary diagnostic data. If the
checkpoint failure itself stops an otherwise active crawl, it is the primary
`checkpoint_failed` reason. This preserves the exception precedence above.

## Compatibility Matrix And Cutover

The implementation plan must map each imported entry point to a named compatibility
profile and its characterization tests. At minimum:

| Surface | Contract to preserve |
| --- | --- |
| Legacy async/blocking recursive helpers | Arguments/defaults, successful-page count, traversal/result order, cookies/user agent, callback behavior except false success, local checkpoint loading |
| Enhanced recursive scraper | FIFO/best-first selection, relative path-depth checks, collected-result page cap, filter/score options, plain-dict traversal metadata and provenance |
| Sitemap and URL-level helpers | Source formats, filtering, defaults, sync/async calling convention, result envelope, policy and credential propagation |
| Enhanced queue and factory exports | Job constructors, priority/status values, awaitable-facing submission, retry bounds, best-effort cancellation, retention and unavailable-service reporting |
| Legacy/enhanced services and endpoints | Omitted/request/config precedence, authorization, task/owner association, summary/ingestion/persistence envelopes and error sanitization |
| Public contracts and downstream consumers | `CrawlRequest`/`CrawlResult` exports, legacy imports, metadata round trips, Watchlists/Workflows/Collections/RAG/Web Search/evaluation consumers actually present in inventory |

Approved intentional changes are bounded actual work, secure lifecycle access,
credential-safe public projections, terminalization/cancellation correctness, and
checkpoint interruption/retention fixes. They also include explicit overload and
incomplete-crawl reporting through the mappings above. Characterization does not
freeze those bugs. Conversely, those fixes are not permission to normalize unrelated entry-point
differences or remove supported features.

Use lossless adapters for non-secret article extra fields and nested plain-dict
metadata. Keep request presence information at parsing/adaptation boundaries rather
than guessing omission from a value equal to a default. Refresh the import inventory
and architecture documentation as consumers switch. Remove only newly superseded
private code with proven ownership and coverage; broader cleanup remains Phase 7.

Compare captured fixtures and deterministic fake executions. Do not shadow a live
request by issuing duplicate network, analyzer, summary, or ingestion operations.

## Verification And Acceptance

Use repository pytest fixtures, fake transports, injected clocks, and explicit
synchronization barriers. Avoid sleep-dependent race tests and real external APIs
for unit/contract coverage. Retain browser integration coverage where transport and
resource ownership cannot be proven by fakes alone.

Required coverage:

1. Public imports/signatures/constructors, return envelopes, configuration presence
   and precedence, extra-field/metadata conversions, and consumer inventory parity.
2. FIFO/best-first order, score tags, seed-relative depth, profile-specific page
   counts, sitemap/link identity, filters/robots, and cyclic/duplicate frontiers.
3. Atomic competing admissions, failed/retried attempts, redirects/preflight and
   additional discovery work, byte enforcement, bounded collections, and parent
   control outcomes bypassing neither analyzer nor browser nor queue guards.
4. Submission versus stop, retry versus stop, cancel versus completion, polling
   versus execution timeout, detached waiters, retention eviction, and exactly-once
   terminal publication. Assert workers/futures/bookkeeping do not remain orphaned.
   Exercise multiple individually bounded crawls and standalone callers sharing
   count/byte capacity, all priorities, oversized payload rejection before admission,
   overload versus shutdown, retry lease reuse at full capacity, draining lease
   release, and byte/count history eviction without losing subscriber delivery.
   Exercise batches larger than the producer window but within their work budgets,
   stable output order across windows, bounded materialized payloads/subscriptions,
   shared pressure with and without owned in-flight work, and cancellation/failure
   after partial admission without orphaning children or touching unrelated jobs.
5. In-flight interruption, stable restored order, committed result deduplication,
   legacy and versioned loading, known budget/time preservation, unknown-history
   disclosure, invalid files/versions/bindings, concurrent writers, atomic replace
   failures, and preservation of the original cancellation/exception.
6. Owner/admin authorization and shared lifecycle denial, nested credential-safe
   projections/logs, egress on resumed URLs, and unchanged browser attestation.
7. Late extraction completion, retained permits until physical exit, no late result
   resurrection, shutdown/drain reporting, initialization races, and unchanged
   shared executor reload/shutdown/fork behavior.
8. Service and consumer summaries, cookies, deduplication, ingestion and persistence
   through the canonical per-page plan with unchanged analyzer advice/results.
   Verify every outcome mapping, the HTTP failure statuses, partial-result and
   ephemeral-receipt round trips, persistent/ingestion receipt propagation, typed
   legacy partial-result failures, cancellation without new service side effects,
   and preservation of the primary outcome when checkpoint/summary/storage fails.
   Protect established pre-admission and mid-crawl structured failure lists, including
   articles followed by a transport-failure entry with no false completion callback.
   Race cancellation against terminal crawl publication and during summarization/
   persistence afterward; assert one immutable crawl outcome and a separately
   canceled service operation, with already committed data retained.

Property-based tests should exercise frontier normalization/deduplication/order and
budget/state-transition invariants. Deterministic tests separately exercise race
interleavings and each checkpoint boundary. Existing Web_Scraping, WebScraping,
Services, Security, preflight, article orchestration, resource-governance, and
affected-consumer suites remain regression evidence; do not replace them with only
new isolated tests.

Architecture and import-inventory tests prevent reverse dependencies and duplicate
execution engines. Run scoped lint/compile and Bandit for touched Python during
implementation, using the project virtual environment. Record failures/skips and
their reasons; do not disable tests or claim a broader certification than was run.
This documentation-only task requires documentation/link/task-format checks, not a
claim that the future Phase 5 behavior has passed these tests.

## Staging And Review Gates

The later implementation plan should divide this specification into four reviewable
stages, each using test-first changes and independently verified cutovers:

1. Characterize compatibility profiles and define internal execution/discovery
   contracts without moving behavior prematurely.
2. Establish process-local job lifecycle, cancellation, completion projections, and
   service authorization boundaries with deterministic queue tests.
3. Move discovery/traversal, parent accounting, and supported recursive continuation
   behind the injected crawl boundary with interruption/security coverage.
4. Switch adapters/consumers, refresh inventories/docs, and certify combined browser,
   analyzer, lifecycle, service, and downstream compatibility.

These are sequencing constraints, not an approved implementation plan. A smaller
stage may be split into reviewable tasks/PRs without changing this design. Runtime
cutovers must not execute both implementations or introduce user-visible duplicate
side effects. Preserve a clear rollback commit boundary for each switch; after a
versioned checkpoint is written, downgrade loading compatibility must be considered
before rolling back checkpoint code.

Written-spec/ADR review and explicit approval are complete. Next gates are review
of the [detailed implementation plan](../plans/2026-11-01-web-scraping-phase-5-crawl-process-local-jobs.md)
with named tests and concrete safety defaults, and user selection of execution
method. Do not implement under design approval alone.

## References And Review Evidence

- [Overall refactor design](2026-07-03-web-scraping-refactor-design.md), Phase 5.
- [Phase 4 design](2026-07-26-web-scraping-phase-4-extraction-article-orchestration-design.md).
- [Import inventory](../../Design/WebScraping_Refactor_Import_Inventory.md).
- Current recursive traversal/checkpoint and queue code in
  `Article_Extractor_Lib.py` and `enhanced_web_scraping.py`; public models in
  `contracts/requests.py` and `contracts/results.py`.
- Existing permit/drain ownership in `orchestration/executor.py` and its regression
  coverage; existing configuration/traversal/service tests under
  `tldw_Server_API/tests/Web_Scraping/`.
- Read-only conversational reviews used immutable revision
  `4c4f197f68481664c58d4553bbfbb45dae157e28`. This document was checked against the
  newer code baseline recorded above; implementation planning must recheck changed
  consumers and policy before assigning code moves.

## Written-Spec Self-Review

Self-review completed on 2026-10-04 for completeness, internal consistency, scope,
and ambiguity. It checked ownership/dependency direction, accounting, cancellation,
stop outcomes, storage trust, legacy migration, compatibility, and verification.
The review clarified credential reacquisition and same-process resume exclusion
while owned work drains. Placeholder checks and relative document-link checks passed.

Remaining tradeoffs are deliberate: process-local queued work is not durable,
checkpoint replay is possible after a crash, unknown legacy/crash accounting is
disclosed, and running extraction threads are drained rather than claimed preempted.
This self-review did not authorize implementation. Written-spec approval was
subsequently provided on 2026-11-01, after the amendments below.

### Review Amendments (2026-11-01)

The user approved addressing the written review's three findings: explicit public
outcomes through the service/ingestion boundary, aggregate queue admission bounds,
and replacement of identifiers that collided with newer work on `dev`.

Implementation choices made in this amendment for written review are immediate
overload rejection rather than admission waiting, an additive `crawl_outcome`
projection with separate storage dispositions, and typed interruption carrying
partial results only where an existing legacy failure contract cannot express an
incomplete traversal. Bounded internal producer pacing is not admission waiting.
These choices preserve successful return shapes and collected data while removing
ambiguous success reporting. They are design decisions, not implemented behavior.

The design task is now TASK-15282 and its proposed decision is ADR-085. Replacement
numbers were checked against refreshed `origin/dev` and every changed-file page
of all open PRs, including PRs with more than 100 files. The original dated task
history is retained under the new task identity. No unrelated task or ADR was
renumbered, and the design branch was not rebased as part of this amendment.

The approved follow-up review narrows three boundaries: established mid-crawl
structured failure lists stay compatible; batches use bounded producer windows
and own cleanup of accepted siblings; and terminal crawl outcomes are published
once before post-processing, so later service cancellation is represented
separately. These refinements add no backend migration or product implementation.
