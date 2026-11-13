# Moderation

The Moderation module provides configurable content policy checks, review
workflows, supervised policy evaluation, governance helpers, category taxonomy,
and family-wizard policy materialization. It is safety-sensitive and should stay
explicit about when text is blocked, warned, redacted, queued for review, or
allowed.

## Start Here

- Runtime checks: `moderation_service.py`.
- Supervised and governance policy helpers: `supervised_policy.py`,
  `governance_utils.py`, `governance_io.py`, and `conflict_resolution.py`.
- Reviews: `review_service.py` and `review_store.py`.
- Taxonomy and matching: `category_taxonomy.py` and `semantic_matcher.py`.
- API endpoint: `app/api/v1/endpoints/moderation.py`; family setup endpoint:
  `app/api/v1/endpoints/family_wizard.py`.
- Tests: moderation tests under `tests/unit/`, chat integration moderation tests,
  Guardian family-wizard tests, and
  `tests/AuthNZ_Unit/test_moderation_permissions_claims.py`.

## Responsibilities

- Load and apply moderation policy from config and runtime overrides.
- Evaluate literal, regex, category, and optional PII-style rules.
- Support supervised policies and governance schedules/chat-type scoping.
- Persist and process moderation review items.
- Materialize family-wizard settings into moderation/governance policy records.

## Module Map

- `moderation_service.py` evaluates text and applies block/warn/redact actions.
- `supervised_policy.py` composes policy decisions and review requirements.
- `review_service.py` and `review_store.py` own review lifecycle data.
- `category_taxonomy.py` defines category naming and metadata.
- `semantic_matcher.py` supports similarity-style policy matching.
- `family_wizard_materializer.py` converts family wizard input into policy data.

## How It Connects

- Chat and other text-generation endpoints call moderation before or after model
  output depending on route policy.
- Governance and Monitoring share policy schedule/chat-type utility functions.
- AuthNZ permission tests protect moderation endpoint claims.

## Architecture Notes

### Core Flow

- Endpoints and chat flows obtain the effective moderation policy, evaluate input or output text with `ModerationService`, then apply block, warn, redact, allow, or review outcomes.
- Literal, regex, managed-blocklist, category, and override rules are combined through explicit precedence so stricter actions win where rules overlap.
- Review capture flows pass sanitized event data through `review_service.py` into `review_store.py`.
- Family wizard materialization writes settings into moderation and governance policy records rather than special-casing family behavior in chat routes.

### State And Data

- Runtime policy can come from config, managed blocklists, user overrides, and family-wizard materialized records.
- Managed blocklist updates use version/ETag-style concurrency checks in the API path.
- Review storage owns review item lifecycle and idempotent event capture.
- Category taxonomy names and supervised policy schedules are shared with adjacent governance and monitoring helpers.

### Security And Operations

- Live service, supervised-policy and Workflow custom regex operations use a
  killable standard-library `re` child, never an inline fallback. Initial policy
  compiler/lint isolation is outside this contract. Pure evaluator unit APIs are
  not the live execution boundary. Normal `re` flags, raw patterns, literal base
  replacements and Workflow template replacements retain their distinct semantics.
- Do not log raw moderated content, blocklist secrets, or review payloads that may contain sensitive user text.
- Endpoint permissions are guarded by AuthNZ moderation claims; update permission tests when adding moderation routes.
- Block, warn, and redact behavior must stay deterministic so test endpoints and chat integration checks agree.

### Execution Limits

`ExecutionBounds` is frozen. Configuration may tighten, never widen, these
ceilings; nonfinite, nonpositive and boolean values are rejected:

| Limit | Default / Ceiling |
| --- | --- |
| Execution scope, including admission, serialization, child startup and all batches | 2 seconds |
| Active owned workers per API process | 4, immediate nonblocking admission |
| Input / output (aggregate across independent batch fields) | 1,000,000 characters each |
| Independent redaction fields per batch | 4,096, including empty fields |
| Rules | 256 |
| Each pattern / replacement | 4,096 characters each |
| Encoded policy object | 262,144 bytes (256 KiB) |
| Request / response frame, including four-byte prefix | 33,554,432 / 8,388,608 bytes |
| Linux child address space / CPU / core dump | 512 MiB / 3 seconds / 0 |

Character admission and ASCII-escaped JSON byte admission are independent;
non-ASCII content may hit a frame ceiling first. Existing evaluator scan/count
limits are separate and may examine less than the admitted input. A facade that
decides then redacts shares one deadline across both child batches.

Audio transcript/segment redaction uses one fresh `redact_many` child with a
single policy snapshot, not one interpreter per segment. Fields retain separate
anchors, boundaries, literal replacements and per-field evaluator limits; they
are never concatenated. Full transcript and checked segment duplicates count
toward the aggregate character/frame caps. Partial opt-outs are not submitted.
All fields succeed before any mapped payload is returned. Trusted `redact_text`
overrides retain dispatch under the inherited scope; the default path never
matches content in the parent. This bounded amendment is recorded in
`Docs/superpowers/specs/2026-11-03-moderation-audio-batch-amendment.md`.

The two-second deadline starts at execution-scope entry, before admission and
serialization. It is not an end-to-end HTTP or public-call wall-time guarantee:
first trusted module initialization before scope entry, failure reporting, and
kernel cleanup can add wall time. API lifecycle preparation loads the incident
reporter before traffic. This boundary follows the parent's recorded service
operation interpretation; it does not widen the deadline or any safety ceiling.
Direct embeddings needing a stricter end-to-end latency contract require separate
review. The release evidence retains the unpartitioned 2063.809 ms cold wall
sample and the later, separate 78.787 ms scoped diagnostic; the latter does not
explain the former.

Each API process owns its own four-worker allowance: N API processes can own
4*N children, with up to 512 MiB address space per Linux child, in addition to
parents, orchestration and reporting threads. This is not a deployment-wide
quota or a reservation of physical RAM. Admission stays closed while ownership
is quarantined; lifecycle shutdown cancels and kills/reaps owned children.

The trusted launch is `sys.executable -I -S <absolute regex_worker.py>`, with
fixed arguments, no shell, no content in argv/environment/diagnostics, bounded
stdin/stdout frames, and discarded child stderr. The wheel includes the worker,
its minimal stdlib-only graph and these operator notes. Keep the installed
interpreter and sibling files intact; missing artifacts fail closed.

Linux must install and verify address-space, CPU and no-core limits before
policy decoding; failure denies execution. macOS uses CPU/no-core only, with no
RLIMIT_AS guarantee. Windows has no stdlib resource backstops; parent deadlines,
frame/count caps and bounded ownership still apply. Windows runtime validation
is unavailable for this release evidence; source compilation is not an OS pass.

### Failure And Rollout

Operational failures are not user policy violations. HTTP returns safe 503
`moderation_unavailable`; streaming/Audio use the same operational code. The
seven typed internal reasons are `deadline_exceeded`, `capacity_exhausted`,
`size_exceeded`, `worker_unavailable`, `worker_crashed`, `invalid_response`, and
`cleanup_failed`. Cancellation remains cancellation. Failed buffered tails,
success markers, persistence and downstream continuation are withheld;
already-emitted safe stream bytes cannot be recalled.

Normal cancellation is quiet. If child/pipe cleanup cannot be confirmed within
its bounded window, quarantine retains worker/offload ownership and the cleanup
owner attempts sanitized `cleanup_failed` reporting independently of the original
failure or cancellation ordering. The shared report-once claim and independent
reporting slot prevent duplicate primary incidents without masking cancellation
or replacing the original error.

One metadata-only incident uses category `moderation_engine_failure`, severity
`high`, status `needs_review`, source `moderation_engine`, recommendation
`escalate`, and excerpt `[Moderation execution failed; content withheld]`.
Unknown phase produces one `unspecified` item, not input/output duplicates.
Rule and test request phases do not gain `unspecified`. Nested failures report
once per operation; incidents deduplicate atomically in five-minute UTC buckets.
Capture-disabled settings do not disable operational incidents.

Incident SQLite connections use a 250 ms busy wait, including initial WAL
acquisition, not a guarantee against arbitrary filesystem stalls. Reporting has
independent bounded admission. If it fails, enforcement remains blocked, with a
bounded-cardinality failure metric and rate-limited sanitized ERROR; no raw
content, policy, traceback or child diagnostic is a safe fallback log field.

Before activation, stop/quiesce incompatible old phase-restricted readers and
writers, back up the review database, and let lifecycle schema preparation
transactionally upgrade it before serving traffic. Incident writes validate a
prepared schema and never migrate on first failure. The migration retains IDs,
decisions, audit, undo leases, indexes, triggers and foreign keys. Roll out the
service activation and converted callers together, not as an intermediate build.

If pre-traffic preparation fails, API startup retains moderation shutdown
ownership and emits degraded reporting diagnostics rather than preventing
unrelated or disabled-moderation traffic. Execution failures still block affected
content; unavailable/unprepared incident storage never authorizes it and is not
migrated in the write path. Repair and prepare the store before relying on review
incident persistence.

Rollback must retain neutral records and compatible readers. Do not downgrade
to phase-restricted readers after neutral writes, relabel incidents as user
violations, drop rows or destructively reverse the schema. Quiesce traffic and
use a compatible build; any backup restoration needs its own explicit data-loss
decision. Release evidence is recorded in
`Docs/superpowers/reviews/moderation-backend/2026-11-02-killable-re-validation.md`
(plan identity; actual execution 2026-11-03).

### Extension Checklist

- New rule type: update `moderation_service.py`, schemas/endpoints, conflict behavior, and unit/chat integration tests.
- New review field: update `review_store.py`, `review_service.py`, endpoint schemas, and review capture tests.
- New family setup setting: update `family_wizard_materializer.py`, Guardian tests, and the moderation/governance policy mapping.

## Extension Points

- Add rule sources in `moderation_service.py` only after defining conflict and
  precedence behavior.
- Add review fields through `review_store.py`, schemas/endpoints, and review
  tests together.
- Keep governance helper functions pure so they can be reused by Monitoring and
  supervised policy code.

## Testing

- Runtime text checks: `tests/unit/test_moderation_check_text_snippet.py`,
  `tests/unit/test_moderation_redact_categories.py`, and
  `tests/Chat_NEW/integration/test_moderation.py`.
- Endpoint and permission coverage: `tests/AuthNZ_Unit/test_moderation_permissions_claims.py`
  and `tests/unit/test_moderation_test_endpoint_sample.py`.
- Family/governance behavior: `tests/Guardian/test_family_wizard_endpoints.py`,
  `tests/Guardian/test_family_wizard_materialization.py`, and related Guardian
  family-wizard tests. There is no direct
  `tests/Moderation/` directory in this tree; use the adjacent unit and chat
  integration moderation tests listed above.

## Gotchas

- Regex rules can create performance and false-positive risks. Keep scan limits,
  replacement limits, and tests for edge cases.
- Do not log raw moderated content when the failure path may include sensitive
  user input.


## Explicit model classification

In Content Rules, open **Model classification**, choose a configured llama.cpp model, enter one criterion and a sample, then choose **Classify and save for review**. The sample and result are saved to the staff Review queue. This requires Admin and system.configure authority; Review decisions retain their own permissions.

Both benign and flagged results await human review. Flagged evidence must quote the submitted sample exactly. The model cannot change rules or make a final decision, and no confidence score is fabricated. Deterministic testing and Chat enforcement are unchanged. This version supports the audited native async llama.cpp transport; other providers are explicitly refused.

A call has one attempt, a 120-second deadline and bounded input/output. Errors, refusals and incomplete or ungrounded results never count as benign. Stop cancels the owned request, but a result saved just before disconnection can remain. **Retry saved request** recovers that receipt before another inference; editing the form starts a new receipt. Concurrent duplicate submissions can issue multiple calls but create one Review item; there is no cross-worker exactly-once inference promise.

POST /api/v1/moderation/classify accepts request_id(UUID), provider, model, criterion(max 500 characters) and text(max 4000 characters). The response is the saved Review item. Server-authored context.classification contains provider/model, criterion, label, reason and evidence. Reusing an actor's request ID with changed or redacted inputs returns 409. Request admission reuses chat.default; middleware enforces requests, while LLM quotas and best-effort accounting remain dispatcher-owned.
