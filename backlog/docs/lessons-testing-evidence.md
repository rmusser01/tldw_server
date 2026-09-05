# Testing Evidence Lessons

## A record conflict is not evidence for every Personal Context domain

**Incident (TASK-13163, 2026-09-05):** A 316-test targeted checkpoint passed
while the new conflict choices exercised records. Review found that linked
client manifests could enter the same conflict path, although the approved
contract forbids pushing those derivative checkpoints. An unrelated semantic
write advanced the server manifest and stranded its immutable conflict review.
New stale/current client-manifest rejection tests both failed before the fix.

**Evidence and rule:** Check each domain's authority and lifecycle before applying
a shared conflict handler. Cover both permitted semantic inputs and prohibited
derived/control inputs; a broad existing regression count does not establish
that the new behavior is valid for every domain.

## Exercise ordinary ingress after narrow storage repairs

**Incident (TASK-13192, 2026-09-05):** The activation repair fixture seeded
PostgreSQL envelopes directly because ordinary insertion failed elsewhere.
Replacing that seed with `SyncV2Store.insert_envelope` reproduced the failure in
the shared SQL converter: `ELSE ? END` retained a question mark, leaving five
bind slots for six parameters. SQLite passed the same insertion.

**Evidence and rule:** Directional CASE keyword recognition restored ordinary
ingress and all receipt-state tests on both backends. Parser regressions also
retain JSONB operators after `CASE ... END` and before a CASE operand. A narrow
storage test may isolate a separate defect temporarily, but remove its bypass
when fixing that defect and prove the public insertion path on each backend.

## Generic dict lint rules do not apply to `sqlite3.Row`

**Incident (TASK-13144, 2026-08-30):** Ruff's `SIM118` suggestion replaced
`for key in row.keys()` with `for key in row` in `PersonalizationDB`. The new
Personal Context tests stayed green, but the combined existing Personalization
regression run produced eight `IndexError` failures because iterating a
`sqlite3.Row` yields values rather than column names.

**Evidence and rule:** Restoring `.keys()` with a narrow lint exemption returned
the same 53-test combined run to green. Do not apply dict-iteration
simplifications to `sqlite3.Row`; when a shared database wrapper changes, pair
new-feature tests with its existing consumer suite.

## Lifecycle fences must be proven inside the write transaction

**Incident (TASK-13145, 2026-08-30):** The Personal Context service checked the
purge-pending state before proposal and runtime writes, and its sequential tests
passed. Independent review showed a purge could commit after that check but
before either standalone repository transaction, allowing the later write to
recreate encrypted state beyond the purge barrier.

**Evidence and rule:** Passing the expected manifest version into each write and
rechecking both the manifest head and surviving scope state under the same
`BEGIN IMMEDIATE` transaction closed the race; targeted purge-race regressions
then passed. For destructive lifecycle barriers, a service precheck is UX only:
the storage transaction must enforce the same fence.

## Exercise canonical factories, not only service fakes

**Incident (TASK-13148, 2026-08-30):** Personal Context bootstrap unit tests
passed with service fakes, but the first authenticated endpoint test using the
real factory failed with `personal_context_snapshot_unavailable`. The canonical
profile clock emitted arbitrary microseconds while canonical serialization
permits millisecond precision only, so a fresh profile could not be snapshotted.

**Evidence and rule:** Normalizing the service clock at the canonical mutation
boundary made the real authenticated bootstrap, stale-completion, receipt, and
post-link push flow pass. When a feature composes storage, canonical models, and
HTTP dependencies, include at least one test through the production factory;
service fakes cannot prove cross-layer serialization contracts.

## After-commit callbacks may precede outer projection terminalization

**Incident (TASK-13161, 2026-09-03):** The real authenticated factory test pushed
a new Personal Context record successfully, but the canonical after-commit relay
then poisoned its authority batch. Unit relay tests were green. The callback ran
after the canonical database commit but before the enclosing Sync materializer
marked the identical client-ingress envelope applied; the envelope also relied on
projected object state because its wire row omitted `object_revision`.

**Evidence and rule:** A production-factory push followed by a real pull reproduced
the failure. Restricting self-head confirmation to an exact authenticated
client-ingress match in either legal pending/applied state, and deriving its base
revision/hash from current projected state, made the same after-commit relay and
exact-once pull pass. For cross-store callbacks, test the actual outer transaction
ordering and use the store's projected head facts rather than assuming envelope
terminal state or optional wire revision fields are already complete.

**Follow-up (TASK-13170, 2026-09-04):** The original production-factory endpoint
later proved that deriving the outgoing base was insufficient: receipt confirmation
still compared authority lineage to the immutable envelope's absent wire revision,
so repeated polls remained permanently pending. A real two-store test that asserted
raw revision `None`, projected revision 1, and both semantic and manifest successors
at lineage 1→2 caught the gap.

**Correction after review (TASK-13170, 2026-09-04):** Making latest projected state
the missing revision's authority proof then created an unbudgeted persistent read and
permanently rejected a lagging companion after a later legitimate materialization
moved projection forward. The durable fix derives revision only from the immutable
ingress result lineage: an absent revision with no base is genesis 1; an absent
revision with a complete strict base is `base_revision + 1`. Projection may verify
that predecessor before the ingress receipt and may advance transactionally after
authenticated authority finalize, but it must not become historical receipt proof.

## Stable basetemp evidence must respect production path policy

**Incident (TASK-13172, 2026-09-04):** The first combined 14-file certification
run forced `--basetemp=/tmp/tldw-task-13172-remediation-matrix`. Hundreds of
otherwise unrelated cases failed or errored because legacy harnesses construct
`PersonalizationDB.for_path(tmp_path / "personalization.db")`, and the production
trusted-root guard correctly rejects `/private/tmp`. The same first seven gates
passed 214/214 when rerun under pytest's trusted default temporary root, and the
remaining groups passed 239/239 and 300/300.

**Evidence and rule:** Use a stable exact basetemp only for fixtures that place
their application databases under an explicitly configured trusted root. For a
mixed legacy matrix, preserve the production path guard and record conclusive
group results under pytest's trusted temp root; never weaken storage validation
to make evidence paths prettier.

## Security report formatters can fail on deliberate invalid-Unicode fixtures

**Incident (TASK-13173, 2026-09-05):** Bandit's text and JSON report writers both
raised `UnicodeEncodeError` on an existing lone-surrogate continuity-token fixture.
The scanner itself had completed; changing output formats did not fix reporting.

**Evidence and rule:** Serializing the scanner's issue fields with escaped Unicode
allowed an in-memory comparison against HEAD: no new findings, only the two
reviewed subprocess warnings removed, and 15 unchanged fixture findings. Preserve
invalid-input coverage and compare scan results; do not remove the negative test
or add broad suppressions merely to make a report formatter succeed.

## 2026-09-05 — Buddy UAT exposed transport and lifecycle gaps hidden by test doubles

TASK-13182's setup smoke test ignored `expected_version`, so it passed while a real defaults save followed by checkpoint advancement returned 409. Enforcing optimistic version checks in the test reproduced the failure. TASK-13184's direct Python WebSocket probe accepted a server handshake without a selected subprotocol, while Chromium rejected the same handshake before sending any message. Require the browser's protocol contract in the auth regression and verify the real browser boundary. During review, discarded Strict Mode loads and A→B→A persona switches also exposed stale-completion cases that a single switch/remount test did not cover.


## 2026-09-05 — Match CI runtime and await usable editor state

During PR #2884 / TASK-13176, VisualPackEditor tests passed64/64 on local Node26 but failed4/64 on Node20 with the workflow's deterministic shared-UI config. Pack metadata appeared before manifest-derived select values, and a recorded candidate-review request preceded completion of the disabled-button state. The shard also exposed a custom-state selection before its option existed. Waiting for the actual manifest values, available option, and accepted/rejected outcome preserved the assertions and passed93 tests in the six-file CI context on Node20. Reproduce both the CI runtime and package config; a selected pack ID or recorded request is not proof that downstream controls are ready. Do not replace these readiness checks with larger timeouts.

## Cache size is not proof of cache reuse

**Incident (TASK-13187/TASK-13188, 2026-09-04):** Manual llama.cpp snapshot
tests initially modeled a slot token count as `n_past`. Review against pinned
upstream source found that slot JSON uses `n_prompt_tokens` only after a task;
fresh idle slots omit it. The live-harness investigation also found that
completion `tokens_cached` describes the final slot size, not tokens reused.

**Evidence and rule:** Source-derived fresh/busy/malformed-slot regressions
passed after correcting the parser. At upstream commit
`4d9176092d00586775af140581bb0b558ddc4389`, `server-common.cpp:67–71`
serializes reused tokens as `timings.cache_n` and newly processed tokens as
`timings.prompt_n`. The harness now requires those counters and a separate cold
process control. No binary/model was available, so the live test remains skipped
and the production build allowlist remains empty. Pin wire fixtures to their
source, label source-derived versus live evidence, and never infer reuse from
file existence, HTTP success, similar output, or final cache size.

## Snapshot reuse evidence is configuration-specific

**Incident (TASK-13188, 2026-09-04):** The supplied Gemma sliding-window model
saved/restored 2770 tokens on llama.cpp b10816 but reused zero after restart.
Native diagnostics reported missing cache coverage; a one-variable `--swa-full`
test changed reuse to 2770 tokens with only 10 processed. The managed
runner/store/coordinator subsequently reproduced 2770/10 versus a cold control
of 0/2780. At context 16384 the native SWA allocation grew 300 to 3200 MiB.

**Evidence and rule:** Treat cache mode as compatibility identity and explicit
operator configuration, with memory/restart guidance. Do not infer another
architecture's support from a model-family label or one successful configuration.
Publish warm/cold measurements before reuse assertions so failures retain useful
evidence. Managed runtime evidence does not substitute for live client acceptance.

## Activation fixtures must distinguish authorization from publication state

**Incident (TASK-13162, 2026-09-05):** Adding canonical activation checks exposed
older relay fixtures that granted access by writing only Sync metadata. Simply
activating those fixtures would cover the pending source rows their recovery
tests were meant to exercise.

**Evidence and rule:** Production certification now performs real baseline
installation and acknowledgement before creating relay debt. Narrow authority
and budget tests use an explicit typed authorization double while preserving
their real source rows and original assertions. Keep those concerns separate;
never add a production metadata-only fallback to satisfy a test harness. Compare
broader failures against unchanged dev before attributing them to the new guard.

**Follow-up (TASK-13162):** Updating those proof fixtures exposed a deeper failure:
guarded repair verified the canonical ingress receipt, but terminalization accepted
only pending/applied rows, so a previously failed row could never unblock bootstrap.
The original recovery assertion caught it. A focused regression failed on both
SQLite and PostgreSQL before the one-line transition fix; all 22 receipt/state
cases and 48 bootstrap tests then passed. Keep the recovery assertion through
fixture repairs, and test retryable failure states as well as first application.
## 2026-09-05 — Verify session handoff under StrictMode and superseded attempts

TASK-13180 real browser UAT resumed the exact Buddy session with HTTP 200, but
full Live never requested its detail or opened a WebSocket. Effect cleanup set
`mountedRef` false while StrictMode setup replay never restored it; the ordinary
route test passed. A StrictMode route regression reproduced the missing socket,
and restoring the mounted flag during effect setup made both paths pass.
Independent review also found that an older connection's catch/finally could clear
a newer attempt's loading state. Resolve/reject regressions now keep the newer
request pending and prove another Connect click cannot issue a duplicate request.
Test effect replay and overlapping completions, including failure/finally paths,
when mounted/attempt refs guard asynchronous connection state.

## Check final task-file newlines after Backlog serialization

PR2902's pre-commit job rejected two task records after Backlog MCP updates
removed their terminal newline. `git diff --check` had passed, because that check
does not require a final newline. Normalize final newlines after the last task
edit, then include those bytes in pre-commit verification; repeated MCP writes
can otherwise reintroduce the same formatting-only CI failure.

## First-run name checks must include tombstones

**Incident (TASK-13196, 2026-09-05):** pixel-migu seed replay tests preserved
its own deletion receipt, but independent review found that a preexisting deleted
user card with the same name had no receipt. The ordinary name lookup excluded
that tombstone while SQLite's unique name constraint retained it, so canonical
DB initialization failed on every retry. A real factory regression reproduced
the conflict; an explicit include-deleted lookup preserved the user's tombstone
and restored startup. Test both deletion after seeding and deleted user content
that predates the seed.
