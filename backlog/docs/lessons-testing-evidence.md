# Testing Evidence Lessons

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
