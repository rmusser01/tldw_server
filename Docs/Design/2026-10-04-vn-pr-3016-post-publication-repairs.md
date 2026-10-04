# PR3016 Post-Publication Repair Design

Association: original TASK13369; plan: IMPLEMENTATION_PLAN_vn_pr_3016_review.md.
The requester authorized the presented repairs on2026-10-04 with all approvals.
This design supplements the completed VN durability design, not a new generation,
quota, Jobs lease or receipt protocol.

## Qodo Repairs

Move inline legacy-display begin and claim release through the existing drained
owning-thread repository boundary. A cancelled setup drains the operation and
cleans its acquired marker/claim before propagating cancellation. Keep attempt
fencing, Jobs authority and legacy overlap semantics; use real blocking barriers
to prove event-loop responsiveness and cancellation cleanup.

The reload smoke test observes the current server/principal-scoped command
journal. Do not reintroduce a duplicate legacy receipt writer. Failed-job retry
tests own their SQLite/official isolated-PG connections instead of relying on
private JobManager helpers; persisted admission and rollback assertions remain.

## Quota Repairs

`QuotaExceededError` takes two arguments. Correct only the inherited combined
quota call's arity; preserve typed denial and all admission/accounting decisions.
The preexisting strict VN matrix stays byte-exact and must not accept arbitrary
transaction failures as quota denial.

Audio profile caps now come from persisted user/team/org limits, not an implicit
30-minute tier. Set the test user's30-minute override with the existing repository,
invalidate only that user's resolver cache, explicitly opt in to enforcement and
retain DATE and numeric expectations. Verify disabled/no-limit controls through
the official isolated-PG fixture; unavailable fixtures are skips, not PG passes.

## Redis Rollout Restriction

New RG writers refresh a shared window key's `ceil(window)+5` expiry; old writers
do not. If an old writer adds late to a new writer's expiring key, the whole key
can disappear while that old charge is still live. Changing script hashes does
not protect the shared key; changing namespaces would split admission counters.

The approved disposition is a homogeneous, quiesced cutover and rollback. Stop
all API/worker/background RG writers, freeze policies, wait out the largest
configured live request/token window plus the five-second expiry margin, retain
the same Redis namespace/state, then start only the chosen version together.
Do not resume traffic while mixed versions can write. Deployment documentation
records this restriction explicitly. No production TTL/accounting edit, key
deletion, real deployment or mixed-version compatibility claim is made.

## CI And Merge

Normalize only the five exact approved task paths with the official formatter,
preserving frontmatter, prose, IDs/status and all checklist state. Remaining
privilege/email/deadline failures need verified runtime causes before minimal
repairs; neither snapshot regeneration nor assertion weakening is a substitute.

Every code repair receives bounded TDD, touched-scope Bandit and independent
SPEC/QUALITY review. Normal hooks, full new exact-head Qodo, all seven required
contexts, strict current dev/rules and the verbatim human Change summary govern
normal merge. No old-head result, unsupported-runtime suite or unavailable-PG
skip certifies those external gates. PR3067 remains untouched.
