# PostgreSQL Media Bootstrap Sequence Safety

Tracking: TASK-13372.

The synthetic archive API probe stored one of 100 messages. PostgreSQL reported
99 duplicate `media_pkey` errors. Each archive child creates a Media handle;
routine post-core bootstrap calls sequence synchronization before the worker's
user scope is installed. Forced RLS makes `MAX(id)` appear empty, and sequence
synchronization resets the next ID to 1.

Remove sequence maintenance from routine post-core structure checks. Fresh
tables already have initialized serial sequences; ordinary handles must not
rewrite global allocator state from a user's visible subset. Retain the explicit
v18 migration maintenance path and its regression. This does not redesign
offline sequence repair or skip schema migrations and security policy checks.

Validate the routine-bootstrap contract red first, then repeat the actual
non-superuser archive upload with forced RLS, all 300 IDs, retry identity and
cross-user isolation assertions. Keep the throughput gate open if correctness
passes but measured upload speed is below 50 messages/sec.

Validation: the unit boundary regression and real PostgreSQL repeated-handle
test failed before the change. Afterward, 75 schema unit tests and three real
PostgreSQL sequence/FTS tests passed. The full authenticated archive probe
stored all 300 messages, preserved their IDs on a 100-message rerun, and
confirmed owner rows 300 versus other-user rows 0 under forced RLS with a
non-superuser/non-bypass role. Model and outbound guards observed zero attempts.
The measured 5.03 messages/sec remains below the throughput target; TASK-13371
records that separate performance gap.
