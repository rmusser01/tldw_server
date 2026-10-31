# Archive-local email database worker reuse

TASK-13373. Continue authorized SQLite then PostgreSQL throughput work using synthetic data only.

The 300-message worker profiles show SQLite connection configuration consumes 1.66 of 3.26 seconds; PostgreSQL database construction/schema bootstrap consumes 25.75 of 37.97 seconds. Reuse one database handle for each archive on one dedicated worker thread. Capture the request context for every submitted operation, construct lazily, and close on that thread on exit. Never share handles between archives or requests. Keep existing per-message legacy and normalized graph transactions, retries, and nonfatal native errors. Do not introduce a batch transaction, global schema cache, or model calls.

The archive loop submits one child at a time, allowing existing chunk verification and per-child failures to remain unchanged. Cancellation cleanup queues close behind any in-flight write and shuts down the executor. Test thread affinity, context propagation/reset, independent rollback, retry identity, native-write failure, and concurrent request separation. Authenticated guarded HTTP probes validate 300 messages, retry IDs, other-user denial, and PostgreSQL forced RLS. Record timing as small-batch evidence, not a sustained 50 messages/sec or million-message claim.
