# Media Ingest Worker Startup And Capability Design

Date: 2026-07-09
Task: TASK-12100

## Problem

YouTube quick ingest can submit a media ingest job that remains `queued` at 0% because the normal in-process media ingest worker is not started by default in the declarative lifecycle path.

The enqueue path creates jobs with `domain="media_ingest"`, the selected queue, and `job_type="media_ingest_item"`. The worker can process those jobs, but `media_ingest_jobs_task` currently uses a lifecycle predicate that requires `MEDIA_INGEST_JOBS_WORKER_ENABLED` to be explicitly truthy. That contradicts the operations docs, which say the default follows the `media` route policy.

A second issue is capability reporting: `hasMediaIngestWorker` is derived from the heavy-worker flag and route instead of the normal media ingest worker policy.

## Goals

- Start the normal in-process media ingest worker by default whenever the `media` route is enabled.
- Preserve explicit opt-out with `MEDIA_INGEST_JOBS_WORKER_ENABLED=false`.
- Preserve sidecar mode: `TLDW_WORKERS_SIDECAR_MODE=true` must prevent in-process worker startup.
- Keep the heavy worker opt-in unless its route or flag enables it.
- Report `hasMediaIngestWorker` from the normal in-process worker policy.
- Reconcile docs so route-default behavior is clear.

## Non-Goals

- Do not change job enqueue shape, queue names, or worker handler behavior.
- Do not change `route_enabled_predicate` globally.
- Do not prove whether an external sidecar worker is running from the API process.
- Do not add new UI states in this slice.

## Design

Extend the existing worker startup policy helper with an optional injected route checker, then add a lifecycle worker predicate that delegates to it:

`should_start_inprocess_worker(flag_key, route_key, sidecar_mode=context.sidecar_mode, default_stable=..., test_mode=context.test_mode, route_enabled=context.route_enabled)`

When no route checker is passed, the helper keeps its current behavior of consulting the global config route policy. This preserves existing callers while letting declarative lifecycle specs respect the `WorkerLifecycleContext` they are already given.

Injected route checker failures fail closed for in-process worker startup. The existing default-stable fallback remains only for the global config route lookup path.

Use this predicate only for the media ingest lifecycle specs:

- `media_ingest_jobs_task`: `MEDIA_INGEST_JOBS_WORKER_ENABLED`, route `media`, `default_stable=True`.
- `media_ingest_heavy_jobs_task`: `MEDIA_INGEST_HEAVY_JOBS_WORKER_ENABLED`, route `media-ingest-heavy-jobs`, `default_stable=False`.

This keeps current global route-backed workers unchanged while restoring the documented media ingest behavior.

Update `config_info.py` so `hasMediaIngestWorker` uses the same normal in-process worker policy with sidecar awareness. In sidecar mode this should report `false` for the API process, because the API cannot verify an external worker is actually running.

Update docs that currently describe media ingest jobs as opt-in or default-false, including both source and published copies:

- `Docs/API-related/Media_Ingest_Jobs_API.md`
- `Docs/Published/API-related/Media_Ingest_Jobs_API.md`
- `Docs/Deployment/Long_Term_Admin_Guide.md`
- `Docs/Published/Deployment/Long_Term_Admin_Guide.md`

The corrected wording should say the normal worker follows the `media` route policy by default and can be disabled explicitly or skipped in sidecar mode.

## Behavior Matrix

| Case | Expected normal worker result |
| --- | --- |
| `media` route enabled, flag unset, sidecar off | enabled |
| `media` route enabled, `MEDIA_INGEST_JOBS_WORKER_ENABLED=false` | disabled |
| `media` route disabled, flag unset | disabled |
| `MEDIA_INGEST_JOBS_WORKER_ENABLED=true` | enabled, unless sidecar mode is on |
| `TLDW_WORKERS_SIDECAR_MODE=true` | disabled in this API process |

The heavy worker remains disabled by default because its route uses `default_stable=False`.

## Tests

- Unit test the actual `media_ingest_jobs_task` spec predicate for route-default enablement.
- Unit test the actual `media_ingest_jobs_task` spec predicate for route-disabled behavior through the lifecycle context.
- Unit test explicit false disables the normal worker.
- Unit test sidecar mode disables the normal worker.
- Unit test the heavy worker remains off by default.
- Unit test the startup policy helper still works without an injected route checker.
- Unit test one-argument route callbacks and broken injected route callbacks.
- Unit test `hasMediaIngestWorker` uses the normal worker flag and honors sidecar mode.

## Verification

After implementation:

- Run the focused backend tests covering startup policy and config capabilities.
- Run a backend startup smoke check with `MEDIA_INGEST_JOBS_WORKER_ENABLED` unset and confirm `media_ingest_jobs_task` starts when the `media` route is enabled.
- Re-run the YouTube quick-ingest walkthrough far enough to confirm the job leaves `queued` and reaches worker processing or a real ingestion failure.

## Risks

- Local installs may now start the normal ingest worker automatically where they previously did not. This is intended for option 1 and matches the operations docs.
- Multi-worker SQLite deployments remain risky. Existing deployment guidance should continue recommending sidecar workers or Postgres for higher concurrency.
- `hasMediaIngestWorker=false` in sidecar mode does not prove no sidecar exists; it only means this API process did not start one.
