# PR2761 capacity diagnostic preparation and stopped build

TASK-13013.9 remains **In Progress**. This attempt produced no capacity result:
the image export was canceled before a service started, so there were **zero
workload requests, zero telemetry samples and no steady/overload/recovery run**.
The private collector and bootstrap preparation did not run end to end and is
not shipped as repository tooling. Neither its runtime behavior nor the proposed
envelope has been validated.

## Exact build attempt

- Source: `910c526513a03b305b2976fec6f1dee73bacc1e9`, extracted with `git archive`
  into an exclusive temporary directory; exact `Dockerfiles/Dockerfile.prod`.
- Platform: `linux/amd64`; base image resolved to
  `python@sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea`.
- Build reference: `popb05905i3kinboaodahksb9`; started 2026-09-10 19:40:11 UTC,
  canceled 20:12:57 UTC, process exit 130.
- Dependency installation and the runtime `import mcp_unified; import
  tldw_profile_core` build step passed. The final image export did not finish;
  there is no resulting image ID, runtime source verification, or image
  `pip freeze` result from this attempt.
- The initial attempt began before archive extraction completed and failed on
  missing build-context input. It was retried only after extraction completed.
  The later cancellation was a separate host-resource stop, not an application
  test failure or a completed image-build success.

The [build status](PR2761-capacity-diagnostic/build-status.json),
[stage excerpt](PR2761-capacity-diagnostic/build-stage-excerpt.txt),
[input hashes](PR2761-capacity-diagnostic/inputs.json), and
[285 resolved builder packages](PR2761-capacity-diagnostic/resolved-build-packages.txt)
retain the distinction. Package versions came from successful builder
installation output; they are not a lockfile or an exported-image attestation.

## Reviewed pre-traffic profile

The [preset](PR2761-capacity-diagnostic/preset.json) and
[dataset](PR2761-capacity-diagnostic/dataset.json) were fixed before any traffic.
The profile was changed from single-user to **multi-user with one disposable
admin actor** before execution because production AuthNZ selects PostgreSQL
only in multi-user mode. No `TEST_MODE` override was used.

Proposed topology: PostgreSQL AuthNZ, SQLite content/workflows, one API worker
limited to 2 CPUs and 4 GiB, in-memory request governance, production mode
disabled, offline model downloads, and an isolated Redis instance. This is a
diagnostic profile, not the production multiworker deployment. The host was a
shared ARM Docker Desktop VM with 18 vCPUs and about 15.6 GiB RAM; the amd64
image would run under emulation. See [hardware](PR2761-capacity-diagnostic/hardware.json).

The proposed workload alternates authenticated current-user reads with new
synchronous prompt-rendering workflows, asserting `succeeded` and the exact
sentinel output. It invokes no external model. Preflight would require two
distinct completed workflow IDs. These requests were **not executed**.

| Phase | Duration | Concurrency | Pause per worker | Minimum successes per workload | p95 ceiling |
| --- | ---: | ---: | ---: | ---: | ---: |
| Steady | 180 s | 2 | 2 s | 30 | 2 s |
| Overload | 60 s | 32 | 0.02 s | 1 | 5 s |
| Recovery | 120 s | 1 | 2 s | 15 | 2 s |

The preset also preserves explicit rejection/error ratios, storage growth,
resource ceilings and recovery time. No threshold was tuned against a result.
The intended collector measures actual AuthNZ pool occupancy from authenticated
health diagnostics (including its probe connection), WorkflowScheduler queue
depth, and apparent bytes in the owned data/Redis volumes plus the dedicated
PostgreSQL database size. It refuses unavailable metrics, checks the running
image identity, and timestamps the start of collection. This scope does not
measure every database pool or queue, CPU saturation, ingestion, external model
latency, or long-term retention growth.

## Resource stop and cleanup

Shared-host free disk fell from roughly 80 GiB during preparation to about
12 GiB during export, with other unrelated builds running concurrently. The
capacity build was stopped to avoid exhausting the host. Aggregate disk loss
cannot be attributed solely to this build.

Eleven uniquely identified, unshared runtime cache records were rechecked and
removed individually with exact-ID filters. Docker reported approximately
8.56 GB reclaimed; the [cleanup record](PR2761-capacity-diagnostic/targeted-cache-cleanup.json)
contains every ID and result. The builder cache record
`e7c63pc99fswpg1aaps1wp3kd` (reported 8.311 GB) was marked shared and was retained.
No global prune, unrelated image deletion or interruption of another build was
performed. APFS still reported about 12 GiB free immediately afterward, so
physical host-space recovery was **not confirmed** by logical cache reclamation.
During final evidence checks, free space had fallen further to approximately
7 GiB while this capacity build remained stopped; the other resource owners
were informed. Shared cache and unrelated builds remained untouched.

The repository's `tests._plugins.postgres._temp_db_generator` provisioned the
exclusive database and completed cleanup with exit 0. A subsequent database
catalog query verified that database was absent. Docker inspection verified
that no capacity containers, volumes or network existed; deployment never
reached their creation. The exclusive 466 MiB source context and generated
private credentials were removed after input hashes and small preparation records
were retained.

## Follow-up environment and retained preparation

Do not repeat the full build on the same space-constrained shared host. Use a
native amd64 runner with separately budgeted storage for the builder, runtime,
export/unpack scratch and workload data, and serialize this image build. The
observed builder/runtime application layers alone were about 16.5 GB before
base images and export overhead. Recheck free disk before proceeding.

The earlier `d5ba8b5be7a9b74e8b7f74a63b2b7fccd6079135`
[CI app build](https://github.com/rmusser01/tldw_server/actions/runs/34521018978/job/103018263083)
passed. Its retained artifacts are small `.dockerbuild` records; the workflow
uses `push: false` and loads images only into the ephemeral CI runner. Metadata
inspection found no downloadable runnable app image in that run. A later run
can execute the diagnostic in the same job after building, or explicitly
export/publish the selected artifact first. No large artifact was pulled here.

The preset and dataset preserve the proposed experiment. Use the existing
tested `Helper_Scripts/load_tests/release_soak.py` runner and its
[collector contract](../Development/Release_Capacity_Soak.md) after an actual
deployment and collector have been validated on the selected runner. Supply the
verified image/source identity and same-origin collector URL in a complete
profile, keep authentication outside evidence, run preflight terminal-workflow
checks, then execute the preset phases without adjusting thresholds against
failures. Provision PostgreSQL through the existing repository fixture workflow
and verify cleanup of all exclusive resources afterward.

The three unexecuted preparation scripts remain in the private temporary
directory only. Their hashes are retained for traceability. Review identified
that the drafted cleanup routine could stop early when an expected container
was absent. Its drafted runtime source comparison also included all 8,631 API
source files, including tests excluded by `.dockerignore`, and would reject a
correct image. Both are preparation defects requiring correction, not candidate
runtime failures. This preparation must not be treated as a validated replay
utility. Compilation and preliminary Bandit checks do not establish lifecycle
correctness. No new executable utility is included in this evidence package.

TASK-13013.9's measured reference envelope, ingestion evidence, completed live
workflow evidence and overload/recovery acceptance remain open. Any later
candidate source or artifact requires its own immutable identity and fresh run.
