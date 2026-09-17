# StudyPack queued-job startup diagnosis

## Conclusion

The missing native worker is a **product default-enable regression in the declarative startup extraction**, exposed by the clean profile. It is not an intentional harness worker-disable flag, provider/model failure, or204 owner failure. No worker has claimed the retained job, so this observation does not exercise generation or persistence.

Current main enters `lifespan_startup_sequence` → `initialize_startup_worker_bootstrap` → `collect_startup_worker_specs` → `provide_study_privilege_jobs_worker_specs` → `LifecycleWorkerEngine`. The StudyPack spec calls `route_enabled_predicate("STUDY_PACK_JOBS_WORKER_ENABLED", "flashcards")`. That predicate requires a truthy explicit flag AND an enabled route. The runtime30793 profile's saved environment and exclusive dotenv omit the StudyPack flag, queue override, sidecar flag and TEST_MODE. Unset flag therefore disables this spec. The current startup log has no StudyPack worker start line; the engine records disabled inventory without the legacy helper's disabled log.

`buildBackendEnv`, `buildLiveTierBackendEnv` and the private launcher do not set StudyPack false. The launcher uses an explicit safe host-env allowlist, so it also does not accidentally inherit an operator's worker opt-in. This is the clean-env behavior, not a scripted negative control. The launcher still uses the profile's isolated Jobs path; no queue/DB mutation was performed.

## Intended contract and extraction evidence

Exact source before commit8f9bae72da is retained in `pre-declarative-startup_worker_groups.py`. Its `_should_start_worker` returnsFalse for sidecar, otherwise `_env_flag(flag_key, _route_default(route_key))`; route defaults are disabled in test mode. It passes that callback to the StudyPack startup helper. Thus unset flag + enabled flashcards route started the in-process StudyPack worker before extraction.

The April29 worker lifecycle design's Startup Flow explicitly says to evaluate the same environment flags and route defaults. The May3 deprecated-code-removal specification excludes changing worker enablement flags/default behavior. The April2 StudyPack plan says startup is behind an env-guarded flag, but does not require default-off; that wording does not override the actual historical route-default implementation.

Current comparable design: `startup_content_jobs_pollers.media_ingest_worker_predicate` delegates to existing `should_start_inprocess_worker`, preserving route defaults, explicit opt-out, test mode and sidecar semantics. The July9 media worker capability design also documents that route-default approach. A blanket change to generic `route_enabled_predicate` would affect unrelated workers and is not proposed.

## No-I/O causal check

`predicate_probe.py` invokes the actual StudyPack spec predicate and the existing canonical worker policy; it starts no app, worker, task, provider or DB. `predicate-results.json` contains all results:

- Unset or blank flag, route enabled, normal mode: currentFalse; canonicalTrue.
- Explicittrue, route enabled, normal mode: bothTrue.
- Explicitfalse: bothFalse.
- Unset flag, route disabled: bothFalse.
- Explicittrue, route enabled, sidecar mode: currentTrue; canonicalFalse.

This last control is an additional relevant policy-parity gap, not a claim of native sidecar execution. The legacy source independently proves the expected default/sidecar contract. Source hashes and snapshots are retained.

## Native receipt

`native-events-safe.json` retains only job/status metadata from the supplied capture: POST202 at05:54:01.529UTC creates job2, domainstudy_packs, queuedefault, typestudy_pack_generate, statusqueued. Eleven subsequent GET200 receipts through05:54:16.868 remainqueued, with no result or error. This capture alone proves15.339seconds of unchanged queue status; the parent's longer observation is separate. No session headers, note content, provider credentials or whole configuration was copied into the safe receipt.

The producer uses the same `study_pack_jobs_queue()` as the worker, defaulting to `default`. It creates a valid durable job and does not check worker availability. Source inspection also corroborates the drawer issue: submission is blocked only by mutation pending, and spinner follows individual poll `isFetching`; it does not stay busy or show a queued phase across polling gaps. That UI affordance should remain a separately scoped repair rather than being claimed solved by worker startup.

## Recommended repair and native next action

1. For the product default, narrowly have the StudyPack spec use the existing `should_start_inprocess_worker` policy with its real route/test/sidecar context. Preserve explicitfalse and sidecar suppression. Test actual spec/engine registration with unset/blank/true/false, route disabled, test mode, sidecar and teardown; no generic flag sweep. Startup availability rejection is not the primary repair: durable Jobs can validly wait for a configured sidecar, so absence of a local in-process task alone is insufficient evidence that enqueue is invalid.
2. For bounded native acceptance before that repair, the parent may set `STUDY_PACK_JOBS_WORKER_ENABLED=true` in the existing exclusive profile dotenv `/Users/macbook-dev/Documents/GitHub/tldw_server2/.tmp/fresh-uat-recovery-20260916/profiles/tldw-onboarding-uat-recovery-targeted-pg-multi-20260916/Config_Files/.env`, then replace only the owned API using the same profile/Jobs DB/provider configuration. The launcher recomputes child env at launch: editing its saved backend-env JSON alone does not change launch, and putting the flag in the shell is filtered out by its host allowlist. Preserve job2; do not submit again. Observe actual worker registration/start and queued→running→terminal behavior on that same job. The worker can consume other eligible jobs in this queue too; this is not a claim of job2-only execution.
3. Retain this as an explicit opt-in acceptance run, not a fresh-default pass. The default-start and drawer-state defects remain separately assessable. Do not delete/reset pending jobs, change provider, invoke the handler directly, or terminate individual database sessions to manufacture a success.

## Limits

Read-only source/history, selected nonsensitive profile flags, one owned API log search and supplied native receipts only. No runtime/browser/process/database/config/production/test/git/task/tracker mutations. Only this private diagnosis folder was written. No claim that all other worker specs preserve their historical defaults or that enabling this worker guarantees successful generation.
