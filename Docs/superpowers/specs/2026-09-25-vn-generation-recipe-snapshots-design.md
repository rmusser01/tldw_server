# VN Asset Generation Recipe Snapshots

Backlog: TASK-13358. Parent issue: #2021. Base: `dev` at `a2f5e1b816`.

## Problem and Goal

Batch creation currently stores request options, slot IDs, and counts. Fanout later
reads current slots; each variant worker later reads the current pack, slot,
character, and world-book entries. The output's prompt snapshot is written only
after successful generation. An edit while work is queued can change its output,
and Retry currently creates a fresh batch from current values.

Accepted batches must retain the authored inputs used for every variant. Retry
must replay the selected failed batch's recipe; Regenerate must intentionally use
current inputs. The first worker to execute a batch must also pin effective
backend and model selection so later attempts do not silently select another
provider. This slice does not add a new image provider, change Jobs APIs, or claim
exactly-once output persistence after a process crash.

## Options Considered

1. **Dedicated batch recipe and execution fields (chosen).** Persist a versioned
   authored snapshot in the existing per-user VN database when accepting the
   batch; materialize worker-dependent backend/model choices once before fanout.
   The fields have a clear lifecycle, support atomic first-writer resolution,
   and leave arbitrary request options alone. Requires a small compatible DB
   migration and validation.
2. Store snapshots inside `options_json`. Avoids a migration but mixes caller
   options with server-owned state and makes atomic resolution and legacy
   detection harder.
3. Put a full recipe in each child Jobs payload. Replay of an existing child is
   simple, but partial fanout after worker restart can create siblings with
   different inputs unless a batch-level snapshot also exists.

## Stored Contract

Add nullable `recipe_json` and `execution_recipe_json` fields to
`vn_asset_batches`, and a nullable `source_batch_id` for retry provenance. A
missing recipe identifies a legacy batch. The authored recipe is immutable after
the batch insert and includes a version, pack and owner IDs, selected slot IDs,
variant counts, and per-slot values actually used to build a request: rendered
prompt and negative prompt, token and omission metadata, labels, asset type,
dimensions, format, allowed generation parameters, seed policy, and requested
backend/model. Rendering the prompt at acceptance freezes character and
world-book content without copying unrelated character records into a job.
The recipe copies only user-authored model selectors already present in this
per-user database, not API keys or provider credentials. The execution recipe
may include a public hosted-provider model ID sourced from worker configuration.
It is never returned by the status API or included in exports. A selector can
refer to a mutable local file; replay pins the selector, not the file bytes.

The execution recipe records the chosen backend and public model selector for
each selected slot, including a configured hosted-provider default when the
author did not choose a model. It is set once by a transaction before child
creation. Later workers reuse it. For an implicit local model, the execution
record stores the selected CLI mode and a digest of the configured path, not
the path; child workers reject a changed path or mode. Worker credentials and
provider secrets remain resolved at execution and are never copied into the
recipe. A provider or referenced local model may still become unavailable or
change in place on disk; this is not an immutable-artifact guarantee.

## Data Flow

1. The owner-scoped service validates selected slots and item limits, renders
   prompts with current character/world-book data, and inserts the batch and
   authored recipe in one transaction. An enqueue failure leaves a failed batch
   with its recipe for diagnosis. If a configured world book cannot be read,
   acceptance fails rather than silently freezing an incomplete prompt.
2. Fanout loads the stored recipe, resolves each backend/model with the worker's
   registry and configuration, then atomically stores the execution choices if
   no execution recipe exists. It rereads the winning value before creating
   deterministic child jobs. Child IDs and existing Jobs idempotency keys remain
   unchanged. A transient parent fanout failure may replay that same batch;
   terminal variant-failed and completed batches are not reopened.
3. A variant worker validates batch ownership and current pack/slot existence
   for storage, then constructs `ImageGenRequest` from the stored recipe and
   execution choices. It never rerenders a snapshotted prompt from current
   mutable records. The output keeps the existing prompt/context metadata.
4. A variant failure records its batch ID on the slot. The WebUI supplies that
   per-slot source, which may be older than the latest pack batch. The service
   verifies ownership, failed state, recorded provenance, slot membership, and
   snapshot version, then creates
   a new single-slot batch copying that slot's authored and execution recipe.
   It reuses the recorded variant count and seeds. An omitted source ID selects
   the recorded failed batch for that slot for older clients. A different
   `variant_count` or recipe-affecting option is rejected with a clear conflict.
   An unbound Retry with no failed batch is rejected; it does not become Start.
5. Regenerate continues through a fresh `start_generation` call, capturing
   current settings. A batch-level fanout failure with no slot attribution can
   also be retried from its failed batch. An old batch without a snapshot has
   actionable recovery.

## Compatibility and Recovery

The migration adds nullable columns to existing databases; old rows remain
readable. A child of a legacy batch uses the existing live-input behavior, but
its status identifies that it has no frozen recipe. A new Retry of a legacy
batch fails with `vn_asset_recipe_unavailable`, directing the user to Start a
new generation. This prevents an action labeled Retry from silently changing
the recipe. Existing idempotency claims are checked before new batch creation,
so a replayed request returns its original response even if the source batch
state has changed. Invalid recipe versions or owner/slot mismatches fail closed.

An interrupted fanout can resume from the same batch and execution recipe.
Existing deterministic Jobs keys suppress duplicate child enqueue. Counting
successful outputs after a crash between file registration and job completion
remains a separate #2021 recovery slice; this design does not claim that Jobs
idempotency makes output writes exactly once.

## API and User Experience

The Retry request accepts optional `source_batch_id` and retains its required
idempotency key. The returned generation status includes the new batch ID and
retry provenance and per-failed-slot recipe availability where available. A
snapshot-unavailable conflict names the recovery action in user terms. The
existing monitor keeps one Retry per failed slot and binds it to that slot's
recorded failed batch. No new setup screen is needed.

## Verification

Backend tests edit pack, slot, character, and world-book data after acceptance
and assert the generated request remains unchanged. They cover failed and
interrupted fanout, duplicate delivery, first-writer execution resolution,
local model selector pinning, retry provenance, legacy batches, and current-value
Regenerate behavior. API tests cover source ownership, idempotent replay and
conflicting retry overrides. Frontend tests verify the displayed batch ID is
sent and a legacy-snapshot error offers a usable next action. Run scoped Ruff,
Bandit, frontend tests/type checks, OpenAPI drift checks, and browser QA for the
changed monitor.
