# Manual llama.cpp slot snapshots

Snapshots preserve processed runtime context for an administrator to reuse later.
They do not restore a Chatbook conversation, messages, tools, pending approvals or
provider selection. Pause and Resume retain their existing process semantics;
neither action saves or restores a snapshot. Nothing sends a chat automatically.

## Current support and evidence

**No production executable build is currently verified.** The production
`TESTED_TEXT_BUILD_SHA256` allowlist is intentionally empty. An enabled runtime
therefore reports `unsupported_build` until measured live evidence is reviewed and
an exact executable hash is admitted. Enabling this feature is not a promise of
current runtime support. Unit tests and mocked browser tests do not change that.

The initial contract is text-only, single-model, managed runtimes, with a bounded
set of context/cache options. Multimodal projectors, adapters, router/draft modes
and unverified options fail closed. Matching filenames or token counts do not
establish compatibility. There is no force-restore override, import, download or
arbitrary filesystem-path input in the snapshot UI.

Snapshot-enabled launches require a numeric loopback bind, such as `127.0.0.1`
or `::1`. Wildcard, LAN/public addresses and DNS names (including `localhost`)
are rejected; change the profile host to a loopback literal before starting it.
The child exposes native slot-management routes on that local listener, even
while the production build allowlist is empty. Local callers can bypass tldw's
admin checks, compatibility gate and receipts. Use only on a trusted host,
restrict local access, and never forward or proxy those native routes to users.
Loopback binding does not isolate untrusted local users or processes.

Snapshot storage currently requires POSIX ownership locking and descriptor-based
filesystem confinement. Unsupported platforms fail closed when storage is
initialized. Ordinary snapshot-disabled runtime lifecycle remains supported on
Windows; this feature does not add Windows snapshot support.

Protocol fixtures are derived from llama.cpp revision
`4d9176092d00586775af140581bb0b558ddc4389`, not captured from a live executable.
At this revision [server-common.cpp](https://github.com/ggml-org/llama.cpp/blob/4d9176092d00586775af140581bb0b558ddc4389/tools/server/server-common.cpp#L67)
serializes actual prompt reuse as `timings.cache_n` and newly processed prompt
tokens as `timings.prompt_n`. Top-level `tokens_cached` is the final slot cache
size and must not be mistaken for reuse evidence.

## Prepare and operate

1. Open **Admin → llama.cpp**, locate the managed profile in Runtime instances,
   and select **Slot snapshots**. Administrative authorization is required for
   the catalog, slots, mutations and operation receipts.
2. Select **Enable snapshots**. This saves the setting without restarting the
   process. Stop and start that profile explicitly when the panel reports
   **Restart required**. A stopped profile must be started before inspection.
3. Quiesce every caller of the runtime. Slots can contain more than one user's
   context. An idle observation does not reserve the slot against concurrent
   inference, and a slot number does not identify a conversation or user.
4. Process the intended text using the original chat, wait for completion, then
   **Refresh** slots. Save an idle slot with processed tokens. Wait for the
   durable operation to reach **Complete** before lifecycle actions.
5. Inspect saved timestamps (with local timezone), tokens, size, exact copyable
   ID and compatibility reasons. **Keep newest** defaults to 10 and accepts
   1–1000. Applying a limit alone deletes nothing: pruning follows a verified
   successful save. A pruning warning leaves the new snapshot committed.
6. Start the matching managed profile if needed. Choose a compatible saved copy,
   select an idle destination, read the replacement warning, and explicitly
   select **Restore into slot N**. This replaces the destination cache. Failure
   may also clear it. Messages and tool state will not be restored.
7. After **Complete**, open the original conversation in Chatbook to continue.
   A request must actually reach the restored slot with a matching token prefix
   and compatible template/options to reuse it. The native verification harness
   explicitly targets slot 0; normal Chatbook/OpenAI-compatible routing may select
   another slot. One successful reuse does not establish reuse for every chat.

The catalog is sorted newest first and paginated. Disabling snapshots preserves
catalog browse/delete access. **Delete** opens an inline confirmation naming the
permanent target: it removes the saved copy, never erases an active slot. Profile
deletion requires explicitly deleting its saved snapshots first.

## Completion, reload and recovery

The panel announces Validating, Saving, Verifying, Restoring, Complete, Failed or
Outcome unknown without fabricated percentages. Closing the page does not cancel
server work. Reopening the profile recovers the latest operation ID through the
slots endpoint and reads its durable receipt; it does not resubmit the mutation.
Polling runs for active operations while the selected Admin surface is visible.

A read failure is an error, not an empty catalog. Refresh to recover status.
A pre-dispatch failure can be retried manually after resolving its cause. A
timeout, disconnect or malformed response after dispatch can leave the outcome
unknown. **Do not retry Restore.** Use **Stop recovery**, read the warning that
inference for all callers ends, then confirm **Stop runtime and inference**.
The server must confirm that the owned child exited before another launch can
mutate snapshots. Start again explicitly and inspect the catalog and slot state.
If the owner is unavailable, restore ownership/service health before proceeding.
Do not manually delete quarantined working files while a child could still use them.

## Privacy, storage and backups

Snapshots belong to a managed profile and server installation, not to the
administrator who saved them. Treat artifacts, backups and replicas as sensitive
conversation material even though this UI never previews prompts or cache bytes.
Private service-owned directories, hashes and manifest-last publication protect
ownership/integrity; they do not provide encryption at rest. Configure disk
encryption and restrict service-account, root, backup and restore access. Snapshot
deletion does not remove older backups or promise secure physical erasure.
Do not transfer snapshots across installations or expose native slot-management
routes to ordinary users. Shared-slot user attribution is not supported.

## Opt-in live verification

The harness creates fresh, disposable profiles and storage under pytest's
temporary directory. It never opens the production profile store and never
downloads an executable or model. Supply absolute, regular, non-symlink paths
to an operator-approved text model and llama-server executable. The test admits
that candidate hash only in its private in-memory service; it never edits the
production allowlist. It starts real processes on loopback, using CPU, context
16384, parallelism 1 and synthetic public text. Allow sufficient RAM, disk and
processing time for the supplied model.

```sh
source /path/to/tldw_server/.venv/bin/activate
TLDW_SNAPSHOT_LIVE=1 TLDW_SNAPSHOT_DISPOSABLE=YES \
TLDW_SNAPSHOT_EXECUTABLE=/absolute/path/to/llama-server \
TLDW_SNAPSHOT_MODEL=/absolute/path/to/text-model.gguf \
python -m pytest tldw_Server_API/tests/LLM_Local/test_llamacpp_snapshots_live.py \
  -q -s --junitxml=/tmp/llamacpp-snapshot-live.xml
```

The real runner, private store and operation coordinator seed a long prefix, save,
stop, start, restore, and submit the same prefix plus a suffix. A separate cold
process receives the identical request. Success requires explicit nonnegative
`timings.cache_n` and `timings.prompt_n` counters, at least 1024 saved tokens,
reuse of at least 80% of the saved tokens, and restored processing below 25% of
the cold control. Missing counters fail closed. Executable/model SHA-256,
effective options and sanitized token metrics are recorded in the JUnit property
`snapshot_live_evidence` and printed as JSON. No private prompts or generated
answers are included. HTTP 200, file existence, elapsed time alone and similar
answers are not acceptance evidence.

This harness is runtime evidence only. To finish validation, exercise the Admin
flow against the same disposable managed runtime in a browser, and verify an
original disposable Chatbook conversation's messages, tools and approvals before
and after. Check Pause/Resume without snapshot calls, record actual routing and
template behavior, and attach sanitized metrics/screenshots. The committed
Playwright workflow uses mocked APIs and is explicitly labeled as such; it cannot
satisfy those live checks.

### Recorded verification on 2026-09-04

No llama-server/model paths were supplied, so the live test was skipped. No live
reuse metrics, verified production hash or real Chatbook mutation proof is
claimed. TASK-13163 remains open pending that evidence. See the task's Stage 3
report for targeted automated UI and harness validation.

Architecture: [ADR-043](../ADR/043-managed-llamacpp-manual-slot-snapshots.md).
Approved workflow: [design](../Design/2026-09-04-llamacpp-manual-slot-snapshots.md).
