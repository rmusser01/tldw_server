# VN Asset Generation Durability

Backlog: TASK-13356. GitHub: #2021.

## Goal

A generation batch must keep the recipe selected at submission and reach a
recoverable, accurate terminal state after duplicate delivery or worker restart.
The browser must be able to rediscover the latest batch after a reload.

## Recipe Boundary

At batch creation, assemble the effective prompt and negative prompt from the
pack, slot, character, and selected world-book entries. Persist one immutable
recipe per `(batch_id, slot_id, variant_index)` with seed, requested backend,
model, dimensions, format, image parameters, asset type, and labels. Store the
recipes and batch in one ChaChaNotes transaction before creating the parent Job.
The image adapter's availability and the server's configured default backend
remain execution-time checks; a missing explicit backend is recorded as null.
The generated request ID is derived from stable batch/slot/variant IDs.

Mark new batches with `recipe_version=1`. Workers require a matching recipe for
these batches and never substitute current pack, slot, character, or world-book
values. Preexisting `recipe_version=0` batches retain their old behavior until
they finish, so deployment does not strand queued work. The API does not expose
raw recipe text in generation status.

## Replay Boundary

Use each V1 recipe row as its durable per-variant outcome ledger, keyed by
`(batch_id, slot_id, variant_index)`. Record a reserved, hidden item ID before
registering image bytes. Its stable `vn_asset_item:{item_id}` source reference
lets a retry find a file registered before the worker crashed. Reveal the item
as a draft only after storage metadata is attached. A duplicate completed Job
returns the same item without calling the adapter again. Compute batch counters
from recipe outcomes in one transaction instead of incrementing them in worker
memory. Reserved items remain absent from normal item listings until committed;
direct item, review, preferred, and file reads also reject unpublished
reservations. Failed reservations do not consume the pack item limit. A late
failure cannot undo a completed outcome or demote an approved item during
redelivery. A failed variant leaves queued siblings runnable; the batch becomes
failed only when all planned variants have completed or failed. One completed
draft keeps its slot reviewable if another variant fails. A cancelled batch
cannot publish a reserved variant, including when cancellation races the
worker's final transaction. Active recipe slots cannot be deleted until their
batch is terminal. Parent fanout must not overwrite a child worker's newer
processing or terminal batch status. Keep the existing Jobs manager as the
lease, queue, and cancellation authority; the recipe ledger does not create a
second lease clock.

## Browser Recovery

Bind each Start/Retry idempotency receipt to the newly created batch in the
same ChaChaNotes transaction as the batch and recipe rows. A same-key retry
with an unfinished receipt finds that exact owned batch and, if its parent
Job ID was not recorded, calls the Jobs create operation again using the
batch's deterministic Jobs key. It returns the batch status and completes the
receipt rather than creating another batch. A receipt with no committed batch
returns an in-progress conflict for two minutes, then permits the same key and
payload to reclaim it. The receipt-to-batch conditional update fences old and
new contenders so at most one batch commits. Never guess a batch from the
pack's latest batch.

Use the existing latest generation status API on mount and pack selection.
Store only unresolved Start/Retry idempotency keys in tab-scoped storage,
namespaced by owner and pack. Also remember the selected pack ID per owner so
reload can reconcile a pending request on a non-first pack. On reload, replay
the exact pending operation once with its original key; on acknowledgement
remove it and refresh pack details. A still-in-progress conflict keeps the key
for a later retry. Do not infer that the latest batch belongs to that key.
Keep polling bounded and non-overlapping while a batch is active.

## Verification

Tests must prove that editing source records after submission does not alter
the worker request, that a missing V1 recipe fails closed, and that legacy V0
batches still run. Replay tests must simulate duplicate delivery and crashes at
the item/storage boundary. Browser tests must reload during an ambiguous Start
response and recover without submitting a second batch. Run scoped lint,
Bandit, and API/browser checks with each reviewable slice.
