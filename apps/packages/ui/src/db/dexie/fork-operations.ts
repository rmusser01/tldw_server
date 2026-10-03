/** Client dispatch guard, never a server receipt or restored write authority. */
import type { ForkRequestV1, ForkResultV1 } from "@/types/history-selection"

import { historyDigest } from "./history-selection"
import { db } from "./schema"
import type { ForkOperation, ForkOperationContext } from "./types"

const temporary = new Map<string, ForkOperation>()
let temporaryTail = Promise.resolve()
const claims = new WeakSet<object>()
export type ForkDispatchClaim = { readonly record: ForkOperation }
const copy = <T>(value: T): T => structuredClone(value)
const freeze = <T>(value: T): T => {
  if (value && typeof value === "object") {
    Object.values(value).forEach(freeze)
    Object.freeze(value)
  }
  return value
}
const sourceKey = (
  owner_key: string,
  conversation_id: string,
  context: ForkOperationContext
) => historyDigest({ owner_key, conversation_id, ...context })
const contextValue = (context: ForkOperationContext): ForkOperationContext => ({
  kind: context.kind,
  scope:
    context.scope?.type === "workspace"
      ? { type: "workspace", workspaceId: context.scope.workspaceId }
      : { type: "global" }
})
const tableFor = (context: ForkOperationContext) =>
  context.kind === "temporary"
    ? {
        get: async (key: string[]) => copy(temporary.get(JSON.stringify(key))),
        add: async (row: ForkOperation) => {
          temporary.set(
            JSON.stringify([row.owner_key, row.operation_id]),
            copy(row)
          )
        },
        put: async (row: ForkOperation) => {
          temporary.set(
            JSON.stringify([row.owner_key, row.operation_id]),
            copy(row)
          )
        },
        where: (field: "source_key" | "active_intent" | "candidate_key") => ({
          equals: (value: string) => ({
            toArray: async () =>
              copy(
                [...temporary.values()].filter((row) => row[field] === value)
              )
          })
        })
      }
    : db.forkOperations
const transaction = <T>(
  context: ForkOperationContext,
  fn: () => Promise<T>
): Promise<T> => {
  if (context.kind !== "temporary")
    return db.transaction("rw", db.forkOperations, fn)
  const result = temporaryTail.then(fn)
  temporaryTail = result.then(
    () => {},
    () => {}
  )
  return result
}
const requestDigest = (request: ForkRequestV1) =>
  historyDigest({
    operation_id: request.operation_id,
    owner_key: request.owner_key,
    destination_owner_key: request.destination_owner_key,
    input: request.input
  })
/** View revision/digest do not change the requested copy; membership, fences, policy and destination do. */
export const prepareForkOperation = async (
  request: ForkRequestV1,
  suppliedContext: ForkOperationContext
): Promise<ForkOperation> => {
  const context = contextValue(suppliedContext)
  if (
    !request.operation_id ||
    !request.owner_key ||
    request.request_digest !== requestDigest(request)
  )
    throw new Error("fork_request_mismatch")
  const frozenRequest = copy(request)
  const {
    selection_revision: _revision,
    selection_digest: _digest,
    ...selection
  } = request.input.selection as ForkRequestV1["input"]["selection"] & {
    selection_revision?: number
  }
  const source_key = sourceKey(
    request.owner_key,
    selection.conversation_id,
    context
  )
  const active_intent = historyDigest({
    source_key,
    destination: request.destination_owner_key,
    kind: request.input.kind,
    selection
  })
  return transaction(context, async () => {
    const table = tableFor(context)
    const existing = await table.get([request.owner_key, request.operation_id])
    if (existing) {
      if (
        existing.request.request_digest !== request.request_digest ||
        existing.source_key !== source_key
      )
        throw new Error("fork_operation_mismatch")
      return existing
    }
    const pending = await table
      .where("active_intent")
      .equals(active_intent)
      .toArray()
    if (pending.length) return pending[0]
    const record: ForkOperation = {
      owner_key: request.owner_key,
      operation_id: request.operation_id,
      conversation_id: selection.conversation_id,
      context,
      source_key,
      active_intent,
      request: frozenRequest,
      state: "prepared"
    }
    await table.add(record)
    return copy(record)
  })
}
/** The only transition that authorizes dispatch. Reopening any other state cannot claim it. */
export const claimForkOperation = async (
  record: ForkOperation
): Promise<ForkDispatchClaim | null> =>
  transaction(record.context, async () => {
    const table = tableFor(record.context)
    const live = await table.get([record.owner_key, record.operation_id])
    if (
      !live ||
      live.state !== "prepared" ||
      live.request.request_digest !== record.request.request_digest
    )
      return null
    const next: ForkOperation = { ...live, state: "dispatching" }
    await table.put(next)
    const claim = Object.freeze({ record: freeze(copy(next)) })
    claims.add(claim)
    return claim
  })
const updateClaim = async (
  claim: ForkDispatchClaim,
  update: (live: ForkOperation) => ForkOperation
) => {
  if (!claims.has(claim)) throw new Error("fork_dispatch_claim_required")
  return transaction(claim.record.context, async () => {
    const table = tableFor(claim.record.context)
    const live = await table.get([
      claim.record.owner_key,
      claim.record.operation_id
    ])
    if (
      !live ||
      live.request.request_digest !== claim.record.request.request_digest
    )
      throw new Error("fork_operation_mismatch")
    if (live.state === "completed" || live.state === "rejected") return live
    const next = update(live)
    await table.put(next)
    return next
  })
}
export const recordForkCandidate = (
  claim: ForkDispatchClaim,
  childId: string
) =>
  updateClaim(claim, (live) => {
    if (
      !childId ||
      (live.candidate_child_id && live.candidate_child_id !== childId)
    )
      throw new Error("fork_candidate_mismatch")
    return {
      ...live,
      candidate_child_id: childId,
      candidate_key: sourceKey(live.owner_key, childId, live.context),
      state: "partial"
    }
  })
export const finishForkOperation = (
  claim: ForkDispatchClaim,
  result: ForkResultV1
) =>
  updateClaim(claim, (live) => {
    if (
      result.owner_key !== live.owner_key ||
      result.operation_id !== live.operation_id
    )
      throw new Error("fork_result_mismatch")
    const completed =
      result.state === "committed" || result.state === "legacy_completed"
    const rejected = result.state === "rejected" || result.state === "blocked"
    if (live.candidate_child_id && rejected)
      throw new Error("fork_result_side_effect_mismatch")
    const candidate = completed
      ? result.child_id
      : "candidate_child_id" in result
        ? result.candidate_child_id
        : undefined
    if (
      candidate &&
      live.candidate_child_id &&
      candidate !== live.candidate_child_id
    )
      throw new Error("fork_candidate_mismatch")
    return {
      ...live,
      state: completed
        ? "completed"
        : rejected
          ? "rejected"
          : (result.state as "unknown" | "partial"),
      result: copy(result),
      candidate_child_id: candidate ?? live.candidate_child_id,
      candidate_key: candidate
        ? sourceKey(live.owner_key, candidate, live.context)
        : live.candidate_key,
      active_intent: completed || rejected ? undefined : live.active_intent
    }
  })
export const forkOperationResult = (record: ForkOperation): ForkResultV1 =>
  record.result ?? {
    state: record.candidate_child_id ? "partial" : "unknown",
    owner_key: record.owner_key,
    operation_id: record.operation_id,
    code: "fork_dispatch_unresolved",
    candidate_child_id: record.candidate_child_id
  }
/** Read and observation occur in one transaction, so stale readers cannot downgrade completion. */
export const loadForkOperations = async (
  address: {
    owner_key: string
    conversation_id: string
  } & ForkOperationContext
): Promise<ForkOperation[]> => {
  const context = contextValue(address)
  return transaction(context, async () => {
    const table = tableFor(context)
    const records = await table
      .where("source_key")
      .equals(sourceKey(address.owner_key, address.conversation_id, context))
      .toArray()
    for (const record of records) {
      if (record.state === "dispatching") {
        record.state = "unknown"
        await table.put(record)
      }
    }
    return records
  })
}
/** Deliberate new work only; retains the unresolved operation and never dispatches it. */
export const allowNewForkOperation = (record: ForkOperation) =>
  transaction(record.context, async () => {
    const table = tableFor(record.context)
    const live = await table.get([record.owner_key, record.operation_id])
    if (
      live &&
      (live.state === "prepared" ||
        live.state === "unknown" ||
        live.state === "partial")
    )
      await table.put({ ...live, active_intent: undefined })
  })

/** An inspection address is not authority: callers must supply a freshly verified current owner. */
export const findForkCandidate = async (
  address: { owner_key: string; child_id: string } & ForkOperationContext
): Promise<ForkOperation | null> => {
  const context = contextValue(address)
  const matches = await tableFor(context)
    .where("candidate_key")
    .equals(sourceKey(address.owner_key, address.child_id, context))
    .toArray()
  return matches[0] ?? null
}
