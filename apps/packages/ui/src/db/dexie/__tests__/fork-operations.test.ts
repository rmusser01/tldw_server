import type { ForkRequestV1 } from "@/types/history-selection"
import { beforeEach, expect, it, vi } from "vitest"

import * as operations from "../fork-operations"
import { historyDigest } from "../history-selection"

// Serialized transaction double; real IndexedDB qualification belongs to Task 5.
const memory = vi.hoisted(() => {
  const rows = new Map<string, any>()
  let tail = Promise.resolve()
  const table = {
    get: async (key: string[]) =>
      structuredClone(rows.get(JSON.stringify(key))),
    put: async (row: any) =>
      rows.set(
        JSON.stringify([row.owner_key, row.operation_id]),
        structuredClone(row)
      ),
    add: async (row: any) => {
      const key = JSON.stringify([row.owner_key, row.operation_id])
      if (rows.has(key)) throw new Error("duplicate")
      rows.set(key, structuredClone(row))
    },
    where: (field: string) => ({
      equals: (value: string) => ({
        toArray: async () =>
          structuredClone(
            [...rows.values()].filter((row) => row[field] === value)
          )
      })
    })
  }
  return {
    rows,
    forkOperations: table,
    transaction: (_mode: string, _table: any, fn: any) => {
      const result = tail.then(fn)
      tail = result.catch(() => {})
      return result
    }
  }
})
vi.mock("../schema", () => ({ db: memory }))

const context = {
  kind: "native" as const,
  scope: { type: "workspace" as const, workspaceId: "original" }
}
const request = (id = "one", selectionRevision = 1): ForkRequestV1 => {
  const value: any = {
    operation_id: id,
    owner_key: "owner",
    destination_owner_key: "owner",
    input: {
      kind: "normal",
      selection: {
        version: 1,
        owner_key: "owner",
        conversation_id: "source",
        purpose: "fork",
        interpretation: { kind: "parent_graph_v1" },
        cursor: { kind: "after_message", message_id: "m" },
        selection_revision: selectionRevision,
        messages: [{ id: "m", revision: "1" }],
        fences: { conversation: "1", history: "1", settings: "0" },
        storage_context_digest: "context",
        request_context_digest: "projection",
        selection_digest: `view-${selectionRevision}`
      }
    }
  }
  return { ...value, request_digest: historyDigest(value) }
}
const address = { owner_key: "owner", conversation_id: "source", ...context }
beforeEach(() => memory.rows.clear())
it("two views attach to the winning immutable intent and only one claims dispatch", async () => {
  const original = request()
  const [a, b] = await Promise.all([
    operations.prepareForkOperation(original, context),
    operations.prepareForkOperation(request("two", 8), context)
  ])
  expect(b.request).toEqual(original)
  const claims = await Promise.all([
    operations.claimForkOperation(a),
    operations.claimForkOperation(b)
  ])
  expect(claims.filter(Boolean)).toHaveLength(1)
  expect(memory.rows.size).toBe(1)
})
it("a changed selected copy and another workspace are distinct intents", async () => {
  await operations.prepareForkOperation(request(), context)
  const changed = request("changed")
  const value: any = {
    ...changed,
    input: {
      ...changed.input,
      selection: {
        ...changed.input.selection,
        messages: [{ id: "m", revision: "2" }]
      }
    }
  }
  value.request_digest = historyDigest({
    operation_id: value.operation_id,
    owner_key: value.owner_key,
    destination_owner_key: value.destination_owner_key,
    input: value.input
  })
  await operations.prepareForkOperation(value, context)
  await operations.prepareForkOperation(request("other-space"), {
    ...context,
    scope: { type: "workspace", workspaceId: "other" }
  })
  expect(memory.rows.size).toBe(3)
  expect(
    await operations.loadForkOperations({
      ...address,
      owner_key: "new-account"
    })
  ).toEqual([])
  expect(
    await operations.loadForkOperations({
      ...address,
      scope: { type: "global" }
    })
  ).toEqual([])
})
it("same ID cannot mutate the immutable request or original workspace", async () => {
  await operations.prepareForkOperation(request(), context)
  await expect(
    operations.prepareForkOperation(request("one", 9), context)
  ).rejects.toThrow("fork_operation_mismatch")
  await expect(
    operations.prepareForkOperation(request(), {
      ...context,
      scope: { type: "global" }
    })
  ).rejects.toThrow("fork_operation_mismatch")
})
it("reopen observes dispatch as unknown without granting dispatch and late dispatcher refines it", async () => {
  const record = await operations.prepareForkOperation(request(), context)
  const claim = (await operations.claimForkOperation(record))!
  const recovered = await operations.loadForkOperations(address)
  expect(recovered[0].state).toBe("unknown")
  expect(await operations.claimForkOperation(recovered[0])).toBeNull()
  await operations.recordForkCandidate(claim, "child")
  expect(
    (await operations.loadForkOperations(address))[0].candidate_child_id
  ).toBe("child")
  await operations.finishForkOperation(claim, {
    state: "legacy_completed",
    owner_key: "owner",
    operation_id: "one",
    child_id: "child"
  })
  // A stale recovery object cannot overwrite terminal success or acquire dispatch.
  expect(await operations.claimForkOperation(recovered[0])).toBeNull()
  expect((await operations.loadForkOperations(address))[0].state).toBe(
    "completed"
  )
})
it("unknown retains intent until explicit new work is allowed, preserving the old request", async () => {
  const record = await operations.prepareForkOperation(request(), context)
  const claim = (await operations.claimForkOperation(record))!
  await operations.finishForkOperation(claim, {
    state: "unknown",
    owner_key: "owner",
    operation_id: "one",
    code: "response_lost"
  })
  expect(
    (await operations.prepareForkOperation(request("two"), context))
      .operation_id
  ).toBe("one")
  await operations.allowNewForkOperation(record)
  expect(
    (await operations.prepareForkOperation(request("three"), context))
      .operation_id
  ).toBe("three")
  expect(
    (await operations.loadForkOperations(address)).find(
      (row) => row.operation_id === "one"
    )?.request
  ).toEqual(request())
})
it("temporary operations stay memory-only and retain no live owner credentials", async () => {
  const record = await operations.prepareForkOperation(request("temporary"), {
    kind: "temporary",
    scope: { type: "global" }
  })
  expect(await operations.claimForkOperation(record)).toBeTruthy()
  expect(memory.rows.size).toBe(0)
  expect(JSON.stringify(record)).not.toMatch(
    /request_scope|config|validate_lease/
  )
})
it("candidate lookup binds original workspace and owner even after source navigation", async () => {
  const record = await operations.prepareForkOperation(request(), context)
  const claim = (await operations.claimForkOperation(record))!
  await operations.recordForkCandidate(claim, "child")
  expect(
    (
      await operations.findForkCandidate({
        owner_key: "owner",
        child_id: "child",
        ...context
      })
    )?.operation_id
  ).toBe("one")
  expect(
    await operations.findForkCandidate({
      owner_key: "owner",
      child_id: "child",
      ...context,
      scope: { type: "workspace", workspaceId: "elsewhere" }
    })
  ).toBeNull()
  expect(
    await operations.findForkCandidate({
      owner_key: "other",
      child_id: "child",
      ...context
    })
  ).toBeNull()
})
it("simultaneous recovery observation and late completion settle monotonically", async () => {
  const record = await operations.prepareForkOperation(request(), context)
  const claim = (await operations.claimForkOperation(record))!
  await Promise.all([
    operations.loadForkOperations(address),
    operations.finishForkOperation(claim, {
      state: "legacy_completed",
      owner_key: "owner",
      operation_id: "one",
      child_id: "child"
    })
  ])
  const final = (await operations.loadForkOperations(address))[0]
  expect(final.state).toBe("completed")
  expect(final.result).toMatchObject({
    state: "legacy_completed",
    child_id: "child"
  })
  expect(await operations.claimForkOperation(record)).toBeNull()
})
it("dispatch capability cannot be retargeted by mutating its original request", async () => {
  const record = await operations.prepareForkOperation(request(), context)
  const claim = (await operations.claimForkOperation(record))!
  expect(() => {
    ;(claim.record.request as any).owner_key = "other"
  }).toThrow()
  expect(() => {
    ;(claim.record.context.scope as any).workspaceId = "other"
  }).toThrow()
})

it("an abandoned prepared record permits explicit distinct work without deleting the original", async () => {
  const record = await operations.prepareForkOperation(request(), context)
  await operations.allowNewForkOperation(record)
  expect(
    (await operations.prepareForkOperation(request("new"), context))
      .operation_id
  ).toBe("new")
  expect(
    (await operations.loadForkOperations(address)).find(
      (row) => row.operation_id === "one"
    )?.request
  ).toEqual(record.request)
})
