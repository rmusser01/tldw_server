import { vi } from "vitest"
export const memory: Record<string, any> = (() => {
  const tables: Record<string, any> = {}
  for (const name of [
    "chatHistories",
    "messages",
    "userSettings",
    "sessionFiles",
    "compareStates",
    "historySelections",
    "forkOperations",
    "historyProjections"
  ]) {
    const rows = new Map<string, any>()
    const key = (v: any) =>
      JSON.stringify(
        name === "forkOperations" ? [v.owner_key, v.operation_id] : name === "historySelections"
          ? [v.profile_id, v.client_session_id, v.owner_key, v.conversation_id]
          : name === "historyProjections"
            ? [v.owner_key, v.conversation_id, v.projection_id]
            : v.id ?? v.sessionId ?? v.history_id
      )
    tables[name] = {
      rows,
      name,
      get: async (id: any) => structuredClone(rows.get(JSON.stringify(id))),
      put: async (v: any) => rows.set(key(v), structuredClone(v)),
      add: async (v: any) => {
        if (rows.has(key(v))) throw new Error("duplicate")
        rows.set(key(v), structuredClone(v))
      },
      bulkAdd: async (vs: any[]) => {
        for (const v of vs) rows.set(key(v), structuredClone(v))
      },
      update: async (id: any, patch: any) => {
        const k = JSON.stringify(id)
        rows.set(k, { ...rows.get(k), ...structuredClone(patch) })
      },
      delete: async (id: any) => rows.delete(JSON.stringify(id)),
      where: (field: string) => ({
        equals: (v: any) => ({
          modify: async (patch: any) => {
            for (const [k, r] of rows)
              if (r[field] === v) rows.set(k, { ...r, ...patch })
          },
          toArray: async () =>
            structuredClone([...rows.values()].filter((r) => r[field] === v))
        }),
        anyOf: (vs: any[]) => ({
          toArray: async () =>
            [...rows.values()].filter((r) => vs.includes(r[field]))
        })
      })
    }
  }
  let tail = Promise.resolve()
  return {
    ...tables,
    transaction: vi.fn((_mode: any, _tables: any, operation: any) => {
      const result = tail.then(async () => {
        const copies = Object.values(tables).map((t: any) => new Map(t.rows))
        try {
          return await operation({
            abort() {
              throw new Error("aborted")
            }
          })
        } catch (e) {
          Object.values(tables).forEach((t: any, i) => {
            t.rows.clear()
            copies[i].forEach((v, k) => t.rows.set(k, v))
          })
          throw e
        }
      })
      tail = result.catch(() => {})
      return result
    })
  }
})()

export const seedLocalFork = async (comparison: boolean) => {
  for (const table of Object.values(memory)) if (table?.rows) table.rows.clear()
  await memory.userSettings.put({ id: "main", history_profile_id: "profile" })
  await memory.chatHistories.put({
    id: "history-1",
    title: "Source",
    createdAt: 1,
    is_rag: false,
    local_owner_key: "local-history-v1:profile"
  })
  const common = {
    history_provenance: {
      owner_key: "local-history-v1:profile",
      projection_id: null
    },
    history_id: "history-1",
    name: "You",
    content: "question",
    images: []
  }
  await memory.messages.bulkAdd([
    {
      ...common,
      id: "u-old",
      role: "user",
      createdAt: 1,
      parent_message_id: null,
      ...(comparison ? { messageType: "compare:user", clusterId: "c1" } : {})
    },
    {
      ...common,
      id: "a1",
      role: "assistant",
      content: "selected answer",
      createdAt: 2,
      parent_message_id: "u-old",
      ...(comparison
        ? { messageType: "compare:assistant", clusterId: "c1", modelId: "A" }
        : {})
    },
    {
      ...common,
      id: "a2",
      role: "assistant",
      content: "hidden answer",
      createdAt: 3,
      parent_message_id: "u-old",
      ...(comparison
        ? { messageType: "compare:assistant", clusterId: "c1", modelId: "B" }
        : {})
    }
  ])
  if (comparison)
    await memory.compareStates.put({
      history_id: "history-1",
      compareMode: true
    })
  return structuredClone([...memory.messages.rows.values()])
}
export const settings = new Map<string, unknown>()
