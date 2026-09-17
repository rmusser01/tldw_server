import { beforeEach, describe, expect, it, vi } from "vitest"
// Transaction orchestration double only; real IndexedDB qualification is Stage 5.
const memory: Record<string, any> = vi.hoisted(() => {
  const tables: Record<string, any> = {}
  for (const name of [
    "chatHistories",
    "messages",
    "userSettings",
    "sessionFiles",
    "compareStates",
    "historySelections",
    "historyProjections"
  ]) {
    const rows = new Map<string, any>()
    const key = (v: any) =>
      JSON.stringify(
        name === "historySelections"
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
      toArray: async () => structuredClone([...rows.values()]),
      clear: async () => {
        rows.clear()
      },
      bulkPut: async (vs: any[]) => {
        for (const v of vs) rows.set(key(v), structuredClone(v))
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
          delete: async () => {
            for (const [k, r] of rows) if (r[field] === v) rows.delete(k)
          },
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
})
vi.mock("../schema", () => ({ db: memory }))
vi.mock("@/services/tldw/TldwApiClient", () => ({ tldwClient: {} }))
const settingsStorage = vi.hoisted(() => ({
  raw: undefined as any,
  localRaw: undefined as any,
  get: vi.fn(),
  set: vi.fn()
}))
vi.mock("@/utils/safe-storage", () => ({
  createSafeStorage: (options: any) => ({
    get: (key: string) => settingsStorage.get(key, options?.area ?? "sync"),
    set: (key: string, value: unknown) =>
      settingsStorage.set(key, value, options?.area ?? "sync")
  })
}))
import { saveChatSettingsForKey } from "@/services/chat-settings"

vi.mock("../nickname", () => ({
  getAllModelNicknames: vi.fn(async () => ({}))
}))
import {
  captureLocalForkSelection,
  prepareLocalFork,
  commitLocalFork,
  forkRequestDigest
} from "../branch"
import {
  getLocalHistoryOwner,
  captureLocalHistorySnapshot
} from "../history-selection"
import { PageAssistDatabase } from "../chat"
import { selectionDigest } from "@/utils/history-selection"
const row = (
  id: string,
  parent: string | null,
  createdAt: number,
  extra = {}
) => ({
  id,
  history_id: "source",
  name: "You",
  role: "user",
  content: id,
  createdAt,
  parent_message_id: parent,
  ...extra
})
let owner: any
beforeEach(async () => {
  settingsStorage.raw = undefined
  settingsStorage.localRaw = undefined
  settingsStorage.get
    .mockReset()
    .mockImplementation(async (_key, area) =>
      area === "local" ? settingsStorage.localRaw : settingsStorage.raw
    )
  settingsStorage.set
    .mockReset()
    .mockImplementation(async (_key, value, area) => {
      if (area === "local") settingsStorage.localRaw = value
      else settingsStorage.raw = value
    })
  for (const table of Object.values(memory))
    if ((table as any)?.rows) (table as any).rows.clear()
  await memory.userSettings.put({ id: "main", history_profile_id: "profile" })
  await memory.chatHistories.put({
    id: "source",
    title: "Source",
    is_rag: false,
    createdAt: 1,
    last_used_prompt: {
      prompt_id: "private-prompt",
      prompt_content: "literal"
    },
    is_pinned: true
  })
  await memory.messages.bulkAdd([
    row("u1", null, 30, {
      serverMessageId: "inert-source-id",
      metadataExtra: { secret: "omit" }
    }),
    row("a1", "u1", 10, { role: "assistant" }),
    row("u2", "a1", 20)
  ])
  await memory.sessionFiles.put({
    sessionId: "source",
    retrievalEnabled: true,
    createdAt: 1,
    files: [
      {
        id: "file-source",
        filename: "notes.txt",
        type: "text/plain",
        content: "owned text",
        size: 10,
        uploadedAt: 1,
        processed: true,
        processingStatus: "ready",
        ingestJobId: 42,
        documentDraftId: "source-draft",
        ingestBatchId: "batch",
        ingestIdempotencyKey: "key",
        processingRecoveryActions: ["cancel"],
        processingResultRef: { kind: "ingest_job", id: 42 }
      }
    ]
  })
  owner = await getLocalHistoryOwner("source")
})
const view = () => ({
  owner_key: owner.owner_key,
  conversation_id: "source",
  view_session_id: "view",
  selection_revision: 1,
  interpretation: { kind: "parent_graph_v1" as const },
  cursor: { kind: "after_message" as const, message_id: "u2" }
})
const request = async (comparison?: any) => {
  const input = await captureLocalForkSelection(owner, comparison ?? view())
  const value = {
    operation_id: crypto.randomUUID(),
    owner_key: owner.owner_key,
    destination_owner_key: owner.owner_key,
    input,
    request_digest: ""
  }
  return { ...value, request_digest: forkRequestDigest(value) }
}
it("copies exact ancestry despite timestamps, owns files and strips source authority", async () => {
  const result = await commitLocalFork(await prepareLocalFork(await request()))
  expect(result.state).toBe("committed")
  if (result.state !== "committed") throw Error(result.state)
  const children = await memory.messages
    .where("history_id")
    .equals(result.child_id)
    .toArray()
  expect(children.map((r: any) => r.content)).toEqual(["u1", "a1", "u2"])
  expect(children.map((r: any) => r.parent_message_id)).toEqual([
    null,
    result.message_map.u1,
    result.message_map.a1
  ])
  expect(children.map((r: any) => r.depth)).toEqual([0, 1, 2])
  expect(children[0]).not.toHaveProperty("serverMessageId")
  expect(children[0]).not.toHaveProperty("metadataExtra")
  const child = await memory.chatHistories.get(result.child_id)
  expect(child.last_used_prompt).toEqual({ prompt_content: "literal" })
  expect(child.is_pinned).toBe(false)
  const files = await memory.sessionFiles.get(result.child_id)
  expect(files.retrievalEnabled).toBe(true)
  expect(files.files[0].content).toBe("owned text")
  expect(files.files[0].id).not.toBe("file-source")
  for (const key of [
    "ingestJobId",
    "documentDraftId",
    "ingestBatchId",
    "ingestIdempotencyKey",
    "processingRecoveryActions",
    "processingResultRef"
  ])
    expect(files.files[0]).not.toHaveProperty(key)
  expect((await memory.sessionFiles.get("source")).files[0].ingestJobId).toBe(
    42
  )
})
it("ignores excluded source controls but rejects retained file drift", async () => {
  const prepared = await prepareLocalFork(await request())
  const files = await memory.sessionFiles.get("source")
  files.files[0].ingestJobId = 99
  await memory.sessionFiles.put(files)
  expect((await commitLocalFork(prepared)).state).toBe("committed")
  const next = await prepareLocalFork(await request())
  files.files[0].content = "changed"
  await memory.sessionFiles.put(files)
  expect(await commitLocalFork(next)).toMatchObject({
    state: "rejected",
    code: "stale_selection"
  })
})
it("blocks unsupported protected assets before child writes", async () => {
  await memory.messages.update("u2", { images: ["https://protected/image"] })
  await expect(prepareLocalFork(await request())).rejects.toThrow(
    "unsupported_local_message_assets"
  )
  expect(memory.chatHistories.rows.size).toBe(1)
})
it.each(["missing", "duplicate"])(
  "rejects %s manifest members",
  async (kind) => {
    const value = await request()
    const selection: any = value.input.selection
    selection.messages =
      kind === "missing"
        ? [...selection.messages, { id: "missing", revision: "x" }]
        : [...selection.messages, selection.messages[0]]
    selection.selection_digest = selectionDigest(selection)
    value.request_digest = forkRequestDigest(value)
    await expect(prepareLocalFork(value)).rejects.toThrow()
    expect(memory.chatHistories.rows.size).toBe(1)
  }
)
it("rejects actual server mirrors", async () => {
  await memory.chatHistories.update("source", { server_chat_id: "native" })
  await expect(request()).rejects.toThrow("unbound_server_mirror")
})
it("normal send rejects comparison and comparison forks materialize only A", async () => {
  memory.messages.rows.clear()
  await memory.compareStates.put({ history_id: "source", compareMode: true })
  await memory.messages.bulkAdd([
    row("u1", null, 1, { messageType: "compare:user", clusterId: "c1" }),
    row("a1", "u1", 2, {
      role: "assistant",
      messageType: "compare:assistant",
      clusterId: "c1",
      modelId: "A"
    }),
    row("b1", "u1", 3, {
      role: "assistant",
      messageType: "compare:assistant",
      clusterId: "c1",
      modelId: "B"
    }),
    row("u2", "b1", 4, { messageType: "compare:user", clusterId: "c2" }),
    row("a2", "u2", 5, {
      role: "assistant",
      messageType: "compare:assistant",
      clusterId: "c2",
      modelId: "A"
    }),
    row("b2", "u2", 6, {
      role: "assistant",
      messageType: "compare:assistant",
      clusterId: "c2",
      modelId: "B"
    }),
    row("ua", "a2", 7, {
      messageType: "compare:perModelUser",
      clusterId: "c2",
      modelId: "A"
    })
  ])
  expect(
    await captureLocalHistorySnapshot(owner, view(), "send")
  ).toMatchObject({ code: "unsupported_comparison_history" })
  const result = await commitLocalFork(
    await prepareLocalFork(
      await request({
        model_id: "A",
        cluster_id: "c2",
        cursor: { kind: "after_message", message_id: "ua" }
      })
    )
  )
  expect(result.state).toBe("committed")
  if (result.state !== "committed") throw Error(result.state)
  const rows = await memory.messages
    .where("history_id")
    .equals(result.child_id)
    .toArray()
  expect(rows.map((r: any) => r.content)).toEqual([
    "u1",
    "a1",
    "u2",
    "a2",
    "ua"
  ])
  expect(rows[2].parent_message_id).toBe(result.message_map.a1)
  expect(result.message_map).not.toHaveProperty("b1")
})
it("production scoped mutations reject cross-history and missing IDs", async () => {
  await memory.chatHistories.put({
    id: "other",
    local_owner_key: owner.owner_key
  })
  const database = new PageAssistDatabase()
  await expect(database.updateMessage("other", "u1", "bad")).rejects.toThrow(
    "message_owner_mismatch"
  )
  await expect(database.removeMessage("other", "a1")).rejects.toThrow(
    "message_owner_mismatch"
  )
  await expect(database.removeMessage("source", "greeting")).rejects.toThrow(
    "missing_message"
  )
  expect((await memory.messages.get("u1")).content).toBe("u1")
  expect(await memory.messages.get("a1")).toBeDefined()
})
it("commits the prepared child identity and ignores caller attempts to replace its rows", async () => {
  const prepared = await prepareLocalFork(await request())
  const childId = prepared.history.id
  prepared.messages[0].content = "tampered"
  prepared.history.server_chat_id = "injected"
  const result = await commitLocalFork(prepared)
  expect(result).toMatchObject({ state: "committed", child_id: childId })
  expect((await memory.messages.get(prepared.message_map.u1)).content).toBe(
    "u1"
  )
  expect(await memory.chatHistories.get(childId)).not.toHaveProperty(
    "server_chat_id"
  )
})
it("reopened child edits and deletes leave source and its hidden sibling untouched", async () => {
  await memory.messages.add(
    row("hidden", "u1", 50, {
      history_provenance: {
        owner_key: owner.owner_key,
        version: 1,
        projection_id: null
      }
    })
  )
  const result = await commitLocalFork(await prepareLocalFork(await request()))
  if (result.state !== "committed") throw Error(result.state)
  const reopenedOwner = await getLocalHistoryOwner(result.child_id)
  const reopened = await captureLocalHistorySnapshot(
    reopenedOwner,
    {
      ...view(),
      owner_key: reopenedOwner.owner_key,
      conversation_id: result.child_id,
      cursor: { kind: "after_message", message_id: result.message_map.u2 }
    },
    "send"
  )
  expect(reopened.status).toBe("captured")
  if (reopened.status === "captured")
    expect(reopened.rows.map((row) => row.id)).toEqual([
      result.message_map.u1,
      result.message_map.a1,
      result.message_map.u2
    ])
  const database = new PageAssistDatabase()
  await database.updateMessage(
    result.child_id,
    result.message_map.a1,
    "child edited"
  )
  await database.removeMessage(result.child_id, result.message_map.u2)
  expect((await memory.messages.get("u2")).content).toBe("u2")
  expect(await memory.messages.get("a1")).toBeDefined()
  expect(await memory.messages.get("hidden")).toBeDefined()
  expect((await memory.messages.get(result.message_map.a1)).content).toBe(
    "child edited"
  )
  expect(await memory.messages.get(result.message_map.u2)).toBeUndefined()
})
it("transaction rollback removes the entire child on a file write failure", async () => {
  const prepared = await prepareLocalFork(await request())
  const write = vi
    .spyOn(memory.sessionFiles, "add")
    .mockRejectedValueOnce(new Error("file write failed"))
  expect(await commitLocalFork(prepared)).toMatchObject({
    state: "rejected",
    code: "file write failed"
  })
  expect(memory.chatHistories.rows.size).toBe(1)
  expect(memory.messages.rows.size).toBe(3)
  write.mockRestore()
})
it("comparison ancestry determines round and follow-up order despite shuffled timestamps", async () => {
  memory.messages.rows.clear()
  await memory.compareStates.put({ history_id: "source", compareMode: true })
  await memory.messages.bulkAdd([
    row("a2", "u2", 1, {
      role: "assistant",
      messageType: "compare:reply",
      clusterId: "c2",
      modelId: "A"
    }),
    row("ua", "a2", 0, {
      messageType: "compare:perModelUser",
      clusterId: "c2",
      modelId: "A"
    }),
    row("u2", "b1", 7, { messageType: "compare:user", clusterId: "c2" }),
    row("b1", "u1", 2, {
      role: "assistant",
      messageType: "compare:reply",
      clusterId: "c1",
      modelId: "B"
    }),
    row("a1", "u1", 9, {
      role: "assistant",
      messageType: "compare:reply",
      clusterId: "c1",
      modelId: "A"
    }),
    row("u1", null, 10, { messageType: "compare:user", clusterId: "c1" })
  ])
  const result = await commitLocalFork(
    await prepareLocalFork(
      await request({
        model_id: "A",
        cluster_id: "c2",
        cursor: { kind: "after_message", message_id: "ua" }
      })
    )
  )
  if (result.state !== "committed") throw Error(result.state)
  expect(
    (
      await memory.messages
        .where("history_id")
        .equals(result.child_id)
        .toArray()
    ).map((row: any) => row.content)
  ).toEqual(["u1", "a1", "u2", "a2", "ua"])
})
it("ambiguous comparison model threads reject before a child write", async () => {
  memory.messages.rows.clear()
  await memory.compareStates.put({ history_id: "source", compareMode: true })
  await memory.messages.bulkAdd([
    row("u1", null, 1, { messageType: "compare:user", clusterId: "c1" }),
    row("a1", "u1", 2, {
      role: "assistant",
      messageType: "compare:reply",
      clusterId: "c1",
      modelId: "A"
    }),
    row("a2", "u1", 3, {
      role: "assistant",
      messageType: "compare:reply",
      clusterId: "c1",
      modelId: "A"
    })
  ])
  await expect(
    request({
      model_id: "A",
      cluster_id: "c1",
      cursor: { kind: "after_message", message_id: "a2" }
    })
  ).rejects.toThrow("ambiguous_comparison_thread")
  expect(memory.chatHistories.rows.size).toBe(1)
})

it("draft, prefill and summary changes alone preserve the prepared retained context", async () => {
  const prepared = await prepareLocalFork(await request())
  await memory.chatHistories.update("source", {
    draft: "new",
    prefill: "new",
    summary: "new",
    compaction: "new",
    revision: 99
  })
  expect((await commitLocalFork(prepared)).state).toBe("committed")
})
it("retained source edits reject the copy without creating a child", async () => {
  const prepared = await prepareLocalFork(await request())
  await memory.messages.update("a1", { content: "changed" })
  expect(await commitLocalFork(prepared)).toMatchObject({
    state: "rejected",
    code: "stale_selection"
  })
  expect(memory.chatHistories.rows.size).toBe(1)
})
it.each([
  { character_id: 12 },
  { is_rag: true },
  { doc_id: "external" },
  { last_used_prompt: { prompt_id: "external" } }
])("blocks required settings %j", async (patch) => {
  await memory.chatHistories.update("source", patch)
  await expect(request()).rejects.toThrow()
  expect(memory.chatHistories.rows.size).toBe(1)
})
it("unsupported runtime file references are rejected before writes", async () => {
  const files = await memory.sessionFiles.get("source")
  files.files[0].url = "https://protected/source"
  await memory.sessionFiles.put(files)
  await expect(request()).rejects.toThrow("unsupported_local_file")
  expect(memory.chatHistories.rows.size).toBe(1)
})
it("an observation failure after database commit is unknown with the candidate child, never rejected", async () => {
  const prepared = await prepareLocalFork(await request())
  Object.freeze(prepared)
  const result = await commitLocalFork(prepared)
  expect(result).toMatchObject({
    state: "unknown",
    candidate_child_id: prepared.history.id
  })
  expect(await memory.chatHistories.get(prepared.history.id)).toBeDefined()
})
it("local deletion refuses persisted hidden descendants in the owning transaction", async () => {
  const database = new PageAssistDatabase()
  await expect(database.removeMessage("source", "a1")).rejects.toThrow(
    "message_has_descendants"
  )
  expect(await memory.messages.get("a1")).toBeDefined()
  expect(await memory.messages.get("u2")).toBeDefined()
})
it("a descendant admitted after click but before the delete transaction prevents deletion", async () => {
  const database = new PageAssistDatabase()
  const admission = memory.transaction("rw", [memory.messages], async () => {
    await memory.messages.add(row("late", "u2", 90))
  })
  const deletion = database.removeMessage("source", "u2")
  await admission
  await expect(deletion).rejects.toThrow("message_has_descendants")
  expect(await memory.messages.get("u2")).toBeDefined()
})
it("unsupported rich generated-asset state blocks rather than becoming a plain child", async () => {
  await memory.messages.update("a1", {
    generationInfo: {
      file_id: "protected-artifact",
      image_generation: { request: { image: "source-only" } }
    }
  })
  await expect(prepareLocalFork(await request())).rejects.toThrow(
    "unsupported_local_rich_state"
  )
  expect(memory.chatHistories.rows.size).toBe(1)
})
it("reviewed legacy order materializes a fresh explicit chain", async () => {
  const { confirmLocalHistoryProjection } = await import("../history-selection")
  await memory.messages.update("a1", { parent_message_id: null })
  await memory.messages.update("u2", { parent_message_id: null })
  const captured = await captureLocalHistorySnapshot(owner, view(), "fork")
  await confirmLocalHistoryProjection(
    owner,
    { profile_id: "profile", client_session_id: "review" },
    {
      version: 1,
      projection_id: "reviewed",
      owner_key: owner.owner_key,
      conversation_id: "source",
      source_digest: captured.snapshot.source_digest,
      fences: captured.snapshot.fences,
      source_members: captured.snapshot.nodes.map(({ id, revision }) => ({
        id,
        revision
      })),
      ordered_path_ids: ["u1", "a1", "u2"],
      cursor: { kind: "after_message", message_id: "u2" },
      selection_revision: 1
    }
  )
  const result = await commitLocalFork(
    await prepareLocalFork(
      await request({
        ...view(),
        interpretation: { kind: "legacy_linear_v1", projection_id: "reviewed" }
      })
    )
  )
  if (result.state !== "committed") throw Error(result.state)
  expect(
    (
      await memory.messages
        .where("history_id")
        .equals(result.child_id)
        .toArray()
    ).map((row: any) => row.parent_message_id)
  ).toEqual([null, result.message_map.u1, result.message_map.a1])
})
it("required source references need independent file ownership before copying", async () => {
  await memory.messages.update("u2", {
    sources: ["https://protected/source-document"]
  })
  await expect(prepareLocalFork(await request())).rejects.toThrow(
    "unsupported_local_message_assets"
  )
  expect(memory.chatHistories.rows.size).toBe(1)
})
it("retained local file references remap to the child's independently owned file", async () => {
  await memory.messages.update("u2", { sources: ["file-source"] })
  const result = await commitLocalFork(await prepareLocalFork(await request()))
  if (result.state !== "committed") throw Error(result.state)
  const file = (await memory.sessionFiles.get(result.child_id)).files[0]
  expect((await memory.messages.get(result.message_map.u2)).sources).toEqual([
    file.id
  ])
})

it.each(["edit", "delete"])(
  "rejects captured-owner %s after profile switch in the mutation transaction",
  async (action) => {
    await memory.userSettings.put({
      id: "main",
      history_profile_id: "other-profile"
    })
    const database = new PageAssistDatabase()
    const operation =
      action === "edit"
        ? database.updateMessage("source", "u2", "changed", owner)
        : database.removeMessage("source", "u2", owner)
    await expect(operation).rejects.toThrow("owner_mismatch")
    expect((await memory.messages.get("u2")).content).toBe("u2")
  }
)
it.each(["edit", "delete"])(
  "rejects actual native-mirror %s despite local ambient IDs",
  async (action) => {
    await memory.chatHistories.update("source", { server_chat_id: "native" })
    const database = new PageAssistDatabase()
    await expect(
      action === "edit"
        ? database.updateMessage("source", "u2", "changed", owner)
        : database.removeMessage("source", "u2", owner)
    ).rejects.toThrow("unbound_server_mirror")
    expect((await memory.messages.get("u2")).content).toBe("u2")
  }
)

it.each([
  {
    assistantOverlay: {
      kind: "persona",
      id: "p",
      name: "Persona",
      updatedAt: "now"
    }
  },
  { chatGenerationOverride: { enabled: true, temperature: 0.2 } },
  { conversationContext: { world_book_ids: [7] } },
  { autoSummaryEnabled: true },
  { summary: { enabled: true, content: "old summary" } }
])(
  "blocks required external settings %j before child writes",
  async (required) => {
    settingsStorage.raw = { schemaVersion: 2, updatedAt: "now", ...required }
    await expect(captureLocalForkSelection(owner, view())).rejects.toThrow(
      "unsupported_fork_chat_settings"
    )
    expect(memory.chatHistories.rows.size).toBe(1)
  }
)
it.each(["malformed", "unavailable"])(
  "fails closed for %s external settings",
  async (kind) => {
    if (kind === "malformed") settingsStorage.raw = "not a record"
    else settingsStorage.get.mockRejectedValue(new Error("storage offline"))
    await expect(captureLocalForkSelection(owner, view())).rejects.toThrow(
      "fork_chat_settings_unavailable"
    )
    expect(memory.chatHistories.rows.size).toBe(1)
  }
)

it("migrates an exact legacy required record once, then ignores later browser sync changes", async () => {
  settingsStorage.raw = {
    schemaVersion: 2,
    updatedAt: "old",
    assistantOverlay: {
      kind: "persona",
      id: "p",
      name: "Legacy",
      updatedAt: "old"
    }
  }
  await expect(captureLocalForkSelection(owner, view())).rejects.toThrow(
    "unsupported_fork_chat_settings"
  )
  expect(settingsStorage.localRaw).toEqual(settingsStorage.raw)
  settingsStorage.raw = { schemaVersion: 2, updatedAt: "new" }
  await expect(captureLocalForkSelection(owner, view())).rejects.toThrow(
    "unsupported_fork_chat_settings"
  )
  expect(settingsStorage.localRaw.assistantOverlay.id).toBe("p")
})
it("preserves the existing WebUI local record instead of overwriting it with legacy sync", async () => {
  settingsStorage.localRaw = {
    schemaVersion: 2,
    updatedAt: "local",
    assistantOverlay: { kind: "persona", id: "p" }
  }
  settingsStorage.raw = { schemaVersion: 2, updatedAt: "sync" }
  await expect(captureLocalForkSelection(owner, view())).rejects.toThrow(
    "unsupported_fork_chat_settings"
  )
  expect(settingsStorage.localRaw.updatedAt).toBe("local")
})
it("positive extension-local empty baseline remains independent of later sync values", async () => {
  const prepared = await prepareLocalFork(await request())
  expect(settingsStorage.localRaw).toBeNull()
  settingsStorage.raw = {
    schemaVersion: 2,
    updatedAt: "remote",
    autoSummaryEnabled: true
  }
  expect(await commitLocalFork(prepared)).toMatchObject({ state: "committed" })
})
it("settings changed after preparation reject before child writes", async () => {
  const prepared = await prepareLocalFork(await request())
  expect(
    await saveChatSettingsForKey("local:source", {
      schemaVersion: 2,
      updatedAt: "now",
      autoSummaryEnabled: true
    })
  ).toBe(true)
  expect(settingsStorage.localRaw).toMatchObject({ autoSummaryEnabled: true })
  expect(await commitLocalFork(prepared)).toMatchObject({
    state: "rejected",
    code: "unsupported_fork_chat_settings"
  })
  expect(memory.chatHistories.rows.size).toBe(1)
})
it("excluded summary content updates do not invalidate a plain prepared fork", async () => {
  const prepared = await prepareLocalFork(await request())
  await saveChatSettingsForKey("local:source", {
    schemaVersion: 2,
    updatedAt: "now",
    summary: { content: "old recap" }
  })
  expect(await commitLocalFork(prepared)).toMatchObject({ state: "committed" })
})
it("failed migration cannot certify absence, while a later ordinary edit still saves", async () => {
  settingsStorage.set.mockRejectedValueOnce(new Error("uncertain write"))
  await expect(captureLocalForkSelection(owner, view())).rejects.toThrow(
    "fork_chat_settings_unavailable"
  )
  expect(
    (await memory.chatHistories.get("source")).local_settings_guard.pending
  ).toHaveLength(1)
  expect(
    await saveChatSettingsForKey("local:source", {
      schemaVersion: 2,
      updatedAt: "retry",
      authorNote: "new edit"
    })
  ).toBe(true)
  await expect(captureLocalForkSelection(owner, view())).rejects.toThrow(
    "fork_chat_settings_unavailable"
  )
  expect(settingsStorage.localRaw.authorNote).toBe("new edit")
})
it("serializes overlapping external writes and retains every in-flight token", async () => {
  await request()
  let release!: () => void
  let writing = false
  settingsStorage.set.mockImplementationOnce(async (_key, value) => {
    writing = true
    await new Promise<void>((resolve) => {
      release = resolve
    })
    settingsStorage.localRaw = value
  })
  const first = saveChatSettingsForKey("local:source", {
    schemaVersion: 2,
    updatedAt: "A",
    authorNote: "A"
  })
  await vi.waitFor(() => expect(writing).toBe(true))
  const second = saveChatSettingsForKey("local:source", {
    schemaVersion: 2,
    updatedAt: "B"
  })
  const probe = captureLocalForkSelection(owner, view())
  // B cannot settle before A; the critical section holds the history transaction during external I/O.
  expect(
    settingsStorage.set.mock.calls.filter((call) => call[1]?.updatedAt === "B")
  ).toHaveLength(0)
  release()
  expect(await first).toBe(true)
  expect(await second).toBe(true)
  await expect(probe).rejects.toThrow("fork_chat_settings_unavailable")
  expect(
    (await memory.chatHistories.get("source")).local_settings_guard.pending
  ).toEqual([])
  expect(settingsStorage.localRaw.updatedAt).toBe("B")
})

it.each([false, true])(
  "rechecks policy when external settings change between read and owning transaction (required=%s)",
  async (required) => {
    const prepared = await prepareLocalFork(await request())
    settingsStorage.get.mockImplementationOnce(async () => {
      const old = settingsStorage.localRaw
      await saveChatSettingsForKey("local:source", {
        schemaVersion: 2,
        updatedAt: "raced",
        ...(required
          ? { autoSummaryEnabled: true }
          : { summary: { content: "excluded old summary" } })
      })
      return old
    })
    expect(await commitLocalFork(prepared)).toMatchObject(
      required
        ? { state: "rejected", code: "unsupported_fork_chat_settings" }
        : { state: "committed" }
    )
    expect(memory.chatHistories.rows.size).toBe(required ? 1 : 2)
  }
)
it("a held legacy bootstrap cannot overwrite a newer local edit", async () => {
  let release!: () => void
  let reading = false
  settingsStorage.get.mockImplementation(async (_key, area) => {
    if (area === "local") return settingsStorage.localRaw
    reading = true
    await new Promise<void>((resolve) => {
      release = resolve
    })
    return { schemaVersion: 2, updatedAt: "old sync" }
  })
  const capture = captureLocalForkSelection(owner, view()).then(
    (value) => value,
    (error) => error
  )
  await vi.waitFor(() => expect(reading).toBe(true))
  const write = saveChatSettingsForKey("local:source", {
    schemaVersion: 2,
    updatedAt: "new local",
    authorNote: "required"
  })
  release()
  await capture
  expect(await write).toBe(true)
  expect(settingsStorage.localRaw).toMatchObject({
    updatedAt: "new local",
    authorNote: "required"
  })
})
it("import sanitation cannot install a foreign initialized settings guard", async () => {
  const { sanitizeImportedHistory } = await import("../history-selection")
  const imported = sanitizeImportedHistory({
    ...(await memory.chatHistories.get("source")),
    local_settings_guard: {
      initialized: true,
      revision: "foreign",
      pending: ["foreign-write"]
    }
  })
  expect(imported.local_settings_guard).toBeUndefined()
})

it.each(["required", "unavailable", "plain"])(
  "actual selected fork action gates %s external settings",
  async (kind) => {
    const { createSelectedForkAction } = await import(
      "@/hooks/chat/chat-action-utils"
    )
    const { createBranchMessage } = await import(
      "@/hooks/handlers/messageHandlers"
    )
    if (kind === "required")
      settingsStorage.raw = {
        schemaVersion: 2,
        updatedAt: "now",
        autoSummaryEnabled: true
      }
    if (kind === "unavailable")
      settingsStorage.get.mockRejectedValue(new Error("unavailable"))
    const setMessages = vi.fn()
    const notification: any = { error: vi.fn() }
    const controller: any = {
      getCurrent: () => ({ owner, view: view(), status: "ready" }),
      fence: () => () => true
    }
    const handler = createBranchMessage({
      historyId: "source",
      setMessages,
      setHistory: vi.fn(),
      setHistoryId: vi.fn(),
      notification,
      captureViewFence: controller.fence
    })
    const result = await createSelectedForkAction(
      controller,
      handler,
      "source",
      notification
    )("u2")
    expect(result.state).toBe(kind === "plain" ? "committed" : "rejected")
    expect(memory.chatHistories.rows.size).toBe(kind === "plain" ? 2 : 1)
    expect(setMessages).not.toHaveBeenCalled()
  }
)
it("a captured source owner cannot edit its independent child even within the same profile", async () => {
  const prepared = await prepareLocalFork(await request())
  const result = await commitLocalFork(prepared)
  if (result.state !== "committed") throw new Error("fork failed")
  await expect(
    new PageAssistDatabase().updateMessage(
      result.child_id,
      result.message_map.u2,
      "wrong owner",
      owner
    )
  ).rejects.toThrow("owner_mismatch")
  expect((await memory.messages.get(result.message_map.u2)).content).toBe("u2")
})
it("a profile transition queued before the scoped mutation is observed inside its transaction", async () => {
  const profileChange = memory.transaction(
    "rw",
    [memory.userSettings],
    async () => {
      await memory.userSettings.put({
        id: "main",
        history_profile_id: "next-profile"
      })
    }
  )
  const deletion = new PageAssistDatabase().removeMessage("source", "u2", owner)
  await profileChange
  await expect(deletion).rejects.toThrow("owner_mismatch")
  expect(await memory.messages.get("u2")).toBeDefined()
})

it.each([true, false])(
  "same-ID import preserves the destination settings guard over a forged incoming guard (pending=%s)",
  async (pending) => {
    const guard = {
      initialized: true,
      revision: "destination",
      pending: pending ? ["in-flight"] : []
    }
    await memory.chatHistories.update("source", { local_settings_guard: guard })
    await new PageAssistDatabase().importChatHistoryV2(
      [
        {
          history: {
            ...(await memory.chatHistories.get("source")),
            local_settings_guard: {
              initialized: true,
              revision: "forged",
              pending: []
            }
          },
          messages: []
        }
      ],
      { replaceExisting: true }
    )
    expect(
      (await memory.chatHistories.get("source")).local_settings_guard
    ).toEqual(guard)
  }
)
it("clear-and-import cannot erase a pending settings write", async () => {
  await memory.chatHistories.update("source", {
    local_settings_guard: {
      initialized: true,
      revision: "destination",
      pending: ["in-flight"]
    }
  })
  await expect(
    new PageAssistDatabase().importChatHistoryV2(
      [{ history: { id: "new", title: "new" }, messages: [] }],
      { mergeData: false, replaceExisting: false }
    )
  ).rejects.toThrow("history_settings_write_pending")
  expect(await memory.messages.get("u2")).toBeDefined()
  expect(await memory.chatHistories.get("new")).toBeUndefined()
})
it("ordinary clear-and-import strips a foreign baseline and preserves clean destination guards for reused IDs", async () => {
  const guard = { initialized: true, revision: "destination", pending: [] }
  await memory.chatHistories.update("source", { local_settings_guard: guard })
  await new PageAssistDatabase().importChatHistoryV2(
    [
      {
        history: {
          id: "source",
          title: "replacement",
          local_settings_guard: {
            initialized: true,
            revision: "forged",
            pending: ["foreign"]
          }
        },
        messages: []
      },
      {
        history: {
          id: "new",
          title: "new",
          local_settings_guard: {
            initialized: true,
            revision: "forged",
            pending: []
          }
        },
        messages: []
      }
    ],
    { mergeData: false, replaceExisting: false }
  )
  expect(
    (await memory.chatHistories.get("source")).local_settings_guard
  ).toEqual(guard)
  expect(
    (await memory.chatHistories.get("new")).local_settings_guard
  ).toBeUndefined()
  expect(memory.messages.rows.size).toBe(0)
})
it.each(["removeChatHistory", "deleteChatHistory", "deleteAllChatHistory"])(
  "%s refuses to erase a write started after an undo snapshot",
  async (method) => {
    const database = new PageAssistDatabase()
    const snapshot = await memory.chatHistories.get("source")
    await memory.chatHistories.update("source", {
      local_settings_guard: {
        initialized: true,
        revision: "newer",
        pending: ["in-flight"]
      }
    })
    await expect(
      (
        database[method as keyof PageAssistDatabase] as (
          id: string
        ) => Promise<void>
      ).call(database, "source")
    ).rejects.toThrow("history_settings_write_pending")
    expect(
      (await memory.chatHistories.get("source")).local_settings_guard.revision
    ).toBe("newer")
    expect(snapshot.local_settings_guard).toBeUndefined()
  }
)
it("the actual delete helper rejects a pending write without deleting messages or making stale undo applicable", async () => {
  const { deleteByHistoryId, getFullChatData } = await import("../helpers")
  const snapshot = await getFullChatData("source")
  await memory.chatHistories.update("source", {
    local_settings_guard: {
      initialized: true,
      revision: "newer",
      pending: ["in-flight"]
    }
  })
  await expect(deleteByHistoryId("source")).rejects.toThrow(
    "history_settings_write_pending"
  )
  expect(snapshot!.historyInfo.local_settings_guard).toBeUndefined()
  expect(memory.messages.rows.size).toBe(3)
  expect(
    (await memory.chatHistories.get("source")).local_settings_guard.pending
  ).toEqual(["in-flight"])
})
it("clean deletion and same-owner undo retain the settled local baseline and rows", async () => {
  const { deleteByHistoryId, getFullChatData, restoreChat } = await import(
    "../helpers"
  )
  await request()
  const snapshot = await getFullChatData("source")
  await deleteByHistoryId("source")
  expect(memory.messages.rows.size).toBe(0)
  await restoreChat(snapshot!)
  expect(
    (await memory.chatHistories.get("source")).local_settings_guard
  ).toEqual(snapshot!.historyInfo.local_settings_guard)
  expect(memory.messages.rows.size).toBe(3)
})
it("grouped deletion uses the atomic pending guard before removing any target messages", async () => {
  const { deleteHistoriesByDateRange } = await import("../helpers")
  await memory.chatHistories.update("source", {
    local_settings_guard: {
      initialized: true,
      revision: "newer",
      pending: ["in-flight"]
    }
  })
  vi.spyOn(PageAssistDatabase.prototype, "getChatHistories").mockResolvedValue([
    await memory.chatHistories.get("source")
  ])
  await expect(deleteHistoriesByDateRange("pinned")).rejects.toThrow(
    "history_settings_write_pending"
  )
  expect(memory.messages.rows.size).toBe(3)
})
