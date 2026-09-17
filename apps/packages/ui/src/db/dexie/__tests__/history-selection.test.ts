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
            : (v.id ?? v.sessionId ?? v.history_id)
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
      bulkPut: async (vs: any[]) => {
        for (const v of vs) rows.set(key(v), structuredClone(v))
      },
      update: async (id: any, patch: any) => {
        const k = JSON.stringify(id)
        rows.set(k, { ...rows.get(k), ...structuredClone(patch) })
      },
      delete: async (id: any) => rows.delete(JSON.stringify(id)),
      where: (field: string) => ({
        equals: (v: any) => ({
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
vi.mock("../nickname", () => ({ getAllModelNicknames: vi.fn() }))
import * as history from "../history-selection"
import { PageAssistDatabase } from "../chat"
import { resolveHistorySelection } from "@/utils/history-selection"
import type { HistoryViewSelectionV1 } from "@/types/history-selection"
const row = (id: string, parent?: string | null) => ({
  id,
  history_id: "chat",
  content: id,
  name: "User",
  role: "user",
  createdAt: 1,
  ...(parent !== undefined ? { parent_message_id: parent } : {})
})
const view = (
  owner: any,
  cursor: any = { kind: "empty" },
  interpretation: any = { kind: "parent_graph_v1" }
): HistoryViewSelectionV1 => ({
  owner_key: owner.owner_key,
  conversation_id: "chat",
  view_session_id: "view",
  selection_revision: 1,
  cursor,
  interpretation
})
const bookmark = { profile_id: "profile", client_session_id: "session" }
const selection = (capture: any) => {
  const r = resolveHistorySelection(
    capture.snapshot,
    capture.view,
    "send",
    "prepared"
  )
  if (r.status !== "ready") throw new Error(r.status)
  return r.selection
}
beforeEach(async () => {
  Object.values(memory).forEach((t: any) => t.rows?.clear())
  await memory.userSettings.put({
    id: "main",
    user_id: "profile",
    theme: "kept"
  })
  await memory.chatHistories.put({
    id: "chat",
    title: "Chat",
    createdAt: 1,
    is_rag: false
  })
})
describe("local owner history", () => {
  it("atomically reuses the profile", async () => {
    await memory.userSettings.delete("main")
    const owners = await Promise.all([
      history.getLocalHistoryOwner("chat"),
      history.getLocalHistoryOwner("chat")
    ])
    expect(owners[0]).toMatchObject({
      kind: "local",
      conversation_id: "chat",
      profile_id: expect.any(String)
    })
    expect(owners[0]).toEqual(owners[1])
  })
  it("keeps explicit empty bookmarks independent across sessions and preserves settings", async () => {
    const owner = await history.getLocalHistoryOwner("chat")
    await history.saveHistoryBookmark(bookmark, view(owner))
    expect(
      (await history.loadHistoryBookmark(bookmark, owner))?.view.cursor
    ).toEqual({ kind: "empty" })
    expect(
      await history.loadHistoryBookmark(
        { ...bookmark, client_session_id: "other" },
        owner
      )
    ).toBeNull()
    expect((await memory.userSettings.get("main")).theme).toBe("kept")
  })
  it("does not locally own an old unbound server mirror", async () => {
    await memory.chatHistories.update("chat", { server_chat_id: "remote" })
    await expect(history.getLocalHistoryOwner("chat")).rejects.toMatchObject({
      code: "unbound_server_mirror"
    })
  })
  it("keeps two reviewed bases, explicit descendants and immutable replay after append", async () => {
    await memory.messages.bulkPut([row("a"), row("b")])
    const owner = await history.getLocalHistoryOwner("chat")
    const initial = await history.captureLocalHistorySnapshot(
      owner,
      view(owner),
      "send"
    )
    expect(initial.status).toBe("legacy_review_required")
    const confirms = ["a", "b"].map((id) => ({
      version: 1 as const,
      projection_id: `p-${id}`,
      owner_key: owner.owner_key,
      conversation_id: "chat",
      source_digest: initial.snapshot.source_digest,
      fences: initial.snapshot.fences,
      source_members: initial.snapshot.nodes.map(({ id, revision }) => ({
        id,
        revision
      })),
      ordered_path_ids: [id],
      cursor: { kind: "after_message" as const, message_id: id },
      selection_revision: 1
    }))
    for (const confirmation of confirms)
      await history.confirmLocalHistoryProjection(owner, bookmark, confirmation)
    const v = view(owner, confirms[0].cursor, {
      kind: "legacy_linear_v1",
      projection_id: "p-a"
    })
    const admission = await history.appendLocalSelectedUser(
      owner,
      selection(await history.captureLocalHistorySnapshot(owner, v, "send")),
      row("u")
    )
    await history.settleLocalAcceptedAssistant(owner, admission, {
      ...row("reply"),
      role: "assistant"
    })
    await expect(
      history.settleLocalAcceptedAssistant(owner, admission, {
        ...row("reply"),
        role: "assistant"
      })
    ).resolves.toMatchObject({ id: "reply" })
    expect(
      (
        await history.captureLocalHistorySnapshot(
          owner,
          view(
            owner,
            { kind: "after_message", message_id: "reply" },
            v.interpretation
          ),
          "send"
        )
      ).status
    ).toBe("captured")
    expect(
      (
        await history.captureLocalHistorySnapshot(
          owner,
          view(owner, confirms[1].cursor, {
            kind: "legacy_linear_v1",
            projection_id: "p-b"
          }),
          "send"
        )
      ).status
    ).toBe("captured")
    await expect(
      history.confirmLocalHistoryProjection(owner, bookmark, confirms[0])
    ).resolves.toMatchObject({ projection_id: "p-a" })
    await expect(
      history.confirmLocalHistoryProjection(owner, bookmark, {
        ...confirms[0],
        ordered_path_ids: ["b"]
      })
    ).rejects.toMatchObject({ code: "projection_id_conflict" })
    await memory.messages.delete("a")
    await expect(
      history.settleLocalAcceptedAssistant(owner, admission, {
        ...row("late"),
        role: "assistant"
      })
    ).resolves.toMatchObject({ parent_message_id: "u" })
  })
  it("settles empty input with null parent and rejects accepted asset edits", async () => {
    const owner = await history.getLocalHistoryOwner("chat")
    const admission = await history.appendLocalSelectedUser(
      owner,
      selection(
        await history.captureLocalHistorySnapshot(owner, view(owner), "send")
      ),
      { ...row("u"), images: ["data:image/png;base64,YQ=="] }
    )
    expect((await memory.messages.get("u")).parent_message_id).toBeNull()
    await history.settleLocalAcceptedAssistant(owner, admission, {
      ...row("reply"),
      role: "assistant"
    })
    await memory.messages.update("u", {
      images: ["data:image/png;base64,Yg=="]
    })
    await expect(
      history.settleLocalAcceptedAssistant(owner, admission, {
        ...row("new-reply"),
        role: "assistant"
      })
    ).rejects.toMatchObject({ code: "stale_parent" })
  })
  it("reports deleted cursor targets as stale and hashes stored files", async () => {
    await memory.messages.put(row("a", null))
    const owner = await history.getLocalHistoryOwner("chat")
    const v = view(owner, { kind: "after_message", message_id: "a" })
    const captured = await history.captureLocalHistorySnapshot(owner, v, "send")
    await memory.sessionFiles.put({
      sessionId: "chat",
      files: [{ id: "f", content: "new" }],
      retrievalEnabled: true
    })
    expect(
      (await history.captureLocalHistorySnapshot(owner, v, "send")).snapshot
        .storage_context_digest
    ).not.toBe(captured.snapshot.storage_context_digest)
    await memory.messages.delete("a")
    expect(
      (await history.captureLocalHistorySnapshot(owner, v, "send")).status
    ).toBe("stale_selection")
  })
  it("strips forged authority at imports including replaceExisting and ordinary adds", async () => {
    const forged = {
      ...row("forged"),
      history_admission: { input_message_id: "forged" },
      history_provenance: { legacy_projection_id: "fake" },
      tldw_history_admission_v1: { version: 1 },
      serverMessageId: "remote"
    }
    const database = new PageAssistDatabase()
    await database.importChatHistoryV2(
      [
        {
          history: {
            id: "chat",
            title: "Readable",
            server_chat_id: "remote",
            server_scope_key: "forged",
            local_owner_key: "forged"
          },
          messages: [forged]
        }
      ],
      { replaceExisting: true }
    )
    expect(await memory.messages.get("forged")).not.toHaveProperty(
      "history_admission"
    )
    expect(await memory.messages.get("forged")).not.toHaveProperty(
      "serverMessageId"
    )
    expect(await memory.chatHistories.get("chat")).not.toHaveProperty(
      "server_scope_key"
    )
    await database.addMessage({ ...forged, id: "ordinary" } as any)
    expect(await memory.messages.get("ordinary")).not.toHaveProperty(
      "history_provenance"
    )
  })
})

describe("local immutable intent fences", () => {
  it("keeps late legacy empty-path settlement/retry independent of removed old source", async () => {
    await memory.messages.bulkPut([row("a"), row("b")])
    const owner = await history.getLocalHistoryOwner("chat")
    const initial = await history.captureLocalHistorySnapshot(
      owner,
      view(owner),
      "send"
    )
    const confirmation = {
      version: 1 as const,
      projection_id: "empty-base",
      owner_key: owner.owner_key,
      conversation_id: "chat",
      source_digest: initial.snapshot.source_digest,
      fences: initial.snapshot.fences,
      source_members: initial.snapshot.nodes.map(({ id, revision }) => ({
        id,
        revision
      })),
      ordered_path_ids: ["a", "b"],
      cursor: { kind: "before_message" as const, message_id: "a" },
      selection_revision: 1
    }
    await history.confirmLocalHistoryProjection(owner, bookmark, confirmation)
    const chosen = selection(
      await history.captureLocalHistorySnapshot(
        owner,
        view(owner, confirmation.cursor, {
          kind: "legacy_linear_v1",
          projection_id: "empty-base"
        }),
        "send"
      )
    )
    const input = row("u")
    const admission = await history.appendLocalSelectedUser(
      owner,
      chosen,
      input
    )
    await memory.messages.delete("a")
    await expect(
      history.appendLocalSelectedUser(owner, chosen, input)
    ).resolves.toEqual(admission)
    const result = { ...row("reply"), role: "assistant" }
    await history.settleLocalAcceptedAssistant(owner, admission, result)
    await expect(
      history.settleLocalAcceptedAssistant(owner, admission, result)
    ).resolves.toMatchObject({ parent_message_id: "u" })
    const stale = await history.captureLocalHistorySnapshot(
      owner,
      view(owner, confirmation.cursor, {
        kind: "legacy_linear_v1",
        projection_id: "empty-base"
      }),
      "send"
    )
    expect(stale.status).toBe("stale_selection")
    expect(stale.snapshot.nodes.map((n) => n.id)).toEqual(["b", "reply", "u"])
  })
  it("does not replace a new view with a late remote confirmation at the same revision", async () => {
    const owner = await history.getLocalHistoryOwner("chat")
    const firstView = view(owner)
    await history.saveHistoryBookmark(bookmark, firstView)
    const pending = {
      version: 1 as const,
      projection_id: "p",
      owner_key: owner.owner_key,
      conversation_id: "chat",
      source_digest: "s",
      fences: { conversation: "c", history: "h", settings: "s" },
      source_members: [],
      ordered_path_ids: [],
      cursor: { kind: "empty" as const },
      selection_revision: 1
    }
    await history.savePendingHistoryConfirmation(bookmark, firstView, pending)
    const nextView = { ...firstView, view_session_id: "reopened" }
    await history.saveHistoryBookmark(bookmark, nextView)
    await history.acknowledgeHistoryConfirmation(bookmark, {
      ...pending,
      projection_digest: "p",
      created_at: "now"
    })
    expect((await history.loadHistoryBookmark(bookmark, owner))?.view).toEqual(
      nextView
    )
  })
  it("returns the complete deterministic 20,001-row legacy manifest", async () => {
    await memory.messages.bulkPut(
      Array.from({ length: 20001 }, (_, i) =>
        row(`m${String(20000 - i).padStart(5, "0")}`)
      )
    )
    const owner = await history.getLocalHistoryOwner("chat")
    const captured = await history.captureLocalHistorySnapshot(
      owner,
      view(owner),
      "send"
    )
    expect(captured.snapshot.nodes).toHaveLength(20001)
    expect(captured.snapshot.nodes[0].id).toBe("m00000")
    expect(captured.snapshot.nodes.at(-1)?.id).toBe("m20000")
  })
  it("rejects scope invalidation without a committed input", async () => {
    const owner = await history.getLocalHistoryOwner("chat")
    const chosen = selection(
      await history.captureLocalHistorySnapshot(owner, view(owner), "send")
    )
    await expect(
      history.appendLocalSelectedUser(owner, chosen, row("u"), {
        validate_lease: () => false
      })
    ).rejects.toMatchObject({ code: "request_config_scope_changed" })
    expect(await memory.messages.get("u")).toBeUndefined()
  })
})

it("binds local admission input before an asynchronous owner read", async () => {
  const owner = await history.getLocalHistoryOwner("chat")
  const chosen = selection(
    await history.captureLocalHistorySnapshot(owner, view(owner), "send")
  )
  const input = row("u")
  const pending = history.appendLocalSelectedUser(owner, chosen, input)
  input.content = "changed after dispatch"
  await pending
  expect((await memory.messages.get("u")).content).toBe("u")
})
it("ordinary history adds cannot forge a server binding", async () => {
  await new PageAssistDatabase().addChatHistory({
    id: "forged-history",
    title: "Imported",
    createdAt: 1,
    is_rag: false,
    server_chat_id: "remote",
    server_scope_key: "forged"
  })
  expect(await memory.chatHistories.get("forged-history")).not.toHaveProperty(
    "server_scope_key"
  )
})

it("preserves local historical tools and composer metadata with image order", async () => {
  await memory.messages.put({
    ...row("tool", null),
    role: "tool",
    images: ["data:image/png;base64,YQ==", "data:image/png;base64,Yg=="],
    metadataExtra: {
      tool_call_id: "call-1",
      tool_calls: [{ id: "call-1", function: { name: "search" } }]
    },
    messageType: "tool",
    documents: [{ id: "doc", content: "inline" }]
  })
  const owner = await history.getLocalHistoryOwner("chat")
  const captured = await history.captureLocalHistorySnapshot(
    owner,
    view(owner, { kind: "after_message", message_id: "tool" }),
    "send"
  )
  if (captured.status !== "captured") throw Error(captured.status)
  expect(captured.rows[0].role).toBe("tool")
  expect(captured.selected_content[0]).toMatchObject({
    images: ["data:image/png;base64,YQ==", "data:image/png;base64,Yg=="],
    tool_calls: [{ id: "call-1" }],
    extra_metadata: {
      tool_call_id: "call-1",
      sender_role: "tool",
      local_history: {
        messageType: "tool",
        documents: [{ id: "doc", content: "inline" }]
      }
    }
  })
})

it("rolls back projection if its bookmark write fails (transaction double)", async () => {
  await memory.messages.bulkPut([row("a"), row("b")])
  const owner = await history.getLocalHistoryOwner("chat")
  const captured = await history.captureLocalHistorySnapshot(
    owner,
    view(owner),
    "send"
  )
  const confirmation = {
    version: 1 as const,
    projection_id: "atomic",
    owner_key: owner.owner_key,
    conversation_id: "chat",
    source_digest: captured.snapshot.source_digest,
    fences: captured.snapshot.fences,
    source_members: captured.snapshot.nodes.map(({ id, revision }) => ({
      id,
      revision
    })),
    ordered_path_ids: ["a"],
    cursor: { kind: "after_message" as const, message_id: "a" },
    selection_revision: 1
  }
  const put = vi
    .spyOn(memory.historySelections, "put")
    .mockRejectedValueOnce(new Error("storage failure"))
  await expect(
    history.confirmLocalHistoryProjection(owner, bookmark, confirmation)
  ).rejects.toThrow("storage failure")
  expect(
    await memory.historyProjections.get([owner.owner_key, "chat", "atomic"])
  ).toBeUndefined()
  put.mockRestore()
  expect(memory.transaction.mock.calls.at(-1)?.[1]).toEqual(
    expect.arrayContaining([
      memory.historyProjections,
      memory.historySelections,
      memory.sessionFiles,
      memory.userSettings,
      memory.messages,
      memory.chatHistories
    ])
  )
})
it("rejects retained-row, stored-file and settings changes before an unaccepted append", async () => {
  await memory.messages.put(row("a", null))
  const owner = await history.getLocalHistoryOwner("chat")
  const v = view(owner, { kind: "after_message", message_id: "a" })
  const chosen = selection(
    await history.captureLocalHistorySnapshot(owner, v, "send")
  )
  await memory.messages.update("a", { content: "edited" })
  await expect(
    history.appendLocalSelectedUser(owner, chosen, row("u"))
  ).rejects.toMatchObject({ code: "stale_selection" })
  const next = selection(
    await history.captureLocalHistorySnapshot(owner, v, "send")
  )
  await memory.sessionFiles.put({
    sessionId: "chat",
    files: [{ id: "file", content: "changed" }],
    retrievalEnabled: true
  })
  await expect(
    history.appendLocalSelectedUser(owner, next, row("u"))
  ).rejects.toMatchObject({ code: "stale_selection" })
  const last = selection(
    await history.captureLocalHistorySnapshot(owner, v, "send")
  )
  await memory.chatHistories.update("chat", {
    last_used_prompt: { prompt_content: "new inline prompt" }
  })
  await expect(
    history.appendLocalSelectedUser(owner, last, row("u"))
  ).rejects.toMatchObject({ code: "stale_selection" })
  expect(await memory.messages.get("u")).toBeUndefined()
})

it("keeps explicit admitted branches on an unambiguous old root usable", async () => {
  await memory.messages.put(row("root"))
  const owner = await history.getLocalHistoryOwner("chat")
  const v = view(owner, { kind: "after_message", message_id: "root" })
  const chosen = selection(
    await history.captureLocalHistorySnapshot(owner, v, "send")
  )
  await history.appendLocalSelectedUser(owner, chosen, row("left"))
  await history.appendLocalSelectedUser(owner, chosen, row("right"))
  const branch = await history.captureLocalHistorySnapshot(
    owner,
    view(owner, { kind: "after_message", message_id: "left" }),
    "send"
  )
  expect(branch.status).toBe("captured")
  if (branch.status === "captured")
    expect(branch.rows.map((n) => n.id)).toEqual(["root", "left"])
})

it("settles an unchanged accepted input after SessionFiles and retrieval context change", async () => {
  const owner = await history.getLocalHistoryOwner("chat")
  await memory.sessionFiles.put({
    sessionId: "chat",
    files: [{ id: "file", content: "original" }],
    retrievalEnabled: true
  })
  const chosen = selection(
    await history.captureLocalHistorySnapshot(owner, view(owner), "send")
  )
  const admission = await history.appendLocalSelectedUser(owner, chosen, {
    ...row("u"),
    images: ["data:image/png;base64,YQ=="]
  })
  await memory.sessionFiles.put({
    sessionId: "chat",
    files: [{ id: "file", content: "changed" }],
    retrievalEnabled: false
  })
  await expect(
    history.settleLocalAcceptedAssistant(owner, admission, {
      ...row("reply"),
      role: "assistant"
    })
  ).resolves.toMatchObject({ parent_message_id: "u" })
})
it.each([
  { images: ["https://example.test/mutable.png"] },
  {
    documents: [
      { type: "tab" as const, url: "https://example.test/mutable", tabId: 1 }
    ]
  }
])("gates unfenced accepted input assets %o", async (assets) => {
  const owner = await history.getLocalHistoryOwner("chat")
  const chosen = selection(
    await history.captureLocalHistorySnapshot(owner, view(owner), "send")
  )
  await expect(
    history.appendLocalSelectedUser(owner, chosen, { ...row("u"), ...assets })
  ).rejects.toMatchObject({ code: "unsupported_local_message_assets" })
  expect(await memory.messages.get("u")).toBeUndefined()
})

it("marks selected remote image dependencies unavailable without omitting them", async () => {
  await memory.messages.put({
    ...row("a", null),
    images: ["https://example.test/mutable.png"]
  })
  const owner = await history.getLocalHistoryOwner("chat")
  const captured = await history.captureLocalHistorySnapshot(
    owner,
    view(owner, { kind: "after_message", message_id: "a" }),
    "send"
  )
  expect(captured.status).toBe("unsupported_history_capability")
  expect(captured.snapshot.nodes).toHaveLength(1)
})

it("gates an unfenced prompt-library reference while supporting an inline saved prompt", async () => {
  const owner = await history.getLocalHistoryOwner("chat")
  await memory.chatHistories.update("chat", {
    last_used_prompt: { prompt_id: "library-id" }
  })
  expect(
    (await history.captureLocalHistorySnapshot(owner, view(owner), "send"))
      .status
  ).toBe("unsupported_history_capability")
  await memory.chatHistories.update("chat", {
    last_used_prompt: {
      prompt_id: "library-id",
      prompt_content: "saved inline"
    }
  })
  expect(
    (await history.captureLocalHistorySnapshot(owner, view(owner), "send"))
      .status
  ).toBe("captured")
})

it("treats old empty composer image placeholders as no asset", async () => {
  await memory.messages.put({ ...row("a", null), images: [""] })
  const owner = await history.getLocalHistoryOwner("chat")
  const captured = await history.captureLocalHistorySnapshot(
    owner,
    view(owner, { kind: "after_message", message_id: "a" }),
    "send"
  )
  expect(captured.status).toBe("captured")
  if (captured.status === "captured")
    expect(captured.selected_content[0].images).toEqual([])
})

it("loads a historical singular inline image without dropping it", async () => {
  await memory.messages.put({
    ...row("a", null),
    image: "data:image/png;base64,YQ=="
  })
  const owner = await history.getLocalHistoryOwner("chat")
  const captured = await history.captureLocalHistorySnapshot(
    owner,
    view(owner, { kind: "after_message", message_id: "a" }),
    "send"
  )
  if (captured.status !== "captured") throw Error(captured.status)
  expect(captured.selected_content[0].images).toEqual([
    "data:image/png;base64,YQ=="
  ])
})
