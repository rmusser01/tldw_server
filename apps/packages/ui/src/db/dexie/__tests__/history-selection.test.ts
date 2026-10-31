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
    "historyProjections",
    "searchTokens"
  ]) {
    const rows = new Map<string, any>()
    const key = (v: any) =>
      JSON.stringify(
        name === "historySelections"
          ? [v.profile_id, v.client_session_id, v.owner_key, v.conversation_id]
          : name === "historyProjections"
            ? [v.owner_key, v.conversation_id, v.projection_id]
          : name === "searchTokens"
            ? [v.token, v.history_id]
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
const remote = vi.hoisted(() => ({ capture: vi.fn(), confirm: vi.fn() }))
vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    captureHistorySelection: remote.capture,
    confirmHistoryProjection: remote.confirm
  }
}))
import * as service from "@/services/chat-history-selection"
import * as history from "../history-selection"
import { PageAssistDatabase } from "../chat"
import { keepHistoryTurnRecovery } from "@/services/history-turn-keep"
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
  it("Regenerate recovery replays the exact rich local answer after a lost acknowledgement", async () => {
    await memory.messages.put(row("question"))
    const owner = await history.getLocalHistoryOwner("chat")
    const origin = view(owner, { kind: "after_message", message_id: "question" })
    const chosen = selection(await history.captureLocalHistorySnapshot(owner, origin, "send"))
    const reply: Parameters<typeof history.settleLocalSelectedExistingInput>[2] = {
      ...row("alternative"), role: "assistant", name: "Assistant",
      content: "another answer", parent_message_id: "question", createdAt: 2,
      modelId: "chosen-model", metadataExtra: { finish_reason: "stop" }
    }
    await history.settleLocalSelectedExistingInput(owner, chosen, reply)
    const storedBefore = await memory.messages.get("alternative")
    const scope = { profile_id: owner.profile_id, client_session_id: "session" }
    const turn: Parameters<typeof history.saveHistoryTurnRecovery>[2] = {
      operation_id: "lost-ack", origin_view: origin,
      selection_digest: chosen.selection_digest,
      request_context_digest: chosen.request_context_digest,
      owner_key: owner.owner_key, conversation_id: "chat",
      input_id: "question", assistant_id: "alternative", created_at: 2,
      input_text: "question", input_images: [], result_text: "another answer",
      state: "generated_unsaved", outcome: "complete",
      existing_selection: chosen, settlement_payload: reply
    }
    await history.saveHistoryTurnRecovery(scope, origin, turn)
    const [entry] = await history.loadHistoryTurnRecoveries(scope, origin)
    const choose = vi.fn(async () => true)
    const controller = {
      fence: () => () => true,
      getCurrent: () => ({ status: "ready", owner, view: origin, bookmarkScope: scope }),
      choose, refreshRecovery: vi.fn(async () => {})
    } as unknown as Parameters<typeof keepHistoryTurnRecovery>[0]
    await expect(keepHistoryTurnRecovery(controller, entry)).resolves.toMatchObject({
      status: "kept", resultId: "alternative", settledByKeep: true
    })
    expect(await memory.messages.get("alternative")).toEqual(storedBefore)
    expect(choose).toHaveBeenCalledWith({ kind: "after_message", message_id: "alternative" })
    expect(remote.capture).not.toHaveBeenCalled()
  })

  it("settles an existing question as a sibling and replays only its proven answer", async () => {
    await memory.messages.bulkPut([
      row("earlier"),
      { ...row("earlier-answer", "earlier"), role: "assistant" },
      row("question", "earlier-answer"),
      { ...row("original-answer", "question"), role: "assistant" }
    ])
    const owner = await history.getLocalHistoryOwner("chat")
    const chosen = selection(await history.captureLocalHistorySnapshot(
      owner, view(owner, { kind: "after_message", message_id: "question" }), "send"
    ))
    const originalQuestion = await memory.messages.get("question")
    const original = await memory.messages.get("original-answer")
    const reply = { ...row("alternative"), role: "assistant", content: "another answer" }
    await expect(history.settleLocalSelectedExistingInput(owner, chosen, reply))
      .resolves.toMatchObject({ id: "alternative", parent_message_id: "question" })
    expect(await memory.messages.get("original-answer")).toEqual(original)
    expect(await memory.messages.get("question")).toEqual(originalQuestion)
    await memory.messages.update("earlier", { content: "edited ancestor" })
    await expect(history.settleLocalSelectedExistingInput(owner, chosen,
      { ...reply, id: "stale-alternative" })).rejects.toMatchObject({ code: "stale_selection" })
    await expect(history.settleLocalSelectedExistingInput(owner, chosen, reply))
      .resolves.toMatchObject({ id: "alternative" })
    await memory.messages.update("alternative", { content: "edited answer" })
    await expect(history.settleLocalSelectedExistingInput(owner, chosen, reply))
      .rejects.toMatchObject({ code: "message_id_conflict" })
  })

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

// Real bookmark helpers over the transaction orchestration double above.
describe("native projection pending resolution", () => {
  beforeEach(async () => {
    await history.ensureLocalProfileId()
  })
  const owner = () => ({
    kind: "native" as const,
    conversation_id: "chat",
    request_scope: {
      config: {
        serverUrl: "https://server.test",
        authMode: "multi-user" as const
      },
      userId: "alice"
    },
    validate_lease: () => true
  })
  const initialView = view({ owner_key: "native-key" })
  const intent = {
    version: 1 as const,
    owner_key: "native-key",
    conversation_id: "chat",
    projection_id: "old",
    source_digest: "old-source",
    fences: { conversation: "1", history: "1", settings: "1" },
    source_members: [],
    ordered_path_ids: [],
    cursor: { kind: "empty" as const },
    selection_revision: 1
  }
  const prepare = async () => {
    const native = owner()
    remote.capture.mockResolvedValue({
      status: "captured",
      snapshot: {
        version: 1,
        owner_key: "native-key",
        conversation_id: "chat",
        fences: intent.fences,
        nodes: [],
        source_digest: "old-source",
        interpretation_status: { kind: "parent_graph_v1" },
        storage_context_digest: "storage"
      },
      rows: [],
      selected_content: [],
      view: initialView,
      purpose: "send",
      storage_context_digest: "storage"
    })
    await service.captureHistorySnapshot(native, initialView, "send")
    remote.confirm.mockReset()
    return native
  }
  it("allows fresh review after definitive stale-source rejection in the same session", async () => {
    const native = await prepare()
    const rejection = Object.assign(new Error("stale source"), {
      status: 409,
      details: { detail: { status: "stale_selection", code: "stale_source" } }
    })
    remote.confirm.mockRejectedValueOnce(rejection)
    await expect(
      service.confirmLegacyHistoryProjection(
        native,
        bookmark,
        intent,
        initialView
      )
    ).rejects.toBe(rejection)
    expect(
      (await history.loadHistoryBookmark(bookmark, initialView))
        ?.pending_confirmation
    ).toBeUndefined()
    const fresh = {
      ...intent,
      projection_id: "fresh",
      source_digest: "fresh-source"
    }
    remote.confirm.mockResolvedValueOnce({
      ...fresh,
      projection_digest: "digest",
      created_at: "now"
    })
    await service.confirmLegacyHistoryProjection(
      native,
      bookmark,
      fresh,
      initialView
    )
    expect(
      (await history.loadHistoryBookmark(bookmark, initialView))?.view
        .interpretation
    ).toEqual({ kind: "legacy_linear_v1", projection_id: "fresh" })
  })
  it.each([
    new Error("network"),
    new DOMException("aborted", "AbortError"),
    Object.assign(new Error("unknown conflict"), {
      status: 409,
      details: { detail: { status: "stale_selection", code: "unknown" } }
    }),
    Object.assign(new Error("scope"), {
      status: 412,
      details: { detail: { code: "request_config_scope_changed" } }
    }),
    Object.assign(new Error("server"), {
      status: 500,
      details: { detail: { status: "stale_selection", code: "stale_source" } }
    })
  ])(
    "preserves uncertain outcome %s and requires matching replay",
    async (rejection) => {
      const native = await prepare()
      remote.confirm.mockRejectedValueOnce(rejection)
      await expect(
        service.confirmLegacyHistoryProjection(
          native,
          bookmark,
          intent,
          initialView
        )
      ).rejects.toBe(rejection)
      await expect(
        service.confirmLegacyHistoryProjection(
          native,
          bookmark,
          { ...intent, projection_id: "fresh" },
          initialView
        )
      ).rejects.toMatchObject({ code: "pending_confirmation_conflict" })
      expect(
        (await history.loadHistoryBookmark(bookmark, initialView))
          ?.pending_confirmation
      ).toEqual(intent)
      remote.confirm.mockResolvedValueOnce({
        ...intent,
        projection_digest: "digest",
        created_at: "now"
      })
      await service.confirmLegacyHistoryProjection(
        native,
        bookmark,
        intent,
        initialView
      )
      expect(remote.confirm).toHaveBeenCalledTimes(2)
    }
  )
  it("resolves a rejected matching replay after reopening without changing its newer cursor", async () => {
    const native = await prepare()
    remote.confirm.mockRejectedValueOnce(new Error("network"))
    await expect(
      service.confirmLegacyHistoryProjection(
        native,
        bookmark,
        intent,
        initialView
      )
    ).rejects.toThrow("network")
    const reopened = {
      ...initialView,
      view_session_id: "reopened",
      cursor: { kind: "after_message" as const, message_id: "new" }
    }
    await history.saveHistoryBookmark(bookmark, reopened)
    const rejection = Object.assign(new Error("stale source"), {
      status: 409,
      details: { detail: { status: "stale_selection", code: "stale_source" } }
    })
    remote.confirm.mockRejectedValueOnce(rejection)
    await expect(
      service.confirmLegacyHistoryProjection(native, bookmark, intent, reopened)
    ).rejects.toBe(rejection)
    const stored = await history.loadHistoryBookmark(bookmark, initialView)
    expect(stored?.pending_confirmation).toBeUndefined()
    expect(stored?.view).toEqual(reopened)
    const fresh = {
      ...intent,
      projection_id: "fresh",
      source_digest: "fresh-source"
    }
    await history.savePendingHistoryConfirmation(bookmark, reopened, fresh)
    expect(
      await history.rejectPendingHistoryConfirmation(bookmark, reopened, intent)
    ).toBe(false)
    expect(
      (await history.loadHistoryBookmark(bookmark, initialView))
        ?.pending_confirmation
    ).toEqual(fresh)
  })
  it("retains an earlier uncertain intent if a matching replay fails before dispatch", async () => {
    const native = await prepare()
    remote.confirm.mockRejectedValueOnce(new Error("network"))
    await expect(
      service.confirmLegacyHistoryProjection(
        native,
        bookmark,
        intent,
        initialView
      )
    ).rejects.toThrow("network")
    let checks = 0
    native.validate_lease = () => ++checks === 1
    await expect(
      service.confirmLegacyHistoryProjection(
        native,
        bookmark,
        intent,
        initialView
      )
    ).rejects.toMatchObject({ code: "request_config_scope_changed" })
    expect(remote.confirm).toHaveBeenCalledOnce()
    expect(
      (await history.loadHistoryBookmark(bookmark, initialView))
        ?.pending_confirmation
    ).toEqual(intent)
  })
  it("does not erase a concurrent replay when the creating attempt fails before dispatch", async () => {
    const native = await prepare()
    let releaseCreator!: () => void
    let creatorSaved!: () => void
    let replayStarted!: () => void
    let finishReplay!: (projection: unknown) => void
    const creatorGate = new Promise<void>((resolve) => {
      releaseCreator = resolve
    })
    const saved = new Promise<void>((resolve) => {
      creatorSaved = resolve
    })
    const started = new Promise<void>((resolve) => {
      replayStarted = resolve
    })
    const original = memory.transaction.getMockImplementation()
    memory.transaction.mockImplementationOnce(async (...args: any[]) => {
      const result = await original(...args)
      creatorSaved()
      await creatorGate
      return result
    })
    const abort = new AbortController()
    const creator = service.confirmLegacyHistoryProjection(
      native,
      bookmark,
      intent,
      initialView,
      abort.signal
    )
    const creatorRejected = expect(creator).rejects.toThrow()
    await saved
    remote.confirm.mockImplementationOnce(() => {
      replayStarted()
      return new Promise((resolve) => {
        finishReplay = resolve
      })
    })
    const replay = service.confirmLegacyHistoryProjection(
      native,
      bookmark,
      intent,
      initialView
    )
    await started
    abort.abort()
    releaseCreator()
    await creatorRejected
    expect(
      (await history.loadHistoryBookmark(bookmark, initialView))
        ?.pending_confirmation
    ).toEqual(intent)
    finishReplay({ ...intent, projection_digest: "digest", created_at: "now" })
    await replay
    expect(
      (await history.loadHistoryBookmark(bookmark, initialView))
        ?.pending_confirmation
    ).toBeUndefined()
  })
  it.each(["new", "legacy-unknown"])(
    "handles pre-dispatch lease failure conservatively for %s pending state",
    async (kind) => {
      const native = await prepare()
      if (kind === "legacy-unknown") {
        await history.savePendingHistoryConfirmation(
          bookmark,
          initialView,
          intent
        )
        const old = await history.loadHistoryBookmark(bookmark, initialView)
        delete old!.pending_dispatch_started
        await memory.historySelections.put(old)
      }
      let checks = 0
      native.validate_lease = () => ++checks === 1
      await expect(
        service.confirmLegacyHistoryProjection(
          native,
          bookmark,
          intent,
          initialView
        )
      ).rejects.toMatchObject({ code: "request_config_scope_changed" })
      expect(remote.confirm).not.toHaveBeenCalled()
      expect(
        (await history.loadHistoryBookmark(bookmark, initialView))
          ?.pending_confirmation
      ).toEqual(kind === "new" ? undefined : intent)
    }
  )
  it("clears only the exact rejected pending identity and keeps a newer cursor", async () => {
    await history.savePendingHistoryConfirmation(bookmark, initialView, intent)
    const newer = {
      ...initialView,
      view_session_id: "reopened",
      selection_revision: 2,
      cursor: { kind: "after_message" as const, message_id: "new" }
    }
    await history.saveHistoryBookmark(bookmark, newer)
    expect(
      await history.rejectPendingHistoryConfirmation(bookmark, newer, intent)
    ).toBe(false)
    expect(
      await history.rejectPendingHistoryConfirmation(bookmark, initialView, {
        ...intent,
        source_digest: "other"
      })
    ).toBe(false)
    expect(
      (await history.loadHistoryBookmark(bookmark, initialView))
        ?.pending_confirmation
    ).toEqual(intent)
    expect(
      await history.rejectPendingHistoryConfirmation(
        bookmark,
        initialView,
        intent
      )
    ).toBe(true)
    expect(
      (await history.loadHistoryBookmark(bookmark, initialView))?.view
    ).toEqual(newer)
  })
})

it("keeps pending sends outside ancestry and restores them across view identities", async () => {
  const owner = await history.getLocalHistoryOwner("chat")
  const origin = view(owner)
  const pending: any = {
    operation_id: "operation",
    owner_key: owner.owner_key,
    conversation_id: "chat",
    input_id: "input",
    assistant_id: "result",
    created_at: 1,
    input_text: "question",
    input_images: [],
    result_text: "",
    state: "dispatching",
    origin_view: origin,
    selection_digest: "selected",
    request_context_digest: "prepared"
  }
  await history.saveHistoryTurnRecovery(bookmark, origin, pending)
  await history.saveHistoryBookmark(bookmark, {
    ...origin,
    selection_revision: 2
  })
  const recovered = await history.loadHistoryTurnRecoveries(
    { ...bookmark, client_session_id: "reopened" },
    owner
  )
  expect(recovered).toHaveLength(1)
  expect(recovered[0].turn).toMatchObject({
    operation_id: "operation",
    input_text: "question"
  })
  const captured = await history.captureLocalHistorySnapshot(
    owner,
    origin,
    "send"
  )
  expect(captured.snapshot.nodes).toEqual([])
  expect(
    await history.loadHistoryTurnRecoveries(bookmark, {
      ...owner,
      owner_key: "other-account"
    })
  ).toEqual([])
  await history.dismissHistoryTurnRecovery(
    recovered[0].scope,
    owner,
    "operation"
  )
  expect(await history.loadHistoryTurnRecoveries(bookmark, owner)).toEqual([])
})

it("pending turn updates preserve confirmation and reject changed immutable intent", async () => {
  const owner = await history.getLocalHistoryOwner("chat")
  const origin = view(owner)
  const pending: any = {
    operation_id: "operation",
    owner_key: owner.owner_key,
    conversation_id: "chat",
    input_id: "input",
    assistant_id: "result",
    created_at: 1,
    input_text: "question",
    input_images: [],
    result_text: "",
    state: "dispatching",
    origin_view: origin,
    selection_digest: "selected",
    request_context_digest: "prepared"
  }
  await history.saveHistoryTurnRecovery(bookmark, origin, pending)
  const confirmation: any = {
    version: 1,
    projection_id: "projection",
    owner_key: owner.owner_key,
    conversation_id: "chat",
    source_digest: "source",
    fences: {},
    source_members: [],
    ordered_path_ids: [],
    cursor: { kind: "empty" },
    selection_revision: 1
  }
  await history.savePendingHistoryConfirmation(bookmark, origin, confirmation)
  await history.saveHistoryTurnRecovery(bookmark, origin, {
    ...pending,
    state: "unknown",
    result_text: "retained"
  })
  expect(
    (await history.loadHistoryBookmark(bookmark, owner))?.pending_confirmation
  ).toEqual(confirmation)
  await expect(
    history.saveHistoryTurnRecovery(bookmark, origin, {
      ...pending,
      input_text: "different"
    })
  ).rejects.toThrow("history_operation_conflict")
  await history.acknowledgeHistoryConfirmation(bookmark, {
    ...confirmation,
    projection_digest: "digest",
    created_at: "today"
  })
  expect(
    (await history.loadHistoryTurnRecoveries(bookmark, owner))[0].turn
      .result_text
  ).toBe("retained")
})

it("keeps how a retained reply ended and drops unknown outcomes (CS-04)", async () => {
  const owner = await history.getLocalHistoryOwner("chat")
  const origin = view(owner)
  const pending: any = {
    operation_id: "operation",
    owner_key: owner.owner_key,
    conversation_id: "chat",
    input_id: "input",
    assistant_id: "result",
    created_at: 1,
    input_text: "question",
    input_images: [],
    result_text: "partial",
    state: "generated_unsaved",
    origin_view: origin,
    selection_digest: "selected",
    request_context_digest: "prepared",
    outcome: "stopped",
    interruption_reason: "Stopped",
    settled_message_id: "result",
    model_name: "local-model",
    model_id: "provider:local-model",
    credential: "never stored"
  }
  await history.saveHistoryTurnRecovery(bookmark, origin, pending)
  const [kept] = await history.loadHistoryTurnRecoveries(bookmark, owner)
  expect(kept.turn).toMatchObject({
    outcome: "stopped",
    interruption_reason: "Stopped",
    settled_message_id: "result",
    model_name: "local-model",
    model_id: "provider:local-model"
  })
  expect(kept.turn).not.toHaveProperty("credential")

  await history.saveHistoryTurnRecovery(bookmark, origin, {
    ...pending,
    operation_id: "other",
    outcome: "resumed-elsewhere"
  })
  const other = (await history.loadHistoryTurnRecoveries(bookmark, owner)).find(
    (entry) => entry.turn.operation_id === "other"
  )
  expect(other?.turn).not.toHaveProperty("outcome")
})

it("terminal recovery outcomes fence late callbacks without erasing another operation", async () => {
  const owner = await history.getLocalHistoryOwner("chat")
  const origin = view(owner)
  const pending: any = {
    operation_id: "operation",
    owner_key: owner.owner_key,
    conversation_id: "chat",
    input_id: "input",
    assistant_id: "result",
    created_at: 1,
    input_text: "question",
    input_images: [],
    result_text: "",
    state: "dispatching",
    origin_view: origin,
    selection_digest: "selected",
    request_context_digest: "prepared"
  }
  await history.saveHistoryTurnRecovery(bookmark, origin, pending)
  await history.saveHistoryTurnRecovery(bookmark, origin, {
    ...pending,
    operation_id: "other"
  })
  await history.dismissHistoryTurnRecovery(
    bookmark,
    owner,
    "operation",
    "completed"
  )
  await history.saveHistoryTurnRecovery(bookmark, origin, {
    ...pending,
    state: "unknown",
    result_text: "late failure"
  })
  expect(
    (await history.loadHistoryTurnRecoveries(bookmark, owner)).map(
      (entry) => entry.turn.operation_id
    )
  ).toEqual(["other"])
})

it.each(["input_text", "selection_digest", "request_context_digest"] as const)("ordinary server memory recovery rejects changed %s", async field => {
  const owner = await history.getLocalHistoryOwner("chat")
  const origin = view(owner)
  const bookmark = { profile_id: owner.profile_id, client_session_id: "ordinary-memory" }
  const pending = { operation_id: "ordinary-memory", persistence: "server" as const, origin_view: origin,
    owner_key: owner.owner_key, conversation_id: owner.conversation_id, selection_digest: "selected",
    request_context_digest: "prepared", created_at: 1, input_text: "question", input_images: [],
    result_text: "Unconfirmed original server text", state: "unknown" as const }
  await history.saveHistoryTurnRecovery(bookmark, origin, { ...pending, result_text: "", state: "dispatching" })
  try {
    await history.retainHistoryTurnRecoveryInMemory(bookmark, origin, pending)
    expect((await history.loadHistoryTurnRecoveries(bookmark, owner))[0].turn).toMatchObject({ result_text: pending.result_text, recovery_storage: "memory" })
    await expect(history.retainHistoryTurnRecoveryInMemory(bookmark, origin, { ...pending, [field]: "changed" })).rejects.toThrow("history_operation_conflict")
  } finally { expect(await history.dismissHistoryTurnRecovery(bookmark, origin, pending.operation_id, "dismissed")).toBe(true) }
})

it("ordinary window recovery replaces streamed prefixes and restores the latest durable text", async () => {
  const owner = await history.getLocalHistoryOwner("chat"), origin = view(owner)
  const bookmark = { profile_id: owner.profile_id, client_session_id: "ordinary-prefix" }
  const pending = { operation_id: "ordinary-prefix", persistence: "server" as const, origin_view: origin,
    owner_key: owner.owner_key, conversation_id: owner.conversation_id, selection_digest: "selected",
    request_context_digest: "prepared", created_at: 1, input_text: "question", input_images: [],
    result_text: "", state: "dispatching" as const }
  await history.saveHistoryTurnRecovery(bookmark, origin, pending)
  try {
    for (const result_text of ["A", "An", "Answer"])
      await history.retainHistoryTurnRecoveryInMemory(bookmark, origin, { ...pending, result_text, state: "unknown" })
    const [entry] = await history.loadHistoryTurnRecoveries(bookmark, owner)
    expect(entry.turn.result_text).toBe("Answer")
    expect(entry.turn.retained_result_texts).toBeUndefined()
    await history.saveHistoryTurnRecovery(bookmark, origin, { ...pending, result_text: "Answer", state: "unknown" })
    const [saved] = await history.loadHistoryTurnRecoveries(bookmark, owner)
    expect(saved.turn.retained_result_texts).toBeUndefined()
    expect(saved.turn.recovery_storage).toBeUndefined()
  } finally { expect(await history.dismissHistoryTurnRecovery(bookmark, origin, pending.operation_id)).toBe(true) }
})

it("native recovery begins without canonical IDs and binds monotonically to owner evidence", async () => {
  const owner = await history.getLocalHistoryOwner("chat")
  const origin = view(owner)
  const bookmark = { profile_id: owner.profile_id, client_session_id: "native" }
  const pending: any = {
    operation_id: "native",
    persistence: "server",
    origin_view: origin,
    owner_key: owner.owner_key,
    conversation_id: owner.conversation_id,
    selection_digest: "selected",
    request_context_digest: "prepared",
    created_at: 1,
    input_text: "question",
    input_images: [],
    result_text: "",
    state: "dispatching"
  }
  await history.saveHistoryTurnRecovery(bookmark, origin, pending)
  expect(
    (await history.loadHistoryTurnRecoveries(bookmark, owner))[0].turn.input_id
  ).toBeUndefined()
  const accepted = {
    ...pending,
    input_id: "owner-input",
    state: "accepted_unsent",
    admission: {
      version: 1,
      owner_key: owner.owner_key,
      conversation_id: owner.conversation_id,
      input_message_id: "owner-input",
      input_message_revision: "r",
      selection_digest: "selected"
    }
  }
  await history.saveHistoryTurnRecovery(bookmark, origin, accepted)
  await history.saveHistoryTurnRecovery(bookmark, origin, pending)
  expect(
    (await history.loadHistoryTurnRecoveries(bookmark, owner))[0].turn.input_id
  ).toBe("owner-input")
  await expect(
    history.saveHistoryTurnRecovery(bookmark, origin, {
      ...accepted,
      input_id: "forged"
    })
  ).rejects.toThrow()
})

it("a confirmed refused native turn preserves exact input on reopen and cannot regress to uncertain dispatch", async () => {
  const owner = await history.getLocalHistoryOwner("chat")
  const origin = view(owner)
  const pending: Parameters<typeof history.saveHistoryTurnRecovery>[2] = {
    operation_id: "refused-native",
    persistence: "server",
    origin_view: origin,
    owner_key: owner.owner_key,
    conversation_id: owner.conversation_id,
    selection_digest: "selected",
    request_context_digest: "prepared",
    created_at: 1,
    input_text: "  exact refused input  ",
    input_images: [],
    result_text: "",
    state: "dispatching"
  }
  await history.saveHistoryTurnRecovery(bookmark, origin, pending)
  await history.saveHistoryTurnRecovery(bookmark, origin, {
    ...pending,
    state: "refused"
  })
  await history.saveHistoryTurnRecovery(bookmark, origin, pending)
  const reopened = await history.loadHistoryTurnRecoveries(
    { ...bookmark, client_session_id: "reopened" },
    owner
  )
  expect(reopened[0].turn).toMatchObject({
    state: "refused" as const,
    input_text: "  exact refused input  ",
    result_text: ""
  })
  expect(reopened[0].turn.admission).toBeUndefined()
  expect(reopened[0].turn.outcome).toBeUndefined()
  expect(
    (await history.captureLocalHistorySnapshot(owner, origin, "send")).snapshot
      .nodes
  ).toEqual([])
})

it.each(["client", "admission", "result", "content", "outcome"])(
  "refused recovery rejects contradictory %s evidence without rewriting durable input",
  async (contradiction) => {
    const owner = await history.getLocalHistoryOwner("chat")
    const origin = view(owner)
    const pending: Parameters<typeof history.saveHistoryTurnRecovery>[2] = {
      operation_id: "refused-native",
      persistence: "server",
      origin_view: origin,
      owner_key: owner.owner_key,
      conversation_id: owner.conversation_id,
      selection_digest: "selected",
      request_context_digest: "prepared",
      created_at: 1,
      input_text: "exact refused input",
      input_images: [],
      result_text: "",
      state: "dispatching"
    }
    await history.saveHistoryTurnRecovery(bookmark, origin, pending)
    const invalid = {
      ...pending,
      state: "refused" as const,
      ...(contradiction === "client"
        ? { persistence: "client" as const, input_id: "i", assistant_id: "a" }
        : contradiction === "admission"
          ? {
              input_id: "i",
              admission: {
                version: 1 as const,
                owner_key: owner.owner_key,
                conversation_id: owner.conversation_id,
                input_message_id: "i",
                input_message_revision: "r",
                selection_digest: "selected"
              }
            }
          : contradiction === "result"
            ? { assistant_id: "a" }
            : contradiction === "content"
              ? { result_text: " " }
              : { outcome: "stopped" as const })
    }
    await expect(
      history.saveHistoryTurnRecovery(bookmark, origin, invalid)
    ).rejects.toThrow()
    expect(
      (await history.loadHistoryTurnRecoveries(bookmark, owner))[0].turn
    ).toMatchObject({ state: "dispatching", input_text: "exact refused input" })
  }
)

describe("on-device owner preparation", () => {
  const reader = { profileId: "profile", serverScopeKey: "realm-A", isCurrent: () => true }
  it("claims a current scope-only row under its captured profile", async () => {
    await memory.userSettings.update("main", { history_profile_id: "profile" })
    await memory.chatHistories.update("chat", { server_scope_key: "realm-A" })
    expect(await history.getLocalHistoryOwner("chat", reader)).toMatchObject({ profile_id: "profile", owner_key: "local-history-v1:profile" })
  })
  it.each(["replaced profile", "foreign realm", "expired reader"])("does not stamp a row after %s", async reason => {
    await memory.userSettings.update("main", { history_profile_id: reason === "replaced profile" ? "foreign" : "profile" })
    await memory.chatHistories.update("chat", { server_scope_key: reason === "foreign realm" ? "realm-B" : "realm-A" })
    const before = JSON.stringify([await memory.userSettings.get("main"), await memory.chatHistories.get("chat")])
    await expect(history.getLocalHistoryOwner("chat", { ...reader, isCurrent: () => reason !== "expired reader" })).rejects.toMatchObject({ code: "owner_mismatch" })
    expect(JSON.stringify([await memory.userSettings.get("main"), await memory.chatHistories.get("chat")])).toBe(before)
  })
  it("does not initialize a replacement profile for a stale qualified click", async () => {
    await memory.userSettings.delete("main")
    await memory.chatHistories.update("chat", { server_scope_key: "realm-A" })
    await expect(history.getLocalHistoryOwner("chat", reader)).rejects.toMatchObject({ code: "owner_mismatch" })
    expect(await memory.userSettings.get("main")).toBeUndefined()
    expect((await memory.chatHistories.get("chat")).local_owner_key).toBeUndefined()
  })
})

it.each(["missing", "replaced"])("verifies an expected profile without initializing after it is %s", async reason => {
  if (reason === "missing") await memory.userSettings.delete("main")
  else await memory.userSettings.update("main", { history_profile_id: "foreign" })
  const before = JSON.stringify(await memory.userSettings.get("main"))
  await expect(history.ensureLocalProfileId("profile")).rejects.toMatchObject({ code: "owner_mismatch" })
  expect(JSON.stringify(await memory.userSettings.get("main"))).toBe(before)
})


it("explicit edited recovery policy round-trips and cannot become an ordinary checkpoint", async () => {
  const owner = await history.getLocalHistoryOwner("chat")
  const origin = view(owner)
  const pending: Parameters<typeof history.saveHistoryTurnRecovery>[2] = {
    action: "edit_resend",
    operation_id: "edited",
    owner_key: owner.owner_key,
    conversation_id: "chat",
    origin_view: origin,
    selection_digest: "selection",
    request_context_digest: "request",
    input_id: "input",
    assistant_id: "result",
    created_at: 1,
    input_text: "Exact  \n",
    input_images: [],
    result_text: "Reply",
    state: "generated_unsaved",
    outcome: "complete",
    settled_message_id: "result"
  }
  await history.saveHistoryTurnRecovery(bookmark, origin, pending)
  expect(
    (await history.loadHistoryTurnRecoveries(bookmark, owner))[0].turn
  ).toMatchObject(pending)
  await expect(
    history.saveHistoryTurnRecovery(bookmark, origin, {
      ...pending,
      action: undefined
    })
  ).rejects.toThrow("history_operation_conflict")
  expect(
    (await history.loadHistoryTurnRecoveries(bookmark, owner))[0].turn.action
  ).toBe("edit_resend")
})

it("recovery reads refuse an unknown action without downgrading it into automatic ordinary work", async () => {
  const owner = await history.getLocalHistoryOwner("chat")
  const origin = view(owner)
  const pending: Parameters<typeof history.saveHistoryTurnRecovery>[2] = {
    operation_id: "old",
    owner_key: owner.owner_key,
    conversation_id: "chat",
    origin_view: origin,
    selection_digest: "selection",
    request_context_digest: "request",
    input_id: "u",
    assistant_id: "a",
    created_at: 1,
    input_text: "Original",
    input_images: [],
    result_text: "Reply",
    state: "generated_unsaved",
    outcome: "complete",
    settled_message_id: "a"
  }
  await history.saveHistoryTurnRecovery(bookmark, origin, pending)
  const row = await memory.historySelections.get([
    bookmark.profile_id,
    bookmark.client_session_id,
    owner.owner_key,
    "chat"
  ])
  await memory.historySelections.put({
    ...row,
    pending_turns: { old: { ...pending, action: "unrecognized" } }
  })
  await expect(
    history.loadHistoryTurnRecoveries(bookmark, owner)
  ).rejects.toThrow("unsupported_history_action")
  expect(
    (
      await memory.historySelections.get([
        bookmark.profile_id,
        bookmark.client_session_id,
        owner.owner_key,
        "chat"
      ])
    ).pending_turns.old.action
  ).toBe("unrecognized")
  await memory.historySelections.put({
    ...row,
    pending_turns: { old: pending }
  })
  expect(
    (await history.loadHistoryTurnRecoveries(bookmark, owner))[0].turn.action
  ).toBeUndefined()
})


const regenerationRecovery = (origin: HistoryViewSelectionV1): import("../types").HistoryTurnRecovery => ({
  action: "regenerate", operation_id: "regenerated", persistence: "server", origin_view: origin,
  owner_key: origin.owner_key, conversation_id: origin.conversation_id, selection_digest: "NEW REGENERATION SELECTION",
  request_context_digest: "request", created_at: 1, input_text: "  Original input  \n", input_images: ["saved-image"],
  result_text: "", state: "dispatching"
})
it("regeneration recovery retains both the new operation digest and original input admission digest", async () => {
  const owner = await history.getLocalHistoryOwner("chat"), origin = view(owner)
  origin.cursor = { kind: "after_message", message_id: "original-assistant" }
  const pending = regenerationRecovery(origin)
  await history.saveHistoryTurnRecovery(bookmark, origin, pending)
  const accepted = { ...pending, input_id: "original-input", state: "accepted_unsent" as const, admission: { version: 1 as const,
    owner_key: owner.owner_key, conversation_id: "chat", input_message_id: "original-input", input_message_revision: "1", selection_digest: "ORIGINAL ADMISSION" } }
  await history.saveHistoryTurnRecovery(bookmark, origin, accepted)
  expect((await history.loadHistoryTurnRecoveries(bookmark, owner))[0].turn).toMatchObject(accepted)
  for (const patch of [{ action: undefined }, { selection_digest: "different" }, { admission: { ...accepted.admission, selection_digest: "different" } },
    { input_id: "foreign-input" }, { admission: { ...accepted.admission, owner_key: "foreign" } }]) {
    await expect(history.saveHistoryTurnRecovery(bookmark, origin, { ...accepted, ...patch })).rejects.toThrow("history_operation_conflict")
  }
  expect((await history.loadHistoryTurnRecoveries(bookmark, owner))[0].turn).toMatchObject(accepted)
})
it("confirmed saved-image regeneration refusal keeps the exact original attachments without inventing admission", async () => {
  const owner = await history.getLocalHistoryOwner("chat"), origin = view(owner)
  origin.cursor = { kind: "after_message", message_id: "original-assistant" }
  const pending = regenerationRecovery(origin)
  await history.saveHistoryTurnRecovery(bookmark, origin, pending)
  await history.saveHistoryTurnRecovery(bookmark, origin, { ...pending, state: "refused" })
  expect((await history.loadHistoryTurnRecoveries(bookmark, owner))[0].turn).toMatchObject({ ...pending, state: "refused" })
})
it.each(["client", "before_message"])("regeneration recovery refuses unsupported %s persistence or boundary", async boundary => {
  const owner = await history.getLocalHistoryOwner("chat"), origin = view(owner)
  origin.cursor = { kind: boundary === "client" ? "after_message" : "before_message", message_id: "original-assistant" }
  const pending = regenerationRecovery(origin)
  if (boundary === "client") Object.assign(pending, { persistence: "client", input_id: "u", assistant_id: "a" })
  await expect(history.saveHistoryTurnRecovery(bookmark, origin, pending)).rejects.toThrow("unsupported_history_action")
})


it("a refused client-managed native attempt keeps candidate IDs and exact input without admission", async () => {
  const owner = await history.getLocalHistoryOwner("chat")
  const origin = view(owner)
  const pending: Parameters<typeof history.saveHistoryTurnRecovery>[2] = {
    operation_id: "refused-client-native",
    persistence: "client",
    origin_view: origin,
    owner_key: owner.owner_key,
    conversation_id: owner.conversation_id,
    selection_digest: "selected",
    request_context_digest: "prepared",
    created_at: 1,
    input_id: "candidate-user",
    assistant_id: "candidate-assistant",
    input_text: "  exact unsent input\nsecond line  ",
    input_images: ["data:image/png;base64,YQ=="],
    result_text: "",
    state: "dispatching"
  }
  await history.saveHistoryTurnRecovery(bookmark, origin, pending)
  await history.saveHistoryTurnRecovery(bookmark, origin, {
    ...pending,
    state: "refused"
  })
  await history.saveHistoryTurnRecovery(bookmark, origin, pending)
  const reopened = await history.loadHistoryTurnRecoveries(
    { ...bookmark, client_session_id: "reopened" },
    owner
  )
  expect(reopened[0].turn).toMatchObject({ ...pending, state: "refused" })
  expect(reopened[0].turn.admission).toBeUndefined()
  expect(
    (await history.captureLocalHistorySnapshot(owner, origin, "send")).snapshot
      .nodes
  ).toEqual([])
  await expect(
    history.saveHistoryTurnRecovery(bookmark, origin, {
      ...pending,
      state: "unknown",
      settled_message_id: "invented"
    })
  ).rejects.toThrow("history_operation_conflict")
})


describe("acknowledged recovery dismissal", () => {
  const retained = async () => {
    const owner = await history.getLocalHistoryOwner("chat")
    const origin = view(owner)
    const turn = {
      operation_id: "dismiss-ack",
      owner_key: owner.owner_key,
      conversation_id: "chat",
      input_id: "input",
      assistant_id: "result",
      persistence: "client" as const,
      created_at: 1,
      input_text: "  exact input\n  ",
      input_images: [],
      result_text: "<think>raw reasoning</think>  exact reply\n  ",
      state: "generated_unsaved" as const,
      origin_view: origin,
      selection_digest: "selected",
      request_context_digest: "prepared"
    }
    await history.saveHistoryTurnRecovery(bookmark, origin, turn)
    return { owner, turn }
  }

  it("acknowledges only a committed removal, not an already missing record", async () => {
    const { owner } = await retained()
    expect(
      await history.dismissHistoryTurnRecovery(bookmark, owner, "dismiss-ack")
    ).toBe(true)
    expect(
      await history.dismissHistoryTurnRecovery(bookmark, owner, "dismiss-ack")
    ).toBe(false)
    expect(
      (await history.loadHistoryBookmark(bookmark, owner))
        ?.history_turn_outcomes
    ).toEqual({ "dismiss-ack": "dismissed" })
  })

  it.each(["entry", "profile", "read", "write"])(
    "rolls back dismissal after its lease retires at %s",
    async (boundary) => {
      const { owner, turn } = await retained()
      let current = boundary !== "entry"
      const table =
        boundary === "profile" ? memory.userSettings : memory.historySelections
      const method = boundary === "write" ? "put" : "get"
      const original = table[method].bind(table)
      const spy =
        boundary === "entry"
          ? null
          : vi
              .spyOn(table, method)
              .mockImplementationOnce(async (...args: unknown[]) => {
                const result = await original(...args)
                current = false
                return result
              })
      try {
        await expect(
          history.dismissHistoryTurnRecovery(
            bookmark,
            owner,
            "dismiss-ack",
            "dismissed",
            { validate_lease: () => current }
          )
        ).rejects.toMatchObject({ code: "request_config_scope_changed" })
      } finally {
        spy?.mockRestore()
      }
      expect(await history.loadHistoryTurnRecoveries(bookmark, owner)).toEqual([
        { scope: bookmark, turn }
      ])
      expect(
        (await history.loadHistoryBookmark(bookmark, owner))
          ?.history_turn_outcomes
      ).toBeUndefined()
    }
  )

  it("aborts a dismissal retired after its write while the transaction receipt is still pending", async () => {
    const { owner, turn } = await retained()
    const original = new Map(memory.historySelections.rows)
    let finish!: () => void
    let entered!: () => void
    const written = new Promise<void>((resolve) => {
      entered = resolve
    })
    memory.transaction.mockImplementationOnce(
      async (
        _mode: unknown,
        _tables: unknown,
        operation: (tx: { abort: () => void }) => Promise<boolean>
      ) => {
        let aborted = false
        const receipt = await operation({
          abort() {
            aborted = true
          }
        })
        entered()
        await new Promise<void>((resolve) => {
          finish = resolve
        })
        if (aborted) {
          memory.historySelections.rows.clear()
          original.forEach((value, key) =>
            memory.historySelections.rows.set(key, value)
          )
          throw new Error("transaction aborted before commit")
        }
        return receipt
      }
    )
    const controller = new AbortController()
    const pending = history.dismissHistoryTurnRecovery(
      bookmark,
      owner,
      "dismiss-ack",
      "dismissed",
      { signal: controller.signal }
    )
    await written
    controller.abort()
    finish()
    await expect(pending).rejects.toThrow("transaction aborted before commit")
    expect(await history.loadHistoryTurnRecoveries(bookmark, owner)).toEqual([
      { scope: bookmark, turn }
    ])
  })

  it("retains the exact journal when storage rejects its removal", async () => {
    const { owner, turn } = await retained()
    const spy = vi
      .spyOn(memory.historySelections, "put")
      .mockRejectedValueOnce(new Error("journal unavailable"))
    try {
      await expect(
        history.dismissHistoryTurnRecovery(bookmark, owner, "dismiss-ack")
      ).rejects.toThrow("journal unavailable")
    } finally {
      spy.mockRestore()
    }
    expect(await history.loadHistoryTurnRecoveries(bookmark, owner)).toEqual([
      { scope: bookmark, turn }
    ])
  })

  it("refuses an aborted or foreign-profile dismissal without losing the journal", async () => {
    const { owner, turn } = await retained()
    const controller = new AbortController()
    controller.abort()
    await expect(
      history.dismissHistoryTurnRecovery(
        bookmark,
        owner,
        "dismiss-ack",
        "dismissed",
        { signal: controller.signal }
      )
    ).rejects.toMatchObject({ code: "request_config_scope_changed" })
    await expect(
      history.dismissHistoryTurnRecovery(
        { ...bookmark, profile_id: "foreign" },
        owner,
        "dismiss-ack"
      )
    ).rejects.toMatchObject({ code: "bookmark_owner_mismatch" })
    expect(await history.loadHistoryTurnRecoveries(bookmark, owner)).toEqual([
      { scope: bookmark, turn }
    ])
  })
})

// Production ownership/admission/settlement run against the serialized rollback
// table-engine double above. These cases do not qualify native IndexedDB commits.
describe("local assistant settlement with captured document files", () => {
  const file = (id = "new-file") => ({
    id, filename: `${id}.txt`, type: "text/plain", content: "  Original file\n",
    size: 16, uploadedAt: 7, processed: true
  })
  const accepted = async () => {
    const owner = await history.getLocalHistoryOwner("chat")
    const captured = await history.captureLocalHistorySnapshot(owner, view(owner), "send")
    const admission = await history.appendLocalSelectedUser(owner, selection(captured), row("file-user"))
    const result = { ...row("file-reply"), role: "assistant", content: "Original answer" }
    const originalSession = {
      sessionId: "chat", files: [file("existing-file")], retrievalEnabled: true, createdAt: 3
    }
    await memory.sessionFiles.put(originalSession)
    return { owner, admission, result, originalSession }
  }

  it("reuses the exact assistant and file ID without duplicating files or changing session metadata", async () => {
    const { owner, admission, result, originalSession } = await accepted()
    const capturedFile = file()
    await service.settleAcceptedAssistant(owner, admission, result, undefined, [capturedFile])
    const first = await memory.sessionFiles.get("chat")
    await service.settleAcceptedAssistant(owner, admission, result, undefined, [capturedFile])
    expect(await memory.sessionFiles.get("chat")).toEqual(first)
    expect(first).toEqual({ ...originalSession, files: [...originalSession.files, capturedFile] })
    expect(await memory.messages.where("history_id").equals("chat").toArray()).toHaveLength(2)
    expect(await memory.messages.get("file-reply")).toMatchObject({
      content: result.content, parent_message_id: "file-user"
    })
  })

  it("refuses a differing duplicate file ID and retains the original session and accepted assistant", async () => {
    const { owner, admission, result } = await accepted()
    await service.settleAcceptedAssistant(owner, admission, result, undefined, [file()])
    const before = await memory.sessionFiles.get("chat")
    await expect(service.settleAcceptedAssistant(owner, admission,
      { ...result, id: "conflicting-file-reply" }, undefined,
      [{ ...file(), content: "Changed file" }]
    )).rejects.toMatchObject({ code: "local_file_id_conflict" })
    expect(await memory.sessionFiles.get("chat")).toEqual(before)
    expect(await memory.messages.get("conflicting-file-reply")).toBeUndefined()
    expect(await memory.messages.get("file-reply")).toMatchObject({ content: result.content })
  })

  it("rolls back both files and assistant when the original owner lease retires after file put", async () => {
    const { owner, admission, result, originalSession } = await accepted()
    let current = true
    const qualifiedOwner = { ...owner, validate_lease: () => current }
    const put = memory.sessionFiles.put.bind(memory.sessionFiles)
    const spy = vi.spyOn(memory.sessionFiles, "put").mockImplementationOnce(async value => {
      await put(value)
      current = false
    })
    try {
      await expect(service.settleAcceptedAssistant(qualifiedOwner, admission, result, undefined, [file()]))
        .rejects.toMatchObject({ code: "request_config_scope_changed" })
    } finally { spy.mockRestore() }
    expect(await memory.sessionFiles.get("chat")).toEqual(originalSession)
    expect(await memory.messages.get("file-reply")).toBeUndefined()
    expect(await memory.messages.get("file-user")).toMatchObject({ content: "file-user" })
  })

  it("rolls back the new files if the assistant write fails", async () => {
    const { owner, admission, result, originalSession } = await accepted()
    const write = vi.spyOn(memory.messages, "add").mockRejectedValueOnce(new Error("assistant write unavailable"))
    try {
      await expect(service.settleAcceptedAssistant(owner, admission, result, undefined, [file()]))
        .rejects.toThrow("assistant write unavailable")
    } finally { write.mockRestore() }
    expect(await memory.sessionFiles.get("chat")).toEqual(originalSession)
    expect(await memory.messages.get("file-reply")).toBeUndefined()
    expect(await memory.messages.get("file-user")).toMatchObject({ content: "file-user" })
  })

  it("settles the original captured file proposal when its caller mutates it during a held read", async () => {
    const { owner, admission, result, originalSession } = await accepted()
    const proposal = [file()]
    const originalProposal = structuredClone(proposal)
    let entered!: () => void
    let release!: () => void
    const reading = new Promise<void>(resolve => { entered = resolve })
    const held = new Promise<void>(resolve => { release = resolve })
    const get = memory.sessionFiles.get.bind(memory.sessionFiles)
    const spy = vi.spyOn(memory.sessionFiles, "get").mockImplementationOnce(async id => {
      const value = await get(id)
      entered()
      await held
      return value
    })
    const pending = service.settleAcceptedAssistant(owner, admission, result, undefined, proposal)
    try {
      await reading
      proposal[0].content = "Newer caller file"
      proposal.push(file("later-file"))
      release()
      await expect(pending).resolves.toMatchObject({ id: "file-reply" })
    } finally { release(); spy.mockRestore() }
    expect(await memory.sessionFiles.get("chat")).toEqual({
      ...originalSession, files: [...originalSession.files, ...originalProposal]
    })
  })
})


describe("genuine Roleplay recovery receipt arm", () => {
  const original = {
    version: 1 as const,
    conversation_id: "chat",
    assistant_message_id: "resume-assistant",
    input: {
      message_id: "resume-user",
      content: "  Exact saved roleplay\n",
      expected: {
        version: 1 as const,
        snapshot_digest: "sha256:" + "b".repeat(64),
        settings_version: 2,
        history_version: 1,
        tail: { message_id: null, message_version: null }
      }
    }
  }
  const append = {
    version: 1 as const,
    conversation_id: "chat",
    message_id: original.input.message_id,
    message_version: 1,
    parent_message_id: null,
    fence: {
      ...original.input.expected,
      history_version: 2,
      tail: { message_id: original.input.message_id, message_version: 1 }
    }
  }
  const completion = {
    chat_id: "chat",
    character_id: 42,
    provider: "openai",
    model: "original",
    saved: true,
    user_message_id: original.input.message_id,
    assistant_message_id: original.assistant_message_id,
    assistant_content: "Exact generated answer",
    tldw_roleplay_resume_v1: {
      version: 1 as const,
      input_message_id: original.input.message_id,
      input_message_version: 1,
      assistant_message_id: original.assistant_message_id,
      assistant_message_version: 1,
      expected: append.fence,
      fence: {
        ...append.fence,
        history_version: 3,
        tail: { message_id: original.assistant_message_id, message_version: 1 }
      },
      unsaved_reason: null
    }
  }
  const pending = async () => {
    const owner = await history.getLocalHistoryOwner("chat")
    const origin = view(owner)
    const turn: Parameters<typeof history.saveHistoryTurnRecovery>[2] = {
      operation_id: "resume-original",
      owner_key: owner.owner_key,
      conversation_id: "chat",
      origin_view: origin,
      persistence: "server",
      selection_digest: "selected",
      request_context_digest: "resume-intent",
      created_at: 1,
      input_text: original.input.content,
      input_images: [],
      result_text: "",
      state: "dispatching",
      roleplay_resume: { intent: original }
    }
    await history.saveHistoryTurnRecovery(bookmark, origin, turn)
    return { owner, origin, turn }
  }
  it("retains chosen IDs only in immutable intent until a genuine append acknowledgement", async () => {
    const { owner } = await pending()
    const [entry] = await history.loadHistoryTurnRecoveries(bookmark, owner)
    expect(entry.turn.roleplay_resume?.intent).toEqual(original)
    expect(entry.turn.input_id).toBeUndefined()
    expect(entry.turn.assistant_id).toBeUndefined()
    expect(entry.turn.admission).toBeUndefined()
  })
  it("promotes original receipts without fabricating H1 admission and keeps them on late downgrade", async () => {
    const { owner, origin, turn } = await pending()
    const accepted = {
      ...turn,
      input_id: original.input.message_id,
      state: "accepted_unsent" as const,
      roleplay_resume: { intent: original, append }
    }
    await history.saveHistoryTurnRecovery(bookmark, origin, accepted)
    const settled = {
      ...accepted,
      state: "generated_unsaved" as const,
      result_text: completion.assistant_content,
      assistant_id: original.assistant_message_id,
      settled_message_id: original.assistant_message_id,
      outcome: "complete" as const,
      roleplay_resume: { intent: original, append, completion }
    }
    await history.saveHistoryTurnRecovery(bookmark, origin, settled)
    await history.saveHistoryTurnRecovery(bookmark, origin, turn)
    expect(
      (await history.loadHistoryTurnRecoveries(bookmark, owner))[0].turn
    ).toMatchObject(settled)
  })
  it.each([
    "input-id",
    "assistant-id",
    "foreign-owner",
    "changed-input",
    "images",
    "h1-admission",
    "wrong-fence"
  ])("refuses unsupported Roleplay journal evidence %s", async (change) => {
    const { origin, turn } = await pending()
    const patch =
      change === "input-id"
        ? { input_id: original.input.message_id }
        : change === "assistant-id"
          ? { assistant_id: original.assistant_message_id }
          : change === "foreign-owner"
            ? {
                roleplay_resume: {
                  intent: { ...original, conversation_id: "foreign" }
                }
              }
            : change === "changed-input"
              ? {
                  roleplay_resume: {
                    intent: {
                      ...original,
                      input: { ...original.input, content: "different" }
                    }
                  }
                }
              : change === "images"
                ? { input_images: ["image"] }
                : change === "h1-admission"
                  ? {
                      admission: {
                        version: 1 as const,
                        owner_key: turn.owner_key,
                        conversation_id: "chat",
                        input_message_id: original.input.message_id,
                        input_message_revision: "1",
                        selection_digest: "selected"
                      }
                    }
                  : {
                      roleplay_resume: {
                        intent: original,
                        append: {
                          ...append,
                          fence: { ...append.fence, settings_version: 3 }
                        }
                      }
                    }
    await expect(
      history.saveHistoryTurnRecovery(bookmark, origin, { ...turn, ...patch })
    ).rejects.toThrow()
  })
  it("does not promote an unsaved completion to canonical assistant identity", async () => {
    const { origin, turn } = await pending()
    const unsaved = {
      ...completion,
      saved: false,
      assistant_message_id: null,
      tldw_roleplay_resume_v1: {
        ...completion.tldw_roleplay_resume_v1,
        assistant_message_version: null,
        fence: null,
        unsaved_reason: "history_fence_changed"
      }
    }
    await expect(
      history.saveHistoryTurnRecovery(bookmark, origin, {
        ...turn,
        input_id: original.input.message_id,
        assistant_id: original.assistant_message_id,
        result_text: completion.assistant_content,
        state: "generated_unsaved",
        roleplay_resume: { intent: original, append, completion: unsaved }
      })
    ).rejects.toThrow()
  })
  it("whitelists credential-free recovery but keeps all unacknowledged generated text", async () => {
    const { owner, origin, turn } = await pending()
    const roleplay_resume = {
      ...turn.roleplay_resume!,
      servicePromptConfig: { apiKey: "never-store" }
    }
    await history.saveHistoryTurnRecovery(bookmark, origin, {
      ...turn,
      state: "generated_unsaved",
      result_text: "x".repeat(1_000_001),
      roleplay_resume
    })
    const [entry] = await history.loadHistoryTurnRecoveries(bookmark, owner)
    expect(entry.turn.roleplay_resume).toEqual({ intent: original })
    expect(entry.turn.result_text).toHaveLength(1_000_001)
  })
  it("freezes original Roleplay recovery before its optional validator is loaded", async () => {
    const { owner, origin, turn } = await pending()
    const record = structuredClone(turn)
    const storing = history.saveHistoryTurnRecovery(bookmark, origin, record)
    record.input_text = "Replaced input"
    record.roleplay_resume!.intent.input.content = "Replaced input"
    await storing
    const [entry] = await history.loadHistoryTurnRecoveries(bookmark, owner)
    expect(entry.turn.input_text).toBe(original.input.content)
    expect(entry.turn.roleplay_resume!.intent.input.content).toBe(
      original.input.content
    )
  })

  it("drops unused nested completion diagnostics before durable recovery", async () => {
    const { owner, origin, turn } = await pending()
    await history.saveHistoryTurnRecovery(bookmark, origin, {
      ...turn,
      input_id: original.input.message_id,
      assistant_id: original.assistant_message_id,
      result_text: completion.assistant_content,
      state: "generated_unsaved",
      outcome: "complete",
      settled_message_id: original.assistant_message_id,
      roleplay_resume: {
        intent: original,
        append,
        completion: {
          ...completion,
          lorebook_diagnostics: [
            { apiKey: "never-store", nested: { credential: "never-store" } }
          ]
        } as unknown as NonNullable<typeof turn.roleplay_resume>
      }
    })
    const [entry] = await history.loadHistoryTurnRecoveries(bookmark, owner)
    expect(JSON.stringify(entry.turn)).not.toContain("never-store")
    expect(entry.turn.roleplay_resume?.completion?.saved).toBe(true)
  })

  it("does not resurrect window-only recovery after another view commits dismissal", async () => {
    const { owner, origin, turn } = await pending()
    await history.retainHistoryTurnRecoveryInMemory(bookmark, origin, {
      ...turn,
      result_text: "Window-only original reply",
      state: "generated_unsaved"
    })
    const key = [
      bookmark.profile_id,
      bookmark.client_session_id,
      owner.owner_key,
      owner.conversation_id
    ]
    const record = await memory.historySelections.get(key)
    await memory.historySelections.put({
      ...record,
      pending_turns: {},
      history_turn_outcomes: { [turn.operation_id]: "dismissed" }
    })
    expect(await history.loadHistoryTurnRecoveries(bookmark, owner)).toEqual([])
    expect(
      await history.dismissHistoryTurnRecovery(
        bookmark,
        owner,
        turn.operation_id
      )
    ).toBe(false)
  })
})

describe("explicit imported local binding", () => {
  const options = () => ({ signal: new AbortController().signal, validate_lease: () => true })
  it("binds only the current browser profile and preserves imported content", async () => {
    const original = await memory.chatHistories.get("chat")
    await memory.messages.put({ ...row("image-question"), images: ["http://asset.test/missing.png"] })
    const saved = await memory.messages.get("image-question")
    const owner = await history.bindUnboundLocalHistory("chat", options())
    expect(owner).toMatchObject({ kind: "local", profile_id: "profile", conversation_id: "chat" })
    expect(await memory.chatHistories.get("chat")).toEqual({ ...original, local_owner_key: owner.owner_key })
    expect(await memory.messages.get("image-question")).toEqual(saved)
  })
  it.each([
    { local_owner_key: "foreign-profile" },
    { server_scope_key: "foreign-account" },
    { server_chat_id: "native" },
    { message_source: "server" },
    { local_settings_guard: { revision: "pending", pending: ["other-write"] } }
  ])("refuses an existing ownership/native/pending-write fence: %j", async patch => {
    await memory.chatHistories.update("chat", patch)
    const original = await memory.chatHistories.get("chat")
    await expect(history.bindUnboundLocalHistory("chat", options())).rejects.toMatchObject({ name: "HistorySelectionError" })
    expect(await memory.chatHistories.get("chat")).toEqual(original)
  })
  it("does not install authority after its caller lease expires", async () => {
    const original = await memory.chatHistories.get("chat")
    await expect(history.bindUnboundLocalHistory("chat", { ...options(), validate_lease: () => false })).rejects.toMatchObject({ code: "request_config_scope_changed" })
    expect(await memory.chatHistories.get("chat")).toEqual(original)
  })
  it("rolls back binding when cancellation interrupts the actual write", async () => {
    const controller = new AbortController()
    const original = await memory.chatHistories.get("chat")
    const update = memory.chatHistories.update
    memory.chatHistories.update = async (...args: any[]) => { const result = await update(...args); controller.abort(); return result }
    try {
      await expect(history.bindUnboundLocalHistory("chat", { signal: controller.signal, validate_lease: () => true })).rejects.toMatchObject({ code: "request_config_scope_changed" })
      expect(await memory.chatHistories.get("chat")).toEqual(original)
    } finally { memory.chatHistories.update = update }
  })
})


it.each([
  { local_owner_key: "foreign-profile", message_source: "web-ui" },
  { server_scope_key: "foreign-account", server_chat_id: "native-source", message_source: "server" }
])("public restoration cannot erase destination ownership and make it adoptable: %j", async ownership => {
  await memory.chatHistories.update("chat", ownership)
  await new PageAssistDatabase().importChatHistoryV2([{ history: { id: "chat", title: "Incoming copy", createdAt: 2, is_rag: false, message_source: "web-ui" }, messages: [] }])
  expect(await memory.chatHistories.get("chat")).toMatchObject(ownership)
  await expect(history.bindUnboundLocalHistory("chat", { signal: new AbortController().signal, validate_lease: () => true })).rejects.toMatchObject({ name: "HistorySelectionError" })
})


it.each(["valid", "owner", "digest", "question", "promote-unknown"])("B02 server existing-question recovery retains real-store authority (%s)", async mode => {
  const owner = await history.getLocalHistoryOwner("chat")
  await memory.messages.put(row("saved-question", null))
  const origin = view(owner, {kind: "after_message", message_id: "saved-question"})
  const captured = await history.captureLocalHistorySnapshot(owner, origin, "send")
  const chosen = selection(captured)
  const scope = {profile_id: owner.profile_id, client_session_id: "b02-existing"}
  const pending: any = {operation_id: "new-explicit-answer", persistence: "server", origin_view: origin,
    owner_key: owner.owner_key, conversation_id: "chat", input_id: "saved-question", existing_selection: chosen,
    selection_digest: chosen.selection_digest, request_context_digest: chosen.request_context_digest,
    created_at: 1, input_text: "saved-question", input_images: [], result_text: "", state: "dispatching"}
  if (mode === "promote-unknown") {
    await history.saveHistoryTurnRecovery(scope, origin, {...pending, state: "unknown", input_id: undefined, existing_selection: undefined})
    await expect(history.saveHistoryTurnRecovery(scope, origin, pending)).rejects.toThrow()
    expect((await history.loadHistoryTurnRecoveries(scope, owner))[0].turn.state).toBe("unknown")
    return
  }
  if (mode === "owner") pending.existing_selection = {...chosen, owner_key: "foreign"}
  if (mode === "digest") pending.existing_selection = {...chosen, request_context_digest: "changed"}
  if (mode === "question") pending.input_id = "other"
  if (mode !== "valid") { await expect(history.saveHistoryTurnRecovery(scope, origin, pending)).rejects.toThrow(); return }
  await history.saveHistoryTurnRecovery(scope, origin, pending)
  await history.saveHistoryTurnRecovery(scope, origin, {...pending, state: "accepted_unsent"})
  const records = await history.loadHistoryTurnRecoveries(scope, owner)
  expect(records[0].turn).toMatchObject({input_id: "saved-question", state: "accepted_unsent", existing_selection: chosen})
  expect(records[0].turn.admission).toBeUndefined()
  await expect(history.saveHistoryTurnRecovery(scope, origin, {...pending, existing_selection: undefined})).rejects.toThrow()
})
