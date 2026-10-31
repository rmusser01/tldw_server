// @vitest-environment jsdom
import { describe, expect, it } from "vitest"
import { waitFor } from "@testing-library/react"
import { historyDigest } from "@/db/dexie/history-selection"
import { resolveHistorySelection, selectionDigest } from "@/utils/history-selection"
import { createFakeTldwServer, deferred, FAKE_MODEL, type FakeTldwServer, type JsonBody } from "./fake-tldw-server"

const request = (server: FakeTldwServer, path: string, body: JsonBody) => server.fetch(server.url + path, {
  method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body)
})
const prepare = async (server: FakeTldwServer, id = "protected") => {
  const chat = server.seedChat({ id, title: "Protected edits", turns: [
    { role: "user", content: "First question" }, { role: "assistant", content: "First answer" },
    { role: "user", content: "Original question" }, { role: "assistant", content: "Original answer" }
  ] })
  const target = chat.messages[2]
  const capture = await (await request(server, `/api/v1/chat/conversations/${id}/history/selection`, {
    purpose: "send", view: { view_session_id: "protected-view", owner_key: "harness-native-owner", conversation_id: id,
      interpretation: { kind: "parent_graph_v1" }, cursor: { kind: "before_message", message_id: target.id }, selection_revision: 1 }
  })).json()
  const context = { model: FAKE_MODEL, stream: false, save_to_db: true, conversation_id: id,
    messages: [...capture.rows.map((row: { id: string }) => {
      const original = chat.messages.find(message => message.id === row.id)!
      return { role: original.role, content: original.content }
    }), { role: "user", content: "Edited question" }] }
  const selected = resolveHistorySelection(capture.snapshot, capture.view, "send", historyDigest(context))
  if (selected.status !== "ready") throw new Error("Fixture prefix is unavailable")
  const body: JsonBody = { ...context, tldw_history_selection_v1: selected.selection, tldw_history_branch: true,
    tldw_history_edit_v1: { version: 1, message_id: target.id, message_revision: String(target.version) } }
  return { chat, target, body }
}

describe("fake API protected edit admission", () => {
  it.each(["completion", "append"])("branches an edited input from its original prefix through %s", async operation => {
    const server = createFakeTldwServer()
    const { chat, body } = await prepare(server)
    const originals = structuredClone(chat.messages)
    const response = operation === "completion"
      ? await request(server, "/api/v1/chat/completions", body)
      : await request(server, `/api/v1/chats/${chat.id}/messages`, { id: "new-edit", role: "user", content: "Edited question",
          tldw_history_selection_v1: body.tldw_history_selection_v1, tldw_history_branch: true, tldw_history_edit_v1: body.tldw_history_edit_v1 })
    expect(response.status).toBe(operation === "completion" ? 200 : 201)
    expect(chat.messages.slice(0, originals.length)).toEqual(originals)
    const input = chat.messages[originals.length]
    expect(input).toMatchObject({ role: "user", content: "Edited question", parent_message_id: originals[1].id })
    if (operation === "completion") expect(chat.messages.at(-1)?.parent_message_id).toBe(input.id)
  })

  it.each(["extra-target-key", "unsupported-version", "missing-branch", "changed-target", "deleted-target", "wrong-target-role", "forged-prefix", "changed-context", "invalid-messages", "invalid-prefix"])(
    "refuses %s without consuming provider work or writing rows", async failure => {
      const server = createFakeTldwServer()
      const { chat, target, body } = await prepare(server)
      if (failure === "extra-target-key") body.tldw_history_edit_v1.extra = true
      if (failure === "unsupported-version") body.tldw_history_edit_v1.version = 2
      if (failure === "missing-branch") body.tldw_history_branch = false
      if (failure === "changed-target") target.version++
      if (failure === "deleted-target") chat.messages = chat.messages.filter(row => row.id !== target.id)
      if (failure === "wrong-target-role") target.role = "assistant"
      if (failure === "forged-prefix") {
        body.tldw_history_selection_v1.messages[0].revision = "999"
        body.tldw_history_selection_v1.selection_digest = selectionDigest(body.tldw_history_selection_v1)
      }
      if (failure === "changed-context") body.messages.at(-1).content = "Unreviewed replacement"
      if (failure === "invalid-messages") body.messages = {}
      if (failure === "invalid-prefix") body.tldw_history_selection_v1.messages = {}
      const originalState = structuredClone(chat.messages)
      server.planCompletion({ reply: "Still queued after refusal" })
      const response = await request(server, "/api/v1/chat/completions", body)
      expect(response.status).toBe(409)
      expect(chat.messages).toEqual(originalState)
      const valid = await prepare(server, "later-valid")
      const answer = await (await request(server, "/api/v1/chat/completions", valid.body)).json()
      expect(answer.choices[0].message.content).toBe("Still queued after refusal")
    }
  )

  it.each(["changed-target", "deleted-target", "wrong-target-role", "duplicate-input"])("append refuses %s without changing rows", async failure => {
    const server = createFakeTldwServer()
    const { chat, target, body } = await prepare(server)
    if (failure === "changed-target") target.version++
    if (failure === "deleted-target") chat.messages = chat.messages.filter(row => row.id !== target.id)
    if (failure === "wrong-target-role") target.role = "assistant"
    const originalState = structuredClone(chat.messages)
    const response = await request(server, `/api/v1/chats/${chat.id}/messages`, {
      id: failure === "duplicate-input" ? chat.messages[0].id : "new-edit", role: "user", content: "Edited question",
      tldw_history_selection_v1: body.tldw_history_selection_v1, tldw_history_branch: true, tldw_history_edit_v1: body.tldw_history_edit_v1
    })
    expect(response.status).toBe(409)
    expect(chat.messages).toEqual(originalState)
  })

  it.each(["deleted-conversation", "replaced-conversation"])("refuses held settlement after %s retirement", async retirement => {
    const server = createFakeTldwServer()
    const { chat, body } = await prepare(server)
    const hold = deferred()
    server.planCompletion({ gate: hold.promise })
    const response = request(server, "/api/v1/chat/completions", body)
    try {
      await waitFor(() => expect(chat.messages).toHaveLength(5))
      const retained = structuredClone(chat.messages)
      if (retirement === "deleted-conversation") server.chats.delete(chat.id)
      else server.seedChat({ id: chat.id, title: "Replacement", turns: [{ role: "user", content: "Another namespace" }] })
      const replacement = server.chats.get(chat.id)
      const replacementRows = replacement ? structuredClone(replacement.messages) : undefined
      hold.resolve()
      expect((await response).status).toBe(409)
      expect(chat.messages).toEqual(retained)
      expect(replacement?.messages).toEqual(replacementRows)
    } finally { hold.resolve() }
  })

  it.each(["completion", "append", "ordinary-append"])("refuses %s admission after conversation replacement during helper loading", async operation => {
    const server = createFakeTldwServer()
    const { chat, body } = await prepare(server)
    const originalRows = structuredClone(chat.messages)
    const path = operation === "completion" ? "/api/v1/chat/completions" : `/api/v1/chats/${chat.id}/messages`
    const input = operation === "completion" ? body : operation === "ordinary-append" ? { id: "ordinary-input", role: "user", content: "Ordinary question" } : { id: "new-edit", role: "user", content: "Edited question",
      tldw_history_selection_v1: body.tldw_history_selection_v1, tldw_history_branch: true, tldw_history_edit_v1: body.tldw_history_edit_v1 }
    const response = request(server, path, input)
    await Promise.resolve()
    expect(server.find("POST", path)).toHaveLength(1)
    const replacement = server.seedChat({ id: chat.id, title: "Replacement", turns: originalRows.map(row => ({ role: row.role as "user" | "assistant", content: row.content })) })
    const replacementRows = structuredClone(replacement.messages)
    expect((await response).status).toBe(409)
    expect(chat.messages).toEqual(originalRows)
    expect(replacement.messages).toEqual(replacementRows)
  })

  it("admits the edited input before provider work and preserves admission after its original target changes", async () => {
    const server = createFakeTldwServer()
    const { chat, target, body } = await prepare(server)
    const hold = deferred()
    server.planCompletion({ gate: hold.promise, reply: "Accepted edit reply" })
    const response = request(server, "/api/v1/chat/completions", body)
    try {
      await waitFor(() => expect(chat.messages).toHaveLength(5))
      const admitted = chat.messages.at(-1)!
      expect(admitted).toMatchObject({ role: "user", content: "Edited question" })
      target.version++
      hold.resolve()
      expect((await response).status).toBe(200)
      expect(chat.messages.at(-1)).toMatchObject({ role: "assistant", parent_message_id: admitted.id })
    } finally { hold.resolve() }
  })

  it.each(["changed-input", "deleted-input"])("refuses held reply settlement after %s without adding an assistant", async failure => {
    const server = createFakeTldwServer()
    const { chat, body } = await prepare(server)
    const hold = deferred()
    server.planCompletion({ gate: hold.promise })
    const response = request(server, "/api/v1/chat/completions", body)
    try {
      await waitFor(() => expect(chat.messages).toHaveLength(5))
      const admitted = chat.messages.at(-1)!
      if (failure === "changed-input") admitted.version++
      else chat.messages = chat.messages.filter(row => row.id !== admitted.id)
      const retained = structuredClone(chat.messages)
      hold.resolve()
      expect((await response).status).toBe(409)
      expect(chat.messages).toEqual(retained)
    } finally { hold.resolve() }
  })
})
