// @vitest-environment jsdom
/**
 * Red-first reproductions for #3106 (UX G05, CM-01): Regenerate and
 * Edit -> "Save & Send" must work on a loaded saved chat instead of failing
 * with raw error codes. Regeneration uses a genuinely admitted saved exchange. Edit & resend
 * asserts preserved canonical originals and the new selected branch.
 */
import {
  act,
  completionMessages,
  createFakeTldwServer,
  getHarnessDb,
  HARNESS_WAIT,
  HeaderNewChatButton,
  openSavedServerChat,
  renderPlayground,
  sendFromComposer,
  waitForChatIdle,
  resetPlaygroundHarness,
  screen,
  SidebarServerChatButton,
  visibleNotifications,
  within,
  waitFor,
  withUnhandledRejectionsRecorded
} from "./harness/playground-harness"
import { deferred } from "./harness/fake-tldw-server"
import { historyDigest } from "@/db/dexie/history-selection"
import { resolveHistorySelection } from "@/utils/history-selection"
import { beforeEach, describe, expect, it } from "vitest"
import { useStoreMessageOption } from "@/store/option"

const seedSavedChat = (
  server: ReturnType<typeof createFakeTldwServer>,
  turns = [
    { role: "user" as const, content: "Saved question" },
    { role: "assistant" as const, content: "Saved answer" }
  ],
  admitted = true
) =>
  server.seedChat({
    id: "saved-chat",
    title: "Saved topic",
    admitted,
    turns
  })

const renderWithSavedChat = async (
  turns?: Parameters<typeof seedSavedChat>[1],
  admitted = true
) => {
  const server = createFakeTldwServer()
  seedSavedChat(server, turns, admitted)
  const view = await renderPlayground({
    server,
    extras: <SidebarServerChatButton chatId="saved-chat" title="Saved topic" />
  })
  await openSavedServerChat(view, {
    title: "Saved topic",
    lastMessage: turns?.at(-1)?.content ?? "Saved answer"
  })
  return view
}

const earlierTurn = [
  { role: "user" as const, content: "First question" },
  { role: "assistant" as const, content: "First answer" },
  { role: "user" as const, content: "Original question" },
  { role: "assistant" as const, content: "Original answer" },
  { role: "user" as const, content: "Later question" },
  { role: "assistant" as const, content: "Later answer" }
]

const editMiddleQuestion = async (view: Awaited<ReturnType<typeof renderWithSavedChat>>) => {
  await view.user.click(screen.getAllByRole("button", { name: "Edit" })[2])
  const editor = await waitFor(() => {
    const field = document.querySelector<HTMLTextAreaElement>(
      "form[data-chat-message-editor] textarea"
    )
    expect(field).not.toBeNull()
    return field as HTMLTextAreaElement
  }, HARNESS_WAIT)
  expect(editor).toHaveValue("Original question")
  await view.user.clear(editor)
  await view.user.type(editor, "Edited question")
  return editor
}

/** The last message of every completion request sent so far. */
const lastCompletionTurns = (server: ReturnType<typeof createFakeTldwServer>) =>
  server.completionRequests().map((request) => completionMessages(request).at(-1))

/**
 * Wait until the action produced an outcome (a completion request or a
 * notification), then require that no raw history error code was shown and
 * that the expected turn was sent.
 */
const expectActionSent = async (
  server: ReturnType<typeof createFakeTldwServer>,
  turn: { role: "user"; content: string }
) => {
  await waitFor(
    () =>
      expect(server.completionRequests().length + visibleNotifications().length).toBeGreaterThan(0),
    HARNESS_WAIT
  )
  expect(visibleNotifications().filter((text) => /unsupported_history/.test(text))).toEqual([])
  await waitFor(() => expect(lastCompletionTurns(server)).toEqual([turn]), HARNESS_WAIT)
}

describe("Playground message actions on a saved chat (#3106)", { timeout: 60_000 }, () => {
  beforeEach(async () => {
    await resetPlaygroundHarness()
  })

  it("CM-01 (#3106): Regenerate adds a sibling answer under the saved question", async () => {
    const view = await renderWithSavedChat([
      { role: "user", content: "Earlier question" },
      { role: "assistant", content: "Earlier answer" },
      { role: "user", content: "Saved question" },
      { role: "assistant", content: "Saved answer" }
    ])
    const chat = view.server.chats.get("saved-chat")!
    const originalRows = structuredClone(chat.messages)
    view.server.setDefaultReply("Alternative answer")

    await withUnhandledRejectionsRecorded(async () => {
      await view.user.click(screen.getByRole("button", { name: "Regenerate" }))
      await expectActionSent(view.server, { role: "user", content: "Saved question" })
      expect(completionMessages(view.server.completionRequests()[0]).filter(
        ({ role }) => role === "user" || role === "assistant"
      )).toEqual([
        { role: "user", content: "Saved question" }
      ])
      expect(view.server.completionRequests()[0].body).toMatchObject({ save_to_db: true, metadata: { tldw_regenerate_from_message_id: originalRows[3].id },
        tldw_history_selection_v1: { messages: originalRows.map(row => ({ id: row.id, revision: String(row.version) })) } })
      await waitForChatIdle()
      await waitFor(() => expect(chat.messages).toHaveLength(5), HARNESS_WAIT)
    })
    expect(chat.messages.slice(0, 4)).toEqual(originalRows)
    expect(chat.messages[4]).toMatchObject({
      role: "assistant",
      content: "Alternative answer",
      parent_message_id: originalRows[2].id
    })
    expect(view.server.find("POST", /\/chats\/saved-chat\/messages$/).filter(
      ({ body }) => body?.role === "user"
    )).toHaveLength(0)
    const settlements = view.server.find("POST", /\/chats\/saved-chat\/messages$/).filter(
      ({ body }) => body?.role === "assistant" && body?.tldw_history_existing_input_v1
    )
    expect(settlements).toHaveLength(0)
  })

  it("legacy existing-input settlement refuses occupied answer ID after question or result state changes", async () => {
    const view = await renderWithSavedChat()
    const chat = view.server.chats.get("saved-chat")!
    const captured = await (await view.server.fetch(view.server.url + "/api/v1/chat/conversations/saved-chat/history/selection", {
      method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ purpose: "send",
        required_capability: "tldw_history_existing_input_v1", view: { view_session_id: "legacy-existing", owner_key: "harness-native-owner",
          conversation_id: chat.id, interpretation: { kind: "parent_graph_v1" }, cursor: { kind: "after_message", message_id: chat.messages[0].id }, selection_revision: 1 } })
    })).json()
    const selected = resolveHistorySelection(captured.snapshot, captured.view, "send", historyDigest({ fixture: "legacy-existing-replay" }))
    if (selected.status !== "ready") throw new Error("Expected owned legacy question")
    const settlement = { body: { id: crypto.randomUUID(), role: "assistant", content: "Alternative answer",
      parent_message_id: chat.messages[0].id, tldw_history_existing_input_v1: selected.selection } }
    const replay = (body = settlement.body) => view.server.fetch("http://tldw.test/api/v1/chats/saved-chat/messages", {
      method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify(body)
    })
    expect((await replay()).status).toBe(201)
    expect((await replay(Object.fromEntries(Object.entries(settlement.body!).reverse()))).status).toBe(201)
    chat.messages[0].version += 1
    expect((await replay()).status).toBe(409)
    chat.messages[0].version -= 1
    chat.messages[2].content = "edited alternative"
    expect((await replay()).status).toBe(409)
  })

  it("Regenerate leaves the original selected answer visible while the provider is paused", async () => {
    const view = await renderWithSavedChat()
    const chat = view.server.chats.get("saved-chat")!
    const originalRows = structuredClone(chat.messages)
    const release = deferred()
    view.server.planCompletion({ gate: release.promise, reply: "Alternative answer" })
    await view.user.click(screen.getByRole("button", { name: "Regenerate" }))
    try {
      await waitFor(() => expect(view.server.completionRequests()).toHaveLength(1), HARNESS_WAIT)
      expect(screen.getByText("Saved answer")).toBeVisible()
      expect(chat.messages).toEqual(originalRows)
      expect(view.server.find("POST", /\/chats\/saved-chat\/messages$/)).toHaveLength(0)
    } finally {
      release.resolve()
      await waitForChatIdle()
    }
  })

  it("Regenerate refuses a question edited away and back before provider dispatch", async () => {
    const view = await renderWithSavedChat()
    const chat = view.server.chats.get("saved-chat")!
    const releaseCapture = deferred()
    const originalFetch = globalThis.fetch
    let held = false
    globalThis.fetch = async (input, init) => {
      const url = new URL(typeof input === "string" ? input : input instanceof URL ? input.href : input.url,
        view.server.url)
      const raw = init?.body ?? (input instanceof Request ? await input.clone().text() : "")
      if (url.pathname.endsWith("/history/selection") && raw &&
          JSON.parse(String(raw)).view?.cursor?.message_id === chat.messages.at(-1)?.id) {
        held = true
        await releaseCapture.promise
      }
      return originalFetch(input, init)
    }
    try {
      await view.user.click(screen.getByRole("button", { name: "Regenerate" }))
      await waitFor(() => expect(held).toBe(true), HARNESS_WAIT)
      chat.messages[0].content = "temporarily changed"
      chat.messages[0].version += 1
      chat.messages[0].content = "Saved question"
      chat.messages[0].version += 1
      releaseCapture.resolve()
      await waitForChatIdle()
      await waitFor(() => expect(visibleNotifications().join(" | ")).toContain("The saved message or conversation changed. Your original exchange and draft are kept."), HARNESS_WAIT)
      expect(view.server.completionRequests()).toHaveLength(0)
      expect(view.server.find("POST", /\/chats\/saved-chat\/messages$/)).toHaveLength(0)
      expect(chat.messages).toHaveLength(2)
    } finally {
      releaseCapture.resolve()
      try {
        await waitForChatIdle()
      } finally {
        globalThis.fetch = originalFetch
      }
    }
  })

  // CM-01 #3106 — selected user Edit-and-Send currently throws before capturing its ancestry.
  it("CM-01 (#3106): Edit -> Save & Send branches from the selected user without rewriting later turns", async () => {
    const view = await renderWithSavedChat(earlierTurn)
    view.server.setDefaultReply("Edited answer")
    const chat = view.server.chats.get("saved-chat")!
    const originalRows = structuredClone(chat.messages)
    const capturesBefore = view.server.find("POST", /\/history\/selection$/).length
    let unhandled: unknown[] = []

    await withUnhandledRejectionsRecorded(async (reasons) => {
      unhandled = reasons
      await editMiddleQuestion(view)
      await view.user.click(screen.getByRole("button", { name: "Save & Send" }))
      await waitFor(() => {
        const alert = document.querySelector("form[data-chat-message-editor] [role=alert]")?.textContent ?? ""
        expect(
          view.server.completionRequests().length,
          `editor=${alert}; notices=${visibleNotifications().join(" | ")}; captures=${view.server.find("POST", /\/history\/selection$/).length}`
        ).toBe(1)
      }, HARNESS_WAIT)
      const prompt = completionMessages(view.server.completionRequests()[0]).filter(
        ({ role }) => role === "user" || role === "assistant"
      )
      expect(prompt).toEqual([
        { role: "user", content: "First question" },
        { role: "assistant", content: "First answer" },
        { role: "user", content: "Edited question" }
      ])
      expect(prompt).not.toContainEqual({ role: "assistant", content: "Original answer" })
      const sendCaptures = view.server
        .find("POST", /\/history\/selection$/)
        .slice(capturesBefore)
        .filter(({ body }) => body?.purpose === "send" && body?.view?.cursor?.kind === "before_message" &&
          body.view.cursor.message_id === "saved-chat-m3")
      expect(sendCaptures).toHaveLength(1)
      expect(sendCaptures[0].body?.view?.cursor).toEqual({
        kind: "before_message",
        message_id: "saved-chat-m3"
      })
      await waitForChatIdle()
      await waitFor(() => expect(chat.messages).toHaveLength(8), HARNESS_WAIT)
      expect(chat.messages.slice(0, 6)).toEqual(originalRows)
      const [editedUser, editedAnswer] = chat.messages.slice(6)
      expect([editedUser.role, editedUser.content, editedUser.parent_message_id]).toEqual([
        "user", "Edited question", "saved-chat-m2"
      ])
      expect([editedAnswer.role, editedAnswer.content, editedAnswer.parent_message_id]).toEqual([
        "assistant", "Edited answer", editedUser.id
      ])
      expect(new Set(chat.messages.map(({ id }) => id)).size).toBe(8)
      await waitFor(
        () => expect(document.querySelector("form[data-chat-message-editor]")).toBeNull(),
        HARNESS_WAIT
      )
    })
    expect(unhandled).toEqual([])
  })

  it("CM-01 (#3106): a rejected Edit-and-Send keeps the edited draft mounted", async () => {
    const view = await renderWithSavedChat(earlierTurn)
    const chat = view.server.chats.get("saved-chat")!
    const originalRows = structuredClone(chat.messages)
    const rejectedInputs: Array<{
      content?: string
      tldw_history_selection_v1?: { messages?: Array<{ id: string }> }
    }> = []
    const admission = deferred()
    const originalFetch = globalThis.fetch
    let unhandled: unknown[] = []
    globalThis.fetch = async (input, init) => {
      const url = new URL(
        typeof input === "string" ? input : input instanceof URL ? input.href : input.url,
        view.server.url
      )
      const method = String(init?.method ?? (input instanceof Request ? input.method : "GET"))
      if (method === "POST" && url.pathname === "/api/v1/chats/saved-chat/messages") {
        const raw = init?.body ?? (input instanceof Request ? await input.clone().text() : "")
        const body = JSON.parse(String(raw))
        if (body.tldw_history_selection_v1) {
          rejectedInputs.push(body)
          await admission.promise
          return new Response(JSON.stringify({
            detail: { status: "stale_selection", code: "stale_source" }
          }), {
            status: 409,
            headers: { "content-type": "application/json" }
          })
        }
      }
      return originalFetch(input, init)
    }
    try {
      await withUnhandledRejectionsRecorded(async (reasons) => {
        unhandled = reasons
        const editor = await editMiddleQuestion(view)
        const form = editor.closest("form[data-chat-message-editor]")
        try {
          await view.user.click(screen.getByRole("button", { name: "Save & Send" }))
          await waitFor(() => expect(rejectedInputs).toHaveLength(1), HARNESS_WAIT)
          expect(document.querySelector("form[data-chat-message-editor] textarea")).toBe(editor)
          expect(editor.isConnected).toBe(true)
          expect(form).toHaveAttribute("aria-busy", "true")
          expect(editor).toHaveValue("Edited question")
          expect(screen.getByText("Later answer")).toBeInTheDocument()
          expect(view.server.completionRequests()).toHaveLength(0)
          expect(chat.messages).toEqual(originalRows)
        } finally {
          admission.resolve()
          if (rejectedInputs.length) await waitForChatIdle()
        }
        expect(rejectedInputs[0].content).toBe("Edited question")
        expect(rejectedInputs[0].tldw_history_selection_v1?.messages?.map(({ id }) => id)).toEqual([
          "saved-chat-m1", "saved-chat-m2"
        ])
        expect(view.server.completionRequests()).toHaveLength(0)
        expect(chat.messages).toEqual(originalRows)
        await waitFor(() => {
          expect(document.querySelector("form[data-chat-message-editor] textarea")).toBe(editor)
          expect(form).toHaveAttribute("aria-busy", "false")
          expect(editor).toHaveValue("Edited question")
        }, HARNESS_WAIT)
        expect(screen.getByText("Later answer")).toBeInTheDocument()
      })
    } finally {
      admission.resolve()
      globalThis.fetch = originalFetch
      expect(unhandled).toEqual([])
    }
  })

  it("CM-01 (#3106): a replaced selection during Edit-and-Send preparation never admits the old draft", async () => {
    const server = createFakeTldwServer()
    seedSavedChat(server, earlierTurn)
    const view = await renderPlayground({
      server,
      extras: <><SidebarServerChatButton chatId="saved-chat" title="Saved topic" /><HeaderNewChatButton /></>
    })
    await openSavedServerChat(view, { title: "Saved topic", lastMessage: "Later answer" })
    const originalRows = structuredClone(server.chats.get("saved-chat")!.messages)
    const capture = deferred()
    const selectedCaptures: unknown[] = []
    const originalFetch = globalThis.fetch
    globalThis.fetch = async (input, init) => {
      const url = new URL(
        typeof input === "string" ? input : input instanceof URL ? input.href : input.url,
        server.url
      )
      const raw = init?.body ?? (input instanceof Request ? await input.clone().text() : "")
      if (
        String(init?.method ?? (input instanceof Request ? input.method : "GET")) === "POST" &&
        /\/history\/selection$/.test(url.pathname) &&
        raw && JSON.parse(String(raw)).purpose === "send"
      ) {
        selectedCaptures.push(JSON.parse(String(raw)))
        await capture.promise
      }
      return originalFetch(input, init)
    }
    let unhandled: unknown[] = []
    try {
      await withUnhandledRejectionsRecorded(async (reasons) => {
        unhandled = reasons
        await editMiddleQuestion(view)
        try {
          await view.user.click(screen.getByRole("button", { name: "Save & Send" }))
          await waitFor(() => expect(selectedCaptures).toHaveLength(1), HARNESS_WAIT)
          await view.user.click(screen.getByRole("button", { name: "Harness header: New chat" }))
        } finally {
          capture.resolve()
        }
        await waitForChatIdle()
        expect(server.find("POST", "/api/v1/chats/saved-chat/messages")).toHaveLength(0)
        expect(server.completionRequests()).toHaveLength(0)
        expect(server.chats.get("saved-chat")!.messages).toEqual(originalRows)
      })
    } finally {
      capture.resolve()
      globalThis.fetch = originalFetch
      expect(unhandled).toEqual([])
    }
  })

  it("CM-01 (#3106): lost admission acknowledgement cannot resend the edited draft with a new ID", async () => {
    const view = await renderWithSavedChat(earlierTurn)
    const chat = view.server.chats.get("saved-chat")!
    const originalRows = structuredClone(chat.messages)
    const selectedInputs: Array<{ id: string }> = []
    const originalFetch = globalThis.fetch
    globalThis.fetch = async (input, init) => {
      const url = new URL(
        typeof input === "string" ? input : input instanceof URL ? input.href : input.url,
        view.server.url
      )
      const raw = init?.body ?? (input instanceof Request ? await input.clone().text() : "")
      const body = raw ? JSON.parse(String(raw)) : null
      if (
        String(init?.method ?? (input instanceof Request ? input.method : "GET")) === "POST" &&
        url.pathname === "/api/v1/chats/saved-chat/messages" &&
        body?.tldw_history_selection_v1
      ) {
        selectedInputs.push({ id: body.id })
        await originalFetch(input, init)
        throw new TypeError("connection lost after admission")
      }
      return originalFetch(input, init)
    }
    let unhandled: unknown[] = []
    try {
      await withUnhandledRejectionsRecorded(async (reasons) => {
        unhandled = reasons
        const editor = await editMiddleQuestion(view)
        await view.user.click(screen.getByRole("button", { name: "Save & Send" }))
        await waitFor(() => expect(selectedInputs).toHaveLength(1), HARNESS_WAIT)
        await waitForChatIdle()
        expect(chat.messages.slice(0, 6)).toEqual(originalRows)
        expect(chat.messages).toHaveLength(7)
        expect(chat.messages[6].id).toBe(selectedInputs[0].id)
        expect(view.server.completionRequests()).toHaveLength(0)
        await waitFor(() => {
          expect(document.querySelector("form[data-chat-message-editor] textarea")).toBe(editor)
          expect(editor.closest("form")).toHaveAttribute("aria-busy", "false")
          expect(editor).toHaveValue("Edited question")
        }, HARNESS_WAIT)
        await view.user.click(screen.getByRole("button", { name: "Save & Send" }))
        await waitForChatIdle()
        expect(selectedInputs).toHaveLength(1)
        expect(chat.messages).toHaveLength(7)
        expect(view.server.completionRequests()).toHaveLength(0)
      })
    } finally {
      globalThis.fetch = originalFetch
      expect(unhandled).toEqual([])
    }
  })

  it("CM-01 (#3106): a view replaced after Edit-and-Send admission cannot dispatch the old provider turn", async () => {
    const view = await renderWithSavedChat(earlierTurn)
    const chat = view.server.chats.get("saved-chat")!
    const originalRows = structuredClone(chat.messages)
    const savedView = async () => (await getHarnessDb().historySelections.toArray())
      .find((row) => row.conversation_id === "saved-chat")?.view
    const originalSelection = await savedView()
    const capturesBefore = view.server.find("POST", /\/history\/selection$/).length
    const acknowledgement = deferred()
    const admittedInputs: Array<{ id: string; content: string }> = []
    const originalFetch = globalThis.fetch
    globalThis.fetch = async (input, init) => {
      const url = new URL(
        typeof input === "string" ? input : input instanceof URL ? input.href : input.url,
        view.server.url
      )
      const raw = init?.body ?? (input instanceof Request ? await input.clone().text() : "")
      const body = raw ? JSON.parse(String(raw)) : null
      if (
        String(init?.method ?? (input instanceof Request ? input.method : "GET")) === "POST" &&
        url.pathname === "/api/v1/chats/saved-chat/messages" &&
        body?.tldw_history_selection_v1
      ) {
        const response = await originalFetch(input, init)
        admittedInputs.push({ id: body.id, content: body.content })
        await acknowledgement.promise
        return response
      }
      return originalFetch(input, init)
    }
    let unhandled: unknown[] = []
    try {
      await withUnhandledRejectionsRecorded(async (reasons) => {
        unhandled = reasons
        try {
          await editMiddleQuestion(view)
          await view.user.click(screen.getByRole("button", { name: "Save & Send" }))
          await waitFor(() => expect(admittedInputs).toHaveLength(1), HARNESS_WAIT)
          expect(chat.messages.slice(0, 6)).toEqual(originalRows)
          expect(chat.messages).toHaveLength(7)
          expect(view.server.completionRequests()).toHaveLength(0)
          await view.user.click(screen.getByRole("button", { name: "Use no prior messages" }))
          await waitFor(() => {
            const captures = view.server.find("POST", /\/history\/selection$/).slice(capturesBefore)
            expect(captures.some(({ body }) => body?.view?.cursor?.kind === "empty")).toBe(true)
          }, HARNESS_WAIT)
          const replacement = await waitFor(async () => {
            const current = await savedView()
            expect(current?.cursor).toEqual({ kind: "empty" })
            expect(current?.selection_revision).toBeGreaterThan(originalSelection?.selection_revision ?? -1)
            return current
          }, HARNESS_WAIT)
          acknowledgement.resolve()
          await waitForChatIdle()
          expect(view.server.completionRequests()).toHaveLength(0)
          expect(chat.messages.slice(0, 6)).toEqual(originalRows)
          expect(chat.messages).toHaveLength(7)
          expect([chat.messages[6].id, chat.messages[6].role, chat.messages[6].content, chat.messages[6].parent_message_id]).toEqual([
            admittedInputs[0].id, "user", "Edited question", "saved-chat-m2"
          ])
          expect(await savedView()).toEqual(replacement)
        } finally {
          acknowledgement.resolve()
          if (admittedInputs.length) await waitForChatIdle()
        }
      })
    } finally {
      acknowledgement.resolve()
      globalThis.fetch = originalFetch
      expect(unhandled).toEqual([])
    }

})
})

describe(
  "Playground message actions on a saved chat (#3106)",
  { timeout: 60_000 },
  () => {
    beforeEach(async () => {
      await resetPlaygroundHarness()
    })

    it("keeps edited text and canonical originals when protected edit support is absent", async () => {
      const server = createFakeTldwServer()
      seedSavedChat(server)
      const originals = structuredClone(server.chats.get("saved-chat")!.messages)
      const fetch = server.fetch
      server.fetch = async (...args: Parameters<typeof fetch>) => {
        const response = await fetch(...args)
        const input = args[0]
        const url = new URL(typeof input === "string" ? input : input instanceof URL ? input.href : input.url)
        if (url.pathname !== "/openapi.json") return response
        const spec = await response.json()
        for (const path of ["/api/v1/chat/completions", "/api/v1/chats/{chat_id}/messages"])
          delete spec.paths[path].post.requestBody.content["application/json"].schema.properties.tldw_history_edit_v1
        return new Response(JSON.stringify(spec), { headers: { "Content-Type": "application/json" } })
      }
      const view = await renderPlayground({ server, extras: <SidebarServerChatButton chatId="saved-chat" title="Saved topic" /> })
      await openSavedServerChat(view, { title: "Saved topic", lastMessage: "Saved answer" })
      await view.user.click(screen.getAllByRole("button", { name: "Edit" })[0])
      const editor = document.querySelector<HTMLTextAreaElement>("form[data-chat-message-editor] textarea")!
      await view.user.clear(editor)
      await view.user.type(editor, "Keep this private edit")
      await view.user.click(screen.getByRole("button", { name: "Save & Send" }))
      await screen.findByText(/does not support this edit action/i, {}, HARNESS_WAIT)
      expect(editor).toHaveValue("Keep this private edit")
      expect(server.completionRequests()).toHaveLength(0)
      expect(server.find("POST", /\/chats\/saved-chat\/messages$/)).toHaveLength(0)
      expect(server.chats.get("saved-chat")!.messages).toEqual(originals)
    })

    it("a held edited reply keeps the original assistant passive and the composer Stop active", async () => {
      const view = await renderWithSavedChat()
      const hold = deferred()
      view.server.planCompletion({ gate: hold.promise })
      try {
        await view.user.click(
          screen.getAllByRole("button", { name: "Edit" })[0]
        )
        const editor = document.querySelector<HTMLTextAreaElement>(
          "form[data-chat-message-editor] textarea"
        )!
        await view.user.clear(editor)
        await view.user.type(editor, "Edited question")
        await view.user.click(
          screen.getByRole("button", { name: "Save & Send" })
        )
        await waitFor(
          () => expect(view.server.completionRequests()).toHaveLength(1),
          HARNESS_WAIT
        )
        expect(useStoreMessageOption.getState().streaming).toBe(true)
        const source = screen
          .getAllByTestId("chat-message")
          .find((row) => row.getAttribute("data-role") === "assistant")!
        expect(source).toHaveTextContent("Saved answer")
        expect(source).not.toHaveAttribute("aria-busy")
        expect(
          within(source).queryByTestId("chat-message-stop-streaming")
        ).not.toBeInTheDocument()
        expect(
          within(source).queryByTestId("playground-streaming-plain-text")
        ).not.toBeInTheDocument()
        expect(
          within(source).queryByText(/Generating response|Processing\.\.\./)
        ).not.toBeInTheDocument()
        expect(
          screen.getByRole("button", { name: /Stop streaming/i })
        ).toBeEnabled()
        expect(
          screen.getByRole("button", { name: "Save & Send" })
        ).toBeDisabled()
      } finally {
        hold.resolve()
        await waitFor(
          () => expect(useStoreMessageOption.getState().streaming).toBe(false),
          HARNESS_WAIT
        )
      }
    })

    it("final edited-result publication cannot overwrite a synchronous replacement draft", async () => {
      const view = await renderWithSavedChat()
      const replacement = [
        { id: "new-user", isBot: false, name: "You", message: "NEW DRAFT" }
      ]
      let replaced = false
      const unsubscribe = useStoreMessageOption.subscribe((state, previous) => {
        if (
          !replaced &&
          state.history !== previous.history &&
          state.history.some((row) => row.content === "Edited question")
        ) {
          replaced = true
          state.startNewChatDraft()
          useStoreMessageOption.setState({
            historyId: null,
            serverChatId: null,
            history: [{ role: "user", content: "NEW DRAFT" }],
            messages: replacement
          })
        }
      })
      try {
        await view.user.click(
          screen.getAllByRole("button", { name: "Edit" })[0]
        )
        const editor = document.querySelector<HTMLTextAreaElement>(
          "form[data-chat-message-editor] textarea"
        )!
        await view.user.clear(editor)
        await view.user.type(editor, "Edited question")
        await view.user.click(
          screen.getByRole("button", { name: "Save & Send" })
        )
        await waitFor(() => expect(replaced).toBe(true), HARNESS_WAIT)
        await act(async () => {
          await Promise.resolve()
        })
        expect(useStoreMessageOption.getState().messages).toEqual(replacement)
        expect(useStoreMessageOption.getState().history).toEqual([
          { role: "user", content: "NEW DRAFT" }
        ])
      } finally {
        unsubscribe()
      }
    })

    // Raw legacy seed rows carry no immutable input admission and remain an honest negative control.
    it(
      "CM-01 (#3106): an unqualified legacy saved reply refuses regeneration without duplicating its question",
      async () => {
        const view = await renderWithSavedChat(undefined, false)

        await withUnhandledRejectionsRecorded(async () => {
          await view.user.click(
            screen.getByRole("button", { name: "Regenerate" })
          )
          await waitFor(() => expect(visibleNotifications().length).toBeGreaterThan(0), HARNESS_WAIT)
          expect(visibleNotifications().join(" ")).toContain("original exchange and draft are kept")
          expect(visibleNotifications().join(" ")).not.toMatch(/unsupported_history|invalid_history/)
          expect(view.server.chats.get("saved-chat")!.messages.map(row => row.content)).toEqual(["Saved question", "Saved answer"])
          expect(screen.getByText("Saved answer")).toBeInTheDocument()
        })
      }
    )

    // CM-01 #3106 — messageHandlers.ts:201-209 always throws "unsupported_history_edit_and_send" for a user message
    // sent from the editor (called from PlaygroundChat.tsx:1424-1425), and EditMessageForm.tsx:41-44 closes the editor first.
    it("CM-01 (#3106): Edit -> Save & Send on a loaded server chat sends the edited text", async () => {
      const view = await renderWithSavedChat()
      const originals = structuredClone(
        view.server.chats.get("saved-chat")!.messages
      )

      await withUnhandledRejectionsRecorded(async () => {
        // The first Edit action belongs to the first (user) message.
        await view.user.click(
          screen.getAllByRole("button", { name: "Edit" })[0]
        )
        const editor = await waitFor(() => {
          const field = document.querySelector<HTMLTextAreaElement>(
            "form[data-chat-message-editor] textarea"
          )
          expect(field).not.toBeNull()
          return field as HTMLTextAreaElement
        }, HARNESS_WAIT)
        await view.user.clear(editor)
        await view.user.type(editor, "Edited question")
        await view.user.click(
          screen.getByRole("button", { name: "Save & Send" })
        )
        await expectActionSent(view.server, {
          role: "user",
          content: "Edited question"
        })
        await waitFor(() => {
          expect(
            document.querySelector("form[data-chat-message-editor]")
          ).toBeNull()
          expect(screen.getByText("Edited question")).toBeInTheDocument()
          expect(screen.queryByText("Saved question")).not.toBeInTheDocument()
        }, HARNESS_WAIT)
        const saved = view.server.chats.get("saved-chat")!.messages
        expect(
          saved.filter((row) =>
            originals.some((original) => original.id === row.id)
          )
        ).toEqual(originals)
        const newUser = saved.find(
          (row) =>
            row.role === "user" &&
            !originals.some((original) => original.id === row.id)
        )!
        expect(newUser).toMatchObject({
          content: "Edited question",
          parent_message_id: originals[0].parent_message_id
        })
        expect(
          saved.filter(
            (row) =>
              row.role === "assistant" &&
              !originals.some((original) => original.id === row.id)
          )
        ).toEqual([expect.objectContaining({ parent_message_id: newUser.id })])
        const admitted = view.server
          .find("POST", /\/chats\/saved-chat\/messages$/)
          .filter((request) => request.body?.role === "user")
        expect(admitted).toHaveLength(1)
        expect(admitted[0].body).toMatchObject({
          tldw_history_branch: true,
          tldw_history_selection_v1: {
            cursor: { kind: "before_message", message_id: originals[0].id }
          }
        })
      })
    })
  }
)


const renderWithAdmittedExchange = async () => {
  const server = createFakeTldwServer(); server.setDefaultReply("Saved answer")
  const initial = await renderPlayground({ server })
  await sendFromComposer(initial, "Saved question")
  expect(server.chats.size).toBe(1)
  const chat = [...server.chats.values()][0]
  initial.unmount()
  const view = await renderPlayground({ server, keepBrowserState: true,
    extras: <SidebarServerChatButton chatId={chat.id} title={chat.title} /> })
  await openSavedServerChat(view, { title: chat.title, lastMessage: "Saved answer" })
  return { view, chat }
}

describe("native admitted regeneration through the actual Playground", { timeout: 60_000 }, () => {
  beforeEach(async () => { await resetPlaygroundHarness() })
  it.each([false, true])("CM-01 preserves one canonical question, original reply and unsent composer through a held reply %s", async held => {
    const { view, chat } = await renderWithAdmittedExchange()
    const original = structuredClone(chat.messages)
    const priorCompletions = view.server.completionRequests().length
    const hold = deferred()
    view.server.planCompletion({ reply: "Regenerated alternative", ...(held ? { gate: hold.promise } : {}) })
    await view.user.type(view.composer, "UNSENT NEXT QUESTION")
    try {
      await withUnhandledRejectionsRecorded(async () => {
        await view.user.click(screen.getByRole("button", { name: "Regenerate" }))
        await waitFor(() => expect(view.server.completionRequests()).toHaveLength(priorCompletions + 1), HARNESS_WAIT)
        if (held) {
          expect(useStoreMessageOption.getState().streaming).toBe(true)
          const saved = screen.getAllByTestId("chat-message").find(row => row.getAttribute("data-role") === "assistant")!
          expect(saved).toHaveTextContent("Saved answer")
          expect(saved).not.toHaveAttribute("aria-busy")
          expect(within(saved).queryByTestId("chat-message-stop-streaming")).toBeNull()
          expect(screen.getByRole("button", { name: /Stop streaming/i })).toBeEnabled()
          expect(view.composer).toHaveValue("UNSENT NEXT QUESTION")
        }
        hold.resolve(); await waitForChatIdle()
        expect(useStoreMessageOption.getState().messages.map(row => row.message), JSON.stringify({
          notices: visibleNotifications(), canonical: chat.messages.map(row => ({ id: row.id, role: row.role, content: row.content }))
        })).toContain("Regenerated alternative")
        await waitFor(() => expect(screen.getAllByTestId("chat-message")
          .some(row => row.getAttribute("data-role") === "assistant" && row.textContent?.includes("Regenerated alternative"))).toBe(true), HARNESS_WAIT)
        expect(view.composer).toHaveValue("UNSENT NEXT QUESTION")
        expect(chat.messages.filter(row => original.some(old => old.id === row.id))).toEqual(original)
        expect(chat.messages.filter(row => row.role === "user")).toEqual([original[0]])
        const alternative = chat.messages.filter(row => !original.some(old => old.id === row.id))
        expect(alternative).toEqual([expect.objectContaining({ role: "assistant", content: "Regenerated alternative", parent_message_id: original[0].id })])
        const request = view.server.completionRequests().at(-1)!.body!
        expect(request).toMatchObject({ save_to_db: true, tldw_history_branch: true,
          metadata: { tldw_regenerate_from_message_id: original[1].id },
          tldw_history_selection_v1: { cursor: { kind: "after_message", message_id: original[1].id } } })
        expect(view.server.find("POST", new RegExp(`/chats/${chat.id}/messages$`)).filter(row => row.body?.role === "user")).toHaveLength(1)
        expect(visibleNotifications().join(" ")).not.toMatch(/unsupported_history|invalid_history/)
      })
    } finally { hold.resolve(); await waitForChatIdle() }
  })
})
