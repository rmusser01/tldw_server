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
  deferred,
  HARNESS_WAIT,
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
import { beforeEach, describe, expect, it } from "vitest"
import { useStoreMessageOption } from "@/store/option"

const seedSavedChat = (server: ReturnType<typeof createFakeTldwServer>) =>
  server.seedChat({
    id: "saved-chat",
    title: "Saved topic",
    turns: [
      { role: "user", content: "Saved question" },
      { role: "assistant", content: "Saved answer" }
    ]
  })

const renderWithSavedChat = async () => {
  const server = createFakeTldwServer()
  seedSavedChat(server)
  const view = await renderPlayground({
    server,
    extras: <SidebarServerChatButton chatId="saved-chat" title="Saved topic" />
  })
  await openSavedServerChat(view, { title: "Saved topic", lastMessage: "Saved answer" })
  return view
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
        const view = await renderWithSavedChat()

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

const savedAssistantRow = () =>
  screen.getAllByTestId("chat-message").filter(row => row.getAttribute("data-role") === "assistant").at(-1)!

const clickContinue = async (view: Awaited<ReturnType<typeof renderPlayground>>) => {
  await view.user.click(within(savedAssistantRow()).getByRole("button", { name: "More actions" }))
  await view.user.click(await screen.findByRole("button", { name: "Continue Response" }, HARNESS_WAIT))
}

describe("native admitted continuation through the actual Playground", { timeout: 60_000 }, () => {
  beforeEach(async () => { await resetPlaygroundHarness() })

  it("CM-01 continues a saved reply as one longer saved alternative and keeps the original", async () => {
    const { view, chat } = await renderWithAdmittedExchange()
    const original = structuredClone(chat.messages)
    const priorCompletions = view.server.completionRequests().length
    view.server.planCompletion({ reply: " and the rest of it." })
    await view.user.type(view.composer, "UNSENT NEXT QUESTION")
    await withUnhandledRejectionsRecorded(async () => {
      await clickContinue(view)
      await waitFor(() => expect(view.server.completionRequests()).toHaveLength(priorCompletions + 1), HARNESS_WAIT)
      await waitForChatIdle()
      expect(visibleNotifications().join(" ")).not.toMatch(/unsupported_history|invalid_history/)
      await waitFor(() => expect(savedAssistantRow()).toHaveTextContent("Saved answer and the rest of it."), HARNESS_WAIT)
      expect(chat.messages.filter(row => original.some(old => old.id === row.id))).toEqual(original)
      const alternative = chat.messages.filter(row => !original.some(old => old.id === row.id))
      expect(alternative).toEqual([expect.objectContaining({
        role: "assistant", content: "Saved answer and the rest of it.", parent_message_id: original[0].id })])
      const request = view.server.completionRequests().at(-1)!.body!
      expect(request).toMatchObject({ save_to_db: true, tldw_history_branch: true,
        metadata: { tldw_continue_from_message_id: original[1].id },
        tldw_history_selection_v1: { cursor: { kind: "after_message", message_id: original[1].id } } })
      expect(request.metadata).not.toHaveProperty("tldw_regenerate_from_message_id")
      expect(view.server.find("POST", new RegExp(`/chats/${chat.id}/messages$`)).filter(row => row.body?.role === "user")).toHaveLength(1)
      expect(view.composer).toHaveValue("UNSENT NEXT QUESTION")
      // The original reply stays available as the other version of this answer.
      expect(within(savedAssistantRow()).getByTestId("variant-count-label")).toHaveTextContent("2 of 2")
    })
  })

  it("CM-01 explains a refused Continue in plain words and keeps the saved reply", async () => {
    const view = await renderWithSavedChat()
    await withUnhandledRejectionsRecorded(async () => {
      await clickContinue(view)
      await waitFor(() => expect(visibleNotifications().length).toBeGreaterThan(0), HARNESS_WAIT)
      const notices = visibleNotifications().join(" ")
      expect(notices).not.toMatch(/unsupported_history|invalid_history/)
      expect(notices).toMatch(/continue/i)
      expect(notices).toContain("original exchange and draft are kept")
      expect(view.server.chats.get("saved-chat")!.messages.map(row => row.content)).toEqual(["Saved question", "Saved answer"])
      expect(screen.getByText("Saved answer")).toBeInTheDocument()
    })
  })
})
