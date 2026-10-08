// @vitest-environment jsdom
/**
 * Red-first reproductions for #3106 (UX G05, CM-01): Regenerate and
 * Edit -> "Save & Send" must work on a loaded saved chat instead of failing
 * with raw error codes. Regenerate remains `it.fails` and is not a positive result. Edit & resend
 * asserts preserved canonical originals and the new selected branch.
 */
import {
  act,
  completionMessages,
  createFakeTldwServer,
  HARNESS_WAIT,
  openSavedServerChat,
  renderPlayground,
  resetPlaygroundHarness,
  screen,
  SidebarServerChatButton,
  visibleNotifications,
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

    // CM-01 #3106 — useChatActions.ts:4766-4781 hands the HistorySelection controller to regenerate whenever it is not
    // idle (any loaded chat), and messageHandlers.ts:88-95 then always throws "unsupported_history_regeneration".
    it.fails(
      "CM-01 (#3106): Regenerate on a loaded server chat sends a completion for the last user turn",
      async () => {
        const view = await renderWithSavedChat()

        await withUnhandledRejectionsRecorded(async () => {
          await view.user.click(
            screen.getByRole("button", { name: "Regenerate" })
          )
          await expectActionSent(view.server, {
            role: "user",
            content: "Saved question"
          })
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
