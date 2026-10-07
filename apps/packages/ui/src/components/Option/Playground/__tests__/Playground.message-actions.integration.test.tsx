// @vitest-environment jsdom
/**
 * Red-first reproductions for #3106 (UX G05, CM-01): Regenerate and
 * Edit -> "Save & Send" must work on a loaded saved chat instead of failing
 * with raw error codes. Each `it.fails` asserts the correct behaviour and
 * passes only while the defect reproduces.
 */
import {
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

describe("Playground message actions on a saved chat (#3106)", { timeout: 60_000 }, () => {
  beforeEach(async () => {
    await resetPlaygroundHarness()
  })

  // CM-01 #3106 — useChatActions.ts:4766-4781 hands the HistorySelection controller to regenerate whenever it is not
  // idle (any loaded chat), and messageHandlers.ts:88-95 then always throws "unsupported_history_regeneration".
  it.fails("CM-01 (#3106): Regenerate on a loaded server chat sends a completion for the last user turn", async () => {
    const view = await renderWithSavedChat()

    await withUnhandledRejectionsRecorded(async () => {
      await view.user.click(screen.getByRole("button", { name: "Regenerate" }))
      await expectActionSent(view.server, { role: "user", content: "Saved question" })
    })
  })

  // CM-01 #3106 — messageHandlers.ts:201-209 always throws "unsupported_history_edit_and_send" for a user message
  // sent from the editor (called from PlaygroundChat.tsx:1424-1425), and EditMessageForm.tsx:41-44 closes the editor first.
  it.fails("CM-01 (#3106): Edit -> Save & Send on a loaded server chat sends the edited text", async () => {
    const view = await renderWithSavedChat()

    await withUnhandledRejectionsRecorded(async () => {
      // The first Edit action belongs to the first (user) message.
      await view.user.click(screen.getAllByRole("button", { name: "Edit" })[0])
      const editor = await waitFor(() => {
        const field = document.querySelector<HTMLTextAreaElement>(
          "form[data-chat-message-editor] textarea"
        )
        expect(field).not.toBeNull()
        return field as HTMLTextAreaElement
      }, HARNESS_WAIT)
      await view.user.clear(editor)
      await view.user.type(editor, "Edited question")
      await view.user.click(screen.getByRole("button", { name: "Save & Send" }))
      await expectActionSent(view.server, { role: "user", content: "Edited question" })
    })
  })
})
