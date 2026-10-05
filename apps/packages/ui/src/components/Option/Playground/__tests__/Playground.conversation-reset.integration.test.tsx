// @vitest-environment jsdom
/**
 * Regression tests for #3106 (UX G05): "New chat" and "Clear conversation"
 * must start a clean conversation. The CS-01 and CS-05 cases began as red-first
 * `it.fails` reproductions; the reset of the HistorySelection controller and
 * the persisted session on every New chat/Clear path turned them into `it`.
 */
import {
  clickHeaderNewChat,
  completionMessages,
  createFakeTldwServer,
  deferred,
  FAKE_API_KEY,
  FAKE_SERVER_URL,
  HARNESS_WAIT,
  HeaderNewChatButton,
  openSavedServerChat,
  renderPlayground,
  resetPlaygroundHarness,
  screen,
  sendFromComposer,
  SidebarServerChatButton,
  waitFor,
  type FakeTldwServer,
  type PlaygroundView
} from "./harness/playground-harness"
import { beforeEach, describe, expect, it } from "vitest"
import { ChatSidebar } from "@/components/Common/ChatSidebar"
import { useClearChat } from "@/hooks/chat/useClearChat"
import { useChatShortcuts } from "@/hooks/keyboard/useKeyboardShortcuts"
import { buildChatSurfaceScopeKeyFromConfig } from "@/services/chat-surface-scope"
import { useStoreMessageOption } from "@/store/option"
import { usePlaygroundSessionStore } from "@/store/playground-session"

/** The sessionStorage key the Playground's HistorySelectionProvider keeps its reference under. */
const PLAYGROUND_REFERENCE_KEY = "tldw-h1-playground-reference"

/**
 * Stand-in for the app layout's Ctrl+Shift+U binding (Layout.tsx `useChatShortcuts(clearChat)`):
 * the real shortcut hook wired to the real `useClearChat`, outside the Playground's provider.
 */
const NewChatShortcut = () => {
  const clearChat = useClearChat()
  useChatShortcuts(clearChat, true)
  return null
}

const sendButton = () => screen.getByRole("button", { name: "Send message" })

/** Send a first local turn, then start a new chat through `startNewChat` and send an unrelated question. */
const expectNewChatStartsClean = async (
  view: PlaygroundView,
  startNewChat: () => Promise<void>
) => {
  await sendFromComposer(view, "First topic question")
  const previousHistoryId = useStoreMessageOption.getState().historyId as string
  await startNewChat()
  await sendFromComposer(view, "Unrelated question")
  expect(completionMessages(view.server.completionRequests()[1])).toEqual([
    { role: "user", content: "Unrelated question" }
  ])
  expect(useStoreMessageOption.getState().historyId).not.toBe(previousHistoryId)
  expect(screen.queryByText("First topic question")).not.toBeInTheDocument()
}

/** A server chat's messages as sorted "role: content" lines (a turn's rows can share a timestamp). */
const serverMessages = (server: FakeTldwServer, chatId: string) =>
  (server.chats.get(chatId)?.messages ?? []).map((message) => `${message.role}: ${message.content}`).sort()

describe("Playground conversation reset (#3106)", { timeout: 60_000 }, () => {
  beforeEach(async () => {
    await resetPlaygroundHarness()
  })

  // CS-01 #3106 — useClearChat.ts:93-150 resets messages/history/historyId/serverChatId but never the
  // HistorySelection controller; useChatActions.ts:3450-3455 and normalChatMode.ts:642-643 then reuse the stale owner.
  // A fresh chat on a connected server is owned by a server chat from its first
  // send (#3195), so "the previous chat" is that server chat.
  it("CS-01 (#3106): after New chat the next request carries no prior messages and is not appended to the previous chat", async () => {
    const server = createFakeTldwServer()
    const view = await renderPlayground({ server, extras: <HeaderNewChatButton /> })
    await sendFromComposer(view, "First topic question")
    const previousChatId = useStoreMessageOption.getState().serverChatId as string
    expect(previousChatId).toBeTruthy()
    expect(serverMessages(server, previousChatId)).toHaveLength(2)

    await clickHeaderNewChat(view)
    await sendFromComposer(view, "Unrelated question")

    expect(completionMessages(server.completionRequests()[1])).toEqual([
      { role: "user", content: "Unrelated question" }
    ])
    expect(serverMessages(server, previousChatId)).toEqual([
      "assistant: Harness reply",
      "user: First topic question"
    ])
    expect(useStoreMessageOption.getState().serverChatId).not.toBe(previousChatId)
    expect(screen.queryByText("First topic question")).not.toBeInTheDocument()
  })

  // CS-01 #3106 — same root cause: after a saved server chat was opened, the native owner survives New chat
  // (useClearChat.ts:93-150), so the next send is admitted into the old conversation (normalChatMode.ts:642-643).
  it("CS-01 (#3106): after New chat from a saved server chat the next send posts nothing to that chat and carries no prior messages", async () => {
    const server = createFakeTldwServer()
    server.seedChat({
      id: "saved-chat",
      title: "Saved topic",
      turns: [
        { role: "user", content: "Saved question" },
        { role: "assistant", content: "Saved answer" }
      ]
    })
    const view = await renderPlayground({
      server,
      extras: (
        <>
          <HeaderNewChatButton />
          <SidebarServerChatButton chatId="saved-chat" title="Saved topic" />
        </>
      )
    })
    await openSavedServerChat(view, { title: "Saved topic", lastMessage: "Saved answer" })

    await clickHeaderNewChat(view)
    await sendFromComposer(view, "Unrelated question")

    expect(server.find("POST", "/api/v1/chats/saved-chat/messages")).toEqual([])
    expect(server.completionRequests().map(completionMessages)).toEqual([
      [{ role: "user", content: "Unrelated question" }]
    ])
    expect(server.chats.get("saved-chat")?.messages).toHaveLength(2)
  })

  // CS-05 #3106 — PlaygroundForm.tsx:3468-3497 handleClearContext only calls setHistory([]) (L3487); messages and
  // the HistorySelection cursor stay, and normalChatMode.ts:848-886 rebuilds the request from that selection.
  it("CS-05 (#3106): Clear conversation removes the messages and the next request carries no prior messages", async () => {
    const server = createFakeTldwServer()
    const view = await renderPlayground({ server })
    await sendFromComposer(view, "Sensitive first question")

    await view.user.click(screen.getByRole("button", { name: "More tools" }))
    await view.user.click(await screen.findByText("Clear conversation"))
    await view.user.click(await screen.findByRole("button", { name: "Confirm" }))

    await waitFor(
      () => expect(screen.queryByText("Sensitive first question")).not.toBeInTheDocument(),
      { timeout: 3_000 }
    )
    await sendFromComposer(view, "Next question")
    expect(completionMessages(server.completionRequests()[1])).toEqual([
      { role: "user", content: "Next question" }
    ])
  })

  it("CS-01 (#3106): New chat from the sidebar '+' starts a clean conversation", async () => {
    const server = createFakeTldwServer()
    const view = await renderPlayground({ server, extras: <ChatSidebar collapsed /> })
    await expectNewChatStartsClean(view, async () => {
      await view.user.click(screen.getByTestId("chat-sidebar-new-chat"))
    })
  })

  it("CS-01 (#3106): New chat from Ctrl+Shift+U starts a clean conversation", async () => {
    const server = createFakeTldwServer()
    const view = await renderPlayground({ server, extras: <NewChatShortcut /> })
    await expectNewChatStartsClean(view, async () => {
      await view.user.keyboard("{Control>}{Shift>}u{/Shift}{/Control}")
      await waitFor(() => expect(useStoreMessageOption.getState().historyId).toBeNull(), HARNESS_WAIT)
    })
  })

  it("CS-01 (#3106): New chat drops the stored history-selection reference and the persisted session", async () => {
    const server = createFakeTldwServer()
    const view = await renderPlayground({ server, extras: <HeaderNewChatButton /> })
    await sendFromComposer(view, "First topic question")
    // The chat's owner since #3195: the server chat created by the first send.
    const previousChatId = useStoreMessageOption.getState().serverChatId as string
    expect(previousChatId).toBeTruthy()
    await waitFor(() => {
      expect(JSON.parse(sessionStorage.getItem(PLAYGROUND_REFERENCE_KEY) ?? "{}")).toMatchObject({
        conversation_id: previousChatId
      })
    }, HARNESS_WAIT)

    await clickHeaderNewChat(view)

    // A reload must not reopen the conversation the user left.
    expect(sessionStorage.getItem(PLAYGROUND_REFERENCE_KEY)).toBeNull()
    expect(usePlaygroundSessionStore.getState()).toMatchObject({
      historyId: null,
      serverChatId: null,
      historySelectionReference: null
    })
  })

  it("CS-05 (#3106): Clear conversation on a saved server chat posts nothing to it and the next request carries no prior messages", async () => {
    const server = createFakeTldwServer()
    server.seedChat({
      id: "saved-chat",
      title: "Saved topic",
      turns: [
        { role: "user", content: "Saved question" },
        { role: "assistant", content: "Saved answer" }
      ]
    })
    const view = await renderPlayground({
      server,
      extras: <SidebarServerChatButton chatId="saved-chat" title="Saved topic" />
    })
    await openSavedServerChat(view, { title: "Saved topic", lastMessage: "Saved answer" })

    await view.user.click(screen.getByRole("button", { name: "More tools" }))
    await view.user.click(await screen.findByText("Clear conversation"))
    // antd confirm dialogs live outside the React tree, and jsdom never finishes
    // their leave animation, so an earlier test's dialog can linger: use the newest.
    const confirmButtons = await screen.findAllByRole("button", { name: "Confirm" })
    await view.user.click(confirmButtons[confirmButtons.length - 1])
    await waitFor(() => expect(screen.queryByText("Saved answer")).not.toBeInTheDocument(), HARNESS_WAIT)

    await sendFromComposer(view, "Unrelated question")
    expect(server.find("POST", "/api/v1/chats/saved-chat/messages")).toEqual([])
    expect(server.completionRequests().map(completionMessages)).toEqual([
      [{ role: "user", content: "Unrelated question" }]
    ])
  })

  it("CS-01 (#3106): Send stays disabled while a restored chat's history is still loading; New chat makes the selection idle and sendable", async () => {
    const server = createFakeTldwServer()
    server.seedChat({
      id: "saved-chat",
      title: "Saved topic",
      turns: [
        { role: "user", content: "Saved question" },
        { role: "assistant", content: "Saved answer" }
      ]
    })
    // Hold the restore's history capture so the selection stays "loading" while
    // the surface still addresses no conversation.
    const capture = deferred()
    const serverFetch = server.fetch
    // The session an earlier visit persisted: a saved server chat in this server scope.
    usePlaygroundSessionStore.setState({
      serverChatId: "saved-chat",
      scopeKey: buildChatSurfaceScopeKeyFromConfig({
        serverUrl: FAKE_SERVER_URL,
        authMode: "single-user",
        apiKey: FAKE_API_KEY
      }),
      lastUpdated: Date.now()
    })
    const view = await renderPlayground({
      server: {
        ...server,
        fetch: async (input: RequestInfo | URL, init?: RequestInit) => {
          const url = typeof input === "string" ? input : input instanceof URL ? input.href : input.url
          if (/\/history\/selection$/.test(new URL(url, FAKE_SERVER_URL).pathname)) await capture.promise
          return serverFetch(input, init)
        }
      },
      extras: <HeaderNewChatButton />
    })
    try {
      await waitFor(() => expect(sendButton()).toBeDisabled(), HARNESS_WAIT)
      await view.user.click(view.composer)
      await view.user.type(view.composer, "Unrelated question")
      await view.user.keyboard("{Enter}")
      expect(view.composer).toHaveValue("Unrelated question")

      await clickHeaderNewChat(view)
      await waitFor(() => expect(sendButton()).toBeEnabled(), HARNESS_WAIT)
    } finally {
      capture.resolve()
    }
    await view.user.click(sendButton())
    await waitFor(() => expect(server.completionRequests()).toHaveLength(1), HARNESS_WAIT)
    expect(completionMessages(server.completionRequests()[0])).toEqual([
      { role: "user", content: "Unrelated question" }
    ])
    expect(server.find("POST", "/api/v1/chats/saved-chat/messages")).toEqual([])
    expect(screen.queryByText("Saved answer")).not.toBeInTheDocument()
  })
})
