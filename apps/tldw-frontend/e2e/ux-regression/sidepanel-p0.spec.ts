/**
 * Extension side-panel tests from the 2026-10-02 UX review (G04, #3105;
 * tracking #3101). XS-01, XS-07 and XP-08 are fixed and guard their fixes with
 * plain assertions; see e2e/ux-regression/README.md.
 *
 * The tests load the built extension into Chromium against the runner's
 * isolated backend (e2e/utils/extension-sidepanel.ts). They seed chats through
 * the API and never send from the composer: on current dev the first send in a
 * fresh browser is not reliably delivered (#3106).
 */
import type { SeedApi } from "../utils/seed-api"
import { createCharacter, createChatWithMessages, createSeedApi, warmBackendOnce } from "../utils/seed-api"
import { addServerChatMessage, listServerChatMessages, readServerChat, type ServerChatState } from "../utils/chat-api"
import { SidePanelChat, expect, test, type SidePanelTab } from "../utils/extension-sidepanel"
import { skipIfServerUnavailable } from "../utils/fixtures"

/** A single lowercase word, so the server's full-text search treats it as one term. */
const uniqueWord = (prefix: string) =>
  `${prefix}${Date.now().toString(36)}${Math.random().toString(36).slice(2, 6)}`

type SeededChat = { chatId: string; title: string; question: string; answer: string }

/** A saved two-message server chat whose title and messages carry one unique word. */
async function seedChat(api: SeedApi, characterId: number, name: string): Promise<SeededChat> {
  const word = uniqueWord(name.toLowerCase())
  const title = `${name} ${word}`
  const question = `${name} question ${word}`
  const answer = `${name} answer ${word}`
  const chatId = await createChatWithMessages(api, {
    characterId,
    title,
    messages: [
      { role: "user", content: question },
      { role: "assistant", content: answer },
    ],
  })
  return { chatId, title, question, answer }
}

/** Poll the server until the chat reads as deleted or `timeoutMs` passes; return the last state. */
async function settleServerChat(api: SeedApi, chatId: string, timeoutMs = 5_000): Promise<ServerChatState> {
  const deadline = Date.now() + timeoutMs
  let state = await readServerChat(api, chatId)
  while (!state.deleted && Date.now() < deadline) {
    await new Promise((resolve) => setTimeout(resolve, 250))
    state = await readServerChat(api, chatId)
  }
  return state
}

/** Copy a stale side panel might show instead of refreshing (XP-08's recommended fix). */
const STALE_NOTICE = /updated in another (window|tab)|newer messages|out of date/i

test.describe("Extension side-panel P0 reproductions", () => {
  test("XS-01: opening a past chat from search leaves the other tabs' conversations alone", async ({
    extension,
    serverInfo,
    request,
  }) => {
    skipIfServerUnavailable(serverInfo)
    const api = createSeedApi(request)
    await warmBackendOnce(api)
    const characterId = await createCharacter(api, `UXR XS-01 ${uniqueWord("char")}`)
    const chats = [await seedChat(api, characterId, "Alpha"), await seedChat(api, characterId, "Bravo")]
    const [chat1, chat2] = chats

    const panel = new SidePanelChat(await extension.openSidePanel())

    // Tab A: chat 1, opened from the history search.
    await panel.openServerChatFromSearch(chat1.title, chat1.chatId)
    await panel.closeSidebar()
    await expect(panel.transcript.getByText(chat1.answer)).toBeVisible()
    await expect
      .poll(async () => (await panel.readTabs()).tabs.find((tab) => tab.serverChatId === chat1.chatId)?.messages, {
        message: "precondition: tab A holds chat 1",
      })
      .toEqual(expect.arrayContaining([chat1.question, chat1.answer]))

    // From tab A, look up chat 2.
    await panel.openServerChatFromSearch(chat2.title, chat2.chatId)
    await panel.closeSidebar()
    await expect(panel.transcript.getByText(chat2.answer)).toBeVisible()

    const chatName = (chatId: string | null) => {
      const index = chats.findIndex((chat) => chat.chatId === chatId)
      return index >= 0 ? `chat ${index + 1}` : null
    }
    const heldChats = (tab: SidePanelTab) =>
      chats
        .filter((chat) => tab.messages.includes(chat.question) || tab.messages.includes(chat.answer))
        .map((chat) => chatName(chat.chatId))
    // A tab bound to a chat holds that chat's messages and no other chat's; an
    // unbound tab holds none.
    const misplaced = async () => {
      const { tabs } = await panel.readTabs()
      const problems = tabs.flatMap((tab) => {
        const own = chatName(tab.serverChatId)
        const held = heldChats(tab)
        const name = `${own ? `${own}'s tab` : "unbound tab"} ("${tab.label}")`
        return [
          ...(own && !held.includes(own) ? [`${name} lost ${own}`] : []),
          ...held.filter((chat) => chat !== own).map((chat) => `${name} holds ${chat}`),
        ]
      })
      if (!tabs.some((tab) => tab.serverChatId === chat1.chatId)) problems.push("no tab is bound to chat 1")
      return problems
    }

    // XS-01 (#3105, fixed): opening a past chat from search used to overwrite the current tab.
    await expect
      .poll(misplaced, { timeout: 5_000, message: "each side-panel tab should hold only its own chat" })
      .toEqual([])
  })

  test("XS-07: 'Delete' on a side-panel chat deletes the conversation, not just its tab", async ({
    extension,
    serverInfo,
    request,
  }) => {
    skipIfServerUnavailable(serverInfo)
    const api = createSeedApi(request)
    await warmBackendOnce(api)
    const characterId = await createCharacter(api, `UXR XS-07 ${uniqueWord("char")}`)
    const chat = await seedChat(api, characterId, "Delta")

    const panel = new SidePanelChat(await extension.openSidePanel())
    await panel.openServerChatFromSearch(chat.title, chat.chatId)
    await panel.clearSearch()

    await panel.chooseTabMenuItem(chat.title, "Delete")
    const dialog = panel.page.getByRole("dialog", { name: "Delete conversation" })
    await expect(dialog).toContainText("moves to Trash")
    await dialog.getByRole("button", { name: "Move to Trash", exact: true }).click()
    await expect(dialog).toBeHidden()
    // Precondition: the delete went through in the UI (the tab is gone).
    await expect(panel.tabRow(chat.title)).toHaveCount(0)

    // XS-07 (#3105, fixed): Delete used to close the tab and leave the chat on the server.
    const server = await settleServerChat(api, chat.chatId)
    await panel.search(chat.title)
    const noMatches = panel.sidebar.getByText("No matches found")
    await expect(panel.searchResult(chat.title).or(noMatches).first()).toBeVisible()
    const listed = await panel.searchResult(chat.title).allInnerTexts()
    expect(
      {
        server: server.deleted ? "deleted" : `still saved (GET returns ${server.status})`,
        sidePanelSearch: listed.length ? `still lists it: ${listed.join(" | ").replace(/\s+/g, " ")}` : "no match",
      },
      "Delete should remove the conversation from the server and from side-panel search"
    ).toEqual({ server: "deleted", sidePanelSearch: "no match" })
  })

  test("XS-07: 'Rename' on a side-panel chat renames the conversation, not just its tab", async ({
    extension,
    serverInfo,
    request,
  }) => {
    skipIfServerUnavailable(serverInfo)
    const api = createSeedApi(request)
    await warmBackendOnce(api)
    const characterId = await createCharacter(api, `UXR XS-07 ${uniqueWord("char")}`)
    const chat = await seedChat(api, characterId, "Echo")
    const newTitle = `Renamed ${uniqueWord("echo")}`

    const panel = new SidePanelChat(await extension.openSidePanel())
    await panel.openServerChatFromSearch(chat.title, chat.chatId)
    await panel.clearSearch()

    await panel.chooseTabMenuItem(chat.title, "Rename")
    const dialog = panel.page.getByRole("dialog", { name: "Rename conversation" })
    await dialog.getByRole("textbox").fill(newTitle)
    await dialog.getByRole("button", { name: "Save", exact: true }).click()
    await expect(dialog).toBeHidden()
    // Precondition: the rename went through in the UI.
    await expect(panel.tabRow(newTitle)).toBeVisible()

    // XS-07 (#3105, fixed): Rename used to relabel only the local tab.
    await expect
      .poll(async () => (await readServerChat(api, chat.chatId)).title, {
        timeout: 5_000,
        message: "the server's chat should carry the new title",
      })
      .toBe(newTitle)
  })

  test("XP-08: a reopened side panel shows turns added to its chat elsewhere", async ({
    extension,
    serverInfo,
    request,
  }) => {
    skipIfServerUnavailable(serverInfo)
    const api = createSeedApi(request)
    await warmBackendOnce(api)
    const characterId = await createCharacter(api, `UXR XP-08 ${uniqueWord("char")}`)
    const chat = await seedChat(api, characterId, "Charlie")

    const panel = new SidePanelChat(await extension.openSidePanel())
    await panel.openServerChatFromSearch(chat.title, chat.chatId)
    await panel.closeSidebar()
    await expect(panel.transcript.getByText(chat.answer)).toBeVisible()

    // Continue the chat in another client (the full page), as a reply to its
    // latest message.
    const followUp = `Charlie follow-up ${uniqueWord("turn")}`
    const reply = `Charlie reply ${uniqueWord("turn")}`
    const leaf = (await listServerChatMessages(api, chat.chatId)).at(-1)
    expect(leaf?.content, "precondition: the seeded chat ends with its answer").toBe(chat.answer)
    const followUpId = await addServerChatMessage(api, chat.chatId, { role: "user", content: followUp, parentId: leaf!.id })
    await addServerChatMessage(api, chat.chatId, { role: "assistant", content: reply, parentId: followUpId })
    expect((await listServerChatMessages(api, chat.chatId)).map((message) => message.content)).toEqual([
      chat.question,
      chat.answer,
      followUp,
      reply,
    ])

    // Close and reopen the side panel.
    await panel.page.close()
    const reopened = new SidePanelChat(await extension.openSidePanel())
    // Precondition: the panel restored the chat.
    await expect(reopened.transcript.getByText(chat.answer)).toBeVisible()

    // XP-08 (#3105, fixed): side-panel tabs used never to refresh from the server.
    await expect(
      reopened.transcript.getByText(reply).or(reopened.page.getByText(STALE_NOTICE)).first(),
      "the reopened side panel should show the turn added in another client, or warn that its copy is out of date"
    ).toBeVisible({ timeout: 10_000 })
  })
})
