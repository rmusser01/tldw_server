/**
 * Chat P0 tests from the 2026-10-02 UX review (tracking #3101). CS-02 is fixed
 * and guards its fix with a plain assertion; see e2e/ux-regression/README.md.
 *
 * These reproductions seed chats through the API and never send from the
 * composer: on current dev the first send in a fresh browser is not reliably
 * delivered (#3106).
 */
import type { Page, Response } from "@playwright/test"
import { test, expect, skipIfServerUnavailable } from "../utils/fixtures"
import { ChatPage } from "../utils/page-objects"
import { createCharacter, createChatWithMessages, createSeedApi, warmBackendOnce } from "../utils/seed-api"

/** A single lowercase word, so the server's full-text search treats it as one term. */
const uniqueWord = (prefix: string) =>
  `${prefix}${Date.now().toString(36)}${Math.random().toString(36).slice(2, 6)}`

const isConversationSearch = (term: string) => (response: Response) => {
  const url = new URL(response.url())
  return url.pathname.replace(/\/$/, "") === "/api/v1/chats/conversations" && url.searchParams.get("query") === term
}

/** Open /chat's history sidebar and its Recent conversations section. */
async function openChatHistory(page: Page) {
  const chat = new ChatPage(page)
  await chat.goto()
  const sidebar = page.getByTestId("chat-sidebar")
  const toggle = sidebar.getByTestId("chat-sidebar-toggle")
  await expect(toggle).toBeVisible()
  if (/expand sidebar/i.test((await toggle.getAttribute("aria-label")) ?? "")) await toggle.click()
  await expect(toggle).toHaveAccessibleName(/collapse sidebar/i)

  const recent = sidebar.getByRole("button", { name: /recent conversations/i })
  if ((await recent.getAttribute("aria-expanded")) !== "true") await recent.click()
  await expect(recent).toHaveAttribute("aria-expanded", "true")
  const search = sidebar.getByPlaceholder("Search chats...")
  await expect(search).toBeVisible()
  return { sidebar, search }
}

/** Type a history search and wait for the server's answer to it. */
async function searchHistory(page: Page, search: ReturnType<Page["getByPlaceholder"]>, term: string) {
  const answered = page.waitForResponse(isConversationSearch(term))
  await search.fill(term)
  const response = await answered
  expect(response.ok()).toBe(true)
  const payload = await response.json()
  return Array.isArray(payload?.items) ? (payload.items as Array<{ id?: string }>) : []
}

test.describe("Chat P0 reproductions", () => {
  test("CS-02: chat history search finds a past chat by what was said in it", async ({
    authedPage,
    serverInfo,
    request,
  }) => {
    skipIfServerUnavailable(serverInfo)
    const api = createSeedApi(request)
    await warmBackendOnce(api)

    // A realistic past chat: a bland title, the memorable detail only in the
    // messages.
    const titleWord = uniqueWord("sync")
    const contentWord = uniqueWord("codeword")
    const title = `Weekly sync ${titleWord}`
    const characterId = await createCharacter(api, `UXR CS-02 ${titleWord}`)
    const chatId = await createChatWithMessages(api, {
      characterId,
      title,
      messages: [
        { role: "user", content: `Remember that the launch codeword is ${contentWord}.` },
        { role: "assistant", content: `Noted: the launch codeword is ${contentWord}.` },
      ],
    })

    const { sidebar, search } = await openChatHistory(authedPage)
    // The row's own button; its "More actions: <title>" menu button also matches the title.
    const chatRow = sidebar.getByRole("button", { name: new RegExp(`^${title}\\b`) })

    // Precondition: the history search lists this chat when searched by title.
    const byTitle = await searchHistory(authedPage, search, titleWord)
    expect(byTitle.map((item) => item.id)).toContain(chatId)
    await expect(chatRow).toBeVisible()

    const byContent = await searchHistory(authedPage, search, contentWord)

    // CS-02 (#3108, fixed): history search used to match titles only.
    await expect(
      chatRow,
      `the chat should be listed; the server's search returned ${JSON.stringify(byContent.map((item) => item.id))}`
    ).toBeVisible({ timeout: 5_000 })
  })
})
