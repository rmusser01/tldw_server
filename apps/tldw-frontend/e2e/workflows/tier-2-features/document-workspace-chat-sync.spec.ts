import { randomUUID } from "node:crypto"
import path from "node:path"
import { readFile } from "node:fs/promises"
import type { APIRequestContext, Browser, Page } from "@playwright/test"
import { test, expect } from "../../utils/fixtures"
import { DocumentWorkspacePage } from "../../utils/page-objects"
import { seedAuth, TEST_CONFIG } from "../../utils/helpers"

const api = TEST_CONFIG.serverUrl
const modelId = "gpt-4o-mini"
const fixtureDir = path.resolve(process.cwd(), "../../tldw_Server_API/tests/Media_Ingestion_Modification/test_media")
test.use({ actionTimeout: 15_000 })
test.skip(process.env.TLDW_E2E_DOCUMENT_CHAT_SYNC !== "1",
  "Requires a disposable multi-user backend for account registration.")

async function register(request: APIRequestContext) {
  const username = `document-sync-${randomUUID().slice(0, 12)}`
  const password = `Qm9!${randomUUID().slice(0, 8).split("").join("%")}`
  const response = await request.post(`${api}/api/v1/auth/register`, {
    data: { username, password, email: `${username}@example.com` }
  })
  expect(response.ok(), `Registration returned HTTP ${response.status()}`).toBe(true)
  return { username, password }
}

async function login(request: APIRequestContext, form: { username: string; password: string }) {
  const response = await request.post(`${api}/api/v1/auth/login`, { form })
  expect(response.ok(), `Login returned HTTP ${response.status()}`).toBe(true)
  const tokens = await response.json() as { access_token: string; refresh_token: string }
  expect(Boolean(tokens.access_token)).toBe(true)
  return tokens
}

async function upload(request: APIRequestContext, token: string, extension: "pdf" | "epub", title: string) {
  const headers = { Authorization: `Bearer ${token}` }
  const response = await request.post(`${api}/api/v1/media/add`, {
    headers,
    multipart: {
      media_type: extension === "pdf" ? "pdf" : "ebook",
      title, perform_analysis: "false", keep_original_file: "true",
      files: {
        name: `sample.${extension}`,
        mimeType: extension === "pdf" ? "application/pdf" : "application/epub+zip",
        buffer: await readFile(path.join(fixtureDir, `sample.${extension}`))
      }
    }
  })
  expect(response.ok(), `Media ingestion returned HTTP ${response.status()}`).toBe(true)
  const result = (await response.json()).results[0]
  const mediaId = Number(result.db_id)
  expect(mediaId, `Media ingestion result status ${result.status}`).toBeGreaterThan(0)
  const original = await request.get(`${api}/api/v1/media/${mediaId}/file`, { headers })
  expect(original.ok(), `Original file returned HTTP ${original.status()}`).toBe(true)
  const detail = await request.get(`${api}/api/v1/media/${mediaId}`, { headers })
  expect(detail.ok()).toBe(true)
  const body = await detail.json()
  expect(body.uuid).toEqual(expect.any(String))
  const text = body.content.text as string
  const word = extension === "epub" ? text.match(/\brabbit\b/i)?.[0] : text.match(/[a-zA-Z]{7,}/)?.[0]
  expect(word).toEqual(expect.any(String))
  return { mediaId, uuid: body.uuid as string, word: word!, title }
}

async function openChat(browser: Browser, tokens: Awaited<ReturnType<typeof login>>, documentTitle: string) {
  const context = await browser.newContext()
  const page = await context.newPage()
  page.on("console", message => {
    if (/document conversations refresh/i.test(message.text())) console.info(message.text())
  })
  await seedAuth(page, { authTokens: tokens, allowOffline: false })
  await page.addInitScript(() => {
    localStorage.setItem("dw-tips-tour-completed", "true")
    localStorage.setItem("tldw-tutorials", JSON.stringify({
      state: { completedTutorials: [], seenPromptPages: ["/document-workspace"] }, version: 0
    }))
  })
  const completions: unknown[] = []
  await page.route("**/api/v1/chat/completions", async route => {
    completions.push(route.request().postDataJSON())
    await route.fulfill({
      status: 200, contentType: "text/event-stream",
      body: `data: ${JSON.stringify({ choices: [{ delta: { content: "Verified answer from the test model." } }] })}\n\ndata: [DONE]\n\n`
    })
  })
  const input = await openWorkspaceChat(page, documentTitle)
  await expect(page.getByRole("checkbox", { name: "Save to server" })).toBeChecked()
  return { context, page, input, completions }
}

async function openWorkspaceChat(page: Page, documentTitle: string) {
  const workspace = new DocumentWorkspacePage(page)
  await workspace.goto()
  await workspace.assertPageReady()
  const dismiss = page.getByRole("button", { name: "Not now", exact: true })
  if (await dismiss.isVisible()) await dismiss.click()
  await page.keyboard.press("Escape")
  const chatTab = page.getByRole("tab", { name: "Chat", exact: true })
  const openDocuments = page.getByRole("tab", { name: documentTitle, exact: true })
  if (!await openDocuments.isVisible()) {
    await workspace.openPicker()
    const document = workspace.pickerModal.getByRole("listitem").filter({ hasText: documentTitle })
    await document.getByRole("button", { name: "Open", exact: true }).click()
  }
  if (!await chatTab.isVisible()) await workspace.toggleRightButton.click()
  if (await chatTab.getAttribute("aria-selected") !== "true") await chatTab.click()
  await page.evaluate(model => {
    const store = (window as typeof window & {
      __tldw_useStoreMessageOption?: { setState: (state: Record<string, unknown>) => void }
    }).__tldw_useStoreMessageOption
    if (!store) throw new Error("WebUI message settings store is unavailable")
    store.setState({ selectedModel: model, toolChoice: "none", ragSearchMode: "fts", fileRetrievalEnabled: true })
  }, modelId)
  const input = page.getByRole("textbox", { name: "Ask about this document..." })
  await expect(input).toBeEnabled()
  return input
}

const list = (page: Page) => page.getByRole("tablist", { name: "Conversations" })
const replies = (page: Page) => page.getByRole("article", { name: /assistant message/i })
  .getByText("Verified answer from the test model.", { exact: true })
const sendButton = (page: Page) => page.getByRole("button", { name: "Send", exact: true })

function savedReply(page: Page) {
  return page.waitForResponse(response =>
    /\/chats\/[^/]+\/messages\?/.test(response.url()) &&
    response.request().method() === "POST" && response.request().postDataJSON()?.role === "assistant", { timeout: 30_000 })
    .catch(async error => {
      const status = (await page.getByRole("status").allTextContents()).join(" ").slice(0, 600)
      throw new Error(`Expected a saved reply. Sidebar status was: ${status || "(empty)"}`, { cause: error })
    })
}

for (const extension of ["pdf", "epub"] as const) {
  test(`syncs the same ${extension.toUpperCase()} chat and sources between two signed-in browsers`, async ({ browser, request }) => {
    test.setTimeout(180_000)
    const owner = await register(request)
    const ownerTokens = await login(request, owner)
    const secondTokens = await login(request, owner)
    const foreign = await login(request, await register(request))
    const fixture = await upload(request, ownerTokens.access_token, extension, `Owned ${extension} sync fixture`)
    const otherCopy = await upload(request, foreign.access_token, extension, `Other ${extension} sync fixture`)
    const first = await openChat(browser, ownerTokens, fixture.title)
    const second = await openChat(browser, secondTokens, fixture.title)
    const third = await openChat(browser, foreign, otherCopy.title)
    try {
      const title = `Private ${extension} conversation`
      await expect(first.page.locator(extension === "pdf" ? ".react-pdf__Page__canvas" : "iframe").first()).toBeVisible()
      await second.input.fill("Independent draft")
      await first.input.fill(title)
      const settled = savedReply(first.page)
      await sendButton(first.page).click()
      const settlement = await settled
      expect(settlement.status()).toBe(201)
      const chatId = settlement.url().match(/\/chats\/([^/]+)\/messages/)?.[1]
      expect(chatId).toEqual(expect.any(String))
      expect((await settlement.json()).metadata_extra?.document_chat_sources_v1).toBeUndefined()
      await expect(replies(first.page)).toBeVisible()
      await expect(list(first.page).getByRole("tab", { name: title })).toBeVisible()

      await second.page.bringToFront()
      await second.page.evaluate(() => window.dispatchEvent(new Event("focus")))
      const synced = list(second.page).getByRole("tab", { name: title })
      await expect(synced).toHaveCount(1)
      await expect(second.input).toHaveValue("Independent draft")
      await synced.click()
      await expect(replies(second.page)).toBeVisible()

      await second.page.getByRole("switch", { name: /use document content/i }).click()
      if (extension === "epub") {
        const unmatched = "zyzzyvanodatapassage"
        await second.input.fill(unmatched)
        await sendButton(second.page).click()
        await expect(second.page.getByRole("status").filter({ hasText: /couldn't find supporting evidence/i })).toBeVisible()
        await expect(second.input).toHaveValue(unmatched)
        await expect(replies(second.page)).toHaveCount(1)
      }
      await second.input.fill(fixture.word)
      const retrieval = second.page.waitForResponse(response => response.url().includes("/api/v1/rag/search") &&
        response.request().method() === "POST", { timeout: 30_000 })
      const savedGroundedReply = savedReply(second.page)
      await sendButton(second.page).click()
      const grounded = await retrieval
      expect(grounded.ok()).toBe(true)
      expect(grounded.request().postDataJSON().include_media_ids).toEqual([fixture.mediaId])
      expect((await savedGroundedReply).status()).toBe(201)
      await expect(replies(second.page)).toHaveCount(2)
      await expect(second.page.getByText(/\d+ sources?/).last()).toBeVisible()
      const messages = await request.get(`${api}/api/v1/chats/${chatId}/messages?include_metadata=true`, {
        headers: { Authorization: `Bearer ${secondTokens.access_token}` }
      })
      expect(messages.ok()).toBe(true)
      const { messages: rows } = await messages.json()
      const sources = rows.find((row: { metadata_extra?: { document_chat_sources_v1?: unknown } }) =>
        row.metadata_extra?.document_chat_sources_v1
      )?.metadata_extra.document_chat_sources_v1
      expect(sources).toEqual(expect.arrayContaining([expect.objectContaining({ media_uuid: fixture.uuid, excerpt: expect.any(String) })]))
      await first.page.bringToFront()
      await first.page.evaluate(() => window.dispatchEvent(new Event("focus")))
      await first.page.getByRole("button", { name: "New chat" }).click()
      await list(first.page).getByRole("tab", { name: title, exact: true }).click()
      await expect(replies(first.page)).toHaveCount(2)
      await expect(first.page.getByText(/\d+ sources?/).last()).toBeVisible()
      await expect(list(third.page).getByRole("tab", { name: title })).toHaveCount(0)
      expect(second.completions.length).toBeGreaterThan(0)
    } finally {
      await Promise.allSettled([first.context.close(), second.context.close(), third.context.close()])
    }
  })
}

test("confirms and saves one legacy browser chat to the server", async ({ browser, request }) => {
  test.setTimeout(120_000)
  const owner = await register(request)
  const ownerTokens = await login(request, owner)
  const fixture = await upload(request, ownerTokens.access_token, "epub", "Legacy save fixture")
  const first = await openChat(browser, ownerTokens, fixture.title)
  const second = await openChat(browser, await login(request, owner), fixture.title)
  try {
    const title = "Legacy browser question"
    const saveMode = first.page.getByRole("checkbox", { name: "Save to server" })
    await saveMode.uncheck()
    await first.input.fill(title)
    await sendButton(first.page).click()
    const save = first.page.getByRole("button", { name: "Save to server", exact: true })
    await expect(save).toBeEnabled()
    await expect(list(second.page).getByRole("tab", { name: title, exact: true })).toHaveCount(0)
    const priorUuid = await first.page.evaluate(docId => new Promise<string | undefined>((resolve, reject) => {
      const open = indexedDB.open("PageAssistDatabase")
      open.onerror = () => reject(open.error)
      open.onsuccess = () => {
        const db = open.result
        const tx = db.transaction("chatHistories", "readwrite")
        const histories = tx.objectStore("chatHistories")
        const rows = histories.getAll()
        let uuid: string | undefined
        rows.onsuccess = () => {
          const history = rows.result.find(row => row.doc_id === docId && !row.server_chat_id)
          if (!history) { tx.abort(); return }
          uuid = history.document_media_uuid
          delete history.document_media_uuid
          histories.put(history)
        }
        tx.oncomplete = () => { db.close(); resolve(uuid) }
        tx.onabort = () => { db.close(); reject(tx.error ?? new Error("Browser history was missing")) }
      }
    }), `document:${fixture.mediaId}`)
    expect(priorUuid).toBe(fixture.uuid)
    await openWorkspaceChat(first.page, fixture.title)
    await expect(first.page.getByRole("tabpanel", { name: title }).getByText(title, { exact: true })).toBeVisible()
    const imports: string[] = []
    first.page.on("request", req => {
      if (req.method() === "POST" && new URL(req.url()).pathname === "/api/v1/chats/import")
        imports.push(req.url())
    })
    await save.click()
    const confirm = first.page.getByRole("dialog", { name: "Confirm this document" })
    await expect(confirm.getByText(fixture.title, { exact: false })).toBeVisible()
    expect(imports).toHaveLength(0)
    await confirm.getByRole("button", { name: "Save to server" }).click()
    await expect(save).toHaveCount(0)
    expect(imports).toHaveLength(1)
    await second.page.bringToFront()
    await second.page.evaluate(() => window.dispatchEvent(new Event("focus")))
    const linked = list(second.page).getByRole("tab", { name: title, exact: true })
    await expect(linked).toHaveCount(1)
    await linked.click()
    await expect(replies(second.page)).toBeVisible()
    await openWorkspaceChat(first.page, fixture.title)
    await expect(list(first.page).getByRole("tab", { name: title, exact: true })).toHaveCount(1)
  } finally {
    await Promise.allSettled([first.context.close(), second.context.close()])
  }
})

test("recovers a lost create response and publishes only after the first question is saved", async ({ browser, request }) => {
  test.setTimeout(120_000)
  const owner = await register(request)
  const ownerTokens = await login(request, owner)
  const fixture = await upload(request, ownerTokens.access_token, "epub", "First question recovery fixture")
  const first = await openChat(browser, ownerTokens, fixture.title)
  try {
    const attempts: Array<{ id: string; status: number }> = []
    await first.page.route(/\/api\/v1\/chats\/\?.*document_media_uuid=/, async route => {
      if (route.request().method() !== "POST") return route.continue()
      const response = await route.fetch()
      attempts.push({ id: route.request().postDataJSON().id, status: response.status() })
      if (attempts.length === 1) await route.abort("failed")
      else await route.fulfill({ response })
    })
    const question = "Retained first question"
    await first.input.fill(question)
    await sendButton(first.page).click()
    await expect(first.page.getByRole("status").filter({ hasText: /server save unconfirmed/i })).toBeVisible()
    await expect(first.input).toHaveValue(question)
    expect(attempts.map(attempt => attempt.status)).toEqual([201])
    const recovered = await openWorkspaceChat(first.page, fixture.title)
    await expect(recovered).toHaveValue(question)

    let refused = false
    await first.page.route(/\/api\/v1\/chats\/[^/]+\/messages\?.*document_media_uuid=/, async route => {
      if (refused || route.request().method() !== "POST" || route.request().postDataJSON()?.role !== "user")
        return route.continue()
      refused = true
      await route.fulfill({
        status: 409,
        json: { detail: "First question was refused before saving." },
        headers: {
          "Access-Control-Allow-Origin": new URL(first.page.url()).origin,
          "Access-Control-Allow-Credentials": "true"
        }
      })
    })
    await sendButton(first.page).click()
    await expect.poll(() => refused).toBe(true)
    await expect(sendButton(first.page)).toBeEnabled()
    await expect(recovered).toHaveValue(question)
    const discover = async () => {
      const response = await request.get(`${api}/api/v1/media/${fixture.mediaId}/chat-conversations?media_uuid=${fixture.uuid}`, {
        headers: { Authorization: `Bearer ${ownerTokens.access_token}` }
      })
      expect(response.ok()).toBe(true)
      return (await response.json()).items
    }
    expect(await discover()).toEqual([])
    const settled = savedReply(first.page)
    await sendButton(first.page).click()
    expect((await settled).status()).toBe(201)
    expect(attempts).toEqual([{ ...attempts[0], status: 201 }, { ...attempts[0], status: 200 }])
    await expect(list(first.page).getByRole("tab", { name: question, exact: true })).toHaveCount(1)
    await expect(first.page.getByRole("article").getByText(question, { exact: true })).toHaveCount(1)
    await expect(replies(first.page)).toBeVisible()
    expect(await discover()).toEqual([expect.objectContaining({ id: attempts[0].id })])
  } finally {
    await first.context.close()
  }
})
